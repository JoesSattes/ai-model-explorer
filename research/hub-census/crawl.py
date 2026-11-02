"""Read the safetensors headers of every sampled repo and record what the weights say.

Each repo runs in its own process with a hard time limit, so one huge or stuck repo cannot stall the run.
Output is JSONL, one line per repo, appended as it goes: stop and restart at any time and it picks up where it left off.

Failure handling
  401/403 (gated, private)  -> status "gated"; popularity stats from the frame are still kept
  404                       -> "not_found" (deleted or renamed since the frame was built)
  no weights we can read    -> "no_weights" (GGUF-only, ONNX-only, adapters without base, code-only repos)
  429 / 5xx                 -> retried inside the package, honouring Retry-After, with exponential back-off
  over --timeout seconds    -> "timeout" (the process is killed)
Set HF_TOKEN for higher rate limits and access to gated repos you have accepted.
"""
import argparse, json, multiprocessing as mp, os, sys, time
import pandas as pd

KEEP = ["downloads", "downloads_all_time", "likes", "library", "pipeline_tag", "gated", "stratum", "weight"]


def _kinds(comps):
    out = []
    for c in comps or []:
        for st in c.get("stacks") or []:
            out.append({"comp": c.get("id"), "n": st.get("n"), "kinds": st.get("kinds") or c.get("kinds"),
                        "pattern": st.get("pattern")})
    return out


def probe(repo: str, token):
    """Everything we learn about one repo. Runs in a child process."""
    from ai_model_explorer._hub import Hub
    from ai_model_explorer.api import _remote
    t0 = time.time()
    hub = Hub(token)
    tensors, cfg, read, nfiles, extra = _remote(repo, hub)
    t_fetch = time.time() - t0
    eng = extra.pop("engine")
    an = eng.infer(tensors)                       # manifest only: no config, no code
    fam_manifest = an.get("family")
    check = eng.fuse(an, cfg) if cfg else None
    fam_cfg = eng.family({"family": fam_manifest, "comps": an.get("comps")}, cfg) if cfg else None
    tot = an.get("totals") or {}
    hub_params = extra.get("hub_params")
    checked = [v for v in (check or {}).values() if isinstance(v, dict) and v.get("checked")]
    cfg = cfg or {}
    return {
        "status": "ok",
        "files_read": nfiles, "header_bytes": read, "weight_bytes": extra.get("weight_bytes"),
        "tensors": len(tensors),
        "family_manifest": fam_manifest, "family_config": fam_cfg,
        "params": tot.get("params"), "stored_bytes": tot.get("bytes"),
        "hub_params": hub_params,
        "params_match_hub": (abs(tot.get("params", 0) - hub_params) / hub_params <= 0.001) if hub_params else None,
        "quant": an.get("quant"),
        "hp": an.get("hp"),
        "components": [{"id": c.get("id"), "role": c.get("role"), "modality": c.get("modality"),
                        "params": c.get("params"), "layers": c.get("layers")} for c in an.get("comps") or []],
        "stacks": _kinds(an.get("comps")),
        "has_config": bool(cfg),
        "config_architectures": cfg.get("architectures"),
        "config_model_type": cfg.get("model_type"),
        "config_checked": len(checked),
        "config_consistent": sum(1 for v in checked if v.get("consistent")),
        "config_mismatch": [v for v in checked if not v.get("consistent")][:3],
        "rate_limit_wait_s": round(hub.waited, 1),
        "t_fetch_s": round(t_fetch, 2), "t_total_s": round(time.time() - t0, 2),
    }


def _child(repo, token, q):
    try:
        q.put(probe(repo, token))
    except PermissionError as e:
        q.put({"status": "gated", "error": str(e)[:200]})
    except FileNotFoundError as e:
        msg = str(e)
        q.put({"status": "not_found" if "repository not found" in msg else "no_weights", "error": msg[:200]})
    except ConnectionError as e:
        q.put({"status": "network", "error": str(e)[:200]})
    except Exception as e:  # noqa: BLE001 - every failure is data here
        msg = f"{type(e).__name__}: {e}"
        if "gated repository" in msg:
            st = "gated"
        elif "old pickle format" in msg:
            st = "legacy_pickle"   # pre-1.6 torch.save: no zip index, would need the whole file
        else:
            st = "error"
        q.put({"status": st, "error": msg[:300]})


def run(sample: pd.DataFrame, out: str, workers: int, timeout: float, gap: float, token):
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [r for r in sample.to_dict("records") if r["id"] not in done]
    total = len(todo)
    print(f"{len(done)} already done, {len(todo)} to go", file=sys.stderr)
    ctx = mp.get_context("fork")
    running = {}  # id -> (proc, queue, start, row)
    n = 0
    with open(out, "a") as fh:
        def finish(rid, res, row, t):
            nonlocal n
            rec = {"id": rid, **{k: row.get(k) for k in KEEP}, **res, "wall_s": round(t, 2),
                   "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            n += 1
            if n % 25 == 0 or res["status"] != "ok":
                print(f"[{n}/{total}] {rid}: {res['status']}", file=sys.stderr)

        while todo or running:
            while todo and len(running) < workers:
                row = todo.pop(0)
                q = ctx.Queue()
                p = ctx.Process(target=_child, args=(row["id"], token, q), daemon=True)
                p.start()
                running[row["id"]] = (p, q, time.time(), row)
                time.sleep(gap)
            time.sleep(0.05)
            for rid, (p, q, t0, row) in list(running.items()):
                el = time.time() - t0
                res = None
                if not q.empty():
                    res = q.get()
                elif not p.is_alive():
                    res = q.get() if not q.empty() else {"status": "error", "error": f"worker exited {p.exitcode}"}
                elif el > timeout:
                    p.kill()
                    res = {"status": "timeout", "error": f"over {timeout:.0f}s"}
                if res is not None:
                    p.join(1)
                    del running[rid]
                    finish(rid, res, row, el)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="out/sample.parquet")
    ap.add_argument("--out", default="out/census.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--gap", type=float, default=0.25, help="seconds between starting repos (politeness)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (testing)")
    a = ap.parse_args()
    s = pd.read_parquet(a.sample)
    if a.limit:
        s = s.head(a.limit)
    run(s, a.out, a.workers, a.timeout, a.gap, os.environ.get("HF_TOKEN"))
