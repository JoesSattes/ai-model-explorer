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
import argparse, gzip, json, multiprocessing as mp, os, sys, time
import pandas as pd

KEEP = ["downloads", "downloads_all_time", "likes", "library", "pipeline_tag", "origin", "stratum", "weight", "also_frontier", "author", "created_at"]


def _kinds(comps):
    out = []
    for c in comps or []:
        for st in c.get("stacks") or []:
            out.append({"comp": c.get("id"), "n": st.get("n"), "kinds": st.get("kinds") or c.get("kinds"),
                        "pattern": st.get("pattern")})
    return out


TEXT_SUBS = ["text_config", "llm_config", "language_config", "thinker_config", "decoder", "encoder_config"]


def config_hp(cfg):
    """Hyperparameters the checkpoint's config.json declares: the reference the weights-only values are scored against."""
    if not cfg:
        return {}
    tc = next((cfg[k] for k in TEXT_SUBS if isinstance(cfg.get(k), dict) and (cfg[k].get("hidden_size") or cfg[k].get("d_model")
                                                                              or isinstance(cfg[k].get("text_config"), dict))), None)
    src = [tc["text_config"], tc, cfg] if tc and isinstance(tc.get("text_config"), dict) else [tc, cfg] if tc else [cfg]

    def g(*ks):
        for s_ in src:
            for k in ks:
                v = s_.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return v
        return None
    return {"d": g("hidden_size", "d_model", "n_embd", "dim", "embed_dim"),
            "layers": g("num_hidden_layers", "n_layer", "num_layers", "decoder_layers", "n_layers"),
            "heads": g("num_attention_heads", "n_head", "num_heads", "decoder_attention_heads", "n_heads"),
            "kv_heads": g("num_key_value_heads", "num_kv_heads", "n_kv_heads", "multi_query_group_num"),
            "head_dim": g("head_dim", "kv_channels", "attention_head_dim"),
            "ffn": g("intermediate_size", "ffn_dim", "d_ff", "n_inner", "decoder_ffn_dim"),
            "vocab": g("vocab_size", "padded_vocab_size"),
            "experts": g("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"),
            "topk": g("num_experts_per_tok", "moe_topk", "top_k", "num_selected_experts")}


def dtype_stats(tensors):
    out = {}
    for v in tensors.values():
        if not isinstance(v, dict) or "shape" not in v:
            continue
        n = 1
        for x in v["shape"]:
            n *= x
        d = out.setdefault(v.get("dtype", "?"), [0, 0])
        d[0] += 1
        d[1] += n
    return out


def probe(repo: str, token):
    """Everything we learn about one repo. Runs in a child process."""
    from ai_model_explorer._hub import Hub
    from ai_model_explorer.api import _remote
    t0 = time.time()
    hub = Hub(token)
    info = hub.model_info(repo)
    names = [x.get("rfilename", "") for x in info.get("siblings", [])]
    if not any(n.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt")) for n in names):
        kind = ("adapter" if any(n.endswith("adapter_config.json") for n in names) else
                "gguf_only" if any(n.endswith(".gguf") for n in names) else
                "onnx_only" if any(n.endswith(".onnx") for n in names) else
                "mlpackage_only" if any(".mlpackage/" in n or ".mlmodelc/" in n for n in names) else
                "keras_or_other" if any(n.endswith((".h5", ".keras", ".msgpack", ".zip", ".npz", ".nemo", ".tflite")) for n in names) else
                "no_weight_files")
        return {"status": "no_weights", "no_weights_kind": kind, "files": len(names)}
    if any(n.endswith("adapter_model.safetensors") for n in names) and not any(
            n.endswith(".safetensors") and "adapter" not in n for n in names):
        return {"status": "no_weights", "no_weights_kind": "adapter", "files": len(names)}
    tensors, cfg, read, nfiles, extra = _remote(repo, hub, info)
    t_fetch = time.time() - t0
    eng = extra.pop("engine")
    meta = {"files_read": nfiles, "header_bytes": read, "weight_bytes": extra.get("weight_bytes"),
            "hub_params": extra.get("hub_params"), "duplicates_skipped": extra.get("duplicates_skipped")}
    out = analyze(eng, tensors, cfg, meta)
    out.update({"rate_limit_wait_s": round(hub.waited, 1), "t_fetch_s": round(t_fetch, 2), "t_total_s": round(time.time() - t0, 2)})
    # the raw manifest and config, so later rule changes can be re-scored offline (rescore.py) without crawling again
    out["_raw"] = {"tensors": {n: [v.get("dtype"), v.get("shape")] for n, v in tensors.items() if isinstance(v, dict)},
                   "config": cfg, "meta": meta}
    return out


def analyze(eng, tensors, cfg, meta):
    """Everything the weights (and config, when present) say. Shared by crawl.py and rescore.py."""
    an = eng.infer(tensors)                       # manifest only: no config, no code
    fam_manifest = an.get("family")
    hint = eng.quant_hint(cfg)
    if cfg and any(hint.values()) and an.get("quant"):  # bits/group from config change the unpacked shapes: count again with them
        an = eng.infer(tensors, qhint=hint)
    check = eng.fuse(an, cfg) if cfg else None
    fam_cfg = eng.family({"family": fam_manifest, "comps": an.get("comps")}, cfg) if cfg else None
    tot = an.get("totals") or {}
    hub_params = meta.get("hub_params")
    checked = [v for v in (check or {}).values() if isinstance(v, dict) and v.get("checked")]
    fused = {}
    if check:
        b = next((c for c in an.get("comps") or [] if c.get("role") == "backbone"), None)
        fused = (check.get(b["id"]) if b else None) or {}
    cfg = cfg or {}
    return {
        "config_hp": config_hp(cfg),
        "hp_fused": {k: fused.get(k) for k in ("heads", "kv_heads", "head_dim", "topk")} if fused else None,
        "dtypes": dtype_stats(tensors),
        "status": "ok",
        **meta,
        "tensors": len(tensors),
        "family_manifest": fam_manifest, "family_config": fam_cfg,
        "params": tot.get("params"), "stored_bytes": tot.get("bytes"),
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
            raw = res.pop("_raw", None)
            if raw is not None:
                with gzip.open(out.replace(".jsonl", "") + "_manifests.jsonl.gz", "at") as mz:
                    mz.write(json.dumps({"id": rid, **raw}, default=str) + "\n")
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
