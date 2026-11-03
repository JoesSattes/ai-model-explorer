"""Read one submitted Hugging Face repo and score how ready it is to deploy. Used by the model-submission workflow.

usage: python submit.py owner/name ISSUE_NUMBER BOARD.json BADGE_DIR > comment.md
Writes or updates the repo's row in BOARD.json and a shields.io endpoint file in BADGE_DIR. Prints the reply comment.
Exit code 0 when the repo was read, 2 when it could not be (the comment explains why).
"""
import datetime, json, multiprocessing as mp, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels  # noqa: E402

TIERS = [(4.8e9, "Phone"), (12e9, "Laptop"), (24e9, "Desktop GPU"), (80e9, "Workstation"), (float("inf"), "Cluster")]
COLORS = {"Phone": "2ea44f", "Laptop": "44cc11", "Desktop GPU": "dfb317", "Workstation": "fe7d37", "Cluster": "e05d44"}
SITE = "https://joessattes.github.io/ai-model-explorer"


def fmt_n(n):
    for u, k in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n and abs(n) >= k:
            return f"{n / k:.1f}{u}"
    return str(n)


def fmt_b(n):
    for u, k in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n and abs(n) >= k:
            return f"{n / k:.1f} {u}"
    return f"{n} B"


def _child(repo, q):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from crawl import probe
    from ai_model_explorer._hub import Hub
    try:
        r = probe(repo, os.environ.get("HF_TOKEN"))
        r.pop("_raw", None)
        try:
            info = Hub(os.environ.get("HF_TOKEN")).model_info(repo)
            r["license"] = (info.get("cardData") or {}).get("license")
            r["downloads"], r["likes"] = info.get("downloads"), info.get("likes")
            r["tags"], r["created_at"] = info.get("tags") or [], info.get("createdAt")
            r["pipeline_tag"] = r.get("pipeline_tag") or info.get("pipeline_tag")
            r["origin"] = r.get("origin") or labels.hub_origin(r["tags"])
        except Exception:
            pass
        q.put(r)
    except PermissionError:
        q.put({"status": "gated"})
    except Exception as e:  # noqa: BLE001
        q.put({"status": "error", "error": f"{type(e).__name__}: {e}"[:300]})


def read(repo, timeout=240):
    q = mp.Queue()
    p = mp.Process(target=_child, args=(repo, q), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        return {"status": "timeout"}
    return q.get() if not q.empty() else {"status": "error", "error": f"worker exited {p.exitcode}"}


def score(r):
    h, f = r.get("hp") or {}, r.get("hp_fused") or {}
    params, kv = r.get("params") or 0, h.get("kv_bytes_per_token_bf16") or 0
    e, k, ex = h.get("experts"), f.get("topk") or h.get("experts_per_token"), h.get("expert_params")
    act = params if not e else (int(params - ex * (1 - k / e)) if k and ex else None)
    need = (params * 0.5 + kv * 8192) * 1.1                     # 4-bit weights, 8k context, 10% runtime
    tier = next(name for lim, name in TIERS if need <= lim)
    hub = r.get("hub_params")
    checks = {"safetensors": (r.get("files_read") or 0) > 0,
              "config.json": bool(r.get("has_config")),
              "config agrees with weights": (r.get("config_checked") or 0) > 0 and r.get("config_checked") == r.get("config_consistent"),
              "Hub parameter count correct": bool(hub) and abs(params - hub) / hub <= 0.001,
              "license declared": bool(r.get("license"))}
    q = r.get("quant")
    return {"family": r.get("family_manifest"), "params": params, "active": act, "stored_bytes": r.get("stored_bytes"),
            "storage": "+".join(sorted(q)) if isinstance(q, dict) and q else "unquantized", "d": h.get("d"), "layers": h.get("layers"),
            "heads": f.get("heads") or h.get("heads"), "kv_heads": f.get("kv_heads") or h.get("kv_heads"), "experts": e,
            "vocab": h.get("vocab"), "kv_bytes_per_token": kv, "ssm_layers": h.get("ssm_layers") or 0,
            "need_4bit_8k_bytes": int(need), "tier": tier, "checks": checks, "score": sum(checks.values()),
            "license": r.get("license"), "downloads": r.get("downloads"), "likes": r.get("likes"), "header_bytes": r.get("header_bytes")}


def main(repo, issue, board_path, badge_dir):
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,95}", repo or ""):
        print(f"I could not read a repo id from this issue. Please edit the issue so the **Hugging Face repo** field is `owner/name`.")
        return 2
    r = read(repo)
    if r.get("status") != "ok":
        why = {"gated": "it is gated or private, so its headers cannot be read without accepting its license",
               "no_weights": "it has no safetensors or PyTorch weights (GGUF-only and adapter-only repos are not supported yet)",
               "not_found": "it does not exist (check the spelling)", "timeout": "reading it took longer than four minutes"}
        print(f"I could not add **{repo}**: {why.get(r.get('status'), 'reading it failed')}.")
        if r.get("error"):
            print(f"\n```\n{r['error']}\n```")
        return 2
    s = score(r)
    board = json.load(open(board_path)) if os.path.exists(board_path) else []
    board = [b for b in board if b.get("id") != repo]
    tags = r.get("tags") or []
    board.append({"id": repo, "issue": int(issue), "submitted": datetime.date.today().isoformat(), "src": "submitted",
                  "th": labels.thai(repo, tags, s["family"]), "origin": labels.origin(r.get("origin"), r.get("quant") or {}, repo, tags),
                  "lab": labels.is_lab(repo), "pt": r.get("pipeline_tag"), **s})
    board.sort(key=lambda b: (-b["score"], b["need_4bit_8k_bytes"]))
    json.dump(board, open(board_path, "w"), indent=1)
    os.makedirs(badge_dir, exist_ok=True)
    slug = repo.replace("/", "--")
    json.dump({"schemaVersion": 1, "label": "deploy-ready", "message": f"{s['tier']} · {s['score']}/5",
               "color": COLORS[s["tier"]]}, open(os.path.join(badge_dir, slug + ".json"), "w"))
    badge = f"https://img.shields.io/endpoint?url={SITE}/census/badge/{slug}.json"
    page = f"{SITE}/census/?m={repo}"
    tick = lambda b: "✅" if b else "❌"
    print(f"**{repo}** is on the board.\n")
    print(f"| | |\n|---|---|\n| Family | {s['family']} |\n| Parameters | {fmt_n(s['params'])}" +
          (f" ({fmt_n(s['active'])} active per token)" if s["active"] and s["active"] != s["params"] else "") + " |")
    print(f"| Stored | {fmt_b(s['stored_bytes'] or 0)} ({s['storage']}) |\n| Shape | width {s['d']} · {s['layers']} layers · heads {s['heads']}/{s['kv_heads']}" +
          (f" · {s['experts']} experts" if s["experts"] else "") + " |")
    print(f"| KV cache | {fmt_b(s['kv_bytes_per_token'])} per token |\n| Memory at 4-bit, 8k context | {fmt_b(s['need_4bit_8k_bytes'])} → **{s['tier']}** |\n")
    print("Checks: " + " · ".join(f"{tick(v)} {k}" for k, v in s["checks"].items()) + f" (**{s['score']}/5**)\n")
    print(f"Badge for your model card (it updates when you resubmit):\n\n```markdown\n[![deploy-ready]({badge})]({page})\n```\n")
    print(f"Full anatomy: {SITE}/en/#hf={repo} · Board: {SITE}/census/#board\n\n<sub>Read from {fmt_b(s['header_bytes'] or 0)} of headers; no weights were downloaded.</sub>")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
