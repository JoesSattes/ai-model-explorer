"""Practical questions about one or many models: whether it fits a device, what differs between two checkpoints,
many repos in one table, the Deploy-Ready board's verdict, and where a model sits in the Hub Weight Census."""
from __future__ import annotations

import csv
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Union

from .model import Model, explore, fmt_b, fmt_n

# memory a model may use, in GB. Unified-memory devices count only the usable share: 75% (Mac, Jetson),
# 60% (Raspberry Pi), 40% (phone).
DEVICES = {
    "rtx-3060": 12, "rtx-4060": 8, "rtx-4060-ti-16": 16, "rtx-4070": 12, "rtx-4080": 16, "rtx-4090": 24, "rtx-5090": 32,
    "t4": 16, "l4": 24, "a10g": 24, "l40s": 48, "a100-40": 40, "a100-80": 80, "h100": 80, "h200": 141, "b200": 192,
    "mi300x": 192, "jetson-orin-nano": 8 * 0.75, "jetson-agx-orin-64": 64 * 0.75, "mac-16": 16 * 0.75, "mac-24": 24 * 0.75,
    "mac-36": 36 * 0.75, "mac-48": 48 * 0.75, "mac-64": 64 * 0.75, "mac-128": 128 * 0.75, "mac-192": 192 * 0.75,
    "raspberry-pi-5-8": 8 * 0.6, "phone-12": 12 * 0.4,
}


def fits(model: Union[str, Model], device: Union[str, float], context: int = 8192, batch: int = 1,
         weight_bits: Optional[float] = None, kv_bits: int = 16, overhead: float = 0.10, token: Optional[str] = None) -> dict:
    """Does the model fit in `device` (a name from DEVICES, or GB)? Weights + KV cache + `overhead` for runtime
    buffers. Also returns the longest context that fits and, if it does not fit, the first of
    8-bit weights, 4-bit weights, 4-bit weights with 8-bit KV cache, or 4-bit weights with 4-bit KV cache that would."""
    m = model if isinstance(model, Model) else explore(model, token=token)
    have = (DEVICES[device.lower()] if isinstance(device, str) else float(device)) * 1e9
    mem = m.memory(context=context, batch=batch, weight_bits=weight_bits, kv_bits=kv_bits)
    need = mem["total_bytes"] * (1 + overhead)
    kvt = m.kv_cache_bytes_per_token(kv_bits)
    room = have / (1 + overhead) - mem["weights_bytes"]
    max_ctx = int(room / (kvt * batch)) if kvt and room > 0 else (0 if room <= 0 else None)
    tries = []
    if need > have:
        for wb, kb in ((8, 16), (4, 16), (4, 8), (4, 4)):
            r = m.memory(context=context, batch=batch, weight_bits=wb, kv_bits=kb)
            if r["total_bytes"] * (1 + overhead) <= have:
                tries.append(f"{wb}-bit weights, {kb}-bit KV cache: {fmt_b(r['total_bytes'] * (1 + overhead))}")
                break
    return {"model": m.name, "device": device, "fits": need <= have, "need_bytes": int(need), "have_bytes": int(have),
            "headroom_bytes": int(have - need), "max_context": max_ctx, "suggestion": tries[0] if tries else None, **mem}


def fits_text(r: dict) -> str:
    ok = "fits" if r["fits"] else "does NOT fit"
    dev = r["device"] if isinstance(r["device"], str) else f"{r['device']:g} GB"
    out = [f"{r['model']} on {dev}: {ok}  (needs {fmt_b(r['need_bytes'])} of {fmt_b(r['have_bytes'])})"]
    if r["max_context"] == 0:
        out.append("  the weights alone are larger than the memory at this precision")
    elif r["max_context"] is not None:
        out.append(f"  longest context that fits at this setting: {r['max_context']:,} tokens")
    if r["suggestion"]:
        out.append(f"  would fit with {r['suggestion']}")
    return "\n".join(out)


def diff(a: Union[str, Model], b: Union[str, Model], token: Optional[str] = None) -> List[dict]:
    """What differs between two checkpoints: family, size, storage, every hyperparameter, components and block types.
    Useful for a fine-tune vs its base, a quantized copy vs the original, or two versions of a model."""
    A = a if isinstance(a, Model) else explore(a, token=token)
    B = b if isinstance(b, Model) else explore(b, token=token)
    rows = []

    def add(field, x, y, fmt=str):
        if x != y:
            rows.append({"field": field, "a": fmt(x) if x is not None else "-", "b": fmt(y) if y is not None else "-"})
    add("family", A.family, B.family)
    add("parameters", A.params, B.params, fmt_n)
    add("stored size", A.stored_bytes, B.stored_bytes, fmt_b)
    add("quantization", ", ".join(sorted(A.data.get("quantization") or {})) or "none",
        ", ".join(sorted(B.data.get("quantization") or {})) or "none")
    for k in sorted(set(A.hp) | set(B.hp)):
        if k.endswith("_from") or isinstance(A.hp.get(k), (dict, list)) or isinstance(B.hp.get(k), (dict, list)):
            continue
        add(k, A.hp.get(k), B.hp.get(k), fmt_b if k.startswith("kv_bytes") else fmt_n if k.endswith("_params") else str)
    add("components", ", ".join(c["id"] for c in A.components), ", ".join(c["id"] for c in B.components))
    lab = lambda m: ", ".join(sorted({t.get("label", "?") for c in m.components for s in c.get("stacks", []) for t in s.get("types", [])}))
    add("block types", lab(A), lab(B))
    return rows


def diff_text(rows: List[dict], a: str, b: str) -> str:
    if not rows:
        return f"{a} and {b}: same architecture, size and storage."
    w = max(len(r["field"]) for r in rows)
    return "\n".join([f"{'':{w}}  {a}  →  {b}"] + [f"{r['field']:{w}}  {r['a']}  →  {r['b']}" for r in rows])


SCAN_FIELDS = ["repo", "family", "params", "active_params", "stored_bytes", "quantization", "d", "layers", "heads", "kv_heads",
               "head_dim", "experts", "vocab", "kv_bytes_per_token", "hub_params", "matches_hub", "has_config",
               "config_consistent", "header_bytes", "error"]


def scan(repos: Iterable[str], token: Optional[str] = None, workers: int = 4) -> List[dict]:
    """Inspect many repos (in parallel) and return one flat row each; failures become rows with `error`."""
    def one(repo):
        try:
            m = explore(repo, token=token)
            hp, d, hub = m.hp, m.data, (m.data.get("source") or {}).get("hub_params")
            chk = [v for v in (d.get("config_check") or {}).values() if isinstance(v, dict) and v.get("checked")]
            return {"repo": repo, "family": m.family, "params": m.params, "active_params": m.active_params,
                    "stored_bytes": m.stored_bytes, "quantization": "+".join(sorted(d.get("quantization") or {})) or "",
                    "d": hp.get("d"), "layers": hp.get("layers"), "heads": hp.get("heads"), "kv_heads": hp.get("kv_heads"),
                    "head_dim": hp.get("head_dim"), "experts": hp.get("experts"), "vocab": hp.get("vocab"),
                    "kv_bytes_per_token": m.kv_cache_bytes_per_token(16), "hub_params": hub,
                    "matches_hub": (abs(m.params - hub) / hub <= 0.001) if hub else None, "has_config": d.get("has_config"),
                    "config_consistent": all(v.get("consistent") for v in chk) if chk else None,
                    "header_bytes": d.get("header_bytes"), "error": None}
        except Exception as e:  # noqa: BLE001 - a bad repo is a row, not a crash
            return {"repo": repo, "error": f"{type(e).__name__}: {e}"[:200]}
    repos = [r.strip() for r in repos if r and r.strip() and not r.strip().startswith("#")]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(one, repos))


def to_csv(rows: List[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SCAN_FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


# ---- the Deploy-Ready board, locally
TIERS = [(4.8e9, "Phone"), (12e9, "Laptop"), (24e9, "Desktop GPU"), (80e9, "Workstation"), (float("inf"), "Cluster")]
SITE = "https://joessattes.github.io/ai-model-explorer"
SUBMIT = "https://github.com/JoesSattes/ai-model-explorer/issues/new?template=submit-model.yml"


def ready(model: Union[str, Model], token: Optional[str] = None) -> dict:
    """The Deploy-Ready board's verdict for one model, computed on your machine: the smallest device class it fits at
    4-bit weights with an 8k context (+10% runtime) and five metadata checks. Same rules as the board."""
    m = model if isinstance(model, Model) else explore(model, token=token)
    d, src = m.data, m.data.get("source") or {}
    kv = m.kv_cache_bytes_per_token(16) or 0
    need = (m.params * 0.5 + kv * 8192) * 1.1
    tier = next(name for lim, name in TIERS if need <= lim)
    chk = [v for v in (d.get("config_check") or {}).values() if isinstance(v, dict) and v.get("checked")]
    hub, repo = src.get("hub_params"), src.get("repo")
    on_hub = src.get("kind") == "huggingface"
    checks = {"safetensors": (d.get("files_read") or 0) > 0, "config.json": bool(d.get("has_config")),
              "config agrees with weights": bool(chk) and all(v.get("consistent") for v in chk),
              # a local folder has no Hub metadata: these two do not apply (None) rather than fail
              "Hub parameter count correct": (bool(hub) and abs(m.params - hub) / hub <= 0.001) if on_hub else None,
              "license declared": bool(src.get("license")) if on_hub else None}
    return {"model": m.name, "tier": tier, "need_4bit_8k_bytes": int(need), "params": m.params, "active_params": m.active_params,
            "kv_bytes_per_token": kv, "checks": checks, "score": sum(1 for v in checks.values() if v),
            "out_of": sum(1 for v in checks.values() if v is not None),
            "submit_url": f"{SUBMIT}&repo={repo}&title=%5BModel%5D%20{repo}" if repo else None}


def ready_text(r: dict) -> str:
    out = [f"{r['model']}: fits on {r['tier']}  (4-bit weights, 8k context: {fmt_b(r['need_4bit_8k_bytes'])})",
           f"  {fmt_n(r['params'])} parameters" + (f", {fmt_n(r['active_params'])} active per token"
                                                    if r["active_params"] and r["active_params"] != r["params"] else "")
           + (f", KV cache {fmt_b(r['kv_bytes_per_token'])} per token" if r["kv_bytes_per_token"] else "")]
    out += [f"  {'–' if v is None else '✓' if v else '✗'} {k}" + ("  (not on the Hub)" if v is None else "") for k, v in r["checks"].items()]
    out.append(f"  score {r['score']}/{r['out_of']}")
    if r["submit_url"]:
        out.append(f"  put it on the board and get a badge: {r['submit_url']}")
    return "\n".join(out)


def _census_rows(refresh: bool = False) -> List[dict]:
    """The census table the website shows (popular readable models), cached for a day."""
    import json, os, time, urllib.request
    from .api import CACHE_DIR
    p = CACHE_DIR / "census-models.json"
    if not refresh and p.exists() and time.time() - p.stat().st_mtime < 86400:
        return json.loads(p.read_text(encoding="utf-8"))
    req = urllib.request.Request(f"{SITE}/census/data/models.json", headers={"User-Agent": "ai-model-explorer"})
    data = urllib.request.urlopen(req, timeout=60).read()
    rows = json.loads(data)
    if not os.environ.get("AME_NO_CACHE"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return rows


def census(model: Union[str, Model], token: Optional[str] = None, rows: Optional[List[dict]] = None, n: int = 5) -> dict:
    """Where a model sits among the popular language models of the Hub Weight Census: size and KV-cache percentiles,
    and the popular original releases closest to it in parameters and KV cache per token."""
    import math
    m = model if isinstance(model, Model) else explore(model, token=token)
    rows = rows if rows is not None else _census_rows()
    # the census definition of a language model: decoder or prefix-LM anatomy above 50M parameters whose Hub pipeline
    # generates text (or none is declared); the pipeline removes dual encoders and CTC models the family rules call decoders
    gen = {"", "text-generation", "text2text-generation", "image-text-to-text", "any-to-any", "audio-text-to-text",
           "video-text-to-text", "visual-question-answering", "conversational"}
    lm = [r for r in rows if r.get("fam") in ("decoder", "prefix-lm") and (r.get("p") or 0) > 5e7 and r.get("kv")
          and (r.get("pt") or "") in gen]
    kv = m.kv_cache_bytes_per_token(16)
    pct = lambda xs, v: round(100 * sum(1 for x in xs if x < v) / len(xs)) if xs and v else None
    near = []
    if kv:
        dist = lambda r: math.hypot(math.log10(r["p"] / m.params), math.log10(r["kv"] / kv))
        near = sorted((r for r in lm if r["id"] != m.name and r.get("origin") == "original"), key=dist)[:n]
    me = next((r for r in rows if r["id"] == m.name), None)
    return {"model": m.name, "in_census": me is not None, "release": (me or {}).get("origin"), "provider": (me or {}).get("lab"),
            "thai": (me or {}).get("th"), "params_percentile": pct([r["p"] for r in lm], m.params),
            "kv_percentile": pct([r["kv"] for r in lm], kv), "popular_models": len(lm),
            "nearest": [{"id": r["id"], "params": r["p"], "kv_bytes_per_token": r["kv"], "downloads": r["dl"]} for r in near],
            "page": f"{SITE}/census/?m={m.name}" if (m.data.get("source") or {}).get("kind") == "huggingface" else None}


def census_text(r: dict) -> str:
    out = [r["model"] + ("" if r["in_census"] else "  (not in the census table; compared anyway)")]
    if r["in_census"]:
        out.append("  " + ", ".join(x for x in [r["release"] and f"release: {r['release']}", r["provider"] and "leading provider",
                                                r["thai"] and f"Thai: {r['thai']}"] if x))
    if r["params_percentile"] is not None:
        out.append(f"  larger than {r['params_percentile']}% of {r['popular_models']:,} popular language models")
    if r["kv_percentile"] is not None:
        out.append(f"  KV cache per token larger than {r['kv_percentile']}% of them")
    if r["nearest"]:
        out.append("  closest popular originals (size and KV cache):")
        out += [f"    {x['id']:48} {fmt_n(x['params']):>7}  {fmt_b(x['kv_bytes_per_token']):>9}/token  {fmt_n(x['downloads']):>6} downloads"
                for x in r["nearest"]]
    if r["page"]:
        out.append(f"  on the web: {r['page']}")
    return "\n".join(out)
