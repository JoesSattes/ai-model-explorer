"""Practical questions about one or many models: will it fit on this device, what changed between two checkpoints,
and a whole list of repos in one table."""
from __future__ import annotations

import csv
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Union

from .model import Model, explore, fmt_b, fmt_n

# memory a model may use, in GB. Unified-memory machines keep ~25% for the system.
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
    buffers. Also returns the longest context that fits and cheaper settings that would make it fit."""
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
