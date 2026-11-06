"""inspect(): the anatomy of a model from its weight manifest, for a Hugging Face repo or local safetensors files."""
from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from typing import Optional, Union

from ._engine import Engine
from ._hub import Hub


def _local_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 0 < n < 200_000_000:
            raise ValueError(f"{path} is not a safetensors file")
        h = json.loads(f.read(n).decode("utf-8"))
    h.pop("__metadata__", None)
    return h, 8 + n


def _local(path: Path):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*.safetensors") if not re.search(r"(^|/)(optimizer|adapter[._])", str(p)))
    if not files:
        raise FileNotFoundError(f"no .safetensors file under {path}")
    tensors, read = {}, 0
    base = path if path.is_dir() else path.parent
    for p in files:
        h, b = _local_header(p)
        read += b
        rel = p.relative_to(base).parent.as_posix()
        prefix = "" if rel in ("", ".") else rel.replace("/", ".") + "."
        for k, v in h.items():
            tensors[prefix + k] = {"dtype": v["dtype"], "shape": v["shape"]}
    cfg = None
    c = base / "config.json"
    if c.exists():
        cfg = json.loads(c.read_text(encoding="utf-8"))
    return tensors, cfg, read, len(files), None


CACHE_DIR = Path(os.environ.get("AME_CACHE", Path.home() / ".cache" / "ai-model-explorer"))


def _cache_path(repo: str, sha: Optional[str]) -> Optional[Path]:
    if not sha or os.environ.get("AME_NO_CACHE"):
        return None
    return CACHE_DIR / (repo.replace("/", "--") + "@" + sha[:12] + ".json")


def _remote(repo: str, hub: Hub, info: Optional[dict] = None, cache: bool = True):
    info = info or hub.model_info(repo)
    extra_info = {"downloads": info.get("downloads"), "hub_params": (info.get("safetensors") or {}).get("total"),
                  "weight_bytes": sum(s.get("size") or 0 for s in info.get("siblings", [])), "revision": info.get("sha"),
                  "license": (info.get("cardData") or {}).get("license"), "pipeline_tag": info.get("pipeline_tag"), "tags": info.get("tags")}
    cp = _cache_path(repo, info.get("sha")) if cache else None
    if cp and cp.exists():  # same commit, same headers: nothing to fetch
        try:
            c = json.loads(cp.read_text(encoding="utf-8"))
            return c["tensors"], c["cfg"], c["read"], c["nfiles"], {"engine": Engine(fetch=hub.get), **extra_info,
                                                                    "duplicates_skipped": c.get("dups"), "cached": True}
        except Exception:  # a broken cache file is simply refetched
            pass
    files = [{"n": s["rfilename"], "size": s.get("size") or 0} for s in info.get("siblings", [])]
    names = {f["n"] for f in files}
    base = f"https://huggingface.co/{repo}/resolve/main/"
    cfg = mi = None
    if "config.json" in names:
        try:
            cfg = hub.json(base + "config.json")
        except Exception:
            cfg = None
    if "model_index.json" in names:
        try:
            mi = hub.json(base + "model_index.json")
        except Exception:
            mi = None
    opt = {"prefer": (cfg or {}).get("default_size"),
           "only": [k for k, v in (mi or {}).items() if not k.startswith("_") and isinstance(v, list) and v and v[0]] if mi else None}
    eng = Engine(fetch=hub.get)
    st = [f["n"] for f in files if f["n"].endswith(".safetensors") and not re.search(r"(^|/)(checkpoint-\d+|runs|global_step\d+)/", f["n"])]
    if 1 < len(st) <= 96:
        hub.prefetch_headers([base + n for n in st])
    mf = eng.hf_manifest(repo, files, opt)
    if not mf:
        raise FileNotFoundError(f"{repo}: no .safetensors or PyTorch checkpoint found")
    if cp:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps({"tensors": mf["tensors"], "cfg": cfg, "read": mf["read"], "nfiles": mf["nfiles"],
                                      "dups": mf.get("duplicates") or None}), encoding="utf-8")
        except OSError:
            pass
    return mf["tensors"], cfg, mf["read"], mf["nfiles"], {"engine": eng, **extra_info, "duplicates_skipped": mf.get("duplicates") or None}


def inspect(model: Union[str, os.PathLike], token: Optional[str] = None, config: Optional[dict] = None,
            include_tensors: bool = False, cache: bool = True) -> dict:
    """Recover a model's anatomy without downloading its weights.

    model:  a Hugging Face repo id ("Qwen/Qwen3-8B", or a huggingface.co URL), a .safetensors file, or a folder of them.
    token:  Hugging Face token for gated or private repos (default: $HF_TOKEN).
    config: a config.json dict to fuse instead of the repo's own.
    cache:  keep the headers of each repo commit under ~/.cache/ai-model-explorer (or $AME_CACHE) and reuse them.
    Returns a JSON-serializable dict: family, totals, hyperparameters, components with their layer stacks and block
    types, edges between components, quantization, the config check, and how many bytes were read.
    """
    s = str(model)
    if os.path.exists(s):
        tensors, cfg, read, nfiles, extra = _local(Path(s))
        eng = Engine()
        source = {"kind": "local", "path": str(Path(s).resolve())}
    else:
        m = re.search(r"(?:huggingface\.co/)?([\w.-]+/[\w.-]+)", s)
        if not m:
            raise ValueError("expected a Hugging Face repo id (owner/name) or a path to .safetensors")
        repo = re.sub(r"/(tree|blob|resolve)/.*$", "", m.group(1))
        tensors, cfg, read, nfiles, extra = _remote(repo, Hub(token), cache=cache)
        eng = extra.pop("engine")
        source = {"kind": "huggingface", "repo": repo, **{k: v for k, v in extra.items() if v is not None}}
    if config is not None:
        cfg = config
    anatomy, logical = eng.analyze(tensors, qhint=eng.quant_hint(cfg))
    check = eng.fuse(anatomy, cfg) if cfg else None
    if cfg:
        anatomy["family"] = eng.family({"family": anatomy.get("family"), "comps": anatomy.get("comps")}, cfg)
    out = {
        "source": source,
        "files_read": nfiles,
        "header_bytes": read,
        "tensor_count": len(tensors),
        "family": anatomy.get("family"),
        "totals": anatomy.get("totals"),
        "hyperparameters": anatomy.get("hp"),
        "quantization": anatomy.get("quant"),
        "components": anatomy.get("comps"),
        "edges": anatomy.get("edges"),
        "evidence": anatomy.get("evidence"),
        "config_check": check,
        "has_config": cfg is not None,
    }
    out["hyperparameters_with_config"] = _merge_hp(anatomy, check)
    out["breakdown"] = _breakdown(anatomy.get("tensors") or {}, logical, anatomy.get("comps") or [])
    out["config"] = cfg
    roles = anatomy.get("tensors") or {}
    keep = {n: t for n, t in roles.items() if t.get("kind") in ("embed", "head") or (t.get("kind") == "attn" and t.get("role") in ("k", "v"))}
    back = next((c["id"] for c in anatomy.get("comps") or [] if c.get("role") == "backbone"), None)
    ffn = sum(_prod(logical.get(n, [[], 0])[0]) for n, t in roles.items() if t.get("kind") == "ffn" and t.get("comp") == back)
    try:
        out["plan"] = eng.plan({"comps": anatomy.get("comps"), "hp": anatomy.get("hp"), "edges": anatomy.get("edges"), "family": anatomy.get("family"), "tensors": keep},
                               {n: logical.get(n, [[], 0])[0] for n in keep}, cfg,
                               {"total": (anatomy.get("totals") or {}).get("params"), "ffn_params": ffn})
    except Exception as e:  # the plan is advice; never fail the inspection because of it
        out["plan"] = {"error": str(e), "levers": []}
    if include_tensors:
        out["tensors"] = anatomy.get("tensors")
    return out


def _prod(s) -> int:
    p = 1
    for x in s or []:
        p *= x
    return p


def _breakdown(roles: dict, logical: dict, comps: list) -> dict:
    """Parameters and stored bytes grouped by part type, by component and by layer stack, plus the size of any
    multi-token-prediction layers (used for speculative decoding, not for ordinary next-token prediction)."""
    out = {"by_kind": {}, "by_component": {}, "by_stack": {}, "mtp": {"params": 0, "expert_params": 0}}
    mtp_items = [st["mtpItem"] + "." for c in comps for st in c.get("stacks") or [] if st.get("mtp") and st.get("mtpItem")]
    for name, r in roles.items():
        shape, nbytes = logical.get(name, [[], 0])
        n = _prod(shape)
        if mtp_items and any(name.startswith(p) for p in mtp_items):
            out["mtp"]["params"] += n
            if r.get("role") == "expert":
                out["mtp"]["expert_params"] += n
        for key, val in (("by_kind", r.get("kind") or "other"), ("by_component", r.get("comp") or "(none)"),
                         ("by_stack", r.get("stack") or "(outside stacks)")):
            d = out[key].setdefault(val, {"params": 0, "bytes": 0, "tensors": 0})
            d["params"] += n
            d["bytes"] += nbytes
            d["tensors"] += 1
    for key in ("by_kind", "by_component", "by_stack"):
        out[key] = dict(sorted(out[key].items(), key=lambda kv: -kv[1]["params"]))
    return out


def _merge_hp(anatomy: dict, check) -> dict:
    """Hyperparameters from the weights, with heads / kv heads / head size / experts per token taken from config.json
    for the largest component when the config was checked against the weights and agrees."""
    hp = dict(anatomy.get("hp") or {})
    if not check:
        return hp
    comps = sorted(anatomy.get("comps") or [], key=lambda c: c.get("params") or 0, reverse=True)
    main = check.get(comps[0]["id"]) if comps else None
    main = main or next((v for v in check.values() if isinstance(v, dict)), None)
    if main and main.get("checked") and main.get("consistent"):
        for k_src, k_dst in (("heads", "heads"), ("kv_heads", "kv_heads"), ("head_dim", "head_dim"), ("topk", "experts_per_token")):
            if main.get(k_src):
                hp[k_dst] = main[k_src]
        hp["head_dim_from"] = "config.json, checked against the weights"
    elif main and main.get("skipped"):  # latent attention: head widths cannot be checked, but the counts still come from config
        for k_src, k_dst in (("heads", "heads"), ("topk", "experts_per_token")):
            if main.get(k_src):
                hp[k_dst] = main[k_src]
        hp["head_dim_from"] = "config.json (latent attention: not checkable against the shapes)"
    return hp


def _fmt_n(n) -> str:
    if n is None:
        return "?"
    for u, k in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= k:
            return f"{n / k:.2f}{u}".replace(".00", "")
    return str(n)


def _fmt_b(n) -> str:
    if n is None:
        return "?"
    for u, k in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= k:
            return f"{n / k:.1f} {u}"
    return f"{n} B"


def summary(a: dict) -> str:
    """A short human-readable report of an inspect() result."""
    src = a["source"]
    name = src.get("repo") or src.get("path")
    t = a.get("totals") or {}
    hp = a.get("hyperparameters_with_config") or a.get("hyperparameters") or {}
    lines = [f"{name}",
             f"  family {a.get('family')} · {_fmt_n(t.get('params'))} parameters · {_fmt_b(t.get('bytes'))} of weights"
             + (f" · stored as {', '.join(a['quantization'])}" if a.get("quantization") else ""),
             f"  read {_fmt_b(a.get('header_bytes'))} from {a.get('files_read')} file(s), {a.get('tensor_count')} tensors"]
    keys = [("d", "width"), ("layers", "layers"), ("heads", "heads"), ("kv_heads", "kv heads"), ("head_dim", "head dim"),
            ("ffn", "ffn"), ("experts", "experts"), ("experts_per_token", "active experts"), ("vocab", "vocab")]
    hps = " · ".join(f"{lab} {hp[k]}" for k, lab in keys if hp.get(k))
    if hps:
        lines.append("  " + hps)
    lines.append("  components:")
    for c in a.get("components") or []:
        role = c.get("role") or ""
        head = f"    {c.get('id')}" + (f" ({role})" if role else "") + (f" · d={c['d']}" if c.get("d") else "")
        lines.append(head)
        for st in c.get("stacks") or []:
            tys = st.get("types") or []
            types = ", ".join(f"{ty.get('label')}" + (f" ×{ty['count']}" if ty.get("count") and len(tys) > 1 else "") for ty in tys)
            mtp = " (last one is a multi-token-prediction layer)" if st.get("mtp") else ""
            lines.append(f"      {st.get('path')}: {st.get('n')} blocks · {types}{mtp}")
    for e in a.get("edges") or []:
        lines.append(f"    {e.get('from')} → {e.get('to')}" + (f" (via {e['via']})" if e.get("via") else ""))
    chk = a.get("config_check")
    if chk:
        bad = [k for k, v in chk.items() if isinstance(v, dict) and v.get("checked") and v.get("consistent") is False]
        lines.append("  config.json: " + ("agrees with the weights" if not bad else "disagrees with the weights for " + ", ".join(bad)))
    return "\n".join(lines)
