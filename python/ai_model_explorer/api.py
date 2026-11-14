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


# ggml tensor types: name, elements per block, bytes per block
_GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
         8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
         13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
         18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
         23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
         29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17)}


def _gguf_inspect(path: Path) -> dict:
    """The token tables and sizes of a GGUF file, from its header (tensor infos); no tensor data is read.
    Enough for vocab_tax(): llama.cpp names the tables token_embd, output (absent when tied) and per_layer_token_embd."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        ver = struct.unpack("<I", f.read(4))[0]
        rn = (lambda: struct.unpack("<Q", f.read(8))[0]) if ver >= 2 else (lambda: struct.unpack("<I", f.read(4))[0])
        n_t, n_kv = rn(), rn()
        SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        rs = lambda: f.read(rn()).decode("utf-8", errors="replace")
        kv = {}

        def val(t):
            if t == 8:
                return rs()
            if t == 9:
                et, n = struct.unpack("<I", f.read(4))[0], rn()
                if et == 8:
                    for _ in range(n):
                        f.seek(rn(), 1)
                elif et == 9:
                    for _ in range(n):
                        val(9)
                else:
                    f.seek(SZ[et] * n, 1)
                return None
            b = f.read(SZ[t])
            return struct.unpack("<" + {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t], b)[0]
        for _ in range(n_kv):
            k = rs(); kv[k] = val(struct.unpack("<I", f.read(4))[0])
        tens = []
        for _ in range(n_t):
            name = rs(); nd = struct.unpack("<I", f.read(4))[0]
            dims = [rn() for _ in range(nd)]
            typ = struct.unpack("<I", f.read(4))[0]; f.read(8)
            tn, bs, bb = _GGML.get(typ, (f"type{typ}", 1, 2))
            n = 1
            for d in dims:
                n *= d
            tens.append({"name": name, "dims": dims, "dtype": tn, "params": n, "bytes": n // bs * bb})
    arch = kv.get("general.architecture")
    by = {t["name"]: t for t in tens}
    emb = by.get("token_embd.weight")
    tables, kinds = [], {"embed": {"params": 0, "bytes": 0}, "head": {"params": 0, "bytes": 0}, "other": {"params": 0, "bytes": 0}}
    V = emb["dims"][1] if emb and len(emb["dims"]) > 1 else 0
    for nm, kind, what in (("token_embd.weight", "embed", "input embedding"), ("output.weight", "head", "output head (untied)"),
                           ("per_layer_token_embd.weight", "embed", "per-layer embeddings")):
        t = by.get(nm)
        if t and len(t["dims"]) > 1 and t["dims"][1] == V:
            tables.append({"name": nm, "what": what, "kind": kind, "width": t["dims"][0], "params": t["params"],
                           "dtype": t["dtype"], "bpp": t["bytes"] / t["params"]})
    for t in tens:
        k = next((x["kind"] for x in tables if x["name"] == t["name"]), "other")
        kinds[k]["params"] += t["params"]; kinds[k]["bytes"] += t["bytes"]
    dt = {}
    for t in tens:
        dt[t["dtype"]] = dt.get(t["dtype"], 0) + t["bytes"]
    return {"source": {"kind": "local", "path": str(path.resolve()), "format": "gguf"}, "family": arch,
            "name": kv.get("general.name"), "totals": {"params": sum(t["params"] for t in tens), "bytes": sum(t["bytes"] for t in tens)},
            "breakdown": {"by_kind": {k: v for k, v in kinds.items() if v["params"]}},
            "plan": {"vocab": {"size": V, "tables": tables, "tied": "output.weight" not in by}} if tables else {},
            "quantization": {"gguf_types_by_bytes": dict(sorted(dt.items(), key=lambda x: -x[1]))},
            "hyperparameters": {k.split(".", 1)[1]: v for k, v in kv.items() if arch and k.startswith(arch + ".") and isinstance(v, (int, float))},
            "tensors_read": len(tens), "note": "GGUF: token tables and sizes from the header; the full anatomy reads safetensors"}


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
    if os.path.isfile(s) and s.lower().endswith(".gguf"):
        return _gguf_inspect(Path(s))
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
    ffn_by = {}
    for n, t in roles.items():
        if t.get("kind") in ("ffn", "moe") and t.get("comp"):
            ffn_by[t["comp"]] = ffn_by.get(t["comp"], 0) + _prod(logical.get(n, [[], 0])[0])
    try:
        out["plan"] = eng.plan({"comps": anatomy.get("comps"), "hp": anatomy.get("hp"), "edges": anatomy.get("edges"), "family": anatomy.get("family"), "tensors": keep},
                               {n: logical.get(n, [[], 0])[0] for n in keep}, cfg,
                               {"total": (anatomy.get("totals") or {}).get("params"), "ffn_by_comp": ffn_by, "name": source.get("repo") or source.get("path") or "",
                                "moe_params": sum(_prod(logical.get(n, [[], 0])[0]) for n, t in roles.items() if t.get("kind") == "moe" and t.get("comp") == back)})
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
    enc = [c for c in a.get("components") or [] if c.get("role") == "encoder" and c.get("layers")]
    if a.get("family") == "encoder-decoder" and enc and hp.get("layers"):
        hps = hps.replace(f"layers {hp['layers']}", f"layers {enc[0]['layers']} encoder + {hp['layers']} decoder", 1)
    if hps:
        lines.append("  " + hps)
    lines.append("  components:")
    for c in a.get("components") or []:
        role = c.get("role") or ""
        if role == "backbone" and a.get("family") == "encoder-decoder":
            role = "decoder"
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
