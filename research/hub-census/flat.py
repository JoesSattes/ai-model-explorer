"""Flat, one-row-per-record parquet tables for the Hugging Face Dataset Viewer.

    python flat.py census out/census.jsonl out/census.parquet
    python flat.py corpus ../data/corpus-510.jsonl.gz out/corpus.parquet
"""
import gzip, json, sys
import pandas as pd


def _hp(r, k):
    h = r.get("hp") or {}
    return h.get(k) if isinstance(h, dict) else None


def census(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        f, c = r.get("hp_fused") or {}, r.get("config_hp") or {}
        q = r.get("quant")
        rows.append({
            "id": r["id"], "status": r.get("status"), "downloads_30d": r.get("downloads"), "downloads_all_time": r.get("downloads_all_time"),
            "likes": r.get("likes"), "created_at": r.get("created_at"), "library": r.get("library"), "pipeline_tag": r.get("pipeline_tag"),
            "origin": r.get("origin"), "stratum": r.get("stratum"), "weight": r.get("weight"), "frontier": bool(r.get("also_frontier")),
            "family": r.get("family_manifest"), "family_config": r.get("family_config"),
            "params": r.get("params"), "hub_params": r.get("hub_params"), "params_match_hub": r.get("params_match_hub"),
            "storage": "+".join(sorted(q)) if isinstance(q, dict) and q else ("unquantized" if r.get("status") == "ok" else None),
            "d": _hp(r, "d"), "layers": _hp(r, "layers"), "heads": f.get("heads") or _hp(r, "heads"),
            "kv_heads": f.get("kv_heads") or _hp(r, "kv_heads"), "head_dim": f.get("head_dim") or _hp(r, "head_dim"),
            "ffn": _hp(r, "ffn"), "experts": _hp(r, "experts"), "experts_per_token": f.get("topk"), "vocab": _hp(r, "vocab"),
            "ssm_layers": _hp(r, "ssm_layers"), "kv_bytes_per_token_bf16": _hp(r, "kv_bytes_per_token_bf16"),
            "has_config": r.get("has_config"), "config_model_type": r.get("config_model_type"),
            "config_architectures": ",".join(r.get("config_architectures") or []) or None,
            "config_d": c.get("d"), "config_layers": c.get("layers"), "config_heads": c.get("heads"), "config_kv_heads": c.get("kv_heads"),
            "config_vocab": c.get("vocab"), "config_experts": c.get("experts"),
            "header_bytes": r.get("header_bytes"), "weight_bytes": r.get("weight_bytes"), "tensors": r.get("tensors"),
            "no_weights_kind": r.get("no_weights_kind"), "error": r.get("error")})
    return pd.DataFrame(rows)


def corpus(path):
    rows = []
    op = gzip.open if path.endswith(".gz") else open
    for l in op(path, "rt"):
        if not l.strip():
            continue
        r = json.loads(l.replace("Infinity", "null").replace("NaN", "null"))
        kinds = {}
        for g in (r.get("gt") or {}).values():
            kinds[g["kind"]] = kinds.get(g["kind"], 0) + 1
        hp = r.get("hp") or {}
        rows.append({"type": r.get("type"), "model_class": r.get("cls"), "params": r.get("params"), "tensors": len(r.get("tensors") or {}),
                     "block_lists": len(r.get("lists") or []), "block_types": len({json.dumps(x.get("cls")) for x in r.get("lists") or []}),
                     **{f"tensors_{k}": v for k, v in sorted(kinds.items())},
                     **{f"config_{k}": v for k, v in hp.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)}})
    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c.startswith("tensors_")]:
        df[c] = df[c].fillna(0).astype("int64")
    return df


if __name__ == "__main__":
    kind, src, dst = sys.argv[1:4]
    df = census(src) if kind == "census" else corpus(src)
    df.to_parquet(dst, index=False)
    print(f"{len(df):,} rows, {df.shape[1]} columns -> {dst}")
