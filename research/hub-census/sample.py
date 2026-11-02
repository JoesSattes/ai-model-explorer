"""Draw a stratified sample from the frame, so results can be weighted back to the whole Hub.

Strata
  top        the N most-downloaded repos (30-day). Taken with certainty, weight 1. This is "what people use".
  rest       everything else, split by  created year x 30-day download tier x safetensors tag,
             sampled at random with proportional allocation (at least --min-per-stratum each).
             weight = stratum size / stratum sample size. This is "what the Hub contains".

Sample size: for a proportion, n = 1.96^2 * 0.25 / e^2  ->  ~2,400 for +/-2%, ~9,600 for +/-1% (95% CI).
The pilot (default 500 + 500) gives about +/-4.4% on the random half; use its failure and timing rates to size the full run.
"""
import argparse, json, os
import numpy as np
import pandas as pd

TIERS = [(-1, 0, "0"), (0, 99, "1-99"), (99, 9_999, "100-9,999"), (9_999, 999_999, "10k-999k"), (999_999, 1e18, "1M+")]


def tier(d):
    for lo, hi, name in TIERS:
        if lo < d <= hi:
            return name
    return "0"


# Providers whose releases define the state of the art; their recent originals form the "frontier" stratum, which tests
# the method on architectures it was not written against (see --since).
FRONTIER = ["Qwen", "deepseek-ai", "google", "meta-llama", "mistralai", "microsoft", "nvidia", "openai", "zai-org",
            "moonshotai", "MiniMaxAI", "ibm-granite", "allenai", "HuggingFaceTB", "LiquidAI", "tencent", "baidu",
            "stepfun-ai", "XiaomiMiMo", "inclusionAI", "ByteDance-Seed", "THUDM", "openbmb", "internlm", "OpenGVLab",
            "CohereLabs", "ai21labs", "tiiuae", "apple", "facebook", "amazon", "Salesforce", "black-forest-labs",
            "stabilityai", "Lightricks", "Wan-AI", "hexgrad", "kyutai", "sesame", "ServiceNow-AI", "swiss-ai", "arcee-ai",
            "PrimeIntellect", "NousResearch", "Zyphra", "jinaai", "BAAI", "Alibaba-NLP", "intfloat", "nomic-ai"]


def frontier(f: pd.DataFrame, n: int, since: str, providers=None) -> pd.DataFrame:
    """The n most-downloaded original releases from the frontier providers created on or after `since`."""
    prov = set(providers or FRONTIER)
    g = f[f["author"].isin(prov) & (f["created_at"] >= pd.Timestamp(since, tz="UTC"))]
    if "origin" in g:
        g = g[g["origin"] == "original"]
    g = g.sort_values("downloads", ascending=False).head(n).copy()
    g["stratum"], g["weight"] = "frontier", 1.0
    return g


def stratify(f: pd.DataFrame, top_n: int, rest_n: int, min_per: int, seed: int, originals: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = f.sort_values("downloads", ascending=False).reset_index(drop=True)
    top = f.head(top_n).copy()
    if originals and "origin" in f:  # the main models: most-downloaded original releases (not quantized / fine-tuned / merged copies)
        extra = f[(f["origin"] == "original") & ~f["id"].isin(top["id"])].head(max(0, originals - int((top["origin"] == "original").sum())))
        top = pd.concat([top, extra])
    top["stratum"], top["weight"] = "top", 1.0
    rest = f[~f["id"].isin(top["id"])].copy()
    rest["stratum"] = ("y" + rest["created_at"].dt.year.fillna(0).astype(int).astype(str) + "|d" +
                       rest["downloads"].map(tier) + "|st" + rest["has_safetensors_tag"].astype(int).astype(str))
    sizes = rest["stratum"].value_counts()
    alloc = (sizes / sizes.sum() * rest_n).round().astype(int).clip(lower=min_per)
    alloc = np.minimum(alloc, sizes)
    parts = []
    for s, k in alloc.items():
        grp = rest[rest["stratum"] == s]
        pick = grp.iloc[rng.choice(len(grp), size=int(k), replace=False)].copy()
        pick["weight"] = len(grp) / max(int(k), 1)
        parts.append(pick)
    out = pd.concat([top] + parts, ignore_index=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="out/frame.parquet")
    ap.add_argument("--out", default="out/sample.parquet")
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--rest", type=int, default=500)
    ap.add_argument("--min-per-stratum", type=int, default=2)
    ap.add_argument("--originals", type=int, default=0, help="also read the N most-downloaded original releases in full")
    ap.add_argument("--frontier", type=int, default=0, help="read the N most-downloaded originals from frontier providers created since --since")
    ap.add_argument("--since", default="2025-11-01", help="frontier window start (YYYY-MM-DD)")
    ap.add_argument("--providers", default="", help="comma-separated provider list (default: built-in FRONTIER)")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    f = pd.read_parquet(a.frame)
    s = stratify(f, a.top, a.rest, a.min_per_stratum, a.seed, a.originals)
    if a.frontier:
        fr = frontier(f, a.frontier, a.since, a.providers.split(",") if a.providers else None)
        s.loc[s["id"].isin(fr["id"]), "also_frontier"] = True   # already read in another stratum: flag, do not read twice
        s = pd.concat([s, fr[~fr["id"].isin(s["id"])]], ignore_index=True)
        s["also_frontier"] = s["also_frontier"].fillna(False).astype(bool) | (s["stratum"] == "frontier")
    s.to_parquet(a.out, index=False)
    info = {"frame": len(f), "sample": len(s), "seed": a.seed, "top": a.top, "rest_target": a.rest,
            "strata": int(s["stratum"].nunique()), "frontier_since": a.since if a.frontier else None,
            "frontier_repos": int(s.get("also_frontier", pd.Series(dtype=bool)).sum()) if a.frontier else 0, "weight_sum": round(float(s["weight"].sum()))}
    with open(os.path.join(os.path.dirname(a.out) or ".", "sample_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2))
