"""The ecosystem as the weights describe it: every finding of the census with its uncertainty and, for changes over
time, a test for trend. Called by insights.py; writes out/insights/ecosystem.json and the figures eco_*.png / eco_*.pdf.

Populations
  popular    repos read in full (top downloads and frontier providers), CI fixtures removed: a complete enumeration of
             what people use, so intervals describe the variability of the process, not sampling error
  original   popular repos the Hub does not tag as a derivative (or read as a frontier release)
  LM         decoder or prefix-LM anatomy above 50M parameters with a text-generating Hub pipeline, or with no pipeline
             and a config.json of a generative decoder / prefix-LM type (see is_lm)
  Hub-wide   all strata with sampling weights (Kish effective n)
The Hub dates every repository older than March 2022 to 2 March 2022, so the first year bin reads "<=2022".

Statistics
  proportions   Wilson 95% interval
  trend         Cochran-Armitage test across years, and the odds ratio per year from a logistic fit (Wald 95% interval)
  medians       bootstrap 95% interval (2,000 resamples, fixed seed)
  monotone      Spearman rank correlation with year
  scaling       least-squares slope in log-log space, bootstrap 95% interval, R^2
  two groups    Mann-Whitney U (two-sided)
"""
import collections, json, math, os

import numpy as np
import pandas as pd

FIXTURE = r"internal-testing|(?:^|[/_-])tiny(?:[-_]|$)|random|dummy|unit-test|^unslothai/"
LMF = ("decoder", "prefix-lm")
GEN = {"text-generation", "text2text-generation", "image-text-to-text", "any-to-any", "audio-text-to-text", "video-text-to-text",
       "visual-question-answering", "conversational"}


NOT_GEN = r"clip|siglip|flava|owl|esm|bert|roberta|layoutlm|fnet|dit|vit|wav2vec|hubert|whisper|t5|bart|led|bigbird|pegasus|marian|m2m"


def is_lm(df):
    """Generative language models: a decoder or prefix-LM anatomy above 50M parameters, and either a Hub pipeline that
    generates text, or, when no pipeline is declared, a config.json whose own architecture is a decoder or prefix-LM of
    a generative model type. Repos with neither are left out: the family rules alone place some dual encoders, CTC speech
    models and encoder-decoders among decoders, which would otherwise contaminate the earliest years."""
    pt = df["pipeline_tag"].fillna("")
    fc = df["family_config"] if "family_config" in df else pd.Series(None, index=df.index)
    mt = (df["config_model_type"] if "config_model_type" in df else pd.Series("", index=df.index)).fillna("").astype(str)
    by_config = (pt == "") & fc.isin(LMF) & ~mt.str.contains(NOT_GEN, case=False, regex=True)
    # family: from the weights, or from the configuration when the two disagree (tied-embedding decoders the weights
    # alone place among encoders, about 4% of generative LMs, would otherwise drop out and bias the small-model results)
    fam = df["family_manifest"].isin(LMF) | (fc.isin(LMF) & ~mt.str.contains(NOT_GEN, case=False, regex=True))
    return fam & (df["params"] > 5e7) & (pt.isin(GEN) | by_config)


RNG = np.random.default_rng(2026)
BLUE, ORANGE, AQUA, YELLOW, RED, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c4242a", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"p": round(p, 4), "lo": round(max(0, c - h), 4), "hi": round(min(1, c + h), 4), "k": int(k), "n": int(n)}


def _phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def armitage(x, k, n):
    """Cochran-Armitage test for a trend in proportions k/n over scores x. Returns z and two-sided p."""
    x, k, n = (np.asarray(v, float) for v in (x, k, n))
    N, K = n.sum(), k.sum()
    if N == 0 or K in (0, N):
        return {"z": 0.0, "p": 1.0}
    p = K / N
    t = (x * (k - n * p)).sum()
    v = p * (1 - p) * ((n * x * x).sum() - (n * x).sum() ** 2 / N)
    z = t / math.sqrt(v) if v > 0 else 0.0
    return {"z": round(float(z), 2), "p": float(2 * (1 - _phi(abs(z))))}


def logit_trend(x, y, iters=50):
    """Odds ratio per unit of x from a logistic fit y ~ 1 + x (IRLS), with a Wald 95% interval."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 10 or y.min() == y.max():
        return None
    X = np.c_[np.ones_like(x), x - x.mean()]
    b = np.zeros(2)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ b))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    se = math.sqrt(np.linalg.inv(H)[1, 1])
    return {"or": round(math.exp(b[1]), 3), "lo": round(math.exp(b[1] - 1.96 * se), 3), "hi": round(math.exp(b[1] + 1.96 * se), 3),
            "p": float(2 * (1 - _phi(abs(b[1] / se))))}


def boot_median(v, B=2000):
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return None
    if len(v) == 1:
        return {"med": float(v[0]), "lo": float(v[0]), "hi": float(v[0]), "n": 1}
    m = np.median(RNG.choice(v, (B, len(v))), axis=1)
    return {"med": float(np.median(v)), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": int(len(v))}


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return {"rho": round(float(r.statistic), 3), "p": float(r.pvalue), "n": int(len(x))}


def loglog(x, y, groups=None, B=1000):
    """Slope of log10(y) on log10(x). The interval resamples whole groups (providers) when given: checkpoints of one
    family share their design, so treating them as independent would make the interval too narrow."""
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    ok = np.isfinite(lx) & np.isfinite(ly)
    lx, ly = lx[ok], ly[ok]
    g = np.asarray(groups)[ok] if groups is not None else np.arange(len(lx))
    b, a = np.polyfit(lx, ly, 1)
    r2 = 1 - ((ly - (a + b * lx)) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    ug = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in ug}
    bs = []
    for _ in range(B):
        pick = np.concatenate([idx[k] for k in RNG.choice(ug, len(ug))])
        if np.ptp(lx[pick]) > 0:
            bs.append(np.polyfit(lx[pick], ly[pick], 1)[0])
    return {"slope": round(float(b), 3), "lo": round(float(np.percentile(bs, 2.5)), 3), "hi": round(float(np.percentile(bs, 97.5)), 3),
            "intercept": round(float(a), 3), "r2": round(float(r2), 3), "n": int(len(lx)), "clusters": int(len(ug))}


def mwu(a, b):
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(r.statistic), "p": float(r.pvalue), "na": int(len(a)), "nb": int(len(b)),
            "med_a": float(np.median(a)), "med_b": float(np.median(b)), "auc": round(float(r.statistic / (len(a) * len(b))), 3)}


def by_year(df, flag):
    """Per-year Wilson intervals plus Armitage and logistic trend for a boolean column."""
    g = df[df["year"].notna()].copy()
    g["yb"] = g["year"].clip(lower=2022).astype(int)
    rows = [{"year": int(y), **wilson(int(s[flag].sum()), len(s))} for y, s in g.groupby("yb")]
    late = g[g["yb"] >= 2023]
    return {"years": rows, "armitage": armitage([r["year"] for r in rows], [r["k"] for r in rows], [r["n"] for r in rows]),
            "logit": logit_trend(g["yb"], g[flag].astype(float)), "logit_from_2023": logit_trend(late["yb"], late[flag].astype(float))}


# ---------------------------------------------------------------- data
def _hp(df, src, keys):
    for k in keys:
        df[k] = df[src].apply(lambda h, k=k: (h or {}).get(k) if isinstance(h, dict) else None)


def _parents(t):
    return [x.split(":", 2)[2] for x in (t if t is not None else []) if str(x).startswith("base_model:") and str(x).count(":") >= 2]


def _relation(t):
    rel = [x.split(":")[1] for x in (t if t is not None else []) if str(x).startswith("base_model:") and str(x).count(":") >= 2]
    return next((r for r in ("quantized", "adapter", "merge", "finetune") if r in rel), "original")


def _main_dtype(dt):
    if not isinstance(dt, dict) or not dt:
        return None
    return max(dt.items(), key=lambda kv: kv[1][1] if isinstance(kv[1], (list, tuple)) else 0)[0]


def load(census):
    d = pd.DataFrame([json.loads(l) for l in open(census)])
    sp = os.path.join(os.path.dirname(os.path.abspath(census)), "sample.parquet")
    if os.path.exists(sp):
        d = d.merge(pd.read_parquet(sp, columns=["id", "tags"]), on="id", how="left")
    else:
        d["tags"] = None
    d = d[~d["id"].str.contains(FIXTURE, case=False, regex=True)].copy()
    d["created"] = pd.to_datetime(d["created_at"], errors="coerce", utc=True)
    d["year"] = d["created"].dt.year
    _hp(d, "hp", ["d", "layers", "heads", "kv_heads", "head_dim", "vocab", "experts", "expert_params", "tie", "attn_layers",
                  "ssm_layers", "kv_bytes_per_token_bf16", "kv_latent", "ffn"])
    _hp(d, "hp_fused", ["topk"])
    d["fheads"] = d["hp_fused"].apply(lambda h: (h or {}).get("heads") if isinstance(h, dict) else None).fillna(d["heads"])
    d["fkv"] = d["hp_fused"].apply(lambda h: (h or {}).get("kv_heads") if isinstance(h, dict) else None).fillna(d["kv_heads"])
    d["mdt"] = d["dtypes"].apply(_main_dtype)
    d["rel"] = d["tags"].apply(_relation)
    return d


# ---------------------------------------------------------------- findings
def analyse(census, frame_stats=None):
    d = load(census)
    pop = d[d["stratum"].isin(["top", "frontier"])]
    ok = pop[pop["status"] == "ok"].copy()
    orig = ok[(ok["origin"] == "original") | ok["also_frontier"].fillna(False).astype(bool)].copy()
    lm = orig[is_lm(orig)].copy()
    F = {"populations": {"repos": len(d), "popular": len(pop), "popular_readable": len(ok), "original": len(orig), "original_lm": len(lm),
                         "dated_2022_03_02": round(float((pop["created"].dt.strftime("%Y-%m-%d") == "2022-03-02").mean()), 3)}}

    # 1. reading cost
    okall = d[d["status"] == "ok"]
    wb, hb = float(okall["weight_bytes"].sum()), float(okall["header_bytes"].sum())
    x = okall[(okall["tensors"] > 0) & (okall["header_bytes"] > 0)]
    F["cost"] = {"weight_bytes": wb, "header_bytes": hb, "ratio": round(wb / hb), "bytes_per_tensor": boot_median(x["header_bytes"] / x["tensors"]),
                 "header_vs_tensors": loglog(x["tensors"], x["header_bytes"], x["author"].values), "seconds": boot_median(okall["t_total_s"].dropna())}

    # 2. architecture trends (original LMs)
    lm["moe"] = lm["experts"].fillna(0) > 0
    lm["hybrid"] = (lm["ssm_layers"].fillna(0) > 0) & (lm["attn_layers"].fillna(0) > 0)   # attention-free stacks (RWKV, Mamba) are not hybrids
    lm["gqa"] = (lm["fkv"].notna() & lm["fheads"].notna()) & (lm["fkv"] < lm["fheads"])
    lm["latent"] = lm["kv_latent"].fillna(False).astype(bool)
    lm["quant"] = lm["quant"].apply(lambda q: bool(q))
    lm["vlm"] = lm["family_manifest"] == "prefix-lm"
    F["trends"] = {k: by_year(lm, k) for k in ("moe", "hybrid", "gqa", "latent", "quant", "vlm")}

    # 3. mixtures of experts: how sparse
    moe = lm[lm["moe"]].copy()
    moe["active_ratio"] = moe.apply(lambda r: (r["params"] - r["expert_params"] * (1 - r["topk"] / r["experts"])) / r["params"]
                                    if r["expert_params"] and r["topk"] and r["experts"] else np.nan, axis=1)
    my = moe[moe["year"] >= 2023]
    F["moe"] = {"by_year": [{"year": int(y), "n": len(g), "experts": boot_median(g["experts"]), "topk": boot_median(g["topk"].dropna()),
                             "active_ratio": boot_median(g["active_ratio"].dropna())} for y, g in my.groupby("year")],
                "active_ratio_vs_year": spearman(my.dropna(subset=["active_ratio"])["year"], my.dropna(subset=["active_ratio"])["active_ratio"]),
                "experts_vs_year": spearman(my["year"], my["experts"])}

    # 4. attention memory: KV cache per token per billion parameters, and when the cache outweighs the weights
    kv = lm[lm["kv_bytes_per_token_bf16"].fillna(0) > 0].copy()
    kv["kvpb"] = kv["kv_bytes_per_token_bf16"] / (kv["params"] / 1e9)
    lm["cache_gt_weights_32k"] = lm["kv_bytes_per_token_bf16"].fillna(0) * 32768 > lm["params"] * 0.5   # 16-bit cache vs 4-bit weights
    kv["yb"] = kv["year"].clip(lower=2022)
    rows = [{"year": int(y), "kv_per_b": boot_median(g["kvpb"])} for y, g in kv.groupby("yb")]
    first, last = rows[0]["kv_per_b"]["med"], rows[-1]["kv_per_b"]["med"]
    late = kv[kv["yb"] >= 2023]
    F["kv"] = {"by_year": rows, "fold_drop": round(first / last, 1), "vs_year": spearman(kv["yb"], kv["kvpb"]),
               "vs_year_from_2023": spearman(late["yb"], late["kvpb"]),
               "cache_gt_weights_32k": by_year(lm, "cache_gt_weights_32k"),
               "gqa_ratio_recent": boot_median((kv[kv["year"] >= 2025]["fheads"] / kv[kv["year"] >= 2025]["fkv"]).dropna())}

    # 5. shape scaling laws of released models: width and depth against parameters
    sc = lm[lm["d"].notna() & lm["layers"].notna() & ~lm["moe"]]
    au = sc["author"].values
    F["scaling"] = {"d": loglog(sc["params"], sc["d"], au), "layers": loglog(sc["params"], sc["layers"], au),
                    "aspect": loglog(sc["params"], sc["d"] / sc["layers"], au), "vocab": loglog(sc["params"], sc["vocab"].fillna(0).replace(0, np.nan), au)}

    # 6. tied embeddings, by size
    t = lm[lm["tie"].notna()].copy()
    t["tie"] = t["tie"].astype(bool)
    bins = [(0, 1e9, "<1B"), (1e9, 3e9, "1-3B"), (3e9, 1e10, "3-10B"), (1e10, 1e11, "10-100B"), (1e11, 1e14, ">100B")]
    F["tie"] = {"by_size": [{"size": lab, **wilson(int(g["tie"].sum()), len(g))} for lo, hi, lab in bins
                            for g in [t[(t["params"] > lo) & (t["params"] <= hi)]] if len(g)],
                "logit_per_decade": logit_trend(np.log10(t["params"]), t["tie"].astype(float))}

    # 7. how weights are stored
    st = orig[orig["mdt"].notna()].copy()
    st["yb"] = st["year"].clip(lower=2022)
    grp = lambda m: "BF16" if m == "BF16" else "F16" if m == "F16" else "F32" if m == "F32" else "FP8" if str(m).startswith("F8") else "packed / int"
    st["fmt"] = st["mdt"].map(grp)
    mix = (pd.crosstab(st["yb"], st["fmt"], normalize="index").round(3))
    big = ok[ok["params"] > 1e9]
    f32 = big[big["mdt"] == "F32"]
    F["storage"] = {"mix_by_year": {int(y): r.to_dict() for y, r in mix.iterrows()},
                    "f32_trend": by_year(st.assign(f32=st["fmt"] == "F32"), "f32"),
                    "f32_over_1b": wilson(len(f32), len(big)), "f32_over_1b_tb": round(float(f32["stored_bytes"].sum()) / 1e12, 2),
                    "f32_over_1b_saved_tb": round(float(f32["stored_bytes"].sum()) / 2e12, 2),
                    "f32_over_1b_downloads": int(f32["downloads"].sum())}

    # 8. the derivative economy
    eco = {}
    if frame_stats:
        o, s = frame_stats.get("origin", {}), frame_stats.get("origin_share_of_downloads", {})
        tot = sum(o.values())
        eco["hub"] = {k: {"repo_share": round(o[k] / tot, 4), "download_share": s.get(k), "lift": round(s.get(k, 0) / (o[k] / tot), 2)}
                      for k in o if tot}
        eco["zero_download_share"] = round(frame_stats["zero_downloads_30d"] / frame_stats["repos"], 4)
        eco["top1000_share"], eco["top10000_share"] = frame_stats.get("share_of_downloads_top_1000"), frame_stats.get("share_of_downloads_top_10000")
    allok = d[d["status"] == "ok"].set_index("id")
    rows = []
    for _, r in ok.iterrows():
        ps = _parents(r["tags"])
        if len(ps) != 1 or ps[0] not in allok.index or ps[0] == r["id"]:
            continue
        p = allok.loc[ps[0]]
        if isinstance(p, pd.DataFrame):
            p = p.iloc[0]
        ch, ph = r["hp"] or {}, p["hp"] or {}
        if not ch.get("d") or not ph.get("d"):   # compare only where both anatomies were recovered
            continue
        diff = {k: ch.get(k) != ph.get(k) for k in ("d", "layers", "vocab", "ffn", "heads")}
        rows.append({"rel": r["rel"], "same": not any(diff.values()), **{"diff_" + k: v for k, v in diff.items()},
                     "dparams": (r["params"] - p["params"]) / p["params"] if p["params"] else None})
    fd = pd.DataFrame(rows)
    eco["fidelity"] = {rel: {"n": len(g), "identical": wilson(int(g["same"].sum()), len(g)),
                             "vocab_changed": wilson(int(g["diff_vocab"].sum()), len(g)),
                             "layers_changed": wilson(int(g["diff_layers"].sum()), len(g)),
                             "params_change_median": boot_median(g["dparams"].dropna())}
                       for rel, g in fd.groupby("rel") if rel in ("finetune", "quantized", "adapter") and len(g) >= 20} if len(fd) else {}
    par = [p for t in pop["tags"] for p in _parents(t)]
    owners = collections.Counter(p.split("/")[0] for p in par)
    tot = sum(owners.values()) or 1
    eco["lineage"] = {"derivatives": len(par), "top_parent_owners": [[k, round(v / tot, 3)] for k, v in owners.most_common(8)],
                      "hhi": round(sum((v / tot) ** 2 for v in owners.values()), 4),
                      "top_parents": collections.Counter(par).most_common(8)}
    qc = ok[ok["rel"] == "quantized"]
    eco["quantizers"] = {"repos": len(qc), "top": [[k, int(v)] for k, v in qc["author"].value_counts().head(8).items()],
                         "formats": {k: round(v, 3) for k, v in qc["quant"].apply(lambda q: "+".join(sorted(q)) if isinstance(q, dict) and q else "unquantized")
                                     .value_counts(normalize=True).head(8).items()}}
    F["derivatives"] = eco

    # 9. metadata reliability
    chk = ok[ok["config_checked"].fillna(0) > 0]
    bad = chk[chk["config_mismatch"].apply(lambda x: bool(x) if isinstance(x, list) else False)]
    paths = collections.Counter("sub-config" if str(m.get("path", "(root)")) != "(root)" else "root"
                                for x in bad["config_mismatch"] for m in x if isinstance(m, dict))
    hub = ok[ok["hub_params"].notna() & (ok["hub_params"] > 0)]
    lic = pop["tags"].apply(lambda t: any(str(x).startswith("license:") for x in (t if t is not None else [])))
    nocfg = ok[~ok["has_config"].fillna(False).astype(bool)]["downloads"]
    cfg = ok[ok["has_config"].fillna(False).astype(bool)]["downloads"]
    F["metadata"] = {"config_disagrees": wilson(len(bad), len(chk)), "disagreement_where": dict(paths),
                     "hub_count_wrong": wilson(int((~hub["params_match_hub"].fillna(False).astype(bool)).sum()), len(hub)),
                     "hub_count_missing": wilson(int(ok["hub_params"].isna().sum()), len(ok)),
                     "license_missing": wilson(int((~lic).sum()), len(pop)),
                     "no_config": wilson(int(len(nocfg)), len(ok)),
                     "downloads_config_vs_none": mwu(cfg, nocfg),
                     "downloads_readable_vs_not": mwu(pop[pop["status"] == "ok"]["downloads"], pop[pop["status"] != "ok"]["downloads"])}

    # 10. what runs where: 4-bit weights, 8k context, +10%
    dep = lm.copy()
    dep["need"] = (dep["params"] * 0.5 + dep["kv_bytes_per_token_bf16"].fillna(0) * 8192) * 1.1
    dep["tier"] = pd.cut(dep["need"], [0, 4.8e9, 12e9, 24e9, 80e9, 1e16], labels=["Phone", "Laptop", "Desktop GPU", "Workstation", "Cluster"])
    dep["yb"] = dep["year"].clip(lower=2022)
    dep["edge"] = dep["need"] <= 12e9
    F["deploy"] = {"tier_by_year": {int(y): g["tier"].value_counts(normalize=True).round(3).to_dict() for y, g in dep.groupby("yb")},
                   "edge_by_year": by_year(dep, "edge"),
                   "download_weighted": (dep.groupby("tier", observed=False)["downloads"].sum() / dep["downloads"].sum()).round(3).to_dict(),
                   "repo_weighted": dep["tier"].value_counts(normalize=True).round(3).to_dict(),
                   "median_params_by_year": {int(y): boot_median(g["params"]) for y, g in dep.groupby("yb")},
                   "largest_by_year": {int(y): float(g["params"].max()) for y, g in dep.groupby("yb")}}
    F["deploy"]["edge_downloads_vs_repos"] = {"repos": round(float(dep["edge"].mean()), 3),
                                              "downloads": round(float(dep[dep["edge"]]["downloads"].sum() / dep["downloads"].sum()), 3)}
    return json.loads(json.dumps(F, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("NaN", "null"))


# ---------------------------------------------------------------- figures
def _ax(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _yl(F):
    return lambda y: "≤2022" if y == 2022 else str(y)


def figures(F, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    yl = _yl(F)
    out = []

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(outdir, f"eco_{name}.{ext}"), dpi=200, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)
        out.append(name)

    # a) architecture adoption with intervals
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fig.patch.set_facecolor(SURF); _ax(ax)
    for k, c, lab in (("gqa", AQUA, "grouped-query attention"), ("moe", BLUE, "mixture of experts"), ("vlm", YELLOW, "vision-language (prefix-LM)"),
                      ("hybrid", ORANGE, "hybrid (state-space / linear attn.)"), ("latent", RED, "latent attention")):
        r = F["trends"][k]["years"]
        x = [z["year"] for z in r]
        ax.plot(x, [z["p"] * 100 for z in r], color=c, lw=2, marker="o", ms=3.5, label=lab)
        ax.fill_between(x, [z["lo"] * 100 for z in r], [z["hi"] * 100 for z in r], color=c, alpha=0.13, lw=0)
    ax.set_xticks(x, [yl(v) for v in x]); ax.set_ylabel("% of popular original LMs", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "adoption")

    # b) attention memory and MoE sparsity
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fig.patch.set_facecolor(SURF); _ax(a1); _ax(a2)
    r = F["kv"]["by_year"]
    x = [z["year"] for z in r]
    a1.plot(x, [z["kv_per_b"]["med"] / 1e3 for z in r], color=BLUE, lw=2, marker="o", ms=3.5)
    a1.fill_between(x, [z["kv_per_b"]["lo"] / 1e3 for z in r], [z["kv_per_b"]["hi"] / 1e3 for z in r], color=BLUE, alpha=.15, lw=0)
    a1.set_yscale("log"); a1.set_xticks(x, [yl(v) for v in x]); a1.set_title("KV cache per token per billion params (KB)", fontsize=9, color=INK)
    m = [z for z in F["moe"]["by_year"] if z["n"] >= 20]   # years with too few mixtures of experts say nothing about a median
    x2 = [z["year"] for z in m]
    a2.plot(x2, [z["active_ratio"]["med"] * 100 if z["active_ratio"] else None for z in m], color=ORANGE, lw=2, marker="o", ms=3.5, label="active share of params")
    a2.set_ylabel("% active per token", color=INK2); a2.set_xticks(x2, [str(v) for v in x2])
    b2 = a2.twinx()
    b2.plot(x2, [z["experts"]["med"] for z in m], color=INK2, lw=1.4, ls="--", marker="s", ms=3, label="experts (median)")
    b2.set_yscale("log", base=2); b2.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top",):
        b2.spines[s].set_visible(False)
    a2.set_title("Mixtures of experts by creation year", fontsize=9, color=INK)
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout()
    save(fig, "memory")

    # c) storage precision by year
    mix = F["storage"]["mix_by_year"]
    ys = sorted(mix, key=int)
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    fig.patch.set_facecolor(SURF); _ax(ax)
    bottom = np.zeros(len(ys))
    for k, c in (("F32", RED), ("F16", YELLOW), ("BF16", BLUE), ("FP8", AQUA), ("packed / int", ORANGE)):
        v = np.array([mix[y].get(k, 0) * 100 for y in ys])
        ax.bar(range(len(ys)), v, bottom=bottom, color=c, width=.62, label=k, edgecolor=SURF, lw=1.5)
        bottom += v
    ax.set_xticks(range(len(ys)), [yl(int(y)) for y in ys]); ax.set_ylabel("% of popular originals", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.16))
    save(fig, "storage")

    # d) derivative economy: share of repos vs share of downloads
    h = F["derivatives"].get("hub") or {}
    if h:
        ks = ["original", "finetune", "quantized", "adapter", "merge"]
        fig, ax = plt.subplots(figsize=(6.0, 2.6))
        fig.patch.set_facecolor(SURF); _ax(ax)
        xx = np.arange(len(ks))
        ax.bar(xx - .18, [h[k]["repo_share"] * 100 for k in ks], .34, color=GRID, edgecolor=INK2, lw=.6, label="share of repositories")
        ax.bar(xx + .18, [(h[k]["download_share"] or 0) * 100 for k in ks], .34, color=BLUE, label="share of downloads")
        for i, k in enumerate(ks):
            ax.text(i + .18, (h[k]["download_share"] or 0) * 100 + 1.5, f"×{h[k]['lift']}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(xx, ["original", "fine-tune", "quantized copy", "adapter", "merge"]); ax.set_ylabel("%", color=INK2)
        ax.legend(frameon=False, fontsize=8)
        save(fig, "economy")

    # e) deployability by year
    tb = F["deploy"]["tier_by_year"]
    ys = sorted(tb, key=int)
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    fig.patch.set_facecolor(SURF); _ax(ax)
    bottom = np.zeros(len(ys))
    for k, c in (("Phone", AQUA), ("Laptop", BLUE), ("Desktop GPU", YELLOW), ("Workstation", ORANGE), ("Cluster", RED)):
        v = np.array([tb[y].get(k, 0) * 100 for y in ys])
        ax.bar(range(len(ys)), v, bottom=bottom, color=c, width=.62, label=k, edgecolor=SURF, lw=1.5)
        bottom += v
    ax.set_xticks(range(len(ys)), [yl(int(y)) for y in ys]); ax.set_ylabel("% of popular original LMs", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.16))
    save(fig, "deploy")
    return out


def run(census, outdir, frame_stats=None):
    F = analyse(census, frame_stats)
    os.makedirs(outdir, exist_ok=True)
    try:
        F["figures"] = figures(F, outdir)
    except ImportError:
        F["figures"] = []
    json.dump(F, open(os.path.join(outdir, "ecosystem.json"), "w"), indent=1)
    return F


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="out/census.jsonl")
    ap.add_argument("--out", default="out/insights")
    a = ap.parse_args()
    fs = os.path.join(os.path.dirname(a.census), "frame_stats.json")
    run(a.census, a.out, json.load(open(fs)) if os.path.exists(fs) else None)
    print(f"-> {a.out}/ecosystem.json")
