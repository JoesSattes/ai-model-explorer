"""Findings from a census run: how model architecture and storage changed year by year, where the Hub's own metadata
is wrong, and which repos ship no specification but their weights. Writes:

  out/insights/insights.json   every number below, for the dataset card and the website
  out/insights/insights.md     the same as readable text
  out/insights/models.json     one compact row per readable popular model (website table)
  out/insights/trends.png, out/insights/storage.png
"""
import argparse, json, math, os
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
LM = ("decoder", "prefix-lm")


def hp(r, k):
    return (r or {}).get(k) if isinstance(r, dict) else None


def load(path):
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    d["created"] = pd.to_datetime(d.get("created_at"), errors="coerce", utc=True)
    d["year"] = d["created"].dt.year
    return d


FIXTURE = r"internal-testing|(?:^|[/_-])tiny[-_]|random|dummy|unit-test|^unslothai/"


def popular(d):
    """Repos read in full (most downloaded, original releases, frontier): what people actually use.
    CI fixtures (tiny random-weight test repos, downloaded millions of times by test suites) are not models: left out."""
    p = d[d["stratum"].isin(["top", "frontier"])]
    return p[~p["id"].str.contains(FIXTURE, case=False, regex=True)]


def trends(ok):
    """Architecture by release year, original releases only (quantized and fine-tuned copies would count a model twice)."""
    orig = ok[(ok.get("origin") == "original") | ok.get("also_frontier", pd.Series(False, index=ok.index)).fillna(False).astype(bool)]
    lm = orig[orig["family_manifest"].isin(LM)].copy()
    lm["moe"] = lm["hp"].apply(lambda h: bool(hp(h, "experts")))
    lm["hybrid"] = lm["hp"].apply(lambda h: (hp(h, "ssm_layers") or 0) > 0)
    lm["latent"] = lm["hp"].apply(lambda h: bool(hp(h, "kv_latent")))
    lm["gqa"] = lm.apply(lambda r: bool(hp(r["hp_fused"], "kv_heads") and hp(r["hp_fused"], "heads")
                                        and r["hp_fused"]["kv_heads"] < r["hp_fused"]["heads"]), axis=1)
    lm["quantized"] = lm["quant"].apply(lambda q: bool(q))
    rows = []
    for y, g in lm[lm["year"].notna()].groupby("year"):
        med = lambda k: (lambda s: None if s.empty else float(s.median()))(pd.Series([hp(h, k) for h in g["hp"]]).dropna())
        rows.append({"year": int(y), "models": len(g), "moe": round(g["moe"].mean(), 3), "hybrid": round(g["hybrid"].mean(), 3),
                     "latent_attention": round(g["latent"].mean(), 3), "gqa": round(g["gqa"].mean(), 3),
                     "quantized": round(g["quantized"].mean(), 3), "median_params": float(g["params"].median()),
                     "median_d": med("d"), "median_layers": med("layers"), "median_head_dim": med("head_dim"),
                     "median_vocab": med("vocab"), "median_kv_bytes_per_token": med("kv_bytes_per_token_bf16")})
    return rows


def storage(ok):
    q = ok.copy()
    q["fmt"] = q["quant"].apply(lambda x: "+".join(sorted(x)) if isinstance(x, dict) and x else "unquantized")
    out = []
    for y, g in q[q["year"].notna()].groupby("year"):
        vc = g["fmt"].value_counts(normalize=True)
        out.append({"year": int(y), **{k: round(float(v), 3) for k, v in vc.items()}})
    return out


def hub_count(ok):
    h = ok[ok["hub_params"].notna() & (ok["hub_params"] > 0)].copy()
    h["r"] = h["params"] / h["hub_params"]
    h["fmt"] = h["quant"].apply(lambda x: "+".join(sorted(x)) if isinstance(x, dict) and x else "unquantized")
    out = []
    for f, g in h.groupby("fmt"):
        if len(g) < 5:
            continue
        wrong = (g["r"] - 1).abs() > 0.001
        out.append({"storage": f, "repos": len(g), "hub_count_differs": round(float(wrong.mean()), 3),
                    "median_true_over_hub": round(float(g.loc[wrong, "r"].median()), 3) if wrong.any() else 1.0})
    return sorted(out, key=lambda r: -r["repos"])


def no_config(ok, min_n=25):
    out = []
    for lib, g in ok.groupby(ok["library"].fillna("(none)")):
        if len(g) >= min_n:
            out.append({"library": lib, "repos": len(g), "no_config": round(float((~g["has_config"].fillna(False).astype(bool)).mean()), 3)})
    return sorted(out, key=lambda r: -r["no_config"])


def model_rows(ok, n=4000):
    rows = []
    for _, r in ok.sort_values("downloads", ascending=False).head(n).iterrows():
        h, f = r["hp"] or {}, r["hp_fused"] or {}
        rows.append({"id": r["id"], "dl": int(r["downloads"] or 0), "likes": int(r["likes"] or 0),
                     "y": int(r["year"]) if not math.isnan(r["year"]) else None, "origin": r.get("origin"),
                     "fam": r["family_manifest"], "p": r["params"], "hub": r["hub_params"] if r["hub_params"] == r["hub_params"] else None,
                     "q": "+".join(sorted(r["quant"])) if isinstance(r["quant"], dict) and r["quant"] else "",
                     "d": h.get("d"), "L": h.get("layers"), "H": f.get("heads") or h.get("heads"), "KV": f.get("kv_heads") or h.get("kv_heads"),
                     "E": h.get("experts"), "V": h.get("vocab"), "kv": h.get("kv_bytes_per_token_bf16"),
                     "ssm": h.get("ssm_layers") or 0, "cfg": bool(r["has_config"]), "lib": r.get("library")})
    return rows


def chart_trends(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.DataFrame(rows)
    t = t[t["models"] >= 30]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    series = (("moe", BLUE, "mixture of experts"), ("hybrid", ORANGE, "hybrid (state-space / linear attention)"),
              ("quantized", AQUA, "published quantized"))
    ends = sorted(((t[k].iloc[-1] * 100, k) for k, _, _ in series))
    ypos, last = {}, -1e9
    for v, k in ends:  # keep end labels at least 2.5 points apart
        ypos[k] = max(v, last + 2.5)
        last = ypos[k]
    for k, c, lab in series:
        ax.plot(t["year"], t[k] * 100, color=c, lw=2, marker="o", ms=8, label=lab)
        ax.annotate(f"{t[k].iloc[-1]*100:.0f}%", (t["year"].iloc[-1], t[k].iloc[-1] * 100), xytext=(t["year"].iloc[-1] + 0.12, ypos[k]),
                    textcoords="data", va="center", color=INK2, fontsize=9)
    ax.set_xticks(t["year"])
    ax.set_ylabel("% of popular original LMs", color=INK2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("What popular original language models are made of, by release year", loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def chart_hub(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = sorted([r for r in rows if r["storage"] != "unquantized"][:8], key=lambda r: r["hub_count_differs"])
    if not t:
        return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.barh([r["storage"] for r in t], [r["hub_count_differs"] * 100 for r in t], color=BLUE, height=.6)
    for i, r in enumerate(t):
        ax.text(r["hub_count_differs"] * 100 + 1, i, f"{r['hub_count_differs']*100:.0f}%  (n={r['repos']})", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("% of repos where the Hub's parameter count differs from the logical count", color=INK2)
    ax.set_xlim(0, 115)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.set_title("The Hub's parameter count, by storage format", loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def main(census, outdir):
    d = load(census)
    os.makedirs(outdir, exist_ok=True)
    pop = popular(d)
    ok = pop[pop["status"] == "ok"]
    T, S, H, C = trends(ok), storage(ok), hub_count(ok), no_config(d[d["status"] == "ok"])
    status = pop["status"].value_counts(normalize=True).round(4).to_dict()
    nwk = pop[pop["status"] == "no_weights"]["no_weights_kind"].value_counts(normalize=True).round(4).to_dict() if "no_weights_kind" in pop else {}
    out = {"repos": len(d), "popular_repos": len(pop), "status_popular": status, "no_weights_kind_popular": nwk,
           "trends": T, "storage_by_year": S, "hub_count_by_storage": H, "no_config_by_library": C[:25]}
    json.dump(out, open(os.path.join(outdir, "insights.json"), "w"), indent=1, default=float)
    json.dump(model_rows(ok), open(os.path.join(outdir, "models.json"), "w"), separators=(",", ":"), default=float)
    chart_trends(T, os.path.join(outdir, "trends.png"))
    chart_hub(H, os.path.join(outdir, "storage.png"))
    t = {r["year"]: r for r in T}
    y0, y1 = min(t), max(t)
    L = ["# Findings", "",
         f"From {len(pop):,} popular repos (most downloaded, original releases, recent frontier releases) out of {len(d):,} read.", ""]
    if y0 in t and y1 in t:
        a, b = t[y0], t[y1]
        L += [f"- **Mixture of experts** went from {a['moe']*100:.0f}% of popular original language models released in {y0} to {b['moe']*100:.0f}% in {y1}.",
              f"- **Hybrid stacks** (state-space or linear-attention layers next to attention): {a['hybrid']*100:.0f}% → {b['hybrid']*100:.0f}%.",
              f"- **Grouped-query attention** (fewer KV heads than query heads): {a['gqa']*100:.0f}% → {b['gqa']*100:.0f}%.",
              f"- Median **head size** {a['median_head_dim']:.0f} → {b['median_head_dim']:.0f}; median **vocabulary** {a['median_vocab']:,.0f} → {b['median_vocab']:,.0f} tokens.",
              f"- Original releases **published already quantized** (FP8, NVFP4, MXFP4 …): {a['quantized']*100:.0f}% → {b['quantized']*100:.0f}%."]
    # only formats that pack two or more values per stored element, where the Hub's element count is clearly not the parameter count
    bad = [r for r in H if r["storage"] != "unquantized" and r["hub_count_differs"] >= 0.5 and r["median_true_over_hub"] >= 1.2]
    if bad:
        L.append("- **The Hub's parameter count is wrong** for packed formats: " + "; ".join(
            f"{r['storage']} {r['hub_count_differs']*100:.0f}% of {r['repos']} repos (true/Hub median {r['median_true_over_hub']}x)" for r in bad[:4]) + ".")
    gg = nwk.get("gguf_only")
    if gg:
        L.append(f"- **{gg*100:.0f}%** of popular repos without readable weights are GGUF-only (llama.cpp format).")
    open(os.path.join(outdir, "insights.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="out/census.jsonl")
    ap.add_argument("--out", default="out/insights")
    a = ap.parse_args()
    main(a.census, a.out)
