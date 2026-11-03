"""Turn census.jsonl into weighted Hub-wide estimates, a markdown summary and two figures.

Two populations are reported side by side:
  "Top N"     the most-downloaded repos (unweighted: we read all of them)
  "Hub-wide"  the random strata, weighted back to the whole frame (weight = stratum size / sampled)
Proportions carry a 95% interval from the weighted (Kish) effective sample size.
"""
import argparse, json, math, os
from collections import Counter
import pandas as pd

BLUE, ORANGE, INK, INK2, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def wprop(mask, w):
    w = w.astype(float)
    if w.sum() == 0:
        return float("nan"), float("nan")
    p = float((w * mask).sum() / w.sum())
    neff = w.sum() ** 2 / (w ** 2).sum()
    return p, 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / neff)


def fmt(p, ci):
    return "-" if math.isnan(p) else f"{p*100:.1f}% ± {ci*100:.1f}"


def table(d, groups, fn):
    rows = []
    for label, part in groups:
        rows.append((label, fn(part)))
    return rows


def main(path, outdir):
    rows = [json.loads(l) for l in open(path)]
    d = pd.DataFrame(rows)
    d["weight"] = d["weight"].fillna(1.0)
    top, rnd = d[d["stratum"] == "top"], d[d["stratum"] != "top"]
    groups = [(f"Top {len(top)} by downloads", top.assign(weight=1.0)), (f"Hub-wide (weighted, n={len(rnd)})", rnd)]
    L = ["# Hub census, pilot", "", f"Repos probed: **{len(d)}**  ·  crawled {d['crawled_at'].min()} → {d['crawled_at'].max()}", ""]

    statuses = ["ok", "gated", "no_weights", "legacy_pickle", "not_found", "timeout", "network", "error"]
    L += ["## What happened when we read each repo", "", "| status | " + " | ".join(g for g, _ in groups) + " |",
          "|---|" + "---|" * len(groups)]
    for s in statuses:
        L.append(f"| {s} | " + " | ".join(fmt(*wprop(p["status"] == s, p["weight"])) for _, p in groups) + " |")

    L += ["", "## Among repos we could read", ""]
    def ok(p): return p[p["status"] == "ok"]
    metrics = {
        "has a config.json": lambda p: p["has_config"] == True,  # noqa: E712
        "**no config.json** (weights are the only spec)": lambda p: p["has_config"] != True,  # noqa: E712
        "config disagrees with weights (≥1 checked field)": lambda p: (p["config_checked"] > p["config_consistent"]),
        "quantized storage": lambda p: p["quant"].apply(lambda q: bool(q)),
        "mixture-of-experts": lambda p: p["hp"].apply(lambda h: bool((h or {}).get("experts"))),
        "has SSM / linear-attention layers": lambda p: p["hp"].apply(lambda h: ((h or {}).get("ssm_layers") or 0) > 0),
        "manifest family = config family": lambda p: (p["family_manifest"] == p["family_config"]) | p["family_config"].isna(),
        "param count = Hub's count (±0.1%)": lambda p: p["params_match_hub"] == True,  # noqa: E712
    }
    L += ["| | " + " | ".join(g for g, _ in groups) + " |", "|---|" + "---|" * len(groups)]
    for name, fn in metrics.items():
        L.append(f"| {name} | " + " | ".join(fmt(*wprop(fn(ok(p)).fillna(False).astype(bool), ok(p)["weight"])) if len(ok(p)) else "-"
                                         for _, p in groups) + " |")

    o = ok(d)
    if len(o):
        ratio = (o["header_bytes"] / o["weight_bytes"].where(o["weight_bytes"] > 0)).dropna()
        L += ["", "## Cost of reading", "",
              f"- header bytes read, median: **{o['header_bytes'].median()/1024:.0f} KB** (p95 {o['header_bytes'].quantile(.95)/1024/1024:.1f} MB)",
              f"- fraction of the repo's weight bytes read, median: **{ratio.median()*100:.4f}%**",
              f"- seconds per repo, median: **{o['t_total_s'].median():.1f}s** (p95 {o['t_total_s'].quantile(.95):.1f}s)",
              f"- time spent waiting on rate limits: {d.get('rate_limit_wait_s', pd.Series([0])).fillna(0).sum():.0f}s total"]

        r_ok = o[o["stratum"] != "top"]
        L += ["", "## Family by weights alone (Hub-wide, weighted share, 95% CI, n = repos behind it)", "",
              "| family | share | n |", "|---|---|---|"]
        for k, cnt in r_ok["family_manifest"].fillna("?").value_counts().items():
            p_, ci = wprop(r_ok["family_manifest"].fillna("?") == k, r_ok["weight"])
            L.append(f"| {k} | {fmt(p_, ci)} | {cnt} |")

        L += ["", "## Popularity vs readability", "", "| status | median downloads (30d) | median likes | repos |", "|---|---|---|---|"]
        for s, g in d.groupby("status"):
            L.append(f"| {s} | {int(g['downloads'].median()):,} | {int(g['likes'].median()):,} | {len(g)} |")

    if len(o):
        h = o[o["hub_params"].notna() & (o["hub_params"] > 0)].copy()
        h["r"] = h["params"] / h["hub_params"]
        h["qf"] = h["quant"].apply(lambda q: ",".join(sorted(q)) if isinstance(q, dict) and q else "")
        def why(r):
            if abs(r.r - 1) <= 0.001: return "equal"
            if r.qf and r.r > 1: return "quantized: Hub counts packed storage elements"
            if r.qf: return "quantized: we undercount (check)"
            if r.files_read > 1 and r.r > 1: return "several components or copies (check)"
            return "other (check)"
        h["why"] = h.apply(why, axis=1)
        L += ["", "## Our parameter count vs the Hub's", "", "| reason | repos | median ours / Hub |", "|---|---|---|"]
        for k, g in h.groupby("why"):
            L.append(f"| {k} | {len(g)} | {g['r'].median():.3f} |")
        L.append(f"| no Hub count at all | {int(o['hub_params'].isna().sum())} | - |")

    if "no_weights_kind" in d:
        nw = d[d["status"] == "no_weights"]
        L += ["", "## Why there were no weights to read", "", "| kind | " + " | ".join(g for g, _ in groups) + " |", "|---|" + "---|" * len(groups)]
        for k in nw["no_weights_kind"].fillna("other").value_counts().index:  # share among repos with no weights
            L.append(f"| {k} | " + " | ".join(fmt(*wprop(q["no_weights_kind"].fillna("other") == k, q["weight"])) if len(q) else "-"
                                              for q in (p[p["status"] == "no_weights"] for _, p in groups)) + " |")
    if "origin" in d and d["origin"].notna().any():
        L += ["", "## Original releases vs derivatives (repos read in full, unweighted)", "",
              "| origin | repos | readable | no config.json | family by weights = family by config |", "|---|---|---|---|---|"]
        tops = d[d["stratum"] == "top"]
        for k, g in tops.groupby(tops["origin"].fillna("?")):
            gk = g[g["status"] == "ok"]
            same = ((gk["family_manifest"] == gk["family_config"]) | gk["family_config"].isna()).mean() if len(gk) else float("nan")
            L.append(f"| {k} | {len(g)} | {len(gk)/len(g)*100:.0f}% | {(gk['has_config'] != True).mean()*100:.0f}% | {same*100:.0f}% |")  # noqa: E712

    if "config_hp" in d:
        L += field_accuracy(d)

    errs = d[d["status"].isin(["error", "no_weights"])]["error"].fillna("").str.slice(0, 80).value_counts().head(10)
    if len(errs):
        L += ["", "## Most common failure messages", ""] + [f"- `{k}` × {v}" for k, v in errs.items()]

    os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, "summary.md"), "w").write("\n".join(L) + "\n")
    figures(d, groups, outdir)
    print("\n".join(L))


FIELDS = ["d", "layers", "heads", "kv_heads", "head_dim", "ffn", "vocab", "experts"]


FIXTURE = r"internal-testing|(?:^|[/_-])tiny[-_]|random|dummy|unit-test|^unslothai/"


def _score(rows, field, fused):
    hit = n = 0
    for _, r in rows.iterrows():
        ref = (r["config_hp"] or {}).get(field)
        if ref is None or ref <= 0 or (field == "ffn" and (r["config_hp"] or {}).get("experts")):  # 0 = placeholder config
            continue
        ours = (r["hp"] or {}).get(field)
        if fused and isinstance(r.get("hp_fused"), dict) and r["hp_fused"].get(field) is not None:
            ours = r["hp_fused"][field]
        n += 1
        hit += ours is not None and abs(float(ours) - float(ref)) < 1e-6
    return hit, n


def field_accuracy(d):
    """Per-field agreement with config.json on the main models, from the weights alone and after config fusion."""
    ok = d[(d["status"] == "ok") & d["has_config"].fillna(False)]
    ok = ok[~ok["id"].str.contains(FIXTURE, case=False, regex=True)]  # CI fixtures (tiny random weights) are not models
    created = pd.to_datetime(ok.get("created_at"), errors="coerce", utc=True) if "created_at" in ok else None
    sets = [("Original releases (top)", ok[(ok["stratum"] == "top") & (ok["origin"] == "original")]),
            ("Frontier providers, recent", ok[ok.get("also_frontier", pd.Series(False, index=ok.index)).fillna(False).astype(bool)])]
    if created is not None:
        sets.append(("Created in 2026", ok[created >= pd.Timestamp("2026-01-01", tz="UTC")]))
    L = ["", "## Weights vs config.json, field by field (repos with a config)", "",
         "Each cell: weights alone / with config fused (heads, kv_heads, head_dim need the config to split width into heads × size). "
         "CI fixtures (tiny random test repos) and placeholder values (0) are excluded.", "",
         "| field | " + " | ".join(f"{k} (n={len(g)})" for k, g in sets) + " |", "|---|" + "---|" * len(sets)]
    for f in FIELDS:
        cells = []
        for _, g in sets:
            h0, n0 = _score(g, f, False)
            h1, n1 = _score(g, f, True)
            cells.append("-" if not n0 else f"{h0/n0*100:.1f}% / {h1/n1*100:.1f}% ({n0})")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    fam = []
    for _, g in sets:
        x = g[g["family_config"].notna()]
        fam.append("-" if not len(x) else f"{(x['family_manifest'] == x['family_config']).mean()*100:.1f}% ({len(x)})")
    L.append("| family | " + " | ".join(fam) + " |")
    return L


def figures(d, groups, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
    st = ["ok", "gated", "no_weights", "legacy_pickle", "timeout", "error"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    h = 0.38
    for i, ((label, p), c) in enumerate(zip(groups, (BLUE, ORANGE))):
        vals = [wprop(p["status"] == s, p["weight"])[0] * 100 for s in st]
        ax.barh([y + (i - .5) * h for y in range(len(st))], vals, height=h - .04, color=c, label=label)
    ax.set_yticks(range(len(st)), st)
    ax.invert_yaxis()
    ax.set_xlabel("% of repos")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", lw=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("What happened when we read each repo", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "status.png"), dpi=160)
    plt.close(fig)

    o = d[(d["status"] == "ok") & (d["weight_bytes"] > 0)]
    if len(o):
        fig, ax = plt.subplots(figsize=(7, 4))
        for (label, p), c in zip(groups, (BLUE, ORANGE)):
            q = o[o["id"].isin(p["id"])]
            ax.scatter(q["weight_bytes"], q["header_bytes"], s=14, color=c, alpha=.75, label=label,
                       edgecolors=SURF, linewidths=.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("repo weight size (bytes)")
        ax.set_ylabel("bytes we read (headers)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e6e5e0", lw=.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        ax.set_title("Headers are a tiny slice of the weights", loc="left", color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "cost.png"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="out/census.jsonl")
    ap.add_argument("--out", default="out/report")
    a = ap.parse_args()
    main(a.census, a.out)
