# Appendix tables from the evaluation outputs: per-model Hub results, per-group summary, failures, corpus per-model accuracy,
# top confusions and the models below 90%.  usage: python appendix_tables.py <rb.json> <res2 dir> <out dir>
import json, sys, os, collections
rb, res, out = sys.argv[1], sys.argv[2], sys.argv[3]
D = json.load(open(rb)); rows = [r for r in D["rows"] if not r.get("err")]; S = D["summary"]
GROUP = {"llm": "Dense LM", "moe": "MoE LM", "ssm": "SSM / hybrid", "enc": "Encoder", "emb": "Embedding", "s2s": "Seq2seq", "speech": "Speech / audio",
         "vision": "Vision", "vlm": "Vision--language", "gen": "Image / video gen.", "ts": "Time series / tabular", "quant": "Quantized release"}
ORDER = list(GROUP)
def tex(s): return str(s).replace("\\", "/").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
def fmt(n):
    if n is None: return "--"
    n = float(n)
    for u, k in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= k: v = n / k; return (f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}") + "\\," + u
    return f"{n:.0f}"
def kb(b): return f"{b/1024:.0f}" if b < 1024 * 1024 else f"{b/1024/1024:.1f}\\,M"
def hub_ok(r):
    if not r.get("apiTotal"): return None
    if r.get("diffusers"): return "den"
    return "=" if r["apiTotal"] in (r["storedHub"], r["params"], r["stored"]) else "$\\ne$"
STOR = {None: "", "fp8": "FP8", "nvfp4": "NVFP4", "mxfp4": "MXFP4", "gptq": "GPTQ", "awq": "AWQ", "mlx": "MLX", "bnb4": "NF4", "int8": "INT8"}
FAM = {"decoder": "dec", "encoder": "enc", "prefix-lm": "prefix", "encoder-decoder": "enc-dec", "diffusion": "diff", "transducer": "trans", "unknown": "?"}

# ---- per-model Hub table (longtable)
L = []
L.append("\\scriptsize\\setlength{\\tabcolsep}{2.5pt}")
L.append("\\begin{longtable}{@{}llrrclrrrr@{}}")
head = "Repository & Storage & Params & Hub & & Family & $d$ & Layers & Header (KB) & Roles \\\\"
L.append("\\caption{All 191 checkpoints read from the Hub, by group. \\emph{Params} is our logical count and \\emph{Hub} the count on the model page; the next column marks agreement ($=$), a difference ($\\ne$, explained in Section~\\ref{sec:hub}) or a diffusers pipeline whose Hub count equals our denoiser (den). $d$ and layers are for the backbone. \\emph{Roles} is role accuracy against labels transferred from the corpus, where the model type is in the corpus.}\\label{tab:hub-all}\\\\")
L.append("\\toprule " + head + " \\midrule\\endfirsthead")
L.append("\\toprule " + head + " \\midrule\\endhead")
L.append("\\bottomrule\\endfoot")
for g in ORDER:
    rs = sorted([r for r in rows if r["cat"] == g], key=lambda r: -(r["params"] or 0))
    if not rs: continue
    L.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textbf{{{GROUP[g]}}} ({len(rs)})}} \\\\")
    for r in rs:
        p = r.get("pred") or {}
        ro = r.get("roles"); ro = f"{ro['acc']*100:.0f}" if ro and ro.get("acc") is not None and ro.get("coverage", 0) >= .5 else ""
        L.append(" & ".join([f"{{\\tiny\\texttt{{{tex(r['repo'])}}}}}", STOR.get(r["quant"], tex(r["quant"] or "")), fmt(r["params"]), fmt(r.get("apiTotal")), hub_ok(r) or "",
                             FAM.get(r["family"], tex(r["family"])), str(p.get("d") or "--"), str(p.get("layers") or "--"), kb(r["headerBytes"]), ro]) + " \\\\")
L.append("\\end{longtable}")
open(os.path.join(out, "app_hub_all.tex"), "w").write("\n".join(L) + "\n")

# ---- per-group summary
T = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{lrrrrrrr}", "\\toprule",
     "Group & Repos & Read & Hub $=$ & Heads (fused) & Roles & Median header & Median time \\\\", "\\midrule"]
fails = collections.Counter(f["cat"] for f in S["fails"])
def med(a): a = sorted(a); return a[len(a)//2] if a else None
tot = [0, 0]
for g in ORDER:
    rs = [r for r in rows if r["cat"] == g]; n = len(rs) + fails.get(g, 0)
    if not n: continue
    hc = [hub_ok(r) for r in rs if hub_ok(r) in ("=", "$\\ne$")]
    hh = [r["hp"]["heads"] for r in rs if r.get("hp") and "heads" in r["hp"]]
    rr = [r["roles"]["acc"] for r in rs if r.get("roles") and r["roles"].get("acc") is not None and r["roles"]["coverage"] >= .5]
    T.append(f"{GROUP[g]} & {n} & {len(rs)} & {sum(x == '=' for x in hc)}/{len(hc)} & " + (f"{sum(h[2] == h[0] for h in hh)}/{len(hh)}" if hh else "--") + " & " +
             (f"{100*sum(rr)/len(rr):.1f}\\% ({len(rr)})" if rr else "--") + f" & {kb(med([r['headerBytes'] for r in rs]))}\\,KB & {med([r['ms'] for r in rs]):.0f}\\,ms \\\\".replace("M\\,KB", "MB"))
T += ["\\bottomrule", "\\end{tabular}", "\\caption{Hub benchmark by group. \\emph{Hub $=$} counts transformers-style repositories whose total agrees with the model page; diffusers pipelines are excluded here (their Hub count equals our denoiser in all 12 cases). \\emph{Heads (fused)} is the head count after configuration fusion, against \\tn{config.json}; \\emph{Roles} is macro role accuracy on models whose type is in the corpus (number of models in brackets).}", "\\label{tab:hub-groups}", "\\end{table}"]
open(os.path.join(out, "app_hub_groups.tex"), "w").write("\n".join(T) + "\n")

# ---- failures
F = ["\\begin{table}[H]", "\\centering", "\\small", "\\begin{tabular}{@{}llp{5.9cm}@{}}", "\\toprule", "Repository & Group & Reason \\\\", "\\midrule"]
REASON = lambda e: "gated: the license must be accepted on the Hub first" if ("Access to" in e or "401" in e) else "old non-zip PyTorch pickle (not read by design)" if "zip-format" in e else "rate limited: one adapter file per language; the loader now skips adapters" if "429" in e else tex(e)
for f in sorted(S["fails"], key=lambda f: (REASON(f["err"]), f["repo"])):
    F.append(f"\\texttt{{{tex(f['repo'])}}} & {GROUP[f['cat']]} & {REASON(f['err'])} \\\\")
F += ["\\bottomrule", "\\end{tabular}", "\\caption{The ten repositories that could not be read.}", "\\label{tab:hub-fail}", "\\end{table}"]
open(os.path.join(out, "app_hub_fail.tex"), "w").write("\n".join(F) + "\n")

# ---- corpus per-model accuracy, three columns
pm = json.load(open(os.path.join(res, "full_dump.json")))["perModel"]; po = {r["type"]: r for r in json.load(open(os.path.join(res, "obf_dump.json")))["perModel"]}
pm = sorted([r for r in pm if r["acc"] is not None], key=lambda r: r["type"])
C = ["\\scriptsize\\setlength{\\tabcolsep}{3pt}", "\\begin{longtable}{@{}lrr@{\\hspace{24pt}}lrr@{}}",
     "\\caption{Per-model role accuracy (\\%) on the corpus: full method / shapes only. Model types are the \\tn{transformers} configuration names.}\\label{tab:corpus-all}\\\\",
     "\\toprule Type & Full & Shapes & Type & Full & Shapes \\\\ \\midrule\\endfirsthead", "\\toprule Type & Full & Shapes & Type & Full & Shapes \\\\ \\midrule\\endhead", "\\bottomrule\\endfoot"]
n = len(pm); per = (n + 1) // 2
for i in range(per):
    cells = []
    for c in range(2):
        j = c * per + i
        if j < n:
            r = pm[j]; o = po.get(r["type"]); cells += [f"\\texttt{{{tex(r['type'])}}}", f"{r['acc']*100:.1f}", f"{o['acc']*100:.1f}" if o and o["acc"] is not None else "--"]
        else: cells += ["", "", ""]
    C.append(" & ".join(cells) + " \\\\")
C.append("\\end{longtable}")
open(os.path.join(out, "app_corpus_all.tex"), "w").write("\n".join(C) + "\n")

# ---- top confusions (full method), by number of tensors
f = json.load(open(os.path.join(res, "full.json")))
X = ["\\begin{table}[H]", "\\centering", "\\small", "\\begin{tabular}{@{}lrp{7.2cm}@{}}", "\\toprule", "Ground truth $\\to$ predicted & Tensors & Models (examples) \\\\", "\\midrule"]
for k, cnt, ex in f["topConfN"][:16]:
    ms = sorted(set(e.split(":")[0] for e in ex))
    X.append(f"{tex(k).replace('→', ' $\\to$ ')} & {cnt} & {tex(', '.join(ms))} \\\\")
X += ["\\bottomrule", "\\end{tabular}", "\\caption{The most frequent confusions of the full method on the corpus, counted in tensors.}", "\\label{tab:conf}", "\\end{table}"]
open(os.path.join(out, "app_conf.tex"), "w").write("\n".join(X) + "\n")
print("ok", len(rows), n)
