# Figures for the paper from the evaluation outputs in research/res2.
import json, sys, os
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt
R = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/lab/arch/res2"
OUT = os.path.join(os.path.dirname(__file__), "..", "fig")
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6})
full, obf, v1 = (json.load(open(f"{R}/{n}.json"))["f1"] for n in ("full", "obf", "v1"))
names = {"attn": "attention", "cross": "cross-attention", "ffn": "feed-forward", "moe": "mixture of experts", "ssm": "SSM / linear attn.", "conv": "conv module", "rnn": "recurrent cell", "mod": "adaLN modulation", "embed": "input stem", "norm": "norm", "head": "output head", "adapter": "adapter"}
ks = list(names)[::-1]
fig, ax = plt.subplots(figsize=(3.3, 3.15))
y = range(len(ks)); h = .26
for k, (src, lab, col) in enumerate([(v1, "name rules", "#bdbdbd"), (obf, "ours: shapes only", "#8fa8d8"), (full, "ours: full", "#1b4f9c")]):
    ax.barh([i + (k - 1) * h for i in y], [src.get(x, 0) * 100 for x in ks], height=h, color=col, label=lab, linewidth=0)
ax.set_yticks(list(y)); ax.set_yticklabels([names[x] for x in ks]); ax.set_xlim(0, 100); ax.set_xlabel("F1 over tensors (%)")
ax.legend(loc="lower left", frameon=False, fontsize=6.6, bbox_to_anchor=(-.45, 1.0), ncol=3, handlelength=1, columnspacing=.8, handletextpad=.4); ax.tick_params(axis="y", length=0)
fig.tight_layout(pad=.3); fig.savefig(f"{OUT}/f1_roles.pdf")
pts = json.load(open(f"{R}/header_points.json"))
fig, ax = plt.subplots(figsize=(3.3, 3.0))
xs = [p[1] / 1e9 for p in pts if p[1] > 0]; ys = [p[0] / 1024 for p in pts if p[1] > 0]
ax.scatter(xs, ys, s=4, color="#1b4f9c", alpha=.55, linewidths=0)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("weights in BF16 (GB)"); ax.set_ylabel("safetensors header (KB)")
for r, lab in ((1e-3, "$10^{-3}$"), (1e-5, "$10^{-5}$"), (1e-7, "$10^{-7}$")):
    xx = [1e-4, 5e3]; ax.plot(xx, [x * 1e9 * r / 1024 for x in xx], ls="--", lw=.5, color="#999")
    yl = 500; xl = yl * 1024 / (1e9 * r)
    if xl > 4e3: xl = 3e3; yl = xl * 1e9 * r / 1024
    if 1e-3 < xl < 5e3 and 1 < yl < 1e3: ax.text(xl, yl, "ratio " + lab, fontsize=6.5, color="#777", ha="right", va="bottom", rotation=0)
ax.set_xlim(1e-3, 5e3); ax.set_ylim(1, 1e3)
fig.tight_layout(pad=.3); fig.savefig(f"{OUT}/header_cost.pdf")
print("ok")
