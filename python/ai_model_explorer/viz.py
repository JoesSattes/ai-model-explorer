"""Draw a model's anatomy with matplotlib: components in data order, each layer stack as a strip of blocks coloured
by block type, and a bar of where the parameters sit."""
from __future__ import annotations

from typing import Optional

THEMES = {
    "light": {"bg": "#ffffff", "panel": "#f4f5f8", "edge": "#d5d8e2", "ink": "#141414", "soft": "#5a5f6e", "dim": "#8a8f9c"},
    "dark": {"bg": "#0b1020", "panel": "#121a2e", "edge": "#2a3454", "ink": "#f4f5f8", "soft": "#b9c0d4", "dim": "#8f97b0"},
}
KIND_COLOR = {"attn": "#5b6bf0", "cross": "#8b96fa", "ffn": "#a78bfa", "moe": "#e5007d", "ssm": "#14a38b", "conv": "#f59e0b",
              "rnn": "#0ea5e9", "mod": "#f97316", "embed": "#64748b", "head": "#94a3b8", "norm": "#cbd5e1", "adapter": "#ff6bb5",
              "other": "#9ca3af"}


def _type_color(label: str) -> str:
    l = (label or "").lower()
    for keys, kind in ((("moe", "expert"), "moe"), (("mamba", "ssm", "linear", "delta", "rwkv", "mixer"), "ssm"),
                       (("cross",), "cross"), (("conv", "conformer"), "conv"), (("lstm", "gru", "rnn"), "rnn"),
                       (("adaln", "dit", "mod"), "mod")):
        if any(k in l for k in keys):
            return KIND_COLOR[kind]
    return KIND_COLOR["attn"]


def _order(model) -> list:
    comps = list(model.components)
    rank = {"encoder": 0, "adapter": 1, "aux": 1, "backbone": 2, "prediction": 3, "joint": 4}
    return sorted(comps, key=lambda c: (rank.get(c.get("role"), 2), -(c.get("params") or 0)))


def plot_model(model, path: Optional[str] = None, theme: str = "light", width: float = 12.0, dpi: int = 200, max_components: int = 5):
    try:
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Rectangle
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install matplotlib") from e
    from .model import KIND_LABEL, fmt_b, fmt_n

    T = THEMES.get(theme, THEMES["light"])
    comps = _order(model)[:max_components]
    n = max(len(comps), 1)
    stacks_max = max([len(c.get("stacks") or []) for c in comps] + [1])
    box_h = 1.0 + 0.72 * stacks_max
    H = 1.85 + box_h + 1.35
    fig = plt.figure(figsize=(width, H * width / 12.0), dpi=dpi, facecolor=T["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, H)
    ax.axis("off")

    # title
    q = model.data.get("quantization") or {}
    pn = f"{fmt_n(model.params)} parameters" + (f" ({fmt_n(model.distinct_params)} distinct)" if model.duplicate_head_params else "")
    sub = f"{model.family} · {pn} · {fmt_b(model.stored_bytes)}" + (f" · {', '.join(q)}" if q else "")
    act = model.active_params
    if model.hp.get("experts") and act:
        sub += f" · {fmt_n(act)} active per token"
    title = model.name if model.data["source"].get("kind") == "huggingface" else "/".join(model.name.rstrip("/").split("/")[-2:])
    ax.text(0.35, H - 0.45, title, fontsize=17, fontweight="bold", color=T["ink"], va="center")
    ax.text(0.35, H - 0.88, sub, fontsize=10.5, color=T["soft"], va="center")
    hp = model.hp
    encs = [c for c in model.components if c.get("role") == "encoder" and c.get("layers")]
    lay = (f"{encs[0]['layers']} encoder + {hp['layers']} decoder layers" if model.family == "encoder-decoder" and encs and hp.get("layers")
           else f"{hp['layers']} layers" if hp.get("layers") else None)
    bits = [lay, f"width {hp['d']}" if hp.get("d") else None,
            f"{hp['heads']} heads / {hp['kv_heads']} KV" if hp.get("heads") and hp.get("kv_heads") else None,
            f"{hp['experts']} experts, {hp['experts_per_token']} per token" if hp.get("experts") and hp.get("experts_per_token") else None,
            f"vocab {hp['vocab']:,}" if hp.get("vocab") else None]
    kvt = model.kv_cache_bytes_per_token()
    if kvt:
        bits.append(f"KV cache {fmt_b(kvt)}/token")
    ax.text(0.35, H - 1.22, "  ·  ".join(b for b in bits if b), fontsize=9, color=T["dim"], va="center")

    # components
    gap = 0.35
    bw = (12 - 0.7 - gap * (n - 1)) / n
    y0 = H - 1.65 - box_h
    types_seen = {}
    centers = {}
    for i, c in enumerate(comps):
        x = 0.35 + i * (bw + gap)
        centers[c.get("id")] = (x, x + bw)
        ax.add_patch(FancyBboxPatch((x, y0), bw, box_h, boxstyle="round,pad=0,rounding_size=0.12", fc=T["panel"], ec=T["edge"], lw=1))
        name = c.get("id") if c.get("id") != "(root)" else "model"
        fit = lambda t, cw: t if len(t) <= int((bw - 0.3) / cw) else "…" + t[-max(int((bw - 0.3) / cw) - 1, 4):]
        name = fit(name, 0.092)
        ax.text(x + 0.18, y0 + box_h - 0.32, name, fontsize=10, fontweight="bold", color=T["ink"], va="center")
        role = "decoder" if model.family == "encoder-decoder" and c.get("role") == "backbone" else c.get("role")
        meta = f"{role} · {fmt_n(c.get('params'))} params" + (f" · d={c['d']}" if c.get("d") and bw > 3 else "")
        meta = meta if len(meta) <= int((bw - 0.3) / 0.072) else meta[:int((bw - 0.3) / 0.072)]
        ax.text(x + 0.18, y0 + box_h - 0.62, meta, fontsize=8.5, color=T["soft"], va="center")
        for j, st in enumerate((c.get("stacks") or [])[:stacks_max]):
            sy = y0 + box_h - 1.02 - j * 0.72
            nb = st.get("n") or 0
            tys = st.get("types") or []
            # block type of every item, in order
            seq = []
            type_of = st.get("typeOf")
            items = st.get("items") or []
            if isinstance(type_of, dict) and items:
                seq = [type_of.get(it, 0) for it in sorted(items, key=lambda s: int(s.rsplit(".", 1)[-1]) if s.rsplit(".", 1)[-1].isdigit() else 0)]
            elif isinstance(type_of, list) and type_of:
                seq = type_of
            if len(seq) != nb:
                seq = [ti for ti, ty in enumerate(tys) for _ in range(ty.get("count") or 0)][:nb] or [0] * nb
            cw = (bw - 0.36) / max(nb, 1)
            for k, ti in enumerate(seq):
                ty = tys[ti] if ti < len(tys) else {}
                col = _type_color(ty.get("label"))
                is_mtp = st.get("mtp") and k == nb - 1
                ax.add_patch(Rectangle((x + 0.18 + k * cw, sy - 0.12), max(cw - min(0.02, cw * 0.2), 0.004), 0.24,
                                       fc=col, ec="none", alpha=0.45 if is_mtp else 1))
            for ty in tys:
                lab = ty.get("label") or "block"
                types_seen.setdefault(lab, _type_color(lab))
            tail = f" · {nb} blocks" + (" · last = MTP" if st.get("mtp") else "")
            room = int((bw - 0.3) / 0.062) - len(tail)
            lab = st.get("path") or ""
            lab = lab if len(lab) <= room else ("…" + lab[-(room - 1):] if room > 6 else "")
            desc = (lab + tail).lstrip(" ·")
            ax.text(x + 0.18, sy - 0.3, desc, fontsize=7.5, color=T["dim"], va="center")
    for e in model.data.get("edges") or []:
        a, b = centers.get(e.get("from")), centers.get(e.get("to"))
        ia, ib = [i for i, c in enumerate(comps) if c.get("id") == e.get("from")], [i for i, c in enumerate(comps) if c.get("id") == e.get("to")]
        if a and b and ia and ib and ib[0] == ia[0] + 1:
            ax.annotate("", xy=(b[0], y0 + box_h / 2), xytext=(a[1], y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=T["dim"], lw=1.2))

    # legend of block types
    lx = 0.35
    for lab, col in list(types_seen.items())[:8]:
        ax.add_patch(Rectangle((lx, y0 - 0.32), 0.18, 0.14, fc=col, ec="none"))
        ax.text(lx + 0.25, y0 - 0.25, lab, fontsize=8, color=T["soft"], va="center")
        lx += 0.45 + 0.075 * len(lab)

    # parameter share bar
    bd = [r for r in model.breakdown("kind") if r["share"] > 0]
    by = 0.45
    ax.text(0.35, by + 0.52, "where the parameters sit", fontsize=8.5, color=T["dim"], va="center")
    x = 0.35
    for r in bd:
        w = 11.3 * r["share"]
        ax.add_patch(Rectangle((x, by), max(w, 0.003), 0.3, fc=KIND_COLOR.get(r["key"], KIND_COLOR["other"]), ec=T["bg"], lw=0.6))
        if w > 1.1:
            ax.text(x + 0.08, by + 0.15, f"{r['name']} {r['share']:.0%}", fontsize=8, color="#ffffff", va="center", fontweight="bold")
        x += w
    small = [r for r in bd if 11.3 * r["share"] <= 1.1 and r["share"] >= 0.005]
    if small:
        ax.text(0.35, by - 0.2, " · ".join(f"{r['name']} {r['share']:.1%}" for r in small), fontsize=7.5, color=T["dim"], va="center")
    ax.text(11.65, 0.12, "AI Model Explorer · read from the safetensors header", fontsize=7, color=T["dim"], ha="right", va="center")

    if path:
        fig.savefig(path, facecolor=T["bg"], dpi=dpi)
        plt.close(fig)
    return fig
