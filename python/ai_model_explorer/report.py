"""A self-contained HTML report for one model: diagram, insights, breakdowns and memory estimates."""
from __future__ import annotations

import html
import io
import urllib.parse

from .model import fmt_b, fmt_n

CSS = {
    "light": "--bg:#ffffff;--panel:#f4f5f8;--line:#e3e5ec;--ink:#141414;--soft:#4d5160;--dim:#7a7f8d;--acc:#5b6bf0;--acc2:#e5007d",
    "dark": "--bg:#0b1020;--panel:#121a2e;--line:#222b44;--ink:#f4f5f8;--soft:#b9c0d4;--dim:#8f97b0;--acc:#8b96fa;--acc2:#ff6bb5",
}


def _table(head, rows) -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def write_report(model, path: str, theme: str = "light") -> str:
    svg = ""
    try:
        buf = io.StringIO()
        fig = model.plot(None, theme=theme)
        fig.savefig(buf, format="svg", facecolor=fig.get_facecolor())
        import matplotlib.pyplot as plt
        plt.close(fig)
        svg = buf.getvalue()
        svg = svg[svg.index("<svg"):]
    except Exception as e:  # matplotlib missing or a drawing problem: the rest of the report still works
        svg = f"<p class='note'>Diagram unavailable: {html.escape(str(e))}</p>"

    hp = model.hp
    src = model.data["source"]
    web = ""
    if src.get("kind") == "huggingface":
        web = ("https://joessattes.github.io/ai-model-explorer/en/#hf=" + urllib.parse.quote(src["repo"], safe="/"))
    facts = [("Family", model.family), ("Parameters", fmt_n(model.params)), ("Stored", fmt_b(model.stored_bytes)),
             ("Storage", ", ".join(model.data.get("quantization") or {}) or "unquantized"),
             ("Active per token", fmt_n(model.active_params) if hp.get("experts") and model.active_params else None),
             ("Layers", hp.get("layers")), ("Width", hp.get("d")), ("Heads / KV heads", f"{hp.get('heads')} / {hp.get('kv_heads')}" if hp.get("heads") else None),
             ("Head size", hp.get("head_dim")), ("FFN width", hp.get("ffn")),
             ("Experts", f"{hp.get('experts')} ({hp.get('experts_per_token')} per token)" if hp.get("experts") else None),
             ("Vocabulary", f"{hp['vocab']:,}" if hp.get("vocab") else None),
             ("KV cache per token (16-bit)", fmt_b(model.kv_cache_bytes_per_token()) if model.kv_cache_bytes_per_token() else None),
             ("Header bytes read", f"{fmt_b(model.data.get('header_bytes'))} from {model.data.get('files_read')} file(s)")]
    facts_html = "".join(f"<div class='fact'><span>{html.escape(k)}</span><b>{html.escape(str(v))}</b></div>" for k, v in facts if v not in (None, "None / None"))

    kinds = _table(["Part", "Parameters", "Share", "Stored"],
                   [(r["name"], fmt_n(r["params"]), f"{r['share']:.1%}", fmt_b(r["bytes"])) for r in model.breakdown("kind")])
    comps = _table(["Component", "Role", "Parameters", "Layer stacks"],
                   [(c.get("id"), c.get("role"), fmt_n(c.get("params")),
                     "; ".join(f"{st.get('path')}: {st.get('n')} × " + ", ".join(f"{t.get('label')} ×{t.get('count')}" for t in st.get("types") or [])
                               for st in c.get("stacks") or []) or "–") for c in model.components])
    mem_rows = []
    for bits, lab in ((None, "as stored"), (16, "16-bit"), (8, "8-bit"), (4, "4-bit")):
        row = [lab]
        for ctx in (4096, 32768, 131072):
            m = model.memory(context=ctx, weight_bits=bits)
            row.append(fmt_b(m["total_bytes"]) if m["kv_cache_bytes"] is not None else fmt_b(m["weights_bytes"]) + " + KV")
        mem_rows.append(row)
    mem = _table(["Weights", "4k context", "32k context", "128k context"], mem_rows)
    ins = "".join(f"<li>{html.escape(x)}</li>" for x in model.insights())
    title = html.escape(model.name)
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · AI Model Explorer report</title>
<style>
:root{{{CSS.get(theme, CSS['light'])}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:32px 24px 48px}}h1{{font-size:28px;margin:0 0 4px}}h2{{font-size:19px;margin:34px 0 10px}}
.sub{{color:var(--soft)}}.sub a{{color:var(--acc)}}.diagram svg{{width:100%;height:auto;border-radius:12px;border:1px solid var(--line);margin-top:18px}}
.facts{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-top:18px}}
.fact{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}}.fact span{{display:block;color:var(--dim);font-size:12px}}.fact b{{font-size:16px}}
ul.ins{{padding-left:20px}}ul.ins li{{margin:6px 0;color:var(--soft)}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}}th{{color:var(--dim);font-weight:600}}
.note{{color:var(--dim);font-size:13px}}footer{{margin-top:40px;color:var(--dim);font-size:13px}}footer a{{color:var(--acc)}}
</style></head><body><div class="wrap">
<h1>{title}</h1>
<div class="sub">Anatomy recovered from the safetensors header, without downloading the weights.{f' <a href="{web}">Open in the web app</a>' if web else ''}</div>
<div class="diagram">{svg}</div>
<div class="facts">{facts_html}</div>
<h2>Insights</h2><ul class="ins">{ins}</ul>
<h2>Where the parameters sit</h2>{kinds}
<h2>Components</h2>{comps}
<h2>Memory to run it</h2>{mem}
<p class="note">Weights plus KV cache for one sequence at 16-bit cache precision. Activations and runtime overhead are not included.</p>
<footer>Made with <a href="https://github.com/JoesSattes/ai-model-explorer">AI Model Explorer</a> (<code>pip install ai-model-explorer</code>).
Method: Sattaya Singkul, <i>Reading the Architecture off the Weights: Manifest-Only Model Anatomy and a Census of 12,974 Hugging Face Repositories</i>, 2026.</footer>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
