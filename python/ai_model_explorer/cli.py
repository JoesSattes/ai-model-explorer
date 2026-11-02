"""Command line for AI Model Explorer.

  ai-model-explorer inspect  MODEL [--json] [--tensors]      anatomy summary (or everything as JSON)
  ai-model-explorer insights MODEL                           findings in plain language
  ai-model-explorer plot     MODEL -o anatomy.png            diagram as PNG, SVG or PDF
  ai-model-explorer report   MODEL -o report.html            self-contained HTML report
  ai-model-explorer memory   MODEL --context 32768           weights + KV cache estimate
  ai-model-explorer compare  MODEL MODEL ... [--csv|--json]  side-by-side table
  ai-model-explorer open     [MODEL]                         the web app on your machine
"""
from __future__ import annotations

import argparse
import csv
import functools
import http.server
import json
import socketserver
import sys
import threading
import urllib.parse
import webbrowser
from importlib import resources

from . import __version__


def _serve(repo, port: int, lang: str, no_browser: bool) -> None:
    web = resources.files("ai_model_explorer").joinpath("web")
    with resources.as_file(web) as root:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
        handler.log_message = lambda *a, **k: None
        with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
            url = f"http://127.0.0.1:{httpd.server_address[1]}/" + ("en/" if lang == "en" else "")
            if repo:
                url += "#hf=" + urllib.parse.quote(repo, safe="/")
            print(f"AI Model Explorer {__version__} running at {url}  (Ctrl+C to stop)")
            if not no_browser:
                threading.Timer(0.4, lambda: webbrowser.open(url)).start()
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print()


def _model_args(p):
    p.add_argument("model", help="Hugging Face repo id (owner/name), huggingface.co URL, .safetensors file or folder")
    p.add_argument("--token", help="Hugging Face token for gated/private repos (default: $HF_TOKEN)")
    p.add_argument("--config", help="path to a config.json to use instead of the repo's own")


def _load_cfg(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ai-model-explorer", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="See any AI model's architecture from its safetensors header, without downloading weights.",
                                epilog=__doc__.split("\n", 2)[2])
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd")

    i = sub.add_parser("inspect", help="anatomy summary, or the full result as JSON")
    _model_args(i)
    i.add_argument("--json", action="store_true", help="print the full result as JSON")
    i.add_argument("--tensors", action="store_true", help="with --json: include the role of every tensor")

    g = sub.add_parser("insights", help="findings in plain language")
    _model_args(g)

    pl = sub.add_parser("plot", help="draw the anatomy to an image")
    _model_args(pl)
    pl.add_argument("-o", "--output", default="anatomy.png", help="output file: .png, .svg or .pdf (default anatomy.png)")
    pl.add_argument("--theme", choices=["light", "dark"], default="light")
    pl.add_argument("--dpi", type=int, default=200)

    r = sub.add_parser("report", help="write a self-contained HTML report")
    _model_args(r)
    r.add_argument("-o", "--output", default="report.html")
    r.add_argument("--theme", choices=["light", "dark"], default="light")

    m = sub.add_parser("memory", help="estimate memory for weights and KV cache")
    _model_args(m)
    m.add_argument("--context", type=int, default=8192, help="tokens per sequence (default 8192)")
    m.add_argument("--batch", type=int, default=1, help="sequences at once (default 1)")
    m.add_argument("--weight-bits", type=float, default=None, help="weight precision in bits (default: as stored)")
    m.add_argument("--kv-bits", type=int, default=16, help="KV cache precision in bits (default 16)")

    c = sub.add_parser("compare", help="compare several models side by side")
    c.add_argument("models", nargs="+")
    c.add_argument("--token")
    c.add_argument("--csv", action="store_true", help="print CSV instead of a Markdown table")
    c.add_argument("--json", action="store_true", help="print JSON")

    o = sub.add_parser("open", help="open the web app in your browser (runs locally)")
    o.add_argument("model", nargs="?", help="Hugging Face repo to load right away, e.g. Qwen/Qwen3-8B")
    o.add_argument("--port", type=int, default=0, help="port (default: any free port)")
    o.add_argument("--lang", choices=["th", "en"], default="en")
    o.add_argument("--no-browser", action="store_true", help="only print the URL")

    a = p.parse_args(argv)
    if a.cmd in (None, "open"):
        _serve(getattr(a, "model", None), getattr(a, "port", 0), getattr(a, "lang", "en"), getattr(a, "no_browser", False))
        return 0

    from .model import compare, compare_table, explore, fmt_b
    try:
        if a.cmd == "compare":
            rows = compare(a.models, token=a.token)
            if a.json:
                print(json.dumps(rows, indent=1))
            elif a.csv:
                w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            else:
                print(compare_table(rows))
            return 0
        mdl = explore(a.model, token=a.token, config=_load_cfg(a.config), include_tensors=getattr(a, "tensors", False))
        if a.cmd == "inspect":
            print(mdl.to_json() if a.json else mdl.summary())
        elif a.cmd == "insights":
            print(mdl.name)
            for line in mdl.insights():
                print(f"  - {line}")
        elif a.cmd == "plot":
            mdl.plot(a.output, theme=a.theme, dpi=a.dpi)
            print(f"wrote {a.output}")
        elif a.cmd == "report":
            mdl.report(a.output, theme=a.theme)
            print(f"wrote {a.output}")
        elif a.cmd == "memory":
            r = mdl.memory(context=a.context, batch=a.batch, weight_bits=a.weight_bits, kv_bits=a.kv_bits)
            wb = "as stored" if a.weight_bits is None else f"{a.weight_bits:g}-bit"
            print(f"{mdl.name}")
            print(f"  weights ({wb}):  {fmt_b(r['weights_bytes'])}")
            if r["kv_cache_bytes"] is None:
                print("  KV cache:         unknown (no attention geometry; add --config)")
            else:
                print(f"  KV cache ({a.kv_bits}-bit, {a.batch} × {a.context:,} tokens):  {fmt_b(r['kv_cache_bytes'])}")
            print(f"  total:            {fmt_b(r['total_bytes'])}  (activations and runtime overhead not included)")
    except (PermissionError, FileNotFoundError, ValueError, RuntimeError, ConnectionError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
