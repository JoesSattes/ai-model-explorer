"""Command line for AI Model Explorer.

  ai-model-explorer inspect  MODEL [--json] [--tensors]      anatomy summary (or everything as JSON)
  ai-model-explorer insights MODEL                           findings in plain language
  ai-model-explorer plot     MODEL -o anatomy.png            diagram as PNG, SVG or PDF
  ai-model-explorer report   MODEL -o report.html            self-contained HTML report
  ai-model-explorer memory   MODEL --context 32768           weights + KV cache estimate
  ai-model-explorer compare  MODEL MODEL ... [--csv|--json]  side-by-side table
  ai-model-explorer fits     MODEL DEVICE [--context N]      will it fit; longest context that fits
  ai-model-explorer diff     A B                             what differs between two checkpoints
  ai-model-explorer scan     [REPOS] [--file F] [-o F.csv]   many repos into one CSV
  ai-model-explorer ready    MODEL [--submit]                Deploy-Ready board verdict
  ai-model-explorer census   MODEL                           position among the Hub's popular language models
  ai-model-explorer devices                                  device names known to fits
  ai-model-explorer open     [MODEL]                         the web app on your machine (also: no command)
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
    p.add_argument("--token", help="Hugging Face token for gated/private repos (default: $HF_TOKEN, then $HUGGING_FACE_HUB_TOKEN)")
    p.add_argument("--config", help="path to a config.json to use instead of the repo's own")


def _load_cfg(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ai-model-explorer", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Recover a model's architecture from its safetensors headers, without downloading the weights.",
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
    pl.add_argument("--theme", choices=["light", "dark"], default="light", help="diagram colors (default light)")
    pl.add_argument("--dpi", type=int, default=200, help="image resolution for PNG (default 200)")

    r = sub.add_parser("report", help="write a self-contained HTML report")
    _model_args(r)
    r.add_argument("-o", "--output", default="report.html", help="output HTML file (default report.html)")
    r.add_argument("--theme", choices=["light", "dark"], default="light", help="report colors (default light)")

    m = sub.add_parser("memory", help="estimate memory for weights and KV cache")
    _model_args(m)
    m.add_argument("--context", type=int, default=8192, help="tokens per sequence (default 8192)")
    m.add_argument("--batch", type=int, default=1, help="sequences at once (default 1)")
    m.add_argument("--weight-bits", type=float, default=None, help="weight precision in bits (default: as stored)")
    m.add_argument("--kv-bits", type=int, default=16, help="KV cache precision in bits (default 16)")

    c = sub.add_parser("compare", help="compare several models side by side")
    c.add_argument("models", nargs="+", help="repo ids, URLs, .safetensors files or folders")
    c.add_argument("--token", help="Hugging Face token (default: $HF_TOKEN)")
    c.add_argument("--csv", action="store_true", help="print CSV instead of a Markdown table")
    c.add_argument("--json", action="store_true", help="print JSON")

    f = sub.add_parser("fits", help="will it fit on a device? longest context that fits, and what would make it fit")
    _model_args(f)
    f.add_argument("device", help="device name (run 'ai-model-explorer devices') or memory in GB, e.g. rtx-4090, mac-64, 24")
    f.add_argument("--context", type=int, default=8192, help="tokens per sequence (default 8192)")
    f.add_argument("--batch", type=int, default=1, help="sequences at once (default 1)")
    f.add_argument("--weight-bits", type=float, default=None, help="weight precision in bits (default: as stored)")
    f.add_argument("--kv-bits", type=int, default=16, help="KV cache precision in bits (default 16)")

    d = sub.add_parser("diff", help="what differs between two checkpoints (fine-tune vs base, quantized vs original)")
    d.add_argument("a", help="first checkpoint (e.g. the base)")
    d.add_argument("b", help="second checkpoint (e.g. the fine-tune)")
    d.add_argument("--token", help="Hugging Face token (default: $HF_TOKEN)")
    d.add_argument("--json", action="store_true", help="print JSON")

    sc = sub.add_parser("scan", help="inspect many repos (args or a file, one per line) into one table")
    sc.add_argument("repos", nargs="*", help="repo ids; or use --file")
    sc.add_argument("--file", help="text file with one repo id per line")
    sc.add_argument("--token", help="Hugging Face token (default: $HF_TOKEN)")
    sc.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    sc.add_argument("-o", "--output", help="write CSV here (default: print CSV)")

    sub.add_parser("devices", help="list the device names known to 'fits'")
    rd = sub.add_parser("ready", help="the Deploy-Ready board's verdict: device class at 4-bit/8k and five metadata checks")
    rd.add_argument("model", help="Hugging Face repo id, URL, .safetensors file or folder")
    rd.add_argument("--json", action="store_true", help="print JSON")
    rd.add_argument("--submit", action="store_true", help="open the board submission for this model in the browser")
    cs = sub.add_parser("census", help="where a model sits among the Hub's popular language models")
    cs.add_argument("model", help="Hugging Face repo id, URL, .safetensors file or folder")
    cs.add_argument("--json", action="store_true", help="print JSON")
    cs.add_argument("--refresh", action="store_true", help="download the census table again")
    for x in (rd, cs):
        x.add_argument("--token", help="Hugging Face token (default: $HF_TOKEN)")

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
    from . import tools
    try:
        if a.cmd == "devices":
            for k, v in tools.DEVICES.items():
                print(f"  {k:22} {v:6.1f} GB usable")
            return 0
        if a.cmd == "ready":
            r = tools.ready(a.model, token=a.token)
            print(json.dumps(r, indent=1) if a.json else tools.ready_text(r))
            if a.submit and r["submit_url"]:
                import webbrowser
                webbrowser.open(r["submit_url"])
            return 0
        if a.cmd == "census":
            r = tools.census(a.model, token=a.token, rows=tools._census_rows(refresh=a.refresh))
            print(json.dumps(r, indent=1) if a.json else tools.census_text(r))
            return 0
        if a.cmd == "diff":
            rows = tools.diff(a.a, a.b, token=a.token)
            print(json.dumps(rows, indent=1) if a.json else tools.diff_text(rows, a.a, a.b))
            return 0
        if a.cmd == "scan":
            repos = list(a.repos) + (open(a.file, encoding="utf-8").read().splitlines() if a.file else [])
            rows = tools.scan(repos, token=a.token, workers=a.workers)
            out = tools.to_csv(rows)
            if a.output:
                open(a.output, "w", encoding="utf-8").write(out)
                bad = sum(1 for r in rows if r.get("error"))
                print(f"wrote {a.output}: {len(rows) - bad} read, {bad} failed")
            else:
                print(out, end="")
            return 0
        if a.cmd == "fits":
            dev = float(a.device) if a.device.replace(".", "", 1).isdigit() else a.device
            if isinstance(dev, str) and dev.lower() not in tools.DEVICES:
                print(f"error: unknown device '{dev}'. Run 'ai-model-explorer devices', or give memory in GB.", file=sys.stderr)
                return 1
            r = tools.fits(explore(a.model, token=a.token, config=_load_cfg(a.config)), dev, context=a.context, batch=a.batch,
                           weight_bits=a.weight_bits, kv_bits=a.kv_bits)
            print(tools.fits_text(r))
            return 0
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
