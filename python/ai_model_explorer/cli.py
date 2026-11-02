"""Command line: `ai-model-explorer open [REPO]` starts the web app locally; `ai-model-explorer inspect REPO` prints the anatomy."""
from __future__ import annotations

import argparse
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


def _serve(repo: str | None, port: int, lang: str, no_browser: bool) -> None:
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ai-model-explorer",
                                description="See any AI model's architecture from its safetensors header, without downloading weights.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd")

    o = sub.add_parser("open", help="open the web app in your browser (runs locally, works offline for files)")
    o.add_argument("repo", nargs="?", help="Hugging Face repo to load right away, e.g. Qwen/Qwen3-8B")
    o.add_argument("--port", type=int, default=0, help="port (default: any free port)")
    o.add_argument("--lang", choices=["th", "en"], default="en")
    o.add_argument("--no-browser", action="store_true", help="only print the URL")

    i = sub.add_parser("inspect", help="print the anatomy of a Hugging Face repo or local .safetensors")
    i.add_argument("model", help="repo id (owner/name), huggingface.co URL, .safetensors file or folder")
    i.add_argument("--json", action="store_true", help="print the full result as JSON")
    i.add_argument("--tensors", action="store_true", help="with --json: include the role of every tensor")
    i.add_argument("--token", help="Hugging Face token for gated/private repos (default: $HF_TOKEN)")
    i.add_argument("--config", help="path to a config.json to use instead of the repo's own")

    a = p.parse_args(argv)
    if a.cmd in (None, "open"):
        _serve(getattr(a, "repo", None), getattr(a, "port", 0), getattr(a, "lang", "en"), getattr(a, "no_browser", False))
        return 0

    from .api import inspect, summary
    cfg = None
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f)
    try:
        r = inspect(a.model, token=a.token, config=cfg, include_tensors=a.tensors)
    except (PermissionError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(r, indent=1, ensure_ascii=False) if a.json else summary(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
