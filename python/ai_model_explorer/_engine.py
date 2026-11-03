"""Run the explorer's own JavaScript (loaders.js, infer.js) inside QuickJS, so Python gets exactly the website's result.

Python does the network I/O: the JS side calls a small fetch() shim that hands each request to Python. Binary bodies
cross the boundary as latin-1 strings (one char per byte), which QuickJS turns back into a Uint8Array.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from importlib import resources
from typing import Callable, Optional

try:
    import quickjs  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("ai-model-explorer needs the 'quickjs' package: pip install quickjs") from e

_PRELUDE = r"""
var window = undefined;
function TextDecoder() {}
TextDecoder.prototype.decode = function (u8) {
  if (!u8) return "";
  var src = u8.buffer && u8.buffer.__s;
  if (src !== undefined && u8.byteOffset === 0 && u8.length === u8.buffer.byteLength) return __pyutf8(src);
  var CH = 0x8000, parts = [];
  for (var i = 0; i < u8.length; i += CH) parts.push(String.fromCharCode.apply(null, u8.subarray(i, i + CH)));
  var s = parts.join("");
  try { return decodeURIComponent(escape(s)); } catch (e) { return s; }
};
function __bytes(s) { var u = new Uint8Array(s.length); for (var i = 0; i < s.length; i++) u[i] = s.charCodeAt(i) & 255; return u; }
function fetch(url, opts) {
  var range = opts && opts.headers && opts.headers.Range ? opts.headers.Range : "";
  var r = JSON.parse(__pyfetch(url, range));
  var body = r.body;
  return Promise.resolve({
    status: r.status, ok: r.status >= 200 && r.status < 300,
    text: function () { return Promise.resolve(new TextDecoder().decode(__bytes(body))); },
    json: function () { return Promise.resolve(JSON.parse(new TextDecoder().decode(__bytes(body)))); },
    arrayBuffer: function () { var b = __bytes(body).buffer; b.__s = body; return Promise.resolve(b); }
  });
}
var module = undefined;
"""


_NODE_RUNNER = r"""
const A = require(process.argv[1]); let s = ""; process.stdin.setEncoding("utf8");
process.stdin.on("data", c => s += c); process.stdin.on("end", () => { const q = JSON.parse(s);
  const anatomy = A.infer(q.t, {lex: q.lex, qhint: q.h}), N = A.normalize(q.t, q.h).T, logical = {};
  for (const k in N) logical[k] = [N[k].shape, N[k].bytes || 0];
  process.stdout.write(JSON.stringify({anatomy, logical})); });
"""


def _js(name: str) -> str:
    return resources.files("ai_model_explorer").joinpath("js", name).read_text(encoding="utf-8")


def _finite(o):
    """Replace inf / nan (legal in Python json, not in JSON) with None."""
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


class Engine:
    def __init__(self, fetch: Optional[Callable[[str, str], tuple]] = None):
        self.ctx = quickjs.Context()
        self.ctx.set_max_stack_size(8 * 1024 * 1024)
        self._fetch = fetch
        self.ctx.add_callable("__pyfetch", self._pyfetch)
        self.ctx.add_callable("__pyget", self._pyget)
        self.ctx.add_callable("__pyutf8", lambda s: s.encode("latin-1").decode("utf-8", "replace"))
        self._inbox = {}
        self.ctx.eval(_PRELUDE)
        self.ctx.eval(_js("loaders.js"))
        self.ctx.eval(_js("infer.js"))

    # ---- host callbacks
    def _pyfetch(self, url, rng):
        if self._fetch is None:
            return json.dumps({"status": 599, "body": ""})
        status, body = self._fetch(url, rng or None)
        return json.dumps({"status": status, "body": body.decode("latin-1")})

    def _pyget(self, key):
        return self._inbox.pop(key)

    def _put(self, key, obj) -> str:
        self._inbox[key] = json.dumps(_finite(obj))
        return f'JSON.parse(__pyget("{key}"))'

    def _await(self, expr: str):
        """Evaluate an async JS expression to completion and return its JSON value."""
        self.ctx.eval(
            "var __res = undefined, __err = undefined;"
            f"Promise.resolve({expr}).then(function (v) {{ __res = JSON.stringify(v === undefined ? null : v); }},"
            " function (e) { __err = String(e && e.message || e); });")
        while self.ctx.execute_pending_job():
            pass
        err = self.ctx.eval("__err")
        if err is not None:
            raise RuntimeError(err)
        res = self.ctx.eval("__res")
        if res is None:
            raise RuntimeError("JavaScript promise did not settle")
        return json.loads(res)

    # ---- API
    def hf_manifest(self, repo: str, files: list, opt: dict):
        return self._await(f'Loaders.hfManifest({json.dumps(repo)}, {self._put("files", files)}, function () {{}}, {self._put("opt", opt)})')

    def analyze(self, tensors: dict, qhint: Optional[dict] = None, lex: bool = True):
        """Anatomy and logical tensor table in one pass. Uses Node.js when installed (3-7x faster on large models,
        e.g. DeepSeek-V3: ~4 s instead of ~15 s); otherwise the embedded QuickJS. Set AME_JS=quickjs to force it."""
        h = {k: v for k, v in (qhint or {}).items() if v is not None}
        node = None if os.environ.get("AME_JS") == "quickjs" else shutil.which("node")
        if node and len(tensors) > 2000:
            try:
                with resources.as_file(resources.files("ai_model_explorer").joinpath("js", "infer.js")) as path:
                    r = subprocess.run([node, "-e", _NODE_RUNNER, str(path)], input=json.dumps(_finite({"t": tensors, "h": h, "lex": lex})),
                                       capture_output=True, text=True, timeout=600)
                if r.returncode == 0:
                    o = json.loads(r.stdout)
                    return o["anatomy"], o["logical"]
            except (OSError, subprocess.SubprocessError, ValueError):
                pass  # fall back to QuickJS
        return self.infer(tensors, lex=lex, qhint=h), self.normalize(tensors, qhint=h)

    def infer(self, tensors: dict, lex: bool = True, qhint: Optional[dict] = None) -> dict:
        h = json.dumps({k: v for k, v in (qhint or {}).items() if v is not None})
        return self._await(f'ArchInfer.infer({self._put("t", tensors)}, {{lex: {"true" if lex else "false"}, qhint: {h}}})')

    def quant_hint(self, cfg: Optional[dict]) -> dict:
        """bits / group size declared in config.json, used to unpack MLX and compressed-tensors weights."""
        if not cfg:
            return {}
        return self._await(f'ArchInfer.quantHint({self._put("c", cfg)})') or {}

    def normalize(self, tensors: dict, qhint: Optional[dict] = None) -> dict:
        """Logical shape and stored bytes of every tensor after undoing quantized storage."""
        return self._await('(function (N) { var o = {}; for (var k in N.T) o[k] = [N.T[k].shape, N.T[k].bytes || 0]; return o; })'
                           f'(ArchInfer.normalize({self._put("n", tensors)}, {json.dumps({k: v for k, v in (qhint or {}).items() if v is not None})}))')

    def family(self, anatomy: dict, config: dict) -> str:
        return self._await(f'ArchInfer.familyFromConfig({self._put("fa", anatomy)}, {self._put("fc", config)})')

    def fuse(self, anatomy: dict, config: dict):
        slim = {"comps": anatomy.get("comps"), "hp": anatomy.get("hp"), "family": anatomy.get("family")}  # the per-tensor map is not needed
        return self._await(f'ArchInfer.fuseConfig({self._put("a", slim)}, {self._put("c", config)})')
