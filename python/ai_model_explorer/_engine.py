"""Run the explorer's own JavaScript (loaders.js, infer.js) inside QuickJS, so Python gets exactly the website's result.

Python does the network I/O: the JS side calls a small fetch() shim that hands each request to Python. Binary bodies
cross the boundary as latin-1 strings (one char per byte), which QuickJS turns back into a Uint8Array.
"""
from __future__ import annotations

import json
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


def _js(name: str) -> str:
    return resources.files("ai_model_explorer").joinpath("js", name).read_text(encoding="utf-8")


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
        self._inbox[key] = json.dumps(obj)
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
        return self._await(f'ArchInfer.fuseConfig({self._put("a", anatomy)}, {self._put("c", config)})')
