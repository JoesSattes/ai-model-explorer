"""Read .tflite (LiteRT) files: which layers are stored at which precision, file size per part, and multiply-adds
counted from the graph itself. A port of the website's reader (src/analyzer.js, src/graph.js); the file is memory-mapped
and only the flatbuffer tables are read, so a 1 GB model opens in about a second."""
import json
import mmap
import os
import re
import struct
from typing import Dict, List, Optional

from .model import fmt_b, fmt_n

_HERE = os.path.dirname(os.path.abspath(__file__))


def _enums():
    p = os.path.join(_HERE, "js", "enums.js")
    if not os.path.exists(p):
        p = os.path.join(_HERE, "..", "..", "src", "enums.js")
    s = open(p, encoding="utf-8").read()
    get = lambda k: {int(a): b for a, b in json.loads(re.search(k + r"=(\{.*?\});", s, re.S).group(1)).items()}
    return get("TFL_OPS"), get("TFL_TYPES")


TFL_OPS, TFL_TYPES = _enums()
WEIGHTED = {"FULLY_CONNECTED", "CONV_2D", "EMBEDDING_LOOKUP", "DEPTHWISE_CONV_2D", "BATCH_MATMUL"}
BITS = {"FLOAT32": 32, "FLOAT16": 16, "BFLOAT16": 16, "INT8": 8, "UINT8": 8, "INT16": 16, "INT4": 4, "UINT4": 4,
        "INT2": 2, "INT32": 32, "INT64": 64, "FLOAT8_E4M3FN": 8, "FLOAT8_E5M2": 8}


def _prod(s):
    n = 1
    for x in s:
        n *= x
    return n


class _FB:
    """Minimal flatbuffer reader over a memory map (little-endian), same slots as the website's fbReader."""

    def __init__(self, b):
        self.b = b

    def u32(self, p): return struct.unpack_from("<I", self.b, p)[0]
    def i32(self, p): return struct.unpack_from("<i", self.b, p)[0]
    def u16(self, p): return struct.unpack_from("<H", self.b, p)[0]

    def field(self, t, slot):
        vt = t - self.i32(t)
        return self.u16(vt + slot) if slot < self.u16(vt) else 0

    def ref(self, p): return p + self.u32(p)

    def vec(self, t, slot):
        f = self.field(t, slot)
        if not f:
            return None
        v = self.ref(t + f)
        return self.u32(v), v + 4

    def tables(self, v):
        return [] if not v else [self.ref(v[1] + 4 * i) for i in range(v[0])]

    def ints(self, v):
        return [] if not v else list(struct.unpack_from("<%di" % v[0], self.b, v[1]))

    def str(self, t, slot):
        v = self.vec(t, slot)
        return bytes(self.b[v[1]:v[1] + v[0]]).decode("utf-8", "replace") if v else ""

    def sc(self, t, slot, kind, d=0):
        f = self.field(t, slot)
        if not f:
            return d
        fmt = {"i8": "<b", "i32": "<i", "u32": "<I", "u64": "<Q", "u8": "<B"}[kind]
        return struct.unpack_from(fmt, self.b, t + f)[0]


def parse(path: str) -> dict:
    """The graph of one .tflite file: tensors (name, shape, type, buffer), ops, buffer sizes."""
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        r = _FB(mm)
        if bytes(mm[4:8]) != b"TFL3":
            raise ValueError(f"{path}: not a TFLite file")
        m = r.u32(0)
        codes = []
        for c in r.tables(r.vec(m, 6)):
            k = max(r.sc(c, 4, "i8"), r.sc(c, 10, "i32"))
            codes.append(TFL_OPS.get(k, f"OP_{k}"))
        bufs = []
        for b in r.tables(r.vec(m, 12)):
            d = r.vec(b, 4)
            bufs.append(d[0] if d else r.sc(b, 8, "u64"))
        sg = r.tables(r.vec(m, 8))[0]
        tensors = [{"name": r.str(t, 10), "shape": r.ints(r.vec(t, 4)), "type": TFL_TYPES.get(r.sc(t, 6, "u8"), "?"),
                    "buffer": r.sc(t, 8, "u32")} for t in r.tables(r.vec(sg, 4))]
        ops = []
        for i, o in enumerate(r.tables(r.vec(sg, 10))):
            ins, outs = r.ints(r.vec(o, 6)), r.ints(r.vec(o, 8))
            ops.append({"i": i, "type": codes[r.sc(o, 4, "u32")], "inputs": ins, "outputs": outs,
                        "name": tensors[outs[0]]["name"] if outs else ""})
        return {"path": path, "bytes": os.path.getsize(path), "tensors": tensors, "ops": ops, "buffers": bufs}
    finally:
        mm.close()


def _weight(p, op, producer):
    for idx in op["inputs"]:
        if idx < 0:
            continue
        t = p["tensors"][idx]
        if p["buffers"][t["buffer"]] > 0:
            return t
        pr = producer.get(idx)
        if pr is not None and p["ops"][pr]["type"] == "DEQUANTIZE":
            s = p["tensors"][p["ops"][pr]["inputs"][0]]
            if p["buffers"][s["buffer"]] > 0:
                return s
    return None


def _group(name, typ, comp, seen):
    def last(rx):
        m = re.findall(rx, name)
        return m[-1] if m else None
    if typ in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        return comp + ".conv", "conv" + (last(r"conv(\d)") or "")
    if typ == "EMBEDDING_LOOKUP":
        return "decoder.embed", "embed"
    if "Logits" in name or name.startswith("serving_default_output"):
        return "decoder.logits", "logits"
    kv = last(r"CrossKV_(\d+)")
    if kv is not None:
        return "encoder.cross_kv" + kv, last(r"Linear_(\w+?_proj)") or "proj"
    idx = [int(x) for x in re.findall(r"(?i)(?:EncoderLayer|DecLayer|DecoderLayer|layers?[._]|blocks?[._]|Layer_)(\d+)", name)]
    sub = last(r"Linear_(\w+?_proj|fc1|fc2)") or last(r"(q_proj|k_proj|v_proj|out_proj|o_proj|fc1|fc2|gate_proj|up_proj|down_proj)") or "linear"
    if not idx:
        return comp + ".other", sub
    g = f"{comp}.L{max(idx):02d}"
    if comp == "decoder" and sub in ("q_proj", "out_proj"):
        k = g + sub
        seen[k] = seen.get(k, 0) + 1
        sub = ("self_" if seen[k] == 1 else "cross_") + sub
    return g, sub


def analyze(p: dict, comp: str) -> dict:
    """Weights per op (group, precision, bytes; shared buffers counted once) and multiply-adds per op."""
    producer = {}
    for o in p["ops"]:
        for t in o["outputs"]:
            producer[t] = o["i"]
    rows, seen_buf, seen, macs, opc = [], set(), {}, 0, {}
    for op in p["ops"]:
        opc[op["type"]] = opc.get(op["type"], 0) + 1
        out = p["tensors"][op["outputs"][0]]["shape"] if op["outputs"] else []
        w = _weight(p, op, producer) if op["type"] in WEIGHTED else None
        if op["type"] == "FULLY_CONNECTED" and w:
            macs += _prod(out) * w["shape"][-1]
        elif "CONV" in op["type"] and w:
            macs += _prod(out) * _prod(w["shape"][1:])
        elif op["type"] == "BATCH_MATMUL":
            a = p["tensors"][op["inputs"][0]]["shape"]
            macs += _prod(out) * (a[-1] if a else 1)
        if not w:
            continue
        g, sub = _group(op["name"], op["type"], comp, seen)
        dup = w["buffer"] in seen_buf
        seen_buf.add(w["buffer"])
        rows.append({"group": g, "sub": sub, "op": op["type"], "shape": w["shape"], "dtype": w["type"],
                     "params": 0 if dup else _prod(w["shape"]), "bytes": 0 if dup else p["buffers"][w["buffer"]],
                     "buffer": w["buffer"], "shared": dup})
    emb = [r for r in rows if r["group"] == "decoder.embed"]
    lg = [r for r in rows if r["group"] == "decoder.logits"]
    if emb and lg and emb[0]["buffer"] == lg[0]["buffer"]:
        for r in emb + lg:
            r["group"] = "decoder.tied_embed"
    return {"component": comp, "file": p["path"], "file_bytes": p["bytes"], "rows": rows, "macs": macs, "ops": opc}


def _mac(n):
    return f"{n / 1e9:.1f} G" if n >= 1e9 else f"{n / 1e6:.1f} M"


def _comp_of(path, n):
    b = os.path.basename(path).lower()
    return "encoder" if "enc" in b else "decoder" if "dec" in b else ("model" if n == 1 else "encoder")


class TfliteModel:
    """One or more .tflite files of a model (an encoder and a decoder file for Whisper exports)."""

    def __init__(self, files: List[str]):
        self.files = files
        self.parts = [analyze(parse(f), _comp_of(f, len(files))) for f in files]
        self.parts.sort(key=lambda p: {"encoder": 0, "model": 1, "decoder": 2}.get(p["component"], 1))

    # --------------------------------------------------------------- numbers
    @property
    def name(self) -> str:
        d = os.path.dirname(os.path.abspath(self.files[0]))
        return os.path.basename(d) or "model"

    @property
    def file_bytes(self) -> int:
        return sum(p["file_bytes"] for p in self.parts)

    @property
    def params(self) -> int:
        return sum(r["params"] for p in self.parts for r in p["rows"])

    @property
    def weight_bytes(self) -> int:
        return sum(r["bytes"] for p in self.parts for r in p["rows"])

    def components(self) -> List[dict]:
        out = []
        for p in self.parts:
            pr = sum(r["params"] for r in p["rows"])
            out.append({"component": p["component"], "file": os.path.basename(p["file"]), "file_bytes": p["file_bytes"],
                        "params": pr, "weight_bytes": sum(r["bytes"] for r in p["rows"]), "macs": p["macs"],
                        "ops": sum(p["ops"].values()), "precision": self._mix(p["rows"])})
        tot = self.file_bytes or 1
        for c in out:
            c["share"] = c["file_bytes"] / tot
        return out

    @staticmethod
    def _mix(rows) -> Dict[str, int]:
        m = {}
        for r in rows:
            if r["params"]:
                m[r["dtype"]] = m.get(r["dtype"], 0) + r["params"]
        return dict(sorted(m.items(), key=lambda kv: -kv[1]))

    def precision(self) -> Dict[str, int]:
        """Parameters per stored type over all files, e.g. {'INT4': ..., 'INT8': ..., 'FLOAT32': ...}."""
        return self._mix([r for p in self.parts for r in p["rows"]])

    def layers(self) -> List[dict]:
        """One row per group (conv stem, each layer, cross-attention K/V, embedding): params, bytes and stored types."""
        out = []
        for p in self.parts:
            gs = {}
            for r in p["rows"]:
                g = gs.setdefault(r["group"], {"group": r["group"], "params": 0, "bytes": 0, "types": {}, "subs": []})
                g["params"] += r["params"]
                g["bytes"] += r["bytes"]
                if r["params"]:
                    g["types"][r["dtype"]] = g["types"].get(r["dtype"], 0) + r["params"]
                g["subs"].append(f"{r['sub']}:{r['dtype']}" + (" (shared)" if r["shared"] else ""))
            out.extend(gs.values())
        for g in out:
            g["dtype"] = "/".join(g["types"]) or "-"
        return out

    def macs(self) -> Dict[str, int]:
        """Multiply-adds counted from the graph's own op shapes (fully connected, convolution, batch matmul), per file:
        the encoder file runs once per input, a decoder file once per call (usually one token). The LiteRT converter
        prints its own estimate when it writes the file ("Estimated count of arithmetic ops"); that estimate counts the
        attention batch-matmuls only in part, so it reads lower (Whisper medium encoder: 499 G there, 582 G here,
        of which 471 G are the fully connected and convolution layers and 111 G the attention scores)."""
        return {p["component"]: p["macs"] for p in self.parts}

    # --------------------------------------------------------------- text
    def summary(self) -> str:
        lines = [f"{self.name}  ({len(self.files)} .tflite file(s), {fmt_b(self.file_bytes)})",
                 f"  {fmt_n(self.params)} weights · stored as " + ", ".join(f"{k} {v / max(self.params, 1):.0%}" for k, v in self.precision().items())]
        for c in self.components():
            lines.append(f"  {c['component']}: {c['file']} {fmt_b(c['file_bytes'])} ({c['share']:.0%}) · {fmt_n(c['params'])} weights · "
                         f"{_mac(c['macs'])} multiply-adds · {c['ops']} ops · "
                         + ", ".join(f"{k} {v / max(c['params'], 1):.0%}" for k, v in c["precision"].items()))
        return "\n".join(lines)

    def insights(self) -> List[str]:
        out = []
        comps = self.components()
        out.append(f"{fmt_b(self.file_bytes)} on disk for {fmt_n(self.params)} weights; stored as "
                   + ", ".join(f"{k} {v / max(self.params, 1):.0%}" for k, v in self.precision().items()) + ".")
        if len(comps) > 1:
            out.append(" / ".join(f"{c['component']} {c['share']:.0%}" for c in comps) + " of the bytes.")
            enc = next((c for c in comps if c["component"] == "encoder"), None)
            dec = next((c for c in comps if c["component"] == "decoder"), None)
            if enc and dec and dec["macs"]:
                out.append(f"Compute from the graph: encoder {enc['macs'] / 1e9:.0f} G multiply-adds per input, decoder "
                           f"{dec['macs'] / 1e6:.1f} M per call. The decoder is {dec['share']:.0%} of the bytes but runs once "
                           f"per token, so time both before choosing what to squeeze.")
        mixed = [g for g in self.layers() if len(g["types"]) > 1]
        hi = [g for g in self.layers() if g["types"] and max(BITS.get(t, 32) for t in g["types"]) > min(BITS.get(t, 32) for p in [self.precision()] for t in p)]
        if mixed:
            out.append(f"{len(mixed)} group(s) mix precisions inside: " + ", ".join(g["group"] for g in mixed[:6]) + ".")
        if len(self.precision()) > 1 and hi:
            out.append("Kept at higher precision: " + ", ".join(f"{g['group']} ({g['dtype']})" for g in hi[:8])
                       + (" …" if len(hi) > 8 else "") + ".")
        tied = [g for g in self.layers() if g["group"] == "decoder.tied_embed"]
        if tied:
            out.append("The embedding and the logits share one buffer (tied), so it is stored once.")
        return out

    def plot(self, path: Optional[str] = None, theme: str = "light"):
        """Bars per group (layers in order), coloured by stored type; returns the matplotlib Figure."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        L = self.layers()
        col = {"FLOAT32": "#9aa3c7", "FLOAT16": "#7b86c9", "BFLOAT16": "#7b86c9", "INT8": "#5b6ef0", "UINT8": "#5b6ef0",
               "INT4": "#ff6bb5", "INT16": "#8f62d6"}
        dark = theme == "dark"
        fig, ax = plt.subplots(figsize=(12, 4.2), dpi=200)
        bg, ink = ("#0b1020", "#eef0ff") if dark else ("white", "#1b1f2e")
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        x = 0
        ticks = []
        for g in L:
            bottom = 0
            for t, n in sorted(g["types"].items(), key=lambda kv: -kv[1]):
                share = n / max(g["params"], 1)
                ax.bar(x, g["bytes"] / 1e6 * share, bottom=bottom, color=col.get(t, "#cccccc"), width=0.85)
                bottom += g["bytes"] / 1e6 * share
            ticks.append(g["group"].replace("encoder.", "E.").replace("decoder.", "D."))
            x += 1
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=90, fontsize=6, color=ink)
        ax.set_ylabel("MB stored", color=ink)
        ax.tick_params(colors=ink)
        for s in ax.spines.values():
            s.set_visible(False)
        used = {t for g in L for t in g["types"]}
        ax.legend(handles=[Patch(color=col.get(t, "#cccccc"), label=t) for t in used], frameon=False, labelcolor=ink, fontsize=8)
        ax.set_title(f"{self.name} · {fmt_b(self.file_bytes)} · stored type per group", color=ink, fontsize=11, loc="left")
        fig.tight_layout()
        if path:
            fig.savefig(path, facecolor=bg)
        return fig

    def to_dict(self) -> dict:
        return {"name": self.name, "files": self.files, "file_bytes": self.file_bytes, "params": self.params,
                "precision": self.precision(), "components": self.components(), "layers": self.layers(), "macs": self.macs()}

    def to_json(self, path: Optional[str] = None, indent: int = 1) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    def __repr__(self):
        return f"<TfliteModel {self.name}: {len(self.files)} file(s), {fmt_b(self.file_bytes)}>"


def tflite_files(model) -> Optional[List[str]]:
    """The .tflite files `model` points to (a file, a folder holding them, or a list), else None."""
    if isinstance(model, (list, tuple)) and model and all(str(m).endswith(".tflite") for m in model):
        return [str(m) for m in model]
    s = str(model)
    if s.endswith(".tflite") and os.path.isfile(s):
        return [s]
    if os.path.isdir(s):
        fs = sorted(os.path.join(s, f) for f in os.listdir(s) if f.endswith(".tflite"))
        has_st = any(f.endswith(".safetensors") for f in os.listdir(s))
        if fs and not has_st:
            return fs
    return None
