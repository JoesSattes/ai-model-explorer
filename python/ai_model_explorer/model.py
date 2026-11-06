"""Model: one object for analysing an inspected model: breakdowns, insights, memory estimates, plots and reports."""
from __future__ import annotations

import json
from typing import Iterable, List, Optional

KIND_LABEL = {"attn": "attention", "cross": "cross-attention", "ffn": "feed-forward", "moe": "mixture of experts",
              "ssm": "state-space / linear attention", "conv": "convolution", "rnn": "recurrent", "mod": "adaLN modulation",
              "embed": "embeddings", "head": "output head", "norm": "norms", "adapter": "adapter / projector", "other": "other"}


def fmt_n(n: Optional[float]) -> str:
    if n is None:
        return "?"
    for u, k in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= k:
            return f"{n / k:.2f}{u}".replace(".00", "")
    return f"{int(n)}"


def fmt_b(n: Optional[float]) -> str:
    if n is None:
        return "?"
    for u, k in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(n) >= k:
            return f"{n / k:.1f} {u}"
    return f"{int(n)} B"


class Model:
    """The anatomy of one model. Create it with `explore()`."""

    def __init__(self, data: dict):
        self.data = data

    # ------------------------------------------------------------------ basics
    @property
    def name(self) -> str:
        s = self.data["source"]
        return s.get("repo") or s.get("path") or "model"

    @property
    def family(self) -> Optional[str]:
        return self.data.get("family")

    @property
    def params(self) -> int:
        return (self.data.get("totals") or {}).get("params") or 0

    @property
    def stored_bytes(self) -> int:
        return (self.data.get("totals") or {}).get("bytes") or 0

    @property
    def hp(self) -> dict:
        return self.data.get("hyperparameters_with_config") or self.data.get("hyperparameters") or {}

    @property
    def components(self) -> list:
        return self.data.get("components") or []

    @property
    def config(self) -> Optional[dict]:
        return self.data.get("config")

    def __repr__(self) -> str:
        return f"<Model {self.name}: {self.family}, {fmt_n(self.params)} parameters>"

    def to_dict(self) -> dict:
        return self.data

    def to_json(self, path: Optional[str] = None, indent: int = 1) -> str:
        s = json.dumps(self.data, indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    def summary(self) -> str:
        from .api import summary
        out = summary(self.data)
        if self.duplicate_head_params:
            out = out.replace(f"{fmt_n(self.params)} parameters", f"{fmt_n(self.params)} parameters stored "
                              f"({fmt_n(self.distinct_params)} distinct; the tied head is stored twice)", 1)
        repo = (self.data.get("source") or {}).get("repo")
        if repo:
            out += f"\n  open in the explorer: https://joessattes.github.io/ai-model-explorer/en/?ref=pip#hf={repo}"
        return out

    # ------------------------------------------------------------------ analysis
    def breakdown(self, by: str = "kind") -> List[dict]:
        """Parameters and stored bytes per part type ("kind"), component ("component") or layer stack ("stack")."""
        key = {"kind": "by_kind", "component": "by_component", "stack": "by_stack"}[by]
        rows = []
        tot = self.params or 1
        for k, v in (self.data.get("breakdown") or {}).get(key, {}).items():
            rows.append({"name": KIND_LABEL.get(k, k) if by == "kind" else k, "key": k, "params": v["params"],
                         "share": v["params"] / tot, "bytes": v["bytes"], "tensors": v["tensors"]})
        return rows

    @property
    def mtp_params(self) -> int:
        return ((self.data.get("breakdown") or {}).get("mtp") or {}).get("params", 0)

    @property
    def active_params(self) -> Optional[int]:
        """Parameters used per token. Equals the total for dense models; for mixtures of experts only the routed
        experts a token visits count. Multi-token-prediction layers are left out (they serve speculative decoding).
        Embeddings are included; some model cards leave the input embedding out (gpt-oss), DeepSeek and Kimi include it."""
        hp = self.hp
        mtp = (self.data.get("breakdown") or {}).get("mtp") or {}
        main = self.params - mtp.get("params", 0)
        e, k, ex = hp.get("experts"), hp.get("experts_per_token"), hp.get("expert_params")
        if not e:
            return main
        if not (k and ex):
            return None
        ex_main = ex - mtp.get("expert_params", 0)
        return int(main - ex_main * (1 - k / e))

    def kv_cache_bytes_per_token(self, bits: int = 16) -> Optional[int]:
        """Bytes of KV cache one token adds, summed over all attention layers. Taken from the weights: the width of the
        key/value projections (exact for grouped-query attention), the compressed vector for latent attention (MLA);
        state-space layers and multi-token-prediction layers keep no per-token cache here."""
        base = (self.data.get("hyperparameters") or {}).get("kv_bytes_per_token_bf16")
        if base:
            return int(base * bits / 16)
        hp = self.hp
        layers = (hp.get("attn_layers") or hp.get("layers") or 0) - (hp.get("mtp_layers") or 0)
        kv, hd = hp.get("kv_heads"), hp.get("head_dim")
        if layers <= 0 or not (kv and hd):
            return None
        return int(2 * layers * kv * hd * bits / 8)

    # ------------------------------------------------------------------ encoder-decoder helpers
    def _cfg(self, *keys, default=None):
        c = self.config or {}
        for k in keys:
            if c.get(k):
                return c[k]
        return default

    @property
    def duplicate_head_params(self) -> int:
        """Parameters of an output head that config.json ties to the embedding but the file stores again (Whisper's
        proj_out). A converter that shares the table (LiteRT, ONNX) keeps one copy."""
        hp, cfg = self.hp, self.config or {}
        head = next((r for r in self.breakdown("kind") if r["key"] == "head"), None)
        if not head or hp.get("tie") or not cfg or cfg.get("tie_word_embeddings", True) is False:
            return 0
        return head["params"] if hp.get("vocab") and hp.get("d") and head["params"] == hp["vocab"] * hp["d"] else 0

    @property
    def distinct_params(self) -> int:
        """Parameters counted once: the stored total minus a duplicated tied head."""
        return self.params - self.duplicate_head_params

    def _encdec(self):
        enc = next((c for c in self.components if c.get("role") == "encoder"), None)
        dec = next((c for c in self.components if c.get("role") == "backbone"), None)
        return (enc, dec) if self.family == "encoder-decoder" and enc and dec else (None, None)

    def encoder_input_length(self) -> Optional[int]:
        """Positions the encoder sees per input (Whisper: 1500 frames for 30 s of audio)."""
        enc, _ = self._encdec()
        if not enc:
            return None
        n = self._cfg("max_source_positions")
        if n:
            return int(n)
        e, d = (enc.get("kinds") or {}).get("embed"), enc.get("d")
        return int(e // d) if e and d else None

    def split(self, converter_layout: bool = True) -> Optional[dict]:
        """Share of the distinct parameters in the encoder and the decoder. With converter_layout, the cross-attention
        K/V projections count with the encoder, because exporters (LiteRT) run them once per input on the encoder side."""
        enc, dec = self._encdec()
        if not enc:
            return None
        tot = self.distinct_params or 1
        e = enc.get("params") or 0
        kv = 0
        if converter_layout:
            d, L = dec.get("d") or 0, dec.get("layers") or 0
            kv = L * (2 * d * d + d)   # K and V weights of the cross-attention, V bias (Whisper's K has none)
        e2 = e + kv
        return {"encoder": e2 / tot, "decoder": 1 - e2 / tot, "encoder_params": e2, "decoder_params": tot - e2,
                "cross_kv_params_moved": kv}

    def cross_kv_bytes(self, bits: int = 16) -> Optional[int]:
        """Cross-attention K/V the decoder reads, computed once per input: decoder layers x 2 x encoder length x width."""
        enc, dec = self._encdec()
        S = self.encoder_input_length()
        if not (enc and S):
            return None
        return int((dec.get("layers") or 0) * 2 * S * (dec.get("d") or 0) * bits / 8)

    def macs(self, context: int = 1) -> Optional[dict]:
        """Rough multiply-accumulates from the shapes: encoder per input and decoder per generated token.
        encoder: L_e (4 S d^2 + 2 S^2 d + 2 S d F) + conv params x S, plus the decoder's cross K/V (L_d 2 S d^2) once per input;
        decoder: L_d (4 d^2 + 2 n d + 2 d^2 + 2 S d + 2 d F) + V d, with n = context so far.
        Attention scores are included; activations, softmax and norms are not, so a profiler will read somewhat differently."""
        enc, dec = self._encdec()
        S = self.encoder_input_length()
        if not (enc and S):
            return None
        de, Le = enc.get("d"), enc.get("layers") or 0
        dd, Ld = dec.get("d"), dec.get("layers") or 0
        Fe = self._cfg("encoder_ffn_dim", default=4 * de)
        Fd = self._cfg("decoder_ffn_dim", default=4 * dd)
        V = self.hp.get("vocab") or 0
        conv = (enc.get("kinds") or {}).get("conv", 0) * S
        e = Le * (4 * S * de * de + 2 * S * S * de + 2 * S * de * Fe) + conv + Ld * 2 * S * dd * de
        t = Ld * (4 * dd * dd + 2 * context * dd + 2 * dd * dd + 2 * S * dd + 2 * dd * Fd) + V * dd
        return {"encoder_per_input": int(e), "decoder_per_token": int(t), "encoder_length": S, "context": context}

    def size(self, bits: float = 8, overrides: Optional[dict] = None) -> int:
        """File size at a mixed precision, in bytes. `bits` applies to every weight matrix; `overrides` maps a
        component id ("model.decoder") or a part type ("embed", "head", "conv", "cross", "attn", "ffn") to its own bits
        (a part type wins over a component). Norm vectors stay 32-bit and a duplicated tied head is counted once.
        Per-channel scales and file overhead are not included."""
        ov = overrides or {}
        tot = 0.0
        for c in self.components:
            cb = ov.get(c.get("id"), bits)
            for k, n in (c.get("kinds") or {}).items():
                if k == "head":
                    n -= self.duplicate_head_params
                b = 32 if k == "norm" else ov.get(k, cb)
                tot += n * b / 8
        return int(tot)

    def memory(self, context: int = 8192, batch: int = 1, weight_bits: Optional[float] = None, kv_bits: int = 16) -> dict:
        """Rough memory to run the model: weights at a chosen precision (default: as stored) plus the KV cache for
        `batch` sequences of `context` tokens. Activations and runtime overhead are not included."""
        w = self.stored_bytes if weight_bits is None else self.params * weight_bits / 8
        kvt = self.kv_cache_bytes_per_token(kv_bits)
        kv = kvt * context * batch if kvt is not None else None
        return {"weights_bytes": int(w), "kv_cache_bytes": kv, "total_bytes": int(w + (kv or 0)), "context": context,
                "batch": batch, "weight_bits": weight_bits if weight_bits is not None else "as stored", "kv_bits": kv_bits}

    def insights(self) -> List[str]:
        """Plain-language findings, the most useful first."""
        out, hp, bd = [], self.hp, self.breakdown("kind")
        q = self.data.get("quantization") or {}
        store = f"stored as {', '.join(q)}" if q else "stored unquantized"
        out.append(f"{fmt_n(self.params)} parameters, {fmt_b(self.stored_bytes)} of weights {store}. "
                   f"At 16-bit that would be {fmt_b(self.distinct_params * 2)}, at 8-bit {fmt_b(self.distinct_params)}, at 4-bit {fmt_b(self.distinct_params / 2)}"
                   + (" (head counted once)." if self.duplicate_head_params else "."))
        if self.mtp_params:
            out.append(f"Includes a multi-token-prediction layer of {fmt_n(self.mtp_params)} parameters, used for speculative "
                       f"decoding; the main model is {fmt_n(self.params - self.mtp_params)}.")
        act = self.active_params
        if hp.get("experts"):
            k = hp.get("experts_per_token")
            if act:
                out.append(f"Mixture of experts: {hp['experts']} experts, {k} used per token, so about {fmt_n(act)} parameters "
                           f"({act / max(self.params, 1):.1%}) do the work for each token, while all {fmt_n(self.params)} must sit in memory.")
            else:
                out.append(f"Mixture of experts with {hp['experts']} experts per layer; the number used per token is not in the manifest (add config.json).")
        if bd:
            top = ", ".join(f"{r['name']} {r['share']:.0%}" for r in bd[:3] if r["share"] >= 0.01)
            out.append(f"Where the parameters sit: {top}.")
        emb = next((r for r in bd if r["key"] == "embed"), None)
        head = next((r for r in bd if r["key"] == "head"), None)
        if emb and emb["share"] >= 0.05:
            if hp.get("tie"):
                out.append(f"Embeddings take {emb['share']:.0%} of the model ({fmt_n(emb['params'])}), and the output head "
                           "reuses the same table.")
            else:
                vocab = emb["share"] + (head["share"] if head else 0)
                out.append(f"Embeddings take {emb['share']:.0%} of the model ({fmt_n(emb['params'])}) and the output head is a "
                           f"separate matrix, so the vocabulary costs {vocab:.0%} in total, a large fixed cost on small devices.")
                cfg = self.config or {}
                if head and cfg and cfg.get("tie_word_embeddings", True) and hp.get("vocab") and hp.get("d") \
                        and head["params"] == hp["vocab"] * hp["d"]:
                    out.append(f"config.json ties the output head to the embedding, yet this file stores the head as its own copy: "
                               f"{fmt_n(head['params'])} are stored twice, so the model has {fmt_n(self.params - head['params'])} "
                               f"distinct parameters. A converter that shares the table (LiteRT, ONNX) counts the smaller number.")
        kvt = self.kv_cache_bytes_per_token()
        if kvt:
            lat = " (compressed latent attention)" if (self.config or {}).get("kv_lora_rank") else ""
            gqa = ""
            if hp.get("heads") and hp.get("kv_heads") and hp["heads"] > hp["kv_heads"] and not lat:
                gqa = f" Grouped-query attention ({hp['heads']} query heads share {hp['kv_heads']} KV heads) makes the cache {hp['heads'] // hp['kv_heads']}× smaller than full multi-head attention."
            tmax = self._cfg("max_target_positions") if self.family == "encoder-decoder" else None
            if tmax:
                out.append(f"Decoder self-attention KV cache: {fmt_b(kvt)} per token at 16-bit, {fmt_b(kvt * tmax)} at the "
                           f"{tmax}-token limit.{gqa}")
            else:
                out.append(f"KV cache{lat}: {fmt_b(kvt)} per token at 16-bit, so {fmt_b(kvt * 32768)} for one 32k-token sequence "
                           f"and {fmt_b(kvt * 131072)} at 128k.{gqa}")
        if hp.get("ssm_layers"):
            out.append(f"Hybrid stack: {hp['ssm_layers']} state-space / linear-attention layers keep a fixed-size state, and only "
                       f"{hp.get('attn_layers', 0)} attention layers grow a KV cache with context length.")
        comps = [c for c in self.components if c.get("role") not in ("backbone",)]
        if comps:
            parts = "; ".join(f"{c.get('role')} {c.get('id')} ({fmt_n(c.get('params'))})" for c in comps[:4])
            if self.family == "encoder-decoder" and len(comps) == 1:
                out.append(f"Encoder {comps[0].get('id')} ({fmt_n(comps[0].get('params'))}) feeds the decoder through cross-attention.")
            else:
                out.append(f"Besides the {'decoder' if self.family == 'encoder-decoder' else 'backbone'}: {parts}.")
        sp, ckv, mc = self.split(), self.cross_kv_bytes(), self.macs()
        if sp:
            raw = self.split(converter_layout=False)
            out.append(f"Encoder {sp['encoder']:.0%} / decoder {sp['decoder']:.0%} of the distinct weights when the cross-attention "
                       f"K/V projections run with the encoder, as an exported model does ({raw['encoder']:.0%} / {raw['decoder']:.0%} by module).")
        if ckv:
            out.append(f"Cross-attention K/V: computed once per input ({self.encoder_input_length()} encoder positions), "
                       f"{fmt_b(ckv)} at 16-bit, then read by every decoder step.")
        if mc:
            out.append(f"Compute: about {mc['encoder_per_input'] / 1e9:.0f} G multiply-adds for the encoder per input and "
                       f"{mc['decoder_per_token'] / 1e6:.0f} M per decoder token. The decoder holds "
                       f"{(sp or {}).get('decoder', 0):.0%} of the weights but runs once per token; time the two before choosing what to squeeze.")
        chk = self.data.get("config_check") or {}
        bad = [k for k, v in chk.items() if isinstance(v, dict) and v.get("checked") and v.get("consistent") is False]
        if bad:
            out.append("config.json disagrees with the weights for " + ", ".join(bad) + "; trust the weights.")
        elif chk:
            out.append("config.json agrees with the weights.")
        elif not self.data.get("has_config"):
            out.append("No config.json: head counts are estimates from shapes.")
        for e in self.data.get("evidence") or []:
            out.append("Note: " + e)
        repo = (self.data.get("source") or {}).get("repo")
        if repo:
            out.append(f"Open it in the explorer: https://joessattes.github.io/ai-model-explorer/en/?ref=pip#hf={repo}")
        return out

    # ------------------------------------------------------------------ output
    def plot(self, path: Optional[str] = None, theme: str = "light", **kw):
        """Draw the anatomy (components, layer stacks coloured by block type, parameter share). Saves to `path`
        (.png, .svg or .pdf) when given and returns the matplotlib Figure."""
        from .viz import plot_model
        return plot_model(self, path=path, theme=theme, **kw)

    def report(self, path: str, theme: str = "light") -> str:
        """Write a self-contained HTML report: diagram, summary, insights, breakdowns and memory estimates."""
        from .report import write_report
        return write_report(self, path, theme=theme)


def explore(model, token: Optional[str] = None, config: Optional[dict] = None, include_tensors: bool = False,
            cache: bool = True) -> Model:
    """Inspect a Hugging Face repo or local .safetensors and return a Model you can analyse, plot and report.
    Headers are cached per repo commit (~/.cache/ai-model-explorer), so a second look is instant."""
    from .tflite import TfliteModel, tflite_files
    tf = tflite_files(model)
    if tf:
        return TfliteModel(tf)
    from .api import inspect
    return Model(inspect(model, token=token, config=config, include_tensors=include_tensors, cache=cache))


def compare(models: Iterable, token: Optional[str] = None) -> List[dict]:
    """One row per model with the numbers people compare most. Accepts repo ids, paths or Model objects."""
    rows = []
    for m in models:
        mm = m if isinstance(m, Model) else explore(m, token=token)
        hp = mm.hp
        kvt = mm.kv_cache_bytes_per_token()
        rows.append({"model": mm.name, "family": mm.family, "params": mm.params, "active_params": mm.active_params,
                     "stored_bytes": mm.stored_bytes, "storage": ", ".join(mm.data.get("quantization") or {}) or "unquantized",
                     "layers": hp.get("layers"), "width": hp.get("d"), "heads": hp.get("heads"), "kv_heads": hp.get("kv_heads"),
                     "experts": hp.get("experts"), "experts_per_token": hp.get("experts_per_token"), "vocab": hp.get("vocab"),
                     "kv_bytes_per_token": kvt})
    return rows


def compare_table(rows: List[dict]) -> str:
    """A Markdown table of compare() rows."""
    cols = [("model", "Model", str), ("family", "Family", str), ("params", "Params", fmt_n), ("active_params", "Active", fmt_n),
            ("stored_bytes", "Stored", fmt_b), ("storage", "Storage", str), ("layers", "Layers", str), ("width", "Width", str),
            ("heads", "Heads", str), ("kv_heads", "KV heads", str), ("experts", "Experts", str), ("vocab", "Vocab", str),
            ("kv_bytes_per_token", "KV/token (16-bit)", fmt_b)]
    head = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = "\n".join("| " + " | ".join("–" if r.get(k) is None else f(r[k]) for k, _, f in cols) + " |" for r in rows)
    return head + body
