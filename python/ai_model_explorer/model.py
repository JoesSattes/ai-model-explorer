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
            out += f"\n  open in the explorer: https://joessattes.github.io/ai-model-explorer/?ref=pip#hf={repo}"
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
        kvp = (self.data.get("plan") or {}).get("kv")
        if kvp and (kvp.get("shared") or kvp.get("sliding")):
            return int(kvp["per_token"] * bits / 16)   # only layers with their own, unbounded cache grow per token
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
        kvp = (self.data.get("plan") or {}).get("kv")
        if kv is not None and kvp and kvp.get("sliding") and kvp.get("window"):
            kv += int(kvp["sliding_per_token"] * kv_bits / 16) * min(context, kvp["window"]) * batch
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
        plan = self.data.get("plan") or {}
        ple = [t for t in (plan.get("vocab") or {}).get("tables", []) if t["what"].startswith("per-layer")]
        if ple:
            pp = sum(t["params"] for t in ple)
            out.append(f"Per-layer embeddings (PLE) hold {fmt_n(pp)} ({pp / max(self.params, 1):.0%} of the model): a lookup of one "
                       f"{ple[0]['width']}-value row per token, so they can sit on flash or be memory-mapped, and they shrink with the vocabulary.")
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
                kvp = plan.get("kv") or {}
                extra = (f" ({kvp['cached']} of {kvp['layers']} layers keep their own cache, {kvp['shared']} share one; "
                         f"{kvp['sliding']} sliding-window layers add at most {fmt_b(kvp['sliding_cap'])})") if kvp.get("shared") or kvp.get("sliding") else ""
                out.append(f"KV cache{lat}: {fmt_b(kvt)} per token at 16-bit{extra}, so {fmt_b(kvt * 32768)} for one 32k-token sequence "
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
        lev = [x for x in plan.get("levers") or [] if x.get("share", 0) >= 0.02]
        if lev:
            out.append("Where to cut (see plan_text()): " + "; ".join(f"{x['title'].lower()} ({x['share']:.0%})" for x in lev[:3]) + ".")
        repo = (self.data.get("source") or {}).get("repo")
        if repo:
            out.append(f"Open it in the explorer: https://joessattes.github.io/ai-model-explorer/?ref=pip#hf={repo}")
        return out

    # ------------------------------------------------------------------ where to cut
    def _tok(self, tokenizer=None, token: Optional[str] = None):
        from . import _tok
        return _tok.load(tokenizer, self.data.get("source") or {}, token)

    def _tokenizer(self, tokenizer=None, token: Optional[str] = None):
        """Pieces of the tokenizer in id order and whether they are byte-level (GPT-2 style) strings.
        tokenizer: a path (tokenizer.json, tokenizer.model, vocab.json, *.tiktoken, tekken.json, .gguf, or a folder),
        a Hub repo id, or a tokenizer.json dict. None reads the model's own folder or repo."""
        t = self._tok(tokenizer, token)
        return t.pieces, t.byte_level

    def vocab_scripts(self, languages: Optional[List[str]] = None, tokenizer=None, token: Optional[str] = None) -> dict:
        """Tokenizer pieces per writing system (Latin, Thai, Han, ...), and how many a set of languages needs.
        Specials, byte-fallback pieces, digits, punctuation and whitespace are always kept."""
        from ._engine import Engine
        pieces, bl = self._tokenizer(tokenizer, token)
        r = Engine().vocab_scripts(pieces, bl, languages)
        r["counts"] = dict(sorted(r["counts"].items(), key=lambda kv: -kv[1]))
        r["byte_level"] = bl
        return r

    @staticmethod
    def _read_corpus(corpus) -> List[str]:
        import os
        if corpus is None:
            return []
        items = [corpus] if isinstance(corpus, str) else list(corpus)
        out = []
        for c in items:
            if isinstance(c, str) and os.path.exists(c):
                with open(c, encoding="utf-8") as f:
                    out += [l.strip() for l in f if l.strip()]
            elif isinstance(c, str):
                out.append(c)
        return out

    def _keep_mask(self, t, languages: List[str], corpus=None, min_count: int = 1, target_resplit: float = 0.01) -> tuple:
        """Script rule, refined to the language level by your own text when corpus is given.
        Script rule: keep pieces whose letters are all in the languages' scripts (+ specials, bytes, digits, punctuation).
        Corpus rule: among those, keep only pieces your text actually uses (>= min_count times), plus every single-character
        piece so any unseen word can still be spelled. A held-out 20% of the text estimates how many tokens would be
        re-split into smaller pieces (input changes); text never fails to encode either way."""
        import collections
        from ._engine import Engine
        eng = Engine()
        mask = eng.vocab_mask(t.pieces, t.byte_level, languages)
        always = eng.vocab_mask(t.pieces, t.byte_level, [])
        keep = [c == "1" for c in mask]
        keep, added0 = self._close_merges(t, keep)   # a few kept pieces are built through removed ones; keep those parents
        info = {"rule": "script", "kept": sum(keep), "merge_closure_added": added0}
        lines = self._read_corpus(corpus)
        if not lines:
            return keep, info
        enc = t.encoder()
        if enc is None:
            raise RuntimeError(f"corpus refinement needs a library that can encode {t.kind} tokenizers "
                               "(pip install tokenizers / sentencepiece / tiktoken / llama-cpp-python)")
        def single(i):
            q = t.pieces[i].replace("\u2581", "").replace("\u0120", "")
            return len(q) <= 1
        def build(sents):
            cnt = collections.Counter(i for s_ in sents for i in enc(s_))
            return [keep[i] and (always[i] == "1" or single(i) or cnt.get(i, 0) >= min_count) for i in range(len(keep))]
        cut = max(1, len(lines) // 5)
        held, train = lines[:cut], lines[cut:] or lines
        k_tr, _ = self._close_merges(t, build(train))
        ho = [i for s_ in held for i in enc(s_)]
        resplit = sum(1 for i in ho if i < len(k_tr) and not k_tr[i]) / max(len(ho), 1)
        enc_all = [enc(s_) for s_ in lines]
        k = build(lines)
        k, added = self._close_merges(t, k)
        cert = self._trim_certificate(enc_all, keep, always, single, min_count, target_resplit)
        info = {"rule": "script + your text", "kept": sum(k), "kept_script_only": sum(keep), "corpus_lines": len(lines),
                "min_count": min_count, "heldout_tokens": len(ho), "heldout_resplit_rate": resplit,
                "merge_closure_added": added, **cert}
        return k, info

    @staticmethod
    def _close_merges(t, keep):
        """Add the BPE merge parents of every kept piece (recursively). Without them a kept piece cannot be produced:
        its row is wasted and its words are split further than the removed pieces alone would cause (2-3x the
        re-split rate in our measurements). Returns (closed keep, number of pieces added)."""
        par = t.merge_parents() if hasattr(t, "merge_parents") else {}
        if not par:
            return keep, 0
        k = list(keep)
        st = [i for i, x in enumerate(k) if x]
        added = 0
        while st:
            i = st.pop()
            for j in par.get(i, ()):
                if j < len(k) and not k[j]:
                    k[j] = True; added += 1; st.append(j)
        return k, added

    @staticmethod
    def _gt_unseen(counts, n_tokens, cand, min_count):
        """Good-Turing estimate of the share of future tokens that fall in pieces the corpus rule removes
        (pieces seen fewer than min_count times): sum over c < min_count of (c+1) * N_{c+1} / N."""
        import collections
        nc = collections.Counter(c for i, c in counts.items() if cand(i))
        return sum((c + 1) * nc.get(c + 1, 0) for c in range(min_count)) / max(n_tokens, 1)

    def _trim_certificate(self, enc_lines, keep, always, single, min_count, target):
        """Predicts, from your text alone, (a) the share of tokens in new text of the same kind that the trimmed tokenizer
        will re-split, and (b) how many lines of text you need to bring that under `target`. Validated across 20 languages
        x 5 tokenizer families: predicted/actual median 0.95 (r = 0.995 in log scale); the line count extrapolates a power
        law fitted on the first half and all of your text (median error 1% on held-out sizes)."""
        import collections, math
        V = len(keep)
        cand = lambda i: i < V and keep[i] and always[i] != "1" and not single(i)
        def est(sub):
            cnt = collections.Counter(i for s_ in sub for i in s_)
            return self._gt_unseen(cnt, sum(cnt.values()), cand, min_count)
        n = len(enc_lines)
        p_full = est(enc_lines)
        out = {"predicted_resplit_rate": p_full, "target_resplit_rate": target, "lines_needed_for_target": None,
               "certificate": "ok" if p_full <= target else "add more text"}
        if n >= 200:
            p_half = est(enc_lines[: n // 2])
            if p_half > 0 and p_full > 0 and p_half > p_full:
                b = math.log(p_half / p_full) / math.log(n / (n // 2))
                out["decay_exponent"] = b
                out["lines_needed_for_target"] = n if p_full <= target else int(math.ceil(n * (p_full / target) ** (1 / b)))
        if n < 1000:
            out["certificate_note"] = "fewer than 1,000 lines: the estimate itself is noisy"
        return out

    def vocab_tax(self, languages: List[str], tokenizer=None, token: Optional[str] = None, coverage: bool = False,
                  keep_latin: bool = True, corpus=None, min_count: int = 1, target_resplit: float = 0.01) -> dict:
        """What the vocabulary costs a user of these languages: the token-table rows they never use, in weights and bytes.
        Works for any model whose manifest has token tables (embedding, untied output head, per-layer embeddings) and any
        tokenizer format _tok reads. coverage=True also tokenizes bundled test sentences of each language and counts the
        tokens that would fall in the removed rows (needs tokenizers / sentencepiece / tiktoken / llama_cpp installed)."""
        from ._engine import Engine
        from . import _tok
        if isinstance(languages, str):
            languages = [x.strip() for x in languages.split(",") if x.strip()]
        eng = Engine()
        asked = list(languages)
        if keep_latin and "en" not in [l.lower() for l in languages]:
            languages = list(languages) + ["en"]   # names, numbers, code and loanwords in Latin letters appear in text of every language
        known = set(eng.languages())
        bad = [l for l in languages if l.lower() not in known]
        if bad:
            raise ValueError(f"unknown language code(s) {bad}; known: {', '.join(sorted(known))}")
        voc = (self.data.get("plan") or {}).get("vocab") or {}
        if not voc.get("tables"):
            raise RuntimeError("no token tables found in this model (not a text model, or the vocabulary is not stored as a table)")
        t = self._tok(tokenizer, token)
        kmask, rule = self._keep_mask(t, languages, corpus, min_count, target_resplit)
        mask = "".join("1" if k else "0" for k in kmask)
        V = voc["size"]
        n = min(V, len(mask))
        removed = sum(1 for i in range(n) if mask[i] == "0")
        kept = V - removed                     # rows past the tokenizer file are kept: they may be special tokens it does not list
        bk = (self.data.get("breakdown") or {}).get("by_kind") or {}
        bpp = {k: (bk[k]["bytes"] / bk[k]["params"]) if bk.get(k, {}).get("params") else 2.0 for k in ("embed", "head")}
        tables, p_rm, b_rm, b_tab = [], 0, 0.0, 0.0
        for x in voc["tables"]:
            pb = bpp["head" if x["kind"] == "head" else "embed"]
            rm = removed * x["width"]
            tables.append({"name": x["name"], "what": x["what"], "width": x["width"], "params": x["params"],
                           "bytes": x["params"] * pb, "removable_params": rm, "removable_bytes": rm * pb})
            p_rm += rm; b_rm += rm * pb; b_tab += x["params"] * pb
        tot_b = (self.data.get("totals") or {}).get("bytes") or sum(v["bytes"] for v in bk.values()) or 1
        out = {"model": self.name, "languages": languages, "asked": asked, "latin_added": languages != asked, "rule": rule,
               "tokenizer": {"format": t.kind, "pieces": len(t), "byte_level": t.byte_level, "path": t.path},
               "vocab_rows": V, "padding_rows": max(V - len(t), 0), "kept_rows": kept, "removed_rows": removed,
               "removed_share_of_rows": removed / V, "tables": tables, "table_bytes": b_tab,
               "removable_params": p_rm, "removable_bytes": b_rm, "share_of_params": p_rm / (self.params or 1),
               "share_of_bytes": b_rm / tot_b, "model_bytes": tot_b}
        if len(t) > V:
            out["warning"] = (f"the tokenizer has {len(t):,} pieces but the token tables have {V:,} rows: "
                              "it does not belong to this model, so the numbers are not meaningful")
        if coverage:
            enc = t.encoder()
            if enc is None:
                out["coverage"] = {"error": f"install a library that can encode {t.kind} tokenizers "
                                            "(pip install tokenizers / sentencepiece / tiktoken / llama-cpp-python)"}
            else:
                cov, fert = {}, {}
                for l in sorted(set(languages) | {"en"}):
                    sents = _tok.corpus(l.lower())
                    if not sents:
                        continue
                    ids = [i for s_ in sents for i in enc(s_)]
                    pud = _tok.corpus(l.lower())[:300]
                    fert[l] = sum(len(enc(s_)) for s_ in pud) / max(len(pud), 1)
                    if l in languages:
                        miss = sum(1 for i in ids if i >= len(mask) or i >= V or mask[i] == "0")
                        cov[l] = {"sentences": len(sents), "tokens": len(ids), "tokens_in_removed_rows": miss, "rate": miss / max(len(ids), 1)}
                out["coverage"] = {"per_language": cov, "tokens_per_sentence": fert,
                                   "tokens_vs_english": {l: fert[l] / fert["en"] for l in fert if "en" in fert and fert["en"]}}
        return out

    def vocab_tax_text(self, languages: List[str], **kw) -> str:
        r = self.vocab_tax(languages, **kw)
        L = "/".join(r["languages"]) + (" (English/Latin kept by default; keep_latin=False to drop it)" if r.get("latin_added") else "")
        lines = [f"{r['model']}: vocabulary tax for {L}",
                 f"tokenizer: {r['tokenizer']['pieces']:,} pieces ({r['tokenizer']['format']}); the token tables have {r['vocab_rows']:,} rows"
                 + (f" ({r['padding_rows']:,} rows past the tokenizer file are kept: padding or special tokens)" if r["padding_rows"] else ""),
                 f"{L} needs {r['kept_rows']:,} rows; {r['removed_rows']:,} ({r['removed_share_of_rows']:.0%}) are never used"]
        if r.get("warning"):
            lines.append("WARNING: " + r["warning"])
        ru = r.get("rule") or {}
        if ru.get("rule") == "script + your text":
            lines.append(f"refined with {ru['corpus_lines']:,} lines of your text: {ru['kept_script_only']:,} -> {ru['kept']:,} pieces kept "
                         f"({ru.get('merge_closure_added', 0):,} of them merge parents that kept pieces need); "
                         f"on a held-out 20% of it, {ru['heldout_resplit_rate']:.3%} of tokens would be split into smaller pieces")
            need = ru.get("lines_needed_for_target")
            lines.append(f"certificate: new text of this kind will see about {ru['predicted_resplit_rate']:.2%} of tokens re-split "
                         f"(target {ru['target_resplit_rate']:.2%}): {ru['certificate']}"
                         + (f"; about {need:,} lines would reach the target" if need and need > ru['corpus_lines'] else "")
                         + (f" ({ru['certificate_note']})" if ru.get("certificate_note") else ""))
        for x in r["tables"]:
            lines.append(f"  {x['what']:28s} {x['name']}: {fmt_b(x['bytes'])}, removable {fmt_b(x['removable_bytes'])}")
        lines.append(f"tax: {fmt_b(r['removable_bytes'])} = {r['share_of_bytes']:.1%} of the model's {fmt_b(r['model_bytes'])} "
                     f"({fmt_n(r['removable_params'])} weights, {r['share_of_params']:.1%})")
        c = r.get("coverage")
        if c and "error" in c:
            lines.append("coverage: " + c["error"])
        elif c:
            for l, v in c["per_language"].items():
                lines.append(f"coverage {l}: {v['tokens_in_removed_rows']} of {v['tokens']:,} tokens in {v['sentences']} test sentences "
                             f"fall in removed rows ({v['rate']:.3%})")
            if c["tokens_vs_english"]:
                lines.append("tokens per sentence vs English: " + ", ".join(f"{l} x{v:.1f}" for l, v in c["tokens_vs_english"].items() if l != "en"))
        lines.append("how: keep the kept rows in the embedding (and head / per-layer tables), remap token ids, keep the tokenizer as is "
                     "(Ushio et al. 2023 showed accuracy holds for the target languages).")
        return "\n".join(lines)

    def trim_vocab(self, languages: List[str], out_dir: str, tokenizer=None, token: Optional[str] = None,
                   keep_latin: bool = True, corpus=None, min_count: int = 1, target_resplit: float = 0.01) -> dict:
        """Write a trimmed tokenizer (tokenizer.json or SentencePiece .model), kept_ids.json (old ids in new-id order) and
        slice_weights.py (how to slice the token tables). With the `tokenizers` / `sentencepiece` library installed it also
        checks that the bundled test sentences of every kept language tokenize to exactly the same pieces as before."""
        import os
        from . import _tok, _trim
        from ._engine import Engine
        if isinstance(languages, str):
            languages = [x.strip() for x in languages.split(",") if x.strip()]
        langs = list(languages) + (["en"] if keep_latin and "en" not in [l.lower() for l in languages] else [])
        t = self._tok(tokenizer, token)
        keep, rule = self._keep_mask(t, langs, corpus, min_count, target_resplit)
        os.makedirs(out_dir, exist_ok=True)
        if t.kind == "hf":
            with open(t.path, encoding="utf-8") as f:
                j = json.load(f)
            nj, kept = _trim.trim_tokenizer_json(j, keep)
            new_path = os.path.join(out_dir, "tokenizer.json")
            with open(new_path, "w", encoding="utf-8") as f:
                json.dump(nj, f, ensure_ascii=False)
        elif t.kind == "spm":
            with open(t.path, "rb") as f:
                nb, kept = _trim.trim_sentencepiece(f.read(), keep)
            new_path = os.path.join(out_dir, os.path.basename(t.path))
            with open(new_path, "wb") as f:
                f.write(nb)
        else:
            raise NotImplementedError(f"trimming {t.kind} tokenizers is not supported yet; vocab_tax() still measures them")
        with open(os.path.join(out_dir, "kept_ids.json"), "w") as f:
            json.dump(kept, f)
        with open(os.path.join(out_dir, "slice_weights.py"), "w") as f:
            f.write(_trim.SLICE_SNIPPET)
        rep = {"languages": langs, "rule": rule, "pieces_before": len(t), "pieces_after": len(kept), "tokenizer": new_path, "verified": None}
        nt = _tok.read_file(new_path)
        e0, e1 = t.encoder(), nt.encoder()
        if e0 and e1:
            same = tot = 0
            for l in langs:
                for s_ in _tok.corpus(l.lower()):
                    a = [t.pieces[i] for i in e0(s_)]
                    b = [nt.pieces[i] for i in e1(s_)]
                    tot += 1; same += a == b
            rep["verified"] = {"sentences": tot, "identical": same}
            if rule.get("rule") != "script":
                rep["verified"]["note"] = "with your text as the rule, sentences unlike it may be split into smaller pieces"
        with open(os.path.join(out_dir, "trim_report.json"), "w") as f:
            json.dump(rep, f, indent=1)
        return rep

    def plan(self, languages: Optional[List[str]] = None, text_only: bool = False, tokenizer=None,
             token: Optional[str] = None, keep_tokens: Optional[int] = None) -> List[dict]:
        """Ranked ways to make the model smaller, with the weights each one removes.
        languages:  e.g. ["th", "en", "zh"]; counts the tokenizer pieces those languages need (reads tokenizer.json)
                    and prices the vocabulary cut. keep_tokens sets the number directly instead.
        text_only:  count the vision / audio towers as removable.
        Each item: id, title, params (weights involved), saves (weights removed, or the halving for quantization),
        share, detail (what was measured) and how (what to do)."""
        p = self.data.get("plan") or {}
        lev = [dict(x) for x in p.get("levers") or []]
        voc = p.get("vocab") or {}
        if languages and keep_tokens is None:
            vs = self.vocab_scripts(languages, tokenizer, token)
            keep_tokens = vs["keep"]
            self._last_scripts = vs
        tot = self.params or 1
        for x in lev:
            if x["id"] == "vocab" and keep_tokens is not None and voc:
                x["keep_tokens"] = keep_tokens
                n_pieces = min((getattr(self, "_last_scripts", None) or {}).get("total") or voc["size"], voc["size"])
                x["saves"] = max(n_pieces - keep_tokens, 0) * voc["row_cost"]   # rows past the tokenizer file are kept
                x["saves_share"] = x["saves"] / tot
                x["detail"] += (f" Keeping {keep_tokens:,} tokens" + (f" for {'/'.join(languages)}" if languages else "")
                                + f" removes {fmt_n(x['saves'])} weights ({x['saves_share']:.0%} of the model).")
            if x["id"] == "ple" and keep_tokens is not None and voc:
                w = sum(t["width"] for t in voc.get("tables", []) if t["what"].startswith("per-layer"))
                x["detail"] += f" After the vocabulary cut it is {fmt_n(keep_tokens * w)}."
            if x["id"].startswith("tower:") and not text_only:
                x["saves"], x["saves_share"] = None, None
                x["title"] += " (pass text_only=True to count it)"
        lev.sort(key=lambda x: -(x.get("saves") or x.get("params") or 0))   # same order as the website
        return lev

    def plan_text(self, languages: Optional[List[str]] = None, text_only: bool = False, **kw) -> str:
        """plan() as a readable report: each lever, what it removes, and the size before and after."""
        lev = self.plan(languages, text_only, **kw)
        lines = [f"{self.name}: {fmt_n(self.params)} weights, {fmt_b(self.params * 2)} at 16-bit"]
        vs = getattr(self, "_last_scripts", None)
        if languages and vs:
            top = ", ".join(f"{k} {v:,}" for k, v in list(vs["counts"].items())[:8])
            lines.append(f"tokenizer: {vs['total']:,} pieces ({top}); {'/'.join(languages)} need {vs['keep']:,}")
        cut = 0
        for i, x in enumerate(lev, 1):
            sv = x.get("saves")
            tag = f"-{fmt_n(sv)} ({x['saves_share']:.0%})" if sv and not x["id"] in ("ffn",) else ""
            lines.append(f"{i}. {x['title']}  [{fmt_n(x['params'])}, {x['share']:.0%} of the model] {tag}".rstrip())
            lines.append(f"   {x['detail']}")
            lines.append(f"   how: {x['how']}")
            if sv and x["id"] not in ("ffn", "ple"):
                cut += sv
        if cut:
            left = self.params - cut
            lines.append(f"cutting the structure above: {fmt_n(self.params)} -> {fmt_n(left)} weights "
                         f"({fmt_b(left * 2)} at 16-bit, {fmt_b(left)} at 8-bit, {fmt_b(left / 2)} at 4-bit)")
        kv = (self.data.get("plan") or {}).get("kv")
        if kv and kv.get("shared") or kv and kv.get("sliding"):
            lines.append(f"KV cache: {kv['cached']} of {kv['layers']} layers keep one ({kv['shared']} share an earlier layer's), "
                         f"{kv['sliding']} of them slide over {kv['window']} tokens: {fmt_b(kv['per_token'])} per token plus at most "
                         f"{fmt_b(kv['sliding_cap'])}, not {fmt_b(kv['naive_per_token'])} per token for every layer.")
        return "\n".join(lines)

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
