"""Offline tests: a tiny Llama-style checkpoint written to a temp folder (header + zero bytes), read back."""
import json
import struct

import pytest

import ai_model_explorer as ame
from ai_model_explorer import tools

D, L, H, KV, HD, F, V = 512, 4, 8, 2, 64, 1376, 1000   # realistic widths: projections of 64 or less read as gates


def _write(path, tensors):
    header, off = {}, 0
    for name, shape in tensors.items():
        n = 2
        for s in shape:
            n *= s
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"\0" * off)


@pytest.fixture(scope="module")
def llama(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-llama")
    t = {"model.embed_tokens.weight": [V, D], "model.norm.weight": [D], "lm_head.weight": [V, D]}
    for i in range(L):
        p = f"model.layers.{i}."
        t.update({p + "input_layernorm.weight": [D], p + "post_attention_layernorm.weight": [D],
                  p + "self_attn.q_proj.weight": [H * HD, D], p + "self_attn.k_proj.weight": [KV * HD, D],
                  p + "self_attn.v_proj.weight": [KV * HD, D], p + "self_attn.o_proj.weight": [D, H * HD],
                  p + "mlp.gate_proj.weight": [F, D], p + "mlp.up_proj.weight": [F, D], p + "mlp.down_proj.weight": [D, F]})
    _write(d / "model.safetensors", t)
    (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": D,
                                               "num_hidden_layers": L, "num_attention_heads": H, "num_key_value_heads": KV,
                                               "intermediate_size": F, "vocab_size": V, "head_dim": HD}))
    return d


def test_anatomy(llama):
    m = ame.explore(str(llama))
    assert m.family == "decoder"
    assert m.hp["d"] == D and m.hp["layers"] == L and m.hp["vocab"] == V
    assert m.params == 2 * V * D + D + L * (2 * D + 2 * H * HD * D + 2 * KV * HD * D + 3 * F * D)


def test_kv_cache(llama):
    m = ame.explore(str(llama))
    assert m.kv_cache_bytes_per_token(16) == 2 * L * KV * HD * 2   # K and V, bf16


def test_fits_and_ready(llama):
    m = ame.explore(str(llama))
    assert tools.fits(m, "phone-12")["fits"]
    r = tools.ready(m)
    assert r["tier"] == "Phone" and r["checks"]["safetensors"] and r["checks"]["config.json"]
    assert r["checks"]["config agrees with weights"]
    assert r["submit_url"] is None          # a local folder cannot be submitted
    assert r["checks"]["license declared"] is None and r["out_of"] == 3 and r["score"] == 3


def test_diff_same(llama):
    assert tools.diff(str(llama), str(llama)) == []


def test_census_offline(llama):
    m = ame.explore(str(llama))
    rows = [{"id": "a/x", "fam": "decoder", "p": 6e7, "kv": 2e3, "dl": 10, "origin": "original", "pt": "text-generation"},
            {"id": "b/y", "fam": "decoder", "p": 1e9, "kv": 1e5, "dl": 20, "origin": "original"},
            {"id": "c/clip", "fam": "decoder", "p": 1.5e7, "kv": 2e3, "dl": 99, "origin": "original", "pt": "zero-shot-image-classification"}]
    r = tools.census(m, rows=rows)
    assert r["popular_models"] == 2                    # the dual encoder is not a language model
    assert r["nearest"][0]["id"] == "a/x" and r["page"] is None


def test_whisper_head_stored_beside_decoder(tmp_path):
    """Whisper with proj_out stored at the root: the decoder stays the backbone (vocab, layers and KV cache from the decoder)."""
    d, f, v, el, dl, pos = 512, 2048, 2000, 6, 2, 1500
    t = {"model.encoder.conv1.weight": [d, 80, 3], "model.encoder.conv2.weight": [d, d, 3],
         "model.encoder.embed_positions.weight": [pos, d], "model.encoder.layer_norm.weight": [d],
         "model.decoder.embed_tokens.weight": [v, d], "model.decoder.embed_positions.weight": [448, d],
         "model.decoder.layer_norm.weight": [d], "proj_out.weight": [v, d]}
    def attn(p):
        t.update({p + ".q_proj.weight": [d, d], p + ".k_proj.weight": [d, d], p + ".v_proj.weight": [d, d], p + ".out_proj.weight": [d, d]})
    for side, n in (("encoder", el), ("decoder", dl)):
        for i in range(n):
            p = f"model.{side}.layers.{i}"
            attn(p + ".self_attn")
            if side == "decoder":
                attn(p + ".encoder_attn")
                t[p + ".encoder_attn_layer_norm.weight"] = [d]
            t.update({p + ".self_attn_layer_norm.weight": [d], p + ".final_layer_norm.weight": [d],
                      p + ".fc1.weight": [f, d], p + ".fc2.weight": [d, f]})
    _write(tmp_path / "model.safetensors", t)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"],
        "d_model": d, "encoder_layers": el, "decoder_layers": dl, "encoder_attention_heads": 8, "decoder_attention_heads": 8,
        "encoder_ffn_dim": f, "decoder_ffn_dim": f, "vocab_size": v, "max_source_positions": pos, "num_mel_bins": 80}))
    m = ame.explore(str(tmp_path))
    assert m.family == "encoder-decoder"
    assert m.hp["vocab"] == v and m.hp["layers"] == dl
    assert m.kv_cache_bytes_per_token() == 2 * dl * d * 2
    assert any("stored twice" in s for s in m.insights())


def test_plan_vocabulary_and_scripts(tmp_path, llama):
    """The plan prices the vocabulary per token row, and tokenizer pieces are counted by script."""
    import shutil
    d = tmp_path / "m"
    shutil.copytree(llama, d)
    pieces = ["<pad>", "<0x41>", "▁the", "▁and", "ing", "▁สวัสดี", "ครับ", "你好", "▁Привет", "▁café", "123", "!"]
    pieces += [f"▁w{i}" for i in range(V - len(pieces))]
    (d / "tokenizer.json").write_text(json.dumps({"model": {"type": "Unigram", "vocab": [[p, 0.0] for p in pieces]}}))
    m = ame.explore(str(d), cache=False)
    lev = {x["id"]: x for x in m.data["plan"]["levers"]}
    assert lev["vocab"]["row_cost"] == 2 * D          # untied: embedding row + head row
    vs = m.vocab_scripts(["th", "en"])
    assert vs["counts"]["Thai"] == 2 and vs["counts"]["Han"] == 1 and vs["counts"]["Cyrillic"] == 1
    assert vs["keep"] == V - 2                         # drops 你好 and Привет; café stays (accented names and loanwords occur in English)
    cut = {x["id"]: x for x in m.plan(["th", "en"])}["vocab"]
    assert cut["saves"] == 2 * 2 * D


def _pb_str(field, s):
    b = s.encode()
    return bytes([field << 3 | 2]) + _varint(len(b)) + b


def _varint(n):
    out = b""
    while True:
        c = n & 0x7F; n >>= 7
        out += bytes([c | (0x80 if n else 0)])
        if not n:
            return out


def test_tokenizer_formats(tmp_path):
    """Every reader returns the same pieces in id order: SentencePiece, tiktoken, tekken, GGUF, vocab.json."""
    import base64
    import struct as st
    from ai_model_explorer import _tok
    words = ["<unk>", "▁the", "สวัสดี", "你好", "▁Привет"]
    # SentencePiece ModelProto: repeated field 1 = {1: piece, 2: score, 3: type}; type 2 = unknown
    sp = b""
    for i, w in enumerate(words):
        inner = _pb_str(1, w) + bytes([2 << 3 | 5]) + st.pack("<f", -i) + (bytes([3 << 3]) + _varint(2) if i == 0 else b"")
        sp += bytes([1 << 3 | 2]) + _varint(len(inner)) + inner
    (tmp_path / "tokenizer.model").write_bytes(sp)
    assert _tok.load(str(tmp_path / "tokenizer.model")).pieces == ["<unk>"] + words[1:]
    raw = [w.replace("▁", " ").encode() for w in words[1:]]
    (tmp_path / "x.tiktoken").write_text("".join(f"{base64.b64encode(b).decode()} {i}\n" for i, b in enumerate(raw)))
    assert _tok.load(str(tmp_path / "x.tiktoken")).pieces == [b.decode() for b in raw]
    (tmp_path / "tekken.json").write_text(json.dumps({"config": {"default_num_special_tokens": 2, "default_vocab_size": 6},
                                                      "vocab": [{"rank": i, "token_bytes": base64.b64encode(b).decode()} for i, b in enumerate(raw)]}))
    tk = _tok.load(str(tmp_path / "tekken.json"))
    assert len(tk) == 6 and tk.pieces[2:] == [b.decode() for b in raw]
    def s(x):
        b = x.encode(); return st.pack("<Q", len(b)) + b
    g = b"GGUF" + st.pack("<I", 3) + st.pack("<Q", 0) + st.pack("<Q", 2)
    g += s("tokenizer.ggml.model") + st.pack("<I", 8) + s("llama")
    g += s("tokenizer.ggml.tokens") + st.pack("<I", 9) + st.pack("<I", 8) + st.pack("<Q", len(words)) + b"".join(s(w) for w in words)
    (tmp_path / "v.gguf").write_bytes(g)
    gg = _tok.load(str(tmp_path / "v.gguf"))
    assert gg.pieces == words and gg.kind == "gguf" and not gg.byte_level
    (tmp_path / "vocab.json").write_text(json.dumps({"Ġthe": 0, "à¸ª": 1}))
    assert _tok.load(str(tmp_path / "vocab.json")).byte_level


def test_vocab_tax(tmp_path, llama):
    """The tax counts the rows a language set never uses, in every token table, priced at the stored dtype."""
    import shutil
    d = tmp_path / "m"
    shutil.copytree(llama, d)
    pieces = ["<pad>", "▁the", "▁สวัสดี", "你好", "▁Привет", "▁café"] + [f"▁w{i}" for i in range(V - 6)]
    (d / "tokenizer.json").write_text(json.dumps({"model": {"type": "Unigram", "vocab": [[p, 0.0] for p in pieces]}}))
    m = ame.explore(str(d), cache=False)
    r = m.vocab_tax(["th", "en"])
    assert r["removed_rows"] == 2 and r["kept_rows"] == V - 2
    assert r["removable_params"] == 2 * 2 * D                    # embedding row + untied head row, per removed token
    assert r["removable_bytes"] == 2 * 2 * D * 2                 # BF16
    assert r["vocab_rows"] == V and r["padding_rows"] == 0
    with pytest.raises(ValueError):
        m.vocab_tax(["xx"])


def test_trim_vocab_exact(tmp_path, llama):
    """A trimmed tokenizer.json tokenizes the kept languages exactly as before, and new ids map back to old ids."""
    tk = pytest.importorskip("tokenizers")
    import shutil
    from ai_model_explorer import _tok
    corpus = _tok.corpus("th") + _tok.corpus("en") + _tok.corpus("ru") + _tok.corpus("zh")
    t = tk.Tokenizer(tk.models.BPE(unk_token=None))
    t.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = tk.decoders.ByteLevel()
    t.train_from_iterator(corpus, tk.trainers.BpeTrainer(vocab_size=V, initial_alphabet=tk.pre_tokenizers.ByteLevel.alphabet()))
    d = tmp_path / "m"
    shutil.copytree(llama, d)
    t.save(str(d / "tokenizer.json"))
    m = ame.explore(str(d), cache=False)
    rep = m.trim_vocab(["th"], str(tmp_path / "out"))          # English is added by default
    assert rep["languages"] == ["th", "en"] and rep["pieces_after"] < rep["pieces_before"]
    assert rep["verified"]["identical"] == rep["verified"]["sentences"]
    kept = json.loads((tmp_path / "out" / "kept_ids.json").read_text())
    nt = tk.Tokenizer.from_file(str(tmp_path / "out" / "tokenizer.json"))
    for s in ["สวัสดีครับ วันนี้อากาศดี", "Hello world 2026", "Привет мир"]:   # Russian still encodes, through bytes
        assert nt.decode(nt.encode(s).ids) == s
    s = "สวัสดีครับ Hello"
    assert [kept[i] for i in nt.encode(s).ids] == t.encode(s).ids


def test_vocab_tax_with_corpus(tmp_path, llama):
    """Your own text refines the keep set to the language level and reports the held-out re-split rate."""
    tk = pytest.importorskip("tokenizers")
    import shutil
    from ai_model_explorer import _tok
    t = tk.Tokenizer(tk.models.BPE(unk_token=None))
    t.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.train_from_iterator(_tok.corpus("th") + _tok.corpus("en") + _tok.corpus("de"),
                          tk.trainers.BpeTrainer(vocab_size=V, initial_alphabet=tk.pre_tokenizers.ByteLevel.alphabet()))
    d = tmp_path / "m"
    shutil.copytree(llama, d)
    t.save(str(d / "tokenizer.json"))
    (tmp_path / "mine.txt").write_text("\n".join(_tok.corpus("th")), encoding="utf-8")
    m = ame.explore(str(d), cache=False)
    a = m.vocab_tax(["th"])
    b = m.vocab_tax(["th"], corpus=[str(tmp_path / "mine.txt")])
    assert b["removed_rows"] > a["removed_rows"]
    assert b["rule"]["rule"] == "script + your text" and 0 <= b["rule"]["heldout_resplit_rate"] < 0.2
