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
