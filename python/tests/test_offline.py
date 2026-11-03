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


def test_diff_same(llama):
    assert tools.diff(str(llama), str(llama)) == []


def test_census_offline(llama):
    m = ame.explore(str(llama))
    rows = [{"id": "a/x", "fam": "decoder", "p": 1e6, "kv": 1e3, "dl": 10, "origin": "original"},
            {"id": "b/y", "fam": "decoder", "p": 1e9, "kv": 1e5, "dl": 20, "origin": "original"}]
    r = tools.census(m, rows=rows)
    assert r["params_percentile"] == 50 and r["nearest"][0]["id"] == "a/x"
