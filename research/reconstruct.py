"""Manifests for models that transformers cannot instantiate, rebuilt from their public source code + config.
These are reconstructions (module names follow the source; checkpoint key prefixes may differ). Shapes follow the
constructor arguments. Used as case studies only, never as ground truth.
"""
import json, sys

def T(shape, dt="F32"): return {"dtype": dt, "shape": list(shape)}

def kumo_block(M, p, C, H, scaling, rope=False, ln_bias=False, mlp_mult=2):
    h = C // H
    M[p + "query_norm.weight"] = T([C]); M[p + "key_value_norm.weight"] = T([C])
    if ln_bias: M[p + "query_norm.bias"] = T([C]); M[p + "key_value_norm.bias"] = T([C])
    M[p + "attn.qkv_lin.weight"] = T([3 * C, C]); M[p + "attn.out_lin.weight"] = T([C, C])
    M[p + "mlp.0.weight"] = T([C])
    if ln_bias: M[p + "mlp.0.bias"] = T([C])
    M[p + "mlp.1.weight"] = T([mlp_mult * C, C]); M[p + "mlp.1.bias"] = T([mlp_mult * C])
    M[p + "mlp.3.weight"] = T([C, mlp_mult * C]); M[p + "mlp.3.bias"] = T([C])
    if scaling == "log": M[p + "query_scaling.head_scale"] = T([H])
    if scaling == "gated":
        M[p + "query_scaling.gate.0.weight"] = T([64, h]); M[p + "query_scaling.gate.0.bias"] = T([64])
        M[p + "query_scaling.gate.2.weight"] = T([h, 64]); M[p + "query_scaling.gate.2.bias"] = T([h])
    if rope: M[p + "query_transform.rope.inv_freq"] = T([h // 2])

def kumo_tabular(size="large", classes=10):
    K = {"small": (128, 4, 4, 128, 512, 12, 8), "medium": (256, 6, 4, 256, 512, 24, 8), "large": (256, 6, 4, 256, 1024, 24, 16)}[size]
    C, nE, hE, ind, Ci, nI, hI = K; F, G, R = 32, 3, 4
    M = {}
    p = "row_embedding."
    M[p + "cell_embedding.num_freq"] = T([F]); M[p + "cell_embedding.cat_freq"] = T([F])
    M[p + "cell_embedding.num_lin.weight"] = T([C, 2 * F]); M[p + "cell_embedding.num_lin.bias"] = T([C])
    M[p + "cell_embedding.cat_lin.weight"] = T([C, 2 * F]); M[p + "cell_embedding.cat_lin.bias"] = T([C])
    M[p + "cell_embedding.nan_lin.weight"] = T([C, G])
    M[p + "y_emb.weight"] = T([classes, C]); M[p + "readout_token"] = T([R, C])
    for i in range(nE):
        q = p + f"col_blocks.{i}."; M[q + "inducing_points"] = T([ind, C])
        kumo_block(M, q + "inducing_block.", C, hE, "log"); kumo_block(M, q + "output_block.", C, hE, None)
    for i in range(nE): kumo_block(M, p + f"row_blocks.{i}.", C, hE, "gated", rope=True)
    M[p + "norm.weight"] = T([C])
    if C * R != Ci: M["row_project.weight"] = T([Ci, C * R]); M["row_project.bias"] = T([Ci])
    p = "icl_block."
    M[p + "y_emb.weight"] = T([classes, Ci])
    for i in range(nI): kumo_block(M, p + f"layers.{i}.", Ci, hI, "log")
    M[p + "norm.weight"] = T([Ci]); M[p + "head.0.weight"] = T([2 * Ci, Ci]); M[p + "head.0.bias"] = T([2 * Ci]); M[p + "head.2.weight"] = T([classes, 2 * Ci]); M[p + "head.2.bias"] = T([classes])
    return M

def kumo_relational(classes=10):
    C, nE, hE, ind, R, nI, hI = 128, 3, 8, 128, 4, 12, 8; Ci = C * R; G = 3
    M = {}
    p = "row_embedding."
    M[p + "lin.weight"] = T([C, G]); M[p + "lin.bias"] = T([C]); M[p + "y_emb.weight"] = T([classes, C]); M[p + "readout_token"] = T([R, C])
    for i in range(nE):
        q = p + f"col_layers.{i}."; M[q + "inducing_points"] = T([ind, C])
        kumo_block(M, q + "inducing_block.", C, hE, None, ln_bias=True); kumo_block(M, q + "output_block.", C, hE, None, ln_bias=True)
    for i in range(nE): kumo_block(M, p + f"row_layers.{i}.", C, hE, None, ln_bias=True)
    M[p + "norm.weight"] = T([C]); M[p + "norm.bias"] = T([C])
    g = "gnn."
    for n in ["edge_type_lin", "skip_lin", "src_lin", "out_lin"]: M[g + n + ".weight"] = T([Ci, Ci]); M[g + n + ".bias"] = T([Ci])
    M[g + "aggregation_lin.weight"] = T([Ci, 5 * Ci])
    for n in ["norm", "out_norm"]: M[g + n + ".weight"] = T([Ci]); M[g + n + ".bias"] = T([Ci])
    p = "icl_block."
    M[p + "y_emb.weight"] = T([classes, Ci])
    for i in range(nI): kumo_block(M, p + f"layers.{i}.", Ci, hI, None, ln_bias=True)
    M[p + "norm.weight"] = T([Ci]); M[p + "norm.bias"] = T([Ci]); M[p + "head.0.weight"] = T([2 * Ci, Ci]); M[p + "head.0.bias"] = T([2 * Ci]); M[p + "head.2.weight"] = T([classes, 2 * Ci]); M[p + "head.2.bias"] = T([classes])
    return M

def minimax_h3(fp8=True):
    d, L, Lr, H, hd, ff, cin, ain, tdim, tE, tH, fq = 5376, 50, 2, 56, 128, 14336, 24, 32, 5120, 2688, 5376, 256
    pd = cin * 1 * 2 * 2
    M = {}
    def lin(n, o, i, bias, q=False):
        if q and fp8: M[n + ".weight"] = T([o, i], "F8_E4M3"); M[n + ".weight_scale"] = T([o, 1], "F32")
        else: M[n + ".weight"] = T([o, i], "BF16")
        if bias: M[n + ".bias"] = T([o], "BF16")
    def attn(p, q):
        for x in ["to_q", "to_k", "to_v"]: lin(p + x, H * hd, d, False, q)
        M[p + "norm_q.weight"] = T([hd], "BF16"); M[p + "norm_k.weight"] = T([hd], "BF16"); lin(p + "to_out.0", d, H * hd, False, q)
    def ff_(p, q): lin(p + "net.0.proj", 2 * ff, d, False, q); lin(p + "net.2", d, ff, False, q)
    lin("proj_in", d, pd, True); lin("audio_proj_in", d, ain, True); lin("context_embedder", d, tdim, True)
    lin("time_embedder.linear_1", tH, fq, True); lin("time_embedder.linear_2", tE, tH, True)
    for i in range(Lr):
        p = f"token_refiner.refiner_blocks.{i}."; M[p + "norm1.weight"] = T([d], "BF16"); attn(p + "attn.", False); M[p + "norm2.weight"] = T([d], "BF16"); ff_(p + "ff.", False)
    M["token_refiner.final_norm.weight"] = T([d], "BF16")
    for i in range(L):
        p = f"transformer_blocks.{i}."; M[p + "norm1.weight"] = T([d], "BF16"); attn(p + "attn.", True); M[p + "norm2.weight"] = T([d], "BF16"); ff_(p + "ff.", True)
        lin(p + "adaln_proj.linear", 6 * d * 3, tE, True)
    M["norm_out.norm.weight"] = T([d], "BF16"); lin("norm_out.linear", 2 * d, tE, True)
    lin("proj_out", pd, d, True); lin("audio_proj_out", ain, d, True)
    return M

if __name__ == "__main__":
    out = {"kumo_tabular_large": kumo_tabular("large"), "kumo_relational": kumo_relational(), "fasth3_omni_fp8_transformer": minimax_h3(True), "fasth3_omni_bf16_transformer": minimax_h3(False)}
    for k, v in out.items():
        n = sum(__import__("math").prod(t["shape"]) for t in v.values())
        print(k, len(v), f"{n/1e6:.1f}M")
    json.dump(out, open(sys.argv[1], "w"))
