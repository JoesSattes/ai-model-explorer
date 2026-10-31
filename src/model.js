// Any model -> one normalized description: family, modality, meta, and an anatomy of parts with real metrics.
// Sources: parsed .tflite graphs (real ops), a safetensors header / tensor list (structure), or a config.json (synthesized structure).
"use strict";
const mprod = a => (a || []).reduce((x, y) => x * Math.max(1, y), 1);
const DTN = {FLOAT32: "fp32", F32: "fp32", FLOAT16: "fp16", F16: "fp16", BF16: "bf16", INT8: "int8", I8: "int8", INT4: "int4", UINT8: "uint8", U8: "uint8", F8_E4M3: "fp8", F8_E5M2: "fp8", I32: "int32", INT32: "int32", I64: "int64"};
const DT_BYTES = {fp32: 4, fp16: 2, bf16: 2, int8: 1, uint8: 1, int4: .5, fp8: 1, int32: 4, int64: 8};

function part() { return {ops: 0, flops: 0, bytes: 0, params: 0, dt: {}, seams: 0, w: [], out: null, types: {}, has: false}; }
function merge(dst, src) { ["ops", "flops", "bytes", "params", "seams"].forEach(k => dst[k] += src[k] || 0); Object.entries(src.dt).forEach(([k, v]) => dst.dt[k] = (dst.dt[k] || 0) + v); Object.entries(src.types).forEach(([k, v]) => dst.types[k] = (dst.types[k] || 0) + v); dst.w.push(...src.w); dst.has = dst.has || src.has; if (src.shared) dst.shared = true; if (src.est) dst.est = true; if (src.out && !dst.out) dst.out = src.out; return dst; }
const sumParts = obj => Object.values(obj || {}).reduce((a, p) => merge(a, p), part());

// ------------------------------------------------------------------ from .tflite ops
function addOp(p, n, seen) {
  p.has = true; p.ops++; p.flops += n.f || 0; p.types[n.t] = (p.types[n.t] || 0) + 1; if (n.cat === "seam") p.seams++;
  if (n.w && n.cat !== "seam") {
    const key = n.w.bid != null ? "b" + n.w.bid : n.w.n;
    if (!seen.has(key)) { seen.add(key); const pr = mprod(n.w.shape), d = DTN[n.w.dtype] || n.w.dtype; p.bytes += n.w.bytes; p.params += pr; p.dt[d] = (p.dt[d] || 0) + pr; p.w.push(n.w); }
    else p.shared = true;
  }
  if (n.o && n.o.length) p.out = n.o;
}
function opRole(n, scope, st, comp) {
  const s = scope.toLowerCase();
  if (/encoder_attn_layer_norm|cross.*norm/.test(s)) { if (st.region !== "cross") { st.region = "cross"; st.bmm = 0; st.used = new Set(); } return "ln"; }
  if (/self_attn_layer_norm|input_layernorm|ln_1|layernorm_before/.test(s)) { if (st.region !== "self" || st.used.size) { st.region = "self"; st.bmm = 0; st.used = new Set(); } return "ln"; }
  if (/final_layer_norm|post_attention_layernorm|ln_2|layernorm_after/.test(s)) { st.region = "ffn"; return "ln"; }
  if (/layer_?norm|rmsnorm/.test(s)) return "ln";
  if (/gate_proj/.test(s)) { st.region = "ffn"; return "gate"; }
  if (/fc1|up_proj|c_fc|intermediate/.test(s)) { st.region = "ffn"; return "fc1"; }
  if (/fc2|down_proj|mlp\/c_proj/.test(s)) { st.region = "ffn"; return "fc2"; }
  if (/activation|gelu|silu/.test(s) || n.t === "GELU") return "act";
  const named = /q_proj|query/.test(s) ? "q" : /k_proj|key/.test(s) ? "k" : /v_proj|value/.test(s) ? "v" : /out_proj|o_proj/.test(s) ? "out" : /c_attn|qkv/.test(s) ? "qkv" : null;
  if (named && (n.t === "FULLY_CONNECTED" || n.cat === "seam")) { st.used.add(named); return named; }
  if (named) return named;
  if (n.t === "FULLY_CONNECTED") { const r = ["q", "k", "v"].find(x => !st.used.has(x)) || "out"; st.used.add(r); return r; }
  if (n.t === "BATCH_MATMUL") return st.bmm++ % 2 === 0 ? "scores" : "context";
  if (n.t === "SOFTMAX") return "softmax";
  if (n.t === "ADD" && /^(out|fc2|res)$/.test(st.last)) return "res";
  if (n.cat === "shape") return "reshape";
  if (comp === "decoder" && st.region === "self" && /^(k|v|cache)$/.test(st.last)) return "cache";
  return st.last && st.last !== "reshape" ? st.last : "misc";
}
function opStemRole(n, scope, comp) {
  const s = scope.toLowerCase();
  if (n.t === "CONV_2D" || /conv|patch/.test(s)) return "conv";
  if (n.t === "EMBEDDING_LOOKUP" || /embed_tokens|^embed|wte/.test(s)) return "embed";
  if (n.t === "GATHER_ND" || /position|wpe/.test(s)) return "pos";
  if (n.t === "SLICE") return "io";
  return comp === "decoder" ? "mask" : "glue";
}
function opHeadRole(n, scope, comp) {
  const s = scope.toLowerCase();
  if (/layer_?norm|\/ln|^ln|norm/.test(s)) return "ln";
  if (comp === "encoder") return n.l === "xattn" || /cross|kv/.test(s) ? "crosskv" : "glue";
  return n.t === "FULLY_CONNECTED" || /logits|lm_head|proj_out/.test(s) ? "logits" : "glue";
}
function anatomyFromOps(graphs) {
  const A = {comps: {}, kind: "ops"};
  graphs.forEach(g => {
    const comp = g.component === "model" ? "decoder" : g.component, seen = new Set();
    const C = A.comps[comp] || (A.comps[comp] = {stem: {}, layers: [], head: {}, inputs: g.inputs, outputs: g.outputs});
    let st = null, curL = -1;
    g.nodes.forEach(n => {
      const scope = g.scopes[n.s] || "", m = n.b.match(/\.L(\d+)$/);
      if (m) {
        const li = +m[1], L = C.layers[li] || (C.layers[li] = {idx: li, regions: {}});
        if (li !== curL) { st = {region: "self", bmm: 0, used: new Set(), last: null}; curL = li; }
        const role = opRole(n, scope, st, comp); st.last = role;
        const R = L.regions[st.region] || (L.regions[st.region] = {}); addOp(R[role] || (R[role] = part()), n, seen);
      } else if (/\.stem$/.test(n.b)) { const r = opStemRole(n, scope, comp); addOp(C.stem[r] || (C.stem[r] = part()), n, seen); }
      else { const r = opHeadRole(n, scope, comp); addOp(C.head[r] || (C.head[r] = part()), n, seen); }
    });
  });
  return finishAnatomy(A);
}

// ------------------------------------------------------------------ from a tensor list (safetensors header / templates / config synthesis)
const TOWER = /(^|\.)(encoder|audio_tower|audio_encoder|visual|vision_tower|vision_model|vision_encoder|image_encoder)\./;
function tensorPlace(name, enc, dec) {
  const s = name.toLowerCase();
  const comp = enc && dec ? (TOWER.test(s) ? "encoder" : "decoder") : enc ? "encoder" : "decoder";
  if (/(^|\.)mtp\./.test(s)) return {comp: "decoder", head: true, role: "mtp"};
  if (comp === "encoder" && /merger|proj1|proj2|ln_post|multi_modal_projector|mm_projector|audio_projection|post_layernorm/.test(s) && !/\.(?:layers|blocks)\.\d+\./.test(s)) return {comp, head: true, role: /norm|ln_post/.test(s) ? "ln" : "proj"};
  const m = s.match(/\.(?:layers|layer|h|blocks|block)\.(\d+)\./);
  if (m) {
    if (/linear_attn|\.mixer\.|deltanet|ssm/.test(s)) return {comp, layer: +m[1], region: "linear", role: /in_proj_qkv|in_proj\.|in_proj_qkvz/.test(s) ? "qkv" : /in_proj_ba|in_proj_[ab]\b|in_proj_z/.test(s) ? "gate" : /conv/.test(s) ? "conv" : /dt_bias|a_log|\.d$/.test(s) ? "state" : /norm/.test(s) ? "ln" : /out_proj/.test(s) ? "out" : "misc"};
    const region = /encoder_attn|crossattention|cross_attn|encdecattention/.test(s) ? "cross"
      : /mlp|intermediate|\.fc[12]\.|ffn|feed_forward|gate_proj|up_proj|down_proj|c_fc|(^|[^n]\.)output\.dense|final_layer_norm|post_attention_layernorm|ln_2|layernorm_after|output\.layernorm|block_sparse_moe|experts/.test(s) && !/attention\.output\.dense|attention\.output\.layernorm/.test(s) ? "ffn" : "self";
    const role = /norm|ln_\d/.test(s) ? "ln" : /c_attn|qkv|query_key_value|wqkv/.test(s) ? "qkv" : /q_proj|\.query\.|\.q\./.test(s) ? "q" : /k_proj|\.key\.|\.k\./.test(s) ? "k" : /v_proj|\.value\.|\.v\./.test(s) ? "v"
      : region === "ffn" ? (/gate|w1\b/.test(s) ? "gate" : /up_proj|fc1|c_fc|intermediate|w3\b/.test(s) ? "fc1" : /down_proj|fc2|c_proj|output\.dense|w2\b/.test(s) ? "fc2" : /router|gate\.weight/.test(s) ? "gate" : "fc1")
      : /o_proj|out_proj|c_proj|output\.dense|\.o\./.test(s) ? "out" : "misc";
    return {comp, layer: +m[1], region, role};
  }
  const stem = /embed|wte|wpe|conv|patch|cls_token|position|pos_embed|token_type|feature_extractor|feature_projection|masked_spec|shared\.weight/.test(s) && !/lm_head|proj_out/.test(s);
  if (stem) return {comp, stem: true, role: /conv|patch|feature/.test(s) ? "conv" : /position|wpe|token_type|cls_token/.test(s) ? "pos" : /norm/.test(s) ? "ln" : "embed"};
  return {comp, head: true, role: /lm_head|proj_out/.test(s) ? "logits" : /pooler/.test(s) ? "pool" : /classifier|score/.test(s) ? "cls" : /norm|ln_f/.test(s) ? "ln" : "misc"};
}
function anatomyFromTensors(tensors, meta, est = "tensors") {
  const names = Object.keys(tensors), lay = n => /\.(?:layers|layer|h|blocks|block)\.\d+\./.test(n);
  const both = names.some(n => TOWER.test(n.toLowerCase()) && lay(n)) && names.some(n => !TOWER.test(n.toLowerCase()) && lay(n) && !/(^|\.)mtp\./.test(n));
  const enc = both || meta.family === "encoder", dec = both || meta.family === "decoder";
  const A = {comps: {}, kind: est};
  const seqEnc = meta.seqEnc || (meta.modality === "audio" ? 1500 : meta.modality === "image" ? (meta.patches || 196) + 1 : 128);
  const get = c => A.comps[c] || (A.comps[c] = {stem: {}, layers: [], head: {}});
  Object.entries(tensors).forEach(([n, v]) => {
    const pl = tensorPlace(n, both || (enc && meta.family !== "decoder"), both || meta.family === "decoder");
    if (pl.layer == null && pl.head && pl.role === "mtp" && !both && meta.family === "encoder") pl.comp = "encoder";
    let comp = pl.comp === "shared" ? (both ? "decoder" : (meta.family === "decoder" ? "decoder" : "encoder")) : pl.comp;
    if (!both) comp = meta.family === "decoder" ? "decoder" : "encoder";
    const C = get(comp), d = DTN[v.dtype] || "fp32", pr = mprod(v.shape);
    let p;
    if (pl.layer != null) { const L = C.layers[pl.layer] || (C.layers[pl.layer] = {idx: pl.layer, regions: {}}); const R = L.regions[pl.region] || (L.regions[pl.region] = {}); p = R[pl.role] || (R[pl.role] = part()); }
    else if (pl.stem) p = C.stem[pl.role] || (C.stem[pl.role] = part()); else p = C.head[pl.role] || (C.head[pl.role] = part());
    p.has = true; p.est = true; p.params += pr; p.bytes += pr * (DT_BYTES[d] || 4); p.dt[d] = (p.dt[d] || 0) + pr; p.w.push({n, shape: v.shape, dtype: v.dtype}); p.out = v.shape;
    const passes = comp === "encoder" ? seqEnc : 1, isMat = v.shape.length >= 2 && !/norm|position|wpe|token_type|cls_token/.test(n.toLowerCase());
    p.flops += isMat ? (pl.role === "embed" ? v.shape[v.shape.length - 1] * passes : 2 * pr * passes) : pr * passes;
  });
  // hybrid stacks: in a linear-attention layer the pre-norm belongs to the linear block, not to a phantom full-attention block
  Object.values(A.comps).forEach(C => C.layers.forEach(L => { if (!L || !L.regions.linear || !L.regions.self) return;
    if (!Object.keys(L.regions.self).some(r => /^(q|k|v|qkv|out)$/.test(r))) { const ln = L.regions.self.ln; if (ln) L.regions.linear.ln = L.regions.linear.ln ? merge(L.regions.linear.ln, ln) : ln; delete L.regions.self; } }));
  // attention core (no weights): scores + context
  Object.entries(A.comps).forEach(([c, C]) => C.layers.forEach(L => ["self", "cross"].forEach(r => { const R = L.regions[r]; if (!R) return;
    const d = c === "encoder" ? (meta.encD || meta.d || 768) : (meta.d || 768), s = c === "encoder" ? seqEnc : 1, ctx = r === "cross" ? seqEnc : c === "encoder" ? seqEnc : 64;
    R.scores = Object.assign(part(), {has: true, est: true, flops: 2 * s * ctx * d}); R.softmax = Object.assign(part(), {has: true, est: true, flops: 5 * s * ctx * (meta.heads || 12)}); R.context = Object.assign(part(), {has: true, est: true, flops: 2 * s * ctx * d});
    if (c === "decoder" && r === "self") R.cache = Object.assign(part(), {has: true, est: true, flops: 2 * d});
    R.res = Object.assign(part(), {has: true, est: true, flops: s * d}); })));
  Object.values(A.comps).forEach(C => C.layers.forEach(L => { const R = L.regions.linear; if (!R) return; const d = meta.d || 768;
    R.state = R.state || part(); Object.assign(R.state, {has: true, est: true}); R.state.flops += 6 * d * 128; R.res = Object.assign(part(), {has: true, est: true, flops: d}); }));
  Object.values(A.comps).forEach(C => C.layers.forEach(L => { if (L.regions.ffn) { L.regions.ffn.act = Object.assign(part(), {has: true, est: true, flops: 0}); L.regions.ffn.res = Object.assign(part(), {has: true, est: true, flops: 0}); } }));
  // tied output head
  const D = A.comps.decoder;
  if (D && meta.tie && !(D.head.logits && D.head.logits.has)) { const V = meta.vocab || 0, d = meta.d || 0; D.head.logits = Object.assign(part(), {has: true, shared: true, est: true, flops: 2 * V * d, out: [V]}); }
  return finishAnatomy(A);
}
function finishAnatomy(A) {
  Object.values(A.comps).forEach(C => {
    C.layers = C.layers.filter(Boolean);
    C.layers.forEach(L => { L.tot = part(); Object.values(L.regions).forEach(R => { R._tot = sumParts(R); merge(L.tot, R._tot); }); });
    C.stemTot = sumParts(C.stem); C.headTot = sumParts(C.head);
    C.layersTot = C.layers.reduce((a, L) => merge(a, L.tot), part());
    C.tot = merge(merge(merge(part(), C.stemTot), C.layersTot), C.headTot);
  });
  return A;
}

// ------------------------------------------------------------------ config.json -> tensor list (common families)
function tensorsFromConfig(cfg) {
  const mt = String(cfg.model_type || "").toLowerCase(), tc = cfg.text_config || {};
  const g = (...k) => { for (const x of k) { if (cfg[x] != null) return cfg[x]; if (tc[x] != null) return tc[x]; } return null; };
  const dt = String(g("torch_dtype", "dtype") || "float32").replace("torch.", ""), DT = dt.includes("bfloat") ? "BF16" : dt.includes("16") ? "F16" : "F32";
  const T = {}, add = (n, s) => T[n] = {dtype: DT, shape: s};
  const audio = /whisper|wav2vec|hubert|wavlm|speech/.test(mt), image = /vit|deit|beit|swin|clip_vision|siglip_vision|dinov/.test(mt);
  const d = g("hidden_size", "d_model", "n_embd", "dim"), V = g("vocab_size"), heads = g("num_attention_heads", "n_head", "encoder_attention_heads", "num_heads") || 1;
  const kv = g("num_key_value_heads", "n_kv_heads") || heads, hd = g("head_dim") || Math.round(d / heads);
  const tie = g("tie_word_embeddings") !== false;
  if (!d) throw new Error("no hidden size (hidden_size / d_model / n_embd)");
  const encL = g("encoder_layers", "num_layers") , decL = g("decoder_layers", "num_decoder_layers");
  const isEncDec = !!cfg.is_encoder_decoder || /whisper|bart|t5|mbart|marian|pegasus|m2m|nllb/.test(mt);
  const meta = {name: cfg._name_or_path || mt, model_type: mt, d, heads, kv_heads: kv, vocab: V, tie, modality: audio ? "audio" : image ? "image" : "text"};
  const attn = (p, cross) => { add(p + ".q_proj.weight", [heads * hd, d]); add(p + ".k_proj.weight", [(cross ? heads : kv) * hd, d]); add(p + ".v_proj.weight", [(cross ? heads : kv) * hd, d]); add(p + ".out_proj.weight", [d, heads * hd]); };
  if (isEncDec) {
    const ffE = g("encoder_ffn_dim", "d_ff") || 4 * d, ffD = g("decoder_ffn_dim", "d_ff") || 4 * d, nE = encL || g("num_hidden_layers"), nD = decL || nE;
    if (audio) { const nm = g("num_mel_bins") || 80; add("model.encoder.conv1.weight", [d, nm, 3]); add("model.encoder.conv2.weight", [d, d, 3]); add("model.encoder.embed_positions.weight", [g("max_source_positions") || 1500, d]); meta.nmels = nm; meta.seqEnc = g("max_source_positions") || 1500; }
    else add("model.encoder.embed_tokens.weight", [V, d]);
    for (let i = 0; i < nE; i++) { const p = `model.encoder.layers.${i}`; add(p + ".self_attn_layer_norm.weight", [d]); attn(p + ".self_attn"); add(p + ".final_layer_norm.weight", [d]); add(p + ".fc1.weight", [ffE, d]); add(p + ".fc2.weight", [d, ffE]); }
    add("model.encoder.layer_norm.weight", [d]);
    add("model.decoder.embed_tokens.weight", [V, d]); add("model.decoder.embed_positions.weight", [g("max_target_positions") || 448, d]);
    for (let i = 0; i < nD; i++) { const p = `model.decoder.layers.${i}`; add(p + ".self_attn_layer_norm.weight", [d]); attn(p + ".self_attn"); add(p + ".encoder_attn_layer_norm.weight", [d]); attn(p + ".encoder_attn", true); add(p + ".final_layer_norm.weight", [d]); add(p + ".fc1.weight", [ffD, d]); add(p + ".fc2.weight", [d, ffD]); }
    add("model.decoder.layer_norm.weight", [d]); if (!tie) add("proj_out.weight", [V, d]);
    Object.assign(meta, {family: "encdec", ffn: ffE, layers: {enc: nE, dec: nD}});
    return {tensors: T, meta};
  }
  const L = g("num_hidden_layers", "n_layer", "num_layers"), ff = g("intermediate_size", "n_inner", "ffn_dim", "mlp_dim") || 4 * d;
  const decoder = /llama|mistral|mixtral|qwen|gemma|phi|gpt|falcon|olmo|starcoder|bloom|opt|mpt|stablelm|cohere|deepseek|granite|smollm|internlm|baichuan|yi|exaone|typhoon/.test(mt) || (cfg.architectures || []).some(a => /CausalLM|LMHead/.test(a));
  const gated = /llama|mistral|mixtral|qwen|gemma|phi3|olmo|cohere|deepseek|granite|internlm|baichuan|exaone/.test(mt) || /silu|swish/.test(String(g("hidden_act", "activation_function") || ""));
  const experts = g("num_local_experts", "num_experts", "n_routed_experts");
  if (decoder) {
    add("model.embed_tokens.weight", [V, d]);
    for (let i = 0; i < L; i++) { const p = `model.layers.${i}`; add(p + ".input_layernorm.weight", [d]); add(p + ".self_attn.q_proj.weight", [heads * hd, d]); add(p + ".self_attn.k_proj.weight", [kv * hd, d]); add(p + ".self_attn.v_proj.weight", [kv * hd, d]); add(p + ".self_attn.o_proj.weight", [d, heads * hd]); add(p + ".post_attention_layernorm.weight", [d]);
      const E = experts || 1, fe = experts ? (g("moe_intermediate_size") || ff) : ff;
      if (gated) { add(p + ".mlp.gate_proj.weight", experts ? [E, fe, d] : [fe, d]); add(p + ".mlp.up_proj.weight", experts ? [E, fe, d] : [fe, d]); add(p + ".mlp.down_proj.weight", experts ? [E, d, fe] : [d, fe]); }
      else { add(p + ".mlp.fc1.weight", [ff, d]); add(p + ".mlp.fc2.weight", [d, ff]); } }
    add("model.norm.weight", [d]); if (!tie) add("lm_head.weight", [V, d]);
    Object.assign(meta, {family: "decoder", ffn: ff, layers: {dec: L}, gated, experts: experts || 0, experts_active: g("num_experts_per_tok") || 0});
    return {tensors: T, meta};
  }
  // encoder-only (BERT-style text, ViT-style image, wav2vec-style audio)
  if (image) { const ps = g("patch_size") || 16, im = g("image_size") || 224, ch = g("num_channels") || 3; add("embeddings.patch_embeddings.projection.weight", [d, ch, ps, ps]); add("embeddings.cls_token", [1, 1, d]); const np = (im / ps) ** 2; add("embeddings.position_embeddings", [1, np + 1, d]); meta.patches = np; meta.labels = g("num_labels") || Object.keys(cfg.id2label || {}).length || 0; }
  else if (audio) { add("feature_extractor.conv_layers.weight", [512, 512, 3 * 7]); add("feature_projection.projection.weight", [d, 512]); meta.seqEnc = 1500; }
  else { add("embeddings.word_embeddings.weight", [V, d]); add("embeddings.position_embeddings.weight", [g("max_position_embeddings") || 512, d]); add("embeddings.LayerNorm.weight", [d]); }
  for (let i = 0; i < L; i++) { const p = `encoder.layer.${i}`; add(p + ".attention.self.query.weight", [d, d]); add(p + ".attention.self.key.weight", [d, d]); add(p + ".attention.self.value.weight", [d, d]); add(p + ".attention.output.dense.weight", [d, d]); add(p + ".attention.output.LayerNorm.weight", [d]); add(p + ".intermediate.dense.weight", [ff, d]); add(p + ".output.dense.weight", [d, ff]); add(p + ".output.LayerNorm.weight", [d]); }
  if (image && meta.labels) add("classifier.weight", [meta.labels, d]); else if (!audio) add("pooler.dense.weight", [d, d]);
  Object.assign(meta, {family: "encoder", ffn: ff, layers: {enc: L}});
  return {tensors: T, meta};
}

// ------------------------------------------------------------------ meta guess for a raw tensor list (HF header without config)
function metaFromTensors(tensors, name) {
  const ns = Object.keys(tensors), s = ns.join(" ").toLowerCase();
  const enc = /(^|\s|\.)encoder\./.test(s), dec = /(^|\s|\.)decoder\./.test(s);
  const family = enc && dec ? "encdec" : /lm_head|\.h\.\d+\.|model\.layers\.\d+/.test(s) || dec ? "decoder" : "encoder";
  const modality = /conv1|mel|feature_extractor/.test(s) ? "audio" : /patch_embed/.test(s) ? "image" : "text";
  const emb = ns.find(n => /embed_tokens|wte|word_embeddings/.test(n)), q = ns.find(n => /q_proj\.weight|query\.weight|c_attn\.weight/.test(n));
  const d = q ? tensors[q].shape[tensors[q].shape.length - 1] : emb ? tensors[emb].shape[1] : 768;
  const fc = ns.find(n => /fc1\.weight|up_proj\.weight|c_fc\.weight|intermediate\.dense\.weight/.test(n));
  const lay = c => new Set(ns.filter(n => (c ? n.includes(c + ".") : true)).map(n => (n.match(/\.(?:layers|layer|h|blocks)\.(\d+)\./) || [])[1]).filter(x => x != null)).size;
  return {name, family, modality, d, vocab: emb ? tensors[emb].shape[0] : null, ffn: fc ? Math.max(...tensors[fc].shape) : 4 * d, heads: null, kv_heads: null,
          tie: family !== "encoder" && !ns.some(n => /lm_head|proj_out/.test(n)), layers: family === "encdec" ? {enc: lay("encoder"), dec: lay("decoder")} : family === "decoder" ? {dec: lay()} : {enc: lay()}, nmels: modality === "audio" ? 80 : null, seqEnc: modality === "audio" ? 1500 : null};
}

// ------------------------------------------------------------------ pipeline: the parts in reading order
function pipeline(M) {
  const A = M.A, E = A.comps.encoder, D = A.comps.decoder, mt = M.meta, mod = mt.modality;
  const cols = [];
  if (mt.family === "transducer" && E) {
    cols.push({kind: "in_audio", io: true}, {kind: "mel", io: true}, {kind: "conv", part: E.stemTot, id: "e_stem", where: "encoder.stem"}, {kind: "enc", stack: true, comp: "encoder", id: "e_stack"});
    const pj = (A.extra || []).filter(x => x.role === "joint" || x.role === "prediction").reduce((a, x) => merge(a, x.tot), merge(part(), E.headTot));
    cols.push({kind: "pj", part: pj, id: "x_pj", where: "prediction + joint"}, {kind: "out_text", io: true}); return cols;
  }
  if (mt.family === "diffusion" && E) {
    cols.push({kind: "in_latent", io: true}, {kind: "stem_lat", part: E.stemTot, id: "e_stem", where: "encoder.stem"}, {kind: "enc", stack: true, comp: "encoder", id: "e_stack"}, {kind: "cls_head", part: E.headTot, id: "e_head", where: "encoder.head", title: "Unpatchify", oshape: "[N, C·p·p]", what: ["norm สุดท้าย (มี adaLN) แล้วแปลงเวกเตอร์กลับเป็นช่องของ latent", "A final adaLN-modulated norm, then project each vector back to a latent patch."]}, {kind: "out_latent", io: true}); return cols;
  }
  if (mod === "features" && E && !D) {
    const rowE = (A.extra || []).filter(x => x.role === "encoder" || x.role === "graph").reduce((a, x) => merge(a, x.tot), merge(part(), E.stemTot));
    cols.push({kind: "in_features", io: true}, {kind: "stem_feat", part: rowE, id: "e_stem", where: "encoder.stem"}, {kind: "enc", stack: true, comp: "encoder", id: "e_stack"}, {kind: "cls_head", part: E.headTot, id: "e_head", where: "encoder.head", title: "Prediction head", oshape: "[query rows, classes]", what: ["อ่านเวกเตอร์ของแถว query แล้ว MLP ให้คะแนนแต่ละคลาส (หรือ quantile สำหรับ regression)", "Read the query rows' vectors and score each class with an MLP (or quantiles for regression)."]}, {kind: "out_pred", io: true}); return cols;
  }
  cols.push({kind: "in_" + mod, io: true});
  if (mod === "audio" && E && E.stem.conv && (mt.nmels || mt.model_type === "whisper" || /whisper/i.test(mt.name || ""))) cols.push({kind: "mel", io: true});
  if (E) {
    cols.push({kind: mod === "image" ? "patch" : mod === "audio" ? "conv" : "embed_enc", part: E.stemTot, id: "e_stem", where: "encoder.stem"});
    cols.push({kind: "enc", stack: true, comp: "encoder", id: "e_stack"});
    if (D) cols.push({kind: mt.family === "prefix" ? "proj" : "kv", part: E.headTot, id: "e_head", where: "encoder.head"});
    else { cols.push({kind: mod === "image" ? "cls_head" : "pool", part: E.headTot, id: "e_head", where: "encoder.head"}); cols.push({kind: mod === "image" ? "out_class" : "out_vec", io: true}); }
  }
  if (D) {
    if (!E) cols.push({kind: "embed_dec", part: D.stemTot, id: "d_stem", where: "decoder.stem"});
    cols.push({kind: "dec", stack: true, comp: "decoder", id: "d_stack"});
    cols.push({kind: "lmhead", part: D.headTot, id: "d_head", where: "decoder.head"});
    cols.push({kind: "out_text", io: true});
  }
  return cols;
}

// ------------------------------------------------------------------ anatomy from the automatic architecture recovery (infer.js)
// compact template format: "name.#.rest|DTYPE|d0,d1|0-23,40" (layer index ranges replace the first '#')
function expandCompact(rows) {
  const T = {};
  for (const r of rows) { const [key, dt, sh, ix] = r.split("|"); const shape = sh ? sh.split(",").map(Number) : [];
    for (const part of ix.split(",")) { const [a, b] = part.split("-").map(Number); if (a < 0) { T[key] = {dtype: dt, shape}; continue; } for (let i = a; i <= (b == null ? a : b); i++) T[key.replace("#", i)] = {dtype: dt, shape}; } }
  return T;
}
const REGION_OF = {attn: "self", cross: "cross", ffn: "ffn", moe: "moe", ssm: "linear", conv: "conv", mod: "mod", rnn: "rnn"};
const UI_FAMILY = {decoder: "decoder", "prefix-lm": "prefix", "encoder-decoder": "encdec", transducer: "transducer", diffusion: "diffusion", encoder: "encoder", unknown: "encoder"};
function slotsOfIR(I, want) {
  const B = I.comps.find(c => c.role === "backbone"), into = id => I.edges.filter(e => e.to === id).map(e => I.comps.find(c => c.id === e.from)).filter(Boolean);
  let tower = null, adapter = null;
  if (B) for (const c of into(B.id)) { if (c.stacks.length) { tower = c; break; } if (c.adapter || c.graph) { const g = into(c.id).find(x => x.stacks.length); if (g) { tower = g; adapter = c; break; } } }
  let fam = UI_FAMILY[I.family] || "encoder";
  // the weights cannot tell a tied decoder from an encoder; config.json (or a template) can
  if (fam === "encoder" && /^(decoder|prefix|encdec)$/.test(want || "")) { fam = want; if (want !== "decoder" && !tower) tower = I.comps.filter(c => c !== B && c.stacks.length && !c.adapter).sort((a, b) => b.params - a.params)[0] || null; }
  if (fam === "encdec" && !tower) tower = I.comps.find(c => c !== B && c.stacks.length && !c.adapter);
  if (fam === "decoder" || fam === "prefix" || fam === "encdec") return {encoder: tower, decoder: B, adapter, fam: tower ? (fam === "decoder" ? "prefix" : fam) : "decoder"};
  return {encoder: B, decoder: null, adapter: null, tower, fam};
}
function anatomyFromIR(I, T, meta) {
  const S0 = slotsOfIR(I, meta.family), A = {comps: {}, kind: "tensors", ir: true, extra: []};
  const slotOf = {}; ["encoder", "decoder"].forEach(k => { if (S0[k]) slotOf[S0[k].id] = k; }); if (S0.adapter && S0.encoder) slotOf[S0.adapter.id] = "encoder";
  const seqEnc = meta.seqEnc || (meta.modality === "audio" ? 1500 : meta.modality === "image" ? (meta.patches || 256) + 1 : meta.modality === "latent" ? 4096 : 128);
  const get = c => A.comps[c] || (A.comps[c] = {stem: {}, layers: [], head: {}, label: null});
  const layerIdx = {}; I.comps.forEach(c => { let off = 0; c.stacks.forEach(st => { layerIdx[st.path] = off; off += st.n; }); });
  const roleMap = r => ({up: "fc1", down: "fc2", norm: "ln", proj: "out", tokens: "embed", positions: "pos", table: "embed", stem: "conv", "input proj": "conv", logits: "logits", classifier: "cls", "pool/proj": "pool", mlp: "cls", joint: "joint", "pred proj": "joint", timestep: "time", "message passing": "proj", bias: null}[r] !== undefined ? ({up: "fc1", down: "fc2", norm: "ln", proj: "out", tokens: "embed", positions: "pos", table: "embed", stem: "conv", "input proj": "conv", logits: "logits", classifier: "cls", "pool/proj": "pool", mlp: "cls", joint: "joint", "pred proj": "joint", timestep: "time", "message passing": "proj", bias: null})[r] : r);
  const modW = new Map(); for (const [n, t] of Object.entries(I.tensors)) if (T[n] && T[n].shape.length >= 2) modW.set(n.slice(0, n.lastIndexOf(".")), t.role);
  for (const [n, t] of Object.entries(I.tensors)) {
    const v = T[n]; if (!v) continue; const slot = slotOf[t.comp];
    const d = DTN[v.dtype] || (v.q ? (v.q.fmt === "nvfp4" || v.q.fmt === "mxfp4" || v.q.bits === 4 ? "int4" : "int8") : "fp32"), pr = mprod(v.shape), bytes = v.bytes || pr * (DT_BYTES[d] || 4);
    let role = roleMap(t.role) || (v.shape.length < 2 ? (modW.has(n.slice(0, n.lastIndexOf("."))) ? roleMap(modW.get(n.slice(0, n.lastIndexOf(".")))) || "misc" : "ln") : t.kind);
    let p;
    if (!slot) { let X = A.extra.find(x => x.id === (t.comp || "(other)")); const ci = I.comps.find(c => c.id === t.comp);
      if (!X) A.extra.push(X = {id: t.comp || "(other)", role: ci ? ci.role : "other", parts: {}}); p = X.parts[t.kind] || (X.parts[t.kind] = part()); }
    else if (t.block != null) {
      const C = get(slot), idx = (layerIdx[t.stack] || 0) + t.block;
      const L = C.layers[idx] || (C.layers[idx] = {idx, regions: {}, type: null});
      let rk = REGION_OF[t.kind];
      if (!rk) { const key = (t.region || "").toLowerCase(); rk = /att|self|mixer|attn|input_layernorm|norm1|ln_1|ln1/.test(key) ? null : /ff|mlp|post_attention|norm2|ln_2|ln2|final_layer_norm|pre_ff/.test(key) ? "ffn" : null; role = "ln"; }
      L._pending = L._pending || []; if (!rk) { L._pending.push({n, v, d, pr, bytes, role}); continue; }
      const R = L.regions[rk] || (L.regions[rk] = {}); p = R[role] || (R[role] = part());
    } else {
      const C = get(slot);
      const stemish = t.kind === "embed" || t.kind === "conv" || t.role === "timestep" || (slot === "encoder" && t.kind === "mod");
      const isProj = t.kind === "adapter" || t.comp === (S0.adapter && S0.adapter.id);
      if (isProj) role = "proj";
      if (t.kind === "norm") role = "ln";
      p = stemish && !isProj ? (C.stem[role] || (C.stem[role] = part())) : (C.head[role] || (C.head[role] = part()));
    }
    p.has = true; p.est = true; p.params += pr; p.bytes += bytes; p.dt[d] = (p.dt[d] || 0) + pr; p.w.push({n, shape: v.shape, dtype: v.dtype}); p.out = v.shape;
    const passes = slot === "encoder" ? seqEnc : 1, isMat = v.shape.length >= 2 && t.kind !== "embed" && t.kind !== "norm";
    let f = isMat ? 2 * pr * passes : pr * passes; if (t.kind === "embed") f = v.shape[v.shape.length - 1] * passes;
    if (t.role === "expert") { const ci = I.comps.find(c => c.id === t.comp); const E = (ci && ci.stacks.flatMap(s => s.types.flatMap(x => x.regions)).find(r => r.kind === "moe") || {}).E || 8; f *= Math.min(1, (meta.topk || Math.min(8, E)) / E); }
    p.flops += f;
  }
  // pre-norms that did not name their region: hand each to the first region of its layer
  Object.values(A.comps).forEach(C => C.layers.forEach(L => { if (!L || !L._pending) return; const first = ["linear", "self", "moe", "conv", "ffn", "cross", "mod", "rnn"].find(k => L.regions[k]) || "self";
    L._pending.forEach(x => { const R = L.regions[first] || (L.regions[first] = {}); const p = R.ln || (R.ln = part()); p.has = true; p.est = true; p.params += x.pr; p.bytes += x.bytes; p.dt[x.d] = (p.dt[x.d] || 0) + x.pr; p.w.push({n: x.n, shape: x.v.shape, dtype: x.v.dtype}); p.flops += x.pr; }); delete L._pending; }));
  // weightless steps: attention core, residuals, activations, recurrent state
  Object.entries(A.comps).forEach(([c, C]) => C.layers.forEach(L => { if (!L) return;
    ["self", "cross"].forEach(r => { const R = L.regions[r]; if (!R) return; const d = c === "encoder" ? (meta.encD || meta.d || 768) : (meta.d || 768), s = c === "encoder" ? seqEnc : 1, ctx = r === "cross" ? seqEnc : c === "encoder" ? seqEnc : 64;
      R.scores = Object.assign(part(), {has: true, est: true, flops: 2 * s * ctx * d}); R.softmax = Object.assign(part(), {has: true, est: true, flops: 5 * s * ctx * (meta.heads || 12)}); R.context = Object.assign(part(), {has: true, est: true, flops: 2 * s * ctx * d});
      if (c === "decoder" && r === "self") R.cache = Object.assign(part(), {has: true, est: true, flops: 2 * d}); R.res = Object.assign(part(), {has: true, est: true, flops: s * d}); });
    if (L.regions.linear) { const R = L.regions.linear; R.state = R.state || part(); Object.assign(R.state, {has: true, est: true}); R.res = Object.assign(part(), {has: true, est: true, flops: meta.d || 768}); }
    ["ffn", "moe"].forEach(r => { if (L.regions[r]) { L.regions[r].act = Object.assign(part(), {has: true, est: true, flops: 0}); L.regions[r].res = Object.assign(part(), {has: true, est: true, flops: 0}); } });
  }));
  const D = A.comps.decoder;
  if (D && meta.tie && !(D.head.logits && D.head.logits.has)) { const V = meta.vocab || 0, d = meta.d || 0; D.head.logits = Object.assign(part(), {has: true, shared: true, est: true, flops: 2 * V * d, out: [V]}); }
  A.extra.forEach(X => X.tot = sumParts(X.parts));
  A.slots = S0;
  return finishAnatomy(A);
}
function metaFromIR(I, cfg, name) {
  const arch0 = cfg && (cfg.architectures || [])[0] || "";
  const S0 = slotsOfIR(I, /CausalLM|LMHead/.test(arch0) ? "decoder" : /ForConditionalGeneration/.test(arch0) ? "prefix" : null), B = I.comps.find(c => c.role === "backbone") || I.comps[0] || {}, tw = S0.tower || S0.encoder !== B && S0.encoder || null;
  const modOf = c => !c ? "text" : c.modality === "signal" ? (c.inProj && c.inProj.some(k => k % 128 === 0 || k % 80 === 0) ? "audio" : "features") : c.modality === "labels" ? "text" : c.modality;
  const hp = I.hp || {}, g = (...k) => { const tc = cfg && (cfg.text_config || cfg.llm_config) || {}; for (const x of k) { if (cfg && cfg[x] != null) return cfg[x]; if (tc[x] != null) return tc[x]; } return null; };
  let fam = S0.fam; const arch = cfg && (cfg.architectures || [])[0] || "";
  if (fam === "encoder" && /CausalLM|LMHead|ForConditionalGeneration/.test(arch) && B.hasEmbed) fam = "decoder";
  const mod = S0.fam === "transducer" ? "audio" : modOf(tw && S0.fam !== "transducer" ? tw : B);
  const pattern = B.stacks ? B.stacks.map(s => s.typeOf.map(i => s.types[i].label)).flat() : [];
  const runs = []; pattern.forEach(l => { const r = runs[runs.length - 1]; if (r && r[0] === l) r[1]++; else runs.push([l, 1]); });
  // provenance of each architectural field: measured from the weights, read from config.json, or estimated
  const F = cfg ? ArchInfer.fuseConfig(I, cfg) : null, fz = F && B.id ? F[B.id] : null, perHead = hp.head_dim_from === "q/k-norm width";
  const src = {heads: fz && fz.heads ? "config" : perHead ? "measured" : hp.heads ? "estimate" : null, topk: fz && fz.topk ? "config" : null};
  return {name, family: fam, src, fused: fz, fusedAll: F, modality: mod === "latent" ? "latent" : mod === "image" ? "image" : mod === "audio" ? "audio" : mod === "video" ? "image" : mod === "features" ? "features" : "text",
    d: B.d || hp.d, encD: tw ? tw.d : null, heads: fz && fz.heads || hp.heads, kv_heads: fz && fz.kv_heads || hp.kv_heads, head_dim: fz && fz.head_dim || hp.head_dim, heads_exact: src.heads !== "estimate",
    vocab: hp.vocab || g("vocab_size"), tie: hp.tie != null ? hp.tie : g("tie_word_embeddings") !== false, ffn: hp.ffn || g("intermediate_size"), experts: hp.experts || 0, topk: fz && fz.topk || g("num_experts_per_tok", "moe_topk", "top_k") || null,
    layers: S0.decoder ? {enc: S0.encoder ? S0.encoder.layers : 0, dec: B.layers} : {enc: B.layers}, layer_pattern: runs.length > 1 ? runs.map(([l, n]) => (n > 1 ? n + "×" : "") + l).slice(0, 8).join(" · ") + (runs.length > 8 ? " …" : "") : null,
    nmels: mod === "audio" ? 128 : null, seqEnc: mod === "audio" ? 1500 : null, irFamily: I.family, quant: I.quant, joint: I.joint || null, timestep: I.timestep || null, model_type: cfg && cfg.model_type || null};
}
