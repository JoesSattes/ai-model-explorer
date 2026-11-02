// infer.js — recover a model's anatomy from its weight manifest alone (tensor names, shapes, dtypes).
// No weights are downloaded and no model code is run. Works in the browser and in Node.
// (c) 2026 Sattaya Singkul · MIT
// Method: S. Singkul, "Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy", 2026 (paper/).
"use strict";
(function (root) {
const prod = s => (s || []).reduce((a, b) => a * b, 1);
const DTB = {F64: 8, F32: 4, F16: 2, BF16: 2, F8_E4M3: 1, F8_E5M2: 1, F8_E8M0: 1, I64: 8, I32: 4, U32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1, F4: .5};
const isInt = s => /^\d+$/.test(s);
const LEX = { // optional name priors; used only to break ties the shapes cannot (the 'nolex' mode switches them off)
  attn: /(^|_)(attn|attention|self_attn|selfattention|mha|msa|sa)($|_)/i, cross: /(encoder_attn|cross_attn|crossattention|cross_attention|enc_dec|encdecattention|src_attn|crossattn)/i,
  ffn: /(^|_)(mlp|ffn|feed_forward\d?|feedforward|intermediate|fc\d|ff|dense_relu_dense)($|_)/i, moe: /(experts?|moe|router|block_sparse)/i,
  ssm: /(mixer|mamba|ssm|linear_attn|deltanet|time_mix|recurrent|rwkv)/i, conv: /(^|_)(conv|conv_module|convolution|dwconv)($|_)/i,
  mod: /(adaln|ada_ln|modulation|scale_shift|norm1_context|time_embed|timestep)/i, q: /(^|_)(q|q_proj|query|wq|to_q)$/i, k: /(^|_)(k|k_proj|key|wk|to_k)$/i,
  v: /(^|_)(v|v_proj|value|wv|to_v)$/i, o: /(^|_)(o|o_proj|out_proj|dense|wo|to_out|out|c_proj|proj)$/i, gate: /(gate|w1|wi_0)/i, up: /(up|w3|wi_1|fc1|c_fc|intermediate|wi$)/i, down: /(down|w2|wo$|fc2|output)/i,
};

// ------------------------------------------------------------------ 1. normalize: quantized storage -> logical tensors
const AUX = /(^|[._])(weight_scale_2|weight_scale_inv|weight_scale|input_scale|output_scale|k_scale|v_scale|scales|qzeros|g_idx|absmax|quant_map|nested_absmax|nested_quant_map|quant_state(\.\w+)?|weight_zero_point|weight_global_scale|input_global_scale|weight_shape|_amax|biases|SCB|weight_format|zeros)$/;
function normalize(manifest) {
  const T = {}, groups = {};
  for (const [n, v] of Object.entries(manifest)) { if (n === "__metadata__" || !v || !v.shape) continue;
    const bq = n.match(/^(.*)\.weight\.(absmax|quant_map|nested_absmax|nested_quant_map|quant_state(?:\..+)?)$/); // bitsandbytes keeps its state under <module>.weight.*
    if (bq) { (groups[bq[1]] || (groups[bq[1]] = {}))[bq[2].replace(/^quant_state\..*/, "quant_state")] = v; continue; }
    const i = n.lastIndexOf("."); const mod = i < 0 ? "" : n.slice(0, i), leaf = n.slice(i + 1); (groups[mod] || (groups[mod] = {}))[leaf] = v; }
  // residual width guess for un-shaping bitsandbytes weights: the commonest length of 1-D float tensors
  const vl = {}; for (const [n, v] of Object.entries(manifest)) if (v && v.shape && v.shape.length === 1 && /F/.test(v.dtype) && v.shape[0] >= 64 && !/\.weight\./.test(n) && !AUX.test(n)) vl[v.shape[0]] = (vl[v.shape[0]] || 0) + 1;
  const dGuess = +(Object.entries(vl).sort((a, b) => b[1] - a[1])[0] || [0])[0];
  const out = {}, quant = {};
  for (const [mod, g] of Object.entries(groups)) {
    const P = m => mod ? mod + "." + m : m;
    const used = new Set();
    const q = (fmt, bits, extra) => { quant[fmt] = (quant[fmt] || 0) + 1; return Object.assign({fmt, bits}, extra || {}); };
    const bytesOf = keys => keys.reduce((a, k) => a + prod(g[k].shape) * (DTB[g[k].dtype] || 4), 0);
    // GPTQ / AWQ
    if (g.qweight && g.scales) { const qw = g.qweight.shape, sc = g.scales.shape, aux = Object.keys(g).filter(k => /^(qweight|qzeros|scales|g_idx)$/.test(k));
      const gptq = qw[1] === sc[1], i = gptq ? qw[0] * 8 : qw[0], o = gptq ? qw[1] : qw[1] * 8;
      out[P("weight")] = {shape: [o, i], dtype: "I4", bytes: bytesOf(aux), q: q(gptq ? "gptq" : "awq", 4, {group: Math.round(i / sc[0])}), src: aux.map(P)}; aux.forEach(k => used.add(k)); }
    // MXFP4 (gpt-oss style): <name>_blocks [.., n/32, 16] + <name>_scales
    for (const k of Object.keys(g)) { const m = k.match(/^(.*)_blocks$/); if (!m || !g[m[1] + "_scales"]) continue; const b = g[k].shape;
      out[P(m[1])] = {shape: [...b.slice(0, -2), b[b.length - 2] * b[b.length - 1] * 2], dtype: "F4", bytes: bytesOf([k, m[1] + "_scales"]), q: q("mxfp4", 4, {group: 32}), src: [P(k)]}; used.add(k); used.add(m[1] + "_scales"); }
    if (!g.weight && !g.qweight) {
      const ks = Object.keys(g), U8 = ks.find(k => g[k].dtype === "U8" && g[k].shape.length === 2), F8s = ks.find(k => /F8/.test(g[k].dtype) && U8 && g[k].shape[0] === g[U8].shape[0] && g[k].shape[1] * 8 === g[U8].shape[1]);
      const F8w = ks.find(k => /F8/.test(g[k].dtype) && g[k].shape.length >= 2 && prod(g[k].shape) > 64), I32 = ks.filter(k => g[k].dtype === "I32" && g[k].shape.length === 2), F16 = ks.find(k => /F16|BF16/.test(g[k].dtype) && g[k].shape.length === 2);
      if (U8 && F8s) { const w = g[U8].shape, aux = ks.filter(k => k !== U8 && (prod(g[k].shape) <= 1 || k === F8s)); out[P(U8)] = {shape: [w[0], w[1] * 2], dtype: "U8", bytes: bytesOf([U8, ...aux]), q: q("nvfp4", 4, {group: 16}), src: aux.map(P)}; used.add(U8); aux.forEach(k => used.add(k)); }
      else if (F8w) { const aux = ks.filter(k => k !== F8w && /F32|BF16|F16/.test(g[k].dtype) && (prod(g[k].shape) === 1 || g[k].shape.length === 2 && g[k].shape[1] === 1 && g[k].shape[0] === g[F8w].shape[0])); out[P(F8w)] = {shape: g[F8w].shape, dtype: g[F8w].dtype, bytes: bytesOf([F8w, ...aux]), q: q("fp8", 8, {}), src: aux.map(P)}; used.add(F8w); aux.forEach(k => used.add(k)); }
      else if (I32.length && F16) { const qw = I32.map(k => g[k]).sort((a, b) => prod(b.shape) - prod(a.shape))[0], qk = I32.find(k => g[k] === qw), sc = g[F16].shape; if (qw.shape[1] === sc[1]) { const aux = ks.filter(k => k !== qk && (g[k].dtype === "I32" || k === F16)); out[P(qk)] = {shape: [qw.shape[1], qw.shape[0] * 8], dtype: "I4", bytes: bytesOf([qk, ...aux]), q: q("gptq", 4, {group: Math.round(qw.shape[0] * 8 / sc[0])}), src: aux.map(P)}; used.add(qk); aux.forEach(k => used.add(k)); } }
    }
    if (g.weight && !used.has("weight")) {
      const w = g.weight, aux = Object.keys(g).filter(k => k !== "weight" && k !== "bias" && AUX.test("." + k)); let shape = w.shape, qq = null;
      const sc = g.weight_scale || g.weight_scale_inv;
      if (w.dtype === "U8" && sc && /F8/.test(sc.dtype) && sc.shape.length === w.shape.length) { shape = [...w.shape.slice(0, -1), w.shape[w.shape.length - 1] * 2]; qq = q("nvfp4", 4, {group: Math.round(shape[shape.length - 1] / sc.shape[sc.shape.length - 1])}); }
      else if (/F8/.test(w.dtype)) { const n = sc ? prod(sc.shape) : 0; qq = q("fp8", 8, {scale: !sc ? "none" : n === 1 ? "per-tensor" : sc.shape.length === 2 && sc.shape[1] > 1 ? "block" : "per-channel"}); }
      else if (w.dtype === "U32" && g.scales && g.biases) { const per = 32 * w.shape[w.shape.length - 1] / (g.scales.shape[g.scales.shape.length - 1] * 64); const bits = [2, 3, 4, 6, 8].includes(per) ? per : 4; shape = [...w.shape.slice(0, -1), w.shape[w.shape.length - 1] * 32 / bits]; qq = q("mlx", bits, {group: 64}); }
      else if ((w.dtype === "U8" || w.dtype === "I8") && g.absmax) { const L = prod(w.shape) * 2; // nf4/fp4: two values per byte; the shape lives in quant_state, outside the header
        shape = dGuess && L % dGuess === 0 ? (/(down_proj|down|o_proj|out_proj|fc2|wo|w2|c_proj|dense_4h_to_h|dense)$/.test(mod) ? [dGuess, L / dGuess] : [L / dGuess, dGuess]) : [L]; qq = q("bnb4", 4, {}); }
      else if (w.dtype === "I8" && g.SCB) qq = q("int8", 8, {});
      else if (w.dtype === "I8" || w.dtype === "U8") { if (sc || g.scales) qq = q("int8", 8, {}); }
      out[P("weight")] = {shape, dtype: w.dtype, bytes: bytesOf(["weight", ...aux]), q: qq, src: aux.map(P)}; used.add("weight"); aux.forEach(k => used.add(k));
    }
    for (const [k, v] of Object.entries(g)) { if (used.has(k)) continue; if (AUX.test("." + k) && Object.keys(g).length > 1 && (g.weight || g.qweight)) continue;
      out[P(k)] = {shape: v.shape, dtype: v.dtype, bytes: prod(v.shape) * (DTB[v.dtype] || 4), q: null}; }
  }
  return {T: out, quant};
}

// ------------------------------------------------------------------ 2. module tree
function buildTree(T) {
  const rootN = {path: "", tok: "", kids: new Map(), ts: [], parent: null};
  for (const n of Object.keys(T)) {
    const parts = n.split("."); let node = rootN;
    for (let i = 0; i < parts.length - 1; i++) { const t = parts[i]; if (!node.kids.has(t)) node.kids.set(t, {path: node.path ? node.path + "." + t : t, tok: t, kids: new Map(), ts: [], parent: node}); node = node.kids.get(t); }
    node.ts.push(n);
  }
  const fill = nd => { nd.all = [...nd.ts]; nd.mods = nd.ts.length ? 1 : 0; for (const k of nd.kids.values()) { fill(k); nd.all.push(...k.all); nd.mods += k.mods; } };
  fill(rootN); return rootN;
}
const walk = (nd, f) => { f(nd); for (const k of nd.kids.values()) walk(k, f); };
const rel = (n, base) => (base ? n.slice(base.length + 1) : n).split(".").map(t => isInt(t) ? "#" : t).join(".");

// ------------------------------------------------------------------ 3. shape algebra helpers
function widthOf(names, T) { // the residual width of a group of tensors: the dimension most matrices agree on
  const c = new Map(), add = (x, w) => c.set(x, (c.get(x) || 0) + w);
  for (const n of names) { const s = T[n].shape; if (s.length === 2) { add(s[0], 1); add(s[1], 1); } else if (s.length === 1 && s[0] > 8) add(s[0], .6); else if (s.length === 3 && s[0] > 8 && s[1] > 8 && s[2] > 8) { add(s[1], .5); add(s[2], .5); } }
  let best = null, bv = 0; for (const [x, v] of c) if (v > bv || v === bv && x > best) { best = x; bv = v; } return best;
}
const mats = (names, T) => names.filter(n => T[n].shape.length === 2 && Math.min(...T[n].shape) > 1);
const vecs = (names, T) => names.filter(n => T[n].shape.length === 1);
const isDW = s => s.length === 3 && s[1] === 1 && s[2] <= 64 && s[0] > 1; // depthwise conv1d [C, 1, k]
const isDW2 = s => s.length === 4 && s[1] === 1 && s[2] <= 15 && s[0] > 1;
const leafTok = n => { const p = n.split("."); return p.length > 1 ? p[p.length - 2] : p[0]; };

// classify one region (a child module of a block, or a set of loose tensors) by its shapes
function classify(names, T, d, ctx) {
  let r = classify0(names, T, d, ctx);
  const weak = /^(other|half_up|half_down)$/.test(r.kind) || r.kind === "ffn" && mats(names, T).length === 2 && mats(names, T).some(n => T[n].shape[0] === d && T[n].shape[1] >= 2.5 * d && T[n].shape[1] % d === 0 && T[n].shape[1] / d === 3);
  if (weak && mats(names, T).length) {
    const Tt = Object.create(T); names.forEach(n => { const v = T[n]; if (v.shape.length === 2) Tt[n] = Object.assign({}, v, {shape: [v.shape[1], v.shape[0]]}); });
    const r2 = classify0(names, Tt, d, Object.assign({}, ctx));
    if (!/^(other|half_up|half_down)$/.test(r2.kind) && (r2.kind !== "ffn" || r.kind !== "ffn")) { r = r2; r.ev.push("weights stored transposed [in, out]"); r.transposed = true; }
  }
  r.d = d; return r;
}
function classify0(names, T, d, ctx) {
  const ev = [], V = vecs(names, T), all = names.map(n => T[n].shape);
  let M = mats(names, T);
  // head-shaped parameters [H, h] (relative-position biases, per-head scales): not projections, but they reveal H and h
  const dims = new Set(M.flatMap(n => T[n].shape));
  const headish = M.filter(n => { const [a, b] = T[n].shape; return a <= 64 && b >= 16 && dims.has(a * b) && M.some(m => m !== n && T[m].shape.includes(a * b)); });
  if (headish.length) { M = M.filter(n => !headish.includes(n)); ctx.headShape = T[headish[0]].shape; }
  const by = k => names.filter(n => T[n].shape.length === k);
  const fused = names.filter(n => T[n].shape.length === 3 && T[n].shape[0] >= 2 && T[n].shape[0] <= 4096 && T[n].shape[1] > 64 && T[n].shape[2] > 64 && (T[n].shape[1] === d || T[n].shape[2] === d));
  if (fused.length && !ctx.expertList && !names.some(n => isDW(T[n].shape))) { const E = Math.max(...fused.map(n => T[n].shape[0])), r = M.find(n => T[n].shape[1] === d && T[n].shape[0] >= E && T[n].shape[0] <= E + 8);
    ev.push(`${fused.length} stacked expert tensors [${T[fused[0]].shape.join("×")}]${r ? ", router [" + T[r].shape.join("×") + "]" : ""}`); return {kind: "moe", E, ev}; }
  if (ctx.expertList || !names.some(n => isDW(T[n].shape)) && names.some(n => T[n].shape.length === 3 && T[n].shape[0] >= 2 && T[n].shape[0] <= 4096 && T[n].shape[1] > 16 && T[n].shape[2] > 16)) {
    const r = M.find(n => T[n].shape[1] === d && (ctx.expertN ? T[n].shape[0] === ctx.expertN : T[n].shape[0] <= 1024));
    const E = ctx.expertN || (names.find(n => T[n].shape.length === 3) ? T[names.find(n => T[n].shape.length === 3)].shape[0] : null);
    if (r && T[r].shape[0] === E || ctx.expertList) { ev.push(`router [${r ? T[r].shape.join("×") : "?"}] and ${E} experts`); return {kind: "moe", E, ev}; }
  }
  const dws = names.filter(n => isDW(T[n].shape));
  const direct = names.filter(n => ctx.root != null && n.slice(0, n.lastIndexOf(".")) === ctx.root);
  const qkvW = new Map(); M.forEach(n => { if (T[n].shape[1] === d) qkvW.set(T[n].shape[0], (qkvW.get(T[n].shape[0]) || 0) + 1); });
  const attnLike = [...qkvW.values()].some(v => v >= 3); // q, k, v of one width next to a conv: span-based dynamic conv attention (ConvBERT), not a conv module
  if (dws.length && !attnLike) {
    const C = T[dws[0]].shape[0], k = T[dws[0]].shape[2];
    const small = direct.filter(n => T[n].shape.length === 1).map(n => T[n].shape[0]).filter(x => x !== d);
    const cnt = new Map(); small.forEach(x => cnt.set(x, (cnt.get(x) || 0) + 1));
    const rep = [...cnt].filter(([, v]) => v >= 2).map(([x]) => x);
    const stateT = names.filter(n => { const s = T[n].shape; return s.length >= 2 && !isDW(s) && s[s.length - 1] > 1 && s[s.length - 1] <= 256 && s.slice(0, -1).some(x => x === C || 2 * x === C || x === C / 2) && s[s.length - 1] < s[s.length - 2]; });
    if (k <= 8 && (rep.length || stateT.length || direct.filter(n => T[n].shape.length <= 1).length >= 2)) { ev.push(`depthwise conv [${T[dws[0]].shape.join("×")}] + state parameters ${rep.length ? "[" + rep[0] + "] ×" + cnt.get(rep[0]) : stateT.length ? "[" + T[stateT[0]].shape.join("×") + "]" : ""} (A, D, Δt)`); return {kind: "ssm", heads: rep[0], ev}; }
    const pw = names.filter(n => T[n].shape.length === 3 && T[n].shape[2] === 1 || T[n].shape.length === 4 && T[n].shape[2] === 1 && T[n].shape[3] === 1);
    if (pw.length || /running_mean|running_var/.test(names.join(" ")) || k > 8) { ev.push(`depthwise conv k=${k} with pointwise convs`); return {kind: "conv", k, ev}; }
  }
  // recurrent cells: W_ih [g*h, x], W_hh [g*h, h], g in {3,4}
  const hh = M.filter(n => { const [o, i] = T[n].shape; return (o === 4 * i || o === 3 * i); });
  if (hh.length && M.length >= 2 && M.every(n => T[n].shape[0] === T[hh[0]].shape[0] && T[n].shape[0] !== d) && !(ctx.lex && !/(^|[._])(weight|bias)_(ih|hh)|lstm|gru|rnn|recur|cell/i.test(M.join(" ")))) { const [o, i] = T[hh[0]].shape; ev.push(`${o / i === 4 ? "LSTM" : "GRU"} gates ${o} = ${o / i}×${i}`); return {kind: "rnn", cell: o / i === 4 ? "lstm" : "gru", h: i, ev}; }
  if (!M.length) {
    const tab = names.find(n => T[n].shape.length === 3 && T[n].shape[0] === 1 && T[n].shape[2] === d && [2, 3, 4, 6, 9, 12].includes(T[n].shape[1]));
    if (tab) { ev.push(`scale/shift table [${T[tab].shape.join("×")}]`); return {kind: "mod", ev}; }
    if (by(3).length || by(4).length || by(5).length) return {kind: "conv", ev: ["conv kernels"]};
    if (V.length && V.every(n => T[n].shape[0] === d || ctx.lex && /norm|ln/.test(n))) return {kind: "norm", ev: ["vectors of width " + d]};
    return {kind: V.length ? "norm" : "other", ev};
  }
  const ins = M.filter(n => T[n].shape[1] === d), outs = M.filter(n => T[n].shape[0] === d);
  // recurrent / linear-attention mixers without a conv: per-head gate projections [<=64, d] or decay/mix vectors on the mixer itself
  const gates = M.filter(n => T[n].shape[1] === d && T[n].shape[0] <= 64);
  const decay = direct.filter(n => { const s = T[n].shape; return s.length === 1 && s[0] === d || s.length === 3 && s[0] === 1 && s[1] === 1; });
  if (outs.length && ins.length >= 2 && (gates.length >= 2 && !ctx.expertList || decay.length >= 2)) { ev.push(gates.length ? `per-head gates [${T[gates[0]].shape.join("×")}]` : `${decay.length} decay/mix vectors`); return {kind: "ssm", ev}; }
  // feed-forward: d -> f (one or two parallel) -> d
  if (M.length <= 3 && outs.length === 1 && ins.length === M.length - 1 && ins.length >= 1) {
    const f = T[outs[0]].shape[1], ups = ins.map(n => T[n].shape[0]);
    if (ups.every(x => x === f) && ins.length <= 2) { ev.push(`${ins.length === 2 ? "gated " : ""}d ${d} → ${f} → ${d}`); return {kind: "ffn", f, gated: ins.length === 2, ev}; }
    if (ins.length === 1 && ups[0] === 2 * f && f > 1.3 * d) { ev.push(`fused gate+up ${ups[0]} = 2×${f}`); return {kind: "ffn", f, gated: true, ev}; }
    if (ins.length === 1 && ups[0] === 2 * f && ctx.lex && LEX.ffn.test(names.join(" "))) { ev.push(`fused gate+up ${ups[0]} = 2×${f}`); return {kind: "ffn", f, gated: true, ev}; }
  }
  // attention: several projections leave d and one projection returns to d
  if (outs.length >= 1 && M.length >= 2) {
    const fed = n => M.some(m => m !== n && T[m].shape[0] === T[n].shape[1]); // prefer the output fed by a non-square input: k/v can be [d, d] when kv width equals d (Qwen3-0.6B)
    const o = outs.find(n => T[n].shape[1] !== d && fed(n)) || outs.find(fed) || outs[0], x = T[o].shape[1];
    const inp = M.filter(n => n !== o);
    // per-head norms (q_norm / k_norm) live in modules that hold no matrix; biases share a module with their weight and are skipped
    const matMods = new Set(names.filter(n => T[n].shape.length >= 2).map(modOf0));
    const qk = ctx.headShape ? [ctx.headShape[1]] : V.filter(n => !matMods.has(modOf0(n))).map(n => T[n].shape[0]).filter(h => h >= 16 && h <= 512 && x % h === 0 && h !== d);
    if (inp.length >= 1) {
      // external inputs: projections whose input is not produced inside the region (low-rank chains such as MLA are internal)
      const outsSet = new Set(M.map(n => T[n].shape[0])), ext = inp.filter(n => !outsSet.has(T[n].shape[1]) || T[n].shape[1] === d);
      const foreign = ext.filter(n => T[n].shape[0] >= 64).map(n => T[n].shape[1]).filter(w => w !== d && w >= 128), cnt = {}; foreign.forEach(w => cnt[w] = (cnt[w] || 0) + 1);
      const cross = Object.values(cnt).some(c => c >= 2) && ext.some(n => T[n].shape[1] === d);
      if (inp.length === 1) { const a = T[inp[0]].shape[0]; if (a === 3 * x || a > x && a < 3 * x && (a - x) % 2 === 0) { ev.push(`fused qkv ${a} = ${x} + 2×${(a - x) / 2}`); return {kind: "attn", x, kv: (a - x) / 2, qk: qk[0], ev}; } }
      else { ev.push(`${inp.length} input projections → [${x}] → output [${d}×${x}]`); return {kind: cross ? "cross" : "attn", x, kv: inferKV(inp, T, x), qk: qk[0], ev}; }
    }
  }
  // modulation (adaLN): one matrix producing k×d
  if (M.length === 1) { const [o, i] = T[M[0]].shape; if (o % d === 0 && (i === d ? [6, 9, 12, 18].includes(o / d) : o / d >= 2 && o / d <= 24 && i < d && i >= 32)) { ev.push(`[${o}×${i}] = ${o / d}×d shift/scale/gate`); return {kind: "mod", ev}; }
    if (i === d) return {kind: "half_up", f: o, ev: [`d ${d} → ${o}`]}; if (o === d) return {kind: "half_down", f: i, ev: [`${i} → d ${d}`]}; }
  return {kind: "other", ev};
}
function inferKV(inp, T, x) { const o = inp.map(n => T[n].shape[0]); if (o.length >= 3) { const s = [...o].sort((a, b) => a - b); return s[0] === s[1] && s[2] >= s[0] ? s[0] : x; } return x; }

// lexical fallback when the shapes are ambiguous or a region is a single loose matrix
function lexKind(key) { for (const k of ["cross", "moe", "ssm", "mod", "attn", "conv", "ffn"]) if (LEX[k].test(key)) return k; return null; }

// ------------------------------------------------------------------ 4. main
function infer(manifest, opts = {}) {
  const lex = opts.lex !== false, ev = [];
  const {T, quant} = normalize(manifest);
  const R = buildTree(T);
  const out = {tensors: {}, comps: [], edges: [], quant, evidence: ev, hp: {}, totals: {params: 0, bytes: 0}};
  for (const [n, v] of Object.entries(T)) { out.totals.params += prod(v.shape); out.totals.bytes += v.bytes; }
  // lists: nodes whose children are all numeric (>= 2)
  const lists = [];
  const listKeys = ks => { if (ks.every(isInt)) return true; const m = ks.map(k => k.match(/^([A-Za-z]+)_(\d+)$/)); return ks.length >= 3 && m.every(x => x && x[1] === m[0][1]); };
  walk(R, nd => { const ks = [...nd.kids.keys()]; if (ks.length >= 2 && listKeys(ks) && !nd.ts.length) lists.push(nd); });
  const composite = item => item.mods >= 2;
  const inItemOf = (nd, L) => nd.path.startsWith(L.path + ".") && nd !== L;
  // experts: a list nested inside another list's item, uniform small MLP items, often with a router of matching width
  const expert = L => {
    const items = [...L.kids.values()], n = items.length;
    if (lex && /expert/i.test(L.tok)) return true;
    const inside = lists.some(O => O !== L && inItemOf(L, O) && [...O.kids.values()].every(composite));
    if (!inside || n < 4) return false;
    const sig = it => it.all.map(x => rel(x, it.path) + ":" + T[x].shape.length).sort().join("|");
    const s0 = sig(items[0]); if (!items.every(it => sig(it) === s0)) return false;
    const m = items[0].all.filter(x => T[x].shape.length >= 2).length; if (m < 2 || m > 3 || items[0].all.some(x => T[x].shape.length === 1 && !/bias/.test(x))) return false;
    const par = L.parent; return par.all.some(x => !x.startsWith(L.path + ".") && T[x].shape.length === 2 && T[x].shape[0] === n) || n >= 8;
  };
  const expertLists = new Set(lists.filter(expert));
  const blockLists = lists.filter(L => !expertLists.has(L) && [...L.kids.values()].filter(composite).length >= 2);
  let inner = blockLists.filter(L => !blockLists.some(O => O !== L && [...L.kids.values()].some(it => O.path.startsWith(it.path + "."))));
  const sigOf = it => it.all.map(x => rel(x, it.path) + ":" + T[x].shape.length).sort().join("|");
  const sub = new Set();
  for (const O of blockLists) { if (inner.includes(O)) continue; const its = [...O.kids.values()];
    const ins = its.map(it => inner.filter(L => L.path.startsWith(it.path + ".") && L.parent === it));
    const short = ins.every(a => a.length >= 1 && a.every(L => L.kids.size <= 4)) && new Set(ins.flatMap(a => a.map(L => L.kids.size))).size === 1;
    const hetero = ins.every(a => a.length === 1 && new Set([...a[0].kids.values()].map(sigOf)).size === a[0].kids.size), multi = ins.every(a => a.length >= 2);
    if (short && (hetero || multi)) { ins.forEach(a => a.forEach(L => sub.add(L))); inner.push(O); } }
  inner = inner.filter(L => !sub.has(L));
  // homes: the module that owns a stack; stage lists (lists of items that themselves hold stacks) fold into their parent
  const homeOf = L => { let h = L.parent; while (h && h.parent && (isInt(h.tok) || lists.includes(h))) h = h.parent; return h; };
  const homes = new Map();
  const under = (nd, h) => !h.path || nd.path === h.path || nd.path.startsWith(h.path + ".");
  for (const L of inner) { let h = homeOf(L);
    // raise the home while its parent holds no other stack (so embeddings / heads next to an 'encoder' wrapper join it)
    const multiRoot = [...R.kids.values()].filter(k => k.all.length >= 3).length > 1; // several top-level components (diffusers folders, VAE + transformer)
    while (h.parent && !inner.some(O => !under(O, h) && under(O, h.parent)) && !(h.parent === R && multiRoot)) h = h.parent;
    if (!homes.has(h)) homes.set(h, []); homes.get(h).push(L); }
  // nested homes (e.g. a vision tower inside the model home): keep both; a tensor belongs to the deepest home that prefixes it
  const homeList = [...homes.keys()].sort((a, b) => b.path.length - a.path.length);
  const ownerHome = name => homeList.find(h => !h.path || name.startsWith(h.path + ".")) || null;
  const comps = homeList.map(h => ({id: h.path || "(root)", path: h.path, node: h, stacks: [], stem: [], head: [], loose: [], d: null, params: 0, bytes: 0, ev: []}));
  const compOfHome = new Map(comps.map(c => [c.node, c]));
  // 4a. blocks and block types
  const blockOf = new Map();
  for (const [h, Ls] of homes) {
    const C = compOfHome.get(h);
    for (const L of Ls) {
      const items = [...L.kids.values()].filter(composite);
      const sigs = items.map(it => it.all.filter(x => !/num_batches_tracked/.test(x)).map(x => rel(x, it.path) + ":" + T[x].shape.length).sort().join("|"));
      const types = []; const tix = sigs.map(s => { let i = types.findIndex(t => t.sig === s); if (i < 0) { types.push({sig: s, count: 0}); i = types.length - 1; } types[i].count++; return i; });
      const st = {path: L.path, n: items.length, items: items.map(i => i.path), typeOf: tix, types};
      items.forEach((it, k) => it.all.forEach(x => blockOf.set(x, {stack: st, item: it, k, t: tix[k]})));
      C.stacks.push(st);
    }
    const bnames = [].concat(...C.stacks.flatMap(s => s.items.map(p => findNode(R, p).all)));
    C.d = widthOf(bnames.filter(x => !expertLists.size || ![...expertLists].some(E => x.startsWith(E.path + "."))), T);
  }
  // 4b. regions inside one representative block per type
  for (const C of comps) for (const st of C.stacks) st.types.forEach((ty, ti) => {
    const k = st.typeOf.indexOf(ti), item = findNode(R, st.items[k]), d = widthOfBlock(item, T, expertLists) || C.d;
    ty.d = d; ty.regions = regionsOf(item, T, d, expertLists, lex);
    ty.label = blockLabel(ty.regions);
  });
  // per-tensor labels for every block of every type (by relative path through the representative)
  for (const C of comps) for (const st of C.stacks) st.items.forEach((p, k) => {
    const ty = st.types[st.typeOf[k]], map = new Map(); ty.regions.forEach(rg => rg.rel.forEach(r => map.set(r, rg)));
    findNode(R, p).all.forEach(x => { const rg = map.get(rel(x, p)); out.tensors[x] = {comp: C.id, stack: st.path, block: k, kind: rg ? rg.kind : "other", region: rg ? rg.key : null, role: rg ? roleOf(x, rg, T, lex) : null}; });
  });
  // a list whose items never return to their own width is not a residual stack: per-layer prediction heads (box MLPs,
  // hypernetworks) or a list of projectors into another component (DeepStack mergers). The list is kept; its tensors are relabelled.
  for (const C of comps) for (const st of C.stacks) {
    if (st.types.length !== 1) continue; const d = st.types[0].d, ns = findNode(R, st.items[0]).all, M2 = ns.filter(n => T[n].shape.length === 2);
    if (!M2.length || M2.length > 4) continue;
    const ins = new Set(M2.map(n => T[n].shape[1])), freeOut = [...new Set(M2.map(n => T[n].shape[0]))].filter(w => !ins.has(w));
    if (freeOut.length !== 1 || freeOut[0] === d || M2.some(n => T[n].shape[0] === d && T[n].shape[1] !== d)) continue;
    const proj = comps.some(c => c !== C && (c.d === freeOut[0] || [4, 9, 16].some(k => c.d * k === d))) || /merger|projector/i.test(lex ? st.path : "");
    st.items.forEach(p => findNode(R, p).all.forEach(x => { const t = out.tensors[x]; if (!(t.kind === "norm" && T[x].shape.length === 1 && !M2.some(m => x.startsWith(modOf0(m) + ".")))) t.kind = proj ? "adapter" : "head"; }));
    st.types[0].label = proj ? "Projector" : "Head MLP";
    ev.push(`${st.path}: items map ${d} → ${freeOut[0]} and never return to ${d}, read as ${proj ? "projectors" : "per-layer prediction heads"}`);
  }
  for (const X of [...comps]) {
    const Y = comps.find(c => c !== X && (!c.path || X.path.startsWith(c.path + ".")));
    if (!Y || !X.stacks.every(st => st.types.every(t => t.label === "Conv" || t.label === "Block"))) continue;
    const xs = Object.entries(out.tensors).filter(([, t]) => t.comp === X.id), xp = xs.reduce((a, [n]) => a + prod(T[n].shape), 0);
    const yp = Object.entries(out.tensors).filter(([, t]) => t.comp === Y.id).reduce((a, [n]) => a + prod(T[n].shape), 0);
    if (xp > .1 * yp) continue;
    xs.forEach(([, t]) => Object.assign(t, {comp: Y.id, block: null, stack: null, kind: "conv", role: "stem"}));
    comps.splice(comps.indexOf(X), 1); homeList.splice(homeList.indexOf(X.node), 1); X.stacks.forEach(st => ev.push(`${st.path}: small conv stack folded into the ${Y.id || "model"} stem`));
  }
  // 4c. everything outside the blocks: stem, head, adapters
  const widths = comps.map(c => c.d);
  const embRows = [];
  for (const [n, v] of Object.entries(T)) {
    if (out.tensors[n]) continue;
    const h = ownerHome(n); const C0 = h ? compOfHome.get(h) : null; const s = v.shape;
    const C = C0 || (s.length === 2 ? comps.filter(c => c.d === s[1]).sort((a, b) => b.stacks.length - a.stacks.length)[0] : null) || null;
    let kind = "other", role = null;
    if (s.length === 1) kind = "norm";
    else if (s.length >= 3 && !(s.length === 3 && s[0] === 1)) { kind = "conv"; role = "stem"; }
    else if (s.length === 2) {
      const [o, i] = s, d = C ? C.d : null;
      const otherW = w => widths.includes(w) && w !== d;
      if (d && (otherW(o) && (i === d || i % d === 0) || otherW(i) && o === d)) { kind = "adapter"; }
      else if (d && i === d && o > 4 * d && o >= 1000 && embRows.length && embRows.includes(o)) { kind = "head"; role = "logits"; }
      else if (d && i === d && o % d === 0 && o / d <= 8 && o > d) { kind = "head"; role = "mlp"; }
      else if (d && i === d && o >= 1000 && o > 2 * d && !/embed|wte|token|tok_emb|word/i.test(lex ? n : "")) { kind = "head_or_embed"; }
      else if (d && i === d && o >= 1000 && o > d) { kind = "embed"; role = "tokens"; embRows.push(o); }
      else if (d && i === d && o !== d && o >= 128 && o < 1000 && (Number.isInteger(Math.log2(o)) || Number.isInteger(Math.log2(o - 2)) || o === 448)) { kind = "embed"; role = "positions"; }
      else if (d && i === d && o < 1000 && widths.every(w => w !== o)) { kind = "head"; role = o <= 64 ? "classifier" : "proj"; }
      else if (d && o === d && i !== d && widths.includes(i)) { kind = "adapter"; }
      else if (d && i === d && widths.includes(o) && o !== d) { kind = "adapter"; }
      else if (d && o === d && i !== d) { kind = "conv"; role = "input proj"; }
      else if (d && i === d && o === d) { kind = "head"; role = "pool/proj"; }
      else if (s[0] === 1 || s[1] === 1) kind = "embed";
      else kind = widths.includes(i) || widths.includes(o) ? "adapter" : "other";
    } else kind = s.length === 3 ? "embed" : "other";
    out.tensors[n] = {comp: C0 ? C0.id : null, kind, role, region: null};
  }
  for (const [n, t] of Object.entries(out.tensors)) if (t.block == null && (t.kind === "head" && t.role !== "logits" || t.kind === "other") && T[n].shape.length === 2 && T[n].shape[0] < 1000) {
    const par = findNode(R, modOf0(n)); const gp = par && par.parent;
    if (par && par.ts.some(m => m !== n && T[m].shape.length === 1 && T[m].shape[0] === T[n].shape[0])) continue;
    if (gp && gp.all.some(m => m !== n && out.tensors[m] && (out.tensors[m].kind === "embed" && T[m].shape.length === 2 && T[m].shape[0] >= 1000 || out.tensors[m].kind === "conv") && Math.abs(modOf0(m).split(".").length - modOf0(n).split(".").length) <= 1)) Object.assign(t, {kind: "embed", role: "table"});
  }
  // token embedding vs untied output head: same [V, d] shape; the head is usually the later / outer one
  const hoe = Object.entries(out.tensors).filter(([, t]) => t.kind === "head_or_embed");
  for (const [n, t] of hoe) {
    const s = T[n].shape, twins = Object.entries(T).filter(([m, v]) => m !== n && v.shape[0] === s[0] && v.shape[1] === s[1] && !out.tensors[m].block);
    let isHead;
    if (lex && /lm_head|output|head|proj_out|classifier|logits/i.test(n)) isHead = true;
    else if (lex && /embed|wte|tok|word|shared/i.test(n)) isHead = false;
    else if (twins.length) isHead = n.split(".").length < twins[0][0].split(".").length || n > twins[0][0];
    else isHead = false;
    Object.assign(t, isHead ? {kind: "head", role: "logits"} : {kind: "embed", role: "tokens"});
  }
  // lexical cleanup for loose tensors the shapes could not place
  if (lex) for (const [n, t] of Object.entries(out.tensors)) if (t.block == null && (t.kind === "other" || t.kind === "adapter" || t.kind === "head" && t.role === "pool/proj" && /merger|projector|multi_modal|mm_proj|connector/i.test(n))) {
    if (/(^|\.)(lm_head|classifier|score|qa_outputs|proj_out|head|joint)/i.test(n)) t.kind = "head";
    else if (/merger|projector|multi_modal|mm_proj|adapter|connector|mlp1/i.test(n)) t.kind = "adapter";
    else if (/embed|wte|wpe|position|pos_emb|token_type|cls_token|mask_token/i.test(n) && t.kind === "other") t.kind = "embed";
    else if (/conv|patch|subsampl|feature_extractor/i.test(n) && t.kind === "other") t.kind = "conv";
  }
  // leftovers outside every home: attach by width (a [V, d] head next to a 'model' wrapper belongs to the d-wide component)
  for (const [n, t] of Object.entries(out.tensors)) if (!t.comp) {
    const s = T[n].shape, w = s.length === 2 ? s[1] : s.length === 1 ? s[0] : null;
    const c = comps.filter(c => c.d === w).sort((a, b) => b.stacks.reduce((x, y) => x + y.n, 0) - a.stacks.reduce((x, y) => x + y.n, 0))[0];
    if (c && !out.joint) t.comp = c.id;
  }
  // tensor-level structure: a bias follows its weight; a leaf module holding only vectors is a normalisation
  const modOf = n => n.slice(0, Math.max(0, n.lastIndexOf(".")));
  const byMod = new Map(); for (const n of Object.keys(T)) { const m = modOf(n); (byMod.get(m) || byMod.set(m, []).get(m)).push(n); }
  for (const [m, ns] of byMod) {
    const nd = findNode(R, m), big = ns.filter(n => T[n].shape.length >= 2);
    if (big.length) { const k = out.tensors[big[0]].kind; ns.filter(n => T[n].shape.length < 2).forEach(n => { const t = out.tensors[n]; if (t.kind === "norm" || t.kind === "other" || t.block == null) t.kind = k; }); }
    else if (nd && nd.kids.size === 0 && ns.every(n => T[n].shape.length <= 1) && (ns.length <= 2 || /running|num_batches/.test(ns.join(" ")) || !lex)) {
      const ln = ns.filter(n => T[n].shape.length === 1).map(n => T[n].shape[0]);
      if (ln.length && ln.every(x => x === ln[0]) && !(nd.parent && nd.parent.ts.length === 0 && false)) ns.forEach(n => { out.tensors[n].kind = "norm"; });
    }
  }
  const vocabs = new Set(Object.entries(out.tensors).filter(([n, t]) => (t.kind === "embed" || t.kind === "head") && T[n].shape.length === 2 && T[n].shape[0] >= 1000).map(([n]) => T[n].shape[0]));
  for (const [n, t] of Object.entries(out.tensors)) if (t.block == null && T[n].shape.length === 1 && vocabs.has(T[n].shape[0]) && t.kind === "norm") Object.assign(t, {kind: "head", role: "bias"});
  // non-block modules can still be recognised by shape: recurrent cells, attention pools, adaLN heads
  for (const [m, ns] of byMod) { if (ns.some(n => out.tensors[n].block != null)) continue; const C = comps.find(c => c.id === out.tensors[ns[0]].comp);
    const c = classify(ns, T, C ? C.d : -1, {lex}); if (c.kind === "rnn" || c.kind === "mod" || c.kind === "ssm") ns.forEach(n => out.tensors[n].kind = c.kind); }
  // adapters between components: a chain of matrices whose free input width matches one component (or k² × it, after
  // pixel-shuffle merging) and whose free output width matches another
  const adGroups = new Map();
  for (const [n, t] of Object.entries(out.tensors)) if (t.block == null && (t.kind === "adapter" || !t.comp && T[n].shape.length === 2)) {
    let key = modOf0(n); const up = modOf0(key); const upN = up ? findNode(R, up) : null;
    if (upN && upN !== R && upN.all.every(m => out.tensors[m] && out.tensors[m].block == null && !homeList.includes(upN)) && upN.all.length <= 12) key = up;
    (adGroups.get(key) || adGroups.set(key, []).get(key)).push(n);
  }
  for (const [key, ns0] of adGroups) {
    const node = findNode(R, key), ns = node ? node.all.filter(m => out.tensors[m] && out.tensors[m].block == null) : ns0;
    const M2 = ns.filter(n => T[n].shape.length === 2); if (!M2.length) continue;
    const outsW = new Set(M2.map(n => T[n].shape[0])), insW = new Set(M2.map(n => T[n].shape[1]));
    const src = [...insW].find(w => !outsW.has(w)) || T[M2[0]].shape[1], dst = [...outsW].find(w => !insW.has(w)) || T[M2[M2.length - 1]].shape[0];
    const from = comps.find(c => c.d === src && c.d !== dst) || comps.find(c => c.d !== dst && [4, 9, 16].some(k => c.d * k === src)) || comps.find(c => c.d === src), to = comps.find(c => c.d === dst && c !== from);
    if (!from || !to) continue;
    const merge = src !== from.d ? Math.round(Math.sqrt(src / from.d)) : 1;
    const A = {id: key || "(adapter)", path: key, stacks: [], d: dst, params: 0, bytes: 0, ev: [`maps ${from.id} (${from.d}${merge > 1 ? ` ×${merge}² merged = ${src}` : ""}) → ${to.id} (${dst})`], adapter: true, merge};
    comps.push(A); ns.forEach(n => Object.assign(out.tensors[n], {comp: A.id, kind: T[n].shape.length === 2 || out.tensors[n].kind !== "norm" ? "adapter" : "norm"}));
    addEdge(out, from.id, A.id, "features"); addEdge(out, A.id, to.id, "tokens");
  }
  {
    const byM = new Map(); for (const [n, t] of Object.entries(out.tensors)) if (t.block == null && T[n].shape.length === 2) { const top = n.split(".")[0]; (byM.get(top) || byM.set(top, []).get(top)).push(n); }
    for (const [top, ns] of byM) { const sq = ns.filter(n => T[n].shape[0] === T[n].shape[1]), C0 = sq.length ? T[sq[0]].shape[0] : 0;
      const agg = ns.find(n => T[n].shape[0] === C0 && T[n].shape[1] % C0 === 0 && T[n].shape[1] / C0 >= 2 && T[n].shape[1] / C0 <= 8);
      if (sq.length >= 2 && agg && !Object.entries(out.tensors).some(([m, t]) => m.startsWith(top + ".") && t.block != null) && findNode(R, top) && findNode(R, top).all.length <= 24) { const G = {id: top, path: top, stacks: [], d: C0, params: 0, bytes: 0, ev: [`message passing: ${sq.length} square maps [${C0}×${C0}] + aggregation [${T[agg].shape.join("×")}] (${T[agg].shape[1] / C0} reducers)`], graph: true};
        comps.push(G); for (const [n, t] of Object.entries(out.tensors)) if (n.startsWith(top + ".")) Object.assign(t, {comp: top, kind: T[n].shape.length === 2 ? "adapter" : t.kind, role: "message passing"}); ev.push(G.ev[0]);
        const to = comps.find(c => c !== G && c.d === C0 && !c.graph), from = comps.find(c => c !== G && c !== to && !c.graph && (c.d === C0 || [2, 4, 8].some(k => c.d * k === C0)));
        if (from) addEdge(out, from.id, G.id, "row embeddings"); if (to) addEdge(out, G.id, to.id, "row embeddings + neighbours"); } }
  }
  // transducers (RNN-T / TDT): a prediction network (label embedding + RNN) and a joint network whose output size is the
  // label vocabulary (+ duration bins for TDT)
  {
    const emb = Object.entries(out.tensors).filter(([n, t]) => t.block == null && t.kind === "embed" && T[n].shape.length === 2 && T[n].shape[0] >= 64);
    const rnnT = Object.entries(out.tensors).filter(([, t]) => t.kind === "rnn");
    for (const [en] of emb) {
      const [V, jp] = T[en].shape; if (V < 2 * jp) continue;
      const head = Object.entries(T).find(([n, v]) => v.shape.length === 2 && v.shape[0] >= V && v.shape[0] <= V + 16 && n !== en && out.tensors[n].block == null);
      if (!head) continue; const [Vo, j] = head[1].shape;
      const encC = comps.filter(c => !c.adapter).sort((a, b) => b.params - a.params)[0] || comps[0];
      const encP = Object.entries(T).find(([n, v]) => v.shape.length === 2 && v.shape[0] === j && encC && v.shape[1] === encC.d && out.tensors[n].block == null);
      if (!encP || !rnnT.length) continue;
      const predRoot = commonPrefix([en, ...rnnT.map(x => x[0])]);
      const pred = {id: predRoot || "prediction", path: predRoot, stacks: [], d: jp, params: 0, bytes: 0, ev: [`label embedding [${V}×${jp}]${rnnT.length ? " + " + (rnnT.length / 4) + " recurrent layer(s)" : ""}`], pred: true};
      const joint = {id: "joint", path: null, stacks: [], d: j, params: 0, bytes: 0, ev: [`encoder ${encC.d} → ${j}, prediction ${jp} → ${j}, output ${Vo} = ${V} labels${Vo > V ? " + " + (Vo - V) + " duration bins (TDT)" : ""}`], joint: true};
      comps.push(pred, joint);
      for (const [n, t] of Object.entries(out.tensors)) if (t.block == null && predRoot && n.startsWith(predRoot + ".")) { t.comp = pred.id; if (T[n].shape.length === 2 && t.kind !== "embed" && t.kind !== "rnn") t.kind = "head", t.role = "pred proj"; }
      const jn = [head[0], encP[0]]; jn.forEach(b => [b, b.replace(/weight$/, "bias")].forEach(m => { if (out.tensors[m]) Object.assign(out.tensors[m], {comp: "joint", kind: "head", role: "joint"}); }));
      out.joint = {dEnc: encC.d, dPred: jp, j, out: Vo, labels: V, durations: Vo - V};
      addEdge(out, encC.id, "joint", "acoustic frames"); addEdge(out, pred.id, "joint", "label context");
      ev.push(`transducer: ${joint.ev[0]}`); break;
    }
  }
  // diffusion denoisers: a timestep MLP (sinusoidal 256 → h → t) feeding adaLN modulation matrices of input width t
  {
    const modIn = new Set(); for (const C of comps) for (const st of C.stacks) for (const ty of st.types) for (const rg of ty.regions) if (rg.kind === "mod") mats(rg.names, T).forEach(n => modIn.add(T[n].shape[1]));
    const free = Object.entries(T).filter(([n, v]) => v.shape.length === 2 && out.tensors[n].block == null);
    for (const [a, va] of free) { if (![128, 256, 320].includes(va.shape[1]) || va.shape[0] <= va.shape[1] || !modIn.size) continue;
      const b = free.find(([m, vb]) => vb.shape[1] === va.shape[0] && modIn.has(vb.shape[0]) && m !== a);
      if (b) { [a, b[0]].forEach(x => [x, x.replace(/weight$/, "bias")].forEach(m => { if (out.tensors[m]) Object.assign(out.tensors[m], {kind: "mod", role: "timestep"}); })); out.timestep = {freq: va.shape[1], width: b[1].shape[0]}; ev.push(`timestep embedder ${va.shape[1]} → ${va.shape[0]} → ${b[1].shape[0]} drives adaLN`); break; }
    }
  }
  // 4d. component roles, modality, edges
  for (const C of comps) {
    const names = Object.entries(out.tensors).filter(([, t]) => t.comp === C.id).map(([n]) => n);
    names.forEach(n => { C.params += prod(T[n].shape); C.bytes += T[n].bytes; });
    const stemConv = names.filter(n => out.tensors[n].kind === "conv" && out.tensors[n].block == null).map(n => T[n].shape);
    // a component fed by a conv stem (audio frames, image patches) has no token table: its [n, d] tables are positions
    C.hasEmbed = !names.some(n => out.tensors[n].kind === "conv" && out.tensors[n].role === "stem" && T[n].shape.length >= 3) && names.some(n => out.tensors[n].kind === "embed" && T[n].shape.length === 2 && T[n].shape[0] >= 1000);
    C.hasHead = names.some(n => out.tensors[n].kind === "head" && out.tensors[n].role === "logits");
    C.inProj = names.filter(n => out.tensors[n].block == null && T[n].shape.length === 2 && T[n].shape[0] === C.d && T[n].shape[1] !== C.d).map(n => T[n].shape[1]);
    C.modality = C.adapter || C.graph ? "features" : C.joint || C.pred ? "labels" : modalityOf(stemConv, names.map(n => T[n].shape), C);
    if (out.timestep && C.stacks.some(st => st.types.some(t => t.regions.some(r => r.kind === "mod")))) C.modality = "latent";
    const kinds = {}; names.forEach(n => { const k = out.tensors[n].kind; kinds[k] = (kinds[k] || 0) + prod(T[n].shape); }); C.kinds = kinds;
    // parallel sibling lists of equal length (XLM: attentions.#, ffns.#) describe the same layers
    // a multi-token-prediction layer stored as the last item of the stack (DeepSeek-V3, GLM-4.5): one extra block of its own type
    // that also holds a [d, 2d] projection joining the hidden state with the next token's embedding
    for (const st of C.stacks) { if (st.n < 4) continue; const num = p => +p.slice(p.lastIndexOf(".") + 1) || 0;
      const ord = st.items.map((p, i) => i).sort((a, b) => num(st.items[a]) - num(st.items[b])), k = ord[ord.length - 1], t = st.typeOf[k];
      const it = findNode(R, st.items[k]), prev = findNode(R, st.items[ord[ord.length - 2]]); const d0 = st.types[t].d || C.d;
      const join = nd => nd && nd.all.some(n => T[n].shape.length === 2 && T[n].shape[0] === d0 && T[n].shape[1] === 2 * d0);
      const nv = nd => nd ? nd.all.filter(n => T[n].shape.length === 1 && T[n].shape[0] === d0).length : 0;
      if (join(it) && !join(prev) && nv(it) >= nv(prev) + 2) { st.mtp = 1; st.mtpItem = st.items[k]; ev.push(`${st.items[k]}: extra layer with a [${d0}×${2 * d0}] projection, read as a multi-token-prediction head`); } }
    const seenL = new Set(); C.layers = 0; for (const st of C.stacks) { const key = st.path.slice(0, st.path.lastIndexOf(".")) + "#" + st.n; if (seenL.has(key)) continue; seenL.add(key); C.layers += st.n - (st.mtp || 0); }
    const dv = {}; C.stacks.forEach(st => st.types.forEach(t => { if (t.d) dv[t.d] = (dv[t.d] || 0) + t.count; })); const dm = Object.entries(dv).sort((a, b) => b[1] - a[1])[0]; if (dm) C.d = +dm[0];
    C.pattern = C.stacks.map(s => s.typeOf.map(i => (s.types[i].label || "?")[0]).join("")).join(" | ");
  }
  // edges from widths: an adapter maps one component's width to another's; cross-attention reads another component
  const byD = d => comps.filter(c => c.d === d);
  for (const [n, t] of Object.entries(out.tensors)) if (t.kind === "adapter" && T[n].shape.length === 2) {
    const [o, i] = T[n].shape, to = byD(o).find(c => c.id !== t.comp) || byD(o)[0], from = comps.find(c => c.id === t.comp);
    if (to && from && to !== from) addEdge(out, from.id, to.id, `adapter ${i}→${o}`);
  }
  for (const C of comps) for (const st of C.stacks) for (const ty of st.types) for (const rg of ty.regions) if (rg.kind === "cross") {
    const src = comps.find(c => c !== C && (c.d === rg.kvIn || !rg.kvIn)); if (src) addEdge(out, src.id, C.id, "cross-attention");
  }
  // root-level joint networks (transducers): two projections from different widths into one space, then to the vocabulary
  const rootMats = Object.entries(out.tensors).filter(([n, t]) => !t.comp && T[n].shape.length === 2).map(([n]) => n);
  const J = jointOf(rootMats, T, comps);
  if (J) { J.names.forEach(n => Object.assign(out.tensors[n], {kind: "head", role: "joint"})); out.joint = J; ev.push(`joint network: enc ${J.dEnc} + pred ${J.dPred} → ${J.j} → ${J.out} outputs`); }
  // order: sources (raw-signal stems) first, then by edges, the backbone (largest with a head or embedding) last
  const score = c => (c.modality !== "text" ? 0 : 10) + (c.hasHead ? 5 : 0) + out.edges.filter(e => e.to === c.id).length;
  comps.sort((a, b) => score(a) - score(b) || a.params - b.params);
  const back = [...comps].filter(c => !c.adapter && !c.joint && !c.pred && !c.graph).sort((a, b) => (b.hasHead || b.hasEmbed) - (a.hasHead || a.hasEmbed) || b.params - a.params)[0];
  for (const C of comps) C.role = C.graph ? "graph" : C.adapter ? "adapter" : C.joint ? "joint" : C.pred ? "prediction" : C === back ? "backbone" : out.edges.some(e => e.from === C.id) || C.modality !== "text" ? "encoder" : "aux";
  // 4e. hyperparameters of the backbone
  if (back) out.hp = hpOf(back, T, out, comps);
  out.comps = comps.map(c => { delete c.node; return c; });
  out.family = familyOf(out);
  return out;
}
function commonPrefix(ns) { if (!ns.length) return ""; let p = ns[0].split("."); for (const n of ns.slice(1)) { const q = n.split("."); let i = 0; while (i < p.length && p[i] === q[i]) i++; p = p.slice(0, i); } return p.join("."); }
function addEdge(out, a, b, via) { if (!out.edges.some(e => e.from === a && e.to === b)) out.edges.push({from: a, to: b, via}); }
function modOf0(n) { return n.slice(0, Math.max(0, n.lastIndexOf("."))); }
function findChild(item, k) { return item.kids.get(k) || null; }
function findNode(R, p) { let n = R; if (!p) return n; for (const t of p.split(".")) { n = n.kids.get(t); if (!n) return null; } return n; }
function widthOfBlock(item, T, EL) {
  // the block's own normalisation layers (leaf children holding only vectors) carry the residual width
  const nl = [...item.kids.values()].filter(c => c.kids.size === 0 && c.ts.length && c.ts.every(n => T[n].shape.length === 1)).map(c => T[c.ts[0]].shape[0]);
  const loose = item.ts.filter(n => T[n].shape.length === 1).map(n => T[n].shape[0]);
  const all = nl.concat(loose); if (all.length) { const c = {}; all.forEach(x => c[x] = (c[x] || 0) + 1); return +Object.entries(c).sort((a, b) => b[1] - a[1] || b[0] - a[0])[0][0]; }
  return widthOf(item.all.filter(x => ![...EL].some(E => x.startsWith(E.path + "."))), T);
}

function regionsOf(item, T, d, EL, lex) {
  const groups = new Map(), loose = [];
  for (const x of item.all) {
    const r = x.slice(item.path.length + 1).split(".");
    if (r.length === 1) loose.push(x); else { const k = r[0]; (groups.get(k) || groups.set(k, []).get(k)).push(x); }
  }
  const regs = [];
  const leafKids = [...item.kids.values()].every(c => c.kids.size === 0);
  if (leafKids && groups.size >= 2 && !loose.length) { const c = classify(item.all, T, d, {lex, root: item.path}); if (c.kind !== "other" && c.kind !== "norm") return [mk("·", item.all, c, item, T, lex)]; }
  for (const [k, names] of groups) {
    const E = [...EL].find(L => L.path.startsWith(item.path + "." + k + ".") || L.path === item.path + "." + k);
    // a group that wraps a list of sub-layers (T5 'layer.0', 'layer.1'): each sub-layer is its own region
    const kidN = findChild(item, k);
    if (kidN && [...kidN.kids.keys()].length >= 2 && [...kidN.kids.keys()].every(isInt) && [...kidN.kids.values()].every(x => x.mods >= 2)) {
      for (const [kk, sn] of kidN.kids) regs.push(mk(k + "." + kk, sn.all, classify(sn.all, T, d, {lex, root: sn.path, expertList: [...EL].some(L => L.path.startsWith(sn.path + ".")), expertN: ([...EL].find(L => L.path.startsWith(sn.path + ".")) || {kids: {size: 0}}).kids.size}), item, T, lex));
      continue;
    }
    let c = classify(names, T, d, {expertList: !!E, expertN: E ? E.kids.size : 0, lex, root: item.path + "." + k});
    // a module that holds several distinct sub-regions (e.g. one 'layer' module wrapping attention + ffn): split it
    if ((c.kind === "other" || c.kind === "half_up" || c.kind === "half_down") && names.some(n => n.split(".").length - item.path.split(".").length > 2)) {
      const sub = new Map(); names.forEach(n => { const kk = n.slice(item.path.length + 1).split(".").slice(0, 2).join("."); (sub.get(kk) || sub.set(kk, []).get(kk)).push(n); });
      if (sub.size > 1) { for (const [kk, nn] of sub) regs.push(mk(kk, nn, classify(nn, T, d, {lex, root: item.path + "." + kk}), item, T, lex)); continue; }
    }
    if (c.kind !== "moe" && kidN && kidN.kids.size >= 2) {
      const parts = [...kidN.kids.entries()].map(([kk, sn]) => [kk, sn, classify(sn.all, T, d, {lex, root: sn.path})]);
      const conf = new Set(parts.filter(x => x[1].kids.size > 0).map(x => x[2].kind).filter(k => !/^(other|norm|half_up|half_down)$/.test(k)));
      if (conf.size >= 2) { const used = new Set(); parts.forEach(([kk, sn, cc]) => { regs.push(mk(k + "." + kk, sn.all, cc, item, T, lex)); sn.all.forEach(x => used.add(x)); });
        const rest = names.filter(x => !used.has(x)); if (rest.length) regs.push(mk(k + ".·", rest, {kind: rest.every(x => T[x].shape.length === 1) ? "norm" : "other", ev: []}, item, T, lex)); continue; }
    }
    regs.push(mk(k, names, c, item, T, lex));
  }
  // loose matrices directly on the block (fc1 / fc2 style): chain them into a feed-forward
  const lm = loose.filter(x => T[x].shape.length >= 2), lv = loose.filter(x => T[x].shape.length < 2);
  if (lm.length) {
    const c = classify(lm.concat(lv.filter(x => !/norm|ln/i.test(x) || !lex)), T, d, {lex});
    const r = mk("·", lm.concat(lv.filter(x => /bias/.test(x) && lm.some(m => m.replace(/weight$/, "") === x.replace(/bias$/, "")))), c, item, T, lex);
    regs.push(r);
  }
  const used = new Set(regs.flatMap(r => r.names));
  lv.filter(x => !used.has(x)).forEach(x => regs.push(mk(leafTok(x) === item.tok ? "·norm" : "·" + x.split(".").pop(), [x], {kind: T[x].shape.length === 1 ? "norm" : "other", ev: []}, item, T, lex)));
  // halves: an 'up' region followed by a 'down' region of the same width form one feed-forward (BERT intermediate + output)
  const ups = regs.filter(r => r.kind === "half_up"), downs = regs.filter(r => r.kind === "half_down");
  for (const u of ups) { const dn = downs.find(x => x.f === u.f || x.f * 2 === u.f); if (dn) { u.kind = dn.kind = "ffn"; u.ev.push("pairs with " + dn.key); } }
  for (const r of regs) {
    if (r.kind === "half_up" || r.kind === "half_down" || r.kind === "other") { const lk = lex ? lexKind(r.key + " " + r.names.join(" ")) : null; r.kind = lk || (r.kind === "other" ? "other" : "ffn"); if (lk) r.ev.push("by name"); }
    if (lex && r.kind === "attn" && LEX.cross.test(r.key)) r.kind = "cross";
    if (lex && r.kind === "ffn" && LEX.moe.test(r.key + " " + r.names.join(" ")) && r.names.some(n => /router|gate\.weight$|experts/i.test(n))) r.kind = "moe";
  }
  // two attention regions in one block and no width evidence: the one named like cross (or, shape-only, the second) reads the encoder
  const at = regs.filter(r => r.kind === "attn"); if (!lex && at.length === 2 && !regs.some(r => r.kind === "cross")) at[1].kind = "cross";
  // norms: attach each loose vector to the region whose name it echoes, else it stays a norm
  return regs;
}
function mk(key, names, c, item, T, lex) {
  const params = names.reduce((a, n) => a + prod(T[n].shape), 0);
  const kvIn = c.kind === "cross" ? Math.min(...mats(names, T).map(n => T[n].shape[1])) : null;
  return Object.assign({key, names, rel: names.map(n => rel(n, item.path)), params, kvIn, d: c.d}, c, {ev: c.ev || []});
}
function blockLabel(regs) {
  const k = new Set(regs.map(r => r.kind));
  if (k.has("ssm") && !k.has("attn")) return k.has("moe") ? "SSM+MoE" : "SSM";
  if (k.has("moe") && !k.has("attn") && !k.has("ssm")) return "MoE";
  if (k.has("conv") && k.has("attn")) return "Conformer";
  if (k.has("mod") && k.has("attn")) return "DiT";
  if (k.has("cross")) return "Attn+Cross";
  if (k.has("attn") && k.has("moe")) return "Attn+MoE";
  if (k.has("attn")) return k.has("ffn") ? "Transformer" : "Attention";
  if (k.has("ffn") && !k.has("attn")) return "FFN";
  if (k.has("rnn")) return "RNN";
  if (k.has("conv")) return "Conv";
  return "Block";
}
function roleOf(x, rg, T, lex) {
  const s = T[x].shape, leaf = leafTok(x);
  if (s.length === 1) return /bias/.test(x) ? "bias" : "norm";
  if (rg.kind === "attn" || rg.kind === "cross") { if (s.length === 2 && s[0] === rg.d && s[1] === rg.x) return "out"; if (lex) for (const r of ["q", "k", "v", "o"]) if (LEX[r].test(leaf)) return r === "o" ? "out" : r; if (s.length === 2 && rg.x && s[0] > rg.x) return "qkv"; return "proj"; }
  if (rg.kind === "ffn") { if (s.length === 2 && rg.f && s[1] === rg.f) return "down"; if (lex) { if (LEX.gate.test(leaf)) return "gate"; if (LEX.up.test(leaf)) return "up"; } return "up"; }
  if (rg.kind === "moe") { if (s.length === 3 || /experts?\.\d+/.test(x)) return "expert"; if (s.length === 2 && s[0] === rg.E) return "router"; return "shared"; }
  if (rg.kind === "ssm") { if (isDW(s)) return "conv"; if (s.length === 2 && s[0] > s[1]) return "in"; if (s.length === 2) return "out"; return "state"; }
  return rg.kind;
}
function modalityOf(stem, shapes, C) {
  for (const s of stem) {
    if (s.length === 4 && s[1] === 3) return "image";
    if (s.length === 5 && s[1] === 3) return "video";
    if (s.length === 5 && s[1] >= 4) return "latent";
    if (s.length === 4 && s[1] === 1) return "audio";
    if (s.length === 3 && (s[1] === 80 || s[1] === 128 || s[1] === 1 || s[1] === 64)) return "audio";
    if (s.length === 4 && s[1] >= 4 && s[1] <= 64) return "latent";
  }
  if (C.hasEmbed) return "text";
  if (C.inProj) for (const k of C.inProj) { if ([588, 768, 1176, 3072, 2352].includes(k)) return "image"; if ([1536, 6144].includes(k)) return "image"; }
  return stem.length ? "signal" : "features";
}
function jointOf(names, T, comps) {
  const M = names.map(n => ({n, s: T[n].shape}));
  for (const a of M) for (const b of M) {
    if (a === b || a.s[0] !== b.s[0] || a.s[1] === b.s[1]) continue;
    const j = a.s[0], o = M.find(m => m.s[1] === j && m !== a && m !== b);
    if (o && comps.some(c => c.d === a.s[1])) return {names: [a.n, b.n, o.n], dEnc: a.s[1], dPred: b.s[1], j, out: o.s[0]};
  }
  return null;
}
function hpOf(C, T, out, comps) {
  const hp = {d: C.d, layers: C.layers};
  const regs = C.stacks.flatMap(s => s.types.flatMap(t => t.regions.map(r => Object.assign({}, r, {count: t.count}))));
  const at = regs.find(r => r.kind === "attn" && r.x);
  if (at) {
    const pri = at.x >= 2048 ? [128, 64, 256, 96, 80, 160, 192, 112, 32] : [64, 128, 32, 96, 80, 112, 160, 256];
    const hd = at.qk || pri.find(h => at.x % h === 0 && at.kv % h === 0) || null;
    hp.head_dim = hd; hp.heads = hd ? at.x / hd : null; hp.kv_heads = hd ? at.kv / hd : null; hp.head_dim_from = at.qk ? "q/k-norm width" : "prior";
  }
  const ff = regs.find(r => r.kind === "ffn" && r.f); if (ff) hp.ffn = ff.f;
  const moe = regs.find(r => r.kind === "moe"); if (moe) hp.experts = moe.E;
  const emb = Object.entries(out.tensors).filter(([n, t]) => t.comp === C.id && t.kind === "embed" && T[n].shape.length === 2 && T[n].shape[1] === C.d).map(([n]) => T[n].shape[0]).sort((a, b) => b - a)[0];
  if (emb) hp.vocab = emb;
  else if (!out.timestep) { const e2 = Object.entries(out.tensors).filter(([n, t]) => t.kind === "embed" && T[n].shape.length === 2 && T[n].shape[0] >= 100).map(([n]) => T[n].shape[0]).sort((a, b) => b - a)[0]; if (e2) hp.vocab = e2; }
  hp.tie = emb ? !Object.entries(out.tensors).some(([n, t]) => t.kind === "head" && t.role === "logits" && T[n].shape[0] === emb) : null;
  const attnLayers = regs.filter(r => r.kind === "attn").reduce((a, r) => a + r.count, 0), ssmLayers = regs.filter(r => r.kind === "ssm").reduce((a, r) => a + r.count, 0);
  hp.attn_layers = attnLayers; hp.ssm_layers = ssmLayers;
  if (at && hp.kv_heads != null) hp.kv_bytes_per_token_bf16 = attnLayers * 2 * at.kv * 2;
  if (moe) { const ex = Object.entries(out.tensors).filter(([n, t]) => t.comp === C.id && t.role === "expert").reduce((a, [n]) => a + prod(T[n].shape), 0); hp.expert_params = ex; }
  return hp;
}
function familyOf(o) {
  const b = o.comps.find(c => c.role === "backbone"); if (!b) return "unknown";
  const k = b.kinds || {};
  const enc = o.comps.filter(c => c.role === "encoder");
  if (o.joint) return "transducer";
  if (o.timestep || k.mod && b.modality === "latent") return "diffusion";
  const cross = o.comps.some(c => c.stacks.some(s => s.types.some(t => t.regions.some(r => r.kind === "cross"))));
  if (enc.length && cross) return "encoder-decoder"; // an encoder feeding a decoder through cross-attention, head tied or not
  if (enc.length && b.hasHead) return "prefix-lm";
  if (b.hasHead) return "decoder";
  return "encoder"; // also tied-head decoders when there is no config: shapes cannot tell them apart (familyFromConfig can)
}

// The family as the checkpoint's own config declares it (architectures, is_encoder_decoder), used when a config exists.
function familyFromConfig(I, cfg) {
  if (!cfg || !I) return I ? I.family : null;
  if (/^(transducer|diffusion)$/.test(I.family)) return I.family;
  const archs = [].concat(cfg.architectures || [], (cfg.text_config || {}).architectures || []).join(" ");
  const enc = (I.comps || []).some(c => c.role === "encoder");
  if (cfg.is_encoder_decoder || /Seq2Seq|ForSpeechSeq2Seq/.test(archs)) return "encoder-decoder";
  if (/ForCausalLM|LMHeadModel|ForConditionalGeneration|ForImageTextToText|ForMultimodalLM/.test(archs)) return I.family === "encoder-decoder" ? I.family : enc ? "prefix-lm" : "decoder";
  if (/ForMaskedLM|ForSequenceClassification|ForTokenClassification|ForQuestionAnswering|ForImageClassification|ForPreTraining/.test(archs)) return enc ? I.family : "encoder";
  return I.family;
}


// ------------------------------------------------------------------ 5. configuration fusion and consistency check
// Fields the shapes cannot identify (head count vs head size, top-k) are taken from config.json when it exists. The
// sub-config is matched to a component by width (and depth), and the config is checked against the weights.
const num = (o, ks) => { for (const k of ks) if (typeof o[k] === "number") return o[k]; return null; };
function fuseConfig(I, cfg) {
  if (!cfg || typeof cfg !== "object") return null;
  const subs = []; (function walk(o, path, depth) { if (!o || typeof o !== "object" || Array.isArray(o) || depth > 5) return; subs.push({o, path});
    for (const [k, v] of Object.entries(o)) if (v && typeof v === "object" && !Array.isArray(v)) walk(v, path ? path + "." + k : k, depth + 1); })(cfg, "", 0);
  const W = o => num(o, ["hidden_size", "d_model", "n_embd", "dim", "embed_dim", "hidden_dim", "encoder_embed_dim", "width"]);
  const out = {}, topAll = subs.map(x => num(x.o, ["num_experts_per_tok", "moe_topk", "top_k", "num_selected_experts", "moe_k"])).find(v => v != null) || null;
  for (const C of I.comps) {
    if (!C.d) continue;
    const at = C.stacks.flatMap(st => st.types.flatMap(t => t.regions)).find(r => r.kind === "attn" && r.x);
    let best = null, bs = -1;
    for (const x of subs) { if (W(x.o) !== C.d) continue; let sc = 1; const L = num(x.o, ["num_hidden_layers", "num_layers", "n_layer", "encoder_layers", "decoder_layers"]);
      if (L === C.layers) sc += 2; if (C.role === "backbone" && /(text|llm|language|decoder)/.test(x.path)) sc += 1; if (C.role !== "backbone" && /(vision|audio|encoder)/.test(x.path)) sc += 1; if (sc > bs) { bs = sc; best = x; } }
    if (!best) continue;
    const o = best.o, H = num(o, ["num_attention_heads", "n_head", "num_heads", "encoder_attention_heads", "attention_heads", "decoder_attention_heads"]);
    const KV = num(o, ["num_key_value_heads", "num_kv_heads", "n_kv_heads", "multi_query_group_num"]), hd = num(o, ["head_dim", "attention_head_dim", "kv_channels", "qk_head_dim"]);
    const rec = {path: best.path || "(root)", heads: H, kv_heads: KV || (o.multi_query === true ? 1 : H), head_dim: hd || (H && at ? at.x / H : null), topk: num(o, ["num_experts_per_tok", "moe_topk", "top_k", "num_selected_experts"]) || (C.role === "backbone" ? topAll : null)};
    if (at && H && rec.head_dim) { const h = rec.head_dim; rec.checked = true; rec.consistent = H * h === at.x && (!at.kv || at.kv === at.x || rec.kv_heads * h === at.kv); }
    out[C.id] = rec;
  }
  return out;
}

const api = {infer, normalize, buildTree, classify, fuseConfig, familyFromConfig};
if (typeof module !== "undefined") module.exports = api; else root.ArchInfer = api;
})(typeof window !== "undefined" ? window : globalThis);
