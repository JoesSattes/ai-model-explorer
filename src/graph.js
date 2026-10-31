// Graph skeletons for the 3D explorer: ops in execution order, their dataflow edges, blocks and lanes.
// Built from a parsed .tflite (real ops), or from a safetensors header / analysis groups (structure only).

const OP_CAT = {
  FULLY_CONNECTED: "matmul", BATCH_MATMUL: "attn", CONV_2D: "matmul", DEPTHWISE_CONV_2D: "matmul", EMBEDDING_LOOKUP: "matmul",
  SOFTMAX: "attn", GELU: "act", TANH: "act", LOGISTIC: "act", RELU: "act", HARD_SWISH: "act",
  MEAN: "norm", SQUARED_DIFFERENCE: "norm", RSQRT: "norm", SQRT: "norm", SUB: "elem", ADD: "elem", MUL: "elem", DIV: "elem",
  RESHAPE: "shape", TRANSPOSE: "shape", SLICE: "shape", STRIDED_SLICE: "shape", CONCATENATION: "shape", PAD: "shape", GATHER: "shape",
  GATHER_ND: "shape", SQUEEZE: "shape", EXPAND_DIMS: "shape", PACK: "shape", UNPACK: "shape", SPLIT: "shape", BROADCAST_TO: "shape",
  QUANTIZE: "seam", DEQUANTIZE: "seam",
  LESS: "ctrl", LESS_EQUAL: "ctrl", SELECT: "ctrl", SELECT_V2: "ctrl", ONE_HOT: "ctrl", CAST: "ctrl", GREATER: "ctrl", EQUAL: "ctrl",
};
const opCat = t => OP_CAT[t] || "elem";

// "pkg.mod.WhisperEncoderLayer_layers.3;12" -> segments [{cls, attr}]
function scopePath(name) {
  const clean = name.split(";")[0];
  return clean.split("/").slice(1).map(seg => {
    const s = seg.replace(/^(?:[A-Za-z_][\w]*\.)+(?=[A-Z_])/, "");          // drop python module prefix
    const m = s.match(/^([A-Za-z0-9]+?)_(.+)$/);
    return m ? {cls: m[1], attr: m[2]} : {cls: s, attr: ""};
  }).filter(x => x.cls);
}
function layerIndex(path) {
  for (const p of path) {
    let m = p.attr.match(/(?:^|\.)(?:layers|layer|blocks|block|h)\.(\d+)$/); if (m) return +m[1];
    if (/Layer$/.test(p.cls) && /^\d+$/.test(p.attr)) return +p.attr;
    m = p.cls.match(/Layer(\d+)$/); if (m) return +m[1];
  }
  return null;
}
function laneOf(path, type) {
  const s = path.map(p => p.attr + " " + p.cls).join(" ").toLowerCase();
  if (/layer_norm|layernorm|ln_|norm/.test(s)) return "norm";
  if (/encoder_attn|cross/.test(s)) return "xattn";
  if (/self_attn|attention|attn/.test(s)) return "attn";
  if (/fc1|fc2|mlp|ffn|activation|gelu|up_proj|down_proj|gate_proj/.test(s) || type === "GELU") return "ffn";
  return "main";
}
const prod = a => a.reduce((x, y) => x * Math.max(1, y), 1);

function graphFromTflite(p, component, label) {
  const producer = {}; p.ops.forEach(o => o.outputs.forEach(t => producer[t] = o.i));
  const isConst = idx => idx >= 0 && p.buffers[p.tensors[idx].buffer] > 0;
  const nodes = p.ops.map(o => {
    const path = scopePath(o.name);
    let w = null;
    for (const idx of o.inputs) {
      if (idx < 0) continue;
      if (isConst(idx) && opCat(o.type) !== "elem") { const t = p.tensors[idx]; if (prod(t.shape) >= 512) { w = t; break; } }
      const pr = producer[idx]; if (pr != null && p.ops[pr].type === "DEQUANTIZE" && isConst(p.ops[pr].inputs[0])) { w = p.tensors[p.ops[pr].inputs[0]]; break; }
    }
    if (o.type === "DEQUANTIZE" && isConst(o.inputs[0])) w = p.tensors[o.inputs[0]];
    const out = o.outputs.length ? p.tensors[o.outputs[0]].shape : [];
    const ins = o.inputs.filter(i => i >= 0 && producer[i] != null).map(i => producer[i]);
    const ext = o.inputs.filter(i => i >= 0 && producer[i] == null && !isConst(i)).map(i => p.tensors[i].name.replace(/^serving_default_/, "").replace(/:0$/, ""));
    // compute estimate (multiply-adds x2)
    let flops = 0;
    if (w && (o.type === "FULLY_CONNECTED")) flops = 2 * prod(out) * w.shape[w.shape.length - 1];
    else if (w && /CONV/.test(o.type)) flops = 2 * prod(out) * prod(w.shape.slice(1));
    else if (o.type === "BATCH_MATMUL") { const a = p.tensors[o.inputs[0]].shape; flops = 2 * prod(out) * (a[a.length - 1] || 1); }
    else if (o.type === "EMBEDDING_LOOKUP") flops = prod(out);
    else flops = prod(out);
    const wb = w ? (p.buffers[w.buffer] || 0) : 0;
    return {i: o.i, t: o.type, cat: opCat(o.type), path, li: layerIndex(path), lane: laneOf(path, o.type), out, ins, ext, flops,
            w: w ? {n: w.name.split(";")[0].split("/").pop().replace(/^(?:[a-z_]+\.)+/, ""), shape: w.shape, dtype: w.type, bytes: wb, bid: w.buffer} : null};
  });
  // unscoped ops inherit the block of their producer (or the previous op)
  nodes.forEach((n, k) => {
    if (n.li == null && n.path.length <= 1) { const pr = n.ins.length ? nodes[Math.max(...n.ins)] : nodes[k - 1]; if (pr) { n.li = pr.li; if (n.lane === "main") n.lane = pr.lane; n.inherited = true; } }
  });
  // blocks: stem (before first layer) / L0..Ln / head
  let seenLayer = false;
  nodes.forEach(n => {
    if (n.li != null) { seenLayer = true; n.block = component + ".L" + String(n.li).padStart(2, "0"); }
    else n.block = component + (seenLayer ? ".head" : ".stem");
  });
  // keep a stem op that got pulled into a layer by inheritance honest: if it precedes the first real layer op, it's stem
  const firstReal = nodes.findIndex(n => n.li != null && !n.inherited);
  nodes.forEach((n, k) => { if (k < firstReal && n.inherited) n.block = component + ".stem"; });
  const scopes = [], sIdx = {};
  const compact = nodes.map(n => {
    const sc = n.path.map(x => x.attr || x.cls).join("/"); if (!(sc in sIdx)) { sIdx[sc] = scopes.length; scopes.push(sc); }
    return {t: n.t, cat: n.cat, b: n.block, l: n.lane, s: sIdx[sc], o: n.out, in: n.ins, x: n.ext.length ? n.ext : undefined, f: n.flops, w: n.w || undefined};
  });
  return {kind: "ops", name: label, component, scopes, nodes: compact,
          inputs: p.inputs.map(t => ({n: t.name.replace(/^serving_default_/, "").replace(/:0$/, ""), s: t.shape, d: t.type})),
          outputs: p.outputs.map(t => ({n: t.name.replace(/^serving_default_/, "").replace(/:0$/, ""), s: t.shape, d: t.type})),
          bytes: p.bytes};
}

// ---- safetensors header -> structure graph (no ops, every weight tensor is a node) ----
function parseSafetensorsHeader(buf) {
  const dv = new DataView(buf); const n = Number(dv.getBigUint64(0, true));
  const j = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  delete j.__metadata__; return j;
}
const ST_BYTES = {F32: 4, F16: 2, BF16: 2, I8: 1, U8: 1, I32: 4, I64: 8, F64: 8, BOOL: 1, F8_E4M3: 1, F8_E5M2: 1};
function graphFromTensors(tensors, label) {
  // tensors: {name: {dtype, shape}}
  const keyOf = name => {
    const comp = /(^|\.)encoder\./.test(name) ? "encoder" : /(^|\.)decoder\./.test(name) ? "decoder" : "model";
    const m = name.match(/\.(?:layers|layer|blocks|h)\.(\d+)\./); return {comp, li: m ? +m[1] : null};
  };
  const items = Object.entries(tensors).map(([n, v]) => ({n, ...keyOf(n), dtype: v.dtype, shape: v.shape})).filter(x => prod(x.shape) > 1);
  const compOrder = {encoder: 0, model: 1, decoder: 2};
  const phase = x => x.li == null ? (/embed|wte|wpe|conv|patch|positions/.test(x.n) ? 0 : 2) : 1;
  items.sort((a, b) => compOrder[a.comp] - compOrder[b.comp] || phase(a) - phase(b) || (a.li ?? 0) - (b.li ?? 0) || a.n.localeCompare(b.n, "en", {numeric: true}));
  const order = s => /q_proj|query|c_attn|qkv/.test(s) ? 0 : /k_proj|key/.test(s) ? 1 : /v_proj|value/.test(s) ? 2 : /out_proj|o_proj|c_proj|dense/.test(s) && /attn|attention/.test(s) ? 3 : /fc1|up|gate|c_fc/.test(s) ? 5 : /fc2|down/.test(s) ? 6 : 4;
  items.sort((a, b) => compOrder[a.comp] - compOrder[b.comp] || phase(a) - phase(b) || (a.li ?? 0) - (b.li ?? 0) || (/encoder_attn/.test(a.n) - /encoder_attn/.test(b.n)) || order(a.n) - order(b.n) || a.n.localeCompare(b.n));
  const nodes = items.map((x, k) => {
    const path = x.n.split(".").map(a => ({cls: "", attr: a}));
    const block = x.comp + (x.li != null ? ".L" + String(x.li).padStart(2, "0") : phase(x) === 0 ? ".stem" : ".head");
    const params = prod(x.shape), bytes = params * (ST_BYTES[x.dtype] || 4);
    const t = /norm|ln/.test(x.n) ? "NORM" : /embed|wte|wpe/.test(x.n) ? "EMBEDDING" : x.shape.length >= 3 ? "CONV" : x.shape.length === 2 ? "LINEAR" : "BIAS";
    return {t, b: block, l: laneOf(path, ""), s: 0, o: x.shape, in: k ? [k - 1] : [], f: params * 2,
            w: {n: x.n, shape: x.shape, dtype: x.dtype, bytes}, cat: t === "NORM" ? "norm" : t === "BIAS" ? "elem" : "matmul"};
  });
  const comps = [...new Set(items.map(x => x.comp))];
  if (comps.length > 1 && comps.every(c => c !== "model")) {
    // one graph per component (encoder ring + decoder ring), local edge indices
    return comps.map(c => { const idx = items.map((x, k) => x.comp === c ? k : -1).filter(k => k >= 0), loc = {}; idx.forEach((k, i) => loc[k] = i);
      return {kind: "tensors", name: label + " · " + c, component: c, scopes: [""], nodes: idx.map((k, i) => ({...nodes[k], in: i ? [i - 1] : []})), inputs: [], outputs: []}; });
  }
  return [{kind: "tensors", name: label, component: "model", scopes: [""], nodes, inputs: [], outputs: []}];
}
function graphFromViz(v, label) {
  const t = {};
  v.groups.forEach(g => g.subs.forEach(s => { if (s.params) t[g.id.replace(/\.L(\d+)/, ".layers.$1") + "." + s.sub] = {dtype: s.dtype === "FLOAT32" ? "F32" : s.dtype, shape: s.shape}; }));
  return graphFromTensors(t, label);
}

if (typeof module !== "undefined") module.exports = {graphFromTflite, graphFromTensors, graphFromViz, parseSafetensorsHeader, scopePath, opCat};
