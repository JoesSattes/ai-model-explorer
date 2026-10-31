// In-browser model analyzer: TFLite flatbuffers + HF config.json -> one analysis format.
const BYTES_PER = {FLOAT32: 4, FLOAT16: 2, INT32: 4, UINT8: 1, INT64: 8, INT8: 1, INT16: 2, INT4: 0.5, BFLOAT16: 2, FLOAT64: 8, BOOL: 1};

function fbReader(buf) {
  const dv = new DataView(buf), u8 = new Uint8Array(buf);
  const u32 = p => dv.getUint32(p, true), i32 = p => dv.getInt32(p, true), u16 = p => dv.getUint16(p, true);
  const field = (t, slot) => { const vt = t - i32(t); return slot < u16(vt) ? u16(vt + slot) : 0; };
  const ref = p => p + u32(p);
  const tbl = (t, slot) => { const f = field(t, slot); return f ? ref(t + f) : 0; };
  const vec = (t, slot) => { const f = field(t, slot); if (!f) return null; const v = ref(t + f); return {len: u32(v), at: v + 4}; };
  const str = (t, slot) => { const v = vec(t, slot); return v ? new TextDecoder().decode(u8.subarray(v.at, v.at + v.len)) : ""; };
  const sc = (t, slot, kind, dflt = 0) => { const f = field(t, slot); if (!f) return dflt;
    return kind === "i8" ? dv.getInt8(t + f) : kind === "i32" ? i32(t + f) : kind === "u32" ? u32(t + f) :
           kind === "u64" ? u32(t + f) + u32(t + f + 4) * 4294967296 : dv.getUint8(t + f); };
  const tables = v => { const out = []; if (!v) return out; for (let i = 0; i < v.len; i++) out.push(ref(v.at + 4 * i)); return out; };
  const ints = v => { const out = []; if (!v) return out; for (let i = 0; i < v.len; i++) out.push(i32(v.at + 4 * i)); return out; };
  return {u32, field, tbl, vec, str, sc, tables, ints, u8};
}

function parseTflite(buf) {
  const r = fbReader(buf);
  const ident = new TextDecoder().decode(r.u8.subarray(4, 8));
  if (ident !== "TFL3") throw new Error("not a TFLite file (identifier " + JSON.stringify(ident) + ")");
  const m = r.u32(0);
  const codes = r.tables(r.vec(m, 6)).map(c => { const a = r.sc(c, 4, "i8"), b = r.sc(c, 10, "i32"); return TFL_OPS[Math.max(a, b)] || ("OP_" + Math.max(a, b)); });
  const buffers = r.tables(r.vec(m, 12)).map(b => { const d = r.vec(b, 4); return d ? d.len : r.sc(b, 8, "u64"); });
  const sg = r.tables(r.vec(m, 8))[0];
  const tensors = r.tables(r.vec(sg, 4)).map(t => ({name: r.str(t, 10), shape: r.ints(r.vec(t, 4)), type: TFL_TYPES[r.sc(t, 6, "u8")] || "?", buffer: r.sc(t, 8, "u32")}));
  const ops = r.tables(r.vec(sg, 10)).map((o, i) => { const ins = r.ints(r.vec(o, 6)), outs = r.ints(r.vec(o, 8));
    return {i, type: codes[r.sc(o, 4, "u32")], inputs: ins, outputs: outs, name: outs.length ? tensors[outs[0]].name : ""}; });
  const opCounts = {}; ops.forEach(o => opCounts[o.type] = (opCounts[o.type] || 0) + 1);
  return {ops, tensors, buffers, opCounts, bytes: buf.byteLength,
          inputs: r.ints(r.vec(sg, 6)).map(i => tensors[i]), outputs: r.ints(r.vec(sg, 8)).map(i => tensors[i])};
}

// ---- which op owns which weight, and which group it belongs to (port of edgelab.layers) ----
const WEIGHTED = {FULLY_CONNECTED: 1, CONV_2D: 1, EMBEDDING_LOOKUP: 1, DEPTHWISE_CONV_2D: 1, BATCH_MATMUL: 1};
function weightOf(p, op) {
  // a weight = a constant input (has buffer data). DEQUANTIZE in front -> follow it to the stored tensor.
  for (const idx of op.inputs) {
    if (idx < 0) continue;
    let t = p.tensors[idx];
    if (p.buffers[t.buffer] > 0) return t;
    const prod = p.ops.find(o => o.outputs[0] === idx && o.type === "DEQUANTIZE");
    if (prod) { const s = p.tensors[prod.inputs[0]]; if (p.buffers[s.buffer] > 0) return s; }
  }
  return null;
}
function groupOf(name, type, component, seen) {
  const last = (re) => { const m = [...name.matchAll(re)]; return m.length ? m[m.length - 1][1] : null; };
  if (type === "CONV_2D" || type === "DEPTHWISE_CONV_2D") return {group: component + ".conv", sub: "conv" + (last(/conv(\d)/g) || "")};
  if (type === "EMBEDDING_LOOKUP") return {group: "decoder.embed", sub: "embed"};
  if (/Logits/.test(name) || name.startsWith("serving_default_output")) return {group: "decoder.logits", sub: "logits"};
  const kv = last(/CrossKV_(\d+)/g);
  if (kv !== null) return {group: "encoder.cross_kv" + kv, sub: last(/Linear_(\w+?_proj)/g) || "proj"};
  const idxs = [...name.matchAll(/(?:EncoderLayer|DecLayer|DecoderLayer|layers?[._]|blocks?[._]|Layer_)(\d+)/gi)].map(m => +m[1]);
  let sub = last(/Linear_(\w+?_proj|fc1|fc2)/g) || last(/(q_proj|k_proj|v_proj|out_proj|o_proj|fc1|fc2|gate_proj|up_proj|down_proj)/g) || "linear";
  if (!idxs.length) return {group: component + ".other", sub};
  const g = component + ".L" + String(Math.max(...idxs)).padStart(2, "0");
  if (component === "decoder" && (sub === "q_proj" || sub === "out_proj")) {
    const k = g + sub; seen[k] = (seen[k] || 0) + 1; sub = (seen[k] === 1 ? "self_" : "cross_") + sub;
  }
  return {group: g, sub};
}
function analyzeTflite(p, component, label) {
  const rows = [], seenBuf = new Set(), seen = {};
  for (const op of p.ops) {
    if (!WEIGHTED[op.type]) continue;
    const w = weightOf(p, op); if (!w) continue;
    const {group, sub} = groupOf(op.name, op.type, component, seen);
    const params = w.shape.reduce((a, b) => a * b, 1), dup = seenBuf.has(w.buffer); seenBuf.add(w.buffer);
    rows.push({op: op.i, type: op.type, name: op.name, group, sub, shape: w.shape, dtype: w.type, params: dup ? 0 : params,
               bytes: dup ? 0 : p.buffers[w.buffer], buffer: w.buffer});
  }
  const emb = rows.filter(r => r.group === "decoder.embed"), lg = rows.filter(r => r.group === "decoder.logits");
  if (emb.length && lg.length && emb[0].buffer === lg[0].buffer) for (const r of emb.concat(lg)) { r.group = "decoder.tied_embed"; }
  return rows;
}
function vizFromTflite(parts, name) {   // parts: [{component, parsed}]
  const groups = {}, opcounts = {}, files = {};
  for (const {component, parsed} of parts) {
    opcounts[component] = parsed.opCounts; files[component] = parsed.bytes;
    for (const r of analyzeTflite(parsed, component)) {
      const g = groups[r.group] || (groups[r.group] = {id: r.group, component: r.group.split(".")[0], subs: [], params: 0, bytes: 0});
      g.subs.push({tag: r.group + "." + r.sub, sub: r.sub + (r.params ? "" : " (shared)"), shape: r.shape, params: r.params, bytes: r.bytes, dtype: r.dtype, op: r.type});
      g.params += r.params; g.bytes += r.bytes;
    }
  }
  const list = Object.values(groups);
  for (const g of list) { const m = g.id.match(/(\d+)$/); g.kind = /conv$/.test(g.id) ? "conv" : /embed|logits/.test(g.id) ? "embed" : /cross_kv/.test(g.id) ? "cross_kv" : /other$/.test(g.id) ? "other" : "layer"; g.index = m ? +m[1] : -1; }
  const ord = {conv: 0, layer: 1, cross_kv: 2, embed: 3, other: 4};
  list.sort((a, b) => (a.component !== b.component ? (a.component === "encoder" ? -1 : 1) : 0) || ord[a.kind] - ord[b.kind] || a.index - b.index);
  const encL = list.filter(g => g.component === "encoder" && g.kind === "layer"), decL = list.filter(g => g.component === "decoder" && g.kind === "layer");
  const q = list.flatMap(g => g.subs).find(s => /q_proj/.test(s.sub)), fc1 = list.flatMap(g => g.subs).find(s => /fc1/.test(s.sub));
  const emb = list.find(g => g.kind === "embed");
  const conv1 = list.flatMap(g => g.subs).find(s => /conv1/.test(s.sub));
  const meta = {name, source: "tflite", encoder_layers: encL.length, decoder_layers: decL.length,
    d_model: q ? q.shape[0] : null, ffn: fc1 ? fc1.shape[0] : null, vocab: emb ? emb.subs[0].shape[0] : null,
    n_mels: conv1 ? conv1.shape[conv1.shape.length - 1] : 80, files};
  return {meta, groups: list, opcounts, sensitivity: null, flow: null, aihub: null};
}

// ---- HF config.json -> exact structure (no weights needed) ----
function vizFromConfig(cfg, name) {
  const d = cfg.d_model, fe = cfg.encoder_ffn_dim, fd = cfg.decoder_ffn_dim, nm = cfg.num_mel_bins || 80, V = cfg.vocab_size;
  const groups = [], sub = (tag, s, shape) => ({tag, sub: s, shape, params: shape.reduce((a, b) => a * b, 1)});
  const g = (id, component, kind, index, subs) => groups.push({id, component, kind, index, subs, params: subs.reduce((a, s) => a + s.params, 0)});
  g("encoder.conv", "encoder", "conv", -1, [sub("encoder.conv.conv1", "conv1", [d, 3, nm]), sub("encoder.conv.conv2", "conv2", [d, 3, d])]);
  for (let i = 0; i < cfg.encoder_layers; i++) { const L = `encoder.L${String(i).padStart(2, "0")}`;
    g(L, "encoder", "layer", i, [["q_proj", [d, d]], ["k_proj", [d, d]], ["v_proj", [d, d]], ["out_proj", [d, d]], ["fc1", [fe, d]], ["fc2", [d, fe]]].map(([s, sh]) => sub(`${L}.${s}`, s, sh))); }
  for (let i = 0; i < cfg.decoder_layers; i++) g(`encoder.cross_kv${i}`, "encoder", "cross_kv", i, ["k_proj", "v_proj"].map(s => sub(`encoder.cross_kv${i}.${s}`, s, [d, d])));
  for (let i = 0; i < cfg.decoder_layers; i++) { const L = `decoder.L${String(i).padStart(2, "0")}`;
    g(L, "decoder", "layer", i, [["self_q_proj", [d, d]], ["k_proj", [d, d]], ["v_proj", [d, d]], ["self_out_proj", [d, d]], ["cross_q_proj", [d, d]], ["cross_out_proj", [d, d]], ["fc1", [fd, d]], ["fc2", [d, fd]]].map(([s, sh]) => sub(`${L}.${s}`, s, sh))); }
  g("decoder.tied_embed", "decoder", "embed", -1, [sub("decoder.tied_embed.embed", "embed + logits (one buffer)", [V, d])]);
  groups.forEach(x => { x.bytes = x.params * 4; x.subs.forEach(s => { s.bytes = s.params * 4; s.dtype = "FLOAT32"; }); });
  return {meta: {name, source: "config", d_model: d, ffn: fe, encoder_layers: cfg.encoder_layers, decoder_layers: cfg.decoder_layers, vocab: V, n_mels: nm,
                 heads: cfg.encoder_attention_heads, max_target: cfg.max_target_positions},
          groups, opcounts: null, sensitivity: null, flow: null, aihub: null};
}
if (typeof module !== "undefined") module.exports = {parseTflite, analyzeTflite, vizFromTflite, vizFromConfig};
