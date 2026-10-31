// evaluate infer.js against the meta-device corpus. usage: node eval.js corpus.jsonl [--nolex] [--obf] [--v1] [--quant fmt] [--only a,b] [--dump]
const fs = require("fs"), path = require("path"), vm = require("vm");
const {infer} = require("/home/claude/ai-model-explorer/src/infer.js");
const args = process.argv.slice(2), file = args[0], has = f => args.includes(f), val = f => args.includes(f) ? args[args.indexOf(f) + 1] : null;
const only = val("--only") ? val("--only").split(",") : null;
// v1 baseline: the explorer's previous regex placement
const ctx = {console, Math, Object, Array, String, Number, Set, Map, JSON}; vm.createContext(ctx);
vm.runInContext(fs.readFileSync("/home/claude/ai-model-explorer/src/model.js", "utf8") + "\n;this.tensorPlace=tensorPlace;this.TOWER=TOWER;", ctx);
function v1kinds(tensors) {
  const names = Object.keys(tensors), lay = n => /\.(?:layers|layer|h|blocks|block)\.\d+\./.test(n);
  const both = names.some(n => ctx.TOWER.test(n.toLowerCase()) && lay(n)) && names.some(n => !ctx.TOWER.test(n.toLowerCase()) && lay(n));
  const o = {};
  for (const n of names) { const p = ctx.tensorPlace(n, both || true, both || true);
    let k = "other";
    if (p.layer != null) k = p.role === "ln" ? "norm" : {self: "attn", cross: "cross", ffn: "ffn", linear: "ssm"}[p.region] || "other";
    else if (p.stem) k = p.role === "conv" ? "conv" : p.role === "ln" ? "norm" : "embed";
    else k = p.role === "ln" ? "norm" : p.role === "logits" || p.role === "cls" || p.role === "pool" ? "head" : p.role === "proj" ? "adapter" : "other";
    o[n] = k; }
  return o;
}
const H = s => { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return "t" + (h >>> 0).toString(36); };
function obf(tensors) { const m = {}, back = {}; for (const [n, v] of Object.entries(tensors)) { const o = n.split(".").map(t => /^\d+$/.test(t) ? t : H(t)).join("."); m[o] = v; back[o] = n; } return {m, back}; }
// synthetic quantized storage of a manifest (NVFP4 / FP8 / GPTQ) for every 2-D linear weight
function quantize(tensors, fmt) {
  const o = {};
  for (const [n, v] of Object.entries(tensors)) {
    const s = v.shape, lin = s.length === 2 && n.endsWith(".weight") && s[1] % 32 === 0 && s[0] % 8 === 0 && !/embed|wte|wpe|position|lm_head|norm/.test(n);
    if (!lin) { o[n] = v; continue; }
    const b = n.slice(0, -7);
    if (fmt === "nvfp4") { o[n] = {dtype: "U8", shape: [s[0], s[1] / 2]}; o[b + ".weight_scale"] = {dtype: "F8_E4M3", shape: [s[0], s[1] / 16]}; o[b + ".weight_scale_2"] = {dtype: "F32", shape: []}; o[b + ".input_scale"] = {dtype: "F32", shape: []}; }
    else if (fmt === "fp8") { o[n] = {dtype: "F8_E4M3", shape: s}; o[b + ".weight_scale"] = {dtype: "F32", shape: [s[0], 1]}; }
    else if (fmt === "gptq") { o[b + ".qweight"] = {dtype: "I32", shape: [s[1] / 8, s[0]]}; o[b + ".qzeros"] = {dtype: "I32", shape: [s[1] / 128, s[0] / 8]}; o[b + ".scales"] = {dtype: "F16", shape: [s[1] / 128, s[0]]}; o[b + ".g_idx"] = {dtype: "I32", shape: [s[1]]}; }
  }
  return o;
}
const K = ["attn", "cross", "ffn", "moe", "ssm", "conv", "rnn", "mod", "embed", "norm", "head", "adapter"];
const agg = {hp1: {}, hpSrc: {}, hdr: [], models: 0, w: 0, wc: 0, n: 0, nc: 0, perK: {}, conf: {}, lists: {gt: 0, hit: 0, pred: 0, cnt: 0, types: 0, typesExact: 0}, hp: {}, params: {n: 0, exact: 0}, fail: [], perModel: []};
const prod = s => s.reduce((a, b) => a * b, 1);
const lines = fs.readFileSync(file, "utf8").split("\n").filter(Boolean);
const seen = new Set();
for (const line of lines) {
  const r = JSON.parse(line); if (r.err || (only && !only.includes(r.type))) continue;
  const key = Object.entries(r.tensors).map(([n, v]) => n + v.shape).join("|"); if (seen.has(key)) continue; seen.add(key);
  let man = r.tensors, back = null;
  if (val("--quant")) man = quantize(man, val("--quant"));
  if (has("--obf")) { const o = obf(man); man = o.m; back = o.back; }
  let pred, I = null;
  try {
    if (has("--v1")) pred = v1kinds(man);
    else { const t0 = process.hrtime.bigint(); I = infer(man, {lex: !has("--nolex")}); agg.ms = agg.ms || []; agg.ms.push(Number(process.hrtime.bigint() - t0) / 1e6); pred = {}; for (const [n, t] of Object.entries(I.tensors)) pred[n] = t.kind; }
  } catch (e) { agg.fail.push(r.type + ": " + e.message); continue; }
  agg.models++; { const hb = Object.entries(r.tensors).reduce((a, [n, v]) => a + n.length + 40 + 6 * v.shape.length, 0) + 8, wb = Object.values(r.tensors).reduce((a, v) => a + prod(v.shape) * 2, 0); agg.hdr.push([hb, wb]); }
  const name = n => back ? back[n] : n;
  const P = {}; for (const [n, k] of Object.entries(pred)) P[name(n).replace(/\.qweight$/, ".weight")] = k;
  // quantized manifests: map back aux-free names
  let mw = 0, mwc = 0;
  // label space: an input stem (token/patch/position tables and conv stems outside the block stacks) counts as 'embed'
  const blockPaths = r.lists.filter(l => !l.expert).map(l => l.path + ".");
  const inBlock = n => blockPaths.some(p => n.startsWith(p));
  const norm = (n, k) => k === "conv" && !inBlock(n) ? "embed" : k;
  for (const [n, g0] of Object.entries(r.gt)) {
    const g = {kind: norm(n, g0.kind)};
    if (g.kind === "other") continue; const w = prod(r.tensors[n].shape) || 1; let p = P[n] === undefined ? undefined : norm(n, P[n]);
    if (p === undefined && val("--quant")) p = P[n.replace(/weight$/, "weight")];
    agg.w += w; agg.n++; mw += w; const ok = p === g.kind; if (ok) { agg.wc += w; agg.nc++; mwc += w; }
    const pk = agg.perK[g.kind] || (agg.perK[g.kind] = {tp: 0, fn: 0, fp: 0, w: 0, wc: 0}); pk.w += w; if (ok) { pk.tp++; pk.wc += w; } else { pk.fn++; const fk = agg.perK[p] || (agg.perK[p] = {tp: 0, fn: 0, fp: 0, w: 0, wc: 0}); fk.fp++; }
    const ck = g.kind + "→" + p; agg.conf[ck] = (agg.conf[ck] || 0) + w; agg.confN = agg.confN || {}; agg.confN[ck] = (agg.confN[ck] || 0) + 1; if (!ok) { agg.ex = agg.ex || {}; (agg.ex[ck] = agg.ex[ck] || []).length < 6 && agg.ex[ck].push(r.type + ":" + n); }
  }
  const row = {type: r.type, acc: mw ? mwc / mw : null, params: r.params};
  // block lists (GT innermost non-expert lists whose items hold >= 2 parameterised sub-modules)
  if (I) {
    const nonexp = r.lists.filter(l => !l.expert);
    const subl = new Set(), outer = new Set();
    for (const O of nonexp) { if (O.inner) continue; const ins = []; for (let i = 0; i < O.n; i++) { const a = nonexp.filter(L => L.inner && L.path.startsWith(O.path + "." + i + ".") && L.path.split(".").length === O.path.split(".").length + 2); ins.push(a); }
      const short = ins.every(a => a.length >= 1 && a.every(L => L.n <= 4)) && new Set(ins.flatMap(a => a.map(L => L.n))).size === 1;
      const hetero = ins.every(a => a.length === 1 && new Set(a[0].types).size === a[0].n), multi = ins.every(a => a.length >= 2);
      if (short && (hetero || multi)) { ins.forEach(a => a.forEach(L => subl.add(L.path))); outer.add(O.path); } }
    const gtL = nonexp.filter(l => (l.inner || outer.has(l.path)) && !subl.has(l.path)).filter(l => { const pre = l.path + ".0."; const mods = new Set(Object.keys(r.tensors).filter(n => n.startsWith(pre)).map(n => n.slice(0, n.lastIndexOf(".")))); return mods.size >= 2; });
    const predS = I.comps.flatMap(c => c.stacks); const pmap = new Map(predS.map(s => [back ? back[s.items[0] + ".x"] ? null : null : s.path, s]));
    const pathOf = s => { if (!back) return s.path; const any = Object.keys(back).find(o => o.startsWith(s.path + ".")); const real = back[any]; return real.split(".").slice(0, s.path.split(".").length).join("."); };
    const pm = new Map(predS.map(s => [pathOf(s), s]));
    agg.lists.gt += gtL.length; agg.lists.pred += predS.length;
    let hit = 0; for (const l of gtL) { const s = pm.get(l.path); if (!s) continue; hit++; agg.lists.hit++;
      if (s.n === l.n) agg.lists.cnt++;
      const a = l.types, b = s.typeOf; agg.lists.types++;
      if (a.length === b.length) { const m1 = new Map(), m2 = new Map(); let ok = true; a.forEach((x, i) => { if (m1.has(x) && m1.get(x) !== b[i]) ok = false; if (m2.has(b[i]) && m2.get(b[i]) !== x) ok = false; m1.set(x, b[i]); m2.set(b[i], x); }); if (ok) agg.lists.typesExact++; else row.typeMiss = (row.typeMiss || []).concat(l.path); }
    }
    row.lists = [gtL.length, hit, predS.length];
    // hyperparameters
    for (const k of ["d", "layers", "heads", "kv_heads", "head_dim", "ffn", "vocab", "experts"]) {
      const g = r.hp[k], p = I.hp[k]; if (g == null || typeof g !== "number") continue; if (k === "ffn" && r.hp.experts) continue; if (k === "kv_heads" && g == null) continue;
      const a = agg.hp[k] || (agg.hp[k] = {n: 0, ok: 0}); a.n++; if (p === g) a.ok++; else (row.hpMiss = row.hpMiss || {})[k] = [g, p];
      if (I.comps.length === 1) { const b = agg.hp1[k] || (agg.hp1[k] = {n: 0, ok: 0}); b.n++; if (p === g) b.ok++; }
      if (k === "heads" || k === "head_dim") { const src = I.hp.head_dim_from || "none", b = agg.hpSrc[k + ":" + src] || (agg.hpSrc[k + ":" + src] = {n: 0, ok: 0}); b.n++; if (p === g) b.ok++; }
    }
    agg.params.n++; if (Math.abs(I.totals.params - r.params) / r.params < 1e-6) agg.params.exact++; else row.paramMiss = [r.params, I.totals.params];
    row.family = I.family; row.comps = I.comps.map(c => c.id + ":" + c.role + ":" + c.d).join(" ");
  }
  agg.perModel.push(row);
}
const f1 = k => { const p = agg.perK[k]; if (!p) return null; const pr = p.tp / Math.max(1, p.tp + p.fp), rc = p.tp / Math.max(1, p.tp + p.fn); return pr + rc ? 2 * pr * rc / (pr + rc) : 0; };
const pm = agg.perModel.filter(r => r.acc != null), macro = pm.reduce((a, r) => a + r.acc, 0) / pm.length;
const res = {models: agg.models, macro_acc: macro, failed: agg.fail.length, acc_w: agg.wc / agg.w, acc_n: agg.nc / agg.n, f1: Object.fromEntries(K.map(k => [k, f1(k)]).filter(x => x[1] != null)),
  wacc: Object.fromEntries(K.filter(k => agg.perK[k] && agg.perK[k].w).map(k => [k, agg.perK[k].wc / agg.perK[k].w])),
  lists: agg.lists, hp: Object.fromEntries(Object.entries(agg.hp).map(([k, v]) => [k, [v.ok, v.n, +(v.ok / v.n).toFixed(3)]])), params: agg.params,
  topConf: Object.entries(agg.conf).filter(([k]) => k.split("→")[0] !== k.split("→")[1]).sort((a, b) => b[1] - a[1]).slice(0, 15).map(([k, v]) => [k, +(v / agg.w * 100).toFixed(2)])};
res.hp1 = Object.fromEntries(Object.entries(agg.hp1).map(([k, v]) => [k, [v.ok, v.n, +(v.ok / v.n).toFixed(3)]])); res.hpSrc = Object.fromEntries(Object.entries(agg.hpSrc).map(([k, v]) => [k, [v.ok, v.n, +(v.ok / v.n).toFixed(3)]]));
{ const h = agg.hdr.map(x => x[0]).sort((a, b) => a - b), q = agg.hdr.map(x => x[0] / x[1]).sort((a, b) => a - b); res.header = {medianKB: h[h.length >> 1] / 1024, p95KB: h[Math.floor(h.length * .95)] / 1024, maxKB: h[h.length - 1] / 1024, medianRatio: q[q.length >> 1]}; }
if (agg.ms) { const m = agg.ms.sort((a, b) => a - b); res.ms = {median: m[m.length >> 1], p95: m[Math.floor(m.length * .95)], max: m[m.length - 1]}; }
res.listsPR = {recall: agg.lists.hit / agg.lists.gt, precision: agg.lists.hit / agg.lists.pred, countExact: agg.lists.cnt / agg.lists.hit, typesExact: agg.lists.typesExact / agg.lists.types};
res.families = {}; res.labels = {}; agg.perModel.forEach(r => { if (r.family) res.families[r.family] = (res.families[r.family] || 0) + 1; });
res.topConfN = Object.entries(agg.confN).filter(([k]) => k.split("→")[0] !== k.split("→")[1]).sort((a, b) => b[1] - a[1]).slice(0, +(process.env.TOPN||25)).map(([k, v]) => [k, v, (agg.ex[k] || []).slice(0, 4)]);
console.log(JSON.stringify(res, null, 1));
if (has("--dump")) fs.writeFileSync(val("--dump"), JSON.stringify({res, perModel: agg.perModel, fail: agg.fail}, null, 1));
