// run infer on the case-study manifests and print a compact anatomy
const fs = require("fs"), path = require("path"), {infer} = require(path.join(__dirname, "../src/infer.js"));
const prod = s => s.reduce((a, b) => a * b, 1);
function quantNVFP4(t, ignore) { const o = {}; for (const [n, v] of Object.entries(t)) { const s = v.shape; if (s.length === 2 && n.endsWith(".weight") && !ignore.test(n) && s[1] % 16 === 0) { const b = n.slice(0, -7); o[n] = {dtype: "U8", shape: [s[0], s[1] / 2]}; o[b + ".weight_scale"] = {dtype: "F8_E4M3", shape: [s[0], s[1] / 16]}; o[b + ".weight_scale_2"] = {dtype: "F32", shape: []}; o[b + ".input_scale"] = {dtype: "F32", shape: []}; } else o[n] = Object.assign({}, v, {dtype: "BF16"}); } return o; }
const cases = {};
const rd = f => JSON.parse(fs.readFileSync(f, "utf8").split("\n")[0]);
cases["parakeet-tdt-0.6b-v3"] = rd("cases/parakeet_tdt_v3.out.jsonl").tensors;
cases["nemotron-3.5-asr-streaming-0.6b"] = rd("cases/nemotron35_asr.out.jsonl").tensors;
cases["Nemotron-3.5-Super-VL-120B-A12B"] = Object.fromEntries(Object.entries(rd("cases/nemotron35_super_vl.out.jsonl").tensors).map(([n, v]) => [n, Object.assign({}, v, {dtype: "BF16"})]));
cases["Nemotron-3-Embed-1B-NVFP4"] = quantNVFP4(rd("cases/nemotron3_embed.out.jsonl").tensors, /embed_tokens|lm_head/);
for (const l of fs.readFileSync("corpus2.jsonl", "utf8").split("\n")) { if (l.includes('"type": "nemo_speechlm"')) { cases["Nemotron-3.5-Transcribe-30B-A4B"] = Object.fromEntries(Object.entries(JSON.parse(l).tensors).map(([n, v]) => [n, Object.assign({}, v, {dtype: "BF16"})])); break; } }
const rec = JSON.parse(fs.readFileSync("cases/reconstructed.json", "utf8"));
cases["Kumo-Tabular (large)"] = rec.kumo_tabular_large; cases["Kumo-Relational"] = rec.kumo_relational;
cases["FastH3-Omni-FP8 (transformer)"] = rec.fasth3_omni_fp8_transformer; cases["FastH3-Omni (BF16 transformer)"] = rec.fasth3_omni_bf16_transformer;
fs.writeFileSync("cases/all_manifests.json", JSON.stringify(cases));
const out = {};
for (const [name, man] of Object.entries(cases)) {
  const t0 = Date.now(), I = infer(man), ms = Date.now() - t0;
  const hdr = Object.entries(man).reduce((a, [n, v]) => a + n.length + 40 + 6 * v.shape.length, 0);
  console.log(`\n## ${name}  tensors=${Object.keys(man).length} params=${(I.totals.params / 1e9).toFixed(3)}B bytes=${(I.totals.bytes / 1e9).toFixed(2)}GB header≈${(hdr / 1024).toFixed(0)}KB family=${I.family} quant=${JSON.stringify(I.quant)} ${ms}ms`);
  for (const c of I.comps) { console.log(`  [${c.role}] ${c.id} d=${c.d} mod=${c.modality} layers=${c.layers} params=${(c.params / 1e6).toFixed(1)}M pattern=${c.pattern.slice(0, 100)}`);
    for (const st of c.stacks) for (const ty of st.types) console.log(`      ${st.path} ×${ty.count} ${ty.label}: ` + ty.regions.filter(r => r.kind !== "norm").map(r => `${r.key}=${r.kind}${r.E ? "(E=" + r.E + ")" : ""}${r.f ? "(f=" + r.f + ")" : ""}${r.x ? "(x=" + r.x + ",kv=" + r.kv + (r.qk ? ",h=" + r.qk : "") + ")" : ""}`).join(" "));
    const out2 = Object.entries(I.tensors).filter(([n, t]) => t.comp === c.id && t.block == null && t.kind !== "norm").map(([n, t]) => n.replace(/\.\d+\./g, ".#.") + ":" + t.kind + (t.role ? "/" + t.role : ""));
    console.log("      outside blocks: " + [...new Set(out2)].slice(0, 12).join(", ")); }
  const loose = Object.entries(I.tensors).filter(([n, t]) => !t.comp).map(([n, t]) => n + ":" + t.kind); if (loose.length) console.log("  no component: " + loose.slice(0, 10).join(", "));
  console.log("  edges: " + JSON.stringify(I.edges) + (I.joint ? " joint=" + JSON.stringify(I.joint) : "") + " hp=" + JSON.stringify(I.hp));
  out[name] = {params: I.totals.params, bytes: I.totals.bytes, family: I.family, comps: I.comps.map(c => ({id: c.id, role: c.role, d: c.d, modality: c.modality, layers: c.layers, params: c.params, pattern: c.pattern})), hp: I.hp, quant: I.quant, joint: I.joint || null, edges: I.edges, hdrKB: hdr / 1024, ms};
}
fs.writeFileSync("res/cases.json", JSON.stringify(out, null, 1));
