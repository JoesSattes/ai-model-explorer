// Analyse the checkpoint benchmark export (bench.html → ame-bench-*.jsonl) with the same recovery code the site runs.
// usage: node realbench.js export.jsonl corpus.jsonl [--out res.json] [--table table.tsv]
// Reference values: parameter totals from the Hub API (safetensors metadata), hyperparameters from config.json,
// and, for model types also in the corpus, per-tensor roles transferred by index-masked name from the class-name ground truth.
"use strict";
const fs = require("fs"), zlib = require("zlib");
const {infer, fuseConfig} = require("../src/infer.js");
const args = process.argv.slice(2), val = f => args.includes(f) ? args[args.indexOf(f) + 1] : null;
const prod = s => s.reduce((a, b) => a * b, 1);
const mask = n => n.split(".").map(t => /^\d+$/.test(t) ? "#" : t).join(".");

// ---- role ground truth by model_type, from the corpus
const gtByType = {};
if (args[1] && fs.existsSync(args[1])) for (const line of fs.readFileSync(args[1], "utf8").split("\n")) {
  if (!line) continue; const r = JSON.parse(line.replace(/(?<=[:\[,]\s*)(-?Infinity|NaN)/g, "null")); if (r.err || !r.type) continue;
  const m = gtByType[r.type] || (gtByType[r.type] = new Map());
  const blockPaths = r.lists.filter(l => !l.expert).map(l => l.path + ".");
  for (const [n, g] of Object.entries(r.gt)) { let k = g.kind; if (k === "conv" && !blockPaths.some(p => n.startsWith(p))) k = "embed"; m.set(mask(n), k); }
}
const normK = (n, k, I) => k === "conv" && I.tensors[n] && I.tensors[n].block == null ? "embed" : k;

// ---- hyperparameters from a config (text sub-config first)
function hp(cfg) {
  if (!cfg) return {};
  const tc = ["text_config", "llm_config", "language_config", "thinker_config", "decoder", "encoder_config"].map(k => cfg[k]).find(x => x && typeof x === "object" && (x.hidden_size || x.d_model || (x.text_config && x.text_config.hidden_size)));
  const src = tc && tc.text_config ? [tc.text_config, tc, cfg] : tc ? [tc, cfg] : [cfg];
  const g = (...ks) => { for (const s of src) for (const k of ks) if (typeof s[k] === "number") return s[k]; return null; };
  return {d: g("hidden_size", "d_model", "n_embd", "dim", "embed_dim"), layers: g("num_hidden_layers", "n_layer", "num_layers", "decoder_layers", "n_layers"),
    heads: g("num_attention_heads", "n_head", "num_heads", "decoder_attention_heads", "n_heads"), kv_heads: g("num_key_value_heads", "num_kv_heads", "n_kv_heads", "multi_query_group_num"),
    head_dim: g("head_dim", "kv_channels", "attention_head_dim"), ffn: g("intermediate_size", "ffn_dim", "d_ff", "n_inner", "decoder_ffn_dim"),
    vocab: g("vocab_size", "padded_vocab_size"), experts: g("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"), model_type: (tc && tc.model_type) || cfg.model_type || null};
}

const rows = [], fails = [];
for (const line of fs.readFileSync(args[0], "utf8").split("\n")) {
  if (!line.trim()) continue; const r = JSON.parse(line);
  if (!r.ok) { fails.push({repo: r.repo, cat: r.cat, err: r.err}); continue; }
  const man = JSON.parse(zlib.gunzipSync(Buffer.from(r.raw, "base64")).toString("utf8"));
  const tensors = {}; for (const [n, [dtype, shape]] of Object.entries(man)) tensors[n] = {dtype, shape};
  const cfgs = JSON.parse(r.cfgs || "{}"), cfg = cfgs["config.json"] || null;
  const row = {repo: r.repo, cat: r.cat, downloads: r.api.downloads, likes: r.api.likes, ntensors: Object.keys(tensors).length, headerBytes: r.read, nfiles: r.nfiles, weightBytes: r.api.weightBytes, apiTotal: r.api.total, apiParams: r.api.parameters};
  let I; const t0 = process.hrtime.bigint();
  try { I = infer(tensors, {lex: true}); } catch (e) { row.err = String(e.message); rows.push(row); continue; }
  row.ms = Number(process.hrtime.bigint() - t0) / 1e6;
  row.params = I.totals.params; row.bytes = I.totals.bytes; row.quant = I.quant && Object.keys(I.quant).length ? Object.keys(I.quant).join("+") : null;
  const stored = Object.values(tensors).reduce((a, v) => a + prod(v.shape), 0); row.stored = stored;
  row.family = I.family; row.comps = I.comps.map(c => `${c.role}:${c.id}:${c.d}:${c.stacks.map(s => s.n + "x" + s.types.map(t => t.label).join("/")).join(",")}`);
  row.labels = [...new Set(I.comps.flatMap(c => c.stacks.flatMap(s => s.types.map(t => t.label))))];
  row.edges = I.edges.length;
  // hyperparameters: manifest only, then fused with the configuration
  const H = hp(cfg); row.mt = H.model_type;
  let F = {}; try { F = fuseConfig(I, cfg) || {}; } catch (e) {}
  const B = I.comps.find(c => c.role === "backbone"), fz = B ? F[B.id] : null;
  row.hp = {}; for (const k of ["d", "layers", "heads", "kv_heads", "head_dim", "ffn", "vocab", "experts"]) {
    const g = H[k]; if (g == null) continue; if (k === "ffn" && H.experts) continue;
    const p = I.hp[k], pf = fz && fz[k] != null ? fz[k] : p; row.hp[k] = [g, p, pf]; }
  row.consistent = fz ? fz.consistent : null; row.checked = fz ? fz.checked : null;
  // roles against transferred ground truth
  const G = H.model_type && gtByType[H.model_type] || cfg && gtByType[cfg.model_type];
  if (G) { let w = 0, wc = 0, n = 0, cov = 0; for (const [nm, v] of Object.entries(tensors)) { const g = G.get(mask(nm)); n++; if (!g || g === "other") continue; cov++; const ww = prod(v.shape) || 1, p = I.tensors[nm] && normK(nm, I.tensors[nm].kind, I); w += ww; if (p === g) wc += ww; }
    row.roles = {coverage: cov / n, acc: w ? wc / w : null}; }
  rows.push(row);
}

// ---- summary
const ok = rows.filter(r => !r.err), S = {exported: rows.length + fails.length, read: rows.length, failedFetch: fails.length, inferErrors: rows.filter(r => r.err).map(r => r.repo)};
const unq = ok.filter(r => !r.quant);
S.paramsExactVsHub = {n: unq.filter(r => r.apiTotal).length, exact: unq.filter(r => r.apiTotal && r.params === r.apiTotal).length, miss: unq.filter(r => r.apiTotal && r.params !== r.apiTotal).map(r => [r.repo, r.params, r.apiTotal])};
S.quantized = ok.filter(r => r.quant).map(r => ({repo: r.repo, fmt: r.quant, logical: r.params, hub: r.apiTotal, ratio: r.apiTotal ? +(r.params / r.apiTotal).toFixed(3) : null}));
const hpAgg = {}; for (const r of ok) for (const [k, [g, p, pf]] of Object.entries(r.hp || {})) { const a = hpAgg[k] || (hpAgg[k] = {n: 0, manifest: 0, fused: 0}); a.n++; if (p === g) a.manifest++; if (pf === g) a.fused++; }
S.hp = hpAgg; S.hpMiss = ok.flatMap(r => Object.entries(r.hp || {}).filter(([, [g, , pf]]) => pf !== g).map(([k, v]) => [r.repo, k, ...v]));
const rl = ok.filter(r => r.roles && r.roles.acc != null && r.roles.coverage >= .5); S.roles = {n: rl.length, macro: rl.reduce((a, r) => a + r.roles.acc, 0) / (rl.length || 1), atLeast99: rl.filter(r => r.roles.acc >= .99).length, low: rl.filter(r => r.roles.acc < .9).map(r => [r.repo, +r.roles.acc.toFixed(3)])};
S.consistency = {checked: ok.filter(r => r.checked).length, consistent: ok.filter(r => r.checked && r.consistent).length, mismatch: ok.filter(r => r.checked && !r.consistent).map(r => r.repo)};
const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
S.header = {medianKB: q(ok.map(r => r.headerBytes), .5) / 1024, p95KB: q(ok.map(r => r.headerBytes), .95) / 1024, maxKB: Math.max(...ok.map(r => r.headerBytes)) / 1024, totalMB: ok.reduce((a, r) => a + r.headerBytes, 0) / 1e6, weightsTB: ok.reduce((a, r) => a + (r.weightBytes || 0), 0) / 1e12};
S.ms = {median: q(ok.map(r => r.ms), .5), p95: q(ok.map(r => r.ms), .95), max: Math.max(...ok.map(r => r.ms))};
S.downloads = ok.reduce((a, r) => a + (r.downloads || 0), 0);
S.families = {}; ok.forEach(r => S.families[r.family] = (S.families[r.family] || 0) + 1);
S.byCat = {}; ok.forEach(r => S.byCat[r.cat] = (S.byCat[r.cat] || 0) + 1);
S.fails = fails;
if (val("--out")) fs.writeFileSync(val("--out"), JSON.stringify({summary: S, rows}, null, 1));
if (val("--table")) fs.writeFileSync(val("--table"), ["repo\tcat\tparams\thub\tfamily\tcomps\tlabels\theaderKB\tms\troleAcc"].concat(ok.map(r => [r.repo, r.cat, r.params, r.apiTotal, r.family, r.comps.length, r.labels.join("/"), (r.headerBytes / 1024).toFixed(1), r.ms.toFixed(1), r.roles && r.roles.acc != null ? r.roles.acc.toFixed(3) : ""].join("\t"))).join("\n"));
console.log(JSON.stringify(S, null, 1));
