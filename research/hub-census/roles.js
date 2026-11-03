// Per-tensor role accuracy on real Hub checkpoints, by label transfer from the corpus.
// For every census repo whose config model_type is also in the corpus, the class-name labels of that type are carried
// over by index-masked tensor name (model.layers.#.self_attn.q_proj.weight -> attn). Repos where fewer than half the
// tensors find a label are not scored. Baselines: v1 name rules with MoE, and the llama.cpp tensor table.
// usage: node roles.js ../data/corpus-510.jsonl.gz out/census_manifests.jsonl.gz out/census.jsonl > out/roles.json
// Method: S. Singkul, "Reading the Architecture off the Weights: Manifest-Only Model Anatomy and a Census of 12,974 Hugging Face Repositories", 2026.
"use strict";
const fs = require("fs"), zlib = require("zlib"), path = require("path"), readline = require("readline");
const {infer} = require(path.join(__dirname, "../../src/infer.js"));
const BL = require(path.join(__dirname, "../baselines.js"));
const [corpusPath, manPath, censusPath] = process.argv.slice(2);
const prod = s => s.reduce((a, b) => a * b, 1);
const mask = n => n.split(".").map(t => /^\d+$/.test(t) ? "#" : t).join(".");
const lines = p => (p.endsWith(".gz") ? zlib.gunzipSync(fs.readFileSync(p)) : fs.readFileSync(p)).toString("utf8").split("\n").filter(Boolean);
const parse = l => { try { return JSON.parse(l); } catch (e) { return JSON.parse(l.replace(/(?<=[:,\[] ?)-?(Infinity|NaN)\b/g, "null")); } };

const gtByType = {};
for (const l of lines(corpusPath)) { const r = parse(l); if (r.err || !r.type) continue;
  const m = gtByType[r.type] || (gtByType[r.type] = new Map()), blocks = (r.lists || []).filter(x => !x.expert).map(x => x.path + ".");
  for (const [n, g] of Object.entries(r.gt)) { let k = g.kind; if (k === "conv" && !blocks.some(p => n.startsWith(p))) k = "embed"; m.set(mask(n), k); } }

const meta = {}; for (const l of lines(censusPath)) { const r = parse(l); meta[r.id] = {stratum: r.stratum, origin: r.origin, frontier: !!r.also_frontier, created: r.created_at, downloads: r.downloads}; }
const FIX = /internal-testing|(?:^|[\/_-])tiny[-_]|random|dummy|unit-test|^unslothai\//i;

(async () => {
  const rows = [];
  const rl = readline.createInterface({input: fs.createReadStream(manPath).pipe(zlib.createGunzip()), crlfDelay: Infinity});
  for await (const l of rl) {
    if (!l) continue; const m = parse(l), cfg = m.config || {}, mt = cfg.model_type || (cfg.text_config || {}).model_type;
    const G = mt && gtByType[mt]; if (!G || FIX.test(m.id)) continue;
    const T = {}; for (const [n, [dtype, shape]] of Object.entries(m.tensors)) T[n] = {dtype, shape};
    let I; try { I = infer(T, {lex: true}); } catch (e) { rows.push({id: m.id, type: mt, err: String(e.message).slice(0, 120)}); continue; }
    const normK = (n, k) => k === "conv" && I.tensors[n] && I.tensors[n].block == null ? "embed" : k;
    const score = pred => { let w = 0, wc = 0, n = 0, cov = 0; for (const [nm, v] of Object.entries(T)) { n++; const g = G.get(mask(nm)); if (!g || g === "other") continue; cov++;
      const ww = prod(v.shape) || 1, p = pred(nm); w += ww; if (p === g) wc += ww; } return {acc: w ? wc / w : null, coverage: n ? cov / n : 0}; };
    const ours = score(nm => I.tensors[nm] && normK(nm, I.tensors[nm].kind));
    if (ours.coverage < 0.5) continue;
    const v1 = BL.v1moe(T), gg = BL.ggufKinds(T), base = P => nm => P[nm] === "conv" && !/\.(layers|layer|h|blocks|block)\.\d+\./.test(nm) ? "embed" : P[nm];
    rows.push({id: m.id, type: mt, ...meta[m.id], acc: ours.acc, coverage: ours.coverage, v1moe: score(base(v1)).acc, gguf: score(base(gg)).acc});
  }
  const ok = rows.filter(r => r.acc != null), mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
  const sum = set => ({repos: set.length, types: new Set(set.map(r => r.type)).size, macro: mean(set.map(r => r.acc)),
    share_99: set.length ? set.filter(r => r.acc >= 0.99).length / set.length : null, v1moe: mean(set.map(r => r.v1moe ?? 0)), gguf: mean(set.map(r => r.gguf ?? 0))});
  const out = {all: sum(ok), originals: sum(ok.filter(r => r.stratum === "top" && r.origin === "original")),
    frontier: sum(ok.filter(r => r.frontier)), released_2026: sum(ok.filter(r => r.created && r.created >= "2026")),
    quantized_copies: sum(ok.filter(r => r.origin === "quantized")), errors: rows.filter(r => r.err).length,
    worst_types: Object.entries(ok.reduce((a, r) => { (a[r.type] = a[r.type] || []).push(r.acc); return a; }, {}))
      .map(([t, v]) => ({type: t, repos: v.length, macro: mean(v)})).filter(x => x.repos >= 5).sort((a, b) => a.macro - b.macro).slice(0, 12),
    rows: ok};
  process.stdout.write(JSON.stringify(out));
})();
