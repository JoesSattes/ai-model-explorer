// Baselines used in the paper (shared by eval.js and realbench.js): the explorer's previous name rules (v1), v1 with a
// MoE label, llama.cpp's tensor-name mapping, and the layer-index regex for stacks.
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
// v1 baseline: the explorer's previous regex placement
const ctx = {console, Math, Object, Array, String, Number, Set, Map, JSON}; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "src", "model.js"), "utf8") + "\n;this.tensorPlace=tensorPlace;this.TOWER=TOWER;", ctx);
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
// v1 with a MoE label: the same rules, but expert, router and shared-expert tensors are labelled MoE (a fairer label space)
function v1moe(tensors) { const o = v1kinds(tensors); for (const n of Object.keys(o)) if (/(^|\.)(experts?|block_sparse_moe|shared_experts?|router)(\.|$)|\.mlp\.gate\.(weight|bias)$|e_score_correction/.test(n)) o[n] = "moe"; return o; }
// llama.cpp baseline: the Hugging Face module paths in gguf-py's tensor_mapping.py, looked up by name (union over architectures)
const GG = (() => { const p = path.join(__dirname, "gguf_roles.json"); if (!fs.existsSync(p)) return null; const d = JSON.parse(fs.readFileSync(p, "utf8")), m = new Map();
  for (const [, role, , pat] of d.patterns) { const k = pat.replace(/\{(bid|i|)\}/g, "#").split(".").map(t => /^\d+$/.test(t) ? "#" : t).join("."); if (!m.has(k)) m.set(k, role); } return m; })();
function ggufKinds(tensors) { const o = {}, mk = n => n.split(".").map(t => /^\d+$/.test(t) ? "#" : t).join(".");
  for (const n of Object.keys(tensors)) { const full = mk(n), parts = full.split("."), mod = parts.slice(0, -1).join(".");
    let r = GG.get(full) || GG.get(mod);
    if (!r && /\.experts\.#\./.test(mod)) r = GG.get(mod.replace(/\.experts\.#\./, ".experts.")); // convert_hf_to_gguf merges per-expert tensors before the lookup
    o[n] = r || "other"; }
  return o; }
// stack baseline: every "<prefix>.(layers|layer|h|blocks|block|resblocks).<i>." is one stack of max(i)+1 identical blocks
function regexStacks(tensors) { const st = new Map(); for (const n of Object.keys(tensors)) { const m = n.match(/^(.*?(?:^|\.)(?:layers|layer|h|blocks|block|resblocks))\.(\d+)\./); if (!m) continue; const p = m[1].replace(/^\./, ""); st.set(p, Math.max(st.get(p) || 0, +m[2] + 1)); }
  return [...st].map(([path, n]) => ({path, n})); }

module.exports = {v1kinds, v1moe, ggufKinds, regexStacks};
