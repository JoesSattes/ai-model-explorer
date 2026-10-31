// bench.js — run the explorer's own loader and recovery over popular Hugging Face checkpoints.
// Each result (repo metadata, config files, and the tensor manifest, gzip-compressed) is kept in this browser (IndexedDB)
// and can be downloaded as one JSON-lines file.
"use strict";
const V = 1, CHUNK = 190000, CONC = 3;
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = n => n == null ? "—" : n >= 1e12 ? (n / 1e12).toFixed(2) + " T" : n >= 1e9 ? (n / 1e9).toFixed(2) + " B" : n >= 1e6 ? (n / 1e6).toFixed(1) + " M" : n >= 1e3 ? (n / 1e3).toFixed(1) + " K" : String(n);
const idOf = r => r.replace(/[^\w.-]+/g, "__");
let stop = false;
const IDB = (() => { let p; const open = () => p || (p = new Promise((res, rej) => { const q = indexedDB.open("mx-bench", 1); q.onupgradeneeded = () => q.result.createObjectStore("r"); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); }));
  const tx = async (mode, f) => { const d = await open(); return new Promise((res, rej) => { const t = d.transaction("r", mode), st = t.objectStore("r"), out = f(st); t.oncomplete = () => res(out && out.result); t.onerror = () => rej(t.error); }); };
  return {put: (k, v) => tx("readwrite", s => s.put(v, k)), all: () => tx("readonly", s => s.getAll()), clear: () => tx("readwrite", s => s.clear())}; })();
const rows = {};

async function gz(str) {
  const cs = new Blob([str]).stream().pipeThrough(new CompressionStream("gzip"));
  const buf = new Uint8Array(await new Response(cs).arrayBuffer()); let bin = "";
  for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  return btoa(bin);
}
async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(`HTTP ${r.status} ${url.split("/resolve/main/")[1] || ""}`); return r.json(); }

function row(repo, cat) {
  if (!rows[repo]) { const tr = document.createElement("tr"); tr.innerHTML = `<td>${esc(cat)}</td><td class="repo">${esc(repo)}</td><td class="st">waiting</td><td class="n"></td><td class="n"></td><td></td><td class="n"></td>`; $("#tb").appendChild(tr); rows[repo] = tr; }
  return rows[repo];
}
function setRow(repo, st, cls, cells) { const tr = row(repo, ""); tr.className = cls || ""; tr.children[2].textContent = st; if (cells) cells.forEach((c, i) => tr.children[3 + i].textContent = c); }

async function one(cat, repo) {
  const t0 = performance.now(); setRow(repo, "listing", "run");
  const info = await getJSON(`https://huggingface.co/api/models/${repo}?blobs=true`);
  if (info.error) throw new Error(info.error);
  const files = (info.siblings || []).map(s => ({n: s.rfilename, size: s.size || 0})), base = `https://huggingface.co/${repo}/resolve/main/`;
  const has = n => files.some(f => f.n === n);
  const cfgs = {};
  for (const n of ["config.json", "model_index.json", "params.json"]) if (has(n)) try { cfgs[n] = await getJSON(base + n); } catch (e) {}
  for (const f of files) if (/^[\w.-]+\/config\.json$/.test(f.n) && Object.keys(cfgs).length < 10) try { cfgs[f.n] = await getJSON(base + f.n); } catch (e) {}
  setRow(repo, "headers", "run");
  const mf = await Loaders.hfManifest(repo, files, (st, n, i, k) => setRow(repo, st === "index" ? "index" : `header ${i}/${k}`, "run"), {prefer: cfgs["config.json"] && cfgs["config.json"].default_size, conc: 8});
  const t1 = performance.now();
  if (!mf) throw new Error("no safetensors / pt weights");
  const man = {}; for (const [n, v] of Object.entries(mf.tensors)) man[n] = [v.dtype, v.shape];
  let I = null, ms = null, ierr = null;
  try { const a = performance.now(); I = ArchInfer.infer(mf.tensors, {lex: true}); ms = performance.now() - a; } catch (e) { ierr = String(e && e.message || e); }
  const raw = await gz(JSON.stringify(man)), nch = 1;
  const cfgStr = JSON.stringify(cfgs);
  const st = info.safetensors || null;
  const rec = {v: V, repo, cat, ok: true, ts: Date.now(), nch, raw, cfgs: cfgStr,
    api: {total: st && st.total || null, parameters: st && st.parameters || null, downloads: info.downloads || 0, likes: info.likes || 0, pipeline: info.pipeline_tag || null, library: info.library_name || null, created: info.createdAt || null, weightBytes: files.filter(f => /\.(safetensors|pt|pth|bin|ckpt)$/.test(f.n)).reduce((a, f) => a + f.size, 0)},
    read: mf.read, nfiles: mf.nfiles, kinds: mf.kinds, ntensors: Object.keys(man).length, msFetch: Math.round(t1 - t0), msInfer: ms, ierr,
    quick: I ? {params: I.totals.params, family: I.family, comps: I.comps.length, d: I.hp.d, layers: I.hp.layers} : null};
  await IDB.put(repo, rec);
  return rec;
}

async function run(onlyFailed) {
  stop = false; $("#go").disabled = true; $("#retry").disabled = true;
  const done = {}; try { (await IDB.all()).forEach(x => done[x.repo] = x); } catch (e) {}
  const todo = BENCH_MODELS.filter(([c, r]) => { const d = done[r]; if (d && d.v === V && d.ok) { const q = d.quick || {}; setRow(r, "done", "ok", [fmt(q.params), fmt(d.api.total), q.family || "", fmt(d.read)]); return false; } if (d && !d.ok && !onlyFailed) { setRow(r, "failed: " + d.err, "bad"); return false; } return true; });
  let next = 0, nOk = Object.values(done).filter(d => d.ok).length, nBad = 0;
  const upd = () => $("#prog").textContent = `${nOk} done · ${nBad} failed this run · ${todo.length - next} left`;
  upd();
  const work = async () => { while (next < todo.length && !stop) { const [cat, repo] = todo[next++]; upd();
    try { const r = await one(cat, repo); nOk++; const q = r.quick || {}; setRow(repo, r.ierr ? "read, infer error" : "done", r.ierr ? "warn" : "ok", [fmt(q.params), fmt(r.api.total), q.family || r.ierr || "", fmt(r.read)]); }
    catch (e) { nBad++; const err = String(e && e.message || e).slice(0, 160); setRow(repo, "failed: " + err, "bad"); try { await IDB.put(repo, {v: V, repo, cat, ok: false, err, ts: Date.now()}); } catch (e2) {} }
    upd(); } };
  await Promise.all(Array.from({length: CONC}, work));
  $("#prog").textContent += stop ? " · paused" : " · finished. Press Download results."; $("#go").disabled = false; $("#retry").disabled = false; $("#dl").disabled = false;
}

async function download() {
  const all = await IDB.all(), blob = new Blob([all.map(x => JSON.stringify(x)).join("\n") + "\n"], {type: "application/x-ndjson"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `ame-bench-${all.filter(x => x.ok).length}-models.jsonl`; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
}
(async () => {
  BENCH_MODELS.forEach(([c, r]) => row(r, c));
  $("#count").textContent = BENCH_MODELS.length;
  try { const r = await fetch("https://huggingface.co/api/models/openai-community/gpt2"); if (!r.ok) throw 0; $("#prog").textContent = "Ready. Hugging Face is reachable."; }
  catch (e) { $("#prog").textContent = "This page cannot reach huggingface.co from here (the browser or host blocks it). Open the copy on GitHub Pages instead."; }
  $("#go").onclick = () => run(false); $("#retry").onclick = () => run(true); $("#pause").onclick = () => { stop = true; }; $("#dl").onclick = download;
  $("#reset").onclick = async () => { if (confirm("Delete the results stored in this browser?")) { await IDB.clear(); location.reload(); } };
  try { const prev = await IDB.all(); prev.forEach(d => { if (d.ok) { const q = d.quick || {}; setRow(d.repo, "done", "ok", [fmt(q.params), fmt(d.api.total), q.family || "", fmt(d.read)]); } else setRow(d.repo, "failed: " + d.err, "bad"); }); if (prev.length) $("#dl").disabled = false; } catch (e) {}
  if (location.hash === "#auto") run(false);
})();
