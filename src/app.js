// Model Explorer — page logic. Any model -> parts -> big picture (with Play), layer map, concepts, report. TH/EN.
"use strict";
(function () {
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fB = b => b >= 1e9 ? (b / 1e9).toFixed(2) + " GB" : b >= 1e6 ? (b / 1e6).toFixed(1) + " MB" : b >= 1e3 ? (b / 1e3).toFixed(0) + " KB" : Math.round(b) + " B";
const fP = p => p >= 1e9 ? (p / 1e9).toFixed(2) + " B" : p >= 1e6 ? (p / 1e6).toFixed(1) + " M" : p >= 1e3 ? (p / 1e3).toFixed(0) + " K" : String(Math.round(p));
const fF = f => f >= 1e12 ? (f / 1e12).toFixed(2) + " TFLOP" : f >= 1e9 ? (f / 1e9).toFixed(1) + " GFLOP" : f >= 1e6 ? (f / 1e6).toFixed(1) + " MFLOP" : f >= 1e3 ? (f / 1e3).toFixed(0) + " KFLOP" : Math.round(f) + " FLOP";
const pct = (a, b) => b ? (a / b * 100).toFixed(a / b < .1 ? 1 : 0) + "%" : "0%";
const L2 = (th, en) => LANG === "en" ? en : th;
const DTC = {fp32: "#1b5fd1", fp16: "#7ea3e6", bf16: "#7ea3e6", int8: "#d4651c", int4: "#eba06d", uint8: "#d4651c", fp8: "#e8b48c"};
const C = {red: "#c4242a", redL: "#c4242a", mag: "#d4651c", magL: "#d4651c", blue: "#1b5fd1", blueL: "#5d86d4", ok: "#247b50", amber: "#9a6200", ink: "#141414", soft: "#4d4d4a", dim: "#6b6b66"};
const PAL = {
  light: {ink: "#141414", soft: "#4d4d4a", dim: "#6b6b66", dim2: "#8a8a85", surf: "#ffffff", raised: "#f3f3f0", active: "#e7e7e3", line: "#d9d9d3",
    enc: "#1b5fd1", dec: "#d4651c", ssm: "#1a8f6a", conf: "#b98900", neg: "#c4242a", ok: "#247b50", amber: "#9a6200", blueL: "#5d86d4",
    ovRGB: "20,20,20", blRGB: "27,95,209", blLRGB: "93,134,212", okRGB: "36,123,80",
    role: {backbone: ["#fbe9dd", "#7a3409"], encoder: ["#e3ecfb", "#123f8c"], adapter: ["#f4ecd6", "#5c4300"], graph: ["#dcf3e6", "#0f4a2c"], prediction: ["#f4ecd6", "#5c4300"], joint: ["#ffedea", "#831c1e"], aux: ["#efefeb", "#4d4d4a"]}},
  dark: {ink: "#f4f5f8", soft: "#b9c0d4", dim: "#8f97b0", dim2: "#7d86a0", surf: "#121a2e", raised: "#18213a", active: "#222c48", line: "#2e3854",
    enc: "#8b96fa", dec: "#ff6bb5", ssm: "#4ade80", conf: "#f5c443", neg: "#ff5a67", ok: "#4ade80", amber: "#f5c443", blueL: "#a3abfa",
    ovRGB: "255,255,255", blRGB: "139,150,250", blLRGB: "163,171,250", okRGB: "74,222,128",
    role: {backbone: ["#2a1630", "#ffb3d9"], encoder: ["#1a2147", "#c3c9ff"], adapter: ["#2b2614", "#f5d98a"], graph: ["#122b20", "#9cf0bd"], prediction: ["#2b2614", "#f5d98a"], joint: ["#2e1519", "#ffb0b6"], aux: ["#1c2438", "#b9c0d4"]}}};
const K = {};
function setTheme(th) {
  th = th === "light" ? "light" : "dark"; document.documentElement.dataset.theme = th; Object.assign(K, PAL[th]);
  Object.assign(C, {red: K.neg, redL: K.neg, mag: K.dec, magL: K.dec, blue: K.enc, blueL: K.blueL, ok: K.ok, amber: K.amber, ink: K.ink, soft: K.soft, dim: K.dim});
  Object.keys(BLK).forEach(k => BLK[k] = /SSM/.test(k) ? K.ssm : /MoE/.test(k) ? K.dec : /Conformer|RNN/.test(k) ? K.conf : /Attention|Transformer|DiT|Cross/.test(k) ? K.enc : "#a9a9a3");
  Object.assign(ROLE_C, K.role);
  Object.assign(DTC, th === "dark" ? {fp32: "#8b96fa", fp16: "#c3c9ff", bf16: "#c3c9ff", int8: "#ff4fa8", int4: "#ffaad6", uint8: "#ff4fa8", fp8: "#ffc6e4"} : {fp32: "#1b5fd1", fp16: "#7ea3e6", bf16: "#7ea3e6", int8: "#d4651c", int4: "#eba06d", uint8: "#d4651c", fp8: "#e8b48c"});
}

// ================================================================== models
const MODELS = [];
// models a visitor loaded are kept in this browser (IndexedDB) so the next visit opens them without re-analysis
const DB = {
  open() { return new Promise((res, rej) => { try { const r = indexedDB.open("mx-model-explorer", 1); r.onupgradeneeded = () => r.result.createObjectStore("models", {keyPath: "id"}); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); } catch (e) { rej(e); } }); },
  async tx(mode, fn) { const db = await this.open(); return new Promise((res, rej) => { const tx = db.transaction("models", mode), st = tx.objectStore("models"), out = fn(st); tx.oncomplete = () => res(out && out.result); tx.onerror = () => rej(tx.error); }); },
  put(o) { return this.tx("readwrite", st => st.put(o)).catch(() => {}); },
  all() { return this.tx("readonly", st => st.getAll()).catch(() => []); },
  del(id) { return this.tx("readwrite", st => st.delete(id)).catch(() => {}); },
};
const pref = (k, v) => { try { if (v === undefined) return localStorage.getItem("mx_" + k); localStorage.setItem("mx_" + k, v); } catch (e) { return null; } };
function persist(def) { DB.put({id: def.id, label: def.label, short: def.short, kind: def.kind, graphs: def.graphs, tensors: def.tensors, meta: def.meta, cfg: def.cfg, noteKey: def.noteKey, noteVars: def.noteVars, savedAt: Date.now()}); }
const S = {M: null, lens: "weights", el: 0, dl: 0, open: null, tokens: 40, step: -1, playing: false, zoom: false, mapLens: "size"};
function register(def) { const i = MODELS.findIndex(m => m.id === def.id); if (i >= 0) MODELS[i] = def; else MODELS.push(def); renderPicker(); return def; }
function build(def) {
  if (def.M) return def.M;
  let M;
  if (def.graphs) { M = {A: anatomyFromOps(def.graphs), graphs: def.graphs, meta: def.meta, kind: "ops"}; }
  else if (def.tensors) {
    const I = ArchInfer.infer(def.tensors, {qhint: ArchInfer.quantHint(def.cfg)}); if (def.cfg) I.family = ArchInfer.familyFromConfig(I, def.cfg); const N = ArchInfer.normalize(def.tensors, ArchInfer.quantHint(def.cfg)).T, auto = metaFromIR(I, def.cfg || null, def.label);
    const meta = Object.assign({}, auto, def.meta || {}); if (I.family === "transducer" || I.family === "diffusion") meta.family = auto.family; ["layer_pattern", "irFamily", "quant", "joint", "timestep", "experts"].forEach(k => { if (auto[k] != null && auto[k] !== 0) meta[k] = auto[k]; });
    M = {A: anatomyFromIR(I, N, meta), meta, kind: def.kind || "tensors", IR: I, T: N}; }
  M.id = def.id; M.label = def.label; M.note = def.note; M.viz = def.viz; M.group = def.group; M.cfg = def.cfg || null;
  attachFragility(M);
  def.M = M; return M;
}
function attachFragility(M) {
  const viz = M.viz; M.hasFrag = !!(viz && viz.sensitivity && viz.sensitivity.dyn_int4); if (!M.hasFrag) return;
  const s4 = viz.sensitivity.dyn_int4, f = viz.flow && viz.flow.dyn_int4 || {};
  ["encoder", "decoder"].forEach(c => { const Cc = M.A.comps[c]; if (!Cc) return;
    Cc.layers.forEach(L => { const id = `${c}.L${String(L.idx).padStart(2, "0")}`; L.tot._frag = s4[id];
      Object.entries(L.regions).forEach(([rn, R]) => { Object.entries(R).forEach(([role, p]) => { if (role === "_tot") return;
        const sub = c === "decoder" && /^(q|out)$/.test(role) ? (rn === "self" ? "self_" : "cross_") + role + "_proj" : /^(q|k|v|out)$/.test(role) ? role + "_proj" : role;
        if (f[id + "." + sub] != null) p._frag = f[id + "." + sub]; }); R._tot._frag = Math.max(0, ...Object.values(R).filter(p => p._frag != null).map(p => p._frag)) || null; }); });
  });
  const E = M.A.comps.encoder, D = M.A.comps.decoder;
  if (E) { E.stemTot._frag = s4["encoder.conv"]; E.headTot._frag = Math.max(s4["encoder.cross_kv0"] || 0, s4["encoder.cross_kv1"] || 0) || null; E.layersTot._frag = Math.max(...E.layers.map(L => L.tot._frag || 0)); }
  if (D) { D.stemTot._frag = s4["decoder.tied_embed"]; D.headTot._frag = s4["decoder.tied_embed"]; D.layersTot._frag = Math.max(...D.layers.map(L => L.tot._frag || 0)); }
}
function setModel(id, keepHash) {
  const def = MODELS.find(m => m.id === id) || MODELS[0]; S.M = build(def); S.el = 0; S.dl = 0; S.open = null; stopPlay(); S.step = -1;
  if (S.lens === "fragility" && !S.M.hasFrag) S.lens = "weights"; if (S.lens === "ops" && S.M.kind !== "ops") S.lens = "weights";
  if (S.mapLens === "frag" && !S.M.hasFrag) S.mapLens = "size";
  try { if (!keepHash && /^[\w-]+$/.test(def.id)) history.replaceState(null, "", "#" + def.id); } catch (e) {}
  pref("last", def.id);
  renderAll();
}
const fam = M => /^(transducer|diffusion)$/.test(M.meta.family) ? M.meta.family : M.A.comps.encoder && M.A.comps.decoder ? (M.meta.family === "prefix" ? "prefix" : "encdec") : M.A.comps.decoder ? "decoder" : "encoder";

// ================================================================== hero + picker
const CATS = ["all", "speech", "language", "vision", "embed", "gen", "prec", "own"];
const catOf = m => m.group === "own" ? "own" : m.group === "prec" ? "prec" : (m.cat || "language");
function sizeOf(m) {
  if (m._size != null) return m._size;
  try { if (m.tensors) { const T = ArchInfer.normalize(m.tensors).T; m._size = Object.values(T).reduce((a, v) => a + v.shape.reduce((x, y) => x * y, 1), 0); } else { const M = build(m); m._size = Object.values(M.A.comps).reduce((a, c) => a + c.tot.params, 0); } } catch (e) { m._size = 0; }
  return m._size;
}
const srcOf = m => m.kind === "ops" || m.graphs ? ".tflite" : m.src === "rec" ? t("src_rec") : m.group === "own" ? (m.kind === "config" ? "config" : t("src_header")) : "config";
function renderPicker() {
  const box = $("#picker"); if (!box) return;
  if (!box.dataset.init) { box.dataset.init = 1;
    box.innerHTML = `<div class="libhead"><input id="libq" type="search" autocomplete="off"><span class="xk" id="libn"></span></div><div class="libtabs" id="libtabs" role="tablist"></div><ul class="liblist" id="liblist"></ul>`;
    $("#libq").oninput = e => { S.libQ = e.target.value.trim().toLowerCase(); renderPicker(); }; }
  $("#libq").placeholder = t("lib_search");
  const cnt = {}; MODELS.forEach(m => cnt[catOf(m)] = (cnt[catOf(m)] || 0) + 1);
  if (!S.libCat || S.libCat !== "all" && !cnt[S.libCat]) S.libCat = "all";
  $("#libtabs").innerHTML = CATS.filter(c => c === "all" || cnt[c]).map(c => `<button role="tab" aria-selected="${S.libCat === c}" class="${S.libCat === c ? "on" : ""}" data-cat="${c}">${esc(t("cat_" + c))}<span>${c === "all" ? MODELS.length : cnt[c]}</span></button>`).join("");
  const q = S.libQ || "", hit = m => !q || [m.label, m.short, ...(m.sub || [])].join(" ").toLowerCase().includes(q);
  const shown = MODELS.filter(m => (S.libCat === "all" || catOf(m) === S.libCat) && hit(m));
  $("#libn").textContent = t("lib_count", {n: shown.length});
  const row = m => `<li class="lrow${S.M && S.M.id === m.id ? " on" : ""}" data-m="${esc(m.id)}" tabindex="0" role="button" aria-pressed="${S.M && S.M.id === m.id}"><b title="${esc(m.label)}">${esc(m.short || m.label)}</b><span class="what">${esc(m.sub ? L2(...m.sub) : m.label)}</span><span class="sz">${sizeOf(m) ? esc(fP(sizeOf(m))) : "–"}</span><span class="src">${esc(srcOf(m))}</span>${m.group === "own" ? `<button class="x" data-del="${esc(m.id)}" aria-label="${t("forget")}">×</button>` : "<span></span>"}</li>`;
  let html = "";
  if (S.libCat === "all") CATS.slice(1).forEach(c => { const g = shown.filter(m => catOf(m) === c); if (g.length) html += `<li class="lgroup">${esc(t("cat_" + c))}</li>` + g.map(row).join(""); });
  else html = shown.map(row).join("");
  $("#liblist").innerHTML = html || `<li class="libempty">${t("lib_none")}</li>`;
  $$("#libtabs [data-cat]").forEach(b => b.onclick = () => { S.libCat = b.dataset.cat; pref("cat", S.libCat); renderPicker(); });
  $$("#liblist [data-m]").forEach(r => { const go = e => { if (e.target.dataset.del) { e.stopPropagation(); const id = e.target.dataset.del; DB.del(id); MODELS.splice(MODELS.findIndex(m => m.id === id), 1); if (S.M && S.M.id === id) setModel(MODELS[0].id); else renderPicker(); return; } setModel(r.dataset.m); $("#arch").scrollIntoView({behavior: "smooth", block: "start"}); };
    r.onclick = go; r.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(e); } }; });
}
function renderHero() {
  const M = S.M, A = M.A, tot = Object.values(A.comps).reduce((a, c) => merge(a, c.tot), part()), ex = (A.extra || []).reduce((a, x) => a + x.tot.params, 0);
  const nL = Object.values(A.comps).reduce((a, c) => a + c.layers.length, 0), ops = M.kind === "ops" ? tot.ops : null;
  const f = fam(M), mod = M.meta.modality, params = M.IR ? M.IR.totals.params : tot.params + ex, bytes = M.IR ? M.IR.totals.bytes : tot.bytes;
  $("#cur").innerHTML = `<div><div class="xk">${t("now_showing")}</div><h3>${esc(M.label)}</h3>
    <div class="tags"><span class="tag e">${t("fam_" + f)}</span><span class="tag d">${t("mod_" + mod)}</span><span class="tag">${t("src_" + M.kind)}</span></div></div>
    <div class="kpis">${[[fP(params), t("kpi_weights")], [fB(bytes), t("kpi_size")], [String(nL), t("kpi_layers")], ops != null ? [String(ops), t("kpi_ops")] : [M.meta.vocab ? M.meta.vocab.toLocaleString() : "–", t("kpi_vocab")]].map(([b, s]) => `<div class="kpi"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("")}</div>
    ${M.note ? `<p class="note">${esc(typeof M.note === "function" ? M.note() : M.note)}</p>` : ""}
    ${/^[\w.-]+\/[\w.-]+$/.test(M.label || "") ? `<p class="note"><a href="${/\/(en|th)\/(index\.html)?$/.test(location.pathname) ? "../" : ""}census/?m=${encodeURIComponent(M.label)}">${t("census_cmp")}</a></p>` : ""}`;
  $("#navModel").textContent = M.short || M.label;
}
(function heroCanvas() {
  const cv = $("#heroCv"); if (!cv) return; const x = cv.getContext("2d"); let W, H, P = [], tt = 0, vis = true;
  function size() { W = cv.width = cv.clientWidth * devicePixelRatio; H = cv.height = cv.clientHeight * devicePixelRatio; }
  addEventListener("resize", size); size(); new IntersectionObserver(e => vis = e[0].isIntersecting).observe(cv);
  for (let i = 0; i < 520; i++) P.push({x: Math.random(), s: .0007 + Math.random() * .0015, o: Math.random() * 6.28, a: Math.random(), lane: Math.floor(Math.random() * 24)});
  (function frame() { requestAnimationFrame(frame); if (!vis) return; tt += .012; x.clearRect(0, 0, W, H); const gx = .55;
    for (const p of P) { p.x += p.s; if (p.x > 1.02) { p.x = -.02; p.lane = Math.floor(Math.random() * 24); }
      const wave = (Math.sin(p.x * 18 + tt * 2 + p.o) * .5 + Math.sin(p.x * 7 - tt + p.o * .5) * .5) * (.12 + .1 * Math.sin(p.o * 3)); let px, py;
      if (p.x < gx) { px = p.x; py = .6 + wave * (1 - p.x / gx * .4); } else { const k = Math.min(1, (p.x - gx) / .08), colx = gx + .06 + p.lane * ((.96 - gx - .06) / 24); px = p.x * (1 - k) + colx * k; py = (.6 + wave * .6) * (1 - k) + (.28 + ((p.o * 97) % 1) * .6) * k; }
      x.fillStyle = px < gx ? `rgba(${K.blLRGB},${.2 + .5 * p.a})` : (p.lane < 20 ? `rgba(${K.blRGB},${.3 + .6 * p.a})` : `rgba(${K.blRGB},${.35 + .6 * p.a})`);
      x.beginPath(); x.arc(px * W, py * H, (1 + p.a * 1.5) * devicePixelRatio, 0, 6.28); x.fill(); } })();
})();

// ================================================================== part texts (what goes in, what comes out, why it matters)
function shapes(M) {
  const m = M.meta, d = m.d || 768, mod = m.modality, Se = mod === "audio" ? (m.seqEnc || 1500) : mod === "image" ? (m.patches || 196) + 1 : "n";
  return {d, de: m.encD || d, Se, V: m.vocab || "V", nm: m.nmels || 80, ff: m.ffn || 4 * d, img: m.image || 224};
}
const EX = {audio: "วันนี้อากาศดีมาก", textDec: ["Edge", " AI", " runs", " on"], textDecNext: [" phones", " devices", " the", " CPU", " your"], textEnc: ["[CLS]", "edge", "ai", "is", "[MASK]", "[SEP]"], img: ["tabby cat", "tiger cat", "lynx", "fox", "dog"]};
function partInfo(col, M) {
  const s = shapes(M), m = M.meta, nE = M.A.comps.encoder ? M.A.comps.encoder.layers.length : 0, nD = M.A.comps.decoder ? M.A.comps.decoder.layers.length : 0, F0 = fam(M);
  const encName = F0 === "diffusion" ? L2("Denoiser (DiT)", "Denoiser (DiT)") : F0 === "transducer" ? L2("Acoustic encoder", "Acoustic encoder") : m.modality === "features" ? L2("ตัวทำนายแบบ in-context", "In-context predictor") : null;
  const J = m.joint || {};
  const ffShare = (() => { const L = (M.A.comps.encoder || M.A.comps.decoder).layers[0]; return L && L.regions.ffn ? pct(L.regions.ffn._tot.params, L.tot.params) : "⅔"; })();
  const pre = m.family === "prefix", dec1 = pre ? `[${m.modality === "audio" ? s.Se : "196"} + n, ${s.d}]` : M.A.comps.encoder ? "[1, " + s.d + "]" : "[n, " + s.d + "]";
  const P = {
    in_audio: {t: ["Audio", "Audio"], what: ["เสียง 16 kHz ยาวสูงสุด 30 วินาที = ตัวเลข 480,000 ค่า · ยังไม่มี weight", "16 kHz sound, up to 30 s = 480,000 numbers. No weights yet."], opt: ["เสียงโทรศัพท์ตัดความถี่เกิน 3.4 kHz ทิ้งตั้งแต่ตรงนี้", "A phone line throws away everything above 3.4 kHz right here."], i: null, o: ["wave", "[480,000]"]},
    in_text: {t: ["Text → tokens", "Text → tokens"], what: ["ตัดข้อความเป็น token แล้วแปลงเป็นเลข id จากพจนานุกรม " + (m.vocab ? m.vocab.toLocaleString() : "") + " คำ", "Cut the text into tokens and map each to an id from a " + (m.vocab ? m.vocab.toLocaleString() : "") + "-entry vocabulary."], opt: ["ภาษาไทยมักใช้ token มากกว่าอังกฤษหลายเท่า → ลำดับยาวขึ้น", "Thai often needs several times more tokens than English → longer sequences."], i: null, o: ["tokens", "[n]"]},
    in_image: {t: ["Image", "Image"], what: [`ภาพสี ${s.img}×${s.img} = ตัวเลข ${(3 * s.img * s.img).toLocaleString()} ค่า`, `A ${s.img}×${s.img} colour image = ${(3 * s.img * s.img).toLocaleString()} numbers.`], opt: ["ความละเอียดภาพกำหนดจำนวน token ทั้งโมเดล", "Image resolution sets the token count for the whole model."], i: null, o: ["image", `[3, ${s.img}, ${s.img}]`]},
    mel: {t: ["Log-Mel", "Log-Mel"], what: [`ตัดเสียงเป็นช่วงละ 10 ms วัดพลังงานใน ${s.nm} ช่องความถี่ = “ภาพของเสียง”`, `Slice the sound into 10 ms frames and measure energy in ${s.nm} bands — a picture of sound.`], opt: ["คำนวณก่อนเข้าโมเดล ไม่มี weight", "Computed before the model. No weights."], i: ["wave", "[480,000]"], o: ["spec", `[${s.nm}, 3000]`]},
    conv: {t: ["Conv stem", "Conv stem"], what: [`conv แปลงภาพเสียงเป็นเวกเตอร์ ${s.de} มิติ และย่อเวลา 3000 → ${s.Se} ช่วง`, `Convolutions turn the spectrogram into ${s.de}-dim vectors and shrink time: 3000 → ${s.Se} steps.`], opt: ["เล็กมาก (< 1%) — กันไว้ราคาถูก", "Tiny (< 1%) — cheap to protect."], i: ["spec", `[${s.nm}, 3000]`], o: ["mat", `[${s.Se}, ${s.de}]`]},
    patch: {t: ["Patch embedding", "Patch embedding"], what: [`ตัดภาพเป็นช่อง 16×16 จำนวน ${m.patches || 196} ช่อง แต่ละช่องกลายเป็นเวกเตอร์ ${s.de} มิติ`, `Cut the image into ${m.patches || 196} 16×16 patches; each becomes a ${s.de}-dim vector.`], opt: ["conv เดียว เล็ก แต่เป็นด่านแรกที่เห็นภาพ", "One small conv — the first thing that sees pixels."], i: ["image", `[3, ${s.img}, ${s.img}]`], o: ["patches", `[${s.Se}, ${s.de}]`]},
    proj: {t: ["Projector", "Projector"], what: [m.modality === "image" ? `รวมช่องภาพ 2×2 เป็นหนึ่ง แล้วแปลงจาก ${s.de} เป็น ${s.d} มิติ (ขนาดของ LLM) → token ภาพ ${Math.round(s.Se / 4)} ตัว วางนำหน้าข้อความ` : `แปลงเวกเตอร์เสียงจาก ${s.de} เป็น ${s.d} มิติ (ขนาดของ LLM) → token เสียง ${s.Se} ตัว วางนำหน้า prompt`, m.modality === "image" ? `Merge 2×2 patches and map ${s.de} → ${s.d} dims (the LLM's width) → ${Math.round(s.Se / 4)} image tokens placed before the text.` : `Map the audio vectors ${s.de} → ${s.d} dims (the LLM's width) → ${s.Se} audio tokens placed before the prompt.`], opt: ["เล็ก แต่เป็นสะพานระหว่างสองโมเดล — error ตรงนี้ LLM ได้ยิน/เห็นผิดทั้งหมด", "Small, but it is the bridge between two models — an error here and the LLM hears or sees everything wrong."], i: ["mat2", `[${s.Se}, ${s.de}]`], o: ["mat", `[${m.modality === "image" ? Math.round(s.Se / 4) : s.Se}, ${s.d}]`]},
    embed_enc: {t: ["Token embedding", "Token embedding"], what: [`token id → หยิบแถวจากตาราง ${s.V}×${s.d} + ตำแหน่ง`, `token id → one row of a ${s.V}×${s.d} table, plus position.`], opt: ["ตารางใหญ่ แต่ขาเข้าอ่านแค่แถวเดียวต่อ token", "A big table, but only one row is read per token."], i: ["tokens", "[n]"], o: ["mat", `[n, ${s.d}]`]},
    embed_dec: {t: ["Token embedding", "Token embedding"], what: [`token id → หยิบแถวจากตาราง ${s.V}×${s.d}`, `token id → one row of the ${s.V}×${s.d} table.`], opt: [m.tie ? "ตารางเดียวกับ logits (tied) — เก็บครั้งเดียว" : "แยกจากตาราง logits", m.tie ? "Same table as the logits (tied) — stored once." : "Separate from the logits table."], i: ["tokens", "[n]"], o: ["mat", `[n, ${s.d}]`]},
    enc: {t: [`${encName || (pre ? (m.modality === "image" ? "Vision" : "Audio") + " encoder" : "Encoder")} × ${nE}`, `${encName || (pre ? (m.modality === "image" ? "Vision" : "Audio") + " encoder" : "Encoder")} × ${nE}`], what: [`${nE} layer ซ้อนกัน แต่ละ layer = attention (ทุกตำแหน่งดูกันเอง) + feed-forward · รันครั้งเดียวต่อ input`, `${nE} stacked layers, each = attention (every position looks at every other) + feed-forward. Runs once per input.`], opt: [`compute หนักเพราะรันทั้ง input พร้อมกัน · feed-forward = ${ffShare} ของ weight ใน layer`, `Heavy compute: it runs the whole input at once. Feed-forward is ${ffShare} of each layer's weights.`], i: ["mat", `[${s.Se}, ${s.de}]`], o: ["mat2", `[${s.Se}, ${s.de}]`]},
    kv: {t: ["Encoder out → K/V", "Encoder out → K/V"], what: ["ผลของ encoder ถูกเก็บไว้เป็น Key/Value ให้ decoder ทุกชั้นมาอ่าน · คำนวณครั้งเดียว อ่านซ้ำทุก token", "The encoder output is kept as Keys/Values for every decoder layer. Made once, read every token."], opt: ["บีบแล้วไม่ช่วยเรื่องเวลา แต่ error ตรงนี้กระจายไปทุก token", "Squeezing it saves no time, but an error here spreads to every token."], i: ["mat2", `[${s.Se}, ${s.d}]`], o: ["kv", `${nD}×2×[${s.Se}, ${s.d}]`]},
    dec: {t: [pre ? `LLM × ${nD}` : `Decoder × ${nD}`, pre ? `LLM × ${nD}` : `Decoder × ${nD}`], what: pre ? [`LLM อ่าน token ${m.modality === "image" ? "ภาพ" : "เสียง"} + prompt ทีเดียวทั้งก้อน (prefill) แล้วเขียนคำตอบทีละ token โดยจดทุกอย่างไว้ใน KV cache${m.layer_pattern ? " · layer เรียงแบบ " + m.layer_pattern : ""}`, `The LLM reads the ${m.modality === "image" ? "image" : "audio"} tokens + prompt in one go (prefill), then writes the answer one token at a time, keeping everything in the KV cache${m.layer_pattern ? " · layer pattern " + m.layer_pattern : ""}.`] : M.A.comps.encoder ? [`สร้างคำทีละ token: ดูคำที่เขียนแล้ว (KV cache) → ฟังเสียง/ข้อมูลจาก encoder (cross-attention) → feed-forward`, `Writes one token at a time: look at what's written (KV cache) → listen to the encoder (cross-attention) → feed-forward.`] : [`${nD} layer: attention ดูเฉพาะ token ก่อนหน้า (KV cache) + feed-forward · รันใหม่ทุก token`, `${nD} layers: attention over earlier tokens only (KV cache) + feed-forward. Runs again for every token.`], opt: ["รันทุก token → อ่าน weight ซ้ำหลายสิบรอบ · byte ต่อ weight กำหนดความเร็ว", "Runs every token → re-reads its weights dozens of times. Bytes per weight set the speed."], i: ["vec", dec1], o: ["vec2", "[1, " + s.d + "]"]},
    lmhead: {t: ["Norm → logits", "Norm → logits"], what: [`คูณเวกเตอร์กับตารางคำศัพท์ ${s.V}×${s.d} → คะแนนของทุกคำ แล้วเลือกคำถัดไป`, `Multiply by the ${s.V}×${s.d} vocabulary table → a score for every token, then pick the next one.`], opt: ["ก้อนเดียวที่ใหญ่สุดของฝั่ง decoder · ภาษาไทยมักพังที่นี่ก่อนเมื่อบีบ int4", "The single biggest block on the decoder side; Thai tends to break here first at int4."], i: ["vec2", "[1, " + s.d + "]"], o: ["probs", `[${s.V}]`]},
    pool: {t: ["Pooler / head", "Pooler / head"], what: ["สรุปทั้งลำดับเป็นเวกเตอร์เดียว (จาก [CLS]) ใช้ต่อกับงาน classify หรือค้นหา", "Summarize the whole sequence into one vector (from [CLS]) for classification or search."], opt: ["เล็ก", "Small."], i: ["mat2", `[n, ${s.d}]`], o: ["vec2", `[${s.d}]`]},
    cls_head: {t: ["Classifier", "Classifier"], what: [`เอาเวกเตอร์ [CLS] ไปคูณ → คะแนนของ ${m.labels || 1000} คลาส`, `Take the [CLS] vector → scores for ${m.labels || 1000} classes.`], opt: ["เล็ก", "Small."], i: ["mat2", `[${s.Se}, ${s.d}]`], o: ["probs_img", `[${m.labels || 1000}]`]},
    out_text: {t: ["Text out", "Text out"], what: M.meta.modality === "audio" ? ["ต่อ token เป็นประโยค แล้ววัด WER/CER", "Join the tokens into a sentence, then measure WER/CER."] : ["ต่อ token ใหม่ท้ายข้อความ แล้ววนกลับไปสร้างคำต่อไป", "Append the new token and loop back for the next one."], opt: [M.meta.modality === "audio" ? "gate ของ workshop: WER บนเสียงโทรศัพท์ห้ามเพิ่มเกิน 2 จุด" : "ความเร็ว = token ต่อวินาที", M.meta.modality === "audio" ? "Workshop gate: phone-audio WER may rise by at most 2 points." : "Speed = tokens per second."], i: ["probs", `[${s.V}]`], o: ["text", ""]},
    out_vec: {t: ["Embedding out", "Embedding out"], what: ["เวกเตอร์ที่สรุปความหมายของข้อความ ใช้ classify / ค้นหา / จัดกลุ่ม", "A vector that captures the meaning of the text — for classification, search, clustering."], opt: ["", ""], i: ["vec2", `[${s.d}]`], o: null},
    in_latent: {t: ["Noisy latent", "Noisy latent"], what: ["วิดีโอ/เสียงในรูป latent ที่ยังมี noise + ขั้นเวลา t + ข้อความเงื่อนไข · ยังไม่มี weight", "The video/audio latent, still noisy, plus the timestep t and the text condition. No weights yet."], opt: ["ความยาววิดีโอ × ความละเอียด = จำนวน token ทั้งหมดของ denoiser", "Clip length × resolution sets the token count for the whole denoiser."], i: null, o: ["patches", "[T·H·W, C]"]},
    in_features: {t: ["Table rows", "Table rows"], what: ["แถวตัวอย่างที่มีคำตอบ (context) + แถวที่ต้องทำนาย (query) · ทุก cell กลายเป็นตัวเลข", "Labelled example rows (context) plus the rows to predict (query). Every cell becomes numbers."], opt: ["ไม่มีการเทรนต่อ: โมเดลเรียนจากแถวตัวอย่างใน prompt", "No fine-tuning: the model learns from the example rows in its input."], i: null, o: ["mat", "[rows, cols]"]},
    stem_lat: {t: ["Patchify + timestep", "Patchify + timestep"], what: [`แบ่ง latent เป็นช่องแล้วแปลงเป็นเวกเตอร์ ${s.d} มิติ · แปลงเลขขั้นเวลาเป็นเวกเตอร์ (sinusoid ${(m.timestep || {}).freq || 256} → MLP) · แปลงข้อความเป็นขนาดเดียวกัน`, `Cut the latent into patches and map each to a ${s.d}-dim vector; turn the timestep into a vector (sinusoid ${(m.timestep || {}).freq || 256} → MLP); map the text to the same width.`], opt: ["เล็ก แต่ timestep MLP คุมทุก block ผ่าน adaLN", "Small, but the timestep MLP steers every block through adaLN."], i: ["patches", "[T·H·W, C]"], o: ["mat", `[N, ${s.d}]`]},
    stem_feat: {t: ["Cell → row embedding", "Cell → row embedding"], what: ["แปลงแต่ละ cell (ตัวเลข/หมวดหมู่) เป็นเวกเตอร์ แล้วใช้ attention ข้ามคอลัมน์รวมเป็นเวกเตอร์ต่อแถว", "Turn each cell (number or category) into a vector, then attend across columns to get one vector per row."], opt: ["ตาราง 100 คอลัมน์ไม่ทำให้โมเดลใหญ่ขึ้น แต่ compute เพิ่ม", "A 100-column table doesn't grow the model, but it grows the compute."], i: ["mat", "[rows, cols]"], o: ["mat", `[rows, ${s.d}]`]},
    pj: {t: ["Prediction + joint", "Prediction + joint"], what: [`prediction network (embedding ${J.labels || "V"}×${J.dPred || ""} + LSTM) จำ label ที่ออกไปแล้ว · joint รวมเวกเตอร์เสียง (${J.dEnc || ""}) กับ label (${J.dPred || ""}) ใน ${J.j || ""} มิติ → คะแนน ${J.out || ""} ค่า${J.durations ? ` = ${J.labels} label + ${J.durations} ช่วงเวลากระโดด (TDT)` : ""}`, `The prediction network (a ${J.labels || "V"}×${J.dPred || ""} label embedding + LSTM) remembers the labels emitted so far; the joint adds the acoustic vector (${J.dEnc || ""}) and the label vector (${J.dPred || ""}) in ${J.j || ""} dims → ${J.out || ""} scores${J.durations ? ` = ${J.labels} labels + ${J.durations} duration bins (TDT)` : ""}.`], opt: ["เล็กมาก แต่รันทุก frame × ทุก label → latency ของ streaming อยู่ที่นี่", "Tiny, but it runs per frame × per label — this is where streaming latency lives."], i: ["mat2", `[T, ${J.dEnc || s.d}]`], o: ["probs", `[${J.out || ""}]`]},
    out_latent: {t: ["Cleaner latent", "Cleaner latent"], what: ["ทำนาย noise/velocity แล้วลบออกหนึ่งขั้น · วนซ้ำหลายขั้น แล้ว VAE ถอดเป็นวิดีโอ+เสียง", "Predict the noise/velocity and remove one step of it. Repeat for every step, then the VAE decodes video + audio."], opt: ["จำนวนขั้น × ขนาด denoiser = เวลาทั้งหมด · โมเดลกลั่น 8 ขั้นเร็วกว่า 50 ขั้น ~6 เท่า", "Steps × denoiser size = total time. An 8-step distilled model is ~6× faster than 50 steps."], i: ["mat2", `[N, ${s.d}]`], o: ["patches", "[T·H·W, C]"]},
    out_pred: {t: ["Prediction", "Prediction"], what: ["ความน่าจะเป็นของแต่ละคลาส (หรือค่าตัวเลข) สำหรับแถว query", "Class probabilities (or a number) for every query row."], opt: ["", ""], i: ["probs_img", ""], o: ["label", ""]},
    out_class: {t: ["Label", "Label"], what: ["คลาสที่คะแนนสูงสุด", "The highest-scoring class."], opt: ["", ""], i: ["probs_img", ""], o: ["label", ""]},
  };
  if (F0 === "diffusion" && col.kind === "enc") P.enc.what = [`${nE} block · แต่ละ block: adaLN (ใช้ timestep) → self-attention ทั้งวิดีโอ+เสียง+ข้อความ → feed-forward · รันครั้งละหนึ่งขั้น denoise`, `${nE} blocks; each: adaLN (from the timestep) → self-attention over video + audio + text → feed-forward. Runs once per denoising step.`];
  if (F0 === "transducer" && col.kind === "enc") P.enc.what = [`${nE} Conformer block: ½FFN → attention → conv module → ½FFN · ทำ subsampling ×8 จึงได้ ~12.5 frame ต่อวินาที`, `${nE} Conformer blocks: ½FFN → attention → conv module → ½FFN, after ×8 subsampling (~12.5 frames per second).`];
  if (m.layer_pattern && (col.kind === "dec" || col.kind === "enc") && !P[col.kind].what[1].includes("pattern")) { P[col.kind].what = P[col.kind].what.map((w, i) => w + (i ? " · layer pattern " : " · ลำดับ layer ") + m.layer_pattern); }
  if (col.title && P[col.kind]) { P[col.kind] = Object.assign({}, P[col.kind], {t: [col.title, col.title]}); if (col.oshape) P[col.kind].o = [P[col.kind].o ? P[col.kind].o[0] : "mat", col.oshape]; if (col.what) P[col.kind].what = col.what; }
  const p = P[col.kind] || {t: [col.kind, col.kind], what: ["", ""], opt: ["", ""]};
  return {title: L2(...p.t), what: L2(...p.what), opt: L2(...p.opt), i: p.i, o: p.o};
}

// ---------------- tiny visuals for the I/O panel
function viz(kind, w = 210, h = 104) {
  let s = "", r = (i, j) => (Math.sin(i * 12.9898 + j * 78.233) * 43758.5453) % 1, ab = v => Math.abs(v);
  if (kind === "wave") { let d = ""; for (let i = 0; i <= 80; i++) { const x = 6 + i * (w - 12) / 80, a = Math.sin(i * .7) * Math.sin(i * .17) * h * .38; d += (i ? "L" : "M") + x.toFixed(1) + " " + (h / 2 + a).toFixed(1); } s = `<path d="${d}" stroke="${C.soft}" stroke-width="2" fill="none"/>`; }
  else if (kind === "spec") { for (let i = 0; i < 26; i++) for (let j = 0; j < 12; j++) { const v = Math.max(0, Math.sin(i * .5 + j * .35) * .5 + .5 - j / 18); s += `<rect x="${4 + i * 7.9}" y="${h - 4 - (j + 1) * 8}" width="7.3" height="7.3" fill="rgba(${K.blLRGB},${(.1 + v * .85).toFixed(2)})"/>`; } }
  else if (kind === "tokens" || kind === "text") {
    const toks = S.M.meta.modality === "audio" ? [EX.audio] : S.M.A.comps.decoder ? (kind === "text" ? [...EX.textDec, EX.textDecNext[0]] : EX.textDec) : EX.textEnc; let x = 6, y = 26;
    toks.forEach((tk, i) => { const tw = Math.max(26, tk.length * 8.4 + 14); if (x + tw > w) { x = 6; y += 32; } const last = kind === "text" && i === toks.length - 1;
      s += `<rect x="${x}" y="${y - 17}" width="${tw}" height="24" rx="7" fill="${last ? `rgba(${K.okRGB},.14)` : `rgba(${K.ovRGB},0.056)`}" stroke="${last ? C.ok : `rgba(${K.ovRGB},0.160)`}"/><text x="${x + tw / 2}" y="${y}" text-anchor="middle" font-size="13" fill="${last ? C.ok : C.ink}">${esc(tk.trim() || "␣")}</text>`;
      if (kind === "tokens" && S.M.meta.modality !== "audio") s += `<text x="${x + tw / 2}" y="${y + 18}" text-anchor="middle" font-size="9.5" fill="${C.dim}" font-family="var(--mono)">${Math.floor(ab(r(i, 3)) * 30000)}</text>`; x += tw + 6; });
    if (kind === "text" && S.M.meta.modality === "audio") s = `<text x="${w / 2}" y="${h / 2 + 8}" text-anchor="middle" font-size="22" font-weight="700" fill="${C.ok}">${EX.audio}</text>`; }
  else if (kind === "image" || kind === "patches") { for (let i = 0; i < 14; i++) for (let j = 0; j < 7; j++) { const v = .5 + .5 * Math.sin(i * .45 + j * .6), c2 = `hsl(${30 + v * 30},${40 + 30 * v}%,${25 + 35 * v}%)`; s += `<rect x="${w / 2 - 98 + i * 14}" y="${4 + j * 14}" width="${kind === "patches" ? 12.5 : 14}" height="${kind === "patches" ? 12.5 : 14}" fill="${c2}"/>`; } }
  else if (kind === "mat" || kind === "mat2") { const cols = 30, rows = 9; for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) { const v = ab(r(i + (kind === "mat2" ? 7 : 0), j)); const c2 = kind === "mat2" ? `rgba(${K.blRGB},${(.15 + .8 * v).toFixed(2)})` : `rgba(${K.blRGB},${(.15 + .8 * v).toFixed(2)})`; s += `<rect x="${4 + i * 6.8}" y="${6 + j * 10.5}" width="6" height="9.5" rx="1.5" fill="${c2}"/>`; } }
  else if (kind === "kv") { ["K", "V"].forEach((l, k) => { for (let i = 0; i < 13; i++) for (let j = 0; j < 7; j++) s += `<rect x="${8 + k * 104 + i * 7}" y="${18 + j * 11}" width="6.2" height="10" rx="1.5" fill="rgba(${K.blRGB},${(.2 + .75 * ab(r(i + k * 9, j))).toFixed(2)})"/>`; s += `<text x="${54 + k * 104}" y="13" text-anchor="middle" font-size="12" font-weight="700" fill="${C.ink}">${l}</text>`; }); }
  else if (kind === "vec" || kind === "vec2") { for (let i = 0; i < 28; i++) { const v = ab(r(i + (kind === "vec2" ? 5 : 0), 2)); s += `<rect x="${6 + i * 7.2}" y="${h / 2 - 9}" width="6.4" height="18" rx="1.5" fill="${kind === "vec2" ? `rgba(${K.blRGB},${(.2 + .75 * v).toFixed(2)})` : `rgba(212,101,28,${(.2 + .75 * v).toFixed(2)})`}"/>`; } }
  else if (kind === "probs" || kind === "probs_img") { const lab = kind === "probs_img" ? EX.img : S.M.meta.modality === "audio" ? ["วันนี้", "วัน", "ทุก", "เมื่อ", "พรุ่ง"] : EX.textDecNext, pr = [.62, .14, .09, .05, .03];
    pr.forEach((v, i) => { s += `<text x="${w * .42}" y="${16 + i * 19}" text-anchor="end" font-size="12" fill="${i ? C.soft : C.ok}">${esc(lab[i].trim())}</text><rect x="${w * .45}" y="${6 + i * 19}" width="${v * w * .8}" height="13" rx="3" fill="${i ? C.mag : C.ok}" opacity="${i ? .55 : 1}"/><text x="${w * .45 + v * w * .8 + 4}" y="${16 + i * 19}" font-size="10" fill="${C.dim}" font-family="var(--mono)">${v.toFixed(2)}</text>`; }); }
  else if (kind === "label") s = `<text x="${w / 2}" y="${h / 2 + 8}" text-anchor="middle" font-size="22" font-weight="700" fill="${C.ok}">${EX.img[0]}</text>`;
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" class="iov">${s}</svg>`;
}

// ================================================================== big picture
const LENS = {
  weights: {unit: p => fB(p.bytes), sub: p => fP(p.params) + " params", v: p => p.bytes},
  compute: {unit: p => fF(p.flops) + (p.est ? "*" : ""), sub: p => p.ops ? p.ops + " ops" : "", v: p => p.flops},
  ops: {unit: p => p.ops + " ops", sub: p => Object.entries(p.types).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([k, v]) => k.replace("FULLY_CONNECTED", "FC").replace("BATCH_MATMUL", "BMM") + "×" + v).join(" "), v: p => p.ops},
  precision: {unit: p => dtLabel(p), sub: p => p.seams ? p.seams + " seams" : "", v: p => p.bytes},
  fragility: {unit: p => p._frag != null ? (p._frag * 100).toFixed(1) + "% err" : "–", sub: () => "int4 alone", v: p => p._frag || 0},
};
function dtLabel(p) { const e = Object.entries(p.dt).sort((a, b) => b[1] - a[1]); if (!e.length) return p.shared ? t("tied_emb") : t("no_weights"); const tt = e.reduce((a, x) => a + x[1], 0); return e.map(([k, v]) => e.length > 1 ? `${k} ${Math.round(v / tt * 100)}%` : k).join(" · "); }
function dtColor(p) { const e = Object.entries(p.dt).sort((a, b) => b[1] - a[1]); if (p.seams && !e.length) return K.neg; return e.length ? DTC[e[0][0]] || K.dim2 : K.active; }
function lensFill(p, max) {
  if (!p || !p.has) return {fill: `rgba(${K.ovRGB},0.024)`};
  if (S.lens === "precision") return {fill: dtColor(p) + "33", stroke: dtColor(p)};
  const k = max ? Math.min(1, LENS[S.lens].v(p) / max) : 0; return {fill: `rgba(${K.blRGB},${(.05 + .45 * Math.sqrt(k)).toFixed(3)})`};
}
function bar(x, y, w, p, max) { if (!p || !p.has || S.lens === "precision") return ""; const k = max ? Math.min(1, LENS[S.lens].v(p) / max) : 0; return `<rect x="${x}" y="${y}" width="${w}" height="5" rx="2.5" fill="rgba(${K.ovRGB},0.064)"/><rect x="${x}" y="${y}" width="${Math.max(2, w * k)}" height="5" rx="2.5" fill="${K.enc}"/>`; }

// every region kind the automatic recovery can name, in reading order inside a layer
const REG = {
  linear: [() => S.M && S.M.IR ? L2("SSM / linear attention", "SSM / linear attention") : "Linear attention", () => L2("state ขนาดคงที่ · ไม่มี KV cache", "fixed-size state · no KV cache")],
  self: [(Lx) => Lx && Lx.regions.linear || (S.M && S.M.A.comps.decoder && S.M.A.comps.decoder.layers.some(L => L.regions.linear)) ? "Full attention" : "Self-attention", (isE) => isE ? "LN → Q K V → softmax → out ⊕" : L2("อ่าน + เขียน KV cache", "reads + writes KV cache")],
  cross: [() => "Cross-attention", () => L2("ฟัง K/V จาก encoder", "listens to the encoder K/V")],
  conv: [() => L2("Convolution module", "Convolution module"), () => L2("pointwise → depthwise → pointwise", "pointwise → depthwise → pointwise")],
  moe: [() => L2("Mixture of experts", "Mixture of experts"), () => { const m = S.M.meta; return m.experts ? L2(`router เลือก ${m.topk || "k"} จาก ${m.experts} expert`, `router picks ${m.topk || "k"} of ${m.experts} experts`) : "router → experts"; }],
  ffn: [() => "Feed-forward", (isE, Lx) => Lx && Lx.regions.ffn && Lx.regions.ffn.gate ? "gate · up → act → down ⊕" : "LN → up → act → down ⊕"],
  mod: [() => L2("adaLN modulation", "adaLN modulation"), () => L2("timestep → shift · scale · gate", "timestep → shift · scale · gate")],
  rnn: [() => "Recurrent cell", () => "LSTM / GRU"],
};
const REG_ORDER = ["linear", "self", "cross", "conv", "moe", "ffn", "mod", "rnn"];
const regTitle = (rk, Lx) => REG[rk] ? REG[rk][0](Lx) : rk;
let LAY = null; // layout of the overview, kept for play/zoom
function layoutCols(cols) {
  const W = 1600, m = 26, wOf = c => c.stack ? 0 : c.io ? 92 : 104, nStack = cols.filter(c => c.stack).length;
  const fixed = cols.reduce((a, c) => a + wOf(c), 0), gaps = (cols.length - 1) * 40;
  const sw = nStack ? Math.max(240, Math.min(420, (W - 2 * m - fixed - gaps) / nStack)) : 0;
  const total = fixed + sw * nStack, gap = (W - 2 * m - total) / Math.max(1, cols.length - 1); let x = m;
  return cols.map(c => { const w = c.stack ? sw : wOf(c); const o = {...c, x, w}; x += w + gap; return o; });
}
function renderBig() {
  const M = S.M, A = M.A, E = A.comps.encoder, D = A.comps.decoder, F = fam(M);
  const cols = layoutCols(pipeline(M)); const y0 = 140, W = 1600, H = 520;
  const nE = E ? E.layers.length : 0, nD = D ? D.layers.length : 0;
  $("#bigH2").textContent = t("big_h2_" + F, {n: cols.length});
  $$("[data-xl]").forEach(b => { b.classList.toggle("on", b.dataset.xl === S.lens); b.disabled = b.dataset.xl === "fragility" ? !M.hasFrag : b.dataset.xl === "ops" ? M.kind !== "ops" : false; });
  $("#lensSay").textContent = t("say_" + S.lens);
  const top = [E && E.stemTot, E && E.layersTot, E && E.headTot, D && D.stemTot, D && D.layersTot, D && D.headTot].filter(Boolean);
  const max = Math.max(...top.map(p => LENS[S.lens].v(p) || 0), 1e-9);
  const Le = E ? E.layers[S.el] : null, Ld = D ? D.layers[S.dl] : null;
  const regMax = Math.max(...[Le, Ld].filter(Boolean).flatMap(l => Object.values(l.regions).map(r => LENS[S.lens].v(r._tot) || 0)), 1e-9);
  const cur = S.step >= 0 && PLAY[S.step] ? PLAY[S.step] : null;
  let s = `<defs><marker id="xa" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="${K.dim2}"/></marker>
    <linearGradient id="rib" x1="0" x2="1"><stop offset="0" stop-color="${K.blueL}" stop-opacity=".32"/><stop offset="1" stop-color="${K.enc}" stop-opacity=".32"/></linearGradient>
    <linearGradient id="ribd" x1="0" x2="1"><stop offset="0" stop-color="${K.dec}" stop-opacity=".32"/><stop offset="1" stop-color="${K.dec}" stop-opacity=".32"/></linearGradient>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="7" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
  const num = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"];
  cols.forEach((c, i) => { const info = partInfo(c, M), cx = c.x + c.w / 2; const sh = info.o ? info.o[1] : "";
    s += `<text x="${cx}" y="34" text-anchor="middle" class="xh">${num[i]} ${esc(info.title)}</text><text x="${cx}" y="54" text-anchor="middle" class="xs" fill="${K.blueL}">${esc(sh)}</text>`; });
  s += `<line x1="20" y1="68" x2="${W - 20}" y2="68" stroke="rgba(${K.ovRGB},0.064)"/>`;
  // ribbons between neighbours in the main row
  for (let i = 0; i < cols.length - 1; i++) { const a = cols[i], b = cols[i + 1]; if (b.kind === "kv" && false) continue;
    const x1 = a.x + a.w, x2 = b.x, yy = y0 + 110, dec = (a.comp === "decoder" || a.kind === "lmhead" || a.kind === "dec" || b.kind === "dec" && F === "decoder" || a.kind === "embed_dec"), w = 16;
    if (a.kind === "kv" && b.kind === "dec") continue;
    s += `<path d="M${x1} ${yy - w / 2} H${x2} V${yy + w / 2} H${x1}Z" fill="url(#${dec ? "ribd" : "rib"})"/><path d="M${x1} ${yy} H${x2}" stroke="${K.ink}" stroke-opacity=".25" stroke-width="1.5" class="flowline"/>`; }
  const sel = id => S.open === id || (cur && cur.id === id);
  const box = (id, x, y, w, h, title, p, mx, wide = false, cap = "") => { const f = lensFill(p, mx), u = p && p.has ? esc(LENS[S.lens].unit(p)) : "", sb = p && p.has ? esc(LENS[S.lens].sub(p) || "") : "";
    const inner = wide ? `<text x="${x + 14}" y="${y + 23}" class="xt">${esc(title)}</text><text x="${x + w - 14}" y="${y + 23}" text-anchor="end" class="xm">${u}</text><text x="${x + 14}" y="${y + 42}" class="xs">${esc(cap)}</text>${h >= 80 ? `<text x="${x + w - 14}" y="${y + 60}" text-anchor="end" class="xs">${sb}</text>` : ""}`
      : `<text x="${x + w / 2}" y="${y + 24}" text-anchor="middle" class="xt">${esc(title)}</text><text x="${x + w / 2}" y="${y + h - 32}" text-anchor="middle" class="xm">${u}</text>${sb.length * 6.6 < w ? `<text x="${x + w / 2}" y="${y + h - 17}" text-anchor="middle" class="xs">${sb}</text>` : ""}`;
    return `<g class="xbox${sel(id) ? " sel" : ""}" data-x="${id}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" fill="${f.fill}" stroke="${f.stroke || `rgba(${K.ovRGB},0.176)`}" stroke-width="${sel(id) ? 2.6 : 1.3}" ${sel(id) ? '' : ""}/>${inner}${bar(x + 12, y + h - 9, w - 24, p, mx)}</g>`; };
  const ioBox = (c, v) => `<g class="xbox${sel(c.kind) ? " sel" : ""}" data-x="${c.kind}"><rect x="${c.x}" y="${y0 + 40}" width="${c.w}" height="140" rx="14" fill="rgba(${K.ovRGB},0.032)" stroke="${sel(c.kind) ? K.dec : `rgba(${K.ovRGB},0.160)`}" stroke-width="${sel(c.kind) ? 2.6 : 1.3}"/>
      <g transform="translate(${c.x + 4},${y0 + 70}) scale(${(c.w - 8) / 210})">${viz(v).replace('class="iov"', "")}</g><text x="${c.x + c.w / 2}" y="${y0 + 200}" text-anchor="middle" class="xs">${t("no_weights")}</text></g>`;
  let zoomLayer = "";
  const stack = (c, Cc, Lx, n, comp) => {
    if (S.zoom && S.zoom.comp === comp && !c._z) { const zw = Math.max(c.w, 640); zoomLayer = stack({...c, _z: true, x: c.x + c.w / 2 - zw / 2, w: zw}, Cc, Lx, n, comp); return ""; }
    const x = c.x, y = y0 - 10, w = c.w, h = 250, isE = comp === "encoder", col = isE ? K.enc : K.dec, idx = isE ? S.el : S.dl;
    let o = ""; for (let k = 3; k >= 1; k--) o += `<rect x="${x + k * 7}" y="${y - k * 7}" width="${w}" height="${h}" rx="18" fill="${isE ? K.raised : K.raised}" stroke="${col}" stroke-opacity="${.15 + .1 * (3 - k)}"/>`;
    const f = lensFill(Cc.layersTot, max);
    o += `<g class="${sel(c.id) ? "selstack" : ""}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="18" fill="${isE ? K.surf : K.surf}" stroke="${col}" stroke-width="${sel(c.id) ? 2.6 : 1.6}" ${sel(c.id) ? '' : ""}/><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="18" fill="${f.fill}" opacity=".6"/></g>
      <text x="${x + 18}" y="${y + 28}" class="xt">${S.M.meta.family === "prefix" ? (isE ? (S.M.meta.modality === "image" ? "Vision" : "Audio") : "LLM") : S.M.meta.family === "diffusion" ? "DiT" : S.M.meta.family === "transducer" ? "Conformer" : isE ? "Encoder" : "Decoder"} ${t("layer")}${Lx && Lx.regions.moe ? " · MoE" : Lx && Lx.regions.linear ? " · SSM" : Lx && Cc.layers.some(L => L.regions.linear) ? " · attn" : ""}</text>
      <g class="xnav" data-nav="${isE ? "e" : "d"}-"><rect x="${x + w - 128}" y="${y + 10}" width="26" height="26" rx="8" class="xr"/><text x="${x + w - 115}" y="${y + 28}" text-anchor="middle" class="xt">‹</text></g>
      <text x="${x + w - 72}" y="${y + 28}" text-anchor="middle" class="xm">${idx + 1} / ${n}</text>
      <g class="xnav" data-nav="${isE ? "e" : "d"}+"><rect x="${x + w - 42}" y="${y + 10}" width="26" height="26" rx="8" class="xr"/><text x="${x + w - 29}" y="${y + 28}" text-anchor="middle" class="xt">›</text></g>`;
    if (Lx) {
      const regs = REG_ORDER.filter(k => Lx.regions[k]).map(k => [k, regTitle(k, Lx), REG[k][1](isE, Lx)]);
      const bh = regs.length >= 4 ? 40 : regs.length === 3 ? 60 : 82, gap = regs.length >= 4 ? 6 : regs.length === 3 ? 8 : 14; let yy = y + 46;
      if (S.zoom && S.zoom.comp === comp) o += zoomSteps(Lx, x, y, w, comp);
      else regs.forEach(([rk, title, cap]) => { o += box((isE ? "e_" : "d_") + rk, x + 16, yy, w - 32, bh, title, Lx.regions[rk]._tot, regMax, true, cap); yy += bh + gap; });
    }
    o += `<text x="${x + w / 2}" y="${y + h + 22}" text-anchor="middle" class="xs">${esc(t("all_layers", {n, v: LENS[S.lens].unit(Cc.layersTot)}))}${isE ? "" : " " + t("per_token")}</text>${bar(x + 40, y + h + 30, w - 80, Cc.layersTot, max)}`;
    return o;
  };
  const embBelow = F === "encdec" || F === "prefix";
  cols.forEach(c => {
    if (c.io) s += ioBox(c, {in_audio: "wave", in_text: "tokens", in_image: "image", mel: "spec", out_text: "text", out_vec: "vec2", out_class: "label", in_latent: "patches", out_latent: "patches", in_features: "mat", out_pred: "label"}[c.kind] || "mat");
    else if (c.stack) s += stack(c, c.comp === "encoder" ? E : D, c.comp === "encoder" ? Le : Ld, c.comp === "encoder" ? nE : nD, c.comp);
    else s += box(c.id, c.x, y0 + 40, c.w, 140, c.title || partInfo(c, M).title.replace("Encoder out → K/V", "K/V ×" + nD).replace("Token embedding", "Embed").replace("Patch embedding", "Patches").replace("Norm → logits", "→ logits").replace("Patchify + timestep", "Patchify + t").replace("Cell → row embedding", "Rows → vec").replace("Prediction + joint", "Pred + joint"), c.part, max);
  });
  s += zoomLayer;
  // decoder-side extras
  if (D) {
    const dc = cols.find(c => c.kind === "dec"), hd = cols.find(c => c.kind === "lmhead"), out = cols.find(c => c.kind === "out_text");
    let emb = cols.find(c => c.kind === "embed_dec");
    if (embBelow) { emb = {x: dc.x, w: Math.min(210, dc.w), id: "d_stem", below: true}; s += box("d_stem", emb.x, y0 + 296, emb.w, 64, L2("Token embed", "Token embed"), D.stemTot, max, true, "token id → vector"); s += `<path d="M${emb.x + 75} ${y0 + 296} V${y0 + 248}" stroke="${K.dec}" stroke-width="2" class="flowline" marker-end="url(#xa)"/>`; }
    const kv = cols.find(c => c.kind === "kv"); if (kv) s += `<path d="M${kv.x + kv.w} ${y0 + 110} C ${kv.x + kv.w + 30} ${y0 + 110}, ${dc.x - 30} ${y0 + 160}, ${dc.x} ${y0 + 160}" stroke="${K.enc}" stroke-width="3" fill="none" stroke-dasharray="6 6" class="flowline" marker-end="url(#xa)"/>`;
    const ex = emb.below ? emb.x + emb.w : emb.x + emb.w / 2, ey = emb.below ? y0 + 345 : y0 + 180;
    s += emb.below ? `<path d="M${out.x + out.w / 2} ${y0 + 180} V${y0 + 345} H${ex + 3}" stroke="${K.dec}" stroke-width="2" fill="none" stroke-dasharray="6 6" class="flowline loopline" marker-end="url(#xa)"/>`
      : `<path d="M${out.x + out.w / 2} ${y0 + 180} V${y0 + 345} H${ex} V${ey + 4}" stroke="${K.dec}" stroke-width="2" fill="none" stroke-dasharray="6 6" class="flowline loopline" marker-end="url(#xa)"/>`;
    s += `<text x="${(out.x + ex) / 2 + 40}" y="${y0 + 338}" text-anchor="middle" class="xs" fill="${K.dec}">${esc(t("next_token", {t: S.tokens}))}</text>`;
    if (D.headTot.shared) { const sx = emb.below ? emb.x + emb.w : emb.x + emb.w / 2 + 14, sy = emb.below ? y0 + 308 : y0 + 180;
      s += emb.below ? `<path d="M${sx} ${sy} H${hd.x + hd.w / 2} V${y0 + 182}" stroke="${K.amber}" stroke-width="1.6" stroke-dasharray="4 5" fill="none" marker-end="url(#xa)"/>` : `<path d="M${sx} ${sy} V${y0 + 290} H${hd.x + hd.w / 2} V${y0 + 182}" stroke="${K.amber}" stroke-width="1.6" stroke-dasharray="4 5" fill="none" marker-end="url(#xa)"/>`;
      s += `<text x="${(sx + hd.x + hd.w / 2) / 2}" y="${(emb.below ? sy : y0 + 290) - 7}" text-anchor="middle" class="xs" fill="${K.amber}">${esc(t("tied_note"))}</text>`; }
  }
  if (!D && (F === "diffusion" || F === "transducer")) { const a = cols[cols.length - 1], b = cols[F === "diffusion" ? 1 : 4] || cols[1]; if (a && b) { const ax = a.x + a.w / 2, bx = b.x + b.w / 2;
    s += `<path d="M${ax} ${y0 + 180} V${y0 + 330} H${bx} V${y0 + 182}" stroke="${K.dec}" stroke-width="2" fill="none" stroke-dasharray="6 6" class="flowline loopline" marker-end="url(#xa)"/><text x="${(ax + bx) / 2}" y="${y0 + 322}" text-anchor="middle" class="xs" fill="${K.dec}">${esc(F === "diffusion" ? L2("ทำซ้ำทุกขั้น denoise (เช่น 8 หรือ 50 ขั้น)", "repeat for every denoising step (e.g. 8 or 50)") : L2("frame ถัดไป · ปล่อย label ได้หลายตัวต่อ frame", "next frame · may emit several labels per frame"))}</text>`; } }
  LAY = {cols, y0, W, H};
  const svg = $("#xsvg"); if (!S.zoomAnim) svg.setAttribute("viewBox", S.zoom ? S.zoom.vb : `0 0 ${W} ${H}`); svg.innerHTML = s;
  svg.classList.toggle("playing", !!cur);
  $$("#xsvg .xbox[data-x]").forEach(g => g.onclick = () => { stopPlay(); S.open = S.open === g.dataset.x ? null : g.dataset.x; S.step = -1; renderBig(); renderDetail(); if (S.open) setTimeout(() => $("#xdetail").scrollIntoView({behavior: "smooth", block: "nearest"}), 50); });
  $$("#xsvg .xnav").forEach(g => g.onclick = e => { e.stopPropagation(); const k = g.dataset.nav; if (k[0] === "e") S.el = (S.el + (k[1] === "+" ? 1 : -1) + nE) % nE; else S.dl = (S.dl + (k[1] === "+" ? 1 : -1) + nD) % nD; renderBig(); renderStrip(); renderDetail(); });
  renderStrip();
}
function zoomSteps(Lx, x, y, w, comp) { // the inside of one layer, drawn small inside the card while zoomed
  const rows = Object.entries(Lx.regions).map(([rk, R]) => [rk, (ORDER[rk] || []).flat().filter(r => R[r] && R[r].has)]).filter(r => r[1].length);
  let o = "", yy = y + 50; const rh = Math.min(62, (190 - 10 * rows.length) / rows.length);
  rows.forEach(([rk, roles], ri) => {
    o += `<text x="${x + 18}" y="${yy + 12}" class="xs" fill="${C.soft}">${esc(regTitle(rk, Lx).toLowerCase())}</text>`;
    const cw = (w - 36) / Math.max(roles.length, 5);
    roles.forEach((r, k) => { const bx = x + 18 + k * cw; o += `<g class="zs" style="animation-delay:${(ri * roles.length + k) * .18}s"><rect x="${bx}" y="${yy + 18}" width="${cw - 6}" height="${rh - 24}" rx="7" fill="rgba(${K.ovRGB},0.056)" stroke="${comp === "encoder" ? K.blueL : K.dec}" stroke-opacity=".6"/>
      <text x="${bx + (cw - 6) / 2}" y="${yy + 18 + (rh - 24) / 2 + 4}" text-anchor="middle" font-size="${cw < 70 ? 10 : 12}" fill="${C.ink}">${esc(stepName(r, true))}</text></g>`; });
    yy += rh + 6;
  });
  o += `<circle r="5" fill="${K.ink}"><animateMotion dur="${1.6 * rows.length}s" repeatCount="indefinite" path="${rows.map((r, i) => `M${x + 18} ${y + 50 + 18 + (rh - 24) / 2 + i * (rh + 6)} H${x + 18 + r[1].length * (w - 36) / Math.max(r[1].length, 5) - 6}`).join(" ")}"/></circle>`;
  return o;
}

// ---------------- Play: walk the parts, input → output, zoom once, loop once
let PLAY = [], playTimer = null;
function buildPlay() {
  const cols = LAY.cols; PLAY = [];
  cols.forEach(c => { PLAY.push({id: c.stack ? c.id : c.io ? c.kind : c.id, col: c, type: c.io ? "io" : c.stack ? "stack" : "part"});
    if (c.stack && !PLAY.some(p => p.type === "zoom")) PLAY.push({id: c.id, col: c, type: "zoom"}); });
  if (S.M.A.comps.decoder) PLAY.push({id: "out_text", col: cols.find(c => c.kind === "out_text"), type: "loop"});
}
const DUR = {io: 3600, part: 4200, stack: 4600, zoom: 7600, loop: 5200};
function goStep(i) {
  if (!LAY) return; if (!PLAY.length) buildPlay(); clearTimeout(playTimer);
  if (i >= PLAY.length) { stopPlay(); S.step = PLAY.length - 1; $("#bPlay").textContent = t("replay"); return; }
  S.step = Math.max(0, i); S.open = null; const st = PLAY[S.step];
  if (st.type === "zoom") zoomTo(st.col); else if (S.zoom) zoomOut();
  renderBig(); renderIO(); renderDetail(); centerCol(st.col);
  if (S.playing) playTimer = setTimeout(() => goStep(S.step + 1), DUR[st.type]);
}
function centerCol(col) { const sc = $("#xsvg").parentElement, svg = $("#xsvg"); if (!col || sc.scrollWidth <= sc.clientWidth + 4) return; const k = svg.clientWidth / LAY.W; sc.scrollTo({left: Math.max(0, (col.x + col.w / 2) * k - sc.clientWidth / 2), behavior: "smooth"}); }
function startPlay() { buildPlay(); S.playing = true; $("#bPlay").textContent = t("pause"); goStep(S.step >= PLAY.length - 1 || S.step < 0 ? 0 : S.step + 1); $("#big").scrollIntoView({behavior: "smooth", block: "start"}); }
function stopPlay() { S.playing = false; clearTimeout(playTimer); const b = $("#bPlay"); if (b) b.textContent = t("play"); }
function animVB(to, done) { const svg = $("#xsvg"), from = svg.getAttribute("viewBox").split(" ").map(Number), tv = to.split(" ").map(Number), t0 = performance.now(); S.zoomAnim = true;
  (function f(now) { const k = Math.min(1, (now - t0) / 750), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2; svg.setAttribute("viewBox", from.map((v, i) => (v + (tv[i] - v) * e).toFixed(1)).join(" ")); if (k < 1) requestAnimationFrame(f); else { S.zoomAnim = false; done && done(); } })(t0); }
function zoomTo(col) { const zw = Math.max(col.w, 640), pad = 30, x = col.x + col.w / 2 - zw / 2 - pad, y = LAY.y0 - 40, w = zw + 2 * pad + 20, h = 300, ar = LAY.W / LAY.H; let W2 = Math.max(w, h * ar), H2 = W2 / ar; const vb = `${(x + w / 2 - W2 / 2).toFixed(0)} ${(y + h / 2 - H2 / 2).toFixed(0)} ${W2.toFixed(0)} ${H2.toFixed(0)}`; S.zoom = {comp: col.comp, vb}; animVB(vb); }
function zoomOut() { S.zoom = null; animVB(`0 0 ${LAY.W} ${LAY.H}`); }
function renderIO() {
  const box = $("#xio"); const st = S.step >= 0 ? PLAY[S.step] : null;
  $("#bDots").innerHTML = PLAY.map((p, i) => `<i class="${i === S.step ? "on" : i < S.step ? "done" : ""}" data-s="${i}"></i>`).join(""); $$("#bDots i").forEach(d => d.onclick = () => { stopPlay(); goStep(+d.dataset.s); });
  if (!st) { box.innerHTML = `<div class="xempty">${t("big_lead")}</div>`; return; }
  const M = S.M, info = partInfo(st.col, M);
  let title = info.title, what = info.what, opt = info.opt, iv = info.i, ov = info.o, nums = "";
  const p = st.col.stack ? (st.col.comp === "encoder" ? M.A.comps.encoder.layersTot : M.A.comps.decoder.layersTot) : st.col.part;
  if (p && p.has) nums = [[t("f_weights"), fB(p.bytes) + " · " + fP(p.params)], [t("f_stored"), dtLabel(p)], [t("f_compute"), fF(p.flops) + (p.est ? " " + t("est") : "") + (st.col.comp === "decoder" || /lmhead|embed_dec/.test(st.col.kind) ? " " + t("per_token") : "")], M.kind === "ops" ? [t("f_ops"), p.ops] : null].filter(Boolean).map(([k, v]) => `<div class="x3row"><span>${k}</span><b>${esc(String(v))}</b></div>`).join("");
  if (st.type === "zoom") { const Cc = M.A.comps[st.col.comp], Lx = Cc.layers[st.col.comp === "encoder" ? S.el : S.dl];
    const steps = Object.keys(Lx.regions).map(rk => (ORDER[rk] || []).flat().filter(r => Lx.regions[rk][r] && Lx.regions[rk][r].has).map(r => stepName(r, true)).join(" → ")).join("  |  ");
    title = t("zoom_title"); what = t("zoom_text", {steps, n: Cc.layers.length}); iv = info.i; ov = info.o; }
  if (st.type === "loop") { title = t("loop_title"); what = t("loop_text", {t: S.tokens}); iv = ["text", ""]; ov = ["tokens", ""]; opt = ""; }
  box.innerHTML = `<div class="iok">${t("step_of", {i: S.step + 1, n: PLAY.length})}</div>
    <div class="iogrid"><div class="ioside">${iv ? `<div class="xk">${t("io_in")}</div>${viz(iv[0])}<div class="ioshape">${esc(iv[1])}</div>` : ""}</div>
      <div class="iomid"><div class="ioarrow">→</div><h3>${esc(title)}</h3><p>${esc(what)}</p>${opt ? `<div class="x3opt"><b>${t("opt_lens")}</b>${esc(opt)}</div>` : ""}</div>
      <div class="ioside">${ov ? `<div class="xk">${t("io_out")}</div>${viz(ov[0])}<div class="ioshape">${esc(ov[1])}</div>` : ""}</div>
      <div class="ionums">${nums}</div></div>`;
}

// ---------------- per-layer strip
function renderStrip() {
  const A = S.M.A, E = A.comps.encoder, D = A.comps.decoder, L = LENS[S.lens];
  const all = [...(E ? E.layers.map(l => ["e", l]) : []), ...(D ? D.layers.map(l => ["d", l]) : [])];
  const mx = Math.max(...all.map(([, l]) => L.v(l.tot) || 0), 1e-9);
  $("#xstrip").innerHTML = `<span class="xk">${esc(t("per_layer", {lens: t("lens_" + S.lens)}))}</span>` + all.map(([c, l]) => {
    const v = L.v(l.tot) || 0, h = S.lens === "precision" ? 26 : Math.max(3, 26 * v / mx), on = c === "e" ? l.idx === (E.layers[S.el] || {}).idx : l.idx === (D.layers[S.dl] || {}).idx;
    return `<button class="xsb${on ? " on" : ""}" data-c="${c}" data-i="${l.idx}" title="${c === "e" ? "encoder" : "decoder"} L${l.idx} · ${esc(L.unit(l.tot))}"><i style="height:${h}px;background:${S.lens === "precision" ? dtColor(l.tot) : c === "e" ? C.blue : C.mag}"></i><span>${c.toUpperCase()}${l.idx}</span></button>`; }).join("");
  $$("#xstrip .xsb").forEach(b => b.onclick = () => { const i = +b.dataset.i; if (b.dataset.c === "e") S.el = E.layers.findIndex(l => l.idx === i); else S.dl = D.layers.findIndex(l => l.idx === i); renderBig(); renderDetail(); });
}

// ---------------- detail of one part
const STEP = {
  ln: ["LayerNorm", "ปรับสเกลเวกเตอร์ก่อนเข้า block · ใน TFLite แตกเป็น op เล็กๆ ~12 ตัว", "Rescale the vector before the block · in TFLite it becomes ~12 small ops"],
  qkv: ["QKV (fused)", "Q, K, V รวมเป็นเมทริกซ์เดียว", "Q, K and V fused into one matrix"],
  q: ["Q · query", "“ตอนนี้ฉันมองหาอะไร”", "“What am I looking for?”"], k: ["K · key", "“ฉันมีอะไรให้ดู”", "“What do I offer?”"], v: ["V · value", "“ถ้าเลือกฉัน จะได้ข้อมูลนี้”", "“What you get if you pick me”"],
  cache: ["KV cache", "ต่อ K,V ของ token นี้ท้าย cache ให้ token ต่อไปไม่ต้องคิดใหม่", "Append this token's K,V to the cache so later tokens don't recompute it"],
  reshape: ["split heads", "reshape/transpose แยกเป็นหลาย head แล้วรวมกลับ — ย้าย memory ไม่ได้คำนวณ", "reshape/transpose into heads and back — memory moves, no math"],
  scores: ["Q·Kᵀ", "ทุกตำแหน่งให้คะแนนกัน · ไม่มี weight แต่ compute โตตามความยาว²", "every position scores every other · no weights, compute grows with length²"],
  softmax: ["softmax", "แปลงคะแนนเป็นสัดส่วนรวมเป็น 1", "turn scores into weights that sum to 1"], context: ["·V", "ถ่วงน้ำหนัก V ตามคะแนน", "weight the values by the scores"],
  out: ["out proj", "รวมทุก head กลับเป็นเวกเตอร์เดียว", "merge the heads back into one vector"], res: ["⊕ residual", "บวกค่าก่อนเข้า block กลับเข้าไป", "add the block's input back"],
  gate: ["gate", "ตัวคุมว่าข้อมูลไหนผ่าน (SwiGLU)", "decides what passes (SwiGLU)"], fc1: ["up", "ขยายแต่ละตำแหน่งเป็นหลายเท่า", "expand each position several times"], act: ["act", "ฟังก์ชันไม่เชิงเส้น", "the non-linearity"], fc2: ["down", "บีบกลับขนาดเดิม", "project back down"],
  conv: ["conv / patch", "แปลงสัญญาณดิบเป็นเวกเตอร์", "turn the raw signal into vectors"], embed: ["embedding", "token id → แถวในตาราง", "token id → a row of the table"], pos: ["+ position", "บอกตำแหน่ง", "add position"],
  mask: ["masks", "mask และตำแหน่งที่จะเขียนลง cache", "masks and the cache write position"], io: ["read caches", "อ่าน cache ที่ส่งเข้ามา", "read the incoming caches"], glue: ["glue", "ปรับรูปร่าง", "reshape glue"],
  crosskv: ["K/V proj", "แปลงผล encoder เป็น K,V ให้ decoder", "turn the encoder output into K,V for the decoder"], logits: ["logits", "คูณกับทั้งตารางคำศัพท์ → คะแนนทุกคำ", "multiply by the whole vocabulary → a score per token"],
  state: ["state update", "อัปเดต state ขนาดคงที่แทนการเก็บ K,V ทุก token (Gated DeltaNet)", "update a fixed-size state instead of storing K,V for every token (Gated DeltaNet)"],
  proj: ["projector", "แปลงเวกเตอร์ของ encoder ให้เป็นขนาดของ LLM", "map the encoder's vectors to the LLM's width"],
  mtp: ["MTP head", "ทาย token ถัดๆ ไปล่วงหน้า ใช้เร่งด้วย speculative decoding", "drafts the next tokens ahead, for speculative decoding"],
  router: ["router", "ให้คะแนนทุก expert แล้วเลือก top-k ต่อ token", "scores every expert and keeps the top-k per token"],
  expert: ["experts", "FFN หลายชุด · แต่ละ token ใช้แค่ไม่กี่ชุด → weight มาก compute น้อย", "many FFNs; each token uses only a few → lots of weight, little compute"],
  shared: ["shared expert", "expert ที่ทุก token ใช้เสมอ", "an expert every token always uses"],
  in: ["in proj", "แตกเป็น x, z, B, C, Δt ในครั้งเดียว", "split into x, z, B, C, Δt in one matrix"],
  mod: ["adaLN", "แปลง timestep เป็น shift/scale/gate ของทุก norm ใน block", "turns the timestep into shift / scale / gate for every norm in the block"],
  time: ["timestep MLP", "เลขขั้น denoise → เวกเตอร์", "denoising step → vector"], rnn: ["LSTM", "จำ label ที่ปล่อยไปแล้ว", "remembers the labels emitted so far"],
  joint: ["joint", "รวมเสียง + label แล้วให้คะแนนทุก label (และช่วงกระโดดของ TDT)", "adds audio + label context and scores every label (and TDT duration)"],
  pool: ["pooler", "สรุปเป็นเวกเตอร์เดียว", "summarize into one vector"], cls: ["classifier", "คะแนนของแต่ละคลาส", "a score per class"], misc: ["other", "อื่นๆ", "other"], seam: ["seams", "QUANTIZE/DEQUANTIZE", "QUANTIZE/DEQUANTIZE"],
};
const stepName = (r, short) => (STEP[r] || [r])[0];
const ORDER = {self: [["ln"], ["qkv", "q", "k", "v"], ["cache"], ["reshape"], ["scores"], ["softmax"], ["context"], ["out"], ["res"]], cross: [["ln"], ["q", "k", "v"], ["reshape"], ["scores"], ["softmax"], ["context"], ["out"], ["res"]],
  ffn: [["ln"], ["gate", "fc1"], ["act"], ["fc2"], ["res"]], linear: [["ln"], ["qkv", "in"], ["gate"], ["conv"], ["state"], ["out"], ["res"]],
  moe: [["ln"], ["router"], ["expert"], ["shared"], ["act"], ["res"]], conv: [["ln"], ["conv"], ["res"]], mod: [["mod"]], rnn: [["rnn"]], stem: [["conv"], ["embed"], ["time"], ["mod"], ["glue"], ["pos"], ["ln"], ["mask"], ["io"]], head: [["ln"], ["crosskv"], ["proj"], ["joint"], ["logits"], ["mtp"], ["pool"], ["cls"], ["out"], ["glue"]]};
function renderDetail() {
  const box = $("#xdetail"), id = S.open, M = S.M, E = M.A.comps.encoder, D = M.A.comps.decoder;
  if (!id) { box.innerHTML = `<div class="xempty">${t("click_hint")}</div>`; return; }
  let parts = null, order = null, crumb = "", title = "", what = "", opt = "";
  const reg = id.match(/^([ed])_(self|cross|ffn|linear|moe|conv|mod|rnn)$/);
  if (reg) { const Cc = reg[1] === "e" ? E : D, L = Cc.layers[reg[1] === "e" ? S.el : S.dl]; parts = L.regions[reg[2]] || {}; order = ORDER[reg[2]] || [Object.keys(parts)]; crumb = `${reg[1] === "e" ? "Encoder" : "Decoder"} › ${t("layer")} ${L.idx}`; title = regTitle(reg[2], L); }
  else { const col = (LAY ? LAY.cols : []).find(c => c.id === id || c.kind === id); if (col) { const info = partInfo(col, M); title = info.title; what = info.what; opt = info.opt; if (col.part) { parts = col.id === "x_pj" ? Object.assign({}, ...(M.A.extra || []).filter(x => /joint|prediction/.test(x.role)).map(x => x.parts), E.head) : col.id === "e_stem" ? E.stem : col.id === "e_head" ? E.head : col.id === "d_stem" ? D.stem : D.head; order = /stem/.test(col.id) ? ORDER.stem : ORDER.head; } crumb = col.comp || (col.where || "").split(".")[0]; }
    if (id === "d_stem") { parts = D.stem; order = ORDER.stem; title = "Token embedding"; crumb = "decoder"; } }
  const tot = parts ? sumParts(parts) : null;
  let svg = "";
  if (parts) {
    const steps = order.map(g => g.filter(r => parts[r] && parts[r].has)).filter(g => g.length); Object.keys(parts).forEach(r => { if (r !== "_tot" && !order.flat().includes(r) && parts[r].has) steps.push([r]); });
    const mx = Math.max(...Object.entries(parts).filter(([k]) => k !== "_tot").map(([, p]) => LENS[S.lens].v(p) || 0), 1e-9);
    const colW = 140, bw = 124, bh = 92, gap = 14, W = Math.max(700, steps.length * colW + 40), rows = Math.max(...steps.map(g => g.length)), H = rows * (bh + gap) + 40, cy = H / 2 + 4;
    svg = `<svg viewBox="0 0 ${W} ${H}" style="min-width:${Math.round(W * .8)}px"><defs><marker id="xa2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="${K.dim2}"/></marker></defs>`;
    steps.forEach((g, i) => { const x = 20 + i * colW, yA = cy - (g.length * (bh + gap) - gap) / 2;
      if (i) svg += `<path d="M${x - colW + bw + 2} ${cy} H${x - 4}" stroke="${K.dim2}" stroke-width="2" class="flowline" marker-end="url(#xa2)"/>`;
      g.forEach((r, k) => { const p = parts[r], y = yA + k * (bh + gap), f = lensFill(p, mx), nm = stepName(r), wt = p.w[0] ? p.w[0].shape.join("×") : (p.out ? `[${p.out.join(",")}]` : "");
        svg += `<g class="xstep" data-r="${r}"><rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="12" fill="${f.fill}" stroke="${f.stroke || (p.seams ? K.neg : `rgba(${K.ovRGB},0.200)`)}" stroke-width="1.4"/>
          <text x="${x + bw / 2}" y="${y + 22}" text-anchor="middle" class="xt" font-size="${nm.length > 14 ? 12.5 : 14.5}">${esc(nm)}</text><text x="${x + bw / 2}" y="${y + 40}" text-anchor="middle" class="xs">${esc(wt)}</text>
          <text x="${x + bw / 2}" y="${y + 62}" text-anchor="middle" class="xm" font-size="14">${esc(LENS[S.lens].unit(p))}</text><text x="${x + bw / 2}" y="${y + 78}" text-anchor="middle" class="xs">${p.ops ? p.ops + " op" + (p.ops > 1 ? "s" : "") : ""}${p.seams ? " · " + p.seams + " seam" : ""}</text></g>`; }); });
    svg += "</svg>";
  }
  const Lsel = reg ? (reg[1] === "e" ? E : D).layers[reg[1] === "e" ? S.el : S.dl] : null;
  const layerFacts = Lsel && Lsel.tot && Lsel.tot.has ? [[t("f_layer"), fB(Lsel.tot.bytes)], ...["self", "cross", "linear", "ffn", "moe"].filter(r => Lsel.regions[r] && Lsel.regions[r]._tot.has)
    .map(r => ["  " + (r === "ffn" || r === "moe" ? t("f_ffn") : r === "cross" ? "cross-attention" : t("f_attn")), fB(Lsel.regions[r]._tot.bytes)])] : [];
  const facts = tot ? [[t("f_weights") + (Lsel ? " (" + esc(title) + ")" : ""), fB(tot.bytes) + " · " + fP(tot.params)], [t("f_stored"), dtLabel(tot)], [t("f_compute"), fF(tot.flops) + (tot.est ? " " + t("est") : "")], M.kind === "ops" ? [t("f_ops"), tot.ops] : null, M.kind === "ops" ? [t("f_seams"), tot.seams] : null, tot._frag != null ? [t("f_frag"), (tot._frag * 100).toFixed(1) + "%"] : null].filter(Boolean) : [];
  box.innerHTML = `<div class="xdh"><div><div class="xk">${esc(crumb)}</div><h3>${esc(title)}</h3>${what ? `<p>${esc(what)}</p>` : ""}</div><button class="btn sm" id="xclose">✕ ${t("close")}</button></div>
    <div class="xdg"><div><div class="hscroll xsteps">${svg || ""}</div><div class="xhov" id="xhov">${parts ? t("see_ops_note") : ""}</div></div>
    <div class="xfacts">${layerFacts.map(([k, v]) => `<div class="x3row"><span>${k}</span><b>${esc(String(v))}</b></div>`).join("")}${facts.map(([k, v]) => `<div class="x3row"><span>${k}</span><b>${esc(String(v))}</b></div>`).join("")}${opt ? `<div class="x3opt"><b>${t("opt_lens")}</b>${esc(opt)}</div>` : ""}</div></div>`;
  $("#xclose").onclick = () => { S.open = null; renderBig(); renderDetail(); };
  $$("#xdetail .xstep").forEach(g => g.onclick = g.onmouseenter = () => { const e = STEP[g.dataset.r] || [g.dataset.r, "", ""], p = parts[g.dataset.r]; const types = Object.entries(p.types).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}×${v}`).join(", ");
    $("#xhov").innerHTML = `<b>${esc(e[0])}</b> — ${esc(L2(e[1], e[2]))}${types ? `<br><span class="mono" style="font-size:12px;color:var(--dim)">${esc(types)}</span>` : ""}`; });
}

// ================================================================== architecture card (automatic recovery from the weight manifest)
const BLK = {SSM: "#1a8f6a", "SSM+MoE": "#1a8f6a", MoE: "#d4651c", Attention: "#1b5fd1", Transformer: "#1b5fd1", "Attn+MoE": "#d4651c", "Attn+Cross": "#1b5fd1", Conformer: "#b98900", DiT: "#1b5fd1", FFN: "#a9a9a3", RNN: "#b98900", Conv: "#a9a9a3", Block: "#a9a9a3"};
const ROLE_C = {backbone: ["#fbe9dd", "#7a3409"], encoder: ["#e3ecfb", "#123f8c"], adapter: ["#f4ecd6", "#5c4300"], graph: ["#dcf3e6", "#0f4a2c"], prediction: ["#f4ecd6", "#5c4300"], joint: ["#ffedea", "#831c1e"], aux: ["#efefeb", "#4d4d4a"]};
setTheme(document.documentElement.dataset.theme || "dark");
const KIND_T = {attn: ["attention", "attention"], cross: ["cross-attention", "cross-attention"], ffn: ["feed-forward", "feed-forward"], moe: ["mixture of experts", "mixture of experts"], ssm: ["SSM / linear attention", "SSM / linear attention"], conv: ["conv module", "conv module"], mod: ["adaLN modulation", "adaLN modulation"], rnn: ["recurrent cell", "recurrent cell"], norm: ["norm", "norm"], other: ["อื่นๆ", "other"]};
function renderArch() {
  const box = $("#archBox"); if (!box) return; const M = S.M, I = M.IR;
  if (!I) { box.innerHTML = `<div class="xempty">${t("arch_ops")}</div>`; return; }
  const comps = I.comps, W = 1600, n = comps.length, gap = 56, bw = Math.min(360, (W - 40 - gap * (n - 1)) / n), bh = 236, y = 34;
  const linked = c => I.edges.some(e => e.from === c.id) ? 1 : 0;
  const order = [...comps].sort((a, b) => { const r = c => ({encoder: 0, graph: 1, adapter: 2, backbone: 3, prediction: 4, aux: 5, joint: 6})[c.role] ?? 5; return r(a) - r(b) || linked(a) - linked(b) || a.params - b.params; });
  const tot = W - 40 - (bw * n + gap * (n - 1)); let x = 20 + tot / 2; const pos = {};
  let svg = `<defs><marker id="aa" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="${K.dim}"/></marker></defs>`;
  order.forEach(c => { pos[c.id] = {x, w: bw}; x += bw + gap; });
  I.edges.forEach(e => { const a = pos[e.from], b = pos[e.to]; if (!a || !b) return; const x1 = a.x + a.w, x2 = b.x, my = y + bh / 2;
    const between = Object.values(pos).some(q => q.x > x1 && q.x + q.w < x2);
    svg += x2 > x1 && !between ? `<path d="M${x1 + 2} ${my} H${x2 - 4}" stroke="${K.dim}" stroke-width="1.6" fill="none" marker-end="url(#aa)"/>` : x2 > x1 ? `<path d="M${a.x + a.w * .6} ${y} C ${a.x + a.w * .6} ${y - 30}, ${b.x + b.w * .4} ${y - 30}, ${b.x + b.w * .4} ${y - 3}" stroke="${K.dim}" stroke-width="1.6" fill="none" marker-end="url(#aa)"/>` : `<path d="M${a.x + a.w / 2} ${y} C ${a.x + a.w / 2} ${y - 30}, ${b.x + b.w / 2} ${y - 30}, ${b.x + b.w / 2} ${y - 3}" stroke="${K.dim}" stroke-width="1.6" fill="none" marker-end="url(#aa)"/>`; });
  const sel = S.archSel && comps.find(c => c.id === S.archSel) ? S.archSel : (comps.find(c => c.role === "backbone") || comps[0]).id; S.archSel = sel;
  order.forEach(c => { const p = pos[c.id], [bg, fg] = ROLE_C[c.role] || ROLE_C.aux, on = c.id === sel, labels = c.stacks.flatMap(st => st.typeOf.map(i => st.types[i].label)), rl = t("role_" + c.role);
    svg += `<g class="acomp" data-c="${esc(c.id)}" tabindex="0" role="button" aria-label="${esc(rl + " " + c.id)}"><rect x="${p.x}" y="${y}" width="${p.w}" height="${bh}" rx="8" fill="${K.surf}" stroke="${on ? K.ink : K.line}" stroke-width="${on ? 2 : 1}"/>
      <rect x="${p.x + 16}" y="${y + 16}" width="${20 + rl.length * 9}" height="26" rx="13" fill="${bg}"/><text x="${p.x + 26}" y="${y + 34}" class="at" fill="${fg}">${esc(rl)}</text>
      <text x="${p.x + 16}" y="${y + 70}" class="am">${esc(c.id.length * 10 > p.w - 32 ? "…" + c.id.slice(-Math.floor((p.w - 44) / 10)) : c.id)}</text>
      <text x="${p.x + 16}" y="${y + 106}" class="ab">${esc(fP(c.params))}</text><text x="${p.x + p.w - 16}" y="${y + 106}" text-anchor="end" class="as">${esc(fB(c.bytes))}</text>
      <text x="${p.x + 16}" y="${y + 132}" class="as">${esc([c.d ? "d " + c.d : "", c.layers ? c.layers + " " + t("layers_w") : "", c.modality && !/text|features|labels/.test(c.modality) ? t("mod_" + c.modality) : ""].filter(Boolean).join(" · "))}</text>`;
    if (labels.length) { const cw = (p.w - 32) / labels.length; labels.forEach((l, i) => svg += `<rect x="${(p.x + 16 + i * cw).toFixed(2)}" y="${y + 150}" width="${Math.max(.8, cw - (cw > 4 ? 1.5 : 0)).toFixed(2)}" height="24" fill="${BLK[l] || "#a9a9a3"}"/>`);
      svg += `<text x="${p.x + 16}" y="${y + 206}" class="as">${esc(summLabels(labels))}</text>`; }
    else { const per = Math.floor((p.w - 32) / 8.4), words = String((c.ev || [])[0] || "").split(" ").flatMap(wd => wd.length > per ? wd.match(new RegExp(".{1," + (per - 1) + "}", "g")) : [wd]), lines = [""]; words.forEach(wd => { if ((lines[lines.length - 1] + " " + wd).length > per) lines.push(wd); else lines[lines.length - 1] += (lines[lines.length - 1] ? " " : "") + wd; });
      lines.slice(0, 4).forEach((ln, k) => svg += `<text x="${p.x + 16}" y="${y + 160 + k * 20}" class="as">${esc(k === 3 && lines.length > 4 ? ln + " …" : ln)}</text>`); }
    svg += `</g>`; });
  const GRP = l => /SSM/.test(l) ? "SSM" : /MoE/.test(l) ? "MoE" : /Conformer/.test(l) ? "Conformer" : /Attention|Transformer|DiT|Cross/.test(l) ? "Attention" : "Other", GC = {Attention: K.enc, SSM: K.ssm, MoE: K.dec, Conformer: K.conf, Other: "#a9a9a3"};
  const used = [...new Set(comps.flatMap(c => c.stacks.flatMap(st => st.types.map(t2 => GRP(t2.label)))))];
  const C0 = comps.find(c => c.id === sel), m = M.meta, hp = Object.assign({}, I.hp || {}), fz = m.fusedAll && m.fusedAll[C0.id];
  const KC = {attn: "Attention", cross: "Attn+Cross", ffn: "FFN", moe: "MoE", ssm: "SSM", conv: "Conformer", mod: "DiT", rnn: "RNN"};
  const rows = C0.stacks.flatMap(st => st.types.map(ty => {
    const regs = ty.regions.filter(r => r.kind !== "norm" && r.kind !== "other"), tp = regs.reduce((a, r) => a + r.params, 0) || 1;
    return `<div class="atype"><div class="ath"><i style="background:${BLK[ty.label] || "#a9a9a3"}"></i><b>${esc(ty.label)}</b><span>× ${ty.count}</span><span class="dim mono">${esc(st.path)}</span><span class="dim">d ${ty.d || C0.d}</span></div>
      <div class="aregs">${regs.map(r => `<div class="areg"><div class="k">${esc(L2(...(KIND_T[r.kind] || [r.kind, r.kind])))}<span>${esc(r.key)}</span></div><div class="bar"><i style="width:${Math.max(2, r.params / tp * 100).toFixed(1)}%;background:${BLK[KC[r.kind]] || "#a9a9a3"}"></i></div><div class="pc">${(r.params / tp * 100).toFixed(0)}%</div>
        <div class="ev">${esc(r.ev.join(" · ") || "")}${r.E && !r.ev.join(" ").includes("experts") ? ` · ${r.E} experts` : ""}${r.x && r.kv && r.kv < r.x ? ` · GQA ${r.x / r.kv}:1` : ""}</div></div>`).join("")}</div></div>`; })).join("");
  // what each value rests on: measured from shapes, read from config.json, or estimated
  const isB = C0.role === "backbone", heads = fz && fz.heads ? fz.heads : isB ? hp.heads : null, kvh = fz && fz.kv_heads ? fz.kv_heads : isB ? hp.kv_heads : null, hd = fz && fz.head_dim ? fz.head_dim : isB ? hp.head_dim : null;
  const hsrc = fz && fz.heads ? "cfg" : isB && hp.head_dim_from === "q/k-norm width" ? "meas" : "est";
  const tag = k => `<span class="src ${k}" title="${esc(t("srcd_" + k))}">${esc(t("srct_" + k))}</span>`;
  const row = (k, v, src) => v == null || v === 0 || v === "" ? "" : `<div class="x3row"><span>${k}</span><b>${esc(String(v))}${src ? tag(src) : ""}</b></div>`;
  const cons = fz && fz.checked ? (fz.consistent ? tag("ok") : tag("bad")) : "";
  box.innerHTML = `<div class="hscroll"><svg id="archSvg" viewBox="0 0 ${W} ${bh + 52}" style="min-width:${Math.max(900, n * 240)}px" role="img" aria-label="${esc(t("arch_h2"))}">${svg}</svg></div>
    <div class="alegend">${used.map(l => `<span><i style="background:${GC[l]}"></i>${esc(t("grp_" + l))}</span>`).join("")}</div>
    <div class="agrid"><div><div class="ahead">${esc(t("role_" + C0.role))} <span class="dim mono" style="font-weight:400">${esc(C0.id)}</span></div>${rows || `<p class="dim" style="margin-top:8px">${esc((C0.ev || []).join(" · ") || t("arch_noblocks"))}</p>`}</div>
      <div class="props"><div class="ahead">${t("arch_hp")}</div>
        ${row(t("hp_family"), t("fam_" + fam(M)) + (I.family === "encoder" && fam(M) === "encoder" && !M.cfg && hp.vocab ? " " + t("hp_tiedq") : ""), M.cfg && /encoder|decoder/.test(fam(M)) ? "cfg" : null)}
        ${row(t("hp_params"), fP(isB ? I.totals.params : C0.params), "meas")}${row(t("hp_stored"), fB(isB ? I.totals.bytes : C0.bytes), "meas")}
        ${row("d_model", C0.d, "meas")}${row(t("hp_layers"), C0.layers, "meas")}
        ${row(t("hp_heads"), heads && kvh ? heads + " / " + kvh + " kv" : null, heads ? hsrc : null)}${row("head_dim", hd, hd ? hsrc : null)}
        ${fz && fz.checked ? `<div class="x3row"><span>${t("hp_check")}</span><b>${cons}</b></div>` : ""}
        ${isB ? row("FFN", hp.ffn, "meas") : ""}${isB ? row(t("hp_experts"), hp.experts ? hp.experts + (m.topk ? ` · top-${m.topk}` : "") : null, hp.experts ? (m.topk ? "cfg" : "meas") : null) : ""}${isB ? row(t("hp_vocab"), hp.vocab, "meas") : ""}
        ${isB ? row(t("hp_kv"), kvText(M, I, hp), "meas") : ""}${row(t("hp_quant"), Object.keys(I.quant || {}).length ? Object.entries(I.quant).map(([k, v]) => k.toUpperCase() + " ×" + v).join(", ") : null, "meas")}
        <div class="ahead" style="margin-top:18px">${t("arch_ev")}</div><ul class="aev">${[...new Set([...I.evidence, ...comps.flatMap(c => (c.ev || []).filter(e => !I.evidence.some(x => x.includes(e))).map(e => c.id + ": " + e))])].slice(0, 8).map(e => `<li>${esc(e)}</li>`).join("") || `<li>${t("arch_ev_none")}</li>`}</ul></div></div>`;
  $$("#archSvg .acomp").forEach(g => { g.onclick = () => { S.archSel = g.dataset.c; renderArch(); }; g.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); S.archSel = g.dataset.c; renderArch(); } }; });
}
// KV cache per token: layer by layer (plan.js) when layers share caches or slide over a window, else the engine's value
function kvText(M, I, hp) {
  if (!(hp.kv_bytes_per_token_bf16 && /decoder|prefix|encdec/.test(fam(M)))) return null;
  try { const sh = {}; for (const [n, x] of Object.entries(M.T || {})) sh[n] = x.shape; const K = Plan.kvProfile(I, sh, M.cfg || {});
    if (K && (K.shared || K.sliding)) return fB(K.per_token) + " / token" + (K.sliding ? " + ≤" + fB(K.sliding_cap) : "") + ` (${K.cached}/${K.layers} layers)`; } catch (e) {}
  return fB(hp.kv_bytes_per_token_bf16) + " / token";
}
function summLabels(ls) { const runs = []; ls.forEach(l => { const r = runs[runs.length - 1]; if (r && r[0] === l) r[1]++; else runs.push([l, 1]); }); const cnt = {}; ls.forEach(l => cnt[l] = (cnt[l] || 0) + 1);
  return runs.length > 6 ? Object.entries(cnt).map(([l, n]) => `${n}× ${l}`).join(" · ") : runs.map(([l, n]) => `${n}× ${l}`).join(" → "); }

// ================================================================== layer map
function renderMap() {
  const M = S.M, A = M.A, svg = $("#mapSvg"); const rows = [];
  ["encoder", "decoder"].forEach(c => { const Cc = A.comps[c]; if (!Cc) return; // name the blocks around the layers by what they hold, so "the 212 MB block" can be found
    const has = (o, k) => o && Object.keys(o).some(x => x !== "_tot" && x.startsWith(k) && o[x].has);
    const tiedHead = Cc.headTot && Cc.headTot.has && !Cc.headTot.bytes;
    const stemLab = has(Cc.stem, "embed") ? (c === "decoder" && (tiedHead || has(Cc.stem, "embed") && !has(Cc.head, "logits")) ? t("lab_tied") : t("lab_embed")) : has(Cc.stem, "conv") ? t("lab_conv") : "stem";
    const headLab = has(Cc.head, "crosskv") ? t("lab_crosskv") : t("lab_head");
    rows.push({c, blocks: [{id: c + ".stem", p: Cc.stemTot, lab: stemLab}, ...Cc.layers.map(L => ({id: c + ".L" + L.idx, p: L.tot, L, lab: String(L.idx)})), {id: c + ".head", p: Cc.headTot, lab: headLab}].filter(b => b.p.has)}); });
  const rd = c => c === "decoder" ? S.tokens : 1;
  const wOf = (b, c) => S.mapLens === "read" ? b.p.bytes * rd(c) : S.mapLens === "frag" ? 1 : Math.max(b.p.bytes, b.p.params * 2, 1);
  const maxRow = Math.max(...rows.map(r => r.blocks.reduce((a, b) => a + wOf(b, r.c), 0))), padL = 190, px = (1280 - padL - 20) / maxRow;
  const fr = rows.flatMap(r => r.blocks.map(b => b.p._frag || 0)).filter(x => x > 0), fmx = Math.max(...fr, 1e-9), fmn = Math.min(...fr, fmx);
  let out = "", y = 30;
  rows.forEach(r => { const h = 112, tot = r.blocks.reduce((a, b) => a + wOf(b, r.c), 0), col = r.c === "encoder" ? C.blue : C.mag;
    out += `<rect x="0" y="${y}" width="4" height="${h}" rx="2" fill="${col}"/><text x="16" y="${y + 22}" class="mt">${r.c === "encoder" ? "Encoder" : "Decoder"}</text><text x="16" y="${y + 42}" class="ms">${r.c === "decoder" ? t("runs_every") : t("runs_once")}</text>
      <text x="16" y="${y + 62}" class="mm">${S.mapLens === "read" ? fB(tot) : S.mapLens === "frag" ? r.blocks.length + " blocks" : fB(r.blocks.reduce((a, b) => a + b.p.bytes, 0))}</text>`;
    let x = padL;
    r.blocks.forEach(b => { const w = Math.max(2, wOf(b, r.c) * px - 2.5); let fill = col, op = 1;
      if (S.mapLens === "prec") fill = dtColor(b.p); if (S.mapLens === "frag") { const s = b.p._frag; const k = !(s > 0) ? -1 : (Math.log(s) - Math.log(fmn)) / ((Math.log(fmx) - Math.log(fmn)) || 1); fill = k < 0 ? K.active : `rgb(${Math.round(60 + 170 * k)},${Math.round(40 + 10 * (1 - k))},${Math.round(120 - 80 * k)})`; }
      out += `<g class="blk" data-c="${r.c}" data-l="${b.L ? b.L.idx : ""}" data-id="${b.id}"><title>${esc(b.id)} · ${fB(b.p.bytes)} · ${esc(dtLabel(b.p))}${b.p._frag != null ? " · int4 err " + (b.p._frag * 100).toFixed(1) + "%" : ""}</title>`;
      if (b.L && (S.mapLens === "size" || S.mapLens === "read") && w > 8) { const att = ["self", "cross", "linear"].reduce((a, r) => a + (b.L.regions[r] ? b.L.regions[r]._tot.bytes : 0), 0), fa = b.p.bytes ? att / b.p.bytes : .33;
        out += `<rect x="${x}" y="${y}" width="${w}" height="${h * fa}" rx="3" fill="${fill}" opacity=".55"/><rect x="${x}" y="${y + h * fa}" width="${w}" height="${h * (1 - fa)}" fill="${fill}"/>`; }
      else out += `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${fill}" opacity="${op}"/>`;
      out += `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="none" stroke="rgba(0,0,0,.3)"/>`;
      if (w > b.lab.length * 7.5) out += `<text x="${x + w / 2}" y="${y - 7}" text-anchor="middle" class="ms">${esc(b.lab)}</text>`;
      else if (!b.L && w > 3 && b.p.bytes) out += `<text x="${x + w / 2}" y="${y + h + 15}" text-anchor="${x + w / 2 > 1150 ? "end" : x < padL + 30 ? "start" : "middle"}" class="ms">${esc(b.lab)}</text>`;
      // the value itself, written in the block (sideways when the block is narrow)
      if (S.mapLens === "size" || S.mapLens === "read") { const v = fB(S.mapLens === "read" ? b.p.bytes * rd(r.c) : b.p.bytes);
        if (w > v.length * 9 + 10) out += `<text x="${x + w / 2}" y="${y + h / 2 + 6}" text-anchor="middle" class="mv mvb">${v}</text>`;
        else if (w >= 13) out += `<text transform="translate(${x + w / 2 + 4.5} ${y + h - 8}) rotate(-90)" class="mv">${v}</text>`; }
      if (S.mapLens === "frag" && b.p._frag > 0 && w > 30) out += `<text x="${x + w / 2}" y="${y + h / 2 + 5}" text-anchor="middle" class="mm" fill="#fff">${(b.p._frag * 100).toFixed(1)}</text>`;
      out += "</g>"; x += wOf(b, r.c) * px; });
    y += h + 40; });
  svg.setAttribute("viewBox", `0 0 1280 ${y}`); svg.innerHTML = out;
  $$("#mapSvg .blk").forEach(g => g.onclick = () => { const c = g.dataset.c, l = g.dataset.l; const Cc = A.comps[c];
    if (l !== "") { if (c === "encoder") S.el = Cc.layers.findIndex(L => String(L.idx) === l); else S.dl = Cc.layers.findIndex(L => String(L.idx) === l); S.open = (c === "encoder" ? "e_" : "d_") + "self"; }
    else S.open = g.dataset.id.endsWith("stem") ? (c === "encoder" ? "e_stem" : "d_stem") : (c === "encoder" ? "e_head" : "d_head");
    stopPlay(); S.step = -1; renderBig(); renderDetail(); renderIO(); $("#xdetail").scrollIntoView({behavior: "smooth", block: "center"}); });
  $$("[data-ml]").forEach(b => { b.classList.toggle("on", b.dataset.ml === S.mapLens); b.disabled = b.dataset.ml === "frag" && !M.hasFrag; });
  $("#mapNote").textContent = S.mapLens === "read" ? t("map_read_note") : S.mapLens === "size" ? t("map_light") : "";
}

// ================================================================== where to cut (src/plan.js, shared with the Python package)
function cutParse(j) { const v = (j.model || {}).vocab; let list = [];
  if (v && !Array.isArray(v)) { for (const [k, i] of Object.entries(v)) list[i] = k; } else list = (v || []).map(x => Array.isArray(x) ? x[0] : x);
  (j.added_tokens || []).forEach(a => { list[a.id] = a.special ? "<" + a.content.replace(/^<|>$/g, "") + ">" : a.content; });
  for (let i = 0; i < list.length; i++) if (list[i] == null) list[i] = "<unused>";
  return {list, bl: /ByteLevel/.test(JSON.stringify([j.decoder, j.pre_tokenizer]))}; }
function renderCut() {
  const box = $("#cutBox"); if (!box) return; const M = S.M, I = M.IR;
  if (!I || !M.T) { box.style.display = "none"; return; } box.style.display = "";
  const shapes = {}; for (const [n, x] of Object.entries(M.T)) shapes[n] = x.shape;
  const total = (I.totals && I.totals.params) || I.comps.reduce((a, c) => a + c.params, 0);
  const st = M.cut || (M.cut = {langs: ["th", "en"], text: false, keep: null, counts: null, n: null, msg: ""});
  if (S.pendingTok && !st.pieces) { const j = S.pendingTok; S.pendingTok = null; setTimeout(() => { try { st.pieces = cutParse(j); const r = Plan.vocabScripts(st.pieces.list, {byte_level: st.pieces.bl}); st.n = r.total; st.keep = r.keep(st.langs); renderCut(); } catch (e) {} }, 0); }
  let P; try { P = Plan.plan(I, shapes, M.cfg || {}, {total, keep_tokens: st.keep, name: M.label || ""}); } catch (e) { box.style.display = "none"; return; }
  const pc = x => (100 * x / total).toFixed(x / total < 0.01 ? 1 : 0) + "%", n0 = x => x.toLocaleString("en-US");
  const kind = id => /vision|visual|image|siglip|vit/i.test(id) ? L2("ส่วนภาพ (vision tower)", "vision") : /audio|speech|whisper|conformer/i.test(id) ? L2("ส่วนเสียง (audio tower)", "audio") : L2("ส่วนอินพุต", "input");
  let cut = 0;
  const items = P.levers.map(l => {
    let title, det, how, saves = l.saves;
    if (l.id === "vocab") { title = t("cut_vocab_t"); det = t("cut_vocab_d", {v: n0(l.vocab), r: n0(l.row_cost), parts: P.vocab.tables.map(x => `${x.what} ${n0(x.width)}`).join(" + ")});
      if (st.keep != null) det += " " + t("cut_vocab_k", {n: n0(st.n), k: n0(st.keep), s: fP(saves), p: pc(saves)}); how = t("cut_vocab_h"); }
    else if (l.id === "ple") { const tb = P.vocab.tables.filter(x => x.what.startsWith("per-layer")); title = t("cut_ple_t"); det = t("cut_ple_d", {n: tb.map(x => x.name).join(", "), s: fP(l.params), p: pc(l.params), w: n0(tb[0].width)}); how = t("cut_ple_h"); }
    else if (l.id.startsWith("tower:")) { const id = l.id.slice(6); title = t("cut_tower_t", {k: kind(id)}); det = t("cut_tower_d", {n: l.detail.split(":")[0].replace(" with ", " + "), s: fP(l.params)}); how = t("cut_tower_h"); if (!st.text) saves = null; }
    else if (l.id === "kv_shared") { title = t("cut_kvs_t"); det = t("cut_kvs_d", {n: P.kv.shared}); how = t("cut_kvs_h"); }
    else if (l.id === "moe") { title = t("cut_moe_t"); det = l.detail; how = t("cut_moe_h"); }
    else if (l.id === "ffn") { title = t("cut_ffn_t"); det = t("cut_ffn_d", {s: fP(l.params), p: pc(l.params), parts: (l.parts || []).map(x => `${x.id} ${fP(x.params)}`).join(", ")}); how = t("cut_ffn_h"); }
    else { title = l.title; det = l.detail; how = l.how; }
    if (saves && l.id !== "ffn" && l.id !== "ple") cut += saves;
    const big = saves && l.id !== "ffn" ? "−" + fP(saves) : fP(l.params);
    return {html: `<div><b>${esc(title)}</b><p>${esc(det)}</p><p class="dim">${esc(how)}</p></div><div class="cutn"><big>${esc(big)}</big><span class="dim">${esc(pc(saves && l.id !== "ffn" ? saves : l.params))}</span><div class="cbar"><i style="width:${Math.min(100, 100 * (saves && l.id !== "ffn" ? saves : l.params) / total).toFixed(1)}%"></i></div></div>`, s: saves && l.id !== "ffn" ? saves : 0};
  }).sort((a, b) => b.s - a.s);
  const K = P.kv, kvLine = K && (K.shared || K.sliding) ? `<p style="margin-top:12px">${esc(t("cut_kv", {c: K.cached, l: K.layers, sh: K.shared, sl: K.sliding, w: K.window, pt: fB(K.per_token), cap: fB(K.sliding_cap), nv: fB(K.naive_per_token)}))}</p>` : "";
  const repo = /^[\w.-]+\/[\w.-]+$/.test(M.label || "") ? M.label : null, LS = ["th", "en", "zh", "ja", "ko", "vi", "id", "ms", "km", "lo", "my", "hi", "ar", "ru"];
  const hasVocab = P.levers.some(l => l.id === "vocab"), hasTower = P.levers.some(l => l.id.startsWith("tower:"));
  box.innerHTML = `<h3>${esc(t("cut_h"))}</h3><p class="dim">${esc(t("cut_lead"))}</p>
    ${hasVocab ? `<div class="chips"><span>${esc(t("cut_langs"))}</span>${LS.map(l => `<button data-cl="${l}" class="${st.langs.includes(l) ? "on" : ""}">${l.toUpperCase()}</button>`).join("")}
      ${repo ? `<button id="cutGo">${esc(t("cut_count"))}</button>` : ""}<label>${esc(t("cut_file"))}<input type="file" id="cutFile" accept=".json" hidden></label><span class="dim" id="cutMsg">${esc(st.msg)}</span></div>` : ""}
    ${hasTower ? `<div class="chips"><label><input type="checkbox" id="cutText" ${st.text ? "checked" : ""}> ${esc(t("cut_text"))}</label></div>` : ""}
    ${items.length ? `<div class="cutl">${items.map(x => x.html).join("")}</div>` : `<p>${esc(t("cut_none"))}</p>`}
    ${cut ? `<div class="cutsum">${esc(t("cut_sum", {a: fP(total), b: fP(total - cut), c: fB((total - cut) * 2), d: fB((total - cut) / 2)}))}</div>` : ""}${kvLine}`;
  const count = pieces => { const r = Plan.vocabScripts(pieces.list, {byte_level: pieces.bl}); st.n = r.total; st.counts = r.counts; st.keep = r.keep(st.langs); st.msg = ""; st.pieces = pieces; renderCut(); };
  const parse = cutParse;
  $$("#cutBox [data-cl]").forEach(b => b.onclick = () => { const l = b.dataset.cl; st.langs = st.langs.includes(l) ? st.langs.filter(x => x !== l) : st.langs.concat(l);
    if (st.pieces) { st.keep = Plan.vocabScripts(st.pieces.list, {byte_level: st.pieces.bl}).keep(st.langs); } renderCut(); });
  const go = $("#cutGo"); if (go) go.onclick = async () => { st.msg = t("cut_loading"); $("#cutMsg").textContent = st.msg;
    try { const r = await fetch(`https://huggingface.co/${repo}/resolve/main/tokenizer.json`); if (!r.ok) throw 0; count(parse(await r.json())); }
    catch (e) { st.msg = t("cut_notok"); $("#cutMsg").textContent = st.msg; } };
  const f = $("#cutFile"); if (f) f.onchange = async () => { if (!f.files[0]) return; try { count(parse(JSON.parse(await f.files[0].text()))); } catch (e) { st.msg = t("cut_notok"); $("#cutMsg").textContent = st.msg; } };
  const tx = $("#cutText"); if (tx) tx.onchange = () => { st.text = tx.checked; renderCut(); };
}

// ================================================================== concepts (numbers from the selected model)
function renderConcepts() {
  const M = S.M, m = M.meta, s = shapes(M), mod = m.modality, L = (M.A.comps.encoder || M.A.comps.decoder).layers[0];
  const share = L && L.regions.ffn ? pct(L.regions.ffn._tot.params, L.tot.params) : "⅔";
  const first = mod === "audio" ? ["c_spec_t", t("c_spec_p", {m: s.nm})] : mod === "image" ? ["c_patch_t", t("c_patch_p", {p: m.patches || 196})] : ["c_tok_t", t("c_tok_p", {v: m.vocab ? m.vocab.toLocaleString() : "?"})];
  $("#c1t").textContent = t(first[0]); $("#c1p").textContent = first[1]; S.c1 = mod;
  $("#c2p").textContent = t("c_attn_p", {h: m.heads || (L && L.regions.self && L.regions.self.q ? "?" : "?")});
  $("#c3p").textContent = t("c_ffn_p", {d: s.d.toLocaleString(), f: (m.ffn || 4 * s.d).toLocaleString(), share});
  $("#c4p").textContent = M.A.comps.decoder ? t("c_kv_p") : t("c_kv_p_enc");
  $("#c5p").textContent = m.tie ? t("c_tie_p_yes", {v: m.vocab ? m.vocab.toLocaleString() : "V", d: s.d}) : M.cfg && M.cfg.tie_word_embeddings !== false && m.vocab ? t("c_tie_p_dup", {v: m.vocab.toLocaleString(), d: s.d}) : t("c_tie_p_no");
  $("#c6p").textContent = t("c_seam_p");
  $("#cFfnD").textContent = s.d; $("#cFfnF").textContent = m.ffn || 4 * s.d;
}
(function conceptAnims() {
  const cv = $("#cSpec"), x = cv.getContext("2d"); let tt = 0; const cols = 120, rows = 40, buf = [];
  const col = i => { const a = []; for (let j = 0; j < rows; j++) { const f = j / rows; a.push(Math.min(1, Math.max(0, Math.sin(j * .9 + i * .07) * .5 + .5) * Math.exp(-f * 2.2) * (.5 + .5 * Math.sin(i * .19)) + Math.random() * .08)); } return a; };
  for (let i = 0; i < cols; i++) buf.push(col(i));
  (function draw() { requestAnimationFrame(draw); tt++; x.clearRect(0, 0, cv.width, cv.height);
    if (S.c1 === "text") { const toks = LANG === "en" ? ["Edge", " AI", " on", " phones", "!"] : ["วัน", "นี้", "อา", "กาศ", "ดี"]; const k = Math.floor(tt / 40) % (toks.length + 2); x.font = "600 30px sans-serif";
      let px = 30; toks.forEach((tk, i) => { const w = x.measureText(tk).width + 26; x.fillStyle = i < k ? `rgba(${K.blRGB},.25)` : `rgba(${K.ovRGB},0.048)`; x.strokeStyle = i < k ? K.dec : `rgba(${K.ovRGB},0.160)`; x.beginPath(); x.roundRect(px, 90, w, 54, 10); x.fill(); x.stroke(); x.fillStyle = K.ink; x.fillText(tk, px + 13, 127); if (i < k) { x.fillStyle = K.blueL; x.font = "500 20px monospace"; x.fillText(String(1000 + i * 4137 % 30000), px + 8, 180); x.font = "600 30px sans-serif"; } px += w + 10; }); return; }
    if (S.c1 === "image") { const n = 14, sz = 260 / n, k = Math.floor(tt / 3) % (n * n + 30); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const v = .5 + .5 * Math.sin(i * .45 + j * .6); x.fillStyle = `hsl(${30 + v * 30},${40 + 30 * v}%,${25 + 35 * v}%)`; const id = j * n + i; x.fillRect(170 + i * sz + (id < k ? 1.5 : 0), 20 + j * sz + (id < k ? 1.5 : 0), sz - (id < k ? 3 : 0), sz - (id < k ? 3 : 0)); } return; }
    if (tt % 3 === 0) { buf.shift(); buf.push(col(tt / 3 + cols)); } const cw = cv.width / cols, ch = cv.height / rows;
    buf.forEach((c, i) => c.forEach((v, j) => { x.fillStyle = `rgba(${Math.round(91 + 140 * v)},${Math.round(107 - 60 * v)},${Math.round(240 - 115 * v)},${.15 + v * .85})`; x.fillRect(i * cw, cv.height - (j + 1) * ch, cw + .5, ch + .5); })); })();
  const toks = ["วัน", "นี้", "อา", "กาศ", "ดี", "มาก"], A = $("#cAttn"); let focus = 0;
  setInterval(() => { focus = (focus + 1) % 6; let s = ""; toks.forEach((tk, i) => { const xx = 60 + i * 96, w = i === focus ? 1 : Math.exp(-Math.abs(i - focus) * .9);
    s += `<text x="${xx}" y="60" text-anchor="middle" fill="${i === focus ? C.magL : C.soft}" font-size="24" font-weight="700">${tk}</text><text x="${xx}" y="262" text-anchor="middle" fill="${C.soft}" font-size="24">${tk}</text><line x1="${60 + focus * 96}" y1="78" x2="${xx}" y2="232" stroke="${C.mag}" stroke-width="${1 + w * 9}" stroke-opacity="${.25 + w * .7}" stroke-linecap="round"/>`; }); A.innerHTML = s; }, 1300);
  const F = $("#cFfn"); let ph = 0;
  (function ffn() { requestAnimationFrame(ffn); ph = (ph + .015) % 1; const e = Math.sin(ph * Math.PI), mid = 40 + 200 * e, d0 = +($("#cFfnD").textContent) || 1024, f0 = +($("#cFfnF").textContent) || 4096;
    F.innerHTML = `<rect x="40" y="110" width="60" height="80" rx="8" fill="${C.blue}"/><text x="70" y="214" text-anchor="middle" fill="${C.soft}" font-size="15" font-family="monospace">${d0}</text><rect x="${300 - mid / 2}" y="${150 - (40 + 100 * e)}" width="${mid}" height="${80 + 200 * e}" rx="10" fill="${C.mag}" opacity=".85"/><text x="300" y="${170 + 100 * e + 34}" text-anchor="middle" fill="${C.soft}" font-size="15" font-family="monospace">${Math.round(d0 + (f0 - d0) * e)}</text><rect x="500" y="110" width="60" height="80" rx="8" fill="${C.blue}"/><text x="530" y="214" text-anchor="middle" fill="${C.soft}" font-size="15" font-family="monospace">${d0}</text><path d="M100 150 H${300 - mid / 2}" stroke="${C.dim}" stroke-width="2" class="flowline"/><path d="M${300 + mid / 2} 150 H500" stroke="${C.dim}" stroke-width="2" class="flowline"/>`; })();
  const K = $("#cKv"); let n = 0;
  setInterval(() => { n = (n + 1) % 9; let s = `<text x="30" y="40" fill="${C.soft}" font-size="16">token ${n + 1}</text>`; for (let i = 0; i <= n; i++) s += `<rect x="${40 + i * 58}" y="80" width="50" height="60" rx="6" fill="${i === n ? C.amber : C.mag}" opacity="${i === n ? 1 : .55}"/><rect x="${40 + i * 58}" y="150" width="50" height="60" rx="6" fill="${i === n ? C.amber : C.mag}" opacity="${i === n ? 1 : .35}"/>`;
    K.innerHTML = s + `<text x="20" y="116" fill="${C.ink}" font-size="18" font-weight="700">K</text><text x="20" y="186" fill="${C.ink}" font-size="18" font-weight="700">V</text>`; }, 900);
  const T = $("#cTie"); let tp = 0;
  setInterval(() => { tp = 1 - tp; T.innerHTML = `<rect x="200" y="70" width="200" height="160" rx="12" fill="${K.dec}" opacity=".25" stroke="${K.dec}"/>${Array.from({length: 10}, (_, i) => `<line x1="210" x2="390" y1="${86 + i * 15}" y2="${86 + i * 15}" stroke="${K.dec}" stroke-opacity=".5"/>`).join("")}
    ${tp ? `<rect x="205" y="131" width="190" height="12" fill="${C.amber}"/><path d="M60 137 H200" stroke="${C.amber}" stroke-width="3" class="flowline"/><text x="60" y="125" fill="${C.soft}" font-size="15">token id</text><text x="300" y="270" text-anchor="middle" fill="${C.amber}" font-size="17">in: 1 row</text>` : `<rect x="200" y="70" width="200" height="160" rx="12" fill="${C.amber}" opacity=".25"/><path d="M400 150 H540" stroke="${C.amber}" stroke-width="3" class="flowline"/><text x="470" y="135" fill="${C.soft}" font-size="15">scores</text><text x="300" y="270" text-anchor="middle" fill="${C.amber}" font-size="17">out: whole table</text>`}`; }, 1800);
  $("#cSeam").innerHTML = `<defs><marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="${C.dim}"/></marker></defs>` + ["float", "QUANTIZE", "int8 op", "int8 op", "DEQUANTIZE", "float op"].map((tx, i) => { const c2 = tx.includes("QUANT") ? C.red : tx.startsWith("int") ? C.mag : C.blue, xx = 14 + i * 97;
    return `<rect x="${xx}" y="110" width="84" height="70" rx="10" fill="${c2}" fill-opacity=".2" stroke="${c2}"/><text x="${xx + 42}" y="150" text-anchor="middle" fill="${C.ink}" font-size="${tx.length > 8 ? 11.5 : 14}" font-weight="600">${tx}</text>` + (i < 5 ? `<path d="M${xx + 84} 145 H${xx + 97}" stroke="${C.dim}" stroke-width="2" marker-end="url(#ar2)"/>` : ""); }).join("");
})();
(function qplay() {
  const cv = $("#qCv"), x = cv.getContext("2d"); let bits = 4, outlier = false, block = false, seed = 7, W = [];
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }, gauss = () => { const u = rnd() || 1e-9, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283 * v); };
  const gen = () => W = Array.from({length: 64}, () => gauss() * .08);
  const quant = (ws, b) => { const q = Math.pow(2, b - 1) - 1, g = block ? 32 : ws.length, out = []; for (let s = 0; s < ws.length; s += g) { const part = ws.slice(s, s + g), sc = Math.max(...part.map(Math.abs)) / q || 1; part.forEach(w => out.push(Math.round(w / sc) * sc)); } return out; };
  function draw() { const ws = W.slice(); if (outlier) ws[20] = .9; const qs = quant(ws, bits), err = Math.sqrt(ws.reduce((a, w, i) => a + (w - qs[i]) ** 2, 0) / ws.reduce((a, w) => a + w * w, 0)), zeros = qs.filter((v, i) => v === 0 && ws[i] !== 0).length;
    const Wd = cv.width, Hd = cv.height, lim = outlier ? 1 : .3, Y = v => Hd / 2 - v / lim * (Hd * .42), X = i => 40 + i * (Wd - 80) / 63; x.clearRect(0, 0, Wd, Hd); x.strokeStyle = `rgba(${K.ovRGB},0.064)`; x.lineWidth = 1;
    const q = Math.pow(2, bits - 1) - 1, sc = Math.max(...ws.slice(0, block ? 32 : 64).map(Math.abs)) / q;
    for (let k = -q; k <= q; k++) { const yy = Y(k * sc); if (yy > 0 && yy < Hd) { x.beginPath(); x.moveTo(30, yy); x.lineTo(block ? Wd / 2 : Wd - 30, yy); x.stroke(); } }
    if (block) { const sc2 = Math.max(...ws.slice(32).map(Math.abs)) / q; for (let k = -q; k <= q; k++) { const yy = Y(k * sc2); if (yy > 0 && yy < Hd) { x.beginPath(); x.moveTo(Wd / 2, yy); x.lineTo(Wd - 30, yy); x.stroke(); } } }
    x.strokeStyle = C.mag; x.lineWidth = 4; x.beginPath(); qs.forEach((v, i) => i ? x.lineTo(X(i), Y(v)) : x.moveTo(X(i), Y(v))); x.stroke();
    ws.forEach((v, i) => { x.fillStyle = qs[i] === 0 && v !== 0 ? C.red : C.blueL; x.beginPath(); x.arc(X(i), Y(v), 7, 0, 6.28); x.fill(); });
    $("#qBitsV").textContent = bits; $("#qLev").textContent = 2 * q + 1; $("#qErr").textContent = (err * 100).toFixed(1) + "%"; $("#qZero").textContent = zeros + " / 64"; $("#qSize").textContent = (bits / 8 + (block ? 2 / 32 : 0)).toFixed(2) + " B"; }
  gen(); draw(); window.qRelabel = () => { $("#qOut").textContent = t(outlier ? "q_out_rm" : "q_out_add"); $("#qBlock").textContent = t(block ? "q_scale32" : "q_scale1"); };
  $("#qBits").oninput = e => { bits = +e.target.value; draw(); };
  $("#qOut").onclick = () => { outlier = !outlier; $("#qOut").classList.toggle("on", outlier); qRelabel(); draw(); };
  $("#qBlock").onclick = () => { block = !block; $("#qBlock").classList.toggle("on", block); qRelabel(); draw(); };
  $("#qNew").onclick = () => { gen(); draw(); };
})();

// ================================================================== report
function shareRows() {
  const A = S.M.A, E = A.comps.encoder, D = A.comps.decoder, rows = [], tk = S.tokens;
  const add = (name, p, per, c) => p && p.has && rows.push({name, p, per, c});
  const sumR = (Cc, r) => Cc.layers.reduce((a, L) => L.regions[r] ? merge(a, L.regions[r]._tot) : a, part());
  const nl = (Cc, r) => Cc.layers.filter(L => L.regions[r]).length;
  const F = fam(S.M), en = F === "diffusion" ? "Denoiser" : F === "transducer" ? "Encoder" : S.M.meta.family === "prefix" ? (S.M.meta.modality === "image" ? "Vision" : "Audio") : "Encoder", per = F === "diffusion" ? "step" : "in";
  if (E) { add(en + " · " + (F === "diffusion" ? "patchify + timestep" : S.M.meta.modality === "audio" ? "conv stem" : S.M.meta.modality === "image" ? "patch embedding" : "embedding"), E.stemTot, per, "e");
    REG_ORDER.forEach(r => add(`${en} · ${regTitle(r).toLowerCase()} ×${nl(E, r)}`, sumR(E, r), per, "e"));
    add(D ? (S.M.meta.family === "prefix" ? en + " · projector" : en + " · final norm / K,V") : en + " · head", E.headTot, per, "e"); }
  (S.M.A.extra || []).forEach(x => add(`${x.id} (${x.role})`, x.tot, F === "transducer" && /joint|prediction/.test(x.role) ? "tok" : "in", "e"));
  if (D) { const dn = S.M.meta.family === "prefix" ? "LLM" : "Decoder"; add(`${dn} · token embedding`, D.stemTot, "tok", "d");
    REG_ORDER.forEach(r => add(`${dn} · ${regTitle(r).toLowerCase()} ×${nl(D, r)}`, sumR(D, r), "tok", "d"));
    add(`${dn} · norm + logits${D.head.mtp ? " + MTP" : ""}`, D.headTot, "tok", "d"); }
  rows.forEach(r => r.read = r.p.bytes * (r.per === "tok" || r.per === "step" ? tk : 1));
  if (D && D.headTot.shared) { const emb = rows.find(r => / · token embedding/.test(r.name) && r.c === "d"), lg = rows.find(r => / · norm \+ logits/.test(r.name)); const tw = D.stemTot.w.find(w => w.shape.length === 2 && w.shape[0] > 1000);
    if (emb && lg && tw) { const tb = emb.p.bytes * (prod(tw.shape) / Math.max(1, emb.p.params)); emb.read = (emb.p.bytes - tb + tb / tw.shape[0]) * tk; lg.read = tb * tk; } }
  return rows;
}
function insights() {
  const M = S.M, A = M.A, E = A.comps.encoder, D = A.comps.decoder, out = [], tot = Object.values(A.comps).reduce((a, c) => merge(a, c.tot), part());
  const rows = shareRows(), big = rows.slice().sort((a, b) => b.p.bytes - a.p.bytes)[0];
  if (big) out.push([C.blue, "i", t("i_big", {p: esc(big.name), s: pct(big.p.bytes, tot.bytes)})]);
  if (E && D) { const rE = rows.filter(r => r.c === "e").reduce((a, r) => a + r.read, 0), rD = rows.filter(r => r.c === "d").reduce((a, r) => a + r.read, 0); out.push([C.mag, "i", t("i_split", {e: pct(E.tot.bytes, tot.bytes), d: pct(D.tot.bytes, tot.bytes), x: rE ? Math.round(rD / rE) : "–"})]); }
  else if (D) out.push([C.mag, "i", t("i_dec", {b: fB(tot.bytes)})]); else out.push([C.blue, "i", t("i_enc", {f: fF(tot.flops)})]);
  const L = (E || D).layers[0]; if (L && L.regions.ffn) out.push([C.mag, "i", t("i_ffn", {s: pct(L.regions.ffn._tot.params, L.tot.params)})]);
  const emb = (D || E).stemTot; if (emb.bytes / tot.bytes > .12) out.push([C.amber, "!", t("i_emb", {b: fB(emb.bytes), s: pct(emb.bytes, tot.bytes)})]);
  if (D && D.headTot.shared) out.push([C.ok, "✓", t("i_tied")]);
  if (D && D.head.logits && D.head.logits.w.length && D.stem.embed && D.stem.embed.w.some(w => D.head.logits.w.some(l => l.shape.join() === w.shape.join()))) out.push([C.red, "!", t("i_split_emb")]);
  if (M.meta.kv_heads && M.meta.heads && M.meta.kv_heads < M.meta.heads) out.push([C.ok, "✓", t("i_gqa", {kv: M.meta.kv_heads, h: M.meta.heads, x: M.meta.heads / M.meta.kv_heads})]);
  if (M.meta.family === "prefix") out.push([C.blueL, "i", t("i_prefix")]);
  if (D) { const lin = D.layers.filter(L => L.regions.linear).length, full = D.layers.filter(L => L.regions.self).length;
    const kvB = D.layers.reduce((a, L) => { const R = L.regions.self; if (!R) return a; const k = R.k && R.k.w[0] ? R.k.w[0].shape[0] : 0, v = R.v && R.v.w[0] ? R.v.w[0].shape[0] : k; return a + (k + v) * 2; }, 0);
    if (kvB) { const n = M.meta.ctx ? Math.min(M.meta.ctx, 131072) : 32768; out.push([C.amber, "!", t("i_kv", {b: fB(kvB), c: fB(kvB * n), n: n.toLocaleString()})]); }
    if (lin) out.push([C.ok, "✓", t("i_hybrid", {l: lin, t: lin + full, f: full})]);
    if (D.head.mtp) out.push([C.blueL, "i", t("i_mtp", {b: fB(D.head.mtp.bytes)})]); }
  const m = M.meta;
  if (m.experts) { const ex = M.IR && M.IR.hp ? M.IR.hp.expert_params : 0; out.push([C.mag, "i", t("i_moe", {e: m.experts, k: m.topk || "?", s: ex ? pct(ex, tot.params) : "?", a: ex && m.topk ? fP(tot.params - (M.IR.hp.mtp_params || 0) - (ex - (M.IR.hp.mtp_expert_params || 0)) * (1 - m.topk / m.experts)) : "?"})]); }
  if (m.quant && Object.keys(m.quant).length) out.push([C.amber, "i", t("i_quant", {q: Object.entries(m.quant).map(([k, v]) => `${k.toUpperCase()} ×${v}`).join(", ")})]);
  if (m.joint) out.push([C.blueL, "i", t(m.joint.durations ? "i_tdt" : "i_rnnt", {v: m.joint.labels, d: m.joint.durations, j: m.joint.j})]);
  if (m.timestep) out.push([C.blueL, "i", t("i_diff", {f: m.timestep.freq, w: m.timestep.width})]);
  if (!D && E && M.IR && E.layers.some(L => L.regions.linear)) { const l = E.layers.filter(L => L.regions.linear).length; out.push([C.ok, "✓", t("i_hybrid", {l, t: E.layers.length, f: E.layers.length - l})]); }
  const dts = tot.dt; if (Object.keys(dts).length > 1) out.push([C.amber, "!", t("i_mixed", {m: dtLabel(tot)})]);
  if (M.kind === "ops") { const ty = tot.types, n = tot.ops, sh = ["RESHAPE", "TRANSPOSE", "SLICE", "CONCATENATION", "PAD", "GATHER_ND", "SQUEEZE", "EXPAND_DIMS"].reduce((a, k) => a + (ty[k] || 0), 0);
    out.push([C.dim, "i", t("i_shape", {s: pct(sh, n)})]); const ln = ["MEAN", "SQUARED_DIFFERENCE", "RSQRT"].reduce((a, k) => a + (ty[k] || 0), 0); if (ln) out.push([C.blueL, "i", t("i_ln", {n: Object.values(A.comps).reduce((a, c) => a + c.layers.reduce((b, Lx) => b + Object.values(Lx.regions).reduce((cc, R) => cc + (R.ln ? R.ln.ops : 0), 0), 0) + (c.headTot && c.head.ln ? c.head.ln.ops : 0), 0)})]);
    if (ty.DEQUANTIZE && !ty.QUANTIZE) out.push([C.red, "!", t("i_wo", {n: ty.DEQUANTIZE})]); if (ty.QUANTIZE) out.push([C.red, "!", t("i_static", {n: ty.QUANTIZE})]); }
  return out;
}
function renderReport() {
  const M = S.M, rows = shareRows(), tb = rows.reduce((a, r) => a + r.p.bytes, 0), fl = r => r.p.flops * (r.per === "tok" ? S.tokens : 1), tf = rows.reduce((a, r) => a + fl(r), 0), tr = rows.reduce((a, r) => a + r.read, 0), ops = M.kind === "ops";
  $("#rIns").innerHTML = insights().map(([c, i, tx]) => `<div class="ins"><i style="background:${c}" aria-hidden="true"></i><div>${tx}</div></div>`).join("");
  const tot = Object.values(M.A.comps).reduce((a, c) => merge(a, c.tot), part()), top = Object.entries(tot.types).sort((a, b) => b[1] - a[1]).slice(0, 12), mx = top.length ? top[0][1] : 1;
  $("#rOps").innerHTML = ops ? top.map(([k, n]) => `<div class="opbar"><span class="mono k" style="color:${/QUANT/.test(k) ? C.redL : C.soft}">${k}</span><div class="v" style="width:${n / mx * 240}px;background:${/QUANT/.test(k) ? C.red : /FULLY|CONV|EMBED|BATCH/.test(k) ? C.mag : C.blue}"></div><span class="mono" style="font-size:12.5px">${n}</span></div>`).join("") : `<p style="color:var(--dim)">${t("r_noops")}</p>`;
  const hdr = [t("th_part"), t("th_weights"), t("th_share"), t("th_stored"), t("th_compute") + (M.A.comps.decoder ? ` (${S.tokens} tok)` : ""), t("th_share"), t("th_read"), ...(ops ? [t("th_ops"), t("th_seams")] : [])];
  const body = rows.map(r => [r.name, fB(r.p.bytes), pct(r.p.bytes, tb), dtLabel(r.p), fF(fl(r)) + (r.p.est ? "*" : ""), pct(fl(r), tf), fB(r.read), ...(ops ? [r.p.ops, r.p.seams || ""] : [])]);
  const foot = [t("total"), fB(tb), "", "", fF(tf), "", fB(tr), ...(ops ? [rows.reduce((a, r) => a + r.p.ops, 0), ""] : [])];
  $("#rTable").innerHTML = `<div class="tscroll"><table class="t" style="min-width:760px"><tr>${hdr.map(h => `<th>${esc(h)}</th>`).join("")}</tr>${body.map((r, i) => `<tr>${r.map((c, j) => `<td${j ? "" : ` style="color:${rows[i].c === "e" ? C.blueL : C.magL}"`}>${esc(String(c))}</td>`).join("")}</tr>`).join("")}<tr style="font-weight:700">${foot.map(c => `<td>${esc(String(c))}</td>`).join("")}</tr></table></div>${rows.some(r => r.p.est) ? `<p class="note">* ${t("est")}</p>` : ""}`;
  S.md = `| ${hdr.join(" | ")} |\n|${hdr.map(() => "---").join("|")}|\n` + body.map(r => `| ${r.join(" | ")} |`).join("\n") + `\n| ${foot.join(" | ")} |\n\nModel: ${M.label} · Model Explorer by Sattaya Singkul`;
}

// ================================================================== loading your own model
async function loadFiles(files) {
  const sts = [...files].filter(f => /\.safetensors$/i.test(f.name));
  if (sts.length) { try { const tensors = {}; let read = 0;
      for (const [k, f] of sts.entries()) { busy(t("st_header", {i: k + 1, n: sts.length}), f.name); const h = new DataView(await f.slice(0, 8).arrayBuffer()), n = Number(h.getBigUint64(0, true));
        const j = JSON.parse(new TextDecoder().decode(await f.slice(8, 8 + n).arrayBuffer())); delete j.__metadata__; read += 8 + n; const shard = /-\d+-of-\d+\.safetensors$/i.test(f.name), clash = Object.keys(j).some(k2 => k2 in tensors);
          const pre = sts.length > 1 && !shard && clash ? f.name.replace(/\.safetensors$/i, "").replace(/[^\w]+/g, "_") + "." : ""; for (const [k2, v] of Object.entries(j)) tensors[pre + k2] = v; }
      // config.json and tokenizer.json picked together with the weights: hyperparameters from the config, vocabulary counts from the tokenizer
      const js = n => [...files].find(f => f.name.toLowerCase() === n); let cfg = null, tok = null;
      try { if (js("config.json")) cfg = JSON.parse(await js("config.json").text()); } catch (e) {}
      try { if (js("tokenizer.json")) tok = JSON.parse(await js("tokenizer.json").text()); } catch (e) {}
      S.pendingTok = tok;
      addOwn({tensors, cfg, meta: null, kind: "tensors", label: sts.map(f => f.name).join(" + "), noteKey: "n_hf_auto", noteVars: {n: Object.keys(tensors).length, r: fB(read), w: fB(sts.reduce((a, f) => a + f.size, 0)), f: sts.length}}, sts.length > 1 ? "Uploaded (" + sts.length + ")" : sts[0].name);
    } catch (e) { ownNote(`<b style="color:${K.neg}">${t("e_read")}</b> ${esc(e.message)}`); } return; }
  const tfl = [...files].filter(f => /\.tflite$/i.test(f.name)); if (!tfl.length) return; ownNote(t("loading"));
  try { const gs = []; for (const f of tfl) { const comp = /enc/i.test(f.name) ? "encoder" : /dec/i.test(f.name) ? "decoder" : (tfl.length === 1 ? "model" : "encoder"); gs.push(graphFromTflite(parseTflite(await f.arrayBuffer()), comp, f.name)); }
    const enc = gs.some(g => g.component === "encoder"), dec = gs.some(g => g.component !== "encoder");
    const meta = {name: tfl.map(f => f.name).join(" + "), family: enc && dec ? "encdec" : enc ? "encoder" : "decoder", modality: gs.some(g => g.nodes.some(n => n.t === "CONV_2D")) && enc ? "audio" : "text", d: null, tie: true};
    const id = "up-" + tfl.map(f => f.name + f.size).join("-").replace(/[^\w-]+/g, "-").slice(0, 60); const def = register({id, label: meta.name, short: tfl.length > 1 ? "Uploaded (" + tfl.length + ")" : tfl[0].name, group: "own", graphs: gs, meta, kind: "ops", noteKey: "n_upload", note: () => t("n_upload")}); def.M = null; persist(def); ownNote(t("saved_note")); setModel(id);
  } catch (e) { ownNote(`<b style="color:${K.neg}">${t("e_read")}</b> ${esc(e.message)}`); }
}
function ownNote(h) { $("#ownNote").innerHTML = h; }
async function hfLoad(input) {
  const m = String(input).trim().match(/(?:huggingface\.co\/)?([\w.-]+\/[\w.-]+)/); if (!m) throw new Error("owner/repo");
  const repo = m[1].replace(/\/(tree|blob|resolve)\/.*$/, ""), base = `https://huggingface.co/${repo}/resolve/main/`;
  busy(t("st_list"), repo);
  let info; try { info = await (await fetch(`https://huggingface.co/api/models/${repo}?blobs=true`)).json(); } catch (e) { throw new Error("BLOCKED"); }
  if (info.error) throw new Error(info.error);
  const files = (info.siblings || []).map(s => ({n: s.rfilename, size: s.size || 0})), tfl = files.filter(f => /\.tflite$/i.test(f.n));
  const total = files.reduce((a, f) => a + f.size, 0);
  let mi = null; if (files.some(f => f.n === "model_index.json")) try { mi = await (await fetch(base + "model_index.json")).json(); } catch (e) {}
  let cfg = null; const cfgF = files.find(f => f.n === "config.json"); if (cfgF) try { busy(t("st_cfg"), "config.json"); cfg = await (await fetch(base + "config.json")).json(); } catch (e) {}
  if (tfl.length) { const pick = tfl.filter(f => /enc|dec/i.test(f.n)).slice(0, 2), chosen = pick.length ? pick : tfl.slice(0, 1), gs = [];
    for (const f of chosen) { busy(t("st_tflite"), `${f.n} (${fB(f.size)})`); gs.push(graphFromTflite(parseTflite(await (await fetch(base + f.n)).arrayBuffer()), /enc/i.test(f.n) ? "encoder" : /dec/i.test(f.n) ? "decoder" : "model", f.n)); }
    return {graphs: gs, kind: "ops", meta: {name: repo, family: chosen.length > 1 ? "encdec" : "decoder", modality: "text", tie: true}, label: repo, noteKey: "n_upload"}; }
  const mf = await Loaders.hfManifest(repo, files, (st, n, i, k, rd) => busy(st === "index" ? t("st_index") : t(st === "pt" ? "st_pt" : "st_header", {i, n: k}), st === "index" ? n : `${n} · ${fB(rd)} ${t("st_sofar")}`), {prefer: cfg && cfg.default_size, only: mi ? Object.entries(mi).filter(([k, v]) => !k.startsWith("_") && Array.isArray(v) && v[0]).map(([k]) => k) : null});
  if (!mf) { if (cfg) { const r = tensorsFromConfig(cfg); return {tensors: r.tensors, meta: {...r.meta, name: repo}, cfg, kind: "config", label: repo, noteKey: "n_cfg", noteVars: {mt: r.meta.model_type}}; } throw new Error("no .tflite / .safetensors / .pt / config.json"); }
  const {tensors, read} = mf, jobs = {length: mf.nfiles};
  busy(t("st_infer"), `${Object.keys(tensors).length} tensors`);
  await new Promise(r => setTimeout(r, 20));
  const n = Object.keys(tensors).length; return {tensors, cfg, meta: null, kind: "tensors", label: repo, noteKey: "n_hf_auto", noteVars: {n, r: fB(read), w: fB(total), f: jobs.length}};
}
function busy(stage, detail) { ownNote(`<span class="spin" aria-hidden="true"></span><b>${esc(stage)}</b> <span class="mono dim">${esc(detail || "")}</span>`); }
function addOwn(r, short) { const id = "own-" + (r.label || "m").replace(/[^\w-]+/g, "-").slice(0, 40); const def = register({id, label: r.label, short: short || r.label.split("/").pop(), group: "own", ...r, note: () => t(r.noteKey, r.noteVars)}); def.M = null; persist(def); ownNote(t("saved_note")); setModel(id); }

// ================================================================== everything
function renderAll() { applyI18n(); renderPicker(); renderHero(); renderArch(); renderCut(); renderBig(); buildPlay(); renderIO(); renderDetail(); renderMap(); renderConcepts(); renderReport(); if (window.qRelabel) qRelabel(); $("#tokV").textContent = S.tokens; $("#tokV2").textContent = S.tokens; }
function wire() {
  { const tb = $("#themeBtn"); if (tb) { const lab = () => { const d = K.ink === PAL.dark.ink; tb.title = LANG === "th" ? (d ? "สลับเป็นธีมสว่าง" : "สลับเป็นธีมมืด") : (d ? "Switch to light theme" : "Switch to dark theme"); tb.setAttribute("aria-label", tb.title); };
    lab(); tb.onclick = () => { const nt = K.ink === PAL.dark.ink ? "light" : "dark"; setTheme(nt); try { localStorage.setItem("mx_theme", nt); } catch (e) {} renderAll(); lab(); }; } }
  $$("[data-lang]").forEach(b => b.onclick = () => { if (/^https?:$/.test(location.protocol) && window.PAGE_ALT && b.dataset.lang !== window.PAGE_LANG) { try { localStorage.setItem("mx_lang", b.dataset.lang); } catch (e) {} location.href = window.PAGE_ALT + location.hash; return; } LANG = b.dataset.lang; try { localStorage.setItem("mx_lang", LANG); } catch (e) {} renderAll(); });
  $$("[data-xl]").forEach(b => b.onclick = () => { S.lens = b.dataset.xl; pref("lens", S.lens); renderBig(); renderDetail(); });
  $$("[data-ml]").forEach(b => b.onclick = () => { S.mapLens = b.dataset.ml; renderMap(); });
  $("#bPlay").onclick = () => S.playing ? stopPlay() : startPlay();
  $("#bNext").onclick = () => { stopPlay(); if (!PLAY.length) buildPlay(); goStep(S.step + 1); };
  $("#bPrev").onclick = () => { stopPlay(); if (!PLAY.length) buildPlay(); goStep(Math.max(0, S.step - 1)); };
  const tok = e => { S.tokens = +e.target.value; $("#tok").value = $("#tok2").value = S.tokens; renderBig(); renderMap(); renderReport(); $("#tokV").textContent = $("#tokV2").textContent = S.tokens; };
  $("#tok").oninput = tok; $("#tok2").oninput = tok;
  $("#file").onchange = e => loadFiles(e.target.files);
  $("#hfGo").onclick = async () => { busy(t("loading"), ""); $("#hfGo").disabled = true; try { const r = await hfLoad($("#hf").value); ownNote(""); addOwn(r); } catch (e) { ownNote(e.message === "BLOCKED" ? t("e_blocked") : `<b style="color:${K.neg}">${t("e_read")}</b> ${esc(e.message)}`); } finally { $("#hfGo").disabled = false; } };
  $("#hf").addEventListener("keydown", e => { if (e.key === "Enter") $("#hfGo").click(); });
  $("#cfgOpen").onclick = () => $("#cfgDlg").classList.add("open"); $("#cfgClose").onclick = () => $("#cfgDlg").classList.remove("open");
  $("#cfgGo").onclick = () => { try { const cfg = JSON.parse($("#cfgTxt").value); const r = tensorsFromConfig(cfg); $("#cfgDlg").classList.remove("open"); addOwn({tensors: r.tensors, meta: r.meta, cfg, kind: "config", label: (cfg._name_or_path || r.meta.model_type || "config") + "", noteKey: "n_cfg", noteVars: {mt: r.meta.model_type}}, r.meta.model_type); } catch (e) { $("#cfgErr").textContent = t("e_cfg") + e.message; } };
  { const pc = $("#pipCopy"); if (pc) pc.onclick = async () => { const code = pc.querySelector("code"), old = code.textContent; try { await navigator.clipboard.writeText(old); code.textContent = t("copied"); } catch (e) { code.textContent = old; } setTimeout(() => code.textContent = old, 1500); }; }
  $("#copyBib").onclick = async () => { try { await navigator.clipboard.writeText($("#citeBib").textContent); $("#copyBib").textContent = t("copied"); } catch (e) { $("#copyBib").textContent = t("cite_fail"); } setTimeout(() => $("#copyBib").textContent = t("cite_copy"), 1800); };
  $("#copyMd").onclick = async () => { try { await navigator.clipboard.writeText(S.md); $("#copyMd").textContent = t("copied"); } catch (e) { $("#copyMd").textContent = t("copy_fail"); } setTimeout(() => $("#copyMd").textContent = t("copy_md"), 1800); };
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), {threshold: 0, rootMargin: "0px 0px -6% 0px"}); $$(".reveal").forEach(el => io.observe(el));
  addEventListener("hashchange", () => { const h = location.hash.slice(1); if (h && MODELS.some(m => m.id === h) && (!S.M || S.M.id !== h)) setModel(h, true); });
}

// ================================================================== templates
(function templates() {
  const PG = window.PRE_GRAPHS || {}, RH = (typeof PRELOADED !== "undefined" && PRELOADED.rehearsal) || {}, TT = window.TEMPLATE_TENSORS || {};
  const wmeta = {name: "distill-whisper-th-medium", model_type: "whisper", family: "encdec", modality: "audio", d: 1024, heads: 16, kv_heads: 16, ffn: 4096, vocab: 51865, tie: true, nmels: 80, seqEnc: 1500};
  if (PG.medium_enc) register({id: "whisper-th-medium", label: "biodatlab/distill-whisper-th-medium", short: "Whisper medium (Thai)", sub: ["เสียง → ข้อความ · .tflite จริง", "speech → text · real .tflite"], group: "tpl", cat: "speech", graphs: [PG.medium_enc, PG.medium_dec], meta: wmeta, note: () => t("n_whisper")});
  const TCAT = {"qwen3-asr-1-7b": "speech", "qwen3-8-27b": "vision", "llama-3-2-1b": "language", "gpt2": "language", "bert-base": "language", "vit-base": "vision"};
  const tpl = (id, key, short, sub) => { const T = TT[key]; if (T) register({id, label: T.meta.name, short, sub, cat: TCAT[id], group: "tpl", tensors: T.tensors, meta: T.meta, cfg: T.cfg || {hidden_size: T.meta.d, num_attention_heads: T.meta.heads, num_key_value_heads: T.meta.kv_heads}, note: () => t(/qwen3_5|qwen3_asr/.test(T.meta.model_type) ? "n_cfgtpl" : "n_tensors", {name: T.meta.name})}); };
  tpl("qwen3-asr-1-7b", "qwen3_asr_17b", "Qwen3-ASR 1.7B", ["เสียง → LLM · ASR", "audio → LLM · ASR"]);
  tpl("qwen3-8-27b", "qwen38_27b", "Qwen3.8 27B", ["ภาพ+ข้อความ · hybrid attention", "vision + text · hybrid attention"]);
  tpl("llama-3-2-1b", "llama32_1b", "Llama 3.2 1B", ["LLM · decoder-only", "LLM · decoder-only"]);
  tpl("gpt2", "gpt2", "GPT-2 small", ["LLM · decoder-only", "LLM · decoder-only"]);
  tpl("bert-base", "bert_base", "BERT base", ["ข้อความ · encoder-only", "text · encoder-only"]);
  tpl("vit-base", "vit_base", "ViT base", ["ภาพ · encoder-only", "image · encoder-only"]);
  const TC = window.TEMPLATE_COMPACT || {};
  const CF = window.TEMPLATE_CONFIGS || {}, CAT = {"nemotron-3-5-super-vl": "vision", "nemotron-3-5-transcribe": "speech", "nemotron-3-5-asr": "speech", "parakeet-tdt-v3": "speech", "nemotron-3-embed-nvfp4": "embed", "fasth3-omni-fp8": "gen", "kumo-tabular": "embed", "kumo-relational": "embed"};
  const nt = (id, key, label, short, sub, src) => { if (!TC[key]) return; register({id, label, short, sub, cat: CAT[id], src, group: "new", tensors: expandCompact(TC[key]), cfg: CF[key] || null, meta: null, note: () => t(src === "rec" ? "n_rec" : src === "q" ? "n_cfgq" : "n_cfgir", {name: label})}); };
  nt("nemotron-3-5-super-vl", "Nemotron-3.5-Super-VL-120B-A12B", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B", "Nemotron 3.5 Super VL", ["Mamba-2 + MoE + attention · 512 experts", "Mamba-2 + MoE + attention · 512 experts"]);
  nt("nemotron-3-5-transcribe", "Nemotron-3.5-Transcribe-30B-A4B", "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B", "Nemotron 3.5 Transcribe", ["เสียง → hybrid MoE LLM", "audio → hybrid MoE LLM"]);
  nt("nemotron-3-5-asr", "nemotron-3.5-asr-streaming-0.6b", "nvidia/nemotron-3.5-asr-streaming-0.6b", "Nemotron 3.5 ASR streaming", ["FastConformer + RNN-T", "FastConformer + RNN-T"]);
  nt("parakeet-tdt-v3", "parakeet-tdt-0.6b-v3", "nvidia/parakeet-tdt-0.6b-v3", "Parakeet TDT 0.6B v3", ["FastConformer + TDT", "FastConformer + TDT"]);
  nt("nemotron-3-embed-nvfp4", "Nemotron-3-Embed-1B-NVFP4", "nvidia/Nemotron-3-Embed-1B-NVFP4", "Nemotron 3 Embed NVFP4", ["embedding · NVFP4", "embedding · NVFP4"], "q");
  nt("fasth3-omni-fp8", "FastH3-Omni-FP8 (transformer)", "FastVideo/FastH3-Omni-FP8 · transformer", "FastH3 Omni FP8", ["video+audio diffusion · FP8", "video+audio diffusion · FP8"], "rec");
  nt("kumo-tabular", "Kumo-Tabular (large)", "nvidia/Kumo-Tabular · large", "Kumo Tabular", ["ตาราง · in-context", "tables · in-context"], "rec");
  nt("kumo-relational", "Kumo-Relational", "nvidia/Kumo-Relational", "Kumo Relational", ["หลายตาราง · GNN + ICL", "multi-table · GNN + ICL"], "rec");
  const rmeta = {name: "whisper rehearsal", model_type: "whisper", family: "encdec", modality: "audio", d: 128, heads: 2, kv_heads: 2, ffn: 256, vocab: 51865, tie: true, nmels: 80, seqEnc: 1500};
  if (PG.reh_fp32_enc) register({id: "reh-fp32", label: "Whisper rehearsal · fp32", short: "fp32", sub: ["มี fragility", "with fragility"], group: "prec", graphs: [PG.reh_fp32_enc, PG.reh_fp32_dec], meta: rmeta, viz: RH.fp32, note: () => t("n_reh")});
  if (PG.reh_wo_enc) register({id: "reh-weightonly", label: "Whisper rehearsal · weight-only int8", short: "weight-only int8", sub: ["ดู seams", "see the seams"], group: "prec", graphs: [PG.reh_wo_enc, PG.reh_wo_dec], meta: rmeta, viz: RH.weightonly_int8, note: () => t("n_reh_wo")});
  if (PG.reh_static_enc) register({id: "reh-static", label: "Whisper rehearsal · static int8", short: "static int8", sub: ["ไฟล์ใหญ่ขึ้น?", "bigger file?"], group: "prec", graphs: [PG.reh_static_enc, PG.reh_static_dec], meta: rmeta, viz: RH.static_int8, note: () => t("n_reh_static")});
  if (PG.reh_int4_enc) register({id: "reh-int4", label: "Whisper rehearsal · dynamic int4", short: "dynamic int4", sub: ["int4", "int4"], group: "prec", graphs: [PG.reh_int4_enc, PG.reh_int4_dec], meta: rmeta, viz: RH.dyn_int4, note: () => t("n_reh_int4")});
})();
// English is the home page; a Thai browser that has not picked a language gets a one-line offer (no automatic redirect)
(function thaiHint() { try {
  if (window.PAGE_LANG !== "en" || !/^https?:$/.test(location.protocol) || localStorage.getItem("mx_lang")) return;
  if (!(navigator.languages || [navigator.language || ""]).some(l => /^th/i.test(l))) return;
  const a = document.createElement("a"); a.href = (window.PAGE_ALT || "th/") + location.hash; a.textContent = "อ่านเป็นภาษาไทย →"; a.id = "thHint";
  a.style.cssText = "position:fixed;right:18px;bottom:18px;z-index:50;padding:10px 16px;border-radius:999px;background:#e8114a;color:#fff;font-weight:600;text-decoration:none;box-shadow:0 6px 20px rgba(0,0,0,.25)";
  a.onclick = () => { try { localStorage.setItem("mx_lang", "th"); } catch (e) {} };
  const x = document.createElement("button"); x.textContent = "×"; x.setAttribute("aria-label", "close");
  x.style.cssText = "margin-left:10px;background:none;border:0;color:#fff;font-size:16px;cursor:pointer";
  x.onclick = e => { e.preventDefault(); e.stopPropagation(); try { localStorage.setItem("mx_lang", "en"); } catch (err) {} a.remove(); };
  a.appendChild(x); document.body.appendChild(a);
} catch (e) {} })();
wire();
S.lens = pref("lens") || "weights";
const h0 = location.hash.slice(1), last = pref("last");
setModel(MODELS.some(m => m.id === h0) ? h0 : MODELS.some(m => m.id === last) ? last : MODELS[0].id, true);
if (/^hf=/.test(h0)) { $("#hf").value = decodeURIComponent(h0.slice(3)); setTimeout(() => $("#hfGo").click(), 50); } // shareable link: #hf=owner/repo
DB.all().then(list => { (list || []).sort((a, b) => b.savedAt - a.savedAt).forEach(o => { if (!MODELS.some(m => m.id === o.id)) register({...o, group: "own", note: () => t(o.noteKey || "n_upload", o.noteVars)}); });
  const want = MODELS.some(m => m.id === h0) ? h0 : last; if (want && S.M.id !== want && MODELS.some(m => m.id === want)) setModel(want, true); else renderPicker(); });
})();
