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
const DTC = {fp32: "#5b6bf0", fp16: "#9aa8ff", bf16: "#9aa8ff", int8: "#e5007d", int4: "#ff8fd0", uint8: "#e5007d", fp8: "#ffb3e0"};
const C = {red: "#e60012", redL: "#ff5a67", mag: "#e5007d", magL: "#ff7cc4", blue: "#5b6bf0", blueL: "#a3abfa", ok: "#4ade80", amber: "#fbbf24", ink: "#f4f5f8", soft: "#b9c0d4", dim: "#7d86a0"};

// ================================================================== models
const MODELS = [];
const S = {M: null, lens: "weights", el: 0, dl: 0, open: null, tokens: 40, step: -1, playing: false, zoom: false, mapLens: "size"};
function register(def) { const i = MODELS.findIndex(m => m.id === def.id); if (i >= 0) MODELS[i] = def; else MODELS.push(def); renderPicker(); return def; }
function build(def) {
  if (def.M) return def.M;
  let M;
  if (def.graphs) { M = {A: anatomyFromOps(def.graphs), graphs: def.graphs, meta: def.meta, kind: "ops"}; }
  else if (def.tensors) { const meta = def.meta || metaFromTensors(def.tensors, def.label); M = {A: anatomyFromTensors(def.tensors, meta, def.kind || "tensors"), meta, kind: def.kind || "tensors"}; }
  M.id = def.id; M.label = def.label; M.note = def.note; M.viz = def.viz; M.group = def.group;
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
  renderAll();
}
const fam = M => M.A.comps.encoder && M.A.comps.decoder ? "encdec" : M.A.comps.decoder ? "decoder" : "encoder";

// ================================================================== hero + picker
function renderPicker() {
  const box = $("#picker"); if (!box) return;
  const g = k => MODELS.filter(m => m.group === k);
  const chip = m => `<button class="mchip${S.M && S.M.id === m.id ? " on" : ""}" data-m="${esc(m.id)}"><b>${esc(m.short || m.label)}</b><span>${esc(m.sub ? L2(...m.sub) : "")}</span></button>`;
  box.innerHTML = `<div class="pk"><div class="xk">${t("pick_templates")}</div><div class="mrow">${g("tpl").map(chip).join("")}</div></div>
    ${g("prec").length ? `<div class="pk"><div class="xk">${t("pick_precision")}</div><div class="mrow">${g("prec").map(chip).join("")}</div></div>` : ""}
    ${g("own").length ? `<div class="pk"><div class="xk">${t("pick_own")}</div><div class="mrow">${g("own").map(chip).join("")}</div></div>` : ""}`;
  $$("#picker [data-m]").forEach(b => b.onclick = () => setModel(b.dataset.m));
}
function renderHero() {
  const M = S.M, A = M.A, tot = Object.values(A.comps).reduce((a, c) => merge(a, c.tot), part());
  const nL = Object.values(A.comps).reduce((a, c) => a + c.layers.length, 0), ops = M.kind === "ops" ? tot.ops : null;
  const f = fam(M), mod = M.meta.modality;
  $("#cur").innerHTML = `<div class="xk">${t("now_showing")}</div><h3>${esc(M.label)}</h3>
    <div class="tags"><span class="tag" style="background:#5b6bf0">${t("fam_" + f)}</span><span class="tag" style="background:#e5007d">${t("mod_" + mod)}</span><span class="tag ghost">${t("src_" + M.kind)}</span></div>
    <div class="kpis">${[[fP(tot.params), t("kpi_weights")], [fB(tot.bytes), t("kpi_size")], [String(nL), t("kpi_layers")], ops != null ? [String(ops), t("kpi_ops")] : [M.meta.vocab ? M.meta.vocab.toLocaleString() : "–", t("kpi_vocab")]].map(([b, s]) => `<div class="kpi"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("")}</div>
    <p class="note">${M.note ? esc(typeof M.note === "function" ? M.note() : M.note) : ""}</p>`;
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
      x.fillStyle = px < gx ? `rgba(163,171,250,${.2 + .5 * p.a})` : (p.lane < 20 ? `rgba(91,107,240,${.3 + .6 * p.a})` : `rgba(229,0,125,${.35 + .6 * p.a})`);
      x.beginPath(); x.arc(px * W, py * H, (1 + p.a * 1.5) * devicePixelRatio, 0, 6.28); x.fill(); } })();
})();

// ================================================================== part texts (what goes in, what comes out, why it matters)
function shapes(M) {
  const m = M.meta, d = m.d || 768, mod = m.modality, Se = mod === "audio" ? (m.seqEnc || 1500) : mod === "image" ? (m.patches || 196) + 1 : "n";
  return {d, Se, V: m.vocab || "V", nm: m.nmels || 80, ff: m.ffn || 4 * d};
}
const EX = {audio: "วันนี้อากาศดีมาก", textDec: ["Edge", " AI", " runs", " on"], textDecNext: [" phones", " devices", " the", " CPU", " your"], textEnc: ["[CLS]", "edge", "ai", "is", "[MASK]", "[SEP]"], img: ["tabby cat", "tiger cat", "lynx", "fox", "dog"]};
function partInfo(col, M) {
  const s = shapes(M), m = M.meta, nE = M.A.comps.encoder ? M.A.comps.encoder.layers.length : 0, nD = M.A.comps.decoder ? M.A.comps.decoder.layers.length : 0;
  const ffShare = (() => { const L = (M.A.comps.encoder || M.A.comps.decoder).layers[0]; return L && L.regions.ffn ? pct(L.regions.ffn._tot.params, L.tot.params) : "⅔"; })();
  const dec1 = M.A.comps.encoder ? "[1, " + s.d + "]" : "[n, " + s.d + "]";
  const P = {
    in_audio: {t: ["Audio", "Audio"], what: ["เสียง 16 kHz ยาวสูงสุด 30 วินาที = ตัวเลข 480,000 ค่า · ยังไม่มี weight", "16 kHz sound, up to 30 s = 480,000 numbers. No weights yet."], opt: ["เสียงโทรศัพท์ตัดความถี่เกิน 3.4 kHz ทิ้งตั้งแต่ตรงนี้", "A phone line throws away everything above 3.4 kHz right here."], i: null, o: ["wave", "[480,000]"]},
    in_text: {t: ["Text → tokens", "Text → tokens"], what: ["ตัดข้อความเป็น token แล้วแปลงเป็นเลข id จากพจนานุกรม " + (m.vocab ? m.vocab.toLocaleString() : "") + " คำ", "Cut the text into tokens and map each to an id from a " + (m.vocab ? m.vocab.toLocaleString() : "") + "-entry vocabulary."], opt: ["ภาษาไทยมักใช้ token มากกว่าอังกฤษหลายเท่า → ลำดับยาวขึ้น", "Thai often needs several times more tokens than English → longer sequences."], i: null, o: ["tokens", "[n]"]},
    in_image: {t: ["Image", "Image"], what: ["ภาพสี 224×224 = ตัวเลข 150,528 ค่า", "A 224×224 colour image = 150,528 numbers."], opt: ["ความละเอียดภาพกำหนดจำนวน token ทั้งโมเดล", "Image resolution sets the token count for the whole model."], i: null, o: ["image", "[3, 224, 224]"]},
    mel: {t: ["Log-Mel", "Log-Mel"], what: [`ตัดเสียงเป็นช่วงละ 10 ms วัดพลังงานใน ${s.nm} ช่องความถี่ = “ภาพของเสียง”`, `Slice the sound into 10 ms frames and measure energy in ${s.nm} bands — a picture of sound.`], opt: ["คำนวณก่อนเข้าโมเดล ไม่มี weight", "Computed before the model. No weights."], i: ["wave", "[480,000]"], o: ["spec", `[${s.nm}, 3000]`]},
    conv: {t: ["Conv stem", "Conv stem"], what: [`conv 2 ชั้นแปลงภาพเสียงเป็นเวกเตอร์ ${s.d} มิติ และลดเวลาครึ่งหนึ่ง 3000 → ${s.Se}`, `Two convolutions turn the spectrogram into ${s.d}-dim vectors and halve time: 3000 → ${s.Se}.`], opt: ["เล็กมาก (< 1%) — กันไว้ราคาถูก", "Tiny (< 1%) — cheap to protect."], i: ["spec", `[${s.nm}, 3000]`], o: ["mat", `[${s.Se}, ${s.d}]`]},
    patch: {t: ["Patch embedding", "Patch embedding"], what: [`ตัดภาพเป็นช่อง 16×16 จำนวน ${m.patches || 196} ช่อง แต่ละช่องกลายเป็นเวกเตอร์ ${s.d} มิติ + [CLS]`, `Cut the image into ${m.patches || 196} 16×16 patches; each becomes a ${s.d}-dim vector, plus a [CLS] token.`], opt: ["conv เดียว เล็ก แต่เป็นด่านแรกที่เห็นภาพ", "One small conv — the first thing that sees pixels."], i: ["image", "[3, 224, 224]"], o: ["patches", `[${s.Se}, ${s.d}]`]},
    embed_enc: {t: ["Token embedding", "Token embedding"], what: [`token id → หยิบแถวจากตาราง ${s.V}×${s.d} + ตำแหน่ง`, `token id → one row of a ${s.V}×${s.d} table, plus position.`], opt: ["ตารางใหญ่ แต่ขาเข้าอ่านแค่แถวเดียวต่อ token", "A big table, but only one row is read per token."], i: ["tokens", "[n]"], o: ["mat", `[n, ${s.d}]`]},
    embed_dec: {t: ["Token embedding", "Token embedding"], what: [`token id → หยิบแถวจากตาราง ${s.V}×${s.d}`, `token id → one row of the ${s.V}×${s.d} table.`], opt: [m.tie ? "ตารางเดียวกับ logits (tied) — เก็บครั้งเดียว" : "แยกจากตาราง logits", m.tie ? "Same table as the logits (tied) — stored once." : "Separate from the logits table."], i: ["tokens", "[n]"], o: ["mat", `[n, ${s.d}]`]},
    enc: {t: [`Encoder × ${nE}`, `Encoder × ${nE}`], what: [`${nE} layer ซ้อนกัน แต่ละ layer = attention (ทุกตำแหน่งดูกันเอง) + feed-forward · รันครั้งเดียวต่อ input`, `${nE} stacked layers, each = attention (every position looks at every other) + feed-forward. Runs once per input.`], opt: [`ส่วนที่ใหญ่ที่สุดของ compute · feed-forward = ${ffShare} ของ weight ใน layer`, `Most of the compute. Feed-forward is ${ffShare} of each layer's weights.`], i: ["mat", `[${s.Se}, ${s.d}]`], o: ["mat2", `[${s.Se}, ${s.d}]`]},
    kv: {t: ["Encoder out → K/V", "Encoder out → K/V"], what: ["ผลของ encoder ถูกเก็บไว้เป็น Key/Value ให้ decoder ทุกชั้นมาอ่าน · คำนวณครั้งเดียว อ่านซ้ำทุก token", "The encoder output is kept as Keys/Values for every decoder layer. Made once, read every token."], opt: ["บีบแล้วไม่ช่วยเรื่องเวลา แต่ error ตรงนี้กระจายไปทุก token", "Squeezing it saves no time, but an error here spreads to every token."], i: ["mat2", `[${s.Se}, ${s.d}]`], o: ["kv", `${nD}×2×[${s.Se}, ${s.d}]`]},
    dec: {t: [`Decoder × ${nD}`, `Decoder × ${nD}`], what: M.A.comps.encoder ? [`สร้างคำทีละ token: ดูคำที่เขียนแล้ว (KV cache) → ฟังเสียง/ข้อมูลจาก encoder (cross-attention) → feed-forward`, `Writes one token at a time: look at what's written (KV cache) → listen to the encoder (cross-attention) → feed-forward.`] : [`${nD} layer: attention ดูเฉพาะ token ก่อนหน้า (KV cache) + feed-forward · รันใหม่ทุก token`, `${nD} layers: attention over earlier tokens only (KV cache) + feed-forward. Runs again for every token.`], opt: ["รันทุก token → อ่าน weight ซ้ำหลายสิบรอบ · byte ต่อ weight กำหนดความเร็ว", "Runs every token → re-reads its weights dozens of times. Bytes per weight set the speed."], i: ["vec", dec1], o: ["vec2", "[1, " + s.d + "]"]},
    lmhead: {t: ["Norm → logits", "Norm → logits"], what: [`คูณเวกเตอร์กับตารางคำศัพท์ ${s.V}×${s.d} → คะแนนของทุกคำ แล้วเลือกคำถัดไป`, `Multiply by the ${s.V}×${s.d} vocabulary table → a score for every token, then pick the next one.`], opt: ["ก้อนเดียวที่ใหญ่สุดของฝั่ง decoder · ภาษาไทยมักพังที่นี่ก่อนเมื่อบีบ int4", "The single biggest block on the decoder side; Thai tends to break here first at int4."], i: ["vec2", "[1, " + s.d + "]"], o: ["probs", `[${s.V}]`]},
    pool: {t: ["Pooler / head", "Pooler / head"], what: ["สรุปทั้งลำดับเป็นเวกเตอร์เดียว (จาก [CLS]) ใช้ต่อกับงาน classify หรือค้นหา", "Summarize the whole sequence into one vector (from [CLS]) for classification or search."], opt: ["เล็ก", "Small."], i: ["mat2", `[n, ${s.d}]`], o: ["vec2", `[${s.d}]`]},
    cls_head: {t: ["Classifier", "Classifier"], what: [`เอาเวกเตอร์ [CLS] ไปคูณ → คะแนนของ ${m.labels || 1000} คลาส`, `Take the [CLS] vector → scores for ${m.labels || 1000} classes.`], opt: ["เล็ก", "Small."], i: ["mat2", `[${s.Se}, ${s.d}]`], o: ["probs_img", `[${m.labels || 1000}]`]},
    out_text: {t: ["Text out", "Text out"], what: M.meta.modality === "audio" ? ["ต่อ token เป็นประโยค แล้ววัด WER/CER", "Join the tokens into a sentence, then measure WER/CER."] : ["ต่อ token ใหม่ท้ายข้อความ แล้ววนกลับไปสร้างคำต่อไป", "Append the new token and loop back for the next one."], opt: [M.meta.modality === "audio" ? "gate ของ workshop: WER บนเสียงโทรศัพท์ห้ามเพิ่มเกิน 2 จุด" : "ความเร็ว = token ต่อวินาที", M.meta.modality === "audio" ? "Workshop gate: phone-audio WER may rise by at most 2 points." : "Speed = tokens per second."], i: ["probs", `[${s.V}]`], o: ["text", ""]},
    out_vec: {t: ["Embedding out", "Embedding out"], what: ["เวกเตอร์ที่สรุปความหมายของข้อความ ใช้ classify / ค้นหา / จัดกลุ่ม", "A vector that captures the meaning of the text — for classification, search, clustering."], opt: ["", ""], i: ["vec2", `[${s.d}]`], o: null},
    out_class: {t: ["Label", "Label"], what: ["คลาสที่คะแนนสูงสุด", "The highest-scoring class."], opt: ["", ""], i: ["probs_img", ""], o: ["label", ""]},
  };
  const p = P[col.kind] || {t: [col.kind, col.kind], what: ["", ""], opt: ["", ""]};
  return {title: L2(...p.t), what: L2(...p.what), opt: L2(...p.opt), i: p.i, o: p.o};
}

// ---------------- tiny visuals for the I/O panel
function viz(kind, w = 210, h = 104) {
  let s = "", r = (i, j) => (Math.sin(i * 12.9898 + j * 78.233) * 43758.5453) % 1, ab = v => Math.abs(v);
  if (kind === "wave") { let d = ""; for (let i = 0; i <= 80; i++) { const x = 6 + i * (w - 12) / 80, a = Math.sin(i * .7) * Math.sin(i * .17) * h * .38; d += (i ? "L" : "M") + x.toFixed(1) + " " + (h / 2 + a).toFixed(1); } s = `<path d="${d}" stroke="${C.soft}" stroke-width="2" fill="none"/>`; }
  else if (kind === "spec") { for (let i = 0; i < 26; i++) for (let j = 0; j < 12; j++) { const v = Math.max(0, Math.sin(i * .5 + j * .35) * .5 + .5 - j / 18); s += `<rect x="${4 + i * 7.9}" y="${h - 4 - (j + 1) * 8}" width="7.3" height="7.3" fill="rgba(163,171,250,${(.1 + v * .85).toFixed(2)})"/>`; } }
  else if (kind === "tokens" || kind === "text") {
    const toks = S.M.meta.modality === "audio" ? [EX.audio] : S.M.A.comps.decoder ? (kind === "text" ? [...EX.textDec, EX.textDecNext[0]] : EX.textDec) : EX.textEnc; let x = 6, y = 26;
    toks.forEach((tk, i) => { const tw = Math.max(26, tk.length * 8.4 + 14); if (x + tw > w) { x = 6; y += 32; } const last = kind === "text" && i === toks.length - 1;
      s += `<rect x="${x}" y="${y - 17}" width="${tw}" height="24" rx="7" fill="${last ? "rgba(74,222,128,.18)" : "rgba(255,255,255,.07)"}" stroke="${last ? C.ok : "rgba(255,255,255,.2)"}"/><text x="${x + tw / 2}" y="${y}" text-anchor="middle" font-size="13" fill="${last ? C.ok : C.ink}">${esc(tk.trim() || "␣")}</text>`;
      if (kind === "tokens" && S.M.meta.modality !== "audio") s += `<text x="${x + tw / 2}" y="${y + 18}" text-anchor="middle" font-size="9.5" fill="${C.dim}" font-family="var(--mono)">${Math.floor(ab(r(i, 3)) * 30000)}</text>`; x += tw + 6; });
    if (kind === "text" && S.M.meta.modality === "audio") s = `<text x="${w / 2}" y="${h / 2 + 8}" text-anchor="middle" font-size="22" font-weight="700" fill="${C.ok}">${EX.audio}</text>`; }
  else if (kind === "image" || kind === "patches") { for (let i = 0; i < 14; i++) for (let j = 0; j < 7; j++) { const v = .5 + .5 * Math.sin(i * .45 + j * .6), c2 = `hsl(${30 + v * 30},${40 + 30 * v}%,${25 + 35 * v}%)`; s += `<rect x="${w / 2 - 98 + i * 14}" y="${4 + j * 14}" width="${kind === "patches" ? 12.5 : 14}" height="${kind === "patches" ? 12.5 : 14}" fill="${c2}"/>`; } }
  else if (kind === "mat" || kind === "mat2") { const cols = 30, rows = 9; for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) { const v = ab(r(i + (kind === "mat2" ? 7 : 0), j)); const c2 = kind === "mat2" ? `rgba(229,0,125,${(.15 + .8 * v).toFixed(2)})` : `rgba(91,107,240,${(.15 + .8 * v).toFixed(2)})`; s += `<rect x="${4 + i * 6.8}" y="${6 + j * 10.5}" width="6" height="9.5" rx="1.5" fill="${c2}"/>`; } }
  else if (kind === "kv") { ["K", "V"].forEach((l, k) => { for (let i = 0; i < 13; i++) for (let j = 0; j < 7; j++) s += `<rect x="${8 + k * 104 + i * 7}" y="${18 + j * 11}" width="6.2" height="10" rx="1.5" fill="rgba(91,107,240,${(.2 + .75 * ab(r(i + k * 9, j))).toFixed(2)})"/>`; s += `<text x="${54 + k * 104}" y="13" text-anchor="middle" font-size="12" font-weight="700" fill="${C.ink}">${l}</text>`; }); }
  else if (kind === "vec" || kind === "vec2") { for (let i = 0; i < 28; i++) { const v = ab(r(i + (kind === "vec2" ? 5 : 0), 2)); s += `<rect x="${6 + i * 7.2}" y="${h / 2 - 9}" width="6.4" height="18" rx="1.5" fill="${kind === "vec2" ? `rgba(229,0,125,${(.2 + .75 * v).toFixed(2)})` : `rgba(255,124,196,${(.2 + .75 * v).toFixed(2)})`}"/>`; } }
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
function dtColor(p) { const e = Object.entries(p.dt).sort((a, b) => b[1] - a[1]); if (p.seams && !e.length) return "#ff2a2a"; return e.length ? DTC[e[0][0]] || "#888" : "#2a3150"; }
function lensFill(p, max) {
  if (!p || !p.has) return {fill: "rgba(255,255,255,.03)"};
  if (S.lens === "precision") return {fill: dtColor(p) + "33", stroke: dtColor(p)};
  const k = max ? Math.min(1, LENS[S.lens].v(p) / max) : 0; return {fill: `rgba(229,0,125,${(.05 + .45 * Math.sqrt(k)).toFixed(3)})`};
}
function bar(x, y, w, p, max) { if (!p || !p.has || S.lens === "precision") return ""; const k = max ? Math.min(1, LENS[S.lens].v(p) / max) : 0; return `<rect x="${x}" y="${y}" width="${w}" height="5" rx="2.5" fill="rgba(255,255,255,.08)"/><rect x="${x}" y="${y}" width="${Math.max(2, w * k)}" height="5" rx="2.5" fill="#ff5aa8"/>`; }

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
  let s = `<defs><marker id="xa" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#7d86a0"/></marker>
    <linearGradient id="rib" x1="0" x2="1"><stop offset="0" stop-color="#a3abfa" stop-opacity=".32"/><stop offset="1" stop-color="#5b6bf0" stop-opacity=".32"/></linearGradient>
    <linearGradient id="ribd" x1="0" x2="1"><stop offset="0" stop-color="#e5007d" stop-opacity=".32"/><stop offset="1" stop-color="#ff7cc4" stop-opacity=".32"/></linearGradient>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="7" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
  const num = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"];
  cols.forEach((c, i) => { const info = partInfo(c, M), cx = c.x + c.w / 2; const sh = info.o ? info.o[1] : "";
    s += `<text x="${cx}" y="34" text-anchor="middle" class="xh">${num[i]} ${esc(info.title)}</text><text x="${cx}" y="54" text-anchor="middle" class="xs" fill="#a3abfa">${esc(sh)}</text>`; });
  s += `<line x1="20" y1="68" x2="${W - 20}" y2="68" stroke="rgba(255,255,255,.08)"/>`;
  // ribbons between neighbours in the main row
  for (let i = 0; i < cols.length - 1; i++) { const a = cols[i], b = cols[i + 1]; if (b.kind === "kv" && false) continue;
    const x1 = a.x + a.w, x2 = b.x, yy = y0 + 110, dec = (a.comp === "decoder" || a.kind === "lmhead" || a.kind === "dec" || b.kind === "dec" && F === "decoder" || a.kind === "embed_dec"), w = 16;
    if (a.kind === "kv" && b.kind === "dec") continue;
    s += `<path d="M${x1} ${yy - w / 2} H${x2} V${yy + w / 2} H${x1}Z" fill="url(#${dec ? "ribd" : "rib"})"/><path d="M${x1} ${yy} H${x2}" stroke="#fff" stroke-opacity=".45" stroke-width="1.5" class="flowline"/>`; }
  const sel = id => S.open === id || (cur && cur.id === id);
  const box = (id, x, y, w, h, title, p, mx, wide = false, cap = "") => { const f = lensFill(p, mx), u = p && p.has ? esc(LENS[S.lens].unit(p)) : "", sb = p && p.has ? esc(LENS[S.lens].sub(p) || "") : "";
    const inner = wide ? `<text x="${x + 14}" y="${y + 23}" class="xt">${esc(title)}</text><text x="${x + w - 14}" y="${y + 23}" text-anchor="end" class="xm">${u}</text><text x="${x + 14}" y="${y + 42}" class="xs">${esc(cap)}</text>${h >= 80 ? `<text x="${x + w - 14}" y="${y + 60}" text-anchor="end" class="xs">${sb}</text>` : ""}`
      : `<text x="${x + w / 2}" y="${y + 24}" text-anchor="middle" class="xt">${esc(title)}</text><text x="${x + w / 2}" y="${y + h - 32}" text-anchor="middle" class="xm">${u}</text>${sb.length * 6.6 < w ? `<text x="${x + w / 2}" y="${y + h - 17}" text-anchor="middle" class="xs">${sb}</text>` : ""}`;
    return `<g class="xbox${sel(id) ? " sel" : ""}" data-x="${id}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" fill="${f.fill}" stroke="${f.stroke || "rgba(255,255,255,.22)"}" stroke-width="${sel(id) ? 2.6 : 1.3}" ${sel(id) ? 'filter="url(#glow)"' : ""}/>${inner}${bar(x + 12, y + h - 9, w - 24, p, mx)}</g>`; };
  const ioBox = (c, v) => `<g class="xbox${sel(c.kind) ? " sel" : ""}" data-x="${c.kind}"><rect x="${c.x}" y="${y0 + 40}" width="${c.w}" height="140" rx="14" fill="rgba(255,255,255,.04)" stroke="${sel(c.kind) ? "#ff7cc4" : "rgba(255,255,255,.2)"}" stroke-width="${sel(c.kind) ? 2.6 : 1.3}"/>
      <g transform="translate(${c.x + 4},${y0 + 70}) scale(${(c.w - 8) / 210})">${viz(v).replace('class="iov"', "")}</g><text x="${c.x + c.w / 2}" y="${y0 + 200}" text-anchor="middle" class="xs">${t("no_weights")}</text></g>`;
  let zoomLayer = "";
  const stack = (c, Cc, Lx, n, comp) => {
    if (S.zoom && S.zoom.comp === comp && !c._z) { const zw = Math.max(c.w, 640); zoomLayer = stack({...c, _z: true, x: c.x + c.w / 2 - zw / 2, w: zw}, Cc, Lx, n, comp); return ""; }
    const x = c.x, y = y0 - 10, w = c.w, h = 250, isE = comp === "encoder", col = isE ? "#5b6bf0" : "#e5007d", idx = isE ? S.el : S.dl;
    let o = ""; for (let k = 3; k >= 1; k--) o += `<rect x="${x + k * 7}" y="${y - k * 7}" width="${w}" height="${h}" rx="18" fill="${isE ? "#10162c" : "#1a1028"}" stroke="${col}" stroke-opacity="${.15 + .1 * (3 - k)}"/>`;
    const f = lensFill(Cc.layersTot, max);
    o += `<g class="${sel(c.id) ? "selstack" : ""}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="18" fill="${isE ? "#121a36" : "#1d1230"}" stroke="${col}" stroke-width="${sel(c.id) ? 2.6 : 1.6}" ${sel(c.id) ? 'filter="url(#glow)"' : ""}/><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="18" fill="${f.fill}" opacity=".6"/></g>
      <text x="${x + 18}" y="${y + 28}" class="xt">${isE ? "Encoder" : "Decoder"} ${t("layer")}</text>
      <g class="xnav" data-nav="${isE ? "e" : "d"}-"><rect x="${x + w - 128}" y="${y + 10}" width="26" height="26" rx="8" class="xr"/><text x="${x + w - 115}" y="${y + 28}" text-anchor="middle" class="xt">‹</text></g>
      <text x="${x + w - 72}" y="${y + 28}" text-anchor="middle" class="xm">${idx + 1} / ${n}</text>
      <g class="xnav" data-nav="${isE ? "e" : "d"}+"><rect x="${x + w - 42}" y="${y + 10}" width="26" height="26" rx="8" class="xr"/><text x="${x + w - 29}" y="${y + 28}" text-anchor="middle" class="xt">›</text></g>`;
    if (Lx) {
      const regs = [["self", L2("Self-attention", "Self-attention"), isE ? "LN → Q K V → softmax → out ⊕" : L2("อ่าน + เขียน KV cache", "reads + writes KV cache")], ["cross", "Cross-attention", L2("ฟัง K/V จาก encoder", "listens to the encoder K/V")], ["ffn", "Feed-forward", Lx.regions.ffn && Lx.regions.ffn.gate ? "gate · up → act → down ⊕" : `LN → up → act → down ⊕`]].filter(r => Lx.regions[r[0]]);
      const bh = regs.length === 3 ? 60 : 82, gap = regs.length === 3 ? 8 : 14; let yy = y + 46;
      if (S.zoom && S.zoom.comp === comp) o += zoomSteps(Lx, x, y, w, comp);
      else regs.forEach(([rk, title, cap]) => { o += box((isE ? "e_" : "d_") + rk, x + 16, yy, w - 32, bh, title, Lx.regions[rk]._tot, regMax, true, cap); yy += bh + gap; });
    }
    o += `<text x="${x + w / 2}" y="${y + h + 22}" text-anchor="middle" class="xs">${esc(t("all_layers", {n, v: LENS[S.lens].unit(Cc.layersTot)}))}${isE ? "" : " " + t("per_token")}</text>${bar(x + 40, y + h + 30, w - 80, Cc.layersTot, max)}`;
    return o;
  };
  const embBelow = F === "encdec";
  cols.forEach(c => {
    if (c.io) s += ioBox(c, {in_audio: "wave", in_text: "tokens", in_image: "image", mel: "spec", out_text: "text", out_vec: "vec2", out_class: "label"}[c.kind] || "mat");
    else if (c.stack) s += stack(c, c.comp === "encoder" ? E : D, c.comp === "encoder" ? Le : Ld, c.comp === "encoder" ? nE : nD, c.comp);
    else s += box(c.id, c.x, y0 + 40, c.w, 140, partInfo(c, M).title.replace("Encoder out → K/V", "K/V ×" + nD).replace("Token embedding", "Embed").replace("Patch embedding", "Patches").replace("Norm → logits", "→ logits"), c.part, max);
  });
  s += zoomLayer;
  // decoder-side extras
  if (D) {
    const dc = cols.find(c => c.kind === "dec"), hd = cols.find(c => c.kind === "lmhead"), out = cols.find(c => c.kind === "out_text");
    let emb = cols.find(c => c.kind === "embed_dec");
    if (embBelow) { emb = {x: dc.x, w: Math.min(210, dc.w), id: "d_stem", below: true}; s += box("d_stem", emb.x, y0 + 296, emb.w, 64, L2("Token embed", "Token embed"), D.stemTot, max, true, "token id → vector"); s += `<path d="M${emb.x + 75} ${y0 + 296} V${y0 + 248}" stroke="#ff7cc4" stroke-width="2" class="flowline" marker-end="url(#xa)"/>`; }
    const kv = cols.find(c => c.kind === "kv"); if (kv) s += `<path d="M${kv.x + kv.w} ${y0 + 110} C ${kv.x + kv.w + 30} ${y0 + 110}, ${dc.x - 30} ${y0 + 160}, ${dc.x} ${y0 + 160}" stroke="#5b6bf0" stroke-width="3" fill="none" stroke-dasharray="6 6" class="flowline" marker-end="url(#xa)"/>`;
    const ex = emb.below ? emb.x + emb.w : emb.x + emb.w / 2, ey = emb.below ? y0 + 345 : y0 + 180;
    s += emb.below ? `<path d="M${out.x + out.w / 2} ${y0 + 180} V${y0 + 345} H${ex + 3}" stroke="#ff7cc4" stroke-width="2" fill="none" stroke-dasharray="6 6" class="flowline loopline" marker-end="url(#xa)"/>`
      : `<path d="M${out.x + out.w / 2} ${y0 + 180} V${y0 + 345} H${ex} V${ey + 4}" stroke="#ff7cc4" stroke-width="2" fill="none" stroke-dasharray="6 6" class="flowline loopline" marker-end="url(#xa)"/>`;
    s += `<text x="${(out.x + ex) / 2 + 40}" y="${y0 + 338}" text-anchor="middle" class="xs" fill="#ff7cc4">${esc(t("next_token", {t: S.tokens}))}</text>`;
    if (D.headTot.shared) { const sx = emb.below ? emb.x + emb.w : emb.x + emb.w / 2 + 14, sy = emb.below ? y0 + 308 : y0 + 180;
      s += emb.below ? `<path d="M${sx} ${sy} H${hd.x + hd.w / 2} V${y0 + 182}" stroke="#fbbf24" stroke-width="1.6" stroke-dasharray="4 5" fill="none" marker-end="url(#xa)"/>` : `<path d="M${sx} ${sy} V${y0 + 290} H${hd.x + hd.w / 2} V${y0 + 182}" stroke="#fbbf24" stroke-width="1.6" stroke-dasharray="4 5" fill="none" marker-end="url(#xa)"/>`;
      s += `<text x="${(sx + hd.x + hd.w / 2) / 2}" y="${(emb.below ? sy : y0 + 290) - 7}" text-anchor="middle" class="xs" fill="#fbbf24">${esc(t("tied_note"))}</text>`; }
  }
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
    o += `<text x="${x + 18}" y="${yy + 12}" class="xs" fill="${C.soft}">${rk === "self" ? "self-attention" : rk === "cross" ? "cross-attention" : "feed-forward"}</text>`;
    const cw = (w - 36) / Math.max(roles.length, 5);
    roles.forEach((r, k) => { const bx = x + 18 + k * cw; o += `<g class="zs" style="animation-delay:${(ri * roles.length + k) * .18}s"><rect x="${bx}" y="${yy + 18}" width="${cw - 6}" height="${rh - 24}" rx="7" fill="rgba(255,255,255,.07)" stroke="${comp === "encoder" ? "#a3abfa" : "#ff7cc4"}" stroke-opacity=".6"/>
      <text x="${bx + (cw - 6) / 2}" y="${yy + 18 + (rh - 24) / 2 + 4}" text-anchor="middle" font-size="${cw < 70 ? 10 : 12}" fill="${C.ink}">${esc(stepName(r, true))}</text></g>`; });
    yy += rh + 6;
  });
  o += `<circle r="5" fill="#fff" filter="url(#glow)"><animateMotion dur="${1.6 * rows.length}s" repeatCount="indefinite" path="${rows.map((r, i) => `M${x + 18} ${y + 50 + 18 + (rh - 24) / 2 + i * (rh + 6)} H${x + 18 + r[1].length * (w - 36) / Math.max(r[1].length, 5) - 6}`).join(" ")}"/></circle>`;
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
  pool: ["pooler", "สรุปเป็นเวกเตอร์เดียว", "summarize into one vector"], cls: ["classifier", "คะแนนของแต่ละคลาส", "a score per class"], misc: ["other", "อื่นๆ", "other"], seam: ["seams", "QUANTIZE/DEQUANTIZE", "QUANTIZE/DEQUANTIZE"],
};
const stepName = (r, short) => (STEP[r] || [r])[0];
const ORDER = {self: [["ln"], ["qkv", "q", "k", "v"], ["cache"], ["reshape"], ["scores"], ["softmax"], ["context"], ["out"], ["res"]], cross: [["ln"], ["q", "k", "v"], ["reshape"], ["scores"], ["softmax"], ["context"], ["out"], ["res"]],
  ffn: [["ln"], ["gate", "fc1"], ["act"], ["fc2"], ["res"]], stem: [["conv"], ["embed"], ["glue"], ["pos"], ["ln"], ["mask"], ["io"]], head: [["ln"], ["crosskv"], ["logits"], ["pool"], ["cls"], ["glue"]]};
function renderDetail() {
  const box = $("#xdetail"), id = S.open, M = S.M, E = M.A.comps.encoder, D = M.A.comps.decoder;
  if (!id) { box.innerHTML = `<div class="xempty">${t("click_hint")}</div>`; return; }
  let parts = null, order = null, crumb = "", title = "", what = "", opt = "";
  const reg = id.match(/^([ed])_(self|cross|ffn)$/);
  if (reg) { const Cc = reg[1] === "e" ? E : D, L = Cc.layers[reg[1] === "e" ? S.el : S.dl]; parts = L.regions[reg[2]] || {}; order = ORDER[reg[2]]; crumb = `${reg[1] === "e" ? "Encoder" : "Decoder"} › ${t("layer")} ${L.idx}`; title = reg[2] === "self" ? "Self-attention" : reg[2] === "cross" ? "Cross-attention" : "Feed-forward"; }
  else { const col = (LAY ? LAY.cols : []).find(c => c.id === id || c.kind === id); if (col) { const info = partInfo(col, M); title = info.title; what = info.what; opt = info.opt; if (col.part) { parts = col.id === "e_stem" ? E.stem : col.id === "e_head" ? E.head : col.id === "d_stem" ? D.stem : D.head; order = /stem/.test(col.id) ? ORDER.stem : ORDER.head; } crumb = col.comp || (col.where || "").split(".")[0]; }
    if (id === "d_stem") { parts = D.stem; order = ORDER.stem; title = "Token embedding"; crumb = "decoder"; } }
  const tot = parts ? sumParts(parts) : null;
  let svg = "";
  if (parts) {
    const steps = order.map(g => g.filter(r => parts[r] && parts[r].has)).filter(g => g.length); Object.keys(parts).forEach(r => { if (r !== "_tot" && !order.flat().includes(r) && parts[r].has) steps.push([r]); });
    const mx = Math.max(...Object.entries(parts).filter(([k]) => k !== "_tot").map(([, p]) => LENS[S.lens].v(p) || 0), 1e-9);
    const colW = 140, bw = 124, bh = 92, gap = 14, W = Math.max(700, steps.length * colW + 40), rows = Math.max(...steps.map(g => g.length)), H = rows * (bh + gap) + 40, cy = H / 2 + 4;
    svg = `<svg viewBox="0 0 ${W} ${H}" style="min-width:${Math.round(W * .8)}px"><defs><marker id="xa2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#7d86a0"/></marker></defs>`;
    steps.forEach((g, i) => { const x = 20 + i * colW, yA = cy - (g.length * (bh + gap) - gap) / 2;
      if (i) svg += `<path d="M${x - colW + bw + 2} ${cy} H${x - 4}" stroke="#7d86a0" stroke-width="2" class="flowline" marker-end="url(#xa2)"/>`;
      g.forEach((r, k) => { const p = parts[r], y = yA + k * (bh + gap), f = lensFill(p, mx), nm = stepName(r), wt = p.w[0] ? p.w[0].shape.join("×") : (p.out ? `[${p.out.join(",")}]` : "");
        svg += `<g class="xstep" data-r="${r}"><rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="12" fill="${f.fill}" stroke="${f.stroke || (p.seams ? "#ff2a2a" : "rgba(255,255,255,.25)")}" stroke-width="1.4"/>
          <text x="${x + bw / 2}" y="${y + 22}" text-anchor="middle" class="xt" font-size="${nm.length > 14 ? 12.5 : 14.5}">${esc(nm)}</text><text x="${x + bw / 2}" y="${y + 40}" text-anchor="middle" class="xs">${esc(wt)}</text>
          <text x="${x + bw / 2}" y="${y + 62}" text-anchor="middle" class="xm" font-size="14">${esc(LENS[S.lens].unit(p))}</text><text x="${x + bw / 2}" y="${y + 78}" text-anchor="middle" class="xs">${p.ops ? p.ops + " op" + (p.ops > 1 ? "s" : "") : ""}${p.seams ? " · " + p.seams + " seam" : ""}</text></g>`; }); });
    svg += "</svg>";
  }
  const facts = tot ? [[t("f_weights"), fB(tot.bytes) + " · " + fP(tot.params)], [t("f_stored"), dtLabel(tot)], [t("f_compute"), fF(tot.flops) + (tot.est ? " " + t("est") : "")], M.kind === "ops" ? [t("f_ops"), tot.ops] : null, M.kind === "ops" ? [t("f_seams"), tot.seams] : null, tot._frag != null ? [t("f_frag"), (tot._frag * 100).toFixed(1) + "%"] : null].filter(Boolean) : [];
  box.innerHTML = `<div class="xdh"><div><div class="xk">${esc(crumb)}</div><h3>${esc(title)}</h3>${what ? `<p>${esc(what)}</p>` : ""}</div><button class="btn sm" id="xclose">✕ ${t("close")}</button></div>
    <div class="xdg"><div><div class="hscroll xsteps">${svg || ""}</div><div class="xhov" id="xhov">${parts ? t("see_ops_note") : ""}</div></div>
    <div class="xfacts">${facts.map(([k, v]) => `<div class="x3row"><span>${k}</span><b>${esc(String(v))}</b></div>`).join("")}${opt ? `<div class="x3opt"><b>${t("opt_lens")}</b>${esc(opt)}</div>` : ""}</div></div>`;
  $("#xclose").onclick = () => { S.open = null; renderBig(); renderDetail(); };
  $$("#xdetail .xstep").forEach(g => g.onclick = g.onmouseenter = () => { const e = STEP[g.dataset.r] || [g.dataset.r, "", ""], p = parts[g.dataset.r]; const types = Object.entries(p.types).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}×${v}`).join(", ");
    $("#xhov").innerHTML = `<b>${esc(e[0])}</b> — ${esc(L2(e[1], e[2]))}${types ? `<br><span class="mono" style="font-size:12px;color:var(--dim)">${esc(types)}</span>` : ""}`; });
}

// ================================================================== layer map
function renderMap() {
  const M = S.M, A = M.A, svg = $("#mapSvg"); const rows = [];
  ["encoder", "decoder"].forEach(c => { const Cc = A.comps[c]; if (!Cc) return; rows.push({c, blocks: [{id: c + ".stem", p: Cc.stemTot, lab: "stem"}, ...Cc.layers.map(L => ({id: c + ".L" + L.idx, p: L.tot, L, lab: String(L.idx)})), {id: c + ".head", p: Cc.headTot, lab: "head"}].filter(b => b.p.has)}); });
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
      if (S.mapLens === "prec") fill = dtColor(b.p); if (S.mapLens === "frag") { const s = b.p._frag; const k = !(s > 0) ? -1 : (Math.log(s) - Math.log(fmn)) / ((Math.log(fmx) - Math.log(fmn)) || 1); fill = k < 0 ? "#2a3150" : `rgb(${Math.round(60 + 170 * k)},${Math.round(40 + 10 * (1 - k))},${Math.round(120 - 80 * k)})`; }
      out += `<g class="blk" data-c="${r.c}" data-l="${b.L ? b.L.idx : ""}" data-id="${b.id}"><title>${esc(b.id)} · ${fB(b.p.bytes)} · ${esc(dtLabel(b.p))}${b.p._frag != null ? " · int4 err " + (b.p._frag * 100).toFixed(1) + "%" : ""}</title>`;
      if (b.L && (S.mapLens === "size" || S.mapLens === "read") && w > 8) { const att = (b.L.regions.self ? b.L.regions.self._tot.bytes : 0) + (b.L.regions.cross ? b.L.regions.cross._tot.bytes : 0), fa = b.p.bytes ? att / b.p.bytes : .33;
        out += `<rect x="${x}" y="${y}" width="${w}" height="${h * fa}" rx="3" fill="${fill}" opacity=".55"/><rect x="${x}" y="${y + h * fa}" width="${w}" height="${h * (1 - fa)}" fill="${fill}"/>`; }
      else out += `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${fill}" opacity="${op}"/>`;
      out += `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="none" stroke="rgba(0,0,0,.3)"/>`;
      if (w > b.lab.length * 7.5) out += `<text x="${x + w / 2}" y="${y - 7}" text-anchor="middle" class="ms">${b.lab}</text>`;
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

// ================================================================== concepts (numbers from the selected model)
function renderConcepts() {
  const M = S.M, m = M.meta, s = shapes(M), mod = m.modality, L = (M.A.comps.encoder || M.A.comps.decoder).layers[0];
  const share = L && L.regions.ffn ? pct(L.regions.ffn._tot.params, L.tot.params) : "⅔";
  const first = mod === "audio" ? ["c_spec_t", t("c_spec_p", {m: s.nm})] : mod === "image" ? ["c_patch_t", t("c_patch_p", {p: m.patches || 196})] : ["c_tok_t", t("c_tok_p", {v: m.vocab ? m.vocab.toLocaleString() : "?"})];
  $("#c1t").textContent = t(first[0]); $("#c1p").textContent = first[1]; S.c1 = mod;
  $("#c2p").textContent = t("c_attn_p", {h: m.heads || (L && L.regions.self && L.regions.self.q ? "?" : "?")});
  $("#c3p").textContent = t("c_ffn_p", {d: s.d.toLocaleString(), f: (m.ffn || 4 * s.d).toLocaleString(), share});
  $("#c4p").textContent = M.A.comps.decoder ? t("c_kv_p") : t("c_kv_p_enc");
  $("#c5p").textContent = m.tie ? t("c_tie_p_yes", {v: m.vocab ? m.vocab.toLocaleString() : "V", d: s.d}) : t("c_tie_p_no");
  $("#c6p").textContent = t("c_seam_p");
  $("#cFfnD").textContent = s.d; $("#cFfnF").textContent = m.ffn || 4 * s.d;
}
(function conceptAnims() {
  const cv = $("#cSpec"), x = cv.getContext("2d"); let tt = 0; const cols = 120, rows = 40, buf = [];
  const col = i => { const a = []; for (let j = 0; j < rows; j++) { const f = j / rows; a.push(Math.min(1, Math.max(0, Math.sin(j * .9 + i * .07) * .5 + .5) * Math.exp(-f * 2.2) * (.5 + .5 * Math.sin(i * .19)) + Math.random() * .08)); } return a; };
  for (let i = 0; i < cols; i++) buf.push(col(i));
  (function draw() { requestAnimationFrame(draw); tt++; x.clearRect(0, 0, cv.width, cv.height);
    if (S.c1 === "text") { const toks = LANG === "en" ? ["Edge", " AI", " on", " phones", "!"] : ["วัน", "นี้", "อา", "กาศ", "ดี"]; const k = Math.floor(tt / 40) % (toks.length + 2); x.font = "600 30px sans-serif";
      let px = 30; toks.forEach((tk, i) => { const w = x.measureText(tk).width + 26; x.fillStyle = i < k ? "rgba(229,0,125,.25)" : "rgba(255,255,255,.06)"; x.strokeStyle = i < k ? "#ff7cc4" : "rgba(255,255,255,.2)"; x.beginPath(); x.roundRect(px, 90, w, 54, 10); x.fill(); x.stroke(); x.fillStyle = "#f4f5f8"; x.fillText(tk, px + 13, 127); if (i < k) { x.fillStyle = "#a3abfa"; x.font = "500 20px monospace"; x.fillText(String(1000 + i * 4137 % 30000), px + 8, 180); x.font = "600 30px sans-serif"; } px += w + 10; }); return; }
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
  setInterval(() => { tp = 1 - tp; T.innerHTML = `<rect x="200" y="70" width="200" height="160" rx="12" fill="#ff4fa8" opacity=".25" stroke="#ff4fa8"/>${Array.from({length: 10}, (_, i) => `<line x1="210" x2="390" y1="${86 + i * 15}" y2="${86 + i * 15}" stroke="#ff4fa8" stroke-opacity=".5"/>`).join("")}
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
    const Wd = cv.width, Hd = cv.height, lim = outlier ? 1 : .3, Y = v => Hd / 2 - v / lim * (Hd * .42), X = i => 40 + i * (Wd - 80) / 63; x.clearRect(0, 0, Wd, Hd); x.strokeStyle = "rgba(255,255,255,.08)"; x.lineWidth = 1;
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
  if (E) { add("Encoder · " + (S.M.meta.modality === "audio" ? "conv stem" : S.M.meta.modality === "image" ? "patch embedding" : "embedding"), E.stemTot, "in", "e"); add(`Encoder · self-attention ×${E.layers.length}`, sumR(E, "self"), "in", "e"); add(`Encoder · feed-forward ×${E.layers.length}`, sumR(E, "ffn"), "in", "e"); add(D ? "Encoder · final norm / K,V" : "Encoder · head", E.headTot, "in", "e"); }
  if (D) { add("Decoder · token embedding", D.stemTot, "tok", "d"); add(`Decoder · self-attention ×${D.layers.length}`, sumR(D, "self"), "tok", "d"); add(`Decoder · cross-attention ×${D.layers.length}`, sumR(D, "cross"), "tok", "d"); add(`Decoder · feed-forward ×${D.layers.length}`, sumR(D, "ffn"), "tok", "d"); add("Decoder · norm + logits", D.headTot, "tok", "d"); }
  rows.forEach(r => r.read = r.p.bytes * (r.per === "tok" ? tk : 1));
  if (D && D.headTot.shared) { const emb = rows.find(r => r.name.startsWith("Decoder · token")), lg = rows.find(r => r.name.startsWith("Decoder · norm")); const tw = D.stemTot.w.find(w => w.shape.length === 2 && w.shape[0] > 1000);
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
  if (M.meta.kv_heads && M.meta.heads && M.meta.kv_heads < M.meta.heads) out.push([C.ok, "✓", t("i_gqa", {kv: M.meta.kv_heads, h: M.meta.heads, x: M.meta.heads / M.meta.kv_heads})]);
  const dts = tot.dt; if (Object.keys(dts).length > 1) out.push([C.amber, "!", t("i_mixed", {m: dtLabel(tot)})]);
  if (M.kind === "ops") { const ty = tot.types, n = tot.ops, sh = ["RESHAPE", "TRANSPOSE", "SLICE", "CONCATENATION", "PAD", "GATHER_ND", "SQUEEZE", "EXPAND_DIMS"].reduce((a, k) => a + (ty[k] || 0), 0);
    out.push([C.dim, "i", t("i_shape", {s: pct(sh, n)})]); const ln = ["MEAN", "SQUARED_DIFFERENCE", "RSQRT"].reduce((a, k) => a + (ty[k] || 0), 0); if (ln) out.push([C.blueL, "i", t("i_ln", {n: Object.values(A.comps).reduce((a, c) => a + c.layers.reduce((b, Lx) => b + Object.values(Lx.regions).reduce((cc, R) => cc + (R.ln ? R.ln.ops : 0), 0), 0) + (c.headTot && c.head.ln ? c.head.ln.ops : 0), 0)})]);
    if (ty.DEQUANTIZE && !ty.QUANTIZE) out.push([C.red, "!", t("i_wo", {n: ty.DEQUANTIZE})]); if (ty.QUANTIZE) out.push([C.red, "!", t("i_static", {n: ty.QUANTIZE})]); }
  return out;
}
function renderReport() {
  const M = S.M, rows = shareRows(), tb = rows.reduce((a, r) => a + r.p.bytes, 0), fl = r => r.p.flops * (r.per === "tok" ? S.tokens : 1), tf = rows.reduce((a, r) => a + fl(r), 0), tr = rows.reduce((a, r) => a + r.read, 0), ops = M.kind === "ops";
  $("#rIns").innerHTML = insights().map(([c, i, tx]) => `<div class="ins"><i style="background:${c}22;color:${c}">${i}</i><div>${tx}</div></div>`).join("");
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
  const tfl = [...files].filter(f => /\.tflite$/i.test(f.name)); if (!tfl.length) return; ownNote(t("loading"));
  try { const gs = []; for (const f of tfl) { const comp = /enc/i.test(f.name) ? "encoder" : /dec/i.test(f.name) ? "decoder" : (tfl.length === 1 ? "model" : "encoder"); gs.push(graphFromTflite(parseTflite(await f.arrayBuffer()), comp, f.name)); }
    const enc = gs.some(g => g.component === "encoder"), dec = gs.some(g => g.component !== "encoder");
    const meta = {name: tfl.map(f => f.name).join(" + "), family: enc && dec ? "encdec" : enc ? "encoder" : "decoder", modality: gs.some(g => g.nodes.some(n => n.t === "CONV_2D")) && enc ? "audio" : "text", d: null, tie: true};
    const id = "up-" + Date.now(); register({id, label: meta.name, short: tfl.length > 1 ? "Uploaded (" + tfl.length + ")" : tfl[0].name, group: "own", graphs: gs, meta, note: () => t("n_upload")}); ownNote(""); setModel(id);
  } catch (e) { ownNote(`<b style="color:#ff5a67">${t("e_read")}</b> ${esc(e.message)}`); }
}
function ownNote(h) { $("#ownNote").innerHTML = h; }
async function hfLoad(input) {
  const m = String(input).trim().match(/(?:huggingface\.co\/)?([\w.-]+\/[\w.-]+)/); if (!m) throw new Error("owner/repo");
  const repo = m[1].replace(/\/(tree|blob|resolve)\/.*$/, ""), base = `https://huggingface.co/${repo}/resolve/main/`;
  let info; try { info = await (await fetch(`https://huggingface.co/api/models/${repo}?blobs=true`)).json(); } catch (e) { throw new Error("BLOCKED"); }
  if (info.error) throw new Error(info.error);
  const files = (info.siblings || []).map(s => ({n: s.rfilename, size: s.size || 0})), tfl = files.filter(f => /\.tflite$/i.test(f.n));
  let cfg = null; if (files.some(f => f.n === "config.json")) try { cfg = await (await fetch(base + "config.json")).json(); } catch (e) {}
  if (tfl.length) { const pick = tfl.filter(f => /enc|dec/i.test(f.n)).slice(0, 2), chosen = pick.length ? pick : tfl.slice(0, 1), gs = [];
    for (const f of chosen) { ownNote(`${t("loading")} ${esc(f.n)} (${fB(f.size)})`); gs.push(graphFromTflite(parseTflite(await (await fetch(base + f.n)).arrayBuffer()), /enc/i.test(f.n) ? "encoder" : /dec/i.test(f.n) ? "decoder" : "model", f.n)); }
    return {graphs: gs, meta: {name: repo, family: chosen.length > 1 ? "encdec" : "decoder", modality: "text", tie: true}, label: repo, note: () => t("n_upload")}; }
  const head = async url => { const b = await (await fetch(url, {headers: {Range: "bytes=0-7"}})).arrayBuffer(); const n = Number(new DataView(b).getBigUint64(0, true)); return parseSafetensorsHeader(await (await fetch(url, {headers: {Range: `bytes=0-${8 + n - 1}`}})).arrayBuffer()); };
  let tensors = {}; const idx = files.find(f => f.n === "model.safetensors.index.json");
  if (idx) { const j = await (await fetch(base + idx.n)).json(); for (const sh of [...new Set(Object.values(j.weight_map))]) Object.assign(tensors, await head(base + sh)); }
  else { const st = files.find(f => /\.safetensors$/.test(f.n)); if (!st) { if (cfg) { const r = tensorsFromConfig(cfg); return {tensors: r.tensors, meta: {...r.meta, name: repo}, kind: "config", label: repo, note: () => t("n_cfg", {mt: r.meta.model_type})}; } throw new Error("no .tflite / .safetensors / config.json"); } Object.assign(tensors, await head(base + st.n)); }
  let meta = metaFromTensors(tensors, repo); if (cfg) { try { const r = tensorsFromConfig(cfg); meta = {...meta, ...r.meta, name: repo}; } catch (e) {} }
  const n = Object.keys(tensors).length; return {tensors, meta, label: repo, note: () => t("n_hf_st", {n})};
}
function addOwn(r, short) { const id = "own-" + (r.label || "m").replace(/[^\w-]+/g, "-").slice(0, 40); register({id, label: r.label, short: short || r.label.split("/").pop(), group: "own", ...r}); MODELS.find(m => m.id === id).M = null; setModel(id); }

// ================================================================== everything
function renderAll() { applyI18n(); renderPicker(); renderHero(); renderBig(); buildPlay(); renderIO(); renderDetail(); renderMap(); renderConcepts(); renderReport(); if (window.qRelabel) qRelabel(); $("#tokV").textContent = S.tokens; $("#tokV2").textContent = S.tokens; }
function wire() {
  $$("[data-lang]").forEach(b => b.onclick = () => { LANG = b.dataset.lang; try { localStorage.setItem("mx_lang", LANG); } catch (e) {} renderAll(); });
  $$("[data-xl]").forEach(b => b.onclick = () => { S.lens = b.dataset.xl; renderBig(); renderDetail(); });
  $$("[data-ml]").forEach(b => b.onclick = () => { S.mapLens = b.dataset.ml; renderMap(); });
  $("#bPlay").onclick = () => S.playing ? stopPlay() : startPlay();
  $("#bNext").onclick = () => { stopPlay(); if (!PLAY.length) buildPlay(); goStep(S.step + 1); };
  $("#bPrev").onclick = () => { stopPlay(); if (!PLAY.length) buildPlay(); goStep(Math.max(0, S.step - 1)); };
  const tok = e => { S.tokens = +e.target.value; $("#tok").value = $("#tok2").value = S.tokens; renderBig(); renderMap(); renderReport(); $("#tokV").textContent = $("#tokV2").textContent = S.tokens; };
  $("#tok").oninput = tok; $("#tok2").oninput = tok;
  $("#file").onchange = e => loadFiles(e.target.files);
  $("#hfGo").onclick = async () => { ownNote(t("loading")); try { const r = await hfLoad($("#hf").value); ownNote(""); addOwn(r); } catch (e) { ownNote(e.message === "BLOCKED" ? t("e_blocked") : `<b style="color:#ff5a67">${t("e_read")}</b> ${esc(e.message)}`); } };
  $("#hf").addEventListener("keydown", e => { if (e.key === "Enter") $("#hfGo").click(); });
  $("#cfgOpen").onclick = () => $("#cfgDlg").classList.add("open"); $("#cfgClose").onclick = () => $("#cfgDlg").classList.remove("open");
  $("#cfgGo").onclick = () => { try { const cfg = JSON.parse($("#cfgTxt").value); const r = tensorsFromConfig(cfg); $("#cfgDlg").classList.remove("open"); addOwn({tensors: r.tensors, meta: r.meta, kind: "config", label: (cfg._name_or_path || r.meta.model_type || "config") + "", note: () => t("n_cfg", {mt: r.meta.model_type})}, r.meta.model_type); } catch (e) { $("#cfgErr").textContent = t("e_cfg") + e.message; } };
  $("#copyMd").onclick = async () => { try { await navigator.clipboard.writeText(S.md); $("#copyMd").textContent = t("copied"); } catch (e) { $("#copyMd").textContent = t("copy_fail"); } setTimeout(() => $("#copyMd").textContent = t("copy_md"), 1800); };
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), {threshold: 0, rootMargin: "0px 0px -6% 0px"}); $$(".reveal").forEach(el => io.observe(el));
  addEventListener("hashchange", () => { const h = location.hash.slice(1); if (h && MODELS.some(m => m.id === h) && (!S.M || S.M.id !== h)) setModel(h, true); });
}

// ================================================================== templates
(function templates() {
  const PG = window.PRE_GRAPHS || {}, RH = (typeof PRELOADED !== "undefined" && PRELOADED.rehearsal) || {}, TT = window.TEMPLATE_TENSORS || {};
  const wmeta = {name: "distill-whisper-th-medium", model_type: "whisper", family: "encdec", modality: "audio", d: 1024, heads: 16, kv_heads: 16, ffn: 4096, vocab: 51865, tie: true, nmels: 80, seqEnc: 1500};
  if (PG.medium_enc) register({id: "whisper-th-medium", label: "biodatlab/distill-whisper-th-medium", short: "Whisper medium (Thai)", sub: ["เสียง → ข้อความ · .tflite จริง", "speech → text · real .tflite"], group: "tpl", graphs: [PG.medium_enc, PG.medium_dec], meta: wmeta, note: () => t("n_whisper")});
  const tpl = (id, key, short, sub) => { const T = TT[key]; if (T) register({id, label: T.meta.name, short, sub, group: "tpl", tensors: T.tensors, meta: T.meta, note: () => t("n_tensors", {name: T.meta.name})}); };
  tpl("llama-3-2-1b", "llama32_1b", "Llama 3.2 1B", ["LLM · decoder-only", "LLM · decoder-only"]);
  tpl("gpt2", "gpt2", "GPT-2 small", ["LLM · decoder-only", "LLM · decoder-only"]);
  tpl("bert-base", "bert_base", "BERT base", ["ข้อความ · encoder-only", "text · encoder-only"]);
  tpl("vit-base", "vit_base", "ViT base", ["ภาพ · encoder-only", "image · encoder-only"]);
  const rmeta = {name: "whisper rehearsal", model_type: "whisper", family: "encdec", modality: "audio", d: 128, heads: 2, kv_heads: 2, ffn: 256, vocab: 51865, tie: true, nmels: 80, seqEnc: 1500};
  if (PG.reh_fp32_enc) register({id: "reh-fp32", label: "Whisper rehearsal · fp32", short: "fp32", sub: ["มี fragility", "with fragility"], group: "prec", graphs: [PG.reh_fp32_enc, PG.reh_fp32_dec], meta: rmeta, viz: RH.fp32, note: () => t("n_reh")});
  if (PG.reh_wo_enc) register({id: "reh-weightonly", label: "Whisper rehearsal · weight-only int8", short: "weight-only int8", sub: ["ดู seams", "see the seams"], group: "prec", graphs: [PG.reh_wo_enc, PG.reh_wo_dec], meta: rmeta, viz: RH.weightonly_int8, note: () => t("n_reh_wo")});
  if (PG.reh_int4_enc) register({id: "reh-int4", label: "Whisper rehearsal · dynamic int4", short: "dynamic int4", sub: ["int4", "int4"], group: "prec", graphs: [PG.reh_int4_enc, PG.reh_int4_dec], meta: rmeta, viz: RH.dyn_int4, note: () => t("n_reh_int4")});
})();
wire();
const h0 = location.hash.slice(1); setModel(MODELS.some(m => m.id === h0) ? h0 : MODELS[0].id, true);
})();
