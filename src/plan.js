// Where to cut: concrete size levers read from the anatomy, the weights' shapes and config.json.
// Shared by the website (app.js) and the Python package (run in QuickJS), so both give the same numbers.
// plan(I, shapes, cfg, opt) -> {total, vocab, kv, levers[]}; vocabScripts(pieces, specials) -> token counts per script.
const Plan = (() => {
  const prod = s => s.reduce((a, b) => a * b, 1);
  const n0 = n => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");   // QuickJS has no locale formatting
  const tcfg = c => (c && (c.text_config || c.language_config || c.llm_config || c.thinker_config && c.thinker_config.text_config)) || c || {};
  const layerNo = n => { const m = n.match(/(?:layers?|blocks?|h|block)\.(\d+)\./); return m ? +m[1] : null; };

  // ---------------------------------------------------------------- KV cache, layer by layer
  function kvProfile(I, shapes, cfg, bits = 16) {
    const c = tcfg(cfg), back = (I.comps || []).find(x => x.role === "backbone"); if (!back) return null;
    if ((I.hp && I.hp.kv_latent) || c.kv_lora_rank) return null;   // latent attention (MLA) caches a compressed vector, not k/v widths
    const by = {};
    for (const [n, t] of Object.entries(I.tensors || {})) {
      if (t.comp !== back.id || t.kind !== "attn" || !(t.role === "k" || t.role === "v") || !/weight$/.test(n)) continue;
      const L = layerNo(n), s = shapes[n]; if (L == null || !s || s.length !== 2) continue;
      (by[L] = by[L] || {})[t.role] = s[0];
    }
    const Ls = Object.keys(by).map(Number).sort((a, b) => a - b); if (!Ls.length) return null;
    const types = c.layer_types || [];
    // layers past num_hidden_layers are extra (multi-token prediction); only num_kv_shared_layers marks shared caches
    const nL = c.num_hidden_layers || Ls[Ls.length - 1] + 1, shared = c.num_kv_shared_layers || 0;
    const win = c.sliding_window || c.attention_window_size || null, B = bits / 8;
    // linear-attention / state-space layers keep a fixed-size state, not a cache that grows with the context
    const rows = Ls.filter(L => L < nL && !/linear|mamba|ssm|recurrent|conv/.test(types[L] || "")).map(L => {
      const w = (by[L].k || 0) + (by[L].v || by[L].k || 0), sliding = /sliding|local|chunk/.test(types[L] || "") && win;
      return {layer: L, kv_width: w, cached: L < nL - shared, sliding: !!sliding, window: sliding ? win : null};
    });
    if (!rows.length) return null;
    const cached = rows.filter(r => r.cached), grow = cached.filter(r => !r.sliding), slide = cached.filter(r => r.sliding);
    const perTok = grow.reduce((a, r) => a + r.kv_width * B, 0), slidePerTok = slide.reduce((a, r) => a + r.kv_width * B, 0);
    const naive = rows.reduce((a, r) => a + r.kv_width * B, 0);
    return {layers: rows.length, cached: cached.length, shared: rows.length - cached.length, full: grow.length, sliding: slide.length, window: win,
            per_token: perTok, sliding_per_token: slidePerTok, sliding_cap: slidePerTok * (win || 0), naive_per_token: naive,
            unused_kv_params: rows.filter(r => !r.cached).reduce((a, r) => a + r.kv_width * (back.d || 0), 0)};
  }

  // ---------------------------------------------------------------- the plan
  function plan(I, shapes, cfg, opt = {}) {
    const c = tcfg(cfg), comps = I.comps || [], back = comps.find(x => x.role === "backbone");
    const total = opt.total || comps.reduce((a, x) => a + (x.params || 0), 0) || 1, hp = I.hp || {};
    const V = hp.vocab || c.vocab_size, d = back && back.d;
    const T = Object.entries(I.tensors || {}).filter(([n]) => shapes[n]);
    // tables indexed by token id: the token embedding (width d, or a factorized narrower one), the output head, and
    // per-layer embeddings (Gemma 3n / 4: width = layers x hidden_size_per_layer_input). Only for a text backbone.
    const perL = c.hidden_size_per_layer_input || null, nLay = c.num_hidden_layers || (back && back.layers) || 0;
    const isPLE = (n, w) => /per_layer/.test(n) || (perL && nLay && w === perL * nLay);
    const textLM = back && back.modality === "text" && I.family !== "diffusion";
    const tables = !textLM ? [] : T.filter(([n, t]) => t.comp === back.id && (t.kind === "embed" || t.kind === "head") && shapes[n].length === 2 && shapes[n][0] === V && !/position|pos_emb|wpe|token_type/i.test(n))
      .map(([n, t]) => { let w = shapes[n][1]; const ple = t.kind === "embed" && isPLE(n, w);
        // wider than d and not per-layer: packed quantized storage the normalizer could not unpack; count d values per row
        if (!ple && d && w > d) w = d;
        const what = t.kind === "head" ? "output head" : ple ? "per-layer embedding (PLE)" : w === d ? "token embedding" :
          w >= 32 && /embed|wte|word|tok|shared/i.test(n) ? "token embedding (factorized)" : null;
        return what && {name: n, kind: t.kind, width: w, params: V * w, what}; }).filter(Boolean);
    if (!tables.some(x => x.kind === "embed" && !x.what.startsWith("per-layer"))) tables.length = 0;   // a classifier head alone is not a vocabulary
    const rowCost = tables.reduce((a, x) => a + x.width, 0), vocabParams = tables.reduce((a, x) => a + x.params, 0);
    const levers = [], pct = x => x / total;
    // 1. vocabulary: every row costs the width of every table indexed by token id
    const pipe0 = comps.some(y => /(^|\.)(vae|unet|loras?|diffusion_models?)(\.|$)/i.test(y.id));
    if (V && V >= 1000 && vocabParams && !pipe0 && (V >= 32000 || vocabParams / total >= 0.05)) {
      const keep = opt.keep_tokens || null;
      const pieces = Math.min(opt.tok_pieces || V, V);   // rows past the tokenizer file are kept: padding or special tokens it does not list
      const cut = keep ? Math.max(pieces - keep, 0) * rowCost : null;
      levers.push({id: "vocab", title: "Trim the vocabulary to the languages you use", params: vocabParams, share: pct(vocabParams),
        saves: cut, saves_share: cut != null ? pct(cut) : null, row_cost: rowCost, vocab: V, keep_tokens: keep,
        detail: `${n0(V)} tokens × ${n0(rowCost)} values per token (` + tables.map(x => `${x.what} ${x.width}`).join(" + ") +
                `). Every token you drop removes ${n0(rowCost)} weights.` + (keep ? "" : " Count the tokens your languages need to see the saving."),
        how: "Count tokenizer pieces by script, keep your scripts plus specials, bytes, digits and punctuation, remap the ids and slice every table above. Words you drop fall back to smaller pieces or bytes (if the tokenizer has byte fallback): retest quality."});
    }
    // 2. per-layer embeddings: a lookup, not compute
    const ple = tables.filter(x => x.what.startsWith("per-layer"));
    if (ple.length) {
      const p = ple.reduce((a, x) => a + x.params, 0);
      levers.push({id: "ple", title: "Per-layer embeddings: keep them out of fast memory", params: p, share: pct(p), saves: null,
        detail: `${ple.map(x => x.name).join(", ")} holds ${(p / 1e9).toFixed(2)}B weights (${(100 * pct(p)).toFixed(0)}% of the model), but each token reads only one row of ${ple[0].width} values.`,
        how: "Memory-map or offload the table to flash/CPU and fetch rows per token, try a lower precision for it (it is only looked up, never multiplied; measure quality), and trim it together with the vocabulary."});
    }
    // 3. modality towers you may not need
    const edges = I.edges || [], fedBy = id => edges.filter(e => e.to === id).map(e => e.from);
    const isAdapterOf = (a, enc) => a.role === "adapter" && (a.id.startsWith(enc.id + ".") || fedBy(a.id).includes(enc.id));
    // an encoder read through cross-attention (Whisper, T5) is the model's input path, not an optional tower
    const crossFed = new Set(edges.filter(e => /cross/.test(e.via || e.label || e.kind || "")).map(e => e.from));
    // an optional input tower: a vision or audio encoder that feeds the language model. Recognised by name, since
    // width-based edges are not reliable enough here; generation pipelines (VAE, LoRA, denoiser), speech-output stacks
    // (codec, vocoder, talker, acoustic decoder) and MTP heads are never offered.
    const pipeline = comps.some(y => /vae|unet|lora|diffusion|(^|\.)(transformer|text_encoders?)(\.|$)/i.test(y.id));
    // a language model that writes text: a [V, d] output head, or config.json saying the head is tied. hp.tie alone
    // is not enough: a dual encoder's text tower (CLIP, SigLIP, CLAP) also has a token table and no head.
    const generative = back && (tables.some(x => x.kind === "head") || c.tie_word_embeddings === true || (cfg || {}).tie_word_embeddings === true);
    const INPUT = /vision|visual|image|vit\b|siglip|clip|(^|\.)sam|audio_tower|audio_encoder|audio_model|speech_encoder|whisper|embed_vision|embed_audio|mm_proj|multi_modal|projector/i;
    const OUTPUT = /decoder|vocoder|codec|talker|acoustic|residual|feat_|token2wav|code_predictor|tokenizer|dit|mtp|draft|nextn|bigvgan|vae/i;
    const modKind = id => /audio|speech|whisper|wav|clap/i.test(id) ? "audio" : "vision";
    const towers = comps.filter(x => textLM && generative && tables.length && !pipeline && (x.role === "encoder" || x.role === "adapter") && x.modality !== "text" &&
                                 !crossFed.has(x.id) && I.family !== "encoder-decoder" && INPUT.test(x.id) && !OUTPUT.test(x.id));
    // the model's own purpose: a dual encoder (CLIP, SigLIP, CLAP) has no optional tower; an ASR model needs its audio
    // tower and an OCR model its vision tower, whatever the request
    const ident = JSON.stringify([(cfg || {}).model_type, (cfg || {}).architectures, opt.name || ""]).toLowerCase();
    const dual = /clip|siglip|clap|align|owl|blip/.test(ident) && !/llava|vl|gemma|qwen/.test(ident);
    const needs = k => k === "audio" ? /asr|transcri|speech|whisper|voxtral|stt/.test(ident) : /ocr/.test(ident);
    const isProj = x => x.role === "adapter" || /projector|multi_modal|mm_proj|connector|embed_vision|embed_audio/i.test(x.id);
    const encs = towers.filter(x => !isProj(x)), removable = new Set();
    // with a single encoder tower every projector belongs to it; otherwise go by the projector's name
    const mk = x => isProj(x) && encs.length === 1 ? modKind(encs[0].id) : modKind(x.id);
    for (const x of towers) {
      // a projector belongs to the tower of its modality (gemma-3: vision_tower + multi_modal_projector)
      if (dual || needs(mk(x))) continue;
      if (isProj(x) && encs.some(y => mk(y) === mk(x))) continue;
      if (x.role === "adapter" && comps.some(y => y.role === "encoder" && isAdapterOf(x, y))) continue;
      const first = encs.find(y => mk(y) === mk(x)) === x;
      const kids = comps.filter(y => y !== x && (y.id.startsWith(x.id + ".") || isAdapterOf(y, x) ||
        (first && isProj(y) && towers.includes(y) && mk(y) === mk(x)))), p = (x.params || 0) + kids.reduce((a, y) => a + (y.params || 0), 0);
      if (p > (back.params || 0)) continue;   // a "tower" larger than the language model is the main model (CLIP, detection, ...)
      const kind = mk(x);
      [x, ...kids].forEach(y => removable.add(y.id));
      levers.push({id: "tower:" + x.id, title: `Drop the ${kind} tower if you only send text`, params: p, share: pct(p), saves: p, saves_share: pct(p),
        detail: `${x.id}${kids.length ? " with " + kids.map(y => y.id).join(", ") : ""}: ${(p / 1e6).toFixed(1)}M weights.`,
        how: `Export only the language model (and the ${kind} tower when a request needs it); the text path never reads these weights.`});
    }
    // 4. KV cache, layer by layer
    const kv = kvProfile(I, shapes, cfg);
    if (kv && kv.unused_kv_params) levers.push({id: "kv_shared", title: "K/V projections that shared-KV layers do not use", params: kv.unused_kv_params,
      share: pct(kv.unused_kv_params), saves: kv.unused_kv_params, saves_share: pct(kv.unused_kv_params),
      detail: `${kv.shared} layers reuse the KV cache of earlier layers (num_kv_shared_layers), yet the file still stores their k/v projections; they are read only when running without a KV cache.`,
      how: "Check that your runtime always runs with a cache and skips them, then leave them out of the export."});
    // 5. where the compute weights sit
    // every component the model keeps (backbone, encoders, adapters), not only the backbone: Whisper's 24 encoder
    // layers hold most of its feed-forward weights. Towers offered for removal above are left out.
    let byComp = opt.ffn_by_comp;
    if (!byComp) { byComp = {}; for (const [n, t] of T) if (t.comp && (t.kind === "ffn" || t.kind === "moe")) byComp[t.comp] = (byComp[t.comp] || 0) + prod(shapes[n]); }
    const keptFfn = Object.entries(byComp).filter(([id]) => !removable.has(id) && comps.some(x => x.id === id)).sort((a, b) => b[1] - a[1]);
    const ffn = keptFfn.reduce((a, [, v]) => a + v, 0);
    const fmtB = v => v >= 1e9 ? (v / 1e9).toFixed(2) + "B" : (v / 1e6).toFixed(1) + "M";
    const ffnParts = keptFfn.map(([id, v]) => `${id} ${fmtB(v)}`).join(", ");
    if (ffn && ffn / total >= 0.05) levers.push({id: "ffn", title: hp.experts ? "Quantize the feed-forward and expert blocks first" : "Quantize the feed-forward blocks first", params: ffn, share: pct(ffn), saves: ffn / 2, saves_share: pct(ffn / 2),
      parts: keptFfn.map(([id, v]) => ({id, params: v})),
      detail: `${fmtB(ffn)} feed-forward weights (${ffnParts}); 8-bit halves them versus 16-bit, 4-bit quarters them.`,
      how: "Weight-only int8/int4 on the FFN with per-channel or group scales; keep norms, router and the embedding row lookups at higher precision until measured."});
    // 6. mixture of experts: all experts sit in memory, a token reads only top-k of them
    const E = hp.experts, k = hp.experts_per_token || c.num_experts_per_tok || c.moe_topk || c.top_k_experts;
    const moe = opt.moe_params != null ? opt.moe_params : T.filter(([n, t]) => t.comp === (back && back.id) && t.kind === "moe").reduce((a, [n]) => a + prod(shapes[n]), 0);
    if (E && moe && moe / total >= 0.2) levers.push({id: "moe", title: "Experts: keep the inactive ones off the fast memory", params: moe, share: pct(moe), saves: null,
      detail: `${E} experts per layer hold ${(moe / 1e9).toFixed(2)}B weights (${(100 * pct(moe)).toFixed(0)}%)` + (k ? `; each token uses ${k}, about ${(100 * k / E).toFixed(1)}% of them.` : "."),
      how: "Offload experts to CPU/flash and cache the hot ones, quantize experts harder than attention, or prune experts your traffic never routes to (measure routing on your own data first)."});
    levers.sort((a, b) => (b.saves || b.params) - (a.saves || a.params));
    return {total, vocab: V ? {size: V, row_cost: rowCost, params: vocabParams, tables} : null, kv, levers};
  }

  // ---------------------------------------------------------------- tokenizer pieces by script
  const SCRIPTS = [["Thai", /\p{Script=Thai}/u], ["Han", /\p{Script=Han}/u], ["Hiragana", /\p{Script=Hiragana}/u], ["Katakana", /\p{Script=Katakana}/u],
    ["Hangul", /\p{Script=Hangul}/u], ["Latin", /\p{Script=Latin}/u], ["Cyrillic", /\p{Script=Cyrillic}/u], ["Arabic", /\p{Script=Arabic}/u],
    ["Devanagari", /\p{Script=Devanagari}/u], ["Greek", /\p{Script=Greek}/u], ["Hebrew", /\p{Script=Hebrew}/u], ["Khmer", /\p{Script=Khmer}/u],
    ["Lao", /\p{Script=Lao}/u], ["Myanmar", /\p{Script=Myanmar}/u], ["Bengali", /\p{Script=Bengali}/u], ["Tamil", /\p{Script=Tamil}/u],
    ["Georgian", /\p{Script=Georgian}/u], ["Armenian", /\p{Script=Armenian}/u], ["Ethiopic", /\p{Script=Ethiopic}/u], ["Telugu", /\p{Script=Telugu}/u], ["Kannada", /\p{Script=Kannada}/u], ["Malayalam", /\p{Script=Malayalam}/u], ["Gujarati", /\p{Script=Gujarati}/u], ["Gurmukhi", /\p{Script=Gurmukhi}/u], ["Oriya", /\p{Script=Oriya}/u], ["Sinhala", /\p{Script=Sinhala}/u], ["Tibetan", /\p{Script=Tibetan}/u], ["Mongolian", /\p{Script=Mongolian}/u], ["Thaana", /\p{Script=Thaana}/u], ["Syriac", /\p{Script=Syriac}/u], ["Javanese", /\p{Script=Javanese}/u], ["Balinese", /\p{Script=Balinese}/u], ["Tifinagh", /\p{Script=Tifinagh}/u], ["Cherokee", /\p{Script=Cherokee}/u], ["Canadian_Aboriginal", /\p{Script=Canadian_Aboriginal}/u]];
  const LX = ["Latin-ASCII", "Latin-ext"];   // accented Latin is kept for English too: names and loanwords (González, café) appear in real English text
  const LANG = {en: LX, th: ["Thai"], zh: ["Han"], cn: ["Han"], ja: ["Han", "Hiragana", "Katakana"], ko: ["Hangul"], vi: LX, id: LX, ms: LX,
    fr: LX, de: LX, es: LX, pt: LX, it: LX, nl: LX, tl: LX, cs: LX, pl: LX, fi: LX, sv: LX, da: LX, no: LX, is: LX, tr: LX, ro: LX, hu: LX, gl: LX,
    ca: LX, et: LX, lt: LX, lv: LX, sk: LX, sl: LX, hr: LX, sw: LX, ru: ["Cyrillic"], uk: ["Cyrillic"], bg: ["Cyrillic"], sr: ["Cyrillic", "Latin-ASCII", "Latin-ext"],
    kk: ["Cyrillic"], ar: ["Arabic"], fa: ["Arabic"], ur: ["Arabic"], hi: ["Devanagari"], mr: ["Devanagari"], ne: ["Devanagari"], bn: ["Bengali"], ta: ["Tamil"],
    el: ["Greek"], he: ["Hebrew"], km: ["Khmer"], lo: ["Lao"], my: ["Myanmar"], ka: ["Georgian"], hy: ["Armenian"], am: ["Ethiopic"], ti: ["Ethiopic"],
    te: ["Telugu"], kn: ["Kannada"], ml: ["Malayalam"], gu: ["Gujarati"], pa: ["Gurmukhi"], or: ["Oriya"], si: ["Sinhala"], bo: ["Tibetan"], dz: ["Tibetan"],
    mn: ["Cyrillic"], dv: ["Thaana"], jv: LX, su: LX, ms_arab: ["Arabic"], ps: ["Arabic"], sd: ["Arabic"], ug: ["Arabic"], yi: ["Hebrew"], be: ["Cyrillic"],
    mk: ["Cyrillic"], ky: ["Cyrillic"], tg: ["Cyrillic"], az: LX, uz: LX, ha: LX, yo: LX, ig: LX, zu: LX, xh: LX, af: LX, mt: LX, ga: LX, cy: LX, eu: LX, sq: LX, bs: LX};
  const B2U = (() => { const bs = []; for (let i = 33; i <= 126; i++) bs.push(i); for (let i = 161; i <= 172; i++) bs.push(i); for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice(); let n = 0; for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
    const m = {}; bs.forEach((b, i) => { m[String.fromCharCode(cs[i])] = b; }); return m; })();
  function decodeByteLevel(s) {
    const bytes = []; for (const ch of s) { const b = B2U[ch]; if (b == null) return s; bytes.push(b); }
    let out = "", i = 0;   // UTF-8 decode; an incomplete sequence becomes U+FFFD (a fragment of a longer character)
    while (i < bytes.length) { const b = bytes[i]; const n = b < 0x80 ? 1 : b >= 0xf0 ? 4 : b >= 0xe0 ? 3 : b >= 0xc0 ? 2 : 0;
      if (!n || i + n > bytes.length) { out += "�"; i++; continue; }
      let cp = n === 1 ? b : b & (0xff >> (n + 1)); for (let k = 1; k < n; k++) cp = (cp << 6) | (bytes[i + k] & 0x3f);
      out += String.fromCodePoint(cp); i += n; }
    return out;
  }
  function scriptOf(piece) {
    if (/^<.*>$/.test(piece) || /^\[.*\]$/.test(piece)) return /^<0x[0-9A-Fa-f]{2}>$/.test(piece) ? "bytes" : "special";
    const s = piece.replace(/[▁\s]/g, ""); if (!s) return "common";
    if (s.includes("�")) return "fragment";
    const found = new Set();
    for (const ch of s) {
      if (/[\p{N}\p{P}\p{S}\p{Z}\p{M}\p{Cf}\p{Script=Common}\p{Script=Inherited}]/u.test(ch)) continue;   // Common/Inherited: shared marks such as the Japanese long vowel ー or the Arabic tatweel   // Cf: zero-width space/joiner, common in Thai text and emoji
      const sc = (SCRIPTS.find(([, re]) => re.test(ch)) || ["other"])[0];
      found.add(sc === "Latin" ? (/[A-Za-z]/.test(ch) ? "Latin-ASCII" : "Latin-ext") : sc);
    }
    return found.size ? [...found].sort().join("+") : "common";
  }
  function vocabScripts(pieces, opt = {}) {
    const byteLevel = !!opt.byte_level, counts = {}, per = new Array(pieces.length);
    pieces.forEach((p, i) => { const s = scriptOf(byteLevel && !/^<.*>$/.test(p) ? decodeByteLevel(p) : p); per[i] = s; counts[s] = (counts[s] || 0) + 1; });
    const keep = langs => {
      const ok = new Set(["special", "bytes", "common", "fragment"]); (langs || []).forEach(l => (LANG[l.toLowerCase()] || [l]).forEach(s => ok.add(s)));
      return per.reduce((a, s) => a + (s.split("+").every(x => ok.has(x)) ? 1 : 0), 0);
    };
    const mask = langs => {   // "1" per kept piece, in id order
      const ok = new Set(["special", "bytes", "common", "fragment"]); (langs || []).forEach(l => (LANG[l.toLowerCase()] || [l]).forEach(x => ok.add(x)));
      return per.map(x => x.split("+").every(y => ok.has(y)) ? "1" : "0").join("");
    };
    return {total: pieces.length, counts, keep, keep_for: langs => keep(langs), mask};
  }
  return {plan, kvProfile, vocabScripts, scriptOf, LANG};
})();
if (typeof module !== "undefined") module.exports = Plan;
