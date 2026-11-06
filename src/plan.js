// Where to cut: concrete size levers read from the anatomy, the weights' shapes and config.json.
// Shared by the website (app.js) and the Python package (run in QuickJS), so both give the same numbers.
// plan(I, shapes, cfg, opt) -> {total, vocab, kv, levers[]}; vocabScripts(pieces, specials) -> token counts per script.
const Plan = (() => {
  const prod = s => s.reduce((a, b) => a * b, 1);
  const tcfg = c => (c && (c.text_config || c.language_config || c.llm_config || c.thinker_config && c.thinker_config.text_config)) || c || {};
  const layerNo = n => { const m = n.match(/(?:layers?|blocks?|h|block)\.(\d+)\./); return m ? +m[1] : null; };

  // ---------------------------------------------------------------- KV cache, layer by layer
  function kvProfile(I, shapes, cfg, bits = 16) {
    const c = tcfg(cfg), back = (I.comps || []).find(x => x.role === "backbone"); if (!back) return null;
    const by = {};
    for (const [n, t] of Object.entries(I.tensors || {})) {
      if (t.comp !== back.id || t.kind !== "attn" || !(t.role === "k" || t.role === "v") || !/weight$/.test(n)) continue;
      const L = layerNo(n), s = shapes[n]; if (L == null || !s || s.length !== 2) continue;
      (by[L] = by[L] || {})[t.role] = s[0];
    }
    const Ls = Object.keys(by).map(Number).sort((a, b) => a - b); if (!Ls.length) return null;
    const nL = c.num_hidden_layers || Ls.length, shared = c.num_kv_shared_layers || 0, types = c.layer_types || [];
    const win = c.sliding_window || c.attention_window_size || null, B = bits / 8;
    const rows = Ls.map(L => {
      const w = (by[L].k || 0) + (by[L].v || by[L].k || 0), sliding = /sliding|local|chunk/.test(types[L] || "") && win;
      return {layer: L, kv_width: w, cached: L < nL - shared, sliding: !!sliding, window: sliding ? win : null};
    });
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
    const tables = T.filter(([n, t]) => (t.kind === "embed" || t.kind === "head") && shapes[n].length === 2 && shapes[n][0] === V)
      .map(([n, t]) => ({name: n, kind: t.kind, width: shapes[n][1], params: prod(shapes[n]),
                         what: t.kind === "head" ? "output head" : shapes[n][1] === d ? "token embedding" : "per-layer embedding (PLE)"}));
    const rowCost = tables.reduce((a, x) => a + x.width, 0), vocabParams = tables.reduce((a, x) => a + x.params, 0);
    const levers = [], pct = x => x / total;
    // 1. vocabulary: every row costs the width of every table indexed by token id
    if (V && vocabParams && (V >= 32000 || vocabParams / total >= 0.05)) {
      const keep = opt.keep_tokens || null;
      const cut = keep ? Math.max(V - keep, 0) * rowCost : null;
      levers.push({id: "vocab", title: "Trim the vocabulary to the languages you use", params: vocabParams, share: pct(vocabParams),
        saves: cut, saves_share: cut != null ? pct(cut) : null, row_cost: rowCost, vocab: V, keep_tokens: keep,
        detail: `${V.toLocaleString("en-US")} tokens × ${rowCost.toLocaleString("en-US")} values per token (` + tables.map(x => `${x.what} ${x.width}`).join(" + ") +
                `). Every token you drop removes ${rowCost.toLocaleString("en-US")} weights.` + (keep ? "" : " Count the tokens your languages need to see the saving."),
        how: "Count tokenizer pieces by script, keep your scripts plus specials, bytes, digits and punctuation, remap the ids and slice every table above. Retest: rare words fall back to bytes."});
    }
    // 2. per-layer embeddings: a lookup, not compute
    const ple = tables.filter(x => x.what.startsWith("per-layer"));
    if (ple.length) {
      const p = ple.reduce((a, x) => a + x.params, 0);
      levers.push({id: "ple", title: "Per-layer embeddings: keep them out of fast memory", params: p, share: pct(p), saves: null,
        detail: `${ple.map(x => x.name).join(", ")} holds ${(p / 1e9).toFixed(2)}B weights (${(100 * pct(p)).toFixed(0)}% of the model), but each token reads only one row of ${ple[0].width} values.`,
        how: "Memory-map or offload the table to flash/CPU and fetch rows per token, quantize it to 4-bit (a lookup tolerates it well), and trim it together with the vocabulary."});
    }
    // 3. modality towers you may not need
    const edges = I.edges || [], fedBy = id => edges.filter(e => e.to === id).map(e => e.from);
    const isAdapterOf = (a, enc) => a.role === "adapter" && (a.id.startsWith(enc.id + ".") || fedBy(a.id).includes(enc.id));
    // an encoder read through cross-attention (Whisper, T5) is the model's input path, not an optional tower
    const crossFed = new Set(edges.filter(e => /cross/.test(e.via || e.label || e.kind || "")).map(e => e.from));
    for (const x of comps.filter(x => (x.role === "encoder" || x.role === "adapter") && x.modality !== "text" && !crossFed.has(x.id) && I.family !== "encoder-decoder")) {
      if (x.role === "adapter" && comps.some(y => y.role === "encoder" && isAdapterOf(x, y))) continue;
      const kids = comps.filter(y => y !== x && (y.id.startsWith(x.id + ".") || isAdapterOf(y, x))), p = (x.params || 0) + kids.reduce((a, y) => a + (y.params || 0), 0);
      const kind = /vision|visual|image|siglip|vit/i.test(x.id) ? "vision" : /audio|speech|whisper|conformer/i.test(x.id) ? "audio" : x.modality;
      levers.push({id: "tower:" + x.id, title: `Drop the ${kind} tower if you only send text`, params: p, share: pct(p), saves: p, saves_share: pct(p),
        detail: `${x.id}${kids.length ? " with " + kids.map(y => y.id).join(", ") : ""}: ${(p / 1e6).toFixed(1)}M weights.`,
        how: `Export only the language model (and the ${kind} tower when a request needs it); the text path never reads these weights.`});
    }
    // 4. KV cache, layer by layer
    const kv = kvProfile(I, shapes, cfg);
    if (kv && kv.unused_kv_params) levers.push({id: "kv_shared", title: "K/V projections that shared-KV layers do not use", params: kv.unused_kv_params,
      share: pct(kv.unused_kv_params), saves: kv.unused_kv_params, saves_share: pct(kv.unused_kv_params),
      detail: `${kv.shared} layers reuse the KV cache of earlier layers (num_kv_shared_layers), yet the file still stores their k/v projections.`,
      how: "Check that your runtime skips them, then leave them out of the export."});
    // 5. where the compute weights sit
    const ffn = opt.ffn_params != null ? opt.ffn_params : T.filter(([n, t]) => t.comp === (back && back.id) && t.kind === "ffn").reduce((a, [n]) => a + prod(shapes[n]), 0);
    if (ffn) levers.push({id: "ffn", title: "Quantize the feed-forward blocks first", params: ffn, share: pct(ffn), saves: ffn / 2, saves_share: pct(ffn / 2),
      detail: `${(ffn / 1e9).toFixed(2)}B feed-forward weights in the backbone; 8-bit halves them versus 16-bit, 4-bit quarters them.`,
      how: "Weight-only int8/int4 on the FFN with per-channel or group scales; keep norms, router and the embedding row lookups at higher precision until measured."});
    levers.sort((a, b) => (b.saves || b.params) - (a.saves || a.params));
    return {total, vocab: V ? {size: V, row_cost: rowCost, params: vocabParams, tables} : null, kv, levers};
  }

  // ---------------------------------------------------------------- tokenizer pieces by script
  const SCRIPTS = [["Thai", /\p{Script=Thai}/u], ["Han", /\p{Script=Han}/u], ["Hiragana", /\p{Script=Hiragana}/u], ["Katakana", /\p{Script=Katakana}/u],
    ["Hangul", /\p{Script=Hangul}/u], ["Latin", /\p{Script=Latin}/u], ["Cyrillic", /\p{Script=Cyrillic}/u], ["Arabic", /\p{Script=Arabic}/u],
    ["Devanagari", /\p{Script=Devanagari}/u], ["Greek", /\p{Script=Greek}/u], ["Hebrew", /\p{Script=Hebrew}/u], ["Khmer", /\p{Script=Khmer}/u],
    ["Lao", /\p{Script=Lao}/u], ["Myanmar", /\p{Script=Myanmar}/u], ["Bengali", /\p{Script=Bengali}/u], ["Tamil", /\p{Script=Tamil}/u]];
  const LX = ["Latin-ASCII", "Latin-ext"];   // English needs only unaccented Latin letters; other Latin-script languages need both
  const LANG = {en: ["Latin-ASCII"], th: ["Thai"], zh: ["Han"], cn: ["Han"], ja: ["Han", "Hiragana", "Katakana"], ko: ["Hangul"], vi: LX, id: LX, ms: LX,
    fr: LX, de: LX, es: LX, pt: LX, it: LX, nl: LX, tl: LX, ru: ["Cyrillic"], ar: ["Arabic"], hi: ["Devanagari"], km: ["Khmer"], lo: ["Lao"], my: ["Myanmar"]};
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
      if (/[\p{N}\p{P}\p{S}\p{Z}\p{M}]/u.test(ch)) continue;
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
    return {total: pieces.length, counts, keep, keep_for: langs => keep(langs)};
  }
  return {plan, kvProfile, vocabScripts, scriptOf, LANG};
})();
if (typeof module !== "undefined") module.exports = Plan;
