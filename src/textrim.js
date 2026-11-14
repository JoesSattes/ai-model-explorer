// Trim the vocabulary with your own text, in the browser: a BPE encoder for tokenizer.json, tiktoken and Tekken files,
// the corpus keep rule with merge closure, and the trim certificate (Good-Turing estimate of how much new text the
// trimmed tokenizer splits differently, and how many lines would bring it under a target). Same rules as the Python
// package (Model.trim_vocab / vocab_tax with corpus=...). SentencePiece .model and GGUF files: use the package.
const TextTrim = (() => {
  // GPT-2 byte <-> unicode table used by byte-level BPE
  const B2U = (() => { const bs = []; for (let i = 33; i <= 126; i++) bs.push(i); for (let i = 161; i <= 172; i++) bs.push(i); for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice(); let n = 0; for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    const m = {}; bs.forEach((b, i) => { m[b] = String.fromCharCode(cs[i]); }); return m; })();
  const enc8 = new TextEncoder();
  const byteStr = s => { let o = ""; for (const b of enc8.encode(s)) o += B2U[b]; return o; };
  const GPT2_RE = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
  const CL100K_RE = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  const O200K_RE = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  // Rust/oniguruma regex -> JavaScript: drop possessive quantifiers, expand (?i:...) into case classes
  function jsRegex(p) {
    p = p.replace(/(\{\d+(?:,\d*)?\}|[?*+])\+/g, "$1");
    p = p.replace(/\(\?i:((?:[^()\\]|\\.)*)\)/g, (_, body) => "(?:" + body.replace(/(\\.|\[[^\]]*\])|([A-Za-z])/g, (m, esc, ch) => esc ? esc : `[${ch.toLowerCase()}${ch.toUpperCase()}]`) + ")");
    return new RegExp(p, "gu");
  }
  const split = (re, s) => { const out = []; let last = 0, m; re.lastIndex = 0;
    while ((m = re.exec(s)) !== null) { if (m.index > last) out.push(s.slice(last, m.index)); if (m[0].length) out.push(m[0]); else re.lastIndex++; last = m.index + m[0].length; }
    if (last < s.length) out.push(s.slice(last)); return out; };

  // ---- tokenizer.json (Hugging Face BPE)
  function fromHF(j) {
    const M = j.model || {}; if (M.type !== "BPE" || !M.vocab || !M.merges) return null;
    const vocab = new Map(Object.entries(M.vocab)), ranks = new Map(), parents = new Map();
    M.merges.forEach((x, r) => { const [a, b] = Array.isArray(x) ? x : x.split(" "); ranks.set(a + "\u0000" + b, r);
      const p = vocab.get(a + b); if (p != null && !parents.has(p) && vocab.has(a) && vocab.has(b)) parents.set(p, [vocab.get(a), vocab.get(b)]); });
    let byteLevel = false, prefixSpace = false; const steps = [];
    const norm = []; (function walkN(n) { if (!n) return; if (n.type === "Sequence") return (n.normalizers || []).forEach(walkN); norm.push(n); })(j.normalizer);
    (function walkP(n) { if (!n) return; if (n.type === "Sequence") return (n.pretokenizers || []).forEach(walkP);
      if (n.type === "ByteLevel") { byteLevel = true; if (n.add_prefix_space) prefixSpace = true; if (n.use_regex !== false) steps.push({re: jsRegex(GPT2_RE)}); }
      else if (n.type === "Split") { const pat = n.pattern || {}; steps.push(pat.Regex != null ? {re: jsRegex(pat.Regex), behavior: n.behavior} : {str: pat.String, behavior: n.behavior}); }
      else if (n.type === "Metaspace") steps.push({meta: n.replacement || "▁", prepend: n.prepend_scheme || (n.add_prefix_space ? "always" : "never"), splitOn: n.split !== false});
      else if (n.type === "Digits") steps.push({digits: !!n.individual_digits});
    })(j.pre_tokenizer);
    if (/ByteLevel/.test(JSON.stringify(j.decoder || {}))) byteLevel = true;
    const normalize = s => { for (const n of norm) {
        if (n.type === "Replace") { const p = n.pattern || {}; s = p.String != null ? s.split(p.String).join(n.content) : s.replace(jsRegex(p.Regex), n.content); }
        else if (n.type === "Prepend") s = n.prepend + s; else if (/^NF/.test(n.type)) s = s.normalize(n.type); else if (n.type === "Lowercase") s = s.toLowerCase(); }
      return s; };
    const pretok = s => { let parts = [s];
      for (const st of steps) { const next = [];
        for (const w of parts) {
          if (st.re) next.push(...split(st.re, w));
          else if (st.str != null) { const xs = w.split(st.str); xs.forEach((x, i) => { if (st.behavior === "MergedWithPrevious") { if (i < xs.length - 1) next.push(x + st.str); else if (x) next.push(x); } else if (st.behavior === "MergedWithNext") { next.push((i ? st.str : "") + x); } else { if (x) next.push(x); if (i < xs.length - 1 && st.behavior !== "Removed") next.push(st.str); } }); }
          else if (st.meta) { let x = w.split(" ").join(st.meta); if (st.prepend === "always" && !x.startsWith(st.meta)) x = st.meta + x; if (st.splitOn) next.push(...x.split(new RegExp(`(?=${st.meta})`, "u")).filter(Boolean)); else next.push(x); }
          else if (st.digits != null) next.push(...split(st.digits ? /\p{N}/gu : /\p{N}+/gu, w));
          else next.push(w);
        } parts = next; }
      return parts; };
    return {kind: "hf", vocab, ranks, parents, byteLevel, prefixSpace, normalize, pretok, ignoreMerges: !!M.ignore_merges, byteFallback: !!M.byte_fallback, unk: M.unk_token};
  }
  // ---- tiktoken / Tekken (ranks over bytes; merges are implied by the ranks)
  function fromRanks(rankBytes, pattern, offset) {   // rankBytes: Array of Uint8Array indexed by rank
    const vocab = new Map(); rankBytes.forEach((b, r) => { if (b) { let s = ""; for (const x of b) s += B2U[x]; vocab.set(s, r + offset); } });
    return {kind: "ranks", vocab, byteLevel: true, normalize: s => s, pretok: s => split(jsRegex(pattern), s), offset, rankOf: s => { const v = vocab.get(s); return v == null ? null : v - offset; } };
  }
  function bpeWord(T, w) {
    if (T.ignoreMerges && T.vocab.has(w)) return [w];
    let sym = Array.from(w);
    const rank = T.kind === "hf" ? (a, b) => T.ranks.get(a + "\u0000" + b) : (a, b) => T.rankOf(a + b);
    while (sym.length > 1) { let best = null, bi = -1;
      for (let i = 0; i < sym.length - 1; i++) { const r = rank(sym[i], sym[i + 1]); if (r != null && (best == null || r < best)) { best = r; bi = i; } }
      if (bi < 0) break; sym.splice(bi, 2, sym[bi] + sym[bi + 1]); }
    return sym;
  }
  function encode(T, text) {
    const ids = []; let s = T.normalize(text);
    if (T.prefixSpace && !s.startsWith(" ")) s = " " + s;
    for (const piece of T.pretok(s)) {
      const w = T.byteLevel ? byteStr(piece) : piece;
      for (const x of bpeWord(T, w)) { const id = T.vocab.get(x);
        if (id != null) ids.push(id);
        else if (T.byteFallback) { for (const b of enc8.encode(x)) { const k = T.vocab.get("<0x" + b.toString(16).toUpperCase().padStart(2, "0") + ">"); if (k != null) ids.push(k); } }
        else if (T.unk != null && T.vocab.has(T.unk)) ids.push(T.vocab.get(T.unk)); } }
    return ids;
  }
  // parents for rank-based vocabularies: the last merge BPE performs when it encodes the piece on its own
  function parentsOf(T) {
    if (T.parents) return T.parents;
    const P = new Map();
    for (const [s, id] of T.vocab) { const r = id - T.offset; const ch = Array.from(s); if (ch.length < 2) continue;
      let sym = ch, last = null;
      while (sym.length > 1) { let best = null, bi = -1;
        for (let i = 0; i < sym.length - 1; i++) { const q = T.rankOf(sym[i] + sym[i + 1]); if (q != null && q <= r && (best == null || q < best)) { best = q; bi = i; } }
        if (bi < 0) break; last = [sym[bi], sym[bi + 1]]; sym.splice(bi, 2, sym[bi] + sym[bi + 1]); }
      if (sym.length === 1 && last && T.vocab.has(last[0]) && T.vocab.has(last[1])) P.set(id, [T.vocab.get(last[0]), T.vocab.get(last[1])]); }
    return (T.parents = P);
  }
  // Load from a tokenizer file's bytes. Returns null when the format has no in-browser encoder here.
  function load(name, buf) {
    const n = name.toLowerCase(), dec = new TextDecoder("utf-8"), u8 = new Uint8Array(buf);
    const b64 = s => { try { return Uint8Array.from(atob(s), c => c.charCodeAt(0)); } catch (e) { return null; } };
    if (n.endsWith(".tiktoken") || n === "tiktoken.model") { const rb = []; dec.decode(u8).split("\n").forEach(l => { const [a, r] = l.trim().split(/\s+/); if (a && r != null) rb[+r] = b64(a); });
      return fromRanks(rb, rb.length > 150000 ? O200K_RE : CL100K_RE, 0); }
    if (!n.endsWith(".json")) return null;
    const j = JSON.parse(dec.decode(u8));
    if (j.config && Array.isArray(j.vocab)) { const ns = j.config.default_num_special_tokens || 0, size = j.config.default_vocab_size || ns + j.vocab.length;
      return fromRanks(j.vocab.slice(0, size - ns).map(x => b64(x.token_bytes)), j.config.pattern || O200K_RE, ns); }
    if (j.model) return fromHF(j);
    return null;
  }
  // Corpus rule + merge closure + certificate. scriptKeep: '1'/'0' string or boolean array over pieces;
  // isAlways(i): specials, bytes, single characters (kept regardless).
  function corpusTrim(T, lines, scriptKeep, isAlways, opt = {}) {
    const m = opt.minCount || 1, tau = opt.target || 0.01, V = scriptKeep.length, sk = i => scriptKeep[i] === "1" || scriptKeep[i] === true;
    const encs = lines.map(l => encode(T, l));
    const est = arr => { const cnt = new Map(); let N = 0; for (const e of arr) for (const i of e) { cnt.set(i, (cnt.get(i) || 0) + 1); N++; }
      const nc = new Map(); for (const [i, c] of cnt) if (c <= m && i < V && sk(i) && !isAlways(i)) nc.set(c, (nc.get(c) || 0) + 1);
      let r = 0; for (let c = 0; c < m; c++) r += (c + 1) * (nc.get(c + 1) || 0); return {cnt, r: N ? r / N : 0}; };
    const full = est(encs), keep = new Uint8Array(V); let kept0 = 0;
    for (let i = 0; i < V; i++) if (sk(i) && (isAlways(i) || (full.cnt.get(i) || 0) >= m)) { keep[i] = 1; kept0++; }
    const P = parentsOf(T), st = []; let added = 0; for (let i = 0; i < V; i++) if (keep[i]) st.push(i);
    while (st.length) { const i = st.pop(), p = P.get(i); if (!p) continue; for (const j of p) if (j < V && !keep[j]) { keep[j] = 1; added++; st.push(j); } }
    let need = null; const n = lines.length;
    if (n >= 200) { const half = est(encs.slice(0, n >> 1)).r;
      if (full.r <= tau) need = n; else if (half > full.r && full.r > 0) need = Math.ceil(n * Math.pow(full.r / tau, 1 / (Math.log(half / full.r) / Math.log(n / (n >> 1))))); }
    return {kept: kept0 + added, keptSeen: kept0, closureAdded: added, predicted: full.r, target: tau, linesNeeded: need, ok: full.r <= tau, lines: n, tokens: encs.reduce((a, e) => a + e.length, 0), keep};
  }
  const loadJSON = j => (j && j.config && Array.isArray(j.vocab)) ? null : (j && j.model ? fromHF(j) : null);
  return {load, loadJSON, encode, corpusTrim, parentsOf, jsRegex};
})();
if (typeof module !== "undefined") module.exports = TextTrim;
