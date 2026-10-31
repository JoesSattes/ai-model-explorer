// loaders.js — read only what describes a model: safetensors headers and PyTorch checkpoint indexes, via HTTP Range.
// Weights are never downloaded. (c) 2026 Sattaya Singkul · MIT
// Method: S. Singkul, "Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy", 2026 (paper/).
"use strict";
(function (root) {
const rangeGet = async (url, a, b) => { const r = await fetch(url, {headers: {Range: `bytes=${a}-${b}`}}); if (r.status === 401 || r.status === 403) throw new Error("gated repository: accept its license on Hugging Face first"); if (!r.ok && r.status !== 206) throw new Error(`HTTP ${r.status}`); return new Uint8Array(await r.arrayBuffer()); };
const utf8 = new TextDecoder();

// ---------------- safetensors: 8-byte little-endian length, then the JSON header
async function safetensorsHeader(url, size) {
  const h = await rangeGet(url, 0, 7), n = Number(new DataView(h.buffer, h.byteOffset, 8).getBigUint64(0, true));
  if (!(n > 0 && n < 200e6)) throw new Error("not a safetensors file");
  const j = JSON.parse(utf8.decode(await rangeGet(url, 8, 8 + n - 1))); const meta = j.__metadata__; delete j.__metadata__;
  return {tensors: j, bytes: 8 + n, meta};
}

// ---------------- PyTorch zip checkpoints (.pt / .pth / .bin): read the zip directory from the end, then data.pkl only
async function torchIndex(url, size) {
  const tail = Math.min(size, 65536 + 22), buf = await rangeGet(url, size - tail, size - 1), dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let eocd = -1; for (let i = buf.length - 22; i >= 0; i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error("not a zip-format PyTorch file (old pickle format is not supported)");
  let cdSize = dv.getUint32(eocd + 12, true), cdOff = dv.getUint32(eocd + 16, true), count = dv.getUint16(eocd + 10, true);
  if (cdOff === 0xffffffff || count === 0xffff) { // zip64
    const loc = eocd - 20; const z64 = Number(dv.getBigUint64(loc + 8, true)); const r = await rangeGet(url, z64, z64 + 55), d2 = new DataView(r.buffer, r.byteOffset, r.byteLength);
    count = Number(d2.getBigUint64(32, true)); cdSize = Number(d2.getBigUint64(40, true)); cdOff = Number(d2.getBigUint64(48, true)); }
  const cd = cdOff >= size - tail ? buf.subarray(cdOff - (size - tail), cdOff - (size - tail) + cdSize) : await rangeGet(url, cdOff, cdOff + cdSize - 1), c = new DataView(cd.buffer, cd.byteOffset, cd.byteLength);
  let p = 0, pkl = null, ents = 0; const sizes = {};
  while (p + 46 <= cd.length && ents < count + 5) { if (c.getUint32(p, true) !== 0x02014b50) break; ents++;
    const method = c.getUint16(p + 10, true); let csize = c.getUint32(p + 20, true), usize = c.getUint32(p + 24, true); const nl = c.getUint16(p + 28, true), xl = c.getUint16(p + 30, true), cl = c.getUint16(p + 32, true); let off = c.getUint32(p + 42, true);
    const name = utf8.decode(cd.subarray(p + 46, p + 46 + nl));
    if (csize === 0xffffffff || usize === 0xffffffff || off === 0xffffffff) { let q = p + 46 + nl; const end = q + xl; while (q < end) { const id = c.getUint16(q, true), ln = c.getUint16(q + 2, true); if (id === 1) { let r = q + 4; if (usize === 0xffffffff) { usize = Number(c.getBigUint64(r, true)); r += 8; } if (csize === 0xffffffff) { csize = Number(c.getBigUint64(r, true)); r += 8; } if (off === 0xffffffff) { off = Number(c.getBigUint64(r, true)); } } q += 4 + ln; } }
    if (/(^|\/)data\.pkl$/.test(name)) pkl = {off, csize, method, name};
    const m = name.match(/\/data\/(\d+)$/); if (m) sizes[m[1]] = usize;
    p += 46 + nl + xl + cl; }
  if (!pkl) throw new Error("no data.pkl in the checkpoint");
  if (pkl.method !== 0) throw new Error("compressed data.pkl is not supported");
  const lh = await rangeGet(url, pkl.off, pkl.off + 29), ld = new DataView(lh.buffer, lh.byteOffset, 30), start = pkl.off + 30 + ld.getUint16(26, true) + ld.getUint16(28, true);
  const data = await rangeGet(url, start, start + pkl.csize - 1);
  const obj = unpickle(data);
  return {tensors: tensorsFromState(obj), bytes: tail + cdSize + pkl.csize + 30};
}

// a minimal unpickler for torch.save output: tensors become {dtype, shape}; everything else stays plain data
const ST = {FloatStorage: "F32", HalfStorage: "F16", BFloat16Storage: "BF16", DoubleStorage: "F64", LongStorage: "I64", IntStorage: "I32", ShortStorage: "I16", CharStorage: "I8", ByteStorage: "U8", BoolStorage: "BOOL", Float8_e4m3fnStorage: "F8_E4M3", Float8_e5m2Storage: "F8_E5M2"};
const TD = {float32: "F32", float16: "F16", bfloat16: "BF16", float64: "F64", int64: "I64", int32: "I32", int16: "I16", int8: "I8", uint8: "U8", bool: "BOOL", float8_e4m3fn: "F8_E4M3", float8_e5m2: "F8_E5M2"};
function unpickle(b) {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength); let i = 0; const stack = [], marks = [], memo = {};
  const MARK = {mark: 1}, pop = () => stack.pop(), popMark = () => { const m = marks.pop(); return stack.splice(m); };
  const glob = (mod, name) => ({__global: mod + "." + name});
  const call = (f, args) => {
    const g = f && f.__global || "";
    if (/_rebuild_tensor/.test(g)) { const st = args[0] || {}; return {__tensor: true, dtype: st.dtype || "F32", shape: (args[2] || []).slice()}; }
    if (/_rebuild_parameter/.test(g)) return args[0];
    if (/OrderedDict|^builtins\.dict$/.test(g)) return {};
    if (/torch\.Size/.test(g)) return args[0];
    if (/_codecs\.encode/.test(g)) return args[0];
    return {__obj: g, args};
  };
  while (i < b.length) {
    const op = b[i++];
    switch (op) {
      case 0x80: i++; break; // PROTO
      case 0x95: i += 8; break; // FRAME
      case 0x7d: stack.push({}); break; // EMPTY_DICT
      case 0x5d: stack.push([]); break; // EMPTY_LIST
      case 0x29: stack.push([]); break; // EMPTY_TUPLE
      case 0x28: marks.push(stack.length); break; // MARK
      case 0x58: { const n = dv.getUint32(i, true); i += 4; stack.push(utf8.decode(b.subarray(i, i + n))); i += n; break; } // BINUNICODE
      case 0x8c: { const n = b[i++]; stack.push(utf8.decode(b.subarray(i, i + n))); i += n; break; } // SHORT_BINUNICODE
      case 0x8d: { const n = Number(dv.getBigUint64(i, true)); i += 8; stack.push(utf8.decode(b.subarray(i, i + n))); i += n; break; }
      case 0x42: { const n = dv.getUint32(i, true); i += 4; stack.push(b.subarray(i, i + n)); i += n; break; } // BINBYTES
      case 0x43: { const n = b[i++]; stack.push(b.subarray(i, i + n)); i += n; break; } // SHORT_BINBYTES
      case 0x71: memo[b[i++]] = stack[stack.length - 1]; break; // BINPUT
      case 0x72: memo[dv.getUint32(i, true)] = stack[stack.length - 1]; i += 4; break; // LONG_BINPUT
      case 0x94: memo[Object.keys(memo).length] = stack[stack.length - 1]; break; // MEMOIZE
      case 0x68: stack.push(memo[b[i++]]); break; // BINGET
      case 0x6a: stack.push(memo[dv.getUint32(i, true)]); i += 4; break; // LONG_BINGET
      case 0x63: { let e = b.indexOf(10, i); const mod = utf8.decode(b.subarray(i, e)); i = e + 1; e = b.indexOf(10, i); const nm = utf8.decode(b.subarray(i, e)); i = e + 1; stack.push(glob(mod, nm)); break; } // GLOBAL
      case 0x93: { const nm = pop(), mod = pop(); stack.push(glob(mod, nm)); break; } // STACK_GLOBAL
      case 0x4a: stack.push(dv.getInt32(i, true)); i += 4; break; // BININT
      case 0x4b: stack.push(b[i++]); break; // BININT1
      case 0x4d: stack.push(dv.getUint16(i, true)); i += 2; break; // BININT2
      case 0x8a: { const n = b[i++]; let v = 0; for (let k = n - 1; k >= 0; k--) v = v * 256 + b[i + k]; if (n && b[i + n - 1] & 0x80) v -= Math.pow(256, n); i += n; stack.push(v); break; } // LONG1
      case 0x47: stack.push(dv.getFloat64(i, false)); i += 8; break; // BINFLOAT
      case 0x4e: stack.push(null); break; case 0x88: stack.push(true); break; case 0x89: stack.push(false); break;
      case 0x74: stack.push(popMark()); break; // TUPLE
      case 0x85: stack.push([pop()]); break; case 0x86: { const b2 = pop(), a = pop(); stack.push([a, b2]); break; } case 0x87: { const c2 = pop(), b2 = pop(), a = pop(); stack.push([a, b2, c2]); break; }
      case 0x6c: stack.push(popMark()); break; // LIST
      case 0x64: { const it = popMark(), d = {}; for (let k = 0; k < it.length; k += 2) d[it[k]] = it[k + 1]; stack.push(d); break; } // DICT
      case 0x73: { const v = pop(), k = pop(); stack[stack.length - 1][k] = v; break; } // SETITEM
      case 0x75: { const it = popMark(), d = stack[stack.length - 1]; for (let k = 0; k < it.length; k += 2) d[it[k]] = it[k + 1]; break; } // SETITEMS
      case 0x61: { const v = pop(); stack[stack.length - 1].push(v); break; } // APPEND
      case 0x65: { const it = popMark(); stack[stack.length - 1].push(...it); break; } // APPENDS
      case 0x52: { const args = pop(), f = pop(); stack.push(call(f, args)); break; } // REDUCE
      case 0x81: { const args = pop(), cls = pop(); stack.push(call(cls, args)); break; } // NEWOBJ
      case 0x62: { const st = pop(), o = stack[stack.length - 1]; if (o && typeof o === "object" && st && typeof st === "object" && !Array.isArray(st)) Object.assign(o, st); break; } // BUILD
      case 0x51: { const pid = pop(); // BINPERSID: ('storage', <StorageType or dtype>, key, location, numel)
        const kind = pid[1] && pid[1].__global || ""; const nm = kind.split(".").pop(); stack.push({dtype: ST[nm] || TD[nm] || "F32", key: pid[2]}); break; }
      case 0x2e: return stack.pop(); // STOP
      case 0x30: pop(); break; case 0x32: stack.push(stack[stack.length - 1]); break; // POP, DUP
      case 0x31: popMark(); break; // POP_MARK
      default: throw new Error("unsupported pickle opcode 0x" + op.toString(16));
    }
  }
  return stack.pop();
}
// flatten nested dicts down to the tensors (checkpoints often wrap the state dict: {"state_dict": {...}, "config": {...}})
function tensorsFromState(o, pre = "", out = {}, depth = 0) {
  if (!o || typeof o !== "object" || depth > 6) return out;
  for (const [k, v] of Object.entries(o)) {
    const n = pre ? pre + "." + k : k;
    if (v && v.__tensor) out[n] = {dtype: v.dtype, shape: v.shape};
    else if (v && typeof v === "object" && !(v instanceof Uint8Array)) tensorsFromState(v, n, out, depth + 1);
  }
  return out;
}
// strip a wrapper prefix every tensor shares (e.g. "state_dict." or "model.")
function stripCommon(T) { const ks = Object.keys(T); if (ks.length < 2) return T; const f = ks[0].split(".")[0]; if (!ks.every(k => k.startsWith(f + ".")) || !/^(state_dict|model_state_dict|module|ema|params|weights)$/.test(f)) return T; const o = {}; for (const k of ks) o[k.slice(f.length + 1)] = T[k]; return o; }

// ---------------- a Hugging Face repo: which files describe the weights, then their headers (shared by the explorer and the benchmark)
async function hfManifest(repo, files, prog = () => {}, opt = {}) {
  const conc = opt.conc || 6;
  const base = `https://huggingface.co/${repo}/resolve/main/`;
  const folder = n => n.includes("/") ? n.slice(0, n.lastIndexOf("/")) : "";
  // skip optimizer/training files, per-language adapters (MMS ships 1,000+ of them) and alternate-precision duplicates
  const skip = n => /(^|\/)(consolidated|optimizer|training_args|scheduler|rng_state|adapter[._])/i.test(n) || /^(\.|original\/)/.test(n);
  const getJSON = async u => { const r = await fetch(u); const t = await r.text();
    if (!r.ok) throw new Error(r.status === 401 || r.status === 403 || /^Access to/.test(t) ? "gated repository: accept its license on Hugging Face first" : `HTTP ${r.status}`);
    try { return JSON.parse(t); } catch (e) { throw new Error(/^Access to/.test(t) ? "gated repository: accept its license on Hugging Face first" : "unreadable index file"); } };
  // a diffusers repo (model_index.json) is described by its component folders; single-file copies and LoRAs at the root are skipped
  if (opt.only && opt.only.length) files = files.filter(f => opt.only.includes(folder(f.n)));
  else files = files.filter(f => !/(^|\/)[^/]*lora[^/]*\.safetensors$/i.test(f.n) || files.every(g => /lora/i.test(g.n) || !/\.safetensors$/.test(g.n)));
  const jobs = [], seen = new Set();
  for (const ix of files.filter(f => /\.safetensors\.index\.json$/.test(f.n) && !skip(f.n) && !/\.(fp16|bf16|fp32)\.safetensors\.index\.json$/.test(f.n) || /\.(fp16|bf16)\.safetensors\.index\.json$/.test(f.n) && !files.some(g => g.n === f.n.replace(/\.(fp16|bf16)\.safetensors/, ".safetensors")))) {
    prog("index", ix.n); const j = await getJSON(base + ix.n), dir = folder(ix.n);
    for (const sh of [...new Set(Object.values(j.weight_map || {}))]) { const n = dir ? dir + "/" + sh : sh; seen.add(n); jobs.push({n, prefix: dir, kind: "st"}); } }
  let loose = files.filter(f => /\.safetensors$/.test(f.n) && !seen.has(f.n) && !skip(f.n));
  // a folder with both x.safetensors and x.fp16.safetensors describes the same weights twice: keep one
  loose = loose.filter(f => !(/\.(fp16|bf16|fp32)\.safetensors$/.test(f.n) && loose.some(g => g !== f && g.n === f.n.replace(/\.(fp16|bf16|fp32)\.safetensors$/, ".safetensors"))));
  { const byDir = {}; loose.forEach(f => (byDir[folder(f.n)] = byDir[folder(f.n)] || []).push(f)); // never read more than 32 loose files per folder
    loose = Object.values(byDir).flatMap(g => g.length > 32 ? g.sort((a, b) => b.size - a.size).slice(0, 4) : g); }
  const rootLoose = loose.filter(f => !folder(f.n));
  loose.forEach(f => jobs.push({n: f.n, prefix: folder(f.n) || (rootLoose.length > 1 ? f.n.replace(/\.safetensors$/, "").replace(/[^\w]+/g, "_") : ""), kind: "st"}));
  if (!jobs.length) {
    const pts = files.filter(f => /\.(pt|pth|bin|ckpt)$/.test(f.n) && !skip(f.n) && !/optimizer|scheduler|training/.test(f.n));
    const groups = {}; pts.forEach(f => (groups[folder(f.n)] = groups[folder(f.n)] || []).push(f));
    let dirs = Object.keys(groups); if (opt.prefer && groups[opt.prefer]) dirs = [opt.prefer]; else if (dirs.length > 1) dirs = dirs.sort((a, b) => groups[b].reduce((x, f) => x + f.size, 0) - groups[a].reduce((x, f) => x + f.size, 0)).slice(0, 4);
    dirs.forEach(d => { const g = groups[d].sort((a, b) => (/classif|model|pytorch_model/.test(b.n) - /classif|model|pytorch_model/.test(a.n)) || b.size - a.size); jobs.push({n: g[0].n, prefix: dirs.length > 1 ? d.replace(/\//g, ".") : "", kind: "pt"}); });
  }
  if (!jobs.length) return null;
  const tensors = {}, res = new Array(jobs.length); let read = 0, done = 0, next = 0;
  const work = async () => { while (next < jobs.length) { const k = next++, jb = jobs[k], sz = (files.find(f => f.n === jb.n) || {}).size || 0;
    const r = jb.kind === "pt" ? await torchIndex(base + jb.n, sz) : await safetensorsHeader(base + jb.n, sz);
    res[k] = r; read += r.bytes; done++; prog(jb.kind === "pt" ? "pt" : "header", jb.n, done, jobs.length, read); } };
  await Promise.all(Array.from({length: Math.min(conc, jobs.length)}, work));
  jobs.forEach((jb, k) => { const T = jb.kind === "pt" ? stripCommon(res[k].tensors) : res[k].tensors;
    for (const [n, v] of Object.entries(T)) tensors[(jb.prefix ? jb.prefix.replace(/\//g, ".") + "." : "") + n] = v; });
  return {tensors, read, nfiles: jobs.length, kinds: [...new Set(jobs.map(j => j.kind))]};
}

root.Loaders = {hfManifest, safetensorsHeader, torchIndex, unpickle, tensorsFromState, stripCommon};
})(typeof window !== "undefined" ? window : globalThis);
