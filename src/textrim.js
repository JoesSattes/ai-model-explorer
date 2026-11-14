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
  // llama.cpp pre-tokenizer table (tokenizer.ggml.pre -> regex chain, ignore_merges), from src/llama-vocab.cpp
  const GGUF_PRE = {"flint":{"re":["\\p{N}{1,3}","[一-龥぀-ゟ゠-ヿ]+","[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"default":{"re":["[\\p{P}\\$\\+<=>\\^~\\|]+","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)","\\p{N}+","[0-9][0-9][0-9]"],"im":false},"minicpm5":{"re":["\\p{N}{1,3}","(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"llama3":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"llama-v3":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"llama-bpe":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"falcon3":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"falcon-h1":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"midm-2.0":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"jina-v5-nano":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"lfm2":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"deepseek-llm":{"re":["[\r\n]","\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+","\\s?[!-/:-~！-／：-～‘-‟　-。]+","\\s+$","[一-龥ࠀ-一가-퟿]+","\\p{N}+"],"im":false},"deepseek-coder":{"re":["[\r\n]","\\s?\\p{L}+","\\s?\\p{P}+","[一-龥ࠀ-一가-퟿]+","\\p{N}"],"im":false},"deepseek-v3":{"re":["\\p{N}{1,3}","[一-龥぀-ゟ゠-ヿ]+","[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"],"im":false},"spark2_5":{"re":["\\p{N}{1,3}","[一-龥぀-ゟ゠-ヿ]+","[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+","\\p{N}"],"im":false},"youtu":{"re":["[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+","[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"falcon":{"re":["[\\p{P}\\$\\+<=>\\^~\\|`]+","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)","[0-9][0-9][0-9]"],"im":false},"mpt":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"starcoder":{"re":["\\p{N}","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"gpt-2":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"phi-2":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"jina-es":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"jina-de":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"gigachat":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"jina-v2-es":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"jina-v2-de":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"a.x-4.0":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"mellum":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"modern-bert":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"jais-2":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}"],"im":false},"gemma4":{"re":["[^\\n]+|[\\n]+"],"im":false},"granite-embed-multi-311m":{"re":["[^\\n]+|[\\n]+"],"im":false},"mmbert":{"re":["[^\\n]+|[\\n]+"],"im":false},"sarvam-moe":{"re":["[^\\n]+|[\\n]+"],"im":false},"jina-v1-en":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"jina-v2-code":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"roberta-bpe":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"whitespace":{"re":["\\S+"],"im":false},"refact":{"re":["\\p{N}","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"command-r":{"re":["\\p{N}","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"qwen2":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"deepseek-r1-qwen":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"kormo":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"f2llmv2":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"qwen35":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"stablelm2":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"olmo":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"dbrx":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"smaug-bpe":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"poro-chat":{"re":[" ?[^(\\s|.,!?…。，、।۔،)]+"],"im":false},"chatglm-bpe":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"glm4":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"glm5":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"viking":{"re":[" ?[^(\\s|.,!?…。，、।۔،)]+","\\p{N}"],"im":false},"jais":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"tekken":{"re":["[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"pixtral":{"re":["[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"tekken-num3":{"re":["[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))|\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"smollm":{"re":["\\p{N}","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"codeshell":{"re":["\\p{N}","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"bloom":{"re":[" ?[^(\\s|.,!?…。，、।۔،)]+"],"im":false},"gpt3-finnish":{"re":[" ?[^(\\s|.,!?…。，、।۔،)]+"],"im":false},"exaone":{"re":["\\p{N}","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"exaone4":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"exaone-moe":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"],"im":false},"chameleon":{"re":["<sentinel:[0-9]+>","(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z","([\\t\\n]|    |  )","\\p{N}","[\\p{P}!-/:-@\\[-`{-~]","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"minerva-7b":{"re":["\\p{N}","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"domyn-small":{"re":["[^\\n]+|[\\n]+"],"im":false},"megrez":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"gpt-4o":{"re":["[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"llama4":{"re":["[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"kanana2":{"re":["[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"talkie":{"re":["[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"granite-embed-multi-97m":{"re":["[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true},"tiny_aya":{"re":["\\d{1,3}(?=(?:\\d{3})*\\b)","[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"cohere2moe":{"re":["\\d{1,3}(?=(?:\\d{3})*\\b)","[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"superbpe":{"re":["\\p{N}+","(?=(\\d{3})+(?!\\d))"],"im":false},"trillion":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"granite-docling":{"re":["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"bailingmoe":{"re":["'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"],"im":false},"bailingmoe2":{"re":["'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"],"im":false},"llada-moe":{"re":["'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"],"im":false},"seed-coder":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"hunyuan":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"hunyuan-dense":{"re":["\\p{N}{1,3}","[一-龥぀-ゟ゠-ヿ]+","[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"],"im":false},"hy_v4":{"re":["\\p{N}{1,3}","[一-龥぀-ゟ゠-ヿ]+","[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"],"im":false},"joyai-llm":{"re":["\\p{N}{1,3}","[一-龥぀-ゟ゠-ヿ]+","[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"],"im":false},"kimi-k2":{"re":["\\p{Han}+"],"im":false},"ufakzeka":{"re":["[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"grok-2":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"afmoe":{"re":["\\p{AFMoE_digits}","[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+","[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"laguna":{"re":["[^\\n]+|[\\n]+","(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"minimax-m2":{"re":["[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"solar-open":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"mellum2":{"re":["\\p{N}","'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],"im":false},"k2-horizon":{"re":["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":false},"nemotron-3.5-embed":{"re":["[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],"im":true}};
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
    const M = j.model || {}; if (M.type === "Unigram" && Array.isArray(M.vocab)) return fromHFUnigram(j);
    if (M.type !== "BPE" || !M.vocab || !M.merges) return null;
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
  // ---- SentencePiece (from .model or a GGUF 'llama' vocabulary): BPE merges by piece score, Unigram by Viterbi
  function fromSPM(o) {
    const vocab = new Map(); o.pieces.forEach((p, i) => { if (!vocab.has(p) && (o.types ? o.types[i] === 1 || o.types[i] === 6 || o.types[i] == null : true)) vocab.set(p, i); });
    let maxLen = 1; for (const p of vocab.keys()) maxLen = Math.max(maxLen, Array.from(p).length);
    const normalize = x => { if (o.nmt) x = x.replace(/[\t\n\r\v\f\u0085\u2028\u2029\u200b]/g, " ").replace(/[\u200c\u200d\u2060\ufeff\u0000-\u0008\u000e-\u001f\u007f]/g, ""); if (o.nfkc) x = x.normalize("NFKC"); if (o.removeWs) x = x.replace(/ +/g, " ").trim(); x = x.split(" ").join("\u2581"); if (o.dummy) x = "\u2581" + x; return x; };
    const unkId = o.types ? o.types.indexOf(2) : -1;
    return {kind: o.unigram ? "unigram" : "spmbpe", vocab, scores: o.scores, normalize, pretok: x => [x], byteLevel: false, byteFallback: o.byteFallback, unk: o.unk, unkId: unkId >= 0 ? unkId : null, maxLen};
  }
  function fromHFUnigram(j) {
    const M = j.model, N = [], P = [];
    (function w(n) { if (!n) return; if (n.type === "Sequence") return (n.normalizers || n.pretokenizers || []).forEach(w); N.push(n); })(j.normalizer);
    (function w(n) { if (!n) return; if (n.type === "Sequence") return (n.pretokenizers || []).forEach(w); P.push(n); })(j.pre_tokenizer);
    const ms = P.find(n => n.type === "Metaspace"), types = M.vocab.map((v, i) => i === M.unk_id ? 2 : 1);
    const T = fromSPM({pieces: M.vocab.map(v => v[0]), scores: M.vocab.map(v => v[1]), types, unigram: true, byteFallback: !!M.byte_fallback,
      dummy: !!ms && (ms.prepend_scheme ? ms.prepend_scheme !== "never" : ms.add_prefix_space !== false),
      removeWs: N.some(n => n.type === "Replace" && n.pattern && /\{2,\}/.test(n.pattern.Regex || "")) || N.some(n => n.type === "Strip"),
      nfkc: N.some(n => n.type === "Precompiled" || n.type === "NFKC"), nmt: N.some(n => n.type === "Precompiled")});
    if (!ms || ms.split !== false) T.pretok = x => x.split(/(?=\u2581)/).filter(Boolean);
    return T;
  }
  function spmBpe(T, w) {
    let sym = Array.from(w);
    while (sym.length > 1) { let best = -Infinity, bi = -1;
      for (let i = 0; i < sym.length - 1; i++) { const id = T.vocab.get(sym[i] + sym[i + 1]); if (id != null && T.scores[id] > best) { best = T.scores[id]; bi = i; } }
      if (bi < 0) break; sym.splice(bi, 2, sym[bi] + sym[bi + 1]); }
    return sym;
  }
  function unigram(T, w) {
    const ch = Array.from(w), n = ch.length, best = new Float64Array(n + 1).fill(-Infinity), from = new Int32Array(n + 1).fill(-1); best[0] = 0;
    const UNK = -1e6;
    for (let i = 0; i < n; i++) { if (best[i] === -Infinity) continue; let sub = "";
      for (let L = 1; L <= Math.min(T.maxLen, n - i); L++) { sub += ch[i + L - 1]; const id = T.vocab.get(sub);
        if (id != null) { const sc = best[i] + T.scores[id]; if (sc > best[i + L]) { best[i + L] = sc; from[i + L] = i; } } }
      if (best[i + 1] === -Infinity || from[i + 1] === -1) { const sc = best[i] + UNK; if (sc > best[i + 1]) { best[i + 1] = sc; from[i + 1] = i; } } }
    const out = []; for (let j = n; j > 0; j = from[j]) out.push(ch.slice(from[j], j).join("")); return out.reverse();
  }
  function readGGUF(buf) {
    const dv = new DataView(buf), u8 = new Uint8Array(buf), dec = new TextDecoder("utf-8", {ignoreBOM: true});
    if (dec.decode(u8.slice(0, 4)) !== "GGUF") return null;
    let o = 8; const u64 = () => { const v = Number(dv.getBigUint64(o, true)); o += 8; return v; }, str = () => { const L = u64(); const x = dec.decode(u8.subarray(o, o + L)); o += L; return x; };
    const SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}; o += 8; const nkv = u64(), kv = {};
    const scalar = t => { const v = t === 4 ? dv.getUint32(o, true) : t === 5 ? dv.getInt32(o, true) : t === 6 ? dv.getFloat32(o, true) : t === 7 ? dv.getUint8(o) : t === 0 ? dv.getUint8(o) : t === 1 ? dv.getInt8(o) : null; o += SZ[t]; return v; };
    const val = t => { if (t === 8) return str(); if (t === 9) { const et = dv.getUint32(o, true); o += 4; const c = u64(); const a = new Array(c); for (let i = 0; i < c; i++) a[i] = et === 8 ? str() : et === 9 ? val(9) : scalar(et); return a; } return scalar(t); };
    for (let k = 0; k < nkv; k++) { const key = str(); const t = dv.getUint32(o, true); o += 4; kv[key] = val(t); if (o > buf.byteLength) break; }
    return kv;
  }
  function fromGGUF(buf) {
    const kv = readGGUF(buf); if (!kv || !kv["tokenizer.ggml.tokens"]) return null;
    const toks = kv["tokenizer.ggml.tokens"], merges = kv["tokenizer.ggml.merges"], model = kv["tokenizer.ggml.model"], types = kv["tokenizer.ggml.token_type"];
    if (model === "llama" && kv["tokenizer.ggml.scores"]) return fromSPM({pieces: toks, scores: kv["tokenizer.ggml.scores"], types, unigram: false, byteFallback: true, dummy: kv["tokenizer.ggml.add_space_prefix"] !== false, removeWs: false, nfkc: false});
    if (!merges) return null;
    const j = {model: {type: "BPE", vocab: Object.fromEntries(toks.map((x, i) => [x, i])), merges, byte_fallback: model !== "gpt2"}};
    if (model === "gpt2") { const P = GGUF_PRE[kv["tokenizer.ggml.pre"] || "default"]; if (!P) return null; j.model.ignore_merges = P.im;
      j.pre_tokenizer = {type: "Sequence", pretokenizers: P.re.map(r => ({type: "Split", pattern: {Regex: r}, behavior: "Isolated"})).concat([{type: "ByteLevel", use_regex: false}])}; }
    else j.normalizer = {type: "Replace", pattern: {String: " "}, content: "\u2581"};
    const vocab = new Map(); toks.forEach((x, i) => { if (!vocab.has(x)) vocab.set(x, i); }); j.model.vocab = Object.fromEntries(vocab);
    return fromHF(j);
  }
  function fromSPMModel(u8) {
    const vi = (b, p) => { let r = 0, sh = 0, c; do { c = b[p.i++]; r += (c & 0x7f) * 2 ** sh; sh += 7; } while (c >= 0x80); return r; };
    const dec = new TextDecoder("utf-8", {ignoreBOM: true}), dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength), P = {i: 0};
    const pieces = [], scores = [], types = []; let unigram = true, byteFallback = false, dummy = true, removeWs = true, normName = "nmt_nfkc";
    const sub = (end, fn) => { const Q = {i: P.i}; while (Q.i < end) { const k = vi(u8, Q), f = Math.floor(k / 8), w = k & 7; fn(f, w, Q); } };
    const skip = (w, Q) => { if (w === 0) vi(u8, Q); else if (w === 5) Q.i += 4; else if (w === 1) Q.i += 8; else if (w === 2) { const m = vi(u8, Q); Q.i += m; } };
    while (P.i < u8.length) { const key = vi(u8, P), f = Math.floor(key / 8), w = key & 7;
      if (w !== 2) { skip(w, P); continue; } const L = vi(u8, P), end = P.i + L;
      if (f === 1) { let piece = "", sc = 0, ty = 1; sub(end, (f2, w2, Q) => { if (f2 === 1 && w2 === 2) { const m = vi(u8, Q); piece = dec.decode(u8.subarray(Q.i, Q.i + m)); Q.i += m; } else if (f2 === 2 && w2 === 5) { sc = dv.getFloat32(Q.i, true); Q.i += 4; } else if (f2 === 3 && w2 === 0) ty = vi(u8, Q); else skip(w2, Q); }); pieces.push(piece); scores.push(sc); types.push(ty); }
      else if (f === 2) sub(end, (f2, w2, Q) => { if (f2 === 3 && w2 === 0) unigram = vi(u8, Q) === 1; else if (f2 === 35 && w2 === 0) byteFallback = vi(u8, Q) === 1; else skip(w2, Q); });
      else if (f === 3) sub(end, (f2, w2, Q) => { if (f2 === 1 && w2 === 2) { const m = vi(u8, Q); normName = dec.decode(u8.subarray(Q.i, Q.i + m)); Q.i += m; } else if (f2 === 3 && w2 === 0) dummy = vi(u8, Q) === 1; else if (f2 === 4 && w2 === 0) removeWs = vi(u8, Q) === 1; else skip(w2, Q); });
      P.i = end; }
    if (!pieces.length) return null;
    const unk = pieces[types.indexOf(2)];
    return fromSPM({pieces, scores, types, unigram, byteFallback, dummy, removeWs, nfkc: /nfkc/i.test(normName), nmt: /^nmt/i.test(normName), unk});
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
      const syms = T.kind === "spmbpe" ? spmBpe(T, w) : T.kind === "unigram" ? unigram(T, w) : bpeWord(T, w);
      for (const x of syms) { const id = T.vocab.get(x);
        if (id != null) ids.push(id);
        else if (T.byteFallback) { for (const b of enc8.encode(x)) { const k = T.vocab.get("<0x" + b.toString(16).toUpperCase().padStart(2, "0") + ">"); if (k != null) ids.push(k); } }
        else if (T.unkId != null) { if (ids[ids.length - 1] !== T.unkId || T.kind === "unigram") ids.push(T.unkId); }
        else if (T.unk != null && T.vocab.has(T.unk)) ids.push(T.vocab.get(T.unk)); } }
    return ids;
  }
  // parents for rank-based vocabularies: the last merge BPE performs when it encodes the piece on its own
  function parentsOf(T) {
    if (T.parents) return T.parents;
    if (T.kind === "unigram") return (T.parents = new Map());
    if (T.kind === "spmbpe") { const P = new Map();
      for (const [s, id] of T.vocab) { let sym = Array.from(s); if (sym.length < 2) continue; let last = null;
        while (sym.length > 1) { let best = -Infinity, bi = -1;
          for (let i = 0; i < sym.length - 1; i++) { const q = T.vocab.get(sym[i] + sym[i + 1]); if (q != null && T.scores[q] > best) { best = T.scores[q]; bi = i; } }
          if (bi < 0) break; last = [sym[bi], sym[bi + 1]]; sym.splice(bi, 2, sym[bi] + sym[bi + 1]); }
        if (sym.length === 1 && last) P.set(id, [T.vocab.get(last[0]), T.vocab.get(last[1])]); }
      return (T.parents = P); }
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
    if (n.endsWith(".gguf")) return fromGGUF(buf);
    if (n.endsWith(".model") || n.includes(".model.")) { try { const r = fromSPMModel(u8); if (r) return r; } catch (e) {} }
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
