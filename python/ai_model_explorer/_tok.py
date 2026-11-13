"""Read the vocabulary of any common tokenizer format, without extra dependencies.

Formats: tokenizer.json (Hugging Face), tokenizer.model (SentencePiece), vocab.json (GPT-2 style BPE),
*.tiktoken / tiktoken.model (OpenAI, Qwen, Kimi style), tekken.json (Mistral) and .gguf (llama.cpp).
Encoding real text (for the coverage check) uses an optional library when one is installed:
tokenizers, sentencepiece, tiktoken or llama_cpp.
"""
import base64
import json
import os
import struct
from typing import Callable, List, Optional

# Files tried in a folder or a Hub repo, best first.
CANDIDATES = ("tokenizer.json", "tekken.json", "tokenizer.model", "spiece.model", "sentencepiece.bpe.model",
              "vocab.json", "tiktoken.model", "qwen.tiktoken")


class Tok:
    def __init__(self, pieces: List[str], byte_level: bool, kind: str, path: Optional[str] = None, meta: Optional[dict] = None):
        self.pieces, self.byte_level, self.kind, self.path, self.meta = pieces, byte_level, kind, path, meta or {}

    def __len__(self):
        return len(self.pieces)

    def encoder(self) -> Optional[Callable[[str], List[int]]]:
        """A function text -> token ids, or None when no library for this format is installed."""
        p, k = self.path, self.kind
        try:
            if k == "hf":
                from tokenizers import Tokenizer
                t = Tokenizer.from_file(p)
                return lambda s: t.encode(s, add_special_tokens=False).ids
            if k == "spm":
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor(model_file=p)
                return sp.encode
            if k in ("tiktoken", "tekken"):
                import tiktoken
                ranks = self.meta["ranks"]
                pat = self.meta.get("pattern") or r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
                e = tiktoken.Encoding("x", pat_str=pat, mergeable_ranks=ranks, special_tokens={})
                off = self.meta.get("offset", 0)
                return lambda s: [i + off for i in e.encode_ordinary(s)]
            if k == "gguf":
                import llama_cpp
                m = llama_cpp.Llama(model_path=p, vocab_only=True, verbose=False)
                return lambda s: m.tokenize(s.encode(), add_bos=False)
        except ImportError:
            return None
        return None


# ---------------------------------------------------------------- readers
def _from_tokenizer_json(path: str) -> Tok:
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    m = j.get("model") or {}
    v = m.get("vocab")
    if isinstance(v, dict):
        pieces = [None] * ((max(v.values()) + 1) if v else 0)
        for k, i in v.items():
            pieces[i] = k
    else:
        pieces = [x[0] if isinstance(x, list) else x for x in (v or [])]
    for a in j.get("added_tokens") or []:
        i = a.get("id")
        while len(pieces) <= i:
            pieces.append(None)
        pieces[i] = "<" + a["content"].strip("<>") + ">" if a.get("special") else a["content"]
    pieces = [p if p is not None else "<unused>" for p in pieces]
    txt = json.dumps({k: j.get(k) for k in ("decoder", "pre_tokenizer")})
    return Tok(pieces, "ByteLevel" in txt, "hf", path)


def _from_vocab_json(path: str) -> Tok:
    with open(path, encoding="utf-8") as f:
        v = json.load(f)
    pieces = [None] * (max(v.values()) + 1)
    for k, i in v.items():
        pieces[i] = k
    return Tok([p if p is not None else "<unused>" for p in pieces], True, "bpe-vocab", path)


def _bytes_piece(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")   # an incomplete UTF-8 sequence keeps U+FFFD: a fragment of a longer character


def _from_tiktoken(path: str) -> Tok:
    ranks = {}
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    pieces = ["<unused>"] * (max(ranks.values()) + 1)
    for b, r in ranks.items():
        pieces[r] = _bytes_piece(b)
    return Tok(pieces, False, "tiktoken", path, {"ranks": ranks})


def _from_tekken(path: str) -> Tok:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cfg = d.get("config") or {}
    ns = cfg.get("default_num_special_tokens", len(d.get("special_tokens") or []))
    size = cfg.get("default_vocab_size") or (ns + len(d["vocab"]))
    voc = d["vocab"][: size - ns]
    ranks = {base64.b64decode(x["token_bytes"]): i for i, x in enumerate(voc)}
    pieces = ["<special>"] * ns + [_bytes_piece(base64.b64decode(x["token_bytes"])) for x in voc]
    return Tok(pieces, False, "tekken", path, {"ranks": ranks, "offset": ns, "pattern": cfg.get("pattern")})


def _varint(b: bytes, i: int):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return r, i


def _from_sentencepiece(path: str) -> Tok:
    """ModelProto: field 1 = repeated SentencePiece {1: piece, 2: score, 3: type}. Type 2/3/4/6 = unknown/control/user/byte."""
    with open(path, "rb") as f:
        b = f.read()
    i, pieces = 0, []
    while i < len(b):
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 2:
            n, i = _varint(b, i)
            if fn == 1:
                sub, j, piece, typ = b[i:i + n], 0, "", 1
                while j < len(sub):
                    k2, j = _varint(sub, j)
                    f2, w2 = k2 >> 3, k2 & 7
                    if w2 == 2:
                        m, j = _varint(sub, j)
                        if f2 == 1:
                            piece = sub[j:j + m].decode("utf-8", errors="replace")
                        j += m
                    elif w2 == 5:
                        j += 4
                    elif w2 == 0:
                        val, j = _varint(sub, j)
                        if f2 == 3:
                            typ = val
                    elif w2 == 1:
                        j += 8
                    else:
                        break
                pieces.append("<" + piece.strip("<>") + ">" if typ in (2, 3) else piece)
            i += n
        elif wt == 0:
            _, i = _varint(b, i)
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8
        else:
            break
    return Tok(pieces, False, "spm", path)


def _from_gguf(path: str) -> Tok:
    """Only the header key/values are read; tokenizer.ggml.tokens is a string array."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        ver = struct.unpack("<I", f.read(4))[0]
        rd_n = (lambda: struct.unpack("<Q", f.read(8))[0]) if ver >= 2 else (lambda: struct.unpack("<I", f.read(4))[0])
        f.read(8 if ver >= 2 else 4)   # tensor count
        n_kv = rd_n()
        SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def rd_str():
            return f.read(rd_n()).decode("utf-8", errors="replace")

        def rd_val(t):
            if t == 8:
                return rd_str()
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]; n = rd_n()
                if et == 8:
                    return [rd_str() for _ in range(n)]
                f.seek(SIZES[et] * n, 1) if et in SIZES else [rd_val(et) for _ in range(n)]
                return None
            f.seek(SIZES[t], 1)
            return None

        tokens, model = None, None
        for _ in range(n_kv):
            key = rd_str(); t = struct.unpack("<I", f.read(4))[0]
            v = rd_val(t)
            if key == "tokenizer.ggml.tokens":
                tokens = v
            elif key == "tokenizer.ggml.model":
                model = v
            if tokens is not None and model is not None:
                break
    if not tokens:
        raise ValueError("GGUF file has no tokenizer.ggml.tokens")
    return Tok(tokens, model == "gpt2", "gguf", path, {"model": model})


def read_file(path: str) -> Tok:
    n = os.path.basename(path).lower()
    if n.endswith(".gguf"):
        return _from_gguf(path)
    if n == "tekken.json" or n.startswith("tekken"):
        return _from_tekken(path)
    if n.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            head = f.read(4096)
        if '"model"' in head or '"added_tokens"' in head or '"version"' in head and '"truncation"' in head:
            return _from_tokenizer_json(path)
        if '"config"' in head and '"vocab"' in head:
            return _from_tekken(path)
        return _from_vocab_json(path)
    if n.endswith(".tiktoken") or n == "tiktoken.model":
        return _from_tiktoken(path)
    if n.endswith(".model") or ".model." in n:
        try:
            return _from_sentencepiece(path)
        except Exception:
            return _from_tiktoken(path)   # some repos store tiktoken ranks in a .model file
    raise ValueError(f"unknown tokenizer format: {path}")


def read_dir(d: str) -> Optional[Tok]:
    for c in CANDIDATES:
        p = os.path.join(d, c)
        if os.path.exists(p):
            return read_file(p)
    for f in sorted(os.listdir(d)):
        if f.endswith((".tiktoken", ".gguf")):
            return read_file(os.path.join(d, f))
    return None


def from_hub(repo: str, token: Optional[str] = None, cache_dir: Optional[str] = None) -> Tok:
    """Download only the tokenizer file of a Hub repo (never the weights)."""
    from ._hub import Hub
    h = Hub(token)
    try:
        sib = [s["rfilename"] for s in (h.json(f"https://huggingface.co/api/models/{repo}") or {}).get("siblings", [])]
    except Exception:
        sib = []
    names = [c for c in CANDIDATES if not sib or c in sib] + [s for s in sib if s.endswith(".tiktoken") and "/" not in s]
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "ai-model-explorer", "tokenizers", repo.replace("/", "__"))
    for n in names:
        local = os.path.join(cache_dir, n)
        if not os.path.exists(local):
            st, body = h.get(f"https://huggingface.co/{repo}/resolve/main/{n}")
            if st != 200 or not body:
                continue
            os.makedirs(cache_dir, exist_ok=True)
            with open(local, "wb") as f:
                f.write(body if isinstance(body, bytes) else body.encode())
        try:
            return read_file(local)
        except Exception:
            continue
    raise RuntimeError(f"no tokenizer file found in {repo} (tried {', '.join(names) or 'none'}); pass tokenizer=<path>")


def load(spec=None, source: Optional[dict] = None, token: Optional[str] = None) -> Tok:
    """spec: a path (file or folder), a Hub repo id, a tokenizer.json dict, or None to use the model's own source."""
    if isinstance(spec, Tok):
        return spec
    if isinstance(spec, dict):
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(spec, f)
        return _from_tokenizer_json(p)
    if isinstance(spec, str):
        if os.path.isdir(spec):
            t = read_dir(spec)
            if t:
                return t
            raise RuntimeError(f"no tokenizer file in {spec}")
        if os.path.exists(spec):
            return read_file(spec)
        if "/" in spec:
            return from_hub(spec, token)
        raise RuntimeError(f"tokenizer not found: {spec}")
    src = source or {}
    if src.get("path"):
        p = src["path"]
        if os.path.isfile(p) and p.lower().endswith(".gguf"):
            return read_file(p)
        t = read_dir(p if os.path.isdir(p) else os.path.dirname(p))
        if t:
            return t
    if src.get("repo"):
        return from_hub(src["repo"], token)
    raise RuntimeError("no tokenizer found; pass tokenizer=<path, folder or Hub repo>")


def corpus(lang: str) -> List[str]:
    """Bundled test sentences for a language (PUD; 'th' also gets Thai social-media text). Empty if none."""
    from importlib import resources
    out = []
    for n in (f"{lang}.txt",) + (("th_social.txt",) if lang == "th" else ()):
        try:
            out += [l.strip() for l in resources.files("ai_model_explorer").joinpath("data", "corpus", n).read_text(encoding="utf-8").splitlines() if l.strip()]
        except FileNotFoundError:
            pass
    return out
