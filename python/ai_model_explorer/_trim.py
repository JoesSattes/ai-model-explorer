"""Cut a tokenizer down to the pieces a set of languages needs, so the model's token tables can be sliced to match.

The kept pieces keep their order, so new id = rank of the old id among the kept ids. Text in the kept languages is
tokenized exactly as before (same pieces); any other text still encodes, through shorter pieces or byte fallback.
Supported here: tokenizer.json with a BPE, WordPiece or Unigram model, and SentencePiece .model files.
"""
import json
from typing import List, Tuple


def trim_tokenizer_json(j: dict, keep: List[bool]) -> Tuple[dict, List[int]]:
    """keep[i] for every id of the original tokenizer. Returns (new tokenizer.json dict, kept old ids in new-id order)."""
    j = json.loads(json.dumps(j))
    added = {a["id"] for a in j.get("added_tokens") or []}
    n = len(keep)
    keep = [bool(k) or i in added for i, k in enumerate(keep)]
    m = j["model"]
    typ = m.get("type")
    kept_ids = [i for i in range(n) if keep[i]]
    new_id = {old: new for new, old in enumerate(kept_ids)}
    if typ in ("BPE", "WordPiece", "WordLevel") and isinstance(m.get("vocab"), dict):
        inv = {v: k for k, v in m["vocab"].items()}
        m["vocab"] = {inv[i]: new_id[i] for i in kept_ids if i in inv}
        ks = set(m["vocab"])
        if typ == "BPE":
            mg = []
            for x in m.get("merges") or []:
                a, b = (x.split(" ", 1) if isinstance(x, str) else x)
                if a in ks and b in ks and (a + b) in ks:
                    mg.append(x)
            m["merges"] = mg
    elif typ == "Unigram" and isinstance(m.get("vocab"), list):
        m["vocab"] = [m["vocab"][i] for i in kept_ids if i < len(m["vocab"])]
        if m.get("unk_id") is not None:
            m["unk_id"] = new_id.get(m["unk_id"], 0)
    else:
        raise ValueError(f"cannot trim a tokenizer.json with model type {typ}")
    for a in j.get("added_tokens") or []:
        a["id"] = new_id[a["id"]]
    pp = j.get("post_processor") or {}
    for proc in [pp] + list(pp.get("processors") or []):
        for st in (proc.get("special_tokens") or {}).values():
            st["ids"] = [new_id.get(i, i) for i in st.get("ids", [])]
        for k in ("sep", "cls"):
            if isinstance(proc.get(k), list) and len(proc[k]) == 2:
                proc[k][1] = new_id.get(proc[k][1], proc[k][1])
    return j, kept_ids


def trim_sentencepiece(model_bytes: bytes, keep: List[bool]) -> Tuple[bytes, List[int]]:
    """Drop removed NORMAL pieces from the model; byte fallback covers text that needed them.
    Kept pieces keep their order, so new id = rank among kept ids, as for tokenizer.json."""
    from sentencepiece import sentencepiece_model_pb2 as pb
    mp = pb.ModelProto()
    mp.ParseFromString(model_bytes)
    kept = [i for i, p in enumerate(mp.pieces) if not (i < len(keep) and not keep[i] and p.type == pb.ModelProto.SentencePiece.NORMAL)]
    ks = set(kept)
    out = pb.ModelProto(); out.CopyFrom(mp)
    del out.pieces[:]
    out.pieces.extend(p for i, p in enumerate(mp.pieces) if i in ks)
    return out.SerializeToString(), kept


SLICE_SNIPPET = '''# Slice the token tables to the kept rows (run once, then save the model with the new tokenizer).
import json, torch
keep = torch.tensor(json.load(open("kept_ids.json")))
emb = model.get_input_embeddings(); emb.weight.data = emb.weight.data[keep]; emb.num_embeddings = len(keep)
head = model.get_output_embeddings()
if head is not None and head.weight.data_ptr() != emb.weight.data_ptr():
    head.weight.data = head.weight.data[keep]; head.out_features = len(keep)
model.config.vocab_size = len(keep)
# Per-layer embeddings (Gemma 3n / 4): slice their first dimension with the same `keep` as well.
'''
