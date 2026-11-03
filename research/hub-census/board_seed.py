"""Fill the Deploy-Ready board before anyone submits: the leading providers' most used releases, the most downloaded
language models, and models built for Thai, all scored exactly as a submission is (submit.score).

usage:
  python board_seed.py --census out/census.jsonl --out ../../census/data/board.json
  python board_seed.py --extra board_extra.txt --out ../../census/data/board.json   # read listed repos live (needs network)

--census rebuilds the census part of the board; --extra reads repos the census did not cover and keeps them.
Rows read live keep their place across census rebuilds. Either can be given alone.
"""
import argparse, datetime, json, os

import labels
from submit import read, score

LM = {"decoder", "prefix-lm"}
FIXTURE = r"internal-testing|(?:^|[/_-])tiny(?:[-_]|$)|random|dummy|unit-test|^unslothai/"
GEN = {"text-generation", "text2text-generation", "image-text-to-text", "any-to-any", "audio-text-to-text", "video-text-to-text"}
PER_LAB, TOP_LM, MULTI = 3, 25, 20


def row(r, tags, groups, src):
    r = dict(r)
    r["license"] = labels.license_of(tags)
    s = score(r)
    created = str(r.get("created_at") or "")[:4]
    return {"id": r["id"], "src": src, "groups": sorted(groups), "th": labels.thai(r["id"], tags, r.get("family_manifest")),
            "origin": labels.origin(r.get("origin"), r.get("quant") or {}, r["id"], tags), "lab": labels.is_lab(r["id"]),
            "y": int(created) if created.isdigit() else None, "pt": r.get("pipeline_tag"), **s}


def from_census(path):
    import pandas as pd
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    d = d[(d["status"] == "ok") & ~d["id"].str.contains(FIXTURE, case=False, regex=True) & d["id"].apply(labels.safe)]
    d = d[d["params"].fillna(0) >= 5e7]
    tags = labels.tags_by_id(path)
    d["tg"] = d["id"].apply(lambda i: tags.get(i, []))
    d["orig"] = d.apply(lambda r: labels.origin(r.get("origin"), r["quant"] if isinstance(r["quant"], dict) else {}, r["id"], r["tg"]), axis=1)
    d["th"] = d.apply(lambda r: labels.thai(r["id"], r["tg"], r["family_manifest"]), axis=1)
    d = d.sort_values("downloads", ascending=False)
    d["gen"] = d.apply(lambda r: r["family_manifest"] in LM and (r.get("pipeline_tag") in GEN or not r.get("pipeline_tag")), axis=1)
    orig = d[d["orig"] == "original"]
    groups = {}
    add = lambda ids, g: [groups.setdefault(i, set()).add(g) for i in ids]
    for lab, g in orig[orig["author"].isin(labels.LABS)].groupby("author"):   # each provider's most used releases
        lm = g[g["gen"]]
        add((lm if len(lm) else g)["id"].head(PER_LAB), "lab")
    add(orig[orig["gen"]]["id"].head(TOP_LM), "popular")
    add(d[d["th"].isin(["TH", "TH/EN"])]["id"], "thai")                       # built for Thai: every one, copies included
    add(orig[(orig["th"] == "Multilingual") & orig["family_manifest"].isin(labels.TEXT_FAMS)]["id"].head(MULTI), "thai")
    by = d.set_index("id")
    return [row({"id": i, **by.loc[i].to_dict()}, by.loc[i, "tg"], g, "census") for i, g in groups.items()]


def from_extra(path, have):
    out = []
    for line in open(path):
        line = line.split("#")[0].split()
        if not line or line[0] in have:
            continue
        repo, groups = line[0], set(line[1].split(",")) if len(line) > 1 else {"thai"}
        r = read(repo)
        if r.get("status") != "ok":
            print(f"skip {repo}: {r.get('status')} {r.get('error', '')}"[:200])
            continue
        r["id"] = repo
        out.append(row(r, r.get("tags") or [], groups, "read") | {"read": datetime.date.today().isoformat(),
                                                                  "downloads": r.get("downloads"), "likes": r.get("likes")})
        print(f"read {repo}: {out[-1]['tier']} {out[-1]['score']}/5")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census")
    ap.add_argument("--extra")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    old = json.load(open(a.out)) if os.path.exists(a.out) else []
    if a.census:
        board = from_census(a.census)
        ids = {b["id"] for b in board}
        board += [b for b in old if b.get("src") == "read" and b["id"] not in ids]   # live reads survive a census rebuild
    else:
        board = old
    if a.extra:
        board += from_extra(a.extra, {b["id"] for b in board})
    board.sort(key=lambda b: -(b.get("downloads") or 0))
    clean = lambda v: None if isinstance(v, float) and v != v else v
    board = [{k: clean(v) for k, v in b.items()} for b in board]
    json.dump(board, open(a.out, "w"), separators=(",", ":"), default=float, allow_nan=False)
    print(f"{len(board)} rows -> {a.out}")


if __name__ == "__main__":
    main()
