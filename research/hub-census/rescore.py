"""Re-score a finished crawl with the current recovery rules, offline.

crawl.py keeps every repo's raw manifest (tensor names, dtypes, shapes) and config in census_manifests.jsonl.gz.
After a rule change, run this instead of crawling the Hub again:

    python rescore.py --census out/census.jsonl --out out/census.jsonl
    python summarize.py
Popularity, status and timing fields are kept; everything derived from the weights is recomputed.
"""
import argparse, gzip, json, os, sys

from ai_model_explorer._engine import Engine
from crawl import analyze


def main(census, out):
    recs = {}
    with open(census) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["id"]] = r
    man = census.replace(".jsonl", "") + "_manifests.jsonl.gz"
    if not os.path.exists(man):
        sys.exit(f"{man} not found: this crawl predates raw manifests, crawl again once")
    eng, n, bad = Engine(), 0, 0
    with gzip.open(man, "rt") as fh:
        for line in fh:
            m = json.loads(line)
            if m["id"] not in recs:
                continue
            tensors = {k: {"dtype": d, "shape": s} for k, (d, s) in m["tensors"].items()}
            try:
                new = analyze(eng, tensors, m.get("config"), m.get("meta") or {})
            except Exception as e:  # noqa: BLE001
                new = {"status": "error", "error": f"rescore: {type(e).__name__}: {e}"[:300]}
                bad += 1
                eng = Engine()  # a failed evaluation can leave the JS context in a bad state
            recs[m["id"]].update(new)
            n += 1
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        for r in recs.values():
            fh.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, out)
    print(f"re-scored {n} repos ({bad} failed) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="out/census.jsonl")
    ap.add_argument("--out", default="out/census.jsonl")
    a = ap.parse_args()
    main(a.census, a.out)
