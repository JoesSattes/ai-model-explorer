"""Re-score a finished crawl with the current recovery rules, offline.

crawl.py keeps every repo's raw manifest (tensor names, dtypes, shapes) and config in census_manifests.jsonl.gz.
After a rule change, run this instead of crawling the Hub again:

    python rescore.py --census out/census.jsonl --out out/census.jsonl
    python summarize.py
Popularity, status and timing fields are kept; everything derived from the weights is recomputed.
"""
import argparse, gzip, json, multiprocessing as mp, os, sys

from ai_model_explorer._engine import Engine
from crawl import analyze


_eng = None


def _init():
    global _eng
    _eng = Engine()


def _one(line):
    global _eng
    m = json.loads(line)
    tensors = {k: {"dtype": d, "shape": s} for k, (d, s) in m["tensors"].items()}
    try:
        return m["id"], analyze(_eng, tensors, m.get("config"), m.get("meta") or {})
    except Exception as e:  # noqa: BLE001
        _eng = Engine()  # a failed evaluation can leave the JS context in a bad state
        return m["id"], {"status": "error", "error": f"rescore: {type(e).__name__}: {e}"[:300]}


def main(census, out, workers=None):
    workers = workers or os.cpu_count() or 1
    recs = {}
    with open(census) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["id"]] = r
    man = census.replace(".jsonl", "") + "_manifests.jsonl.gz"
    if not os.path.exists(man):
        sys.exit(f"{man} not found: this crawl predates raw manifests, crawl again once")
    def todo():
        with gzip.open(man, "rt") as fh:
            for line in fh:
                i = line[8:line.index('"', 8)] if line.startswith('{"id": "') else json.loads(line)["id"]
                if i in recs:
                    yield line
    n = bad = 0
    with mp.Pool(workers, initializer=_init) as pool:
        for rid, new in pool.imap_unordered(_one, todo(), chunksize=8):
            recs[rid].update(new)
            n += 1
            bad += new.get("status") == "error" and str(new.get("error", "")).startswith("rescore:")
            if n % 1000 == 0:
                print(f"{n} re-scored", flush=True)
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
    ap.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    a = ap.parse_args()
    main(a.census, a.out, a.workers)
