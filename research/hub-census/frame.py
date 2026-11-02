"""Build the sampling frame: every public model on the Hugging Face Hub with its popularity stats.

Source 1 (default): the community hub-stats dump (cfahlgren1/hub-stats, models.parquet), refreshed daily.
Source 2 (--api):   the Hub /api/models listing, paged by cursor. Slower; use when the dump is unavailable.

Output: frame.parquet with one row per repo and these columns (missing ones are left empty):
  id, author, downloads (30-day), downloads_all_time, likes, created_at, last_modified,
  library, pipeline_tag, tags, gated, has_safetensors_tag
"""
import argparse, json, os, sys, time, urllib.parse, urllib.request
import pandas as pd

EP = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
TOKEN = os.environ.get("HF_TOKEN")
DUMP = EP + "/datasets/cfahlgren1/hub-stats/resolve/main/models.parquet"

ALIASES = {  # dump/API names -> frame names
    "id": ["id", "modelId", "model_id"],
    "author": ["author"],
    "downloads": ["downloads"],
    "downloads_all_time": ["downloadsAllTime", "downloads_all_time"],
    "likes": ["likes"],
    "created_at": ["createdAt", "created_at"],
    "last_modified": ["lastModified", "last_modified"],
    "library": ["library_name", "libraryName", "library"],
    "pipeline_tag": ["pipeline_tag", "pipelineTag"],
    "tags": ["tags"],
    "gated": ["gated"],
}


def _req(url):
    h = {"User-Agent": "ai-model-explorer-census"}
    if TOKEN:
        h["Authorization"] = "Bearer " + TOKEN
    return urllib.request.Request(url, headers=h)


def _open(url, tries=6):
    for i in range(tries):
        try:
            return urllib.request.urlopen(_req(url), timeout=120)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                ra = e.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit() else 2 ** i * 3)
                continue
            raise


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for name, cands in ALIASES.items():
        col = next((c for c in cands if c in df.columns), None)
        out[name] = df[col] if col else None
    if out["author"].isna().all():
        out["author"] = out["id"].astype(str).str.split("/").str[0]
    for c in ("downloads", "downloads_all_time", "likes"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")
    for c in ("created_at", "last_modified"):
        out[c] = pd.to_datetime(out[c], errors="coerce", utc=True)
    out["tags"] = out["tags"].apply(lambda t: [str(x) for x in t] if isinstance(t, (list, tuple)) or hasattr(t, "tolist") else [])
    out["gated"] = out["gated"].apply(lambda g: bool(g) and str(g).lower() not in ("false", "0", "none"))
    out["has_safetensors_tag"] = out["tags"].apply(lambda t: "safetensors" in t)
    # original release or derivative: the Hub tags derivatives as base_model:<relation>:<parent>
    def origin(t):
        rel = [x.split(":")[1] for x in t if x.startswith("base_model:") and x.count(":") >= 2]
        for r in ("quantized", "adapter", "merge", "finetune"):
            if r in rel:
                return r
        return "original"
    out["origin"] = out["tags"].apply(origin)
    return out.dropna(subset=["id"]).drop_duplicates("id").reset_index(drop=True)


def from_dump(path):
    """hub-stats models.parquet is ~1.5 GB. Download it with huggingface_hub (parallel, resumable, progress bar),
    then read only the columns we need so memory stays small."""
    import pyarrow.parquet as pq
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        print("downloading cfahlgren1/hub-stats models.parquet (~1.5 GB)", file=sys.stderr, flush=True)
        got = hf_hub_download("cfahlgren1/hub-stats", "models.parquet", repo_type="dataset",
                              local_dir=os.path.dirname(path) or ".", token=TOKEN)
        if os.path.abspath(got) != os.path.abspath(path):
            os.replace(got, path)
    names = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for cands in ALIASES.values() for c in cands if c in names]
    print(f"reading {len(cols)} of {len(names)} columns: {cols}", file=sys.stderr, flush=True)
    return pq.read_table(path, columns=cols).to_pandas()


def from_api(limit_pages):
    url = EP + "/api/models?" + urllib.parse.urlencode(
        [("limit", 1000), ("sort", "downloads"), ("direction", -1)] +
        [("expand[]", k) for k in ("downloads", "downloadsAllTime", "likes", "createdAt", "lastModified",
                                   "library_name", "pipeline_tag", "tags", "gated")])
    rows, page = [], 0
    while url and page < limit_pages:
        with _open(url) as r:
            rows += json.load(r)
            nxt = r.headers.get("Link", "")
        url = nxt.split(";")[0].strip("<> ") if 'rel="next"' in nxt else None
        page += 1
        print(f"page {page}: {len(rows):,} repos", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def describe(f: pd.DataFrame) -> dict:
    yr = f["created_at"].dt.year
    q = lambda s: {p: int(s.quantile(p / 100)) for p in (50, 90, 99, 99.9)}
    top = f["downloads"].sort_values(ascending=False)
    return {
        "repos": len(f),
        "with_safetensors_tag": int(f["has_safetensors_tag"].sum()),
        "gated": int(f["gated"].sum()),
        "zero_downloads_30d": int((f["downloads"] == 0).sum()),
        "downloads_30d_quantiles": q(f["downloads"]),
        "likes_quantiles": q(f["likes"]),
        "share_of_downloads_top_1000": round(float(top.head(1000).sum() / max(top.sum(), 1)), 4),
        "share_of_downloads_top_10000": round(float(top.head(10000).sum() / max(top.sum(), 1)), 4),
        "created_per_year": {int(k): int(v) for k, v in yr.value_counts().sort_index().items() if pd.notna(k)},
        "origin": f["origin"].value_counts().to_dict(),
        "origin_share_of_downloads": {k: round(float(v), 4) for k, v in (f.groupby("origin")["downloads"].sum() / max(f["downloads"].sum(), 1)).items()},
        "top_libraries": f["library"].fillna("(none)").value_counts().head(15).to_dict(),
        "top_pipeline_tags": f["pipeline_tag"].fillna("(none)").value_counts().head(15).to_dict(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/frame.parquet")
    ap.add_argument("--dump", default="out/models.parquet", help="where to keep the hub-stats download")
    ap.add_argument("--api", action="store_true", help="page the Hub API instead of the hub-stats dump")
    ap.add_argument("--api-pages", type=int, default=100000)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.api:
        raw = from_api(a.api_pages)
    else:
        try:
            raw = from_dump(a.dump)
        except Exception as e:  # dump moved or unreachable: fall back to the API listing
            print(f"hub-stats dump unavailable ({e}); paging the Hub API instead", file=sys.stderr)
            raw = from_api(a.api_pages)
    f = normalize(raw)
    f.to_parquet(a.out, index=False)
    stats = describe(f)
    with open(os.path.join(os.path.dirname(a.out) or ".", "frame_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(json.dumps(stats, indent=2, default=str))
