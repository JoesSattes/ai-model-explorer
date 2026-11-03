"""Publish the labeled corpus (510 architectures, per-tensor ground truth) as a Hugging Face dataset.

usage: python publish_corpus.py <owner/name> [../data/corpus-510.jsonl.gz]   (needs HF_TOKEN with write access)
"""
import datetime, gzip, json, os, sys

from card_meta import yaml, CITATION, CITE_LINE

CARD = """---
license: apache-2.0
pretty_name: Manifest Anatomy Corpus
{meta}
size_categories: [n<1K]
configs:
- config_name: default
  data_files: corpus.parquet
---

# Manifest Anatomy Corpus

Every model type registered in Hugging Face `transformers` 5.22.1 that instantiates from its default configuration,
built on PyTorch's meta device (shapes only, no weights), deduplicated: **{n} architectures, {t:,} tensors**, from
{pmin} to {pmax} parameters. Each record has the tensor manifest a safetensors header would carry and ground truth
that a manifest reader never sees: the role of every tensor from the Python class that owns it, the block lists with
their block types, hyperparameters, and the configuration.

It is the benchmark of S. Singkul, *Reading the Architecture off the Weights: Manifest-Only Model Anatomy and a Census
of 12,974 Hugging Face Repositories*, 2026 ([paper](https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_Manifest-Only_Recovery_of_Neural_Network_Anatomy),
[code](https://github.com/JoesSattes/ai-model-explorer), [web app](https://joessattes.github.io/ai-model-explorer/en/)).
Its companion, [hub-weight-census](https://huggingface.co/datasets/{owner}/hub-weight-census), applies the same
recovery to 12,974 Hub repositories and carries these labels over to them by model type.

{cite_line}

## Files

- `corpus.parquet`: one flat row per architecture (shown in the viewer above): model class, parameters, tensor counts by
  role, block lists, block types and configuration hyperparameters.
- `corpus.jsonl.gz`: the full records, fields below.

## Fields (corpus.jsonl.gz, one JSON object per line)

- `type`: transformers model type; `name`; `cls`: the model class instantiated
- `tensors`: `{{name: {{dtype, shape}}}}`, exactly what a safetensors header lists
- `gt`: `{{name: {{kind, owner}}}}`: role (`attn`, `cross`, `ffn`, `moe`, `ssm`, `conv`, `rnn`, `norm`, `embed`, `head`,
  `adapter`, `mod` (adaLN modulation), `other`) and the class that owns the tensor
- `lists`: block lists with path, length, block-type sequence, item class, expert flag
- `hp`: hyperparameters from the configuration; `cfg`: the configuration; `params`: parameter total; `n`: tensor count

## Use

Score any architecture-recovery, model-search or conversion tool against `gt` and `lists`; the reference results are
in the paper (96.8% macro role accuracy, 95.0% with every name replaced by a random token). Reproduce or extend with
`research/build_corpus.py` and `research/eval.js` in the code repository.

## Limits

Default configurations give small, untrained models whose shapes can differ from released checkpoints; class names are an independent but
imperfect source of labels (see the paper's discussion). Model types that need remote code are not included.
Built {date}. Configurations and shapes derive from `transformers` (Apache-2.0).

{citation}"""


def main(repo, path):
    from huggingface_hub import HfApi
    n = t = 0
    pmin, pmax = float("inf"), 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line.replace("Infinity", "null").replace("NaN", "null")) if line.strip() else None
            if not r or r.get("err"):
                continue
            n += 1
            t += len(r.get("tensors") or {})
            p = r.get("params") or 0
            pmin, pmax = min(pmin, p), max(pmax, p)
    fmt = lambda x: f"{x/1e12:.2f} T" if x >= 1e12 else f"{x/1e9:.1f} B" if x >= 1e9 else f"{x/1e6:.1f} M" if x >= 1e6 else f"{x/1e3:.1f} K"
    card = CARD.format(meta=yaml("corpus"), cite_line=CITE_LINE, citation=CITATION, n=n, t=t, pmin=fmt(pmin), pmax=fmt(pmax), owner=repo.split("/")[0], date=datetime.date.today().isoformat())
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
    import subprocess, tempfile
    pq = os.path.join(tempfile.mkdtemp(), "corpus.parquet")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flat.py"), "corpus", path, pq], check=True)
    api.upload_file(path_or_fileobj=pq, path_in_repo="corpus.parquet", repo_id=repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=path, path_in_repo="corpus.jsonl.gz", repo_id=repo, repo_type="dataset",
                    commit_message="corpus " + datetime.date.today().isoformat())
    print("published https://huggingface.co/datasets/" + repo)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "data", "corpus-510.jsonl.gz"))
