"""Publish a census run as a Hugging Face dataset, with a datasheet.

usage: python publish.py <owner/dataset-name>      (needs HF_TOKEN with write access)
Uploads census.parquet, census.jsonl, census_manifests.jsonl.gz, roles.json, sample.parquet, frame_stats.json,
sample_info.json, report/ and insights/ from out/.
"""
import datetime, json, os, sys

from card_meta import yaml, CITATION, CITE_LINE

CARD = """---
license: mit
pretty_name: Hub Weight Census
{meta}
size_categories: [{size}]
configs:
- config_name: default
  data_files: census.parquet
---

# Hub Weight Census

What the weights on the Hugging Face Hub say about themselves. For each sampled model repository we read only the
safetensors headers (a median of {hdr} per repo, about {frac} of the weight bytes) and recover the architecture from
tensor names, shapes and dtypes with [ai-model-explorer](https://github.com/JoesSattes/ai-model-explorer), then compare
it with the repo's config.json and with the Hub's own parameter count.

Method: S. Singkul, *Reading the Architecture off the Weights: Manifest-Only Model Anatomy and a Census of 12,974 Hugging Face Repositories*, 2026
([paper](https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_Manifest-Only_Recovery_of_Neural_Network_Anatomy)).
Explore it on the web: https://joessattes.github.io/ai-model-explorer/census/

{cite_line}

## Sample ({date})

{sample}

Strata: `top` (most downloaded, read in full), `original` releases (no `base_model:` derivative tag, read in full),
`frontier` (originals from leading providers created since {since}), and a random sample of the rest of the Hub stratified
by creation year x download tier x safetensors tag, with `weight` = stratum size / sampled. Weighted estimates describe
the whole Hub; the top/original/frontier strata describe the most downloaded models.

## Files

- `census.parquet`: one flat row per repo (shown in the viewer above): status, popularity, family, parameters, the Hub's
  count, storage format, width, layers, heads, KV heads, experts, vocabulary, KV bytes per token, and the values config.json declares.
- `census.jsonl`: the full records, fields below.
- `census_manifests.jsonl.gz`: the raw manifest and config.json of every readable repo (see below).
- `roles.json`: per-tensor role accuracy on repos whose model type is in the companion corpus.
- `sample.parquet`: the sampled repos with stratum and weight; `sample_info.json`: sample sizes and settings.
- `frame_stats.json`: statistics of the full Hub frame the sample was drawn from.
- `report/`: summary tables and charts; `insights/`: ecosystem findings (`ecosystem.json`, `insights.md`) and figures.

## Fields (census.jsonl)

`id`, `status` (ok, gated, no_weights + `no_weights_kind`, legacy_pickle, not_found, timeout, error), popularity
(`downloads` 30-day, `downloads_all_time`, `likes`), `library`, `pipeline_tag`, `origin`, `stratum`, `weight`;
for readable repos: `family_manifest` (weights only), `family_config`, `params`, `stored_bytes`, `hub_params`,
`quant`, `dtypes`, `hp` (weights only), `hp_fused`, `config_hp` (as declared), `components`, `stacks`,
`config_checked` / `config_consistent` with the widths behind any mismatch, `header_bytes`, `weight_bytes`, timings.

`census_manifests.jsonl.gz`: for every readable repo, the raw manifest (name, dtype, shape of every tensor) and its
config.json, so any field can be recomputed (`rescore.py`). `roles.json`: per-tensor role accuracy on repos whose model
type is in the companion corpus ([manifest-anatomy-corpus]({corpus_url})), with labels carried over by matching tensor names after layer indices are masked.

## Datasheet

- **Motivation.** Measure how much of the public model ecosystem can be described from weight metadata alone, how
  often no other specification exists, and where the Hub's own metadata is wrong.
- **Collection.** Public Hub API and HTTP range requests on safetensors headers; no weights downloaded, no gated
  content read (gated repos are recorded as such, with their public popularity stats only).
- **Frame.** [cfahlgren1/hub-stats](https://huggingface.co/datasets/cfahlgren1/hub-stats) models snapshot from the run date.
- **Known limits.** `config_hp` is the reference, and configs can themselves be wrong; head counts need a config
  (shapes give heads x head size only); MLX bit-width and group size trade off without a config; repos stored only in
  the pre-1.6 PyTorch pickle format are not parsed.
- **Personal data.** None. Repository ids and authors are public account names.
- **License.** MIT for this dataset; the models described keep their own licenses.

## Findings

{findings}

![Architecture by release year, with 95% intervals](insights/eco_adoption.png)
![KV cache per token per billion parameters, and how sparse mixtures of experts became](insights/eco_memory.png)
![Share of repositories and of downloads by relation to a parent model](insights/eco_economy.png)
![How weights are stored, by year](insights/eco_storage.png)
![Smallest device class each original language model fits at 4-bit, 8k context](insights/eco_deploy.png)
![The Hub's parameter count by storage format](insights/storage.png)

Every number above, with its 95% interval and trend test, is in `insights/ecosystem.json` (written by `ecosystem.py`):
Wilson intervals for shares, bootstrap intervals for medians, Cochran-Armitage and logistic trends over release year,
Spearman for monotone trends, log-log slopes for shape scaling, Mann-Whitney for two groups.

## What it can be used for

- **Architecture trends** with evidence from the weights themselves: mixture of experts, hybrid state-space stacks,
  grouped-query and latent attention, head size, vocabulary, by release year and provider.
- **Metadata quality**: repos whose declared config disagrees with the weights, and formats where the Hub's own
  parameter count is not the model's parameter count.
- **Deployment planning at scale**: stored precision, KV-cache bytes per token and parameter counts for thousands of
  checkpoints without downloading one.
- **Benchmarking** architecture-recovery and model-search tools against `config_hp` and `family_config`.
- **A snapshot to compare against**: re-run the census later (the code and the frozen rules are public) to see what changed.

## Summary tables

{summary}

{citation}"""


def main(repo, out="out"):
    from huggingface_hub import HfApi
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flat.py"), "census",
                    os.path.join(out, "census.jsonl"), os.path.join(out, "census.parquet")], check=True)
    info = json.load(open(os.path.join(out, "sample_info.json")))
    summ = open(os.path.join(out, "report", "summary.md")).read()
    hdr = next((l.split("**")[1] for l in summ.splitlines() if l.startswith("- header bytes")), "?")
    frac = next((l.split("**")[1] for l in summ.splitlines() if l.startswith("- fraction")), "?")
    ins = os.path.join(out, "insights", "insights.md")
    findings = open(ins).read().split("\n", 2)[-1] if os.path.exists(ins) else "(run insights.py)"
    n = info.get("sample", 0)
    size = "1K<n<10K" if n < 10_000 else "10K<n<100K" if n < 100_000 else "100K<n<1M"
    owner = repo.split("/")[0]
    card = CARD.format(meta=yaml("census"), cite_line=CITE_LINE, citation=CITATION, corpus_url=f"https://huggingface.co/datasets/{owner}/manifest-anatomy-corpus", size=size, hdr=hdr, frac=frac, date=datetime.date.today().isoformat(), findings=findings,
                       sample="\n".join(f"- {k}: {v}" for k, v in info.items()), since=info.get("frontier_since") or "-",
                       summary=summ.split("\n", 2)[-1])
    open(os.path.join(out, "README.md"), "w").write(card)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=out,
                      allow_patterns=["README.md", "census.parquet", "census.jsonl", "census_manifests.jsonl.gz", "roles.json", "sample.parquet", "frame_stats.json",
                                      "sample_info.json", "report/*", "insights/*.json", "insights/*.md", "insights/*.png"],
                      commit_message="census run " + datetime.date.today().isoformat())
    print("published https://huggingface.co/datasets/" + repo)


if __name__ == "__main__":
    main(sys.argv[1])
