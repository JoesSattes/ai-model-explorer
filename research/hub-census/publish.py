"""Publish a census run as a Hugging Face dataset, with a datasheet.

usage: python publish.py <owner/dataset-name>      (needs HF_TOKEN with write access)
Uploads census.jsonl, sample.parquet, frame_stats.json, sample_info.json and report/ from out/.
"""
import datetime, json, os, sys

CARD = """---
license: mit
pretty_name: Hub Weight Census
tags: [model-anatomy, safetensors, huggingface-hub, census]
size_categories: [1K<n<10K]
---

# Hub Weight Census

What the weights on the Hugging Face Hub say about themselves. For each sampled model repository we read only the
safetensors headers (a median of {hdr} per repo, about {frac} of the weight bytes) and recover the architecture from
tensor names, shapes and dtypes with [ai-model-explorer](https://github.com/JoesSattes/ai-model-explorer), then compare
it with the repo's config.json and with the Hub's own parameter count.

Method: S. Singkul, *Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy*, 2026.

## Sample ({date})

{sample}

Strata: `top` (most downloaded, read in full), `original` releases (no `base_model:` derivative tag, read in full),
`frontier` (originals from leading providers created since {since}), and a random sample of the rest of the Hub stratified
by creation year x download tier x safetensors tag, with `weight` = stratum size / sampled. Weighted estimates describe
the whole Hub; the top/original/frontier strata describe the models people actually use.

## Fields (census.jsonl)

`id`, `status` (ok, gated, no_weights + `no_weights_kind`, legacy_pickle, not_found, timeout, error), popularity
(`downloads` 30-day, `downloads_all_time`, `likes`), `library`, `pipeline_tag`, `origin`, `stratum`, `weight`;
for readable repos: `family_manifest` (weights only), `family_config`, `params`, `stored_bytes`, `hub_params`,
`quant`, `dtypes`, `hp` (weights only), `hp_fused`, `config_hp` (as declared), `components`, `stacks`,
`config_checked` / `config_consistent` with the widths behind any mismatch, `header_bytes`, `weight_bytes`, timings.

## Datasheet

- **Motivation.** Measure how much of the public model ecosystem can be described from weight metadata alone, how
  often no other specification exists, and where the Hub's own metadata is wrong.
- **Collection.** Public Hub API and HTTP range requests on safetensors headers; no weights downloaded, no gated
  content read (gated repos are recorded as such, with their public popularity stats only).
- **Frame.** [cfahlgren1/hub-stats](https://huggingface.co/datasets/cfahlgren1/hub-stats) models snapshot of the run date.
- **Known limits.** `config_hp` is the reference, and configs can themselves be wrong; head counts need a config
  (shapes give heads x head size only); MLX bit-width and group size trade off without a config; repos read in
  pre-1.6 PyTorch pickle format are not parsed.
- **Personal data.** None. Repository ids and authors are public account names.
- **License.** MIT for this dataset; the models described keep their own licenses.

## Summary

{summary}
"""


def main(repo, out="out"):
    from huggingface_hub import HfApi
    info = json.load(open(os.path.join(out, "sample_info.json")))
    summ = open(os.path.join(out, "report", "summary.md")).read()
    hdr = next((l.split("**")[1] for l in summ.splitlines() if l.startswith("- header bytes")), "?")
    frac = next((l.split("**")[1] for l in summ.splitlines() if l.startswith("- fraction")), "?")
    card = CARD.format(hdr=hdr, frac=frac, date=datetime.date.today().isoformat(),
                       sample="\n".join(f"- {k}: {v}" for k, v in info.items()), since=info.get("frontier_since") or "-",
                       summary=summ.split("\n", 2)[-1])
    open(os.path.join(out, "README.md"), "w").write(card)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=out,
                      allow_patterns=["README.md", "census.jsonl", "sample.parquet", "frame_stats.json", "sample_info.json", "report/*"],
                      commit_message="census run " + datetime.date.today().isoformat())
    print("published https://huggingface.co/datasets/" + repo)


if __name__ == "__main__":
    main(sys.argv[1])
