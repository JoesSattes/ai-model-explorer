"""Publish the vocabulary tax tables as a Hugging Face dataset.

usage: python publish.py <owner/dataset-name>      (needs HF_TOKEN with write access; --dry writes the card only)
Uploads data/*.parquet with a datasheet. The tables are computed from the Hub Weight Census manifests, the tokenizer
families and the trimming experiments described in the card.
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

CARD = """---
license: mit
pretty_name: AI Vocabulary Tax
language: [en, th, zh, ja, ko, vi, id, ms, tl, fr, de, es, pt, it, nl, pl, cs, ru, uk, tr, ar, fa, ur, hi, bn, ta, te, mr, gu, kn, ml, pa, he, el, km, lo, my, am, sw, ka, hy]
annotations_creators: [machine-generated]
source_datasets: [original]
task_categories: [tabular-regression]
tags: [tokenizer, vocabulary-trimming, model-compression, edge-ai, on-device, multilingual, llm, safetensors, huggingface-hub]
size_categories: [100K<n<1M]
configs:
- config_name: models
  data_files: models.parquet
  default: true
- config_name: model_language
  data_files: model_language.parquet
- config_name: tokenizers
  data_files: tokenizers.parquet
- config_name: quality
  data_files: quality.parquet
- config_name: certificate
  data_files: certificate.parquet
- config_name: merge_closure
  data_files: merge_closure.parquet
---

# AI Vocabulary Tax

Open LLMs and speech models ship token tables (input embedding, untied output head, per-layer embeddings) for every
language their vocabulary covers, and every user downloads them. The **vocabulary tax** is the bytes in rows a
deployment never uses. This dataset gives it for **{n_models:,} models** of the Hugging Face Hub and **{n_langs} languages**,
read from the safetensors headers and the tokenizer, with no weights downloaded, and the experiments that show how
much can be trimmed safely.

Try it on the web: [census page, Vocabulary tax](https://joessattes.github.io/ai-model-explorer/census/#vtax) ·
trim your own model with your own text: [AI Model Explorer](https://joessattes.github.io/ai-model-explorer/) (Where to cut) ·
`pip install ai-model-explorer` then `ai-model-explorer tax REPO --lang th --corpus my_text.txt`.

## Two keep rules

- **script**: keep the pieces whose characters are in the deployment's scripts (plus Latin, Common, Inherited, format
  characters, special tokens, byte fallbacks). Text in those scripts is tokenized exactly as before: `resplit_rate` = 0.
- **text**: keep only the pieces that the user's own text uses (`train_lines` lines of Universal Dependencies treebank
  text; Thai adds Wisesight posts), closed under BPE merges so every kept piece stays reachable. `resplit_rate` is the
  real share of tokens of 1,000 held-out lines that the trimmed tokenizer splits differently.

Every language set includes English (Latin), because names, numbers, code and loanwords appear in Latin letters in all of them.

## Tables

| config | rows | what |
|---|---:|---|
| `models` | {n_models:,} | one row per model: tokenizer family, vocabulary, bytes in token tables, total stored bytes, parameters, pipeline tag, 30-day downloads, and the tax for Thai + English under both rules |
| `model_language` | {n_ml:,} | model x language x rule: rows, bytes and share of the model removed, and the re-split rate |
| `tokenizers` | {n_tok:,} | tokenizer family x language x rule (and number of text lines): share of rows removed and re-split rate |
| `quality` | {n_q} | six models (Gemma 3 270M and 1B, Qwen3 0.6B and 1.7B, Llama 3.2 1B, SmolLM3 3B) x 11 languages x rule: bits per character, greedy continuations and Belebele accuracy before and after trimming |
| `certificate` | {n_c} | trim certificate (Good-Turing estimate from the trimming text) and a 20% hold-out estimate against the real re-split rate |
| `merge_closure` | {n_n} | nominal re-split rate (tokens in removed rows) and real rate with and without merge closure |

## How it was made

Token tables and stored sizes come from the safetensors manifests of the
[Hub Weight Census](https://huggingface.co/datasets/{owner}/hub-weight-census). Each model is assigned to one of the
reference tokenizer families by model type and vocabulary size; rows past the end of the tokenizer are kept. Shares in
`model_language` are the family's removable share scaled to the model's tables, so a model that ships a custom tokenizer
with a standard size can be misassigned; the package and the website read a model's own tokenizer when you run them on it.
The `quality` table evaluates trimmed models exactly without editing weights: the trimmed tokenizer's ids are mapped to
the original rows and every softmax runs over the kept rows only.

Main findings: the tax is regressive (for Thai and English, a median 4.4% of model bytes below 0.5B parameters, 2.1%
above 8B under the script rule; 29% below 0.5B with 15,000 lines of text); without merge closure the real re-split rate
is a median 1.9 times the nominal one; the certificate predicts the real rate (correlation 0.996); and where it reports
under 1%, bits per character of the trimmed models rise by at most 2.0%.

## Citation

S. Singkul, *The Vocabulary Tax: Safely Trimming Unused Tokens from LLMs for On-Device Use*, 2026 (preprint in preparation).
Builds on S. Singkul, *Reading the Architecture off the Weights: Manifest-Only Model Anatomy and a Census of 12,974
Hugging Face Repositories*, 2026 ([ResearchGate](https://www.researchgate.net/publication/417293353)).

Code: https://github.com/JoesSattes/ai-model-explorer (MIT). Treebank text is not redistributed; only counts derived from it.
"""


def card(owner):
    n = lambda f: len(pd.read_parquet(os.path.join(D, f + ".parquet")))
    ml = pd.read_parquet(os.path.join(D, "model_language.parquet"), columns=["language"])
    return CARD.format(owner=owner, n_models=n("models"), n_langs=ml.language.nunique(), n_ml=len(ml), n_tok=n("tokenizers"),
                       n_q=n("quality"), n_c=n("certificate"), n_n=n("merge_closure"))


def main(repo):
    owner = repo.split("/")[0] if repo != "--dry" else "JoesSattes"
    text = card(owner)
    open(os.path.join(D, "README.md"), "w", encoding="utf-8").write(text)
    if repo == "--dry":
        print(text[:400]); return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=D, commit_message="vocabulary tax tables from GitHub")
    print("dataset https://huggingface.co/datasets/" + repo)


if __name__ == "__main__":
    main(sys.argv[1])
