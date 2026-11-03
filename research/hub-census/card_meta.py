"""Metadata and citation shared by the dataset cards (publish.py, publish_corpus.py, card_update.py).

Task categories name what the data supports: the census is a table of architecture fields per repo (predict family,
parameters or format from the header); the corpus labels every tensor name with its role (classify each name).
"""

PAPER = "Reading the Architecture off the Weights: Manifest-Only Model Anatomy and a Census of 12,974 Hugging Face Repositories"
PAPER_URL = ("https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_"
             "Manifest-Only_Recovery_of_Neural_Network_Anatomy")
PDF_URL = "https://github.com/JoesSattes/ai-model-explorer/blob/main/paper/manifest-anatomy-singkul-2026.pdf"

COMMON = {
    "language": "[en]",
    "annotations_creators": "[machine-generated]",
    "source_datasets": "[original]",
}
META = {
    "census": dict(COMMON, **{
        "task_categories": "[tabular-classification, tabular-regression]",
        "tags": "[model-architecture, model-metadata, safetensors, huggingface-hub, llm, mixture-of-experts, "
                "quantization, kv-cache, ecosystem-analysis]",
    }),
    "corpus": dict(COMMON, **{
        "task_categories": "[text-classification, tabular-classification]",
        "tags": "[model-architecture, safetensors, transformers, pytorch, tensor-role-labeling, ground-truth, benchmark]",
    }),
}

CITE_LINE = f"**Citation.** If you use this dataset, please cite the paper ([PDF]({PDF_URL}), [ResearchGate]({PAPER_URL})); BibTeX under [Citation](#citation)."

CITATION = f"""## Citation

If you use this dataset, please cite:

```bibtex
@misc{{singkul2026manifest,
  title  = {{{PAPER}}},
  author = {{Singkul, Sattaya}},
  year   = {{2026}},
  note   = {{Preprint}},
  url    = {{{PAPER_URL}}}
}}
```

S. Singkul, *{PAPER}*, 2026. [PDF]({PDF_URL}) · [ResearchGate]({PAPER_URL}) · code: https://github.com/JoesSattes/ai-model-explorer
"""


def yaml(kind):
    return "\n".join(f"{k}: {v}" for k, v in META[kind].items())

