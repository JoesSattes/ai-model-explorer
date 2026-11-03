"""Publish the explorer (and the census page) as a static Hugging Face Space.

usage: python space.py <owner/space-name> <census-dataset> <corpus-dataset>    (needs HF_TOKEN with write access)
The Space serves the same files as GitHub Pages; models are read from the Hub by the visitor's browser.
"""
import os, shutil, sys, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FILES = ["index.html", "en/index.html", "census/index.html", "census/data/insights.json", "census/data/models.json", "census/data/community.json", "census/data/board.json",
         "og.png", "LICENSE", "assets/logo.svg", "assets/logo-32.png", "assets/logo-180.png", "assets/logo-512.png"]

README = """---
title: AI Model Explorer
emoji: 🔍
colorFrom: indigo
colorTo: pink
sdk: static
app_file: en/index.html
pinned: true
license: mit
short_description: Any model's architecture from its safetensors header
datasets:
- {census}
- {corpus}
tags:
- model-architecture
- visualization
- safetensors
- quantization
- edge-ai
---

# AI Model Explorer

Type any Hugging Face model id and see what it is made of: components, layer stacks, block types, the role of every
tensor, parameters and storage format, KV cache per token, and whether `config.json` agrees with the weights. Only the
safetensors headers are read (a few kilobytes to a few megabytes), never the weights.

- **Hub census**: what thousands of popular models are made of, by release year: [`/census/`](census/index.html) ·
  dataset [{census}](https://huggingface.co/datasets/{census})
- **Labelled corpus** of 510 architectures with per-tensor ground truth: [{corpus}](https://huggingface.co/datasets/{corpus})
- **Python**: `pip install ai-model-explorer` · **Code**: https://github.com/JoesSattes/ai-model-explorer
- **Paper**: S. Singkul, *Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy*, 2026
  ([ResearchGate](https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_Manifest-Only_Recovery_of_Neural_Network_Anatomy))

Also at https://joessattes.github.io/ai-model-explorer/ (Thai and English).
"""


def main(repo, census, corpus):
    from huggingface_hub import HfApi
    tmp = tempfile.mkdtemp()
    for f in FILES:
        src = os.path.join(ROOT, f)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(os.path.join(tmp, f)), exist_ok=True)
            shutil.copy(src, os.path.join(tmp, f))
    open(os.path.join(tmp, "README.md"), "w", encoding="utf-8").write(README.format(census=census, corpus=corpus))
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=tmp, commit_message="explorer from GitHub")
    print("space https://huggingface.co/spaces/" + repo)


if __name__ == "__main__":
    main(*sys.argv[1:4])
