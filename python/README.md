<p align="center">
  <img src="https://raw.githubusercontent.com/JoesSattes/ai-model-explorer/main/assets/logo-180.png" width="96" alt="AI Model Explorer logo">
</p>

<h1 align="center">AI Model Explorer</h1>

<p align="center"><b>See inside any AI model on Hugging Face, part by part, without downloading the weights.</b></p>

<p align="center">
  <a href="https://joessattes.github.io/ai-model-explorer/en/">Web app</a> ·
  <a href="https://github.com/JoesSattes/ai-model-explorer">GitHub</a> ·
  <a href="https://github.com/JoesSattes/ai-model-explorer/blob/main/docs/usage.md">Docs</a> ·
  <a href="https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_Manifest-Only_Recovery_of_Neural_Network_Anatomy">Paper</a>
</p>

`ai-model-explorer` reads the safetensors header of a model (tensor names, shapes and types, usually a few tens of
kilobytes) and works out its architecture: components, repeated layers and what each one does, how the parts
connect, the exact parameter count and how every tensor is stored. Nothing else is downloaded.

```bash
pip install ai-model-explorer
```

## Command line

```bash
ai-model-explorer inspect deepseek-ai/DeepSeek-V3
```
```
deepseek-ai/DeepSeek-V3
  family decoder · 684.49B parameters · 688.6 GB of weights · stored as fp8
  read 11.9 MB from 163 file(s), 91991 tensors
  width 7168 · layers 61 · heads 128 · kv heads 128 · head dim 128 · ffn 18432 · experts 256 · active experts 8 · vocab 129280
  components:
    (root) (backbone) · d=7168
      model.layers: 62 blocks · Transformer ×3, Attn+MoE ×58, Attn+MoE ×1 (last one is a multi-token-prediction layer)
  config.json: agrees with the weights
```

| Command | What it does |
|---|---|
| `ai-model-explorer inspect MODEL` | readable summary of a Hugging Face repo, URL, `.safetensors` file or folder (`--json` for everything, `--tensors` for every tensor's role) |
| `ai-model-explorer insights MODEL` | findings in plain language: active parameters per token, where the parameters sit, KV-cache cost, tied embeddings, hybrid layers |
| `ai-model-explorer memory MODEL --context 32768 --weight-bits 4` | memory for the weights at any precision plus the KV cache |
| `ai-model-explorer compare A B C` | side-by-side table (Markdown, `--csv` or `--json`) |
| `ai-model-explorer plot MODEL -o model.png` | the anatomy diagram as PNG, SVG or PDF (`--theme dark`) |
| `ai-model-explorer report MODEL -o model.html` | one self-contained HTML report: diagram, insights, tables, memory |
| `ai-model-explorer open [MODEL]` | the web app on your machine (`--lang th` for Thai) |

![Diagram drawn by ai-model-explorer plot for DeepSeek-V3](https://raw.githubusercontent.com/JoesSattes/ai-model-explorer/main/assets/example-deepseek-v3.png)

Gated or private repos: set `HF_TOKEN` or pass `--token`. A mirror: set `HF_ENDPOINT`.

## Python

```python
from ai_model_explorer import explore, compare, compare_table

m = explore("deepseek-ai/DeepSeek-V3")
m.family, m.params, m.active_params   # 'decoder', 684489845504, 37552297472
m.hp["heads"], m.hp["kv_heads"]       # 128, 128 (from config.json, checked against the weights)
m.breakdown("kind")                   # parameters, share and bytes per part type
m.kv_cache_bytes_per_token()          # 70272 (latent attention)
m.memory(context=32768, weight_bits=4)
for line in m.insights():
    print("-", line)
m.plot("deepseek.png")                # matplotlib Figure; .png, .svg or .pdf
m.report("deepseek.html")

print(compare_table(compare(["Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B-GPTQ-Int4"])))
```

`explore()` returns a `Model`; `inspect()` returns the raw dict it wraps. Every field and method is described in the
[usage guide](https://github.com/JoesSattes/ai-model-explorer/blob/main/docs/usage.md).

## Why it is useful
- **Before downloading**: the anatomy of a 689 GB checkpoint from 12 MB of headers.
- **Quantization and edge planning**: where the parameters sit, which layers keep a KV cache, how each tensor is stored (FP8, NVFP4, GPTQ, AWQ, MXFP4, MLX, bitsandbytes).
- **New architectures**: hybrid attention / state-space / mixture-of-experts stacks, multi-token-prediction layers, encoders feeding an LLM, each labelled with the evidence.
- **Comparisons and reports**: a table across many models, diagrams for slides and papers, an HTML report to share.
- **Batch analysis and agents**: plain data from a single function call.

## What is new
Reading the header is standard practice; tools such as hf-mem estimate memory from it, and HF Viewer draws graphs from a model's config and code. This package interprets the header itself: components, block types and the role of every tensor without config or code (95% of roles still right with every tensor name hidden), exact logical parameter counts under quantized storage, KV cache from the weights including latent attention, and a check of `config.json` against the weights.

## How accurate is it?
On 510 architectures from the transformers library, 96.8% of tensors get the right role (82.3% for the best rule
set that reads names only, 95.0% for this method with every name replaced by a random token). Parameter totals
are exact for all 510, including FP8, NVFP4 and GPTQ storage. On 191 popular Hugging Face checkpoints, totals agree
with the Hub's own count for 163 of 170.

## How it runs
The analysis is the same JavaScript that powers the website (`infer.js`, `loaders.js`), run in an embedded QuickJS
engine, so the Python and web results match. Python 3.9+, two dependencies (`quickjs`, and `matplotlib` for diagrams), no PyTorch, transformers or
Node.js. A typical model takes well under a second; the largest manifests (DeepSeek-V3, 92k tensors) take about ten.

## Citation
Sattaya Singkul. 2026. *Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy.*
Preprint, [ResearchGate](https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_Manifest-Only_Recovery_of_Neural_Network_Anatomy).

MIT License © 2026 Sattaya Singkul
