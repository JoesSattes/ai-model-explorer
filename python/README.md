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
| `ai-model-explorer inspect REPO` | readable summary of a Hugging Face repo, URL, `.safetensors` file or folder |
| `ai-model-explorer inspect REPO --json` | the full result as JSON; add `--tensors` for the role of every tensor |
| `ai-model-explorer open [REPO]` | start the web app on your machine (`--lang th` for Thai) |

Gated or private repos: set `HF_TOKEN` or pass `--token`. A mirror: set `HF_ENDPOINT`.

## Python

```python
from ai_model_explorer import inspect, summary

a = inspect("Qwen/Qwen3-8B")
a["family"]                                   # 'decoder'
a["totals"]["params"]                         # 8190735360
a["hyperparameters_with_config"]["kv_heads"]  # 8
a["components"][0]["stacks"][0]["n"]          # 36
print(summary(a))
```

`inspect(model, token=None, config=None, include_tensors=False)` returns a JSON-serializable dict with `family`,
`totals`, `hyperparameters`, `hyperparameters_with_config`, `quantization`, `components` (with their layer stacks
and block types), `edges`, `evidence`, `config_check`, and how many bytes were read. The fields are described in
the [usage guide](https://github.com/JoesSattes/ai-model-explorer/blob/main/docs/usage.md#what-the-result-contains).

## Why it is useful
- **Before downloading**: the anatomy of a 689 GB checkpoint from 12 MB of headers.
- **Quantization and edge planning**: where the parameters sit, which layers keep a KV cache, how each tensor is stored (FP8, NVFP4, GPTQ, AWQ, MXFP4, MLX, bitsandbytes).
- **New architectures**: hybrid attention / state-space / mixture-of-experts stacks, multi-token-prediction layers, encoders feeding an LLM, each labelled with the evidence.
- **Batch analysis and agents**: plain JSON from a single function call.

## How accurate is it?
On 510 architectures from the transformers library, 96.8% of tensors get the right role (82.3% for the best rule
set that reads names only, 95.0% for this method with every name replaced by a random token). Parameter totals
are exact for all 510, including FP8, NVFP4 and GPTQ storage. On 191 popular Hugging Face checkpoints, totals agree
with the Hub's own count for 163 of 170.

## How it runs
The analysis is the same JavaScript that powers the website (`infer.js`, `loaders.js`), run in an embedded QuickJS
engine, so the Python and web results match. Python 3.9+, one dependency (`quickjs`), no PyTorch, transformers or
Node.js. A typical model takes well under a second; the largest manifests (DeepSeek-V3, 92k tensors) take about ten.

## Citation
Sattaya Singkul. 2026. *Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy.*
Preprint, [ResearchGate](https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_Manifest-Only_Recovery_of_Neural_Network_Anatomy).

MIT License © 2026 Sattaya Singkul
