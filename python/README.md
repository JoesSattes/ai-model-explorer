# AI Model Explorer

See any Hugging Face model's architecture from its safetensors header, without downloading the weights.

```bash
pip install ai-model-explorer
```

## Command line

```bash
ai-model-explorer inspect Qwen/Qwen3-8B
```
```
Qwen/Qwen3-8B
  family decoder · 8.19B parameters · 16.4 GB of weights
  read 46.1 KB from 5 file(s), 399 tensors
  width 4096 · layers 36 · heads 32 · kv heads 8 · head dim 128 · ffn 12288 · vocab 151936
  components:
    (root) (backbone) · d=4096
      model.layers: 36 blocks · Transformer
  config.json: agrees with the weights
```

- `ai-model-explorer inspect REPO --json` prints the full result: components, layer stacks, block types, edges between components, hyperparameters, quantization and the config check. Add `--tensors` to get the role of every tensor.
- `ai-model-explorer inspect ./model.safetensors` or a folder works offline.
- `ai-model-explorer open [REPO]` starts the web app on your machine and opens it in the browser (`--lang th` for Thai).
- Gated or private repos: set `HF_TOKEN` or pass `--token`. A mirror can be set with `HF_ENDPOINT`.

## Python

```python
from ai_model_explorer import inspect, summary

a = inspect("deepseek-ai/DeepSeek-V3")
print(a["totals"]["params"], a["hyperparameters"]["experts"])   # 684489845504 256
print(summary(a))
```

Only the header bytes of each weight file are requested (a median of 79 KB per repo in our benchmark of 191 popular checkpoints), plus `config.json`. The analysis is the same JavaScript code that runs the website, executed in an embedded QuickJS engine, so results match the web app exactly. No Node.js, PyTorch or transformers needed. Typical models take well under a second; the very largest manifests (DeepSeek-V3, 92k tensors) take about ten seconds.

## How accurate is it?
On 510 architectures from the transformers library, 96.8% of tensors get the right role (95.0% with every name replaced by a random token), and parameter totals are exact for all 510, including FP8, NVFP4 and GPTQ storage. Details are in the paper.

## Links
- Web app: https://joessattes.github.io/ai-model-explorer/en/
- Source: https://github.com/JoesSattes/ai-model-explorer
- Paper: Sattaya Singkul, *Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy*, 2026. https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_Manifest-Only_Recovery_of_Neural_Network_Anatomy

MIT License © 2026 Sattaya Singkul
