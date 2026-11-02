# Using AI Model Explorer

Three ways in: the website, the command line, and Python. All three run the same recovery code, so they give the same answer.

- [Website](#website)
- [Command line](#command-line)
- [Python](#python)
- [What the result contains](#what-the-result-contains)
- [Gated repos, mirrors and offline use](#gated-repos-mirrors-and-offline-use)
- [Limits](#limits)

## Website

Open https://joessattes.github.io/ai-model-explorer/en/ (Thai: https://joessattes.github.io/ai-model-explorer/).

1. Type a Hugging Face repo id such as `Qwen/Qwen3-8B` and press **Load**, or drop a `.safetensors`, `.tflite` or `config.json` file on the page.
2. **Anatomy** shows the components in data order (for example vision encoder → projector → language model). Each layer stack is a strip coloured by block type, and every value carries its source: measured from the weights, read from config, or estimated.
3. **Big picture** numbers the parts from input to output and plays a sample through them, showing what each part receives and returns.
4. **Layer map** draws every layer to scale. Lenses switch the view between weights, compute, TFLite ops, precision and fragility.
5. **Report** lists the findings and gives a Markdown table you can copy.

Share a model with a link: `https://joessattes.github.io/ai-model-explorer/en/#hf=deepseek-ai/DeepSeek-V3` opens the page and loads that repo.
Models you open are kept in your browser, so the next visit doesn't load them again.

## Command line

```bash
pip install ai-model-explorer
```

### `inspect`: print the anatomy

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

| Option | Effect |
|---|---|
| `MODEL` | repo id (`owner/name`), a `huggingface.co` URL, a `.safetensors` file, or a folder of them |
| `--json` | print the full result as JSON (see [What the result contains](#what-the-result-contains)) |
| `--tensors` | with `--json`: add the role of every tensor |
| `--token TOKEN` | Hugging Face token for gated or private repos (default: `$HF_TOKEN`) |
| `--config PATH` | use this `config.json` instead of the repo's own |

Exit code is 0 on success and 1 on a readable error (gated repo, repo not found, no network).

### `open`: run the website on your machine

```bash
ai-model-explorer open                    # opens the English page
ai-model-explorer open Qwen/Qwen3-8B      # and loads this repo right away
ai-model-explorer open --lang th          # Thai page
ai-model-explorer open --port 8080 --no-browser
```

The page is served from the installed package on `127.0.0.1`. Files you drop are read locally. Hugging Face repos are fetched by your browser, as on the public site.

## Python

```python
from ai_model_explorer import inspect, summary

a = inspect("Qwen/Qwen3-8B")
print(summary(a))                          # the same text as the CLI
print(a["totals"]["params"])               # 8190735360
print(a["hyperparameters_with_config"])    # width, layers, heads, kv heads, ffn, vocab, ...

local = inspect("./checkpoints/my-model")  # a folder of .safetensors, read offline
```

`inspect(model, token=None, config=None, include_tensors=False) -> dict`

| Argument | Meaning |
|---|---|
| `model` | repo id, `huggingface.co` URL, `.safetensors` path or folder |
| `token` | Hugging Face token (default: `$HF_TOKEN`, then `$HUGGING_FACE_HUB_TOKEN`) |
| `config` | a `config.json` dict to use instead of the repo's own |
| `include_tensors` | add `tensors`, the role of every tensor |

Errors are standard exceptions: `PermissionError` (gated or private), `FileNotFoundError` (repo or file not found), `ConnectionError` (Hugging Face unreachable), `ValueError` (input not understood).

A batch over many repos is a plain loop. Each call reads only headers, so a few hundred models take minutes, not hours:

```python
rows = []
for repo in ["Qwen/Qwen3-8B", "meta-llama/Llama-3.2-1B", "openai/whisper-large-v3"]:
    a = inspect(repo)
    hp = a["hyperparameters_with_config"]
    rows.append((repo, a["family"], a["totals"]["params"], hp.get("layers"), hp.get("heads"), hp.get("kv_heads")))
```

## What the result contains

`inspect()` returns a JSON-serializable dict, and `--json` prints the same thing.

| Key | Content |
|---|---|
| `source` | where the manifest came from: `{"kind": "huggingface", "repo", "weight_bytes", "downloads", "hub_params"}` or `{"kind": "local", "path"}` |
| `files_read`, `header_bytes`, `tensor_count` | how much was read: weight files whose header was fetched, bytes, tensors |
| `family` | `decoder`, `encoder`, `encoder-decoder` (cross-attention), `prefix-lm` (an encoder's output fed into a decoder-only LLM, as in most vision-language and audio LLMs), `transducer`, `diffusion`, or `unknown` |
| `totals` | `params` (logical parameters, quantized storage unpacked) and `bytes` (as stored) |
| `hyperparameters` | read from the weights alone: `d` (width), `layers`, `heads`, `kv_heads`, `head_dim`, `ffn`, `experts`, `vocab`, `tie` (tied embedding), `attn_layers`, `ssm_layers`, `kv_bytes_per_token_bf16`, … |
| `hyperparameters_with_config` | the same, with heads, KV heads, head size and active experts from `config.json` when the config agrees with the weights |
| `quantization` | storage formats found and how many tensors use each, e.g. `{"fp8": 45808}` or `{"gptq": 18624}` |
| `components` | list of parts. Each has `id`, `role` (`backbone`, `encoder`, `adapter`, `prediction` and `joint` for transducers, `aux`), `d`, `params`, `bytes`, `modality`, and `stacks` |
| `components[].stacks` | repeated layers: `path`, `n` (blocks), `types` (block types with `label`, `count`, `d` and `regions`), `mtp` (true when the last block is a multi-token-prediction layer) |
| `edges` | how components connect: `{"from", "to", "via"}`, e.g. encoder → decoder via cross-attention |
| `evidence` | sentences explaining non-obvious decisions, e.g. why a layer was read as a multi-token-prediction head |
| `config_check` | per component: config values for `heads`, `kv_heads`, `head_dim`, `topk`, and whether they are `consistent` with the weights |
| `tensors` | only with `include_tensors` / `--tensors`: for every tensor, `comp`, `stack`, `block`, `kind` (`attn`, `ffn`, `moe`, `ssm`, `conv`, `norm`, `embed`, `head`, …), `region` and `role` |

## Gated repos, mirrors and offline use

- **Gated or private repos** (Llama, Gemma, some FLUX models): accept the license on the model page, create a read token at https://huggingface.co/settings/tokens, then `export HF_TOKEN=hf_...` or pass `--token`. The token is sent only to Hugging Face.
- **Mirror**: `export HF_ENDPOINT=https://your-mirror` to read through a Hugging Face mirror.
- **Offline**: point `inspect` at local `.safetensors` files. Only the headers are read, so this is instant even for large checkpoints.

## Limits

- Heads, KV heads and head size can't always be told apart from shapes alone (a projection of width 4096 can be 32×128 or 64×64). With `config.json` they're taken from the config after checking it against the weights; without it they're estimates and marked as such.
- Very large manifests take longer in Python: DeepSeek-V3 (92k tensors) takes about ten seconds, a typical 7B model well under one.
- Old pickle-only checkpoints without a zip index can't be read without downloading them.
- On Linux ARM there's no prebuilt wheel of the `quickjs` dependency yet, so pip compiles it and needs a C compiler.
