# Using AI Model Explorer

AI Model Explorer has three interfaces: the website, the command line, and Python. All three run the same recovery code, so they give the same answer.

- [Website](#website)
- [Command line](#command-line)
- [Python](#python)
- [What the result contains](#what-the-result-contains)
- [Gated repos, mirrors and offline use](#gated-repos-mirrors-and-offline-use)
- [Limits](#limits)

## Website

Open https://joessattes.github.io/ai-model-explorer/en/ (Thai: https://joessattes.github.io/ai-model-explorer/).

1. Type a Hugging Face repo id such as `Qwen/Qwen3-8B` and press **Load**, or drop a `.safetensors`, `.tflite` or `config.json` file on the page.
2. **Anatomy** shows the components in data order (for example vision encoder → projector → language model). Each layer stack is a strip colored by block type, and every value carries its source: measured from the weights, read from config, or estimated.
3. **Big picture** numbers the parts from input to output and plays a sample through them, showing what each part receives and returns.
4. **Layer map** draws every layer to scale. Lenses switch the view between weights, compute, TFLite ops, precision and fragility.
5. **Report** lists the findings and gives a Markdown table you can copy.

Share a model with a link: `https://joessattes.github.io/ai-model-explorer/en/#hf=deepseek-ai/DeepSeek-V3` opens the page and loads that repo.
Models you open are kept in your browser, so the next visit does not load them again.

## Command line

```bash
pip install ai-model-explorer
```

Running `ai-model-explorer` with no command starts the local web app (the same as `open`). Use `ai-model-explorer --help` for the list of commands.

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

`layers` counts the main stack (61); the 62nd block is the multi-token-prediction layer.

| Option | Effect |
|---|---|
| `MODEL` | repo id (`owner/name`), a `huggingface.co` URL, a `.safetensors` file, or a folder of them |
| `--json` | print the full result as JSON (see [What the result contains](#what-the-result-contains)) |
| `--tensors` | with `--json`: add the role of every tensor |
| `--token TOKEN` | Hugging Face token for gated or private repos (default: `$HF_TOKEN`) |
| `--config PATH` | use this `config.json` instead of the repo's own |

Exit code is 0 on success and 1 on a readable error (gated repo, repo not found, no network).

### `insights`, `memory`, `compare`, `plot`, `report`

```bash
ai-model-explorer insights deepseek-ai/DeepSeek-V3
```
```
deepseek-ai/DeepSeek-V3
  - 684.49B parameters, 688.6 GB of weights stored as fp8. At 16-bit that would be 1.4 TB, at 8-bit 684.5 GB, at 4-bit 342.2 GB.
  - Includes a multi-token-prediction layer of 13.46B parameters, used for speculative decoding; the main model is 671.03B.
  - Mixture of experts: 256 experts, 8 used per token, so about 37.55B parameters (5.5%) do the work for each token, while all 684.49B must sit in memory.
  - Where the parameters sit: mixture of experts 98%, attention 2%.
  - KV cache (compressed latent attention): 70.3 KB per token at 16-bit, so 2.3 GB for one 32k-token sequence and 9.2 GB at 128k.
  - config.json agrees with the weights.
```

```bash
ai-model-explorer memory Qwen/Qwen3-8B --context 32768 --weight-bits 4
```
```
Qwen/Qwen3-8B
  weights (4-bit):  4.1 GB
  KV cache (16-bit, 1 × 32,768 tokens):  4.8 GB
  total:            8.9 GB  (activations and runtime overhead not included)
```

```bash
ai-model-explorer compare Qwen/Qwen3-8B Qwen/Qwen3-30B-A3B-GPTQ-Int4 openai/whisper-large-v3
```
| Model | Family | Params | Active | Stored | Storage | Layers | Width | Heads | KV heads | Experts | Vocab | KV/token (16-bit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-8B | decoder | 8.19B | 8.19B | 16.4 GB | unquantized | 36 | 4096 | 32 | 8 | – | 151936 | 147.5 KB |
| Qwen/Qwen3-30B-A3B-GPTQ-Int4 | decoder | 30.53B | 3.35B | 16.9 GB | gptq | 48 | 2048 | 32 | 4 | 128 | 151936 | 98.3 KB |
| openai/whisper-large-v3 | encoder-decoder | 1.54B | 1.54B | 3.1 GB | unquantized | 32 | 1280 | 20 | 20 | – | 51866 | 163.8 KB |

`--csv` or `--json` for scripts.

```bash
ai-model-explorer plot deepseek-ai/DeepSeek-V3 -o deepseek.png          # also .svg or .pdf, --theme dark
ai-model-explorer report deepseek-ai/DeepSeek-V3 -o deepseek.html       # diagram, insights, tables, memory
```

The diagram puts the components in data order, draws every layer stack as a strip of blocks colored by block type (a faded block marks a multi-token-prediction layer), and ends with a bar of where the parameters sit. The report is one HTML file with no external dependencies, so it can be attached or shared as is.

| Option | Commands | Effect |
|---|---|---|
| `--context N` | memory, fits | tokens per sequence (default 8192) |
| `--batch N` | memory, fits | sequences at once (default 1) |
| `--weight-bits B` | memory, fits | weight precision in bits (default: as stored) |
| `--kv-bits B` | memory, fits | KV cache precision in bits (default 16) |
| `-o FILE` | plot, report | output file (default `anatomy.png`, `report.html`) |
| `--theme light\|dark` | plot, report | color theme (default `light`) |
| `--dpi N` | plot | PNG resolution (default 200) |
| `--token`, `--config` | insights, memory, plot, report, fits | as for `inspect` |

### `fits`, `diff`, `scan`: practical questions

```bash
ai-model-explorer fits Qwen/Qwen3-8B rtx-4090 --context 32768      # or a size in GB: fits MODEL 24
ai-model-explorer devices                                            # the device names fits knows
ai-model-explorer diff Qwen/Qwen3-8B Qwen/Qwen3-8B-FP8               # fine-tune vs base, quantized vs original
ai-model-explorer scan --file repos.txt -o table.csv --workers 8     # many repos into one CSV
```

`fits` adds 10% for runtime buffers to weights + KV cache, reports the longest context that fits, and, when it does not
fit, the first of 8-bit weights, 4-bit weights, 4-bit weights with 8-bit or 4-bit KV cache that would. Unified-memory
devices (Mac, Jetson, phones) count only the share an app can normally use.

`diff` lists only what differs: family, parameters, stored size, quantization, every hyperparameter, components and block
types. `scan` writes one row per repo (family, parameters, active parameters, storage, hyperparameters, KV bytes per
token, the Hub's own count and whether it matches, config check, bytes read); a repo that fails becomes a row with `error`.
Repos come from the arguments, from `--file` (one repo id per line; blank lines and lines starting with `#` are
skipped), or both. Without `-o` the CSV goes to standard output; `--workers` sets the number of parallel requests (default 4).

### `ready`, `census`: the Deploy-Ready board and the Hub census, from the terminal

```bash
ai-model-explorer ready Qwen/Qwen3-8B             # device class at 4-bit / 8k context and five metadata checks
ai-model-explorer ready my-org/my-model --submit  # same, then opens the board submission in the browser
ai-model-explorer census Qwen/Qwen3-8B            # where it sits among the Hub's popular language models
```

`ready` applies the board's rules on your machine: memory for 4-bit weights and an 8k-token KV cache plus 10%, placed
in Phone (≤ 4.8 GB), Laptop (≤ 12), Desktop GPU (≤ 24), Workstation (≤ 80) or Cluster; then safetensors, config.json,
config agrees with the weights, the Hub's parameter count is right, and a license is declared. Run it before you submit.

`census` downloads the census table the website shows (cached for a day; `--refresh` to fetch it again) and reports the
model's size and KV-cache percentiles among popular language models, its release type (original, fine-tune, quantized
copy, merge, adapter), whether it comes from a leading provider or is built for Thai, and the five popular original
releases closest to it in size and cache.

`ready`, `census` and `diff` accept `--json` to print the raw result.

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
from ai_model_explorer import explore, compare, compare_table

m = explore("deepseek-ai/DeepSeek-V3")
m                                   # <Model deepseek-ai/DeepSeek-V3: decoder, 684.49B parameters>
m.params, m.active_params           # 684489845504, 37552297472
m.hp["heads"], m.hp["kv_heads"]     # 128, 128   (config.json, checked against the weights)
m.breakdown("kind")                 # [{'name': 'mixture of experts', 'params': ..., 'share': 0.976, ...}, ...]
m.kv_cache_bytes_per_token()        # 70272      (latent attention: 61 layers × (512 + 64) × 2 bytes)
m.memory(context=32768, weight_bits=4)
m.insights()                        # list of plain-language findings
m.plot("deepseek.png")              # returns the matplotlib Figure
m.report("deepseek.html")
m.to_json("deepseek.json")

print(compare_table(compare(["Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B-GPTQ-Int4"])))
```

| `Model` member | What it gives |
|---|---|
| `name`, `family`, `params`, `stored_bytes`, `hp`, `components`, `config` | the basics; `hp` has heads and KV heads from `config.json` when it agrees with the weights |
| `active_params` | parameters used per token: the total for dense models; for mixtures of experts only the experts a token visits, without multi-token-prediction layers |
| `breakdown(by="kind" \| "component" \| "stack")` | parameters, share, stored bytes and tensor count per group |
| `kv_cache_bytes_per_token(bits=16)` | KV cache one token adds over all attention layers; handles grouped-query and latent attention, skips state-space layers |
| `memory(context, batch, weight_bits, kv_bits)` | weights at a chosen precision plus the KV cache (activations not included) |
| `insights()` | findings in plain language, most useful first |
| `plot(path=None, theme="light")` | the anatomy diagram; `.png`, `.svg` or `.pdf` |
| `report(path, theme="light")` | self-contained HTML report |
| `summary()`, `to_dict()`, `to_json(path=None)` | text summary and raw data |

`compare(models)` takes repo ids, paths or `Model` objects and returns one dict per model; `compare_table(rows)` turns them into Markdown.

The lower-level `inspect(model, token=None, config=None, include_tensors=False) -> dict` returns the raw result that `Model` wraps:

```python
from ai_model_explorer import inspect, summary

a = inspect("Qwen/Qwen3-8B")
print(summary(a))                          # the same text as the CLI
print(a["totals"]["params"])               # 8190735360
print(a["hyperparameters_with_config"])    # width, layers, heads, kv heads, ffn, vocab, ...

local = inspect("./checkpoints/my-model")  # a folder of .safetensors, read offline
```

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
| `family` | as declared by `config.json` when present, otherwise from the shapes: `decoder`, `encoder`, `encoder-decoder` (cross-attention), `prefix-lm` (an encoder's output fed into a decoder-only LLM, as in most vision-language and audio LLMs), `transducer`, `diffusion`, or `unknown` |
| `totals` | `params` (logical parameters, quantized storage unpacked) and `bytes` (as stored) |
| `hyperparameters` | read from the weights alone: `d` (width), `layers`, `heads`, `kv_heads`, `head_dim`, `ffn`, `experts`, `vocab`, `tie` (tied embedding), `attn_layers`, `ssm_layers`, `kv_bytes_per_token_bf16`, … |
| `hyperparameters_with_config` | the same, with heads, KV heads, head size and active experts from `config.json` when the config agrees with the weights |
| `quantization` | storage formats found and how many tensors use each, e.g. `{"fp8": 45808}` or `{"gptq": 18624}` |
| `components` | list of parts. Each has `id`, `role` (`backbone`, `encoder`, `adapter`, `prediction` and `joint` for transducers, `aux`), `d`, `params`, `bytes`, `modality`, and `stacks` |
| `components[].stacks` | repeated layers: `path`, `n` (blocks), `types` (block types with `label`, `count`, `d` and `regions`), `mtp` (true when the last block is a multi-token-prediction layer) |
| `edges` | how components connect: `{"from", "to", "via"}`, e.g. encoder → decoder via cross-attention |
| `evidence` | sentences explaining non-obvious decisions, e.g. why a layer was read as a multi-token-prediction head |
| `breakdown` | parameters and stored bytes `by_kind`, `by_component` and `by_stack`, plus the size of any multi-token-prediction layer (`mtp`) |
| `config` | the `config.json` that was used, if any |
| `config_check` | per component: config values for `heads`, `kv_heads`, `head_dim`, `topk`, and whether they are `consistent` with the weights |
| `tensors` | only with `include_tensors` / `--tensors`: for every tensor, `comp`, `stack`, `block`, `kind` (`attn`, `ffn`, `moe`, `ssm`, `conv`, `norm`, `embed`, `head`, …), `region` and `role` |

## Gated repos, mirrors and offline use

- **Gated or private repos** (Llama, Gemma, some FLUX models): accept the license on the model page, create a read token at https://huggingface.co/settings/tokens, then `export HF_TOKEN=hf_...` or pass `--token`. The token is sent only to Hugging Face.
- **Mirror**: `export HF_ENDPOINT=https://your-mirror` to read through a Hugging Face mirror.
- **Offline**: point `inspect` at local `.safetensors` files. Only the headers are read and no network is needed; the time depends on the tensor count (see [Limits](#limits)).

## Limits

- Heads, KV heads and head size cannot always be told apart from shapes alone (a projection of width 4096 can be 32×128 or 64×64). With `config.json` they are taken from the config after checking it against the weights; without it they are estimates and marked as such.
- Very large manifests take longer in Python: DeepSeek-V3 (92k tensors) takes about 15 seconds with the embedded engine, about 5 with Node.js installed; a typical 7B model takes well under one second. A second look at the same repo commit reuses cached headers.
- Old pickle-only checkpoints without a zip index cannot be read without downloading them.
- On Linux ARM there is no prebuilt wheel of the `quickjs` dependency yet, so pip compiles it and needs a C compiler.
