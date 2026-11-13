<p align="center">
  <img src="https://raw.githubusercontent.com/JoesSattes/ai-model-explorer/main/assets/logo-180.png" width="96" alt="AI Model Explorer logo">
</p>

<h1 align="center">AI Model Explorer</h1>

<p align="center"><b>See inside a Hugging Face model, part by part, from its safetensors headers alone.</b></p>

<p align="center">
  <a href="https://joessattes.github.io/ai-model-explorer/">Web app</a> ·
  <a href="https://github.com/JoesSattes/ai-model-explorer">GitHub</a> ·
  <a href="https://github.com/JoesSattes/ai-model-explorer/blob/main/docs/usage.md">Docs</a> ·
  <a href="https://www.researchgate.net/publication/417293353">Paper</a>
</p>

`ai-model-explorer` reads the safetensors header of a model (tensor names, shapes and types, a median of 80 KB across
12,974 Hub repositories) and works out its architecture: components, repeated layers and what each one does, how the parts
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
| `ai-model-explorer tax MODEL --lang th,en --coverage` | vocabulary tax: the token-table rows (and bytes) your languages never use, for any model and any tokenizer format; `--file list.txt -o tax.jsonl` for many models |
| `ai-model-explorer plot MODEL -o model.png` | the anatomy diagram as PNG, SVG or PDF (`--theme dark`) |
| `ai-model-explorer report MODEL -o model.html` | one self-contained HTML report: diagram, insights, tables, memory |
| `ai-model-explorer fits MODEL rtx-4090 --context 32768` | does it fit on a device (or `24` GB)? the longest context that fits, and the cheapest precision that would make it fit (`devices` lists the names) |
| `ai-model-explorer diff A B` | what changed between two checkpoints: a fine-tune vs its base, a quantized copy vs the original, two versions |
| `ai-model-explorer scan --file repos.txt -o table.csv` | many repos into one CSV (in parallel; failures become rows with an error) |
| `ai-model-explorer ready MODEL [--submit]` | smallest device class at 4-bit weights / 8k context and five metadata checks, as on the Deploy-Ready board |
| `ai-model-explorer census MODEL` | where it sits among the Hub's popular language models, and the closest popular original releases |
| `ai-model-explorer devices` | the device names `fits` accepts |
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

from ai_model_explorer import fits, diff, scan
fits("Qwen/Qwen3-8B", "rtx-4090", context=32768)   # {'fits': True, 'max_context': ..., 'suggestion': ...}
diff("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-FP8")         # [{'field': 'quantization', 'a': 'none', 'b': 'fp8'}, ...]
rows = scan(open("repos.txt").read().split())      # one dict per repo
```

Encoder-decoder models (Whisper and other speech or translation models) have a few extra helpers:

```python
m = explore("biodatlab/distill-whisper-th-medium")
m.distinct_params                  # head counted once when config.json ties it but the file stores a copy
m.split()                          # encoder / decoder share; cross-attention K/V counted with the encoder, as exports do
m.cross_kv_bytes()                 # cross-attention K/V computed once per input (16-bit)
m.macs()                           # rough multiply-adds: encoder per input, decoder per token
m.size(4, {"model.decoder": 8})    # file size with the encoder at 4-bit and the decoder at 8-bit
```

To make a model smaller, ask for a plan: each step comes with the weights it removes, what was measured and what to do.

```python
m = explore("google/gemma-4-E2B-it")
print(m.plan_text(["th", "en", "zh"], text_only=True))   # reads tokenizer.json to count the pieces your languages need
m.plan(["th", "en"])                                     # the same as a list of dicts
m.vocab_scripts()                                        # tokenizer pieces per writing system
```

From the command line: `ai-model-explorer plan google/gemma-4-E2B-it --lang th,en,zh --text-only`.

For Gemma 4 E2B the plan finds that every vocabulary row costs 10,496 weights (1,536 in the token embedding plus 8,960
in the per-layer embeddings), so trimming the vocabulary to the languages you serve is the largest cut, followed by the
audio and vision towers for text-only use; it also reports that 20 of the 35 layers share an earlier layer's KV cache.
The website shows the same plan under "Where to cut" (pick config.json and tokenizer.json together with the weights).

The vocabulary tax works for any model with token tables and any common tokenizer format: `tokenizer.json`,
SentencePiece `tokenizer.model`, `vocab.json`, `*.tiktoken`, Mistral `tekken.json` and llama.cpp `.gguf` (only the
tokenizer file is downloaded, never the weights). `coverage=True` tokenizes bundled test sentences (Parallel UD
treebanks in 21 languages, plus Thai social-media text) and counts the tokens that would land in the removed rows;
it needs one of `tokenizers`, `sentencepiece`, `tiktoken` or `llama-cpp-python`.

```python
r = explore("Qwen/Qwen3-0.6B").vocab_tax(["th", "en"], coverage=True)
r["removable_bytes"], r["share_of_bytes"], r["coverage"]["per_language"]["th"]["rate"]
```

By default the keep set is decided by writing system (with Latin always kept: names, numbers and code appear in every
language), so text in the kept languages tokenizes exactly as before. Languages that share a script (English and German,
Chinese and Japanese) can only be told apart by text: pass your own lines with `corpus=["mine.txt"]` (CLI `--corpus`)
and only the pieces your text uses are kept, plus every single-character piece so unseen words can still be spelled.
The result reports, on a held-out 20% of your text, how many tokens would be split into smaller pieces. On 27k lines of
Thai text this keeps 10-28% of a large multilingual vocabulary at a 0.1-0.6% re-split rate.

Every corpus result also carries a **trim certificate**: a Good-Turing estimate, from your text alone, of the share of
tokens that new text of the same kind will see re-split, and how many lines of text would bring that under a target
(`target_resplit=0.01`, CLI `--target`). Across 20 languages and 5 tokenizer families the estimate is within 5% of the
measured rate (median; r = 0.995 in log scale), and the line count, fitted on half and all of your text, lands within 1%
(median) on a held-out size. Rough guide at a 1% target: Thai about 5k lines, Spanish 7k, German 13k, French 24k; broad
English web text needs far more because its pieces are spread over most of the Latin vocabulary.

`trim_vocab(languages, "out/")` (CLI `tax ... --export out/`) writes the trimmed `tokenizer.json` or SentencePiece
`.model`, `kept_ids.json` and `slice_weights.py`, and checks the bundled test sentences.

Rows past the end of the tokenizer file are kept (they may be special tokens the file does not list), and a tokenizer
with more pieces than the model has rows is reported as a mismatch.

`.tflite` (LiteRT) files open the same way, so you can check what a quantization recipe actually stored:

```python
t = explore("export/")             # a folder with encoder.tflite and decoder.tflite, or one .tflite file
t.precision()                      # parameters per stored type, e.g. {'INT4': ..., 'INT8': ..., 'FLOAT32': ...}
t.layers()                         # each layer: params, bytes on disk, stored type(s)
t.macs()                           # multiply-adds counted from the graph: encoder per input, decoder per call
t.plot("export.png")               # bytes per layer, coloured by stored type
```

`explore()` returns a `Model` (`TfliteModel` for .tflite); `inspect()` returns the raw dict it wraps. Every field and method is described in the
[usage guide](https://github.com/JoesSattes/ai-model-explorer/blob/main/docs/usage.md).

## Why it is useful
- **Before downloading**: the anatomy of a 689 GB checkpoint from 12 MB of headers.
- **Quantization and edge planning**: where the parameters sit, which layers keep a KV cache, how each tensor is stored (FP8, NVFP4, GPTQ, AWQ, MXFP4, MLX, bitsandbytes).
- **New architectures**: hybrid attention / state-space / mixture-of-experts stacks, multi-token-prediction layers, encoders feeding an LLM, each labeled with the evidence.
- **Comparisons and reports**: a table across many models, diagrams for slides and papers, an HTML report to share.
- **Batch analysis**: plain dicts or JSON from a single function call.

## What is new
Reading the header is standard practice; tools such as hf-mem estimate memory from it, and HF Viewer draws graphs from a model's config and code. This package interprets the header itself: components, block types and the role of every tensor without config or code (95% of roles still right with every tensor name hidden), exact logical parameter counts under quantized storage, KV cache from the weights including latent attention, and a check of `config.json` against the weights.

## How accurate is it?
On 510 architectures from the transformers library, 96.8% of tensors get the right role (82.3% for the best rule
set that reads names only, 95.0% for this method with every name replaced by a random token). Parameter totals
are exact for all 510, including FP8, NVFP4 and GPTQ storage. On 191 popular Hugging Face checkpoints, totals agree
with the Hub's own count for 149 of the 158 transformers-style checkpoints that show a count (allowing for the 0-d
scalars the Hub skips); 5 of the 9 differences are packed 4-bit releases where the Hub counts packed storage.

## How it runs
The analysis is the same JavaScript that powers the website (`infer.js`, `loaders.js`), run in an embedded QuickJS
engine, so the Python and web results match. Python 3.9+, two dependencies (`quickjs`, and `matplotlib` for diagrams), no PyTorch or transformers.
A typical model takes well under a second. Headers of a sharded model are fetched in parallel, and each repo commit's
headers are cached (`~/.cache/ai-model-explorer`, or `$AME_CACHE`; `cache=False` or `AME_NO_CACHE=1` to skip). If Node.js
is installed, large manifests are analyzed with it (DeepSeek-V3: about 5 s instead of 15; `AME_JS=quickjs` to opt out).

## Citation
Sattaya Singkul. 2026. *Reading the Architecture off the Weights: Manifest-Only Model Anatomy and a Census of 12,974 Hugging Face Repositories.*
Preprint, [ResearchGate](https://www.researchgate.net/publication/417293353).

MIT License © 2026 Sattaya Singkul
