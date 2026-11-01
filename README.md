# AI Model Explorer

**See inside any AI model, part by part, before you download or squeeze it.**
Live: **https://joessattes.github.io/ai-model-explorer/** · Thai / English · dark theme (light on the toggle)

![AI Model Explorer](og.png)

Paste a Hugging Face repo id and the explorer reads only the file headers, never the weights. It works out the
model's anatomy on its own (components, layer stacks, what each layer does, how the parts connect, parameter
counts) and shows it as a diagram you can walk through with **Play**. Every value says where it came from:
**measured** from the weights, read from **config** and checked against the weights, or **estimated**.

## Bring any model
| Input | What is read | What you get |
|---|---|---|
| Hugging Face repo id | file list, safetensors headers by HTTP range request, shard indexes, `config.json`; for PyTorch zip checkpoints only the zip directory and `data.pkl` | automatic anatomy, exact parameter counts, KV-cache size, precision per tensor |
| `.safetensors` file | its header | the same, offline |
| `.tflite` (LiteRT) file | the full op graph, parsed in your browser | real ops, shapes, dtypes, quantize/dequantize seams |
| `config.json` (paste) | the config values | synthesized structure for common families |

Nothing is uploaded anywhere. Models you open are kept in your own browser (IndexedDB), so the next visit opens
them instantly. Gated repositories need their license accepted on Hugging Face first.

## How the automatic anatomy works
The method is in `src/infer.js` (about 650 lines, no dependencies) and is described in the paper in `paper/`.
From tensor names, shapes and dtypes alone it:

1. normalizes quantized storage to logical shapes (NVFP4, FP8, GPTQ, AWQ, MXFP4, MLX, bitsandbytes NF4);
2. finds repeated blocks in the module tree and groups them into block types (a hybrid Mamba-2 / MoE / attention
   stack gets one type each);
3. labels the parts of each block from shape algebra: attention (fused, GQA, latent, cross), dense and gated
   feed-forward, mixtures of experts, SSM / linear attention, conformer convolution, adaLN modulation, recurrent
   cells, multi-token-prediction layers;
4. links components by matching widths (vision tower → projector → LLM; encoder → joint ← prediction network);
5. reads what shapes cannot show (heads, KV heads, head size, routing top-k) from `config.json`, and checks that
   the config agrees with the weights.

## How well it works
| Benchmark | Result |
|---|---|
| 510 architectures from `transformers` (labels from module classes the method never sees) | 96.8% of tensors get the right role (macro) vs 82.3% for the best name-only baseline; 95.0% with every name replaced by a random token; parameter totals exact for 510/510, also under NVFP4, FP8 and GPTQ |
| Same, head geometry | 73% from shapes alone; 96–98% with the config; 95% of configs agree with their weights |
| 191 popular Hugging Face checkpoints (DeepSeek-V3, Kimi K2, Qwen3, Llama 3.3, gpt-oss, Gemma 3, Whisper, SAM, SDXL, Wan, …) | totals agree with the Hub's own count for 163 of 170; 4 of the 7 differences are the Hub counting packed 4-bit bytes; head count 97% with config; median 79 KB read per repo |

Run the Hub benchmark yourself: open **https://joessattes.github.io/ai-model-explorer/bench.html**, press Start,
then Download results; `node research/realbench.js export.jsonl corpus.jsonl` analyses the export.

## Repository
| Path | What |
|---|---|
| `index.html`, `en/index.html` | the built site, Thai and English (one file each) |
| `src/` | sources: `infer.js` (recovery), `loaders.js` (headers, range requests), `model.js`, `app.js`, `i18n.js`; `python3 build.py` rebuilds both pages |
| `bench/`, `bench.html` | the Hub checkpoint benchmark page and its model list |
| `research/` | corpus builder, evaluation (`eval.js`), Hub analysis (`realbench.js`), results (`res/`), the Hub export (`data/`) |
| `paper/` | LaTeX sources and PDF of *Reading the Architecture off the Weights* |

## Templates
- **Whisper medium (Thai)**: `biodatlab/distill-whisper-th-medium` converted to `.tflite` (random weights, so ops, shapes, sizes and compute are real; accuracy is not)
- **Nemotron 3.5** Super VL, Transcribe, ASR streaming and Embed NVFP4; **Parakeet TDT 0.6B v3**; **FastH3 Omni FP8**; **Kumo** Tabular and Relational: analysed automatically from their weight manifests
- **Qwen3-ASR 1.7B**, **Qwen3.8 27B**, **Llama 3.2 1B**, **GPT-2 small**, **BERT base**, **ViT base**
- **Precision demos**: Whisper rehearsal models in fp32, weight-only int8, static int8 and dynamic int4

## Run locally
**The website** needs no install: open `index.html` in a browser. After editing `src/`, rebuild with
`cd src && python3 build.py` (standard library only).

**Reproducing the paper** needs Python 3.11–3.13 and Node.js 18 or newer:

```bash
git clone https://github.com/JoesSattes/ai-model-explorer.git
cd ai-model-explorer
python3 -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt        # PyTorch and transformers at the versions used in the paper
```

Quick check (about a minute):

```bash
python research/build_corpus.py corpus.jsonl --only gpt2,whisper,llama
node research/eval.js corpus.jsonl     # prints macro_acc 1 for these three
```

Full run: `python research/build_corpus.py corpus.jsonl` builds all 510 architectures (tens of minutes, a few GB of RAM;
models are created on the meta device, nothing is downloaded but configs). Then `node research/eval.js corpus.jsonl`
with `--obf --nolex`, `--quant nvfp4|fp8|gptq`, `--cfg`, or a baseline: `--v1` (name rules), `--v1moe` (name rules
with an MoE label) or `--gguf` (llama.cpp's tensor-name table, built by `python research/gguf_baseline.py`).
Hub benchmark: `node research/realbench.js research/data/hub-bench-191.jsonl.gz corpus.jsonl`.
Figures: `cd paper && python scripts/figs.py` (reads `research/res/`).

## Citation
If you use AI Model Explorer or its recovery method, please cite the paper ([ResearchGate](https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_Manifest-Only_Recovery_of_Neural_Network_Anatomy), [PDF](paper/manifest-anatomy-singkul-2026.pdf)):

```bibtex
@misc{singkul2026reading,
  title  = {Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy},
  author = {Singkul, Sattaya},
  year   = {2026},
  note   = {Preprint},
  howpublished = {ResearchGate},
  url    = {https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_Manifest-Only_Recovery_of_Neural_Network_Anatomy}
}
```

## Credits
Created by **[Sattaya Singkul](https://joessattes.github.io/Sattaya-AI-Website/)** for the workshop *Squeeze Until It Breaks — A Hands-On Edge Inference Contest*, AI Expo Thailand 2026.

## License
[MIT](LICENSE) © 2026 Sattaya Singkul. Keep the copyright notice when you reuse the code.
