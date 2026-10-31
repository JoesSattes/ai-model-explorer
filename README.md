# AI Model Explorer

**See inside any AI model, part by part — before you squeeze it.**
Live: **https://joessattes.github.io/ai-model-explorer/** · Thai / English

![AI Model Explorer](og.png)

AI Model Explorer splits a model into its parts — input, embedding, encoder or decoder stacks, attention, feed-forward, output head — and shows the **real weights, compute, TFLite op count and precision** of each part. Press **Play** to walk through what goes into each part and what comes out.

## Bring any model
| Input | What is read | What you get |
|---|---|---|
| `.tflite` (LiteRT) file | the full op graph, parsed in your browser | real ops, shapes, dtypes, quantize/dequantize seams |
| Hugging Face repo | only the safetensors header (names, shapes, dtypes) | exact weight layout, estimated compute |
| `config.json` (paste) | the config values | synthesized structure for Llama, Qwen, Gemma, Mistral, Phi, GPT-2, BERT, ViT, Whisper, BART, T5 |

Nothing is uploaded anywhere: files are parsed in the browser. Models you load are kept in your own browser (IndexedDB), so your next visit opens them instantly.

## Templates
- **Whisper medium (Thai)** — `biodatlab/distill-whisper-th-medium`, the architecture converted to `.tflite` (random weights, so ops, shapes, sizes and compute are real; accuracy is not)
- **Qwen3-ASR 1.7B** (audio encoder → LLM) and **Qwen3.8 27B** (vision encoder → hybrid linear/full-attention LLM with MTP) — structure built from their official config.json
- **Llama 3.2 1B**, **GPT-2 small**, **BERT base**, **ViT base** — structure from the transformers implementation
- **Precision demos** — Whisper rehearsal models in fp32, weight-only int8 and dynamic int4

## Run locally
Open `index.html` in a browser. To rebuild after editing `src/`: `cd src && python3 build.py`.

## Credits
Created by **[Sattaya Singkul](https://joessattes.github.io/Sattaya-AI-Website/)** for the workshop *Squeeze Until It Breaks — A Hands-On Edge Inference Contest*, AI Expo Thailand 2026.

## License
[MIT](LICENSE) © 2026 Sattaya Singkul. Keep the copyright notice when you reuse the code.
