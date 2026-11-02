# How it works

A safetensors file begins with a small JSON header that lists every tensor's name, shape and storage type. For a 685-billion-parameter model that header is about 12 MB, while the weights are 689 GB. AI Model Explorer reads only that list and works out the architecture from it. The full method and its evaluation are in the paper: [Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy](https://www.researchgate.net/publication/417293353_Reading_the_Architecture_off_the_Weights_Manifest-Only_Recovery_of_Neural_Network_Anatomy).

## 1. Read only the manifest

For a Hugging Face repo the tool asks the Hub for the file list, then requests two byte ranges per weight file: the 8-byte header length, then the header itself. Sharded checkpoints are found through their `.safetensors.index.json`. Optimizer states, LoRA adapters, per-language adapters and duplicate copies in another precision (`model.fp16.safetensors` next to `model.safetensors`) are skipped. For diffusers repos, only the components listed in `model_index.json` are read. PyTorch zip checkpoints are read the same way, from the zip directory at the end of the file plus the pickled index, never the tensor data.

In a benchmark of 191 popular checkpoints, the median read was 79 KB per repo, and 161 MB in total for 12.1 TB of weights.

## 2. Undo quantized storage

Quantized checkpoints store a weight as several tensors: packed integers, scales, zero points, sometimes a group index. The tool recognizes FP8 with block or per-channel scales, NVFP4, MXFP4, GPTQ, AWQ, MLX and bitsandbytes layouts and rebuilds each logical weight with its true shape. That is why parameter counts stay exact for quantized models, where the Hub's own count can report packed bytes instead.

## 3. Find the repeated blocks

Tensor names form a tree (`model.layers.17.self_attn.q_proj.weight`). Numeric path tokens mark lists, and a list whose items have the same shape signature is a stack of repeated blocks. One stack can hold several block types: DeepSeek-V3 has 3 dense blocks, then 58 mixture-of-experts blocks, then a multi-token-prediction block. Hybrid models such as Qwen3-Next or Nemotron-H interleave attention, state-space and MoE blocks in one stack.

## 4. Give each part a role from its shapes

Inside a block, the shapes have a structure that names don't change:

- **Attention**: several projections read the residual width `d`; the query and output are `d × d` (or `d × heads·head_dim`); key and value are narrower when heads are shared (grouped-query attention). Latent attention (DeepSeek's MLA) shows up as a low-rank down-projection followed by an up-projection.
- **Feed-forward**: two or three matrices expand `d` to a larger width and project back; three means a gated unit such as SwiGLU.
- **Mixture of experts**: many identical feed-forward units plus a router whose output size equals the number of experts.
- **State-space / linear attention**: depthwise 1-D convolutions, per-channel decay parameters and input-dependent projections (Mamba, Mamba-2, gated DeltaNet, RWKV).
- **Others**: conformer convolution modules, recurrent cells, adaLN modulation in diffusion transformers, embeddings, norms and output heads.

Names are used only as a tie-breaker. When every name is replaced by a random token, accuracy drops from 96.8% to 95.0%, so most of the work is done by shapes.

## 5. Connect the components

Multi-part models (a vision encoder, a projector and a language model; a speech encoder, a prediction network and a joint network) are linked by matching the output width of one part to the input width of the next. Cross-attention links an encoder to a decoder.

## 6. Check config.json against the weights

Some fields can't be read from shapes. A 4096-wide projection could be 32 heads of 128 or 64 heads of 64. When `config.json` is present, its head counts are taken and checked against the weights: `heads × head_dim` must equal the projection width, KV heads must match the key/value width, and so on. The result reports whether they agree. On the corpus, 95% of configs agree with their weights, and most of the rest use attention variants where the query and value head sizes differ.

## The same code everywhere

The website, the command line and the Python API all run one JavaScript implementation (`src/infer.js`, `src/loaders.js`). The pip package executes it in an embedded QuickJS engine, so results don't drift between the web and Python versions.

## Accuracy at a glance

| Benchmark | Result |
|---|---|
| 510 architectures from `transformers`, ground truth from module classes the method never sees | 96.8% of tensors get the right role (macro), 99.9% micro |
| Same, every name replaced by a random token | 95.0% |
| Best rule set that reads names only | 82.3% |
| Parameter totals | exact for 510 of 510, also under NVFP4, FP8 and GPTQ |
| 191 popular Hub checkpoints | totals agree with the Hub's own count for 163 of 170; 4 of the 7 differences are the Hub counting packed 4-bit bytes |
| Recovery time | median 10 ms per model in the browser |
