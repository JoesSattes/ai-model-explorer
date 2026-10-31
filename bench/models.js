// Popular public checkpoints for the real-world benchmark (ungated repos or ungated mirrors of gated ones).
window.BENCH_MODELS = [
 // ---- language models, dense
 ["llm","openai-community/gpt2"],["llm","openai-community/gpt2-xl"],["llm","EleutherAI/pythia-1.4b"],["llm","facebook/opt-1.3b"],["llm","bigscience/bloom-560m"],
 ["llm","unsloth/Llama-3.2-1B"],["llm","unsloth/Llama-3.2-3B"],["llm","unsloth/Meta-Llama-3.1-8B"],["llm","unsloth/Llama-3.3-70B-Instruct"],["llm","NousResearch/Meta-Llama-3.1-405B"],
 ["llm","Qwen/Qwen2.5-0.5B"],["llm","Qwen/Qwen2.5-7B"],["llm","Qwen/Qwen2.5-72B"],["llm","Qwen/Qwen2.5-Coder-7B-Instruct"],["llm","Qwen/QwQ-32B"],
 ["llm","Qwen/Qwen3-0.6B"],["llm","Qwen/Qwen3-4B"],["llm","Qwen/Qwen3-8B"],["llm","Qwen/Qwen3-32B"],
 ["llm","mistralai/Mistral-7B-v0.1"],["llm","unsloth/mistral-7b-v0.3"],["llm","mistralai/Mistral-Nemo-Instruct-2407"],["llm","mistralai/Mistral-Small-24B-Instruct-2501"],["llm","mistralai/Devstral-Small-2505"],
 ["llm","unsloth/gemma-2-2b"],["llm","unsloth/gemma-2-9b"],["llm","unsloth/gemma-3-1b-it"],["llm","unsloth/gemma-3-27b-it"],
 ["llm","microsoft/phi-2"],["llm","microsoft/Phi-3-mini-4k-instruct"],["llm","microsoft/Phi-4-mini-instruct"],["llm","microsoft/phi-4"],
 ["llm","deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],["llm","deepseek-ai/DeepSeek-R1-Distill-Llama-70B"],
 ["llm","tiiuae/falcon-7b"],["llm","tiiuae/Falcon3-7B-Base"],["llm","allenai/OLMo-2-1124-7B"],["llm","HuggingFaceTB/SmolLM2-1.7B"],["llm","HuggingFaceTB/SmolLM3-3B"],
 ["llm","ibm-granite/granite-3.3-8b-instruct"],["llm","THUDM/glm-4-9b-chat-hf"],["llm","CohereLabs/c4ai-command-r7b-12-2024"],["llm","internlm/internlm3-8b-instruct"],
 ["llm","google/gemma-3-270m"],["llm","baidu/ERNIE-4.5-0.3B-PT"],["llm","LiquidAI/LFM2-1.2B"],["llm","nvidia/Llama-3.1-Nemotron-Nano-8B-v1"],["llm","swiss-ai/Apertus-8B-2509"],
 // ---- language models, mixture of experts
 ["moe","mistralai/Mixtral-8x7B-v0.1"],["moe","Qwen/Qwen1.5-MoE-A2.7B"],["moe","Qwen/Qwen3-30B-A3B"],["moe","Qwen/Qwen3-235B-A22B"],["moe","Qwen/Qwen3-Coder-480B-A35B-Instruct"],["moe","Qwen/Qwen3-Next-80B-A3B-Instruct"],
 ["moe","deepseek-ai/DeepSeek-V2-Lite"],["moe","deepseek-ai/DeepSeek-V3"],["moe","deepseek-ai/DeepSeek-R1"],["moe","deepseek-ai/DeepSeek-V3.1"],
 ["moe","moonshotai/Kimi-K2-Instruct"],["moe","moonshotai/Moonlight-16B-A3B"],["moe","openai/gpt-oss-20b"],["moe","openai/gpt-oss-120b"],
 ["moe","zai-org/GLM-4.5-Air"],["moe","zai-org/GLM-4.5"],["moe","allenai/OLMoE-1B-7B-0924"],["moe","microsoft/Phi-3.5-MoE-instruct"],["moe","ibm-granite/granite-3.1-3b-a800m-instruct"],
 ["moe","baidu/ERNIE-4.5-21B-A3B-PT"],["moe","tencent/Hunyuan-A13B-Instruct"],["moe","MiniMaxAI/MiniMax-M2"],["moe","unsloth/Llama-4-Scout-17B-16E-Instruct"],["moe","meituan-longcat/LongCat-Flash-Chat"],
 // ---- state-space and hybrid
 ["ssm","state-spaces/mamba-2.8b-hf"],["ssm","tiiuae/falcon-mamba-7b"],["ssm","tiiuae/Falcon-H1-7B-Base"],["ssm","nvidia/NVIDIA-Nemotron-Nano-9B-v2"],["ssm","nvidia/Nemotron-H-8B-Base-8K"],
 ["ssm","ai21labs/AI21-Jamba-Mini-1.7"],["ssm","ibm-granite/granite-4.0-h-tiny"],["ssm","Zyphra/Zamba2-2.7B"],["ssm","RWKV/v6-Finch-1B6-HF"],["ssm","NX-AI/xLSTM-7b"],
 // ---- encoders, embeddings, seq2seq
 ["enc","google-bert/bert-base-uncased"],["enc","FacebookAI/roberta-large"],["enc","FacebookAI/xlm-roberta-large"],["enc","distilbert/distilbert-base-uncased"],["enc","microsoft/deberta-v3-base"],
 ["enc","answerdotai/ModernBERT-large"],["enc","google/electra-base-discriminator"],["enc","albert/albert-base-v2"],
 ["emb","sentence-transformers/all-MiniLM-L6-v2"],["emb","sentence-transformers/all-mpnet-base-v2"],["emb","BAAI/bge-m3"],["emb","BAAI/bge-large-en-v1.5"],["emb","intfloat/multilingual-e5-large"],
 ["emb","Qwen/Qwen3-Embedding-0.6B"],["emb","Qwen/Qwen3-Embedding-8B"],["emb","nomic-ai/nomic-embed-text-v1.5"],["emb","Alibaba-NLP/gte-multilingual-base"],["emb","Qwen/Qwen3-Reranker-0.6B"],
 ["s2s","google-t5/t5-base"],["s2s","google/flan-t5-large"],["s2s","google/mt5-base"],["s2s","facebook/bart-large-cnn"],["s2s","facebook/nllb-200-distilled-600M"],["s2s","facebook/m2m100_418M"],["s2s","Helsinki-NLP/opus-mt-en-de"],
 // ---- speech and audio
 ["speech","openai/whisper-tiny"],["speech","openai/whisper-small"],["speech","openai/whisper-large-v3"],["speech","openai/whisper-large-v3-turbo"],["speech","distil-whisper/distil-large-v3"],
 ["speech","facebook/wav2vec2-base-960h"],["speech","facebook/wav2vec2-large-xlsr-53"],["speech","facebook/hubert-large-ls960-ft"],["speech","microsoft/wavlm-large"],["speech","facebook/mms-1b-all"],
 ["speech","facebook/seamless-m4t-v2-large"],["speech","UsefulSensors/moonshine-base"],["speech","Qwen/Qwen2-Audio-7B-Instruct"],["speech","Qwen/Qwen2.5-Omni-7B"],["speech","kyutai/stt-2.6b-en"],
 ["speech","microsoft/speecht5_tts"],["speech","facebook/musicgen-small"],["speech","facebook/encodec_24khz"],["speech","microsoft/VibeVoice-1.5B"],["speech","ResembleAI/chatterbox"],
 ["speech","nari-labs/Dia-1.6B-0626"],["speech","pyannote/segmentation-3.0"],["speech","speechbrain/spkrec-ecapa-voxceleb"],["speech","MIT/ast-finetuned-audioset-10-10-0.4593"],["speech","laion/clap-htsat-unfused"],
 // ---- vision
 ["vision","google/vit-base-patch16-224"],["vision","facebook/dinov2-large"],["vision","microsoft/resnet-50"],["vision","facebook/convnext-base-224"],["vision","microsoft/swin-base-patch4-window7-224"],
 ["vision","openai/clip-vit-large-patch14"],["vision","google/siglip-so400m-patch14-384"],["vision","google/siglip2-base-patch16-224"],["vision","facebook/sam-vit-huge"],["vision","facebook/sam2.1-hiera-large"],
 ["vision","facebook/detr-resnet-50"],["vision","PekingU/rtdetr_r50vd"],["vision","nvidia/segformer-b5-finetuned-ade-640-640"],["vision","facebook/mask2former-swin-large-coco-panoptic"],
 ["vision","depth-anything/Depth-Anything-V2-Large-hf"],["vision","Intel/dpt-large"],["vision","google/owlv2-base-patch16-ensemble"],["vision","IDEA-Research/grounding-dino-base"],["vision","timm/efficientnet_b0.ra_in1k"],
 // ---- vision-language
 ["vlm","Qwen/Qwen2.5-VL-7B-Instruct"],["vlm","Qwen/Qwen2.5-VL-72B-Instruct"],["vlm","Qwen/Qwen3-VL-30B-A3B-Instruct"],["vlm","Qwen/Qwen3-VL-235B-A22B-Instruct"],["vlm","llava-hf/llava-1.5-7b-hf"],
 ["vlm","llava-hf/llava-onevision-qwen2-7b-ov-hf"],["vlm","HuggingFaceTB/SmolVLM2-2.2B-Instruct"],["vlm","mistral-community/pixtral-12b"],["vlm","unsloth/Llama-3.2-11B-Vision-Instruct"],
 ["vlm","unsloth/gemma-3-4b-it"],["vlm","microsoft/Phi-4-multimodal-instruct"],["vlm","deepseek-ai/Janus-Pro-7B"],["vlm","moonshotai/Kimi-VL-A3B-Instruct"],["vlm","zai-org/GLM-4.5V"],
 ["vlm","OpenGVLab/InternVL3-8B-hf"],["vlm","nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"],["vlm","google/paligemma2-3b-pt-224"],["vlm","allenai/Molmo-7B-D-0924"],["vlm","lerobot/smolvla_base"],
 // ---- image and video generation
 ["gen","stable-diffusion-v1-5/stable-diffusion-v1-5"],["gen","stabilityai/stable-diffusion-xl-base-1.0"],["gen","stabilityai/sdxl-turbo"],["gen","black-forest-labs/FLUX.1-schnell"],
 ["gen","Qwen/Qwen-Image"],["gen","PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"],["gen","Efficient-Large-Model/Sana_1600M_1024px_diffusers"],["gen","HiDream-ai/HiDream-I1-Full"],["gen","facebook/DiT-XL-2-256"],
 ["gen","Wan-AI/Wan2.1-T2V-1.3B-Diffusers"],["gen","Wan-AI/Wan2.2-TI2V-5B-Diffusers"],["gen","hunyuanvideo-community/HunyuanVideo"],["gen","Lightricks/LTX-Video"],["gen","THUDM/CogVideoX-2b"],["gen","playgroundai/playground-v2.5-1024px-aesthetic"],
 // ---- time series and tabular
 ["ts","amazon/chronos-t5-small"],["ts","amazon/chronos-bolt-base"],["ts","google/timesfm-2.0-500m-pytorch"],["ts","Salesforce/moirai-1.1-R-base"],["ts","ibm-granite/granite-timeseries-ttm-r2"],["ts","Prior-Labs/TabPFN-v2-clf"],
 // ---- quantized releases
 ["quant","Qwen/Qwen3-8B-FP8"],["quant","Qwen/Qwen3-30B-A3B-GPTQ-Int4"],["quant","Qwen/Qwen2.5-7B-Instruct-AWQ"],["quant","RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"],["quant","nvidia/Llama-3.1-8B-Instruct-FP4"],
 ["quant","nvidia/DeepSeek-R1-0528-FP4"],["quant","mlx-community/Qwen3-8B-4bit"],["quant","unsloth/Qwen3-8B-bnb-4bit"],["quant","unsloth/Llama-3.2-1B-Instruct-bnb-4bit"],["quant","nvidia/Qwen3-8B-FP4"],
];
