"""Build a ground-truth corpus: every transformers model type, default config, instantiated on the meta device.
For each model we write the weight manifest (what a safetensors header would hold) and ground truth taken from
the module tree (class names), which the inference never sees.

usage: python build_corpus.py OUT.jsonl [--only type1,type2] [--config-json path --name N]
"""
import json, sys, signal, re, traceback, os, warnings
warnings.filterwarnings("ignore")
import torch
import transformers
from transformers import AutoConfig
from transformers.models.auto import modeling_auto as ma
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
transformers.logging.set_verbosity_error()

PREF = ["MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES",
        "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES", "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES", "MODEL_FOR_TDT_MAPPING_NAMES",
        "MODEL_FOR_RNNT_MAPPING_NAMES", "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "MODEL_FOR_CTC_MAPPING_NAMES", "MODEL_FOR_MASKED_LM_MAPPING_NAMES",
        "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES", "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES", "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES",
        "MODEL_MAPPING_NAMES"]
DT = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32", torch.uint8: "U8",
      torch.int8: "I8", torch.bool: "BOOL", torch.float64: "F64"}


def model_class(mt):
    for tab in PREF:
        names = getattr(ma, tab, {})
        if mt in names:
            n = names[mt]
            if isinstance(n, (list, tuple)): n = n[0]
            c = getattr(transformers, n, None)
            if c is not None: return c
    return None


# ---------------- ground truth from class names (never visible to the inference)
SUF = [  # (kind, regex on a lowercased class name); nearest ancestor that matches wins, MoE containers override
    ("head", r"(predictionheadtransform|lmpredictionhead|predictionhead|lmhead|onlymlmhead|pooler|classificationhead|mlmhead|predictions)$"),
    ("attn", r"(selfoutput|attentionoutput)$"),
    ("norm", r"(rmsnorm|layernorm|groupnorm|batchnorm\dd?|norm)$"),
    ("cross", r"(crossattention|cross_attention|encdecattention|crossattn)$"),
    ("ssm", r"(mamba2?mixer|mixer|mamba|ssm|deltanet|gateddeltanet|linearattention|lightningattention|rwkv\w*|recurrentblock|rglru|xlstm\w*|retention|hgrn|gla)$"),
    ("attn", r"(attention|attn|selfattention|multiheadattention|sdpaattention)$"),
    ("rnn", r"(lstm|gru|rnn|lstmcell)$"),
    ("mod", r"(adalayernorm\w*|adaln\w*|modulation|timestepembedding|timestepembedder)$"),
    ("conv", r"(convolutionmodule|convmodule|conformerconvolution\w*|convblock|depthwiseconv\w*)$"),
    ("ffn", r"(mlp|feedforward|ffn|intermediate|output|swiglu|geglu|densereludense|densegatedactdense|selfoutput)$"),
    ("embed", r"(embedding|embeddings|embed|patchembed\w*|patchembeddings)$"),
]
MOE = r"(sparsemoeblock|moeblock|sparsemoe|moe|moelayer|experts|expert|router|topkrouter|moegate|sparsemlp|switchsparsemlp|routedexperts)$"


def kind_of(classes, attr, modname):
    cl = [re.sub(r"^.*?(?=[a-z]*$)", "", c.lower()) if False else c.lower() for c in classes]
    if any(re.search(MOE, c) for c in cl[1:]): return "moe"
    for c in reversed(cl[1:]):
        for k, rx in SUF:
            if re.search(rx, c):
                if k == "attn" and re.search(r"encoder_attn|crossattention|cross_attn|cross_attention", modname): return "cross"
                return k
    if re.search(r"(^|\.)(lm_head|head|heads|classifier|score|proj_out|output_proj|qa_outputs|joint\w*)$", modname): return "head"
    last = modname.split(".")[-1].lower() if modname else ""
    if re.search(r"^(fc\d|mlp|ffn|dense|wi|wo|w[123]|c_fc|c_proj)$", last): return "ffn"
    if re.search(r"conv|subsampl|patch|feature_extractor|downsampl", modname.lower()) and not re.search(r"layers\.\d+\.", modname): return "conv"
    if re.search(r"merger|projector|multi_modal|mm_proj|adapter|connector|mlp1", modname.lower()): return "adapter"
    return "other"


def gt_for(model, cfg):
    mods = dict(model.named_modules())
    params = {}
    seen = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen: continue
        seen[id(p)] = n; params[n] = p
    sd = model.state_dict()
    bufs = {n: b for n, b in model.named_buffers() if n in sd and n not in params}
    tensors, gt = {}, {}
    for n, t in list(params.items()) + list(bufs.items()):
        tensors[n] = {"dtype": DT.get(t.dtype, str(t.dtype)), "shape": list(t.shape)}
        parts = n.split(".")
        chain = []
        for i in range(len(parts)):
            m = mods.get(".".join(parts[:i]))
            if m is not None: chain.append(type(m).__name__)
        k = kind_of(chain, parts[-1], ".".join(parts[:-1]))
        if k != "norm" and re.search(r"norm|^ln", ".".join(parts[-2:]).lower()) and len(t.shape) == 1: k = "norm"
        gt[n] = {"kind": k, "owner": chain[-1] if chain else ""}
    # block lists: ModuleLists with >=2 composite children; expert containers excluded
    lists = []
    for name, m in mods.items():
        if not isinstance(m, (torch.nn.ModuleList, torch.nn.Sequential)) or len(m) < 2: continue
        kids = list(m)
        if not all(sum(1 for _ in k.parameters()) >= 2 for k in kids): continue
        parent = mods.get(name.rsplit(".", 1)[0] if "." in name else "")
        pc = type(parent).__name__.lower() if parent is not None else ""
        kc = type(kids[0]).__name__.lower()
        expert = bool(name.endswith("experts") or re.search(r"expert", kc) or (re.search(MOE, pc) and not re.search(r"layer|block", kc)))
        sig = []
        for k in kids:
            comp = sorted(type(x).__name__ for x in k.modules())
            sig.append(type(k).__name__ + "|" + ",".join(comp))
        uniq = {s: i for i, s in enumerate(dict.fromkeys(sig))}
        lists.append({"path": name, "n": len(kids), "expert": expert, "types": [uniq[s] for s in sig], "cls": type(kids[0]).__name__})
    # innermost non-expert lists = blocks
    nonexp = [l for l in lists if not l["expert"]]
    for l in nonexp:
        l["inner"] = not any(o is not l and o["path"].startswith(l["path"] + ".") for o in nonexp)
    return tensors, gt, lists


def hparams(cfg):
    tc = getattr(cfg, "text_config", None) or getattr(cfg, "llm_config", None) or getattr(cfg, "decoder", None) or getattr(cfg, "encoder_config", None)
    if tc is not None and not hasattr(tc, "hidden_size") and isinstance(tc, dict): tc = None
    out = {}
    def g(*keys):
        for src in ([tc] if tc is not None else []) + [cfg]:
            for k in keys:
                v = getattr(src, k, None)
                if isinstance(v, (int, float)) and not isinstance(v, bool): return v
        return None
    out["d"] = g("hidden_size", "d_model", "n_embd", "dim", "embed_dim")
    out["layers"] = g("num_hidden_layers", "n_layer", "num_layers", "decoder_layers")
    out["heads"] = g("num_attention_heads", "n_head", "num_heads", "decoder_attention_heads")
    out["kv_heads"] = g("num_key_value_heads", "num_kv_heads", "n_kv_heads", "multi_query_group_num")
    out["head_dim"] = g("head_dim", "kv_channels", "attention_head_dim")
    out["ffn"] = g("intermediate_size", "ffn_dim", "d_ff", "n_inner", "decoder_ffn_dim")
    out["vocab"] = g("vocab_size")
    out["experts"] = g("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts")
    out["topk"] = g("num_experts_per_tok", "moe_topk", "top_k", "num_selected_experts")
    out["tie"] = getattr(tc if tc is not None else cfg, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", None))
    return out


class TO(Exception): pass
def alarm(*_): raise TO()


def one(mt, cfg=None, name=None):
    if cfg is None: cfg = CONFIG_MAPPING[mt]()
    cls = model_class(cfg.model_type if hasattr(cfg, "model_type") else mt)
    if cls is None: return {"type": mt, "err": "no class"}
    signal.signal(signal.SIGALRM, alarm); signal.alarm(90)
    try:
        with torch.device("meta"):
            m = cls._from_config(cfg) if hasattr(cls, "_from_config") else cls(cfg)
        tensors, gt, lists = gt_for(m, cfg)
    finally:
        signal.alarm(0)
    return {"type": mt, "name": name or mt, "cls": cls.__name__, "n": len(tensors), "params": sum(int(torch.Size(v["shape"]).numel()) for v in tensors.values()),
            "tensors": tensors, "gt": gt, "lists": lists, "hp": hparams(cfg)}


if __name__ == "__main__":
    out = sys.argv[1]
    only = None
    if "--only" in sys.argv: only = sys.argv[sys.argv.index("--only") + 1].split(",")
    if "--config-json" in sys.argv:
        p = sys.argv[sys.argv.index("--config-json") + 1]; nm = sys.argv[sys.argv.index("--name") + 1]
        d = json.load(open(p)); cfg = AutoConfig.for_model(**d) if "model_type" in d else None
        r = one(d["model_type"], cfg, nm)
        open(out, "w").write(json.dumps(r) + "\n"); print(nm, r.get("n"), r.get("params")); sys.exit()
    types = only or list(CONFIG_MAPPING.keys())
    done = set()
    if os.path.exists(out) and not only:
        for line in open(out): done.add(json.loads(line)["type"])
    f = open(out, "a" if not only else "w")
    for i, mt in enumerate(types):
        if mt in done: continue
        try: r = one(mt)
        except TO: r = {"type": mt, "err": "timeout"}
        except Exception as e: r = {"type": mt, "err": (type(e).__name__ + ": " + str(e))[:200]}
        f.write(json.dumps(r) + "\n"); f.flush()
        print(i, mt, r.get("n", r.get("err")), flush=True)
