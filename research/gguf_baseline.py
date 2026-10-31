# Build the llama.cpp name-mapping baseline: read gguf-py/gguf/tensor_mapping.py as data (AST, nothing executed),
# turn every Hugging Face module path it lists into a regex, and map each llama.cpp tensor type to one of our roles.
# usage: python gguf_baseline.py tensor_mapping.py gguf_roles.json
# Source: https://github.com/ggml-org/llama.cpp (MIT), gguf-py/gguf/tensor_mapping.py
import ast, json, re, sys

src = open(sys.argv[1], encoding="utf-8").read()
tree = ast.parse(src)
tables = {}
for node in ast.walk(tree):
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in ("mappings_cfg", "block_mappings_cfg", "arch_block_mappings_cfg"):
        tables[node.target.id] = node.value

def tensor_of(k):
    return k.attr if isinstance(k, ast.Attribute) else None

pairs = []  # (pattern, MODEL_TENSOR name)
def collect(d):
    for k, v in zip(d.keys, d.values):
        t = tensor_of(k)
        if t is None or not isinstance(v, (ast.Tuple, ast.List)): continue
        for e in v.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str): pairs.append((e.value, t))
collect(tables["mappings_cfg"]); collect(tables["block_mappings_cfg"])
for k, v in zip(tables["arch_block_mappings_cfg"].keys, tables["arch_block_mappings_cfg"].values):
    if isinstance(v, ast.Dict): collect(v)  # union over architectures: the baseline is not told the architecture

# llama.cpp tensor type -> our role (first match wins)
RULES = [
    (r"^A_ENC_POS_BIAS|REL_B$|_REL_POS|ATTN_K_REL", "attn"),
    (r"^(NEXTN_|DFLASH_|DSPARK_|DECISION_|LILICORR_|HC_|ALTUP_|LAUREL_|ROPE_|MASKED_EMBD|PER_LAYER_|PLE_|V_TOK_|V_STD|A_ENC_MEL)", "other"),
    (r"NORM|_LN\d?$|_LN_|^ATTN_SUB_NORM|LAYER_SCALE|LAYER_OUT_SCALE|_GAMMA$|_SCALE(_\d)?$", "norm"),
    (r"(_EXP|_EXPS|_SHEXP|_CHEXP|GATE_INP|EXP_PROBS|ROUTED|MOE_LATENT)", "moe"),
    (r"(CROSS_ATTN|QF_CROSS)", "cross"),
    (r"^(SSM_|TIME_MIX_|SHORTCONV_)", "ssm"),
    (r"^(V_MMPROJ|V_MM_|A_MMPROJ|A_MM_|V_MERGER|V_RESMPL|V_MULTI_PROJ|V_DS_|A_QF_PROJ|V_SAM_NECK|V_SAM_NET)", "adapter"),
    (r"(ATTN|_Q$|_K$|_V$|INDEXER|VISEXP_ATTN)", "attn"),
    (r"(FFN|CHANNEL_MIX|_MLP|^FC$|_UP$|_DOWN$|_GATE$|VISEXP_)", "ffn"),
    (r"(CONV|POSNET|CONVNEXT|A_ENC_SE)", "conv"),
    (r"^(TOKEN_EMBD|TOKEN_TYPES|POS_EMBD|V_ENC_EMBD|A_ENC_EMBD|V_ENC_CONV_STEM|A_ENC_INP_PROJ|V_SAM_PATCH|V_SAM_POS|A_ENC_LINEAR_POS)", "embed"),
    (r"^(OUTPUT|CLS|CLS_OUT|A_CTC_OUT|DENSE_\d_OUT|A_ENC_OUT|A_ENC_EMBD_TO_LOGITS)", "head"),
]
def role(t):
    for rx, r in RULES:
        if re.search(rx, t): return r
    return "other"

out, seen = [], set()
for pat, t in pairs:
    r = role(t)
    rx = "^" + re.escape(pat).replace(r"\{bid\}", r"\d+").replace(r"\{i\}", r"\d+").replace(r"\{\}", r"\d+") + r"(\.(weight|bias|scale|[A-Za-z_]+))?$"
    if (rx, r) in seen: continue
    seen.add((rx, r)); out.append([rx, r, t, pat])
# conv layers inside blocks: the llama.cpp names are matched whole, as llama.cpp does
json.dump({"source": "ggml-org/llama.cpp gguf-py/gguf/tensor_mapping.py (master, read 2026-11-01)", "patterns": out}, open(sys.argv[2], "w"), indent=0)
from collections import Counter
print(len(out), "patterns;", Counter(x[1] for x in out))
