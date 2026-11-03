"""Labels the website and the Deploy-Ready board show next to a model: who released it (a leading provider or not),
how it relates to its parent (original, fine-tune, quantized copy, merge, adapter), its pipeline and whether it is
built for Thai. All of it comes from Hub tags, read once from the census sample (out/sample.parquet)."""
import os, re

from sample import FRONTIER

LABS = set(FRONTIER)
ISO = set(("aa ab af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co cr cs cu cv cy da de dv dz ee el en eo "
           "es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu ja jv ka kg ki "
           "kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr "
           "nv ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr ss st su sv sw ta te tg th "
           "ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za zh zu").split())
THAI_ID = re.compile(r"typhoon|openthaigpt|pathumma|wangchan|thalle|(^|[/_.-])thai([/_.-]|$)|(^|/)scb10x/|(^|/)nectec/|(^|/)airesearch/|(^|/)pythainlp/|(^|/)biodatlab/", re.I)
SEA_ID = re.compile(r"sea-?lion|seallm|(^|/)sail/sailor|(^|/)aisingapore/", re.I)
UNSAFE = re.compile(r"nsfw|uncensored|abliterat|porn|hentai|lewd|erotic|nude", re.I)
TEXT_PIPES = {None, "", "text-generation", "text2text-generation", "image-text-to-text", "conversational", "any-to-any",
              "audio-text-to-text", "visual-question-answering", "video-text-to-text"}


def tags_by_id(census_path):
    """{repo: [tags]} from the sample next to census.jsonl, or {} when it is not there."""
    p = os.path.join(os.path.dirname(os.path.abspath(census_path)), "sample.parquet")
    if not os.path.exists(p):
        return {}
    import pandas as pd
    s = pd.read_parquet(p, columns=["id", "tags"])
    return {i: [str(x) for x in (t if t is not None else [])] for i, t in zip(s["id"], s["tags"])}


def langs(tags):
    return [t for t in tags if t in ISO]


TEXT_FAMS = {"decoder", "prefix-lm", "encoder", "encoder-decoder", "transducer"}


def thai(repo, tags, fam=None):
    """'TH', 'TH/EN', 'Multilingual' for language and speech models that read or write Thai, else None.
    Image generators and unreadable repos are left out even when their card lists Thai."""
    if fam is not None and fam not in TEXT_FAMS or not safe(repo):
        return None
    ls = set(langs(tags))
    multi = "multilingual" in tags or len(ls) >= 4
    if "th" in ls:
        return "Multilingual" if multi else "TH/EN" if "en" in ls else "TH"
    if SEA_ID.search(repo):
        return "Multilingual"
    if THAI_ID.search(repo):
        return "TH/EN" if "en" in ls else "TH"
    return None


def hub_origin(tags):
    """The Hub's own relation tag, as frame.py reads it."""
    rel = [t.split(":")[1] for t in tags if t.startswith("base_model:") and t.count(":") >= 2]
    return next((r for r in ("quantized", "adapter", "merge", "finetune") if r in rel), "original")


def parents(tags):
    return [t.split(":", 2)[2] for t in tags if t.startswith("base_model:") and t.count(":") >= 2]


def origin(o, quant, repo="", tags=()):
    """How the repo relates to its parent. Two corrections to the Hub's base_model tag:
    a derivative of the same owner's model (an instruct release of its own base) is that owner's release, so 'original';
    the Hub says 'quantized' when the repo carries any quantized file (an ONNX export, say), so when the safetensors we
    read are not quantized the repo is a fine-tune or re-release of its parent, not a quantized copy."""
    o = o or "original"
    ps = parents(tags)
    if o != "original" and ps and all(p.split("/")[0].lower() == repo.split("/")[0].lower() for p in ps):
        return "original"
    if o == "quantized" and not quant:
        return "finetune"
    return o


def license_of(tags):
    return next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)


def is_lab(repo):
    return repo.split("/")[0] in LABS


def safe(repo):
    return not UNSAFE.search(repo)
