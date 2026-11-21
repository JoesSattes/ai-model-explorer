"""Update the metadata and citation of a published dataset card in place, without re-running the census.

usage: python card_update.py <census|corpus> <owner/dataset>     (needs HF_TOKEN with write access)
"""
import os, re, sys

from card_meta import CITATION, CITE_LINE, META, PAPER_URL


def update(card, kind):
    _, front, body = card.split("---\n", 2)
    keep = [l for l in front.splitlines() if l.split(":", 1)[0] not in META[kind]]
    at = next(i for i, l in enumerate(keep) if l.startswith(("size_categories", "configs")))
    keep[at:at] = [f"{k}: {v}" for k, v in META[kind].items()]
    body = body.split("\n## Citation\n")[0].rstrip() + "\n\n" + CITATION
    if "**Citation.**" not in body:
        i = body.find("\n## ")
        body = body[:i] + "\n" + CITE_LINE + "\n" + body[i:]
    else:
        body = re.sub(r"^\*\*Citation\.\*\*.*$", lambda m: CITE_LINE, body, count=1, flags=re.M)
    # point every ResearchGate record link (old and current) at the paper's DOI
    body = re.sub(r"https?://www\.researchgate\.net/publication/(417293353|417926447)[\w.-]*", PAPER_URL, body)
    # the ResearchGate DOIs used before the SSRN and arXiv records
    body = body.replace("https://doi.org/10.13140/RG.2.2.19473.19048", PAPER_URL).replace("10.13140/RG.2.2.19473.19048", "10.2139/ssrn.7699619")
    body = body.replace("https://doi.org/10.13140/RG.2.2.10665.15200", "https://arxiv.org/abs/2611.19913").replace("10.13140/RG.2.2.10665.15200", "10.48550/arXiv.2611.19913")
    return "---\n" + "\n".join(keep) + "\n---\n" + body


def main(kind, repo):
    from huggingface_hub import HfApi, hf_hub_download
    path = hf_hub_download(repo, "README.md", repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    card = update(open(path, encoding="utf-8").read(), kind)
    HfApi(token=os.environ.get("HF_TOKEN")).upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
        repo_id=repo, repo_type="dataset", commit_message="card: task categories, tags and citation")
    print("updated https://huggingface.co/datasets/" + repo)


if __name__ == "__main__":
    main(*sys.argv[1:3])
