"""Update the metadata and citation of a published dataset card in place, without re-running the census.

usage: python card_update.py <census|corpus> <owner/dataset>     (needs HF_TOKEN with write access)
"""
import os, sys

from card_meta import CITATION, CITE_LINE, META


def update(card, kind):
    _, front, body = card.split("---\n", 2)
    keep = [l for l in front.splitlines() if l.split(":", 1)[0] not in META[kind]]
    at = next(i for i, l in enumerate(keep) if l.startswith(("size_categories", "configs")))
    keep[at:at] = [f"{k}: {v}" for k, v in META[kind].items()]
    body = body.split("\n## Citation\n")[0].rstrip() + "\n\n" + CITATION
    if "**Citation.**" not in body:
        i = body.find("\n## ")
        body = body[:i] + "\n" + CITE_LINE + "\n" + body[i:]
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
