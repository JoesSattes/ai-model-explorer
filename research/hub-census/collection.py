"""Group the project's datasets (and anything else listed) into one Hugging Face collection on the owner's profile.

usage: python collection.py <owner> <dataset> [<dataset> ...]      (needs HF_TOKEN with write access)
"""
import os, sys

TITLE = "AI Model Explorer"
DESC = ("Reading a model's architecture from its safetensors header: the labelled corpus, the Hub census, "
        "and the playground at joessattes.github.io/ai-model-explorer")


def main(owner, datasets):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    col = next((c for c in api.list_collections(owner=owner, limit=100) if c.title == TITLE), None)
    if col is None:
        col = api.create_collection(TITLE, namespace=owner, description=DESC, private=False)
    have = {i.item_id for i in api.get_collection(col.slug).items}
    for d in datasets:
        if d not in have:
            api.add_collection_item(col.slug, item_id=d, item_type="dataset", exists_ok=True)
    print("collection https://huggingface.co/collections/" + col.slug)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
