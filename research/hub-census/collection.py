"""Group the project's datasets (and anything else listed) into one Hugging Face collection on the owner's profile.

usage: python collection.py <owner> <type:repo> [<type:repo> ...]   e.g. space:me/ai-model-explorer dataset:me/x
"""
import os, sys

TITLE = "AI Model Explorer"
DESC = "Model anatomy from safetensors headers: corpus, Hub census, vocabulary tax and the explorer. By AI Model Explorer Lab."  # HF limit: 150 chars


def main(owner, items):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    col = next((c for c in api.list_collections(owner=owner, limit=100) if c.title == TITLE), None)
    if col is None:
        col = api.create_collection(TITLE, namespace=owner, description=DESC[:150], private=False)
    elif col.description != DESC[:150]:
        api.update_collection_metadata(col.slug, description=DESC[:150])
    have = {i.item_id for i in api.get_collection(col.slug).items}
    for it in items:
        kind, rid = it.split(":", 1) if ":" in it else ("dataset", it)
        if rid not in have:
            api.add_collection_item(col.slug, item_id=rid, item_type=kind, exists_ok=True)
    print("collection https://huggingface.co/collections/" + col.slug)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
