"""Copy a run's insights into the website's census page data and mark it final."""
import json, os, shutil, sys

out, site, dataset_url = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(site, exist_ok=True)
d = json.load(open(os.path.join(out, "insights", "insights.json")))
d["preliminary"] = False
d["dataset_url"] = dataset_url
json.dump(d, open(os.path.join(site, "insights.json"), "w"), indent=1)
shutil.copy(os.path.join(out, "insights", "models.json"), os.path.join(site, "models.json"))
print("site data updated")
