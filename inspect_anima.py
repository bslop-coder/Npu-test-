import collections
import json
import re
import struct

import requests
from huggingface_hub import HfApi, hf_hub_url

repo = "circlestone-labs/Anima"
files = HfApi().list_repo_files(repo)
print("FILES")
for f in files:
    print(f)

for f in files:
    if not f.endswith(".safetensors"):
        continue
    url = hf_hub_url(repo, f)
    r = requests.get(url, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", r.content[:8])[0]
    r = requests.get(url, headers={"Range": "bytes=8-%d" % (7 + n)})
    h = json.loads(r.content)
    groups = collections.OrderedDict()
    total = 0
    for k, v in h.items():
        if k == "__metadata__":
            continue
        key = (re.sub(r"\.\d+(?=\.|$)", ".N", k), v["dtype"], tuple(v["shape"]))
        groups[key] = groups.get(key, 0) + 1
        p = 1
        for s in v["shape"]:
            p *= s
        total += p
    print("=====", f, "params %.2fB" % (total / 1e9))
    for (k, d, s), c in groups.items():
        print("%dx %s %s %s" % (c, k, d, list(s)))
