import json, os, sys
import numpy as np
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from mvseg2d.io import load_meta, load_labelmap, labelmap_objects  # noqa
from methods.common import frame_dir  # noqa
DATA = os.path.join(REPO, "stage1", "Data")
def cells(lab, o):
    h, w = lab.shape
    l = lab[(np.arange(1008) * h // 1008)][:, (np.arange(1008) * w // 1008)]
    return int(((l == o).reshape(72, 14, 72, 14).mean(axis=(1, 3)) >= 0.5).sum())
sf = json.load(open(os.path.join(REPO, "MVSeed", "runs", "seedfail.json")))["rows"]
kv = json.loads(open(sys.argv[1]).read()[open(sys.argv[1]).read().index("{"):])["rows"]
kvk = {(r["scene"], r["cam"], r["obj"]): r for r in kv}
labs = {}
n = {"all": [0, 0], "good": [0, 0], "fail": [0, 0], "under": [0, 0], "inv20": [0, 0]}
for r in sf:
    sc = r["scene"]
    if sc not in labs:
        m = load_meta(DATA, sc); labs[sc] = load_labelmap(os.path.join(frame_dir(DATA, sc, int(m["ref_frame"])), "masks", f"{m['ref_cam']}.png"))
    e = cells(labs[sc], r["obj"]) >= 20
    k = (sc, r["cam"], r["obj"]); v = kvk.get(k)
    inv = None if v is None else (v["gt_outside"] + v["gt_occluded"] + v["gt_nodepth"]) / max(v["gt_px"], 1)
    groups = ["all", "good" if r["JF"] >= 0.8 else "fail"] + (["under"] if r["JF"] < 0.8 and r.get("kind", "").startswith("under") else []) \
        + (["inv20"] if inv is not None and inv >= 0.2 else [])
    for g in groups:
        n[g][0] += 1; n[g][1] += e
print({g: f"{b}/{a} eligible (>= 20 ref cells)" for g, (a, b) in n.items()})
