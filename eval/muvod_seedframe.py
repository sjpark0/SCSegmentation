#!/usr/bin/env python3
"""How much of the MUVOD score comes from c_ini's given frame.

MUVOD does not say whether the reference frame -- the one whose mask the model is
handed -- is scored.  We score all 21 frames.  This recomputes the basic J&F^3 with
that one frame dropped from c_ini only, so the write-up can state the size of the
ambiguity instead of ignoring it.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/sjpark/Documents/SCSegmentation")
from eval.report_jf import Store, load, Cfg, MUVOD_CFG, MUVOD_ORDER, mean   # noqa

raw = sys.argv[1] if len(sys.argv) > 1 else "Data/MVSeg/jf_muvod.json"
methods = sys.argv[2:] or ["SegMaskSam3XW1GPS4M", "SegMaskSam3XW0M"]
store = Store(load([raw]))
cfg = Cfg(**MUVOD_CFG).replace(objects="basic")

def cam_score(d, cam, m, drop_seed):
    e = store.entry(d, cam, m)
    if e is None:
        return None
    keep = store.object_filter(d, cfg)
    pf, meta = e["result"]["per_frame"], e["meta"]
    frames = meta["frames"]
    idx = [i for i in range(len(frames))
           if not (drop_seed and frames[i] == meta["start_frame"])]
    js, fs = [], []
    for i, o in enumerate(e["objects"]):
        if keep is not None and o not in keep:
            continue
        js.append(mean([pf["J"][i][k] for k in idx]))
        fs.append(mean([pf["F"][i][k] for k in idx]))
    if not js:
        return None
    return (mean(js) + mean(fs)) / 2

print(f"{'scene':22} " + "  ".join(f"{m[-14:]:>16}" for m in methods))
tot = {m: {"all": [], "drop": []} for m in methods}
for d in [x for x in MUVOD_ORDER if x in store.datasets]:
    ref = store.ref_entry(d, cfg.ref_rule)["cam"]
    line = f"{d:22} "
    for m in methods:
        a, b = [], []
        for cam in store.cams[d]:
            sa = cam_score(d, cam, m, False)
            sb = cam_score(d, cam, m, cam == ref)
            if sa is not None:
                a.append(sa); b.append(sb)
        if not a:
            line += f"{'-':>16}  "; continue
        A, B = mean(a) * 100, mean(b) * 100
        tot[m]["all"].append(A); tot[m]["drop"].append(B)
        line += f"{A:8.2f}{B:8.2f}  "
    print(line)
print(f"\n{'':22} " + "  ".join(f"{'all / no seed':>16}" for _ in methods))
print(f"{'GLOBAL':22} " + "  ".join(
    f"{mean(tot[m]['all']):8.2f}{mean(tot[m]['drop']):8.2f}  " for m in methods))
for m in methods:
    d = mean(tot[m]["all"]) - mean(tot[m]["drop"])
    print(f"{m}: dropping c_ini's given frame costs {d:+.3f} points")
