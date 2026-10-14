"""How much of C0's missed GT can an add-only shell reach at all? missed_b = GT_b & ~C0_b;
U = pixels claimed by no C0 mask of the view (scored objects), conn = 8-connected to C0_b through (C0_b | missed_b & U).
Pair-mean J&F if every missed pixel in U (resp. U & connected) were added (no FP), vs all missed.
Host numpy/cv2 only; INV not applied (so this is an upper bound for INV & U)."""
import json, os, sys, statistics as st
import cv2, numpy as np
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from mvseg2d.io import load_labelmap, load_meta, read_pred_mask
from mvseg2d.metrics import bound_pix, disk, jf_pair
DATA = os.path.join(REPO, "stage1", "Data"); C0 = "geoset_pred_ft_rep_x_e11"
rows = json.load(open(os.path.join(REPO, "MVSeed", "runs", "seedfail.json")))["rows"]
by = {}
for r in rows: by.setdefault((r["scene"], r["cam"]), []).append(r)
out = []
for (scene, cam), rs in sorted(by.items()):
    meta = load_meta(DATA, scene)
    gt_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{cam}.png")); hw = gt_lab.shape
    bp = bound_pix(*hw); se = disk(bp)
    def rd(o):
        p = os.path.join(DATA, scene, "Result", C0, cam, f"{o}.png")
        return read_pred_mask(p) if os.path.isfile(p) else np.zeros(hw, bool)
    objs = sorted({r["obj"] for r in rs}); c0 = {o: rd(o) for o in objs}
    U = ~np.logical_or.reduce([c0[o] for o in objs])
    for r in rs:
        o = r["obj"]; gt = gt_lab == o; c = c0[o]
        miss = gt & ~c; mU = miss & U
        if c.any() and mU.any():
            n, lab = cv2.connectedComponents((c | mU).astype(np.uint8), connectivity=8)
            keep = np.isin(lab, np.unique(lab[c])); mUc = mU & keep
        else:
            mUc = np.zeros_like(mU)
        J0, F0 = jf_pair(gt, c, bp, se)
        Ja, Fa = jf_pair(gt, c | miss, bp, se)
        Ju, Fu = jf_pair(gt, c | mU, bp, se)
        Jc, Fc = jf_pair(gt, c | mUc, bp, se)
        g = ["good605"] if r["JF"] >= 0.8 else (["fail59", "under22"] if r["kind"].startswith("under") else ["fail59"])
        out.append(dict(g=g + ["all664"], miss=int(miss.sum()), mU=int(mU.sum()), mUc=int(mUc.sum()),
                        jf0=(J0 + F0) / 2, jfa=(Ja + Fa) / 2, jfu=(Ju + Fu) / 2, jfc=(Jc + Fc) / 2,
                        claimed_other=int((miss & ~U).sum())))
for grp in ("all664", "good605", "fail59", "under22"):
    v = [x for x in out if grp in x["g"]]
    m = sum(x["miss"] for x in v); mu = sum(x["mU"] for x in v); muc = sum(x["mUc"] for x in v)
    print(f"{grp:8s} n={len(v):3d} missed px {m:,}  in U {mu/m:.3f}  in U&conn {muc/m:.3f} | pair-mean J&F oracle: all-missed {st.mean(x['jfa']-x['jf0'] for x in v):+.4f}  U {st.mean(x['jfu']-x['jf0'] for x in v):+.4f}  U&conn {st.mean(x['jfc']-x['jf0'] for x in v):+.4f}")
