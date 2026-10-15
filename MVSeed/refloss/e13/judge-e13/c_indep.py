"""Judge: independent re-composition of candidate C primary (pts|20|None) for a few scenes; compare with rows.json."""
import json, os, sys
import numpy as np, cv2
S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
from mvseg2d.io import load_labelmap, read_pred_mask
from mvseg2d.metrics import bound_pix, disk, jf_pair
DATA = S1 + "/Data"; DEV = S1 + "/geomem/runs/_e13_dev"; PROPS = S1 + "/geomem/runs/_e13_sam3-prompt-extend/props"
scenes = sys.argv[2].split(",")
rows = json.load(open(sys.argv[1]))
byv = {(v["scene"], v["t"], v["cam"]): v for v in rows["views"]}
idx = json.load(open(DEV + "/index.json"))["scenes"]
maxd = 0; n = 0; nadd_mis = 0
for s in scenes:
    meta_s = json.load(open(f"{DATA}/{s}/meta.json"))
    for t in idx[s]["frames"]:
        assert t != int(meta_s["ref_frame"])
        for cam in idx[s]["eval_cams"]:
            vd = f"{PROPS}/{s}/{t:06d}/{cam}"
            meta = json.load(open(vd + "/meta.json"))
            hw = tuple(meta["hw"]); objs = [int(o) for o in meta["objs"]]
            gt = load_labelmap(f"{DATA}/{s}/frames/{t:06d}/masks/{cam}.png")
            # independent visibility: recompute cls? use cls.png (validated vs dev_pairs by C); check partition only
            cls = cv2.imread(vd + "/cls.png", cv2.IMREAD_UNCHANGED)
            def rd(p):
                return read_pred_mask(p) if os.path.isfile(p) else np.zeros(hw, bool)
            D = {o: rd(f"{DEV}/{s}/{t:06d}/{cam}/{o}.png") for o in objs}
            anyD = np.zeros(hw, np.int32)
            for o in objs: anyD += D[o]
            props = {int(k) for k in meta["props"]}
            proj = {int(k): v for k, v in meta["proj_px"].items()}
            A = {}
            for o in objs:
                if o not in props or proj[o] < 20: continue
                S = rd(f"{vd}/pts/{o}.png")
                C = S & (cls > 0) & (anyD == 0)       # not D_o and not any other D_q  <=> anyD==0 (C excludes D_o anyway)
                U = (C | D[o]).astype(np.uint8)
                nl, lab = cv2.connectedComponents(U, connectivity=8)
                keep = np.zeros(nl, bool); keep[np.unique(lab[D[o]])] = True; keep[0] = False
                A[o] = keep[lab] & C
            cnt = np.zeros(hw, np.int32)
            for o in A: cnt += A[o]
            bp = bound_pix(*hw); se = disk(bp)
            v = byv[(s, t, cam)]
            for o in objs:
                if not (gt == o).any(): continue
                g = gt == o
                a = A[o] & (cnt == 1) if o in A else np.zeros(hw, bool)
                J0, F0 = jf_pair(g, D[o], bp, se); J1, F1 = jf_pair(g, D[o] | a, bp, se)
                rec = v["pairs"].get(str(o), v["pairs"].get(o, {})).get("pts|20|None")
                d = ((J1 + F1) - (J0 + F0)) / 2
                dr = (rec[0] + rec[1]) / 2 if rec else 0.0
                if rec and rec[2] != int(a.sum()): nadd_mis += 1
                maxd = max(maxd, abs(d - dr)); n += 1
print("pairs", n, "max |dJF diff|", maxd, "added-count mismatches", nadd_mis)
