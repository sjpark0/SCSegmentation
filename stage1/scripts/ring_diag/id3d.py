"""Read-only simulation (descriptive, T06 frame 900): would a visibility-independent 3D identity check flag the
learned stage 1's wrong-person masks?  Each predicted mask is lifted with the TARGET view's own Pi3 depth into the
reference frame; its median 3D point is compared with the reference objects' median 3D points (from the input mask).
Flag = nearest reference object != own id.  Also: J after a Hungarian re-assignment of masks to ids by that distance."""
import json, os, sys
import numpy as np
from PIL import Image
from itertools import permutations
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points, scene_scale, to_reference
D = os.path.join(REPO, "stage1/Data/T06_flag_ganggang_F"); C = os.path.join(D, "Cache/pi3")
MASK = os.path.join(REPO, "Data/PlenopticFull/T06_flag_ganggang_F/Mask")
RES = os.path.join(REPO, "PlenopticServer1/CROOM/SegResult/T06_flag_ganggang_F")
CFG = {"lat": "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs", "sub": "SegMaskSam3XW1CGPS4MFsCs10kG2"}
ref = "cam_06"
def load(c):
    z = np.load(f"{C}/{c}.npz"); return z["depth"], z["conf"], z["K"], z["c2w"]
d_r, c_r, K_r, T_r = load(ref); h, w = d_r.shape
vr = (c_r > CONF_MIN) & (d_r > 0); Pr = to_reference(local_points(d_r, K_r), T_r, T_r); sc = scene_scale(Pr, vr)
def small(a): return np.array(Image.fromarray(a).resize((w, h), Image.NEAREST))
labr = small(np.array(Image.open(f"{D}/masks/{ref}.png")))
cent = {o: np.median(Pr[(labr == o) & vr] / sc, 0) for o in range(1, 7)}
def Jfull(a, b):
    u = (a | b).sum(); return (a & b).sum() / u if u else 1.0
res = {k: dict(flag_tp=0, flag_fp=0, wrong=0, correct=0, J_before=[], J_after=[]) for k in CFG}
for i in range(32):
    c = f"cam_{i:02d}"
    if c == ref: continue
    d, cf, K, T = load(c); v = (cf > CONF_MIN) & (d > 0); P = to_reference(local_points(d, K), T, T_r) / sc
    L4 = np.array(Image.open(f"{MASK}/{c}/000900.png")); L = small(L4)
    ids = [o for o in range(1, 7) if (L4 == o).any()]
    for k, cfg in CFG.items():
        preds = {}
        for o in range(1, 7):
            f = f"{RES}/{cfg}/{c}/900/{o}.png"
            m = np.array(Image.open(f)) > 127 if os.path.isfile(f) else np.zeros(L4.shape, bool)
            preds[o] = m
        lifted = {}
        for o in range(1, 7):
            ms = small(preds[o].astype(np.uint8)) > 0
            sel = ms & v
            lifted[o] = np.median(P[sel], 0) if sel.sum() >= 20 else None
        # flags vs truth (truth: which GT person the mask mostly covers, at work res)
        for o in ids:
            ms = small(preds[o].astype(np.uint8)) > 0
            if lifted[o] is None: continue
            cnt = [(ms & (L == j)).sum() for j in range(1, 7)]
            covers = int(np.argmax(cnt)) + 1 if sum(cnt) else 0
            dist = {q: np.linalg.norm(lifted[o] - cent[q]) for q in cent}
            nearest = min(dist, key=dist.get)
            wrong = covers not in (0, o)
            res[k]["wrong" if wrong else "correct"] += 1
            if nearest != o:
                res[k]["flag_tp" if wrong else "flag_fp"] += 1
        # Hungarian (brute force over the present predicted masks) by 3D distance, then J at full res
        objs = [o for o in range(1, 7) if lifted[o] is not None]
        best, bestcost = None, 1e9
        if len(objs) <= 6:
            for perm in permutations(objs):
                cost = sum(np.linalg.norm(lifted[a] - cent[b]) for a, b in zip(objs, perm))
                if cost < bestcost: best, bestcost = perm, cost
        assign = {b: a for a, b in zip(objs, best)} if best else {}
        for o in ids:
            gt = L4 == o
            res[k]["J_before"].append(Jfull(preds[o], gt))
            src = assign.get(o)
            res[k]["J_after"].append(Jfull(preds[src], gt) if src else 0.0)
for k, r in res.items():
    print(f"{k}: masks correct-person {r['correct']}, wrong-person {r['wrong']}; 3D check flags {r['flag_tp']}/{r['wrong']} wrong, "
          f"false flags {r['flag_fp']}/{r['correct']}; mean J {np.mean(r['J_before']):.4f} -> after 3D re-assignment {np.mean(r['J_after']):.4f} "
          f"(J<0.5: {np.sum(np.array(r['J_before'])<0.5)} -> {np.sum(np.array(r['J_after'])<0.5)})")
