"""Read-only, inputs only (c_ini mask at frame t + Pi3 of frame t; no target-view GT): distribution of per-(view, object)
co-visibility on MUVOD DEV-A frames, i.e. how often a co-visibility router would fire on the training-domain rigs."""
import json, os, sys
import numpy as np
from PIL import Image
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points
from geomem.devset import dev_frames, basic_set
D = os.path.join(REPO, "stage1/Data")
scenes = json.load(open(os.path.join(REPO, "stage1/geomem/folds.json")))
scenes = sorted(scenes if isinstance(scenes, dict) else scenes)
allv = {}
for s in scenes:
    meta = json.load(open(f"{D}/{s}/meta.json")); ref = meta["ref_cam"]; objs = basic_set(s)
    vals = []
    for t in dev_frames(D, s, "A"):
        C = f"{D}/{s}/Cache/pi3_frames/{t:06d}"
        mp = f"{D}/{s}/frames/{t:06d}/masks/{ref}.png"
        if not (os.path.isdir(C) and os.path.isfile(mp)): continue
        zr = np.load(f"{C}/{ref}.npz"); d_r, K_r, T_r = zr["depth"], zr["K"], zr["c2w"]; h, w = d_r.shape
        vr = (zr["conf"] > CONF_MIN) & (d_r > 0); Pr = local_points(d_r, K_r)
        lab = np.array(Image.fromarray(np.array(Image.open(mp))).resize((w, h), Image.NEAREST))
        for c in meta["cams"]:
            if c == ref: continue
            z = np.load(f"{C}/{c}.npz"); d_c, K_c, T_c = z["depth"], z["K"], z["c2w"]
            M = np.linalg.inv(T_c) @ T_r
            for o in objs:
                sel = (lab == o) & vr
                if sel.sum() < 5: continue
                X = Pr[sel] @ M[:3, :3].T + M[:3, 3]; Z = X[:, 2]; Zs = np.where(np.abs(Z) < 1e-9, 1e-9, Z)
                u = K_c[0, 0] * X[:, 0] / Zs + K_c[0, 2]; v = K_c[1, 1] * X[:, 1] / Zs + K_c[1, 2]
                ins = (Z > 0) & (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5)
                dd = d_c[np.clip(np.round(v), 0, h - 1).astype(int), np.clip(np.round(u), 0, w - 1).astype(int)]
                vals.append(float((ins & ((dd <= 0) | (Z <= dd * 1.05))).mean()))
    a = np.array(vals); allv[s] = vals
    if a.size:
        print(f"{s:20s} n={a.size:5d} frac<0.25 {np.mean(a<0.25):.2f} frac<0.5 {np.mean(a<0.5):.2f} median {np.median(a):.2f}")
A = np.concatenate([np.array(v) for v in allv.values() if v])
print(f"ALL MUVOD DEV-A (view, object) n={A.size}: frac<0.25 {np.mean(A<0.25):.3f}, frac<0.5 {np.mean(A<0.5):.3f}")
