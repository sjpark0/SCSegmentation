"""Read-only, inputs only (no target GT): viewpoint angle of every camera vs c_ini on DEV-A frames.
anchor 'scene' = median of c_ini valid Pi3 points (ring diagnosis definition); anchor 'obj' = median of c_ini points inside
the basic-set reference masks (union). Also optical-axis angle."""
import json, os, sys
import numpy as np
from PIL import Image
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem import devset
from geomem.pointmap import CONF_MIN, local_points
D = os.path.join(REPO, "stage1/Data")
scenes = sorted(s for s in os.listdir(D) if os.path.isfile(os.path.join(D, s, "meta.json")) and not s.startswith("T06"))
out = {}
def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
for s in scenes:
    m = json.load(open(os.path.join(D, s, "meta.json"))); ref = m["ref_cam"]
    evals = devset.eval_cams(D, s); basic = devset.basic_set(s)
    rows = []
    for t in devset.dev_frames(D, s, "A"):
        C = os.path.join(D, s, "Cache/pi3_frames", f"{t:06d}")
        if not os.path.isdir(C):
            continue
        z = np.load(os.path.join(C, f"{ref}.npz")); T_r = z["c2w"]
        valid = (z["conf"] > CONF_MIN) & (z["depth"] > 0)
        P = local_points(z["depth"], z["K"])
        lab = np.array(Image.open(os.path.join(D, s, "frames", f"{t:06d}", "masks", f"{ref}.png")))
        h, w = valid.shape
        labs = np.array(Image.fromarray(lab).resize((w, h), Image.NEAREST))
        objm = np.isin(labs, basic) & (labs > 0) & valid
        Xs = T_r[:3, :3] @ np.median(P[valid], 0) + T_r[:3, 3]
        Xo = T_r[:3, :3] @ np.median(P[objm], 0) + T_r[:3, 3] if objm.sum() >= 20 else Xs
        for c in m["cams"]:
            if c == ref: continue
            T = np.load(os.path.join(C, f"{c}.npz"))["c2w"]
            rows.append(dict(t=t, cam=c, ev=c in evals,
                             scene=ang(T_r[:3, 3] - Xs, T[:3, 3] - Xs), obj=ang(T_r[:3, 3] - Xo, T[:3, 3] - Xo),
                             axis=float(np.degrees(np.arccos(np.clip(T_r[:3, 2] @ T[:3, 2], -1, 1))))))
    out[s] = rows
    if not rows: print(s, "no cache"); continue
    for key in ("scene", "obj"):
        a = np.array([r[key] for r in rows]); e = np.array([r[key] for r in rows if r["ev"]])
        print(f"{s:20s} {key:5s} all max {a.max():6.1f} p90 {np.percentile(a,90):6.1f} >40 {np.mean(a>40):.2f} >45 {np.mean(a>45):.2f} >50 {np.mean(a>50):.2f} >55 {np.mean(a>55):.2f} | eval max {e.max():6.1f} min {e.min():6.1f} >40 {np.mean(e>40):.2f} >45 {np.mean(e>45):.2f} >50 {np.mean(e>50):.2f}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "angles_devA.json"), "w"))
