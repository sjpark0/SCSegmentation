"""T06 frame 900 (declared contaminated dev): per-camera viewpoint angle (scene / object-union anchors) vs per-view J of
preset, preset_r1, chain (scores_raw rows of ring_diag/d1_d5). Also: which already-decoded views would be relay memory."""
import json, os, sys
import numpy as np
from PIL import Image
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points
C = os.path.join(REPO, "stage1/Data/T06_flag_ganggang_F/Cache/pi3")
ref = "cam_06"
z = np.load(f"{C}/{ref}.npz"); T_r = z["c2w"]; valid = (z["conf"] > CONF_MIN) & (z["depth"] > 0)
P = local_points(z["depth"], z["K"]); h, w = valid.shape
lab = np.array(Image.open(os.path.join(REPO, "stage1/Data/T06_flag_ganggang_F/masks/cam_06.png")))
labs = np.array(Image.fromarray(lab).resize((w, h), Image.NEAREST))
Xs = T_r[:3, :3] @ np.median(P[valid], 0) + T_r[:3, 3]
Xo = T_r[:3, :3] @ np.median(P[(labs > 0) & valid], 0) + T_r[:3, 3]
def ang(X, T):
    u, v = T_r[:3, 3] - X, T[:3, 3] - X
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
rows = json.load(open(os.path.join(REPO, "stage1/scripts/ring_diag/d1_d5/scores_raw.json")))["rows"]
J = {k: {} for k in ("preset", "preset_r1", "nas_chain")}
for k in J:
    for r in rows[k]:
        J[k].setdefault(r["cam"], []).append((r["J_full"], r["covers"], r["p"], r["d"]))
cent = {}
for i in range(32):
    c = f"cam_{i:02d}"
    if c == ref: continue
    T = np.load(f"{C}/{c}.npz")["c2w"]; cent[c] = T[:3, 3]
    a_s, a_o = ang(Xs, T), ang(Xo, T)
    d = J["preset"][c][0][3]
    def s(k): 
        v = J[k][c]; return f"{np.mean([x[0] for x in v]):.3f}/{sum(x[1] not in (0, x[2]) for x in v)}"
    print(f"{c} d{d:2d} scene {a_s:6.1f} obj {a_o:6.1f}  r1 {s('preset_r1')}  preset {s('preset')}  chain {s('nas_chain')}")
