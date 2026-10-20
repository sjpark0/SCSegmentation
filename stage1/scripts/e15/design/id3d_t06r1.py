"""T06 frame 900 (contaminated dev): candidate Fix A rules on the preset's ROUND-1 masks (D1 decode in ring_d15)."""
import json, os, sys
import numpy as np, cv2
from PIL import Image
REPO = "/home/sjpark/Documents/SCSegmentation"; sys.path.insert(0, REPO + "/stage1")
from geomem.pointmap import CONF_MIN, local_points, scene_scale, to_reference
SC = "/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad"
R1 = SC + "/ring_d15/data/T06_flag_ganggang_F/Result/preset_r1"
s = "T06_flag_ganggang_F"; D = REPO + "/stage1/Data"; C = f"{D}/{s}/Cache/pi3"; ref = "cam_06"; K3 = np.ones((3, 3), np.uint8)
def load(c):
    z = np.load(f"{C}/{c}.npz"); return z["depth"], z["conf"], z["K"], z["c2w"]
def to_work(m, hw): return cv2.resize(m.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_AREA) >= 0.5
def lift(mw, P, v):
    sel = (cv2.erode(mw.astype(np.uint8), K3) > 0) & v
    if sel.sum() < 20: sel = mw & v
    return P[sel]
d, cf, K, T_r = load(ref); hw = d.shape; v = (cf > CONF_MIN) & (d > 0); P = to_reference(local_points(d, K), T_r, T_r); sc = scene_scale(P, v); P /= sc
labr = np.array(Image.open(f"{D}/{s}/masks/{ref}.png")); M = {}
for o in range(1, 7):
    R = lift(to_work(labr == o, hw), P, v); c = np.median(R, 0); M[o] = (c, float(np.percentile(np.linalg.norm(R - c, axis=1), 90)))
rows = json.load(open(REPO + "/stage1/scripts/ring_diag/d1_d5/scores_raw.json"))["rows"]["preset_r1"]
dd = {(r["cam"], r["p"]): r for r in rows}
def status(q, n, o, beta=1.0, kappa=1.25, sigma=0.05):
    if n < 20: return "abstain"
    dist = {j: np.linalg.norm(q - M[j][0]) for j in M}
    own_in = dist[o] <= beta * M[o][1] + sigma
    nearer = any(kappa * dist[j] < dist[o] for j in M if j != o)
    return "pass" if own_in and not nearer else ("flag" if (not own_in) and nearer else "abstain")
from collections import Counter
for P_ in ((1.0, 1.25, 0.05), (1.0, 1.25, 0.03), (1.5, 1.25, 0.03), (1.0, 1.0, 0.0)):
    cnt = Counter()
    for i in range(32):
        cam = f"cam_{i:02d}"
        if cam == ref: continue
        dv, cv, Kv, Tv = load(cam); vv = (cv > CONF_MIN) & (dv > 0); Pv = to_reference(local_points(dv, Kv), Tv, T_r) / sc
        for o in range(1, 7):
            r = dd.get((cam, o))
            if r is None: continue
            f = f"{R1}/{cam}/{o}.png"
            pm = np.array(Image.open(f)) > 127 if os.path.isfile(f) else None
            truth = "ok" if r["covers"] == o else ("empty" if r["covers"] == 0 else "wrong")
            far = "far" if r["d"] >= 7 else "near"
            if pm is None or not pm.any(): cnt[(far, truth, "nomask")] += 1; continue
            Q = lift(to_work(pm, hw), Pv, vv)
            stt = status(np.median(Q, 0), len(Q), o, *P_) if len(Q) >= 20 else "abstain"
            cnt[(far, truth, stt)] += 1
    print(P_, dict(sorted(cnt.items())))
