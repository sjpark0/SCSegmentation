"""Descriptive (T06 frame 900; NOT a tuned rule): J of per-object hybrids between the learned stage 1 (lat) and the
chain (sub) frame-900 masks, with routing signals that read no target-view ground truth."""
import json, os, sys
import numpy as np
from PIL import Image
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points, scene_scale, to_reference
D = os.path.join(REPO, "stage1/Data/T06_flag_ganggang_F"); C = os.path.join(D, "Cache/pi3")
RES = os.path.join(REPO, "PlenopticServer1/CROOM/SegResult/T06_flag_ganggang_F/SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs")
ref = "cam_06"
def load(c):
    z = np.load(f"{C}/{c}.npz"); return z["depth"], z["conf"], z["K"], z["c2w"]
d_r, c_r, K_r, T_r = load(ref); h, w = d_r.shape
vr = (c_r > CONF_MIN) & (d_r > 0); Pr = to_reference(local_points(d_r, K_r), T_r, T_r); sc = scene_scale(Pr, vr)
small = lambda a: np.array(Image.fromarray(a).resize((w, h), Image.NEAREST))
labr = small(np.array(Image.open(f"{D}/masks/{ref}.png")))
cent = {o: np.median(Pr[(labr == o) & vr] / sc, 0) for o in range(1, 7)}
lat = {r['camera']: dict(zip(r['objects'], r['result']['J_all'])) for r in json.load(open(f"{REPO}/plenoptic/full/jf_lat.json"))}
sub = {r['camera']: dict(zip(r['objects'], r['result']['J_all'])) for r in json.load(open(f"{REPO}/plenoptic/full/jf_sub.json"))}
oc = {(r['cam'], r['o']): r['objcovis'] for r in json.load(open("objcovis.json"))}
ang = {}
X = None
rows = []
for i in range(32):
    c = f"cam_{i:02d}"
    if c == ref: continue
    d, cf, K, T = load(c); v = (cf > CONF_MIN) & (d > 0); P = to_reference(local_points(d, K), T, T_r) / sc
    for o, jl in lat[c].items():
        if (c, o) not in oc: continue
        f = f"{RES}/{c}/900/{o}.png"
        flag = True
        if os.path.isfile(f):
            ms = small((np.array(Image.open(f)) > 127).astype(np.uint8)) > 0
            sel = ms & v
            if sel.sum() >= 20:
                q = np.median(P[sel], 0)
                dist = {k: np.linalg.norm(q - cent[k]) for k in cent}
                flag = min(dist, key=dist.get) != o
        rows.append(dict(cam=c, o=o, lat=jl, sub=sub[c][o], flag=flag, oc=oc[(c, o)]))
L = np.array([r['lat'] for r in rows]); S = np.array([r['sub'] for r in rows]); F = np.array([r['flag'] for r in rows]); O = np.array([r['oc'] for r in rows])
print(f"pairs {len(rows)}  lat {L.mean():.4f}  sub {S.mean():.4f}  oracle max {np.maximum(L,S).mean():.4f}")
print(f"3D-identity flag -> sub: {np.where(F, S, L).mean():.4f}  (flagged {F.sum()})")
for t in (0.25, 0.5):
    print(f"obj co-visibility < {t} -> sub: {np.where(O < t, S, L).mean():.4f} (routed {np.sum(O<t)}); combined with flag: {np.where((O<t)|F, S, L).mean():.4f}")
