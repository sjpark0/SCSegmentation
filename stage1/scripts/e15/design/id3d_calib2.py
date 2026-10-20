"""Read-only S1-E15 design probe (DEV-A + T06 dev only): 3D identity statistics for GT masks (correct), synthetic swaps
(GT mask of the 3D-nearest other object) and the preset's predicted masks.  Writes rows to a JSON for rule evaluation."""
import json, os, sys
import numpy as np, cv2
from PIL import Image
REPO = "/home/sjpark/Documents/SCSegmentation"
S1 = os.path.join(REPO, "stage1"); sys.path.insert(0, S1)
from geomem.pointmap import CONF_MIN, local_points, scene_scale, to_reference
from geomem.devset import dev_frames, basic_set, eval_cams
D = os.path.join(S1, "Data"); DEVO = os.path.join(D, "_dev", "geoset_rep_xs_A")
K3 = np.ones((3, 3), np.uint8); NMIN = 20

def load(C, c):
    z = np.load(f"{C}/{c}.npz"); return z["depth"], z["conf"], z["K"], z["c2w"]

def to_work(m, hw):
    h, w = hw
    return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5

def lift_set(m_work, P, valid):
    me = cv2.erode(m_work.astype(np.uint8), K3) > 0
    sel = me & valid
    if sel.sum() < NMIN:
        sel = m_work & valid
    return P[sel]

def model(C, ref, lab_full, objs):
    d, cf, K, T = load(C, ref); hw = d.shape
    v = (cf > CONF_MIN) & (d > 0); P = to_reference(local_points(d, K), T, T); sc = scene_scale(P, v); P = P / sc
    M = {}
    for o in objs:
        R = lift_set(to_work(lab_full == o, hw), P, v)
        if len(R) < NMIN: continue
        c = np.median(R, 0)
        m = lab_full == o; H, W = m.shape; b = max(2, int(0.005 * min(H, W)))
        edge = np.zeros_like(m); edge[:b] = edge[-b:] = True; edge[:, :b] = True; edge[:, -b:] = True
        M[o] = dict(c=c, rho=float(np.percentile(np.linalg.norm(R - c, axis=1), 90)), n=len(R),
                    border=float((m & edge).sum() / max(1, m.sum())) * 100.0)
    return M, sc, T, hw

def target(C, cam, T_ref, sc):
    d, cf, K, T = load(C, cam); v = (cf > CONF_MIN) & (d > 0)
    return to_reference(local_points(d, K), T, T_ref) / sc, v, d.shape

def row(M, o, Q, kind, **kw):
    if len(Q) < NMIN or o not in M: return dict(kind=kind, o=o, n=len(Q), ok=False, **kw)
    q = np.median(Q, 0)
    return dict(kind=kind, o=o, n=len(Q), ok=True, d={str(j): float(np.linalg.norm(q - M[j]["c"])) for j in M},
                rho={str(j): M[j]["rho"] for j in M}, rho_q=float(np.percentile(np.linalg.norm(Q - q, axis=1), 90)),
                border={str(j): M[j]["border"] for j in M}, nref=M[o]["n"], **kw)

def read_pred(path, hw_full):
    return np.array(Image.open(path)) > 127 if os.path.isfile(path) else np.zeros(hw_full, bool)

rows = []
folds = json.load(open(os.path.join(S1, "geomem/folds.json")))
for s in sorted(folds):
    meta = json.load(open(f"{D}/{s}/meta.json")); ref = meta["ref_cam"]; b = basic_set(s); ev = eval_cams(D, s)
    for t in dev_frames(D, s, "A"):
        C = f"{D}/{s}/Cache/pi3_frames/{t:06d}"
        labr = np.array(Image.open(f"{D}/{s}/frames/{t:06d}/masks/{ref}.png")); labr = np.where(np.isin(labr, b), labr, 0)
        objs = [int(o) for o in np.unique(labr) if o]
        M, sc, T_ref, hw = model(C, ref, labr, objs)
        for cam in ev:
            P, v, _ = target(C, cam, T_ref, sc)
            lab = np.array(Image.open(f"{D}/{s}/frames/{t:06d}/masks/{cam}.png"))
            labw = cv2.resize(lab, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
            for o in objs:
                g = lab == o
                if not g.any(): continue
                rows.append(row(M, o, lift_set(to_work(g, hw), P, v), "gt", scene=s, t=t, cam=cam, src="muvod"))
                # synthetic swap: GT of the 3D-nearest other modelled object present in this view, labelled o
                if o in M:
                    cands = sorted((np.linalg.norm(M[o]["c"] - M[j]["c"]), j) for j in M if j != o and (lab == j).any())
                    if cands:
                        j = cands[0][1]
                        rows.append(row(M, o, lift_set(to_work(lab == j, hw), P, v), "swap", scene=s, t=t, cam=cam, src="muvod", true=j))
                for stage, sub in (("r1", "_r1"), ("final", "")):
                    pth = os.path.join(DEVO, s, f"{t:06d}", sub, cam, f"{o}.png") if sub else os.path.join(DEVO, s, f"{t:06d}", cam, f"{o}.png")
                    pm = read_pred(pth, g.shape)
                    if not pm.any():
                        rows.append(dict(kind="pred_" + stage, o=o, n=0, ok=False, covers=0, scene=s, t=t, cam=cam, src="muvod")); continue
                    pw = to_work(pm, hw)
                    cnt = {int(j): int((pw & (labw == j)).sum()) for j in np.unique(labw[pw]) if j}
                    covers = max(cnt, key=cnt.get) if cnt else 0
                    rows.append(row(M, o, lift_set(pw, P, v), "pred_" + stage, scene=s, t=t, cam=cam, src="muvod", covers=covers))
    print(s, len(rows), flush=True)
# T06 frame 900 (contaminated dev)
s = "T06_flag_ganggang_F"; C = f"{D}/{s}/Cache/pi3"; ref = "cam_06"
labr = np.array(Image.open(f"{D}/{s}/masks/{ref}.png")); objs = list(range(1, 7))
M, sc, T_ref, hw = model(C, ref, labr, objs)
NAS = os.path.join(REPO, "PlenopticServer1/CROOM/SegResult", s)
for i in range(32):
    cam = f"cam_{i:02d}"
    if cam == ref: continue
    P, v, _ = target(C, cam, T_ref, sc)
    lab = np.array(Image.open(f"{REPO}/Data/PlenopticFull/{s}/Mask/{cam}/000900.png"))
    labw = cv2.resize(lab, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    for o in objs:
        g = lab == o
        if not g.any(): continue
        rows.append(row(M, o, lift_set(to_work(g, hw), P, v), "gt", scene=s, t=900, cam=cam, src="t06"))
        cands = sorted((np.linalg.norm(M[o]["c"] - M[j]["c"]), j) for j in M if j != o and (lab == j).any())
        if cands:
            j = cands[0][1]
            rows.append(row(M, o, lift_set(to_work(lab == j, hw), P, v), "swap", scene=s, t=900, cam=cam, src="t06", true=j))
        for stage, cfg in (("final", "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs"),):
            pm = read_pred(f"{NAS}/{cfg}/{cam}/900/{o}.png", g.shape)
            if not pm.any():
                rows.append(dict(kind="pred_" + stage, o=o, n=0, ok=False, covers=0, scene=s, t=900, cam=cam, src="t06")); continue
            pw = to_work(pm, hw)
            cnt = {int(j): int((pw & (labw == j)).sum()) for j in np.unique(labw[pw]) if j}
            covers = max(cnt, key=cnt.get) if cnt else 0
            rows.append(row(M, o, lift_set(pw, P, v), "pred_" + stage, scene=s, t=900, cam=cam, src="t06", covers=covers))
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "id3d_rows2.json"), "w"))
print("rows", len(rows))
