"""Descriptive (T06 frame 900 = declared contaminated out-of-domain DEV; nothing here registers a constant):
Fix A statistic candidates on the D1 ROUND-1 masks (preset_r1) and the final preset masks.
A-c centroid: median lifted point -> nearest reference-object median (flag = nearest != own).
A-v vote: each lifted mask point -> nearest reference-object POINT (point-to-set), unassigned beyond r_max;
          ov3d = own share; flag = plurality of assigned points != own.
Truth per (cam, person): covers = GT person the mask overlaps most (0 = none); wrong = covers not in (0, own)."""
import json, os, sys
import numpy as np
from PIL import Image
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points, scene_scale, to_reference
S = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(REPO, "stage1/Data/T06_flag_ganggang_F/Cache/pi3")
MASK = os.path.join(REPO, "Data/PlenopticFull/T06_flag_ganggang_F/Mask")
RES = os.path.join(S, "../ring_d15/data/T06_flag_ganggang_F/Result")
ref = "cam_06"
def load(c):
    z = np.load(f"{C}/{c}.npz"); return z["depth"], z["conf"], z["K"], z["c2w"]
d_r, c_r, K_r, T_r = load(ref); h, w = d_r.shape
vr = (c_r > CONF_MIN) & (d_r > 0); Pr = to_reference(local_points(d_r, K_r), T_r, T_r); sc = scene_scale(Pr, vr)
small = lambda a: np.array(Image.fromarray(a).resize((w, h), Image.NEAREST))
labr = small(np.array(Image.open(f"{REPO}/stage1/Data/T06_flag_ganggang_F/masks/{ref}.png")))
def sub(X, n):
    return X if len(X) <= n else X[np.linspace(0, len(X) - 1, n).astype(int)]
R = {o: sub(Pr[(labr == o) & vr] / sc, 1500) for o in range(1, 7)}
cent = {o: np.median(Pr[(labr == o) & vr] / sc, 0) for o in range(1, 7)}
refpts = np.concatenate([R[o] for o in range(1, 7)]); reflab = np.concatenate([[o] * len(R[o]) for o in range(1, 7)])
rows = json.load(open(os.path.join(REPO, "stage1/scripts/ring_diag/d1_d5/scores_raw.json")))["rows"]
dmap = {r["cam"]: r["d"] for r in rows["preset"]}
ang = {}
z0 = np.load(f"{C}/{ref}.npz"); Pl = local_points(z0["depth"], z0["K"])
Xo = T_r[:3, :3] @ np.median(Pl[(labr > 0) & vr], 0) + T_r[:3, 3]
RMAX = (0.1, 0.15, 0.2, 0.3, np.inf)
out = []
for arm in ("preset_r1", "preset"):
    for i in range(32):
        c = f"cam_{i:02d}"
        if c == ref: continue
        d, cf, K, T = load(c); v = (cf > CONF_MIN) & (d > 0); P = to_reference(local_points(d, K), T, T_r) / sc
        u, q = T_r[:3, 3] - Xo, T[:3, 3] - Xo
        a = float(np.degrees(np.arccos(np.clip(u @ q / np.linalg.norm(u) / np.linalg.norm(q), -1, 1))))
        L4 = np.array(Image.open(f"{MASK}/{c}/000900.png")); L = small(L4)
        for o in [o for o in range(1, 7) if (L4 == o).any()]:
            f = f"{RES}/{arm}/{c}/{o}.png"
            m4 = np.array(Image.open(f)) > 127 if os.path.isfile(f) else np.zeros(L4.shape, bool)
            gt = L4 == o; inter = (m4 & gt).sum(); uni = (m4 | gt).sum(); J = inter / uni if uni else 1.0
            ms = small(m4.astype(np.uint8)) > 0
            cnt = [(ms & (L == j)).sum() for j in range(1, 7)]
            covers = int(np.argmax(cnt)) + 1 if sum(cnt) else 0
            sel = ms & v
            rec = dict(arm=arm, cam=c, d=dmap[c], ang=a, o=o, J=float(J), covers=covers, npx=int(sel.sum()))
            if sel.sum() >= 20:
                Q = sub(P[sel], 2000)
                mu = np.median(P[sel], 0)
                dc = {j: float(np.linalg.norm(mu - cent[j])) for j in cent}
                nearest = min(dc, key=dc.get); other = min(dc[j] for j in dc if j != o)
                rec.update(c_near=nearest, c_ratio=dc[o] / other)
                D = np.sqrt(((Q[:, None, :] - refpts[None]) ** 2).sum(-1))       # [nq, nref]
                j = D.argmin(1); dmin = D[np.arange(len(Q)), j]; lab = reflab[j]
                for r in RMAX:
                    ok = dmin <= r
                    labr_ = np.where(ok, lab, 0)
                    bc = np.bincount(labr_, minlength=7)
                    plur = int(np.argmax(bc[1:])) + 1 if bc[1:].sum() else 0
                    rec[f"ov_{r}"] = float((labr_ == o).mean()); rec[f"pl_{r}"] = plur
            out.append(rec)
json.dump(out, open(os.path.join(S, "fixa_sim.json"), "w"))
for arm in ("preset_r1", "preset"):
    rs = [r for r in out if r["arm"] == arm]
    for name, f in (("near d1-6", lambda r: r["d"] <= 6), ("far d7-16", lambda r: r["d"] >= 7)):
        g = [r for r in rs if f(r)]
        wrong = [r for r in g if r["covers"] not in (0, r["o"])]
        good = [r for r in g if r["covers"] == r["o"] and r["J"] >= 0.5]
        empty = [r for r in g if r["covers"] == 0]
        line = f"{arm:10s} {name}: n {len(g)} wrong {len(wrong)} good {len(good)} empty/none {len(empty)} abstain {sum('c_near' not in r for r in g)}"
        line += f" | A-c flag wrong {sum(r.get('c_near', r['o']) != r['o'] for r in wrong)}/{len(wrong)} falseflag good {sum(r.get('c_near', r['o']) != r['o'] for r in good)}/{len(good)}"
        line += f" A-c verified(rho<=0.7) good {sum(r.get('c_near') == r['o'] and r['c_ratio'] <= 0.7 for r in good)}/{len(good)} wrong {sum(r.get('c_near') == r['o'] and r['c_ratio'] <= 0.7 for r in wrong)}"
        print(line)
        for rr in RMAX:
            k, p = f"ov_{rr}", f"pl_{rr}"
            fl_w = sum(r.get(p, r['o']) not in (r['o'],) for r in wrong if k in r)
            fl_g = sum(r.get(p, r['o']) not in (r['o'],) for r in good if k in r)
            ver = lambda r, t: k in r and r[k] >= t
            print(f"    A-v r_max {rr}: flag wrong {fl_w}/{len(wrong)} falseflag good {fl_g}/{len(good)} | verified ov3d>=0.5 good {sum(ver(r,0.5) for r in good)}/{len(good)} wrong {sum(ver(r,0.5) for r in wrong)}/{len(wrong)}; >=0.7 good {sum(ver(r,0.7) for r in good)} wrong {sum(ver(r,0.7) for r in wrong)}")
