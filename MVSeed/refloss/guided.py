"""Probe: edge-aware refinement of a method's binary masks with a guided filter (He et al.),
grayscale guidance, threshold 0.5.  Ref camera frame 0 (the GT prompt) is left untouched."""
import json, os, sys, math
from multiprocessing import Pool
import cv2, numpy as np
REPO = "/host/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "eval"))
import eval_jf as E
ROOT = os.path.join(REPO, "Data", "MVSeg")
METHOD = sys.argv[1]; OUT = sys.argv[2]
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")))
OBJ = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
VARIANTS = [(2, 1e-3), (4, 1e-3), (8, 1e-3), (4, 1e-2), (8, 1e-2), (16, 1e-2)]

def gf(I, p, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k); mp = cv2.boxFilter(p, -1, k)
    cIp = cv2.boxFilter(I * p, -1, k) - mI * mp; vI = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cIp / (vI + eps); b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)

def jf(g, p, bp, se):
    box = E.crop_box(g, p, bp + 2, g.shape)
    if box is None: return 1.0, 1.0
    y0, y1, x0, x1 = box
    return E.db_eval_iou(g[y0:y1, x0:x1], p[y0:y1, x0:x1]), E.db_eval_boundary(p[y0:y1, x0:x1], g[y0:y1, x0:x1], bp, se)

def job(a):
    sc, cam, isref, vdir = a
    gd = os.path.join(ROOT, sc, "Mask", cam)
    files = sorted((f for f in os.listdir(gd) if f.endswith(".png")), key=E.frame_key)
    md = os.path.join(ROOT, sc, METHOD, cam); pframes = {E.frame_key(p): p for p in os.listdir(md)}
    basic = set(OBJ[sc]["seed_ids_c_ini"]); rows = []
    G = [cv2.imread(os.path.join(gd, f), cv2.IMREAD_UNCHANGED) for f in files]
    G = [g[..., 0] if g.ndim == 3 else g for g in G]
    ids = sorted(o for o in set(int(x) for g in G for x in np.unique(g)) - {0} if o in basic)
    for t, f in enumerate(files):
        g_all = G[t]
        h, w = g_all.shape; bp = int(math.ceil(E.BOUND_TH * np.linalg.norm((h, w)))); se = E._disk(bp)
        img = cv2.imread(os.path.join(vdir, "%06d.jpg" % E.frame_key(f)), cv2.IMREAD_GRAYSCALE)
        if img.shape != (h, w): img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        I = img.astype(np.float32) / 255
        for o in ids:
            g = g_all == o
            p = os.path.join(md, pframes[E.frame_key(f)], f"{o}.png")
            pm = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
            pm = np.zeros((h, w), bool) if pm is None else (cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST) if pm.shape != (h, w) else pm) > 127
            r = dict(scene=sc, cam=cam, obj=o, ref=isref, t=t)
            r["J"], r["F"] = jf(g, pm, bp, se)
            for i, (rad, eps) in enumerate(VARIANTS):
                if (isref and t == 0) or not pm.any():
                    q = pm
                else:
                    box = E.crop_box(pm, pm, 3 * rad + 4, (h, w)); y0, y1, x0, x1 = box
                    q = np.zeros_like(pm); q[y0:y1, x0:x1] = gf(I[y0:y1, x0:x1], pm[y0:y1, x0:x1].astype(np.float32), rad, eps) > 0.5
                r["Jg%d" % i], r["Fg%d" % i] = jf(g, q, bp, se)
            rows.append(r)
    return rows

if __name__ == "__main__":
    jobs = []
    for sc in OBJ:
        d = CFG.get(sc) or next((v for v in CFG.values() if v.get("folder") == sc), None)
        for c in d["cam_list"]:
            name = E.cam_name(c, d["prefix"], d["prefix1"])
            jobs.append((sc, name, name == OBJ[sc]["c_ini"], os.path.join(ROOT, sc, "Video", name)))
    with Pool(12) as pool:
        out = [r for rs in pool.imap_unordered(job, jobs) for r in rs]
    json.dump(dict(variants=VARIANTS, rows=out), open(OUT, "w")); print("rows", len(out))
