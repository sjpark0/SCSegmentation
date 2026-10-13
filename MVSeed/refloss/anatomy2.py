"""Error anatomy of a method vs GT, per (scene, cam, obj, frame).  Runs in scsam3 container.
Outputs JF of: method, copy-GT0 baseline, band-oracle (pred with pixels within r px of GT boundary
replaced by GT) for r = bound_pix//2 and bound_pix; plus FN/FP px and their in-band shares."""
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

def jf(g, p, bp, se):
    box = E.crop_box(g, p, bp + 2, g.shape)
    if box is None: return 1.0, 1.0
    y0, y1, x0, x1 = box
    return E.db_eval_iou(g[y0:y1, x0:x1], p[y0:y1, x0:x1]), E.db_eval_boundary(p[y0:y1, x0:x1], g[y0:y1, x0:x1], bp, se)

def job(a):
    sc, cam, isref = a
    gd = os.path.join(ROOT, sc, "Mask", cam)
    files = sorted((f for f in os.listdir(gd) if f.endswith(".png")), key=E.frame_key)
    gts = [cv2.imread(os.path.join(gd, f), cv2.IMREAD_UNCHANGED) for f in files]
    gts = [g[..., 0] if g.ndim == 3 else g for g in gts]
    h, w = gts[0].shape; bp = int(math.ceil(E.BOUND_TH * np.linalg.norm((h, w)))); se = E._disk(bp)
    basic = set(OBJ[sc]["seed_ids_c_ini"])
    ids = sorted(set(int(x) for g in gts for x in np.unique(g)) - {0})
    ids = [o for o in ids if o in basic]
    md = os.path.join(ROOT, sc, METHOD, cam)
    pframes = {E.frame_key(p): p for p in os.listdir(md)}
    rows = []
    for o in ids:
        g0 = gts[0] == o
        for t, f in enumerate(files):
            g = gts[t] == o
            p = os.path.join(md, pframes.get(E.frame_key(f), "x"), f"{o}.png")
            pm = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
            pm = np.zeros((h, w), bool) if pm is None else (cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST) if pm.shape != (h, w) else pm) > 127
            r = dict(scene=sc, cam=cam, obj=o, ref=isref, t=t, gt=int(g.sum()), pred=int(pm.sum()))
            r["J"], r["F"] = jf(g, pm, bp, se)
            r["Jc"], r["Fc"] = jf(g, g0, bp, se)
            if g.any():
                # distance to the GT boundary (both sides)
                gi = g.astype(np.uint8)
                din = cv2.distanceTransform(gi, cv2.DIST_L2, 3); dout = cv2.distanceTransform(1 - gi, cv2.DIST_L2, 3)
                dist = np.where(g, din, dout)
            else:
                dist = np.full((h, w), 1e9, np.float32)
            fn = g & ~pm; fp = pm & ~g
            r["fn"] = int(fn.sum()); r["fp"] = int(fp.sum())
            for k in (1, 2, 3, 5):
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                r["Jd%d" % k], r["Fd%d" % k] = jf(g, cv2.dilate(pm.astype(np.uint8), ker) > 0, bp, se)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            r["Je2"], r["Fe2"] = jf(g, cv2.erode(pm.astype(np.uint8), ker) > 0, bp, se)
            for name, rad in (("h", bp / 2), ("b", bp), ("2", 2), ("4", 4), ("8", 8)):
                band = dist <= rad
                r["fn_" + name] = int((fn & band).sum()); r["fp_" + name] = int((fp & band).sum())
                q = np.where(band, g, pm)
                r["J" + name], r["F" + name] = jf(g, q, bp, se)
            rows.append(r)
    return rows

if __name__ == "__main__":
    jobs = []
    for sc in OBJ:
        d = CFG.get(sc) or next((v for v in CFG.values() if v.get("folder") == sc), None)
        if d is None or not os.path.isdir(os.path.join(ROOT, sc, METHOD)): continue
        ordered = sorted(d["cam_list"]); c_ini = d.get("c_ini", ordered[(len(ordered) - 1) // 2])
        for c in d["cam_list"]:
            name = E.cam_name(c, d["prefix"], d["prefix1"])
            jobs.append((sc, name, name == OBJ[sc]["c_ini"]))
    print(len(jobs), "cameras", flush=True)
    with Pool(12) as pool:
        out = [r for rs in pool.imap_unordered(job, jobs) for r in rs]
    json.dump(out, open(OUT, "w"))
    print("rows", len(out))
