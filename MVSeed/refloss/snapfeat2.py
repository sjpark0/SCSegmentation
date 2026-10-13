"""Ref camera only: per (obj, t>=1) features usable without GT at t: IoU(pred_t, GT0),
image change inside dilated GT0 mask (mean |I_t - I_0|, 0..255), plus IoU(pred_t, pred_{t-1})."""
import json, os, sys, math
from multiprocessing import Pool
import cv2, numpy as np
REPO = "/host/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "eval")); import eval_jf as E
ROOT = os.path.join(REPO, "Data", "MVSeg"); METHOD = sys.argv[1]; OUT = sys.argv[2]
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")))
OBJ = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
def iou(a, b):
    u = np.count_nonzero(a | b); return 1.0 if u == 0 else np.count_nonzero(a & b) / u
def jf(g, p, bp, se):
    box = E.crop_box(g, p, bp + 2, g.shape)
    if box is None: return 1.0, 1.0
    y0, y1, x0, x1 = box
    return E.db_eval_iou(g[y0:y1, x0:x1], p[y0:y1, x0:x1]), E.db_eval_boundary(p[y0:y1, x0:x1], g[y0:y1, x0:x1], bp, se)
def job(a):
    sc, cam, vdir, isref = a
    gd = os.path.join(ROOT, sc, "Mask", cam)
    files = sorted((f for f in os.listdir(gd) if f.endswith(".png")), key=E.frame_key)
    G = [cv2.imread(os.path.join(gd, f), cv2.IMREAD_UNCHANGED) for f in files]; G = [g[..., 0] if g.ndim == 3 else g for g in G]
    h, w = G[0].shape; bp = int(math.ceil(E.BOUND_TH * np.linalg.norm((h, w)))); se = E._disk(bp)
    imgs = []
    for f in files:
        im = cv2.imread(os.path.join(vdir, "%06d.jpg" % E.frame_key(f)), cv2.IMREAD_GRAYSCALE)
        imgs.append((cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) if im.shape != (h, w) else im).astype(np.int16))
    md = os.path.join(ROOT, sc, METHOD, cam); pf = {E.frame_key(p): p for p in os.listdir(md)}
    basic = set(OBJ[sc]["seed_ids_c_ini"])
    ids = sorted(o for o in set(int(x) for g in G for x in np.unique(g)) - {0} if o in basic)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)); rows = []
    for o in ids:
        p0f = os.path.join(md, pf[E.frame_key(files[0])], f"{o}.png")
        g0 = (cv2.imread(p0f, cv2.IMREAD_GRAYSCALE) > 127) if os.path.exists(p0f) else np.zeros((h, w), bool)
        reg = cv2.dilate(g0.astype(np.uint8), ker) > 0; prev = g0
        for t in range(1, len(files)):
            p = os.path.join(md, pf[E.frame_key(files[t])], f"{o}.png")
            pm = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
            pm = np.zeros((h, w), bool) if pm is None else pm > 127
            d = float(np.abs(imgs[t] - imgs[0])[reg].mean()) if reg.any() else 0.0
            Js, Fs = jf(G[t] == o, g0, bp, se)
            rows.append(dict(scene=sc, cam=cam, obj=o, t=t, ref=isref, Js=Js, Fs=Fs, iou_p_g0=iou(pm, g0), iou_p_prev=iou(pm, prev), imgdiff=d, gt0px=int(g0.sum())))
            prev = pm
    return rows
if __name__ == "__main__":
    jobs = []
    for sc in OBJ:
        d = CFG.get(sc) or next((v for v in CFG.values() if v.get("folder") == sc), None)
        for c in d["cam_list"]:
            name = E.cam_name(c, d["prefix"], d["prefix1"])
            jobs.append((sc, name, os.path.join(ROOT, sc, "Video", name), name == OBJ[sc]["c_ini"]))
    with Pool(12) as pool: out = [r for rs in pool.imap_unordered(job, jobs) for r in rs]
    json.dump(out, open(OUT, "w")); print("rows", len(out))
