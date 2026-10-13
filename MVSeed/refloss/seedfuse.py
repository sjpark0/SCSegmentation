"""scsam3: <seed result> <refined result> <out.json>: IoU gate and boundary-band fusion of a SAM 3 redraw (docs/stage1-seedfail.md section 4)."""
import json, os, sys, math
from multiprocessing import Pool
import cv2, numpy as np
REPO = "/host/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "eval")); import eval_jf as E
ROOT = os.path.join(REPO, "Data", "MVSeg"); S1 = os.path.join(REPO, "stage1", "Data")
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json"))); OBJ = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
A, B = sys.argv[1], sys.argv[2]
def jf(g, p, bp, se):
    box = E.crop_box(g, p, bp + 2, g.shape)
    if box is None: return 1.0
    y0, y1, x0, x1 = box
    return 0.5 * (E.db_eval_iou(g[y0:y1, x0:x1], p[y0:y1, x0:x1]) + E.db_eval_boundary(p[y0:y1, x0:x1], g[y0:y1, x0:x1], bp, se))
def rd(sc, name, cam, o, hw):
    p = os.path.join(S1, sc, "Result", name, cam, f"{o}.png")
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
    return np.zeros(hw, bool) if m is None else m > 127
def job(sc):
    d = CFG.get(sc) or next(v for v in CFG.values() if v.get("folder") == sc); st = d["start_frame"]; out = []
    for cam in OBJ[sc]["cameras"]:
        if cam == OBJ[sc]["c_ini"]: continue
        G = cv2.imread(os.path.join(ROOT, d["folder"], "Mask", cam, "%06d.png" % st), cv2.IMREAD_UNCHANGED); G = G[..., 0] if G.ndim == 3 else G
        h, w = G.shape; bp = int(math.ceil(E.BOUND_TH * np.linalg.norm((h, w)))); se = E._disk(bp)
        for o in OBJ[sc]["seed_ids_c_ini"]:
            g = G == o
            if not g.any(): continue
            a = rd(sc, A, cam, o, (h, w)); b = rd(sc, B, cam, o, (h, w))
            u = np.count_nonzero(a | b); iou = np.count_nonzero(a & b) / u if u else 1.0
            r = dict(scene=sc, cam=cam, obj=o, iou_ab=iou, base=jf(g, a, bp, se), ref=jf(g, b, bp, se))
            au = a.astype(np.uint8)
            for rr in (2, 4, 8, 16):
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
                band = (cv2.dilate(au, k) > 0) & ~(cv2.erode(au, k) > 0)
                r[f"band{rr}"] = jf(g, np.where(band, b, a), bp, se)
            out.append(r)
    return out
with Pool(12) as p: R = [r for rs in p.imap_unordered(job, sorted(OBJ)) for r in rs]
json.dump(R, open(sys.argv[3], "w")); print(len(R))
