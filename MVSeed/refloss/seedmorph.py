"""scsam3: dilate / erode the stage-1 seeds by 1-5 px (docs/stage1-seedfail.md section 4) -> morph.json."""
import json, os, sys, math
from multiprocessing import Pool
import cv2, numpy as np
REPO = "/host/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "eval")); import eval_jf as E
ROOT = os.path.join(REPO, "Data", "MVSeg"); S1 = os.path.join(REPO, "stage1", "Data")
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json"))); OBJ = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
def jf(g, p, bp, se):
    box = E.crop_box(g, p, bp + 2, g.shape)
    if box is None: return 1.0
    y0, y1, x0, x1 = box
    return 0.5 * (E.db_eval_iou(g[y0:y1, x0:x1], p[y0:y1, x0:x1]) + E.db_eval_boundary(p[y0:y1, x0:x1], g[y0:y1, x0:x1], bp, se))
def job(sc):
    d = CFG.get(sc) or next(v for v in CFG.values() if v.get("folder") == sc); st = d["start_frame"]; out = []
    for cam in OBJ[sc]["cameras"]:
        if cam == OBJ[sc]["c_ini"]: continue
        G = cv2.imread(os.path.join(ROOT, d["folder"], "Mask", cam, "%06d.png" % st), cv2.IMREAD_UNCHANGED); G = G[..., 0] if G.ndim == 3 else G
        h, w = G.shape; bp = int(math.ceil(E.BOUND_TH * np.linalg.norm((h, w)))); se = E._disk(bp)
        for o in OBJ[sc]["seed_ids_c_ini"]:
            g = G == o
            if not g.any(): continue
            p = os.path.join(S1, sc, "Result", "geoset_pred_ft_rep_e8", cam, f"{o}.png")
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
            a = np.zeros((h, w), np.uint8) if m is None else (m > 127).astype(np.uint8)
            r = dict(scene=sc, cam=cam, obj=o, gt_px=int(g.sum()), w=w, base=jf(g, a > 0, bp, se))
            for k in (1, 2, 3, 5):
                K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                r[f"d{k}"] = jf(g, cv2.dilate(a, K) > 0, bp, se); r[f"e{k}"] = jf(g, cv2.erode(a, K) > 0, bp, se)
            out.append(r)
    return out
with Pool(12) as p: R = [r for rs in p.imap_unordered(job, sorted(OBJ)) for r in rs]
json.dump(R, open("morph.json", "w")); print(len(R))
