"""scsam3: 288/576-grid round-trip ceiling of the frame-0 non-reference GT (docs/stage1-seedfail.md section 3) -> grid0.json."""
import json, os, sys, math
from multiprocessing import Pool
import cv2, numpy as np
REPO = "/host/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "eval")); import eval_jf as E
import torch, torch.nn.functional as F
ROOT = os.path.join(REPO, "Data", "MVSeg")
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json"))); OBJ = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
def jf(g, p, bp, se):
    box = E.crop_box(g, p, bp + 2, g.shape)
    if box is None: return 1.0
    y0, y1, x0, x1 = box
    return 0.5 * (E.db_eval_iou(g[y0:y1, x0:x1], p[y0:y1, x0:x1]) + E.db_eval_boundary(p[y0:y1, x0:x1], g[y0:y1, x0:x1], bp, se))
def rt(g, n):
    t = torch.from_numpy(g.astype(np.float32))[None, None] * 2 - 1
    lo = F.interpolate(F.interpolate(t, size=(1008, 1008), mode="bilinear", align_corners=False), size=(n, n), mode="bilinear", align_corners=False, antialias=True)
    hi = F.interpolate(F.interpolate(lo, size=(1008, 1008), mode="bilinear", align_corners=False), size=g.shape, mode="bilinear", align_corners=False)
    return hi[0, 0].numpy() > 0
def job(sc):
    torch.set_num_threads(2)
    d = CFG.get(sc) or next(v for v in CFG.values() if v.get("folder") == sc); st = d["start_frame"]; out = []
    for cam in OBJ[sc]["cameras"]:
        if cam == OBJ[sc]["c_ini"]: continue
        G = cv2.imread(os.path.join(ROOT, d["folder"], "Mask", cam, "%06d.png" % st), cv2.IMREAD_UNCHANGED); G = G[..., 0] if G.ndim == 3 else G
        h, w = G.shape; bp = int(math.ceil(E.BOUND_TH * np.linalg.norm((h, w)))); se = E._disk(bp)
        for o in OBJ[sc]["seed_ids_c_ini"]:
            g = G == o
            if not g.any(): continue
            out.append(dict(scene=sc, cam=cam, obj=o, g288=jf(g, rt(g, 288), bp, se), g576=jf(g, rt(g, 576), bp, se)))
    return out
with Pool(10) as p: R = [r for rs in p.imap_unordered(job, sorted(OBJ)) for r in rs]
json.dump(R, open("grid0.json", "w")); print(len(R))
