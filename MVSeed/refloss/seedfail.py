"""Stage-1 seed error anatomy on the non-reference annotated cameras (docs/stage1-seedfail.md).

Per (scene, cam, obj) with the object in the basic set: the headline stage-1 seed
(Data/MVSeg/<folder>/MVSeed_geoset_rep/<cam>/<start>/<obj>.png) against the frame-0 ground truth, with
features for sorting failures:

  J, F, JF, gt_px, seed_px, recall, precision
  frac_bg       seed pixels on background (label 0) / seed pixels
  frac_other    seed pixels on the most-overlapped OTHER object / seed pixels  (other_obj, other_basic)
  border        the GT touches the image border (truncated by the frame)
  vis_ratio     gt_px(cam) / gt_px(c_ini)  (frame-relative area; < 1 = smaller or partly hidden here)
  gt_cc, seed_cc   connected components (>= 0.5 % of the mask) of GT / seed
  angle         angle between the optical axes of cam and c_ini (deg, Pi3 c2w)
  oracle_r{2,4,8}  JF after replacing pixels within r px of the GT boundary by the GT (boundary-only error)
  twin          another basic object with the same text label exists (objects_labels.json); otype dynamic/static/env

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \\
        python MVSeed/refloss/seedfail.py MVSeed_geoset_rep seedfail.json
    ... seedfail.py s1:<result> out.json     # a stage-1 result folder, stage1/Data/<scene>/Result/<result>/<cam>/<obj>.png
"""
import json, os, sys, math
from multiprocessing import Pool
import cv2, numpy as np
REPO = "/host/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "eval")); import eval_jf as E
ROOT = os.path.join(REPO, "Data", "MVSeg"); S1 = os.path.join(REPO, "stage1", "Data")
SEEDS = sys.argv[1]; OUT = sys.argv[2]
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")))
OBJ = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))


def jf(g, p, bp, se):
    box = E.crop_box(g, p, bp + 2, g.shape)
    if box is None: return 1.0, 1.0
    y0, y1, x0, x1 = box
    return E.db_eval_iou(g[y0:y1, x0:x1], p[y0:y1, x0:x1]), E.db_eval_boundary(p[y0:y1, x0:x1], g[y0:y1, x0:x1], bp, se)


def ncc(m):
    if not m.any(): return 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return int((st[1:, cv2.CC_STAT_AREA] >= 0.005 * m.sum()).sum())


def labels(sc, folder):
    p = os.path.join(ROOT, folder, "Mask", "objects_labels.json")
    if not os.path.exists(p): return {}
    d = json.load(open(p))
    return {int(k): (v.get("label"), v.get("type")) for k, v in d.items() if str(k).isdigit()}


def axis(sc, cam):
    p = os.path.join(S1, sc, "Cache", "pi3", cam + ".npz")
    if not os.path.exists(p): return None
    c2w = np.load(p)["c2w"]; return c2w[:3, 2] / np.linalg.norm(c2w[:3, 2])


def job(a):
    sc = a; d = CFG.get(sc) or next(v for v in CFG.values() if v.get("folder") == sc); folder = d["folder"]
    st = d["start_frame"]; cini = OBJ[sc]["c_ini"]; basic = OBJ[sc]["seed_ids_c_ini"]; lab = labels(sc, folder)
    read = lambda cam: (lambda g: g[..., 0] if g.ndim == 3 else g)(cv2.imread(os.path.join(ROOT, folder, "Mask", cam, "%06d.png" % st), cv2.IMREAD_UNCHANGED))
    G0 = read(cini); ax0 = axis(sc, cini); rows = []
    for cam in OBJ[sc]["cameras"]:
        if cam == cini: continue
        G = read(cam); h, w = G.shape; bp = int(math.ceil(E.BOUND_TH * np.linalg.norm((h, w)))); se = E._disk(bp)
        ax = axis(sc, cam); ang = None if ax is None or ax0 is None else float(np.degrees(np.arccos(np.clip(ax @ ax0, -1, 1))))
        for o in basic:
            g = G == o
            sp = (os.path.join(S1, sc, "Result", SEEDS[3:], cam, f"{o}.png") if SEEDS.startswith("s1:")
                  else os.path.join(ROOT, folder, SEEDS, cam, str(st), f"{o}.png"))
            s = cv2.imread(sp, cv2.IMREAD_GRAYSCALE) if os.path.exists(sp) else None
            s = np.zeros((h, w), bool) if s is None else (cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST) if s.shape != (h, w) else s) > 127
            J, F = jf(g, s, bp, se); gp, spx = int(g.sum()), int(s.sum())
            inter = int((g & s).sum())
            other = {int(x): int(c) for x, c in zip(*np.unique(G[s & ~g], return_counts=True)) if x != 0}
            oo = max(other, key=other.get) if other else None
            band = {}
            for r in (2, 4, 8):
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)); gu = g.astype(np.uint8)
                bd = (cv2.dilate(gu, k) > 0) & ~(cv2.erode(gu, k) > 0)
                so = np.where(bd, g, s); band[f"oracle_r{r}"] = 0.5 * sum(jf(g, so, bp, se))
            lb, ty = lab.get(o, (None, None))
            twin = lb is not None and any(lab.get(q, (None,))[0] == lb for q in basic if q != o)
            rows.append(dict(scene=sc, cam=cam, obj=o, J=J, F=F, JF=0.5 * (J + F), gt_px=gp, seed_px=spx,
                             recall=inter / gp if gp else None, precision=inter / spx if spx else None,
                             frac_bg=float((s & (G == 0)).sum()) / spx if spx else None,
                             frac_other=(other[oo] / spx) if oo else 0.0, other_obj=oo, other_basic=oo in basic if oo else None,
                             border=bool(g[0].any() or g[-1].any() or g[:, 0].any() or g[:, -1].any()),
                             vis_ratio=gp / max(1, int((G0 == o).sum())), gt_cc=ncc(g), seed_cc=ncc(s), angle=ang,
                             twin=twin, label=lb, otype=ty, shape=[h, w], **band))
    return rows


if __name__ == "__main__":
    with Pool(12) as pool: out = [r for rs in pool.imap_unordered(job, sorted(OBJ)) for r in rs]
    json.dump(out, open(OUT, "w"), indent=0); print("rows", len(out))
