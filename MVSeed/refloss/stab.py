"""Non-reference cameras: GT-free temporal stabilisation of the tracker's own outputs (no seed anchor).

docs/stage2-D-nonref.md section 6.  Per (scene, cam, obj) of the headline METHOD, frames t >= 1 only
(frame 0 = the seed, untouched).  Rules, all on predictions + video frames only:

  hold  (tau, d): out_t = out_{t-1} if IoU(p_t, out_{t-1}) >= tau and mean |I_t - I_{t-1}| over
                  dilate(out_{t-1}, 31) <= d; else p_t.  (chains through a static run)
  cons  (tau, d): out_t = per-pixel majority of {p_s : s >= 1, IoU(p_s, p_t) >= tau,
                  mean |I_s - I_t| over dilate(p_t, 31) <= d} (p_t itself included).
  med   (k, d):   out_t = per-pixel majority of p_{t-k..t+k} (s >= 1) when every one of those frames
                  passes the image test against t with threshold d; else p_t.

Writes per-frame J&F (eval/eval_jf.py J and boundary F, crop as anatomy2/snapfeat2) for the baseline
and every rule: [{scene, cam, obj, ref, n, JF: {rule: [jf_t, t = 0..n-1]}}].

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \\
        python MVSeed/refloss/stab.py SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2 stab_H.json
"""
import json, os, sys, math, hashlib
from multiprocessing import Pool
import cv2, numpy as np
REPO = "/host/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "eval")); import eval_jf as E
ROOT = os.path.join(REPO, "Data", "MVSeg"); METHOD = sys.argv[1]; OUT = sys.argv[2]
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")))
OBJ = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
TAUS, DS, KS = (0.8, 0.9, 0.95), (2, 3, 4, 6), (1, 2, 3)
RULES = [("hold", t, d) for t in TAUS for d in DS] + [("cons", t, d) for t in TAUS for d in DS] + \
        [("med", k, d) for k in KS for d in DS]


def iou(a, b):
    u = np.count_nonzero(a | b); return 1.0 if u == 0 else np.count_nonzero(a & b) / u


def jf(g, p, bp, se):
    box = E.crop_box(g, p, bp + 2, g.shape)
    if box is None: return 1.0
    y0, y1, x0, x1 = box
    return 0.5 * (E.db_eval_iou(g[y0:y1, x0:x1], p[y0:y1, x0:x1]) + E.db_eval_boundary(p[y0:y1, x0:x1], g[y0:y1, x0:x1], bp, se))


def diff(a, b, m):
    reg = cv2.dilate(m.astype(np.uint8), KER) > 0
    return float(np.abs(a - b)[reg].mean()) if reg.any() else 0.0


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
    n = len(files); rows = []
    for o in ids:
        P = []
        for t in range(n):
            p = os.path.join(md, pf[E.frame_key(files[t])], f"{o}.png")
            pm = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
            P.append(np.zeros((h, w), bool) if pm is None else pm > 127)
        cache = {}

        def score(t, m):
            key = (t, hashlib.sha1(np.packbits(m).tobytes()).digest())
            if key not in cache: cache[key] = jf(G[t] == o, m, bp, se)
            return cache[key]
        # image change between frames s, t inside dilate(p_t): computed lazily
        dcache = {}

        def dd(s, t):
            if (s, t) not in dcache: dcache[(s, t)] = diff(imgs[s], imgs[t], P[t])
            return dcache[(s, t)]
        out = {"base": [score(t, P[t]) for t in range(n)]}
        if not isref:
            for r in RULES:
                kind, a1, d = r; seq = [P[0]]
                for t in range(1, n):
                    if kind == "hold":
                        prev = seq[-1]
                        keep = t > 1 and iou(P[t], prev) >= a1 and diff(imgs[t], imgs[t - 1], prev) <= d
                        seq.append(prev if keep else P[t])
                    elif kind == "cons":
                        S = [s for s in range(1, n) if s == t or (iou(P[s], P[t]) >= a1 and dd(s, t) <= d)]
                        seq.append(P[t] if len(S) < 3 else np.sum([P[s] for s in S], 0) * 2 > len(S))
                    else:
                        S = [s for s in range(max(1, t - a1), min(n, t + a1 + 1))]
                        ok = len(S) >= 3 and all(dd(s, t) <= d for s in S)
                        seq.append(np.sum([P[s] for s in S], 0) * 2 > len(S) if ok else P[t])
                out["%s_%s_%s" % r] = [score(t, seq[t]) for t in range(n)]
        rows.append(dict(scene=sc, cam=cam, obj=o, ref=isref, n=n, JF=out))
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
