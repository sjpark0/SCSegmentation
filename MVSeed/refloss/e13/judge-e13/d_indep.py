"""Judge: independent re-composition of candidate D (mv, mv_noapp) from saved grown masks; stage breakdown of where
raw grown pixels are removed (vis / claimed by any D / not 8-connected / contested). Compares added px and dJF with D's
frame JSONs. vis = (C's props cls.png == 0)."""
import json, os, sys
from multiprocessing import Pool
import numpy as np, cv2
S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
from mvseg2d.io import load_labelmap, read_pred_mask
from mvseg2d.metrics import bound_pix, disk, jf_pair
DATA = S1 + "/Data"; DEV = S1 + "/geomem/runs/_e13_dev"; PROPS = S1 + "/geomem/runs/_e13_sam3-prompt-extend/props"
GR = S1 + "/geomem/runs/_e13_geometry-grow/main"
VARS = ("mv", "mv_noapp")

def job(arg):
    s, t, cam = arg
    fr = json.load(open(f"{GR}/frames/{s}_{t:06d}.json"))
    view = [v for v in fr["views"] if v["cam"] == cam][0]
    objs = [int(o) for o in view["objs"]]
    gt = load_labelmap(f"{DATA}/{s}/frames/{t:06d}/masks/{cam}.png"); hw = gt.shape
    cls = cv2.imread(f"{PROPS}/{s}/{t:06d}/{cam}/cls.png", cv2.IMREAD_UNCHANGED)
    invis = cls > 0
    def rd(p): return read_pred_mask(p) if os.path.isfile(p) else np.zeros(hw, bool)
    D = {o: rd(f"{DEV}/{s}/{t:06d}/{cam}/{o}.png") for o in objs}
    anyD = np.zeros(hw, np.int32)
    for o in objs: anyD += D[o]
    z = np.load(f"{GR}/masks/{s}/{t:06d}/{cam}.npz"); hw_w = tuple(int(x) for x in z["hw"])
    bp = bound_pix(*hw); se = disk(bp)
    res = {"stage": {}, "mism": 0, "maxd": 0.0, "n": 0}
    for v in VARS:
        A = {}; st = np.zeros(5, np.int64)  # raw(invis), after unclaimed, after connected, after contest, raw_total(G)
        for o in objs:
            k = f"{v}|{o}"
            if k not in z.files: continue
            g = np.unpackbits(z[k])[: hw_w[0] * hw_w[1]].reshape(hw_w).astype(bool)
            gf = cv2.resize(g.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            raw = gf & invis
            C = raw & (anyD == 0)
            if C.any() and D[o].any():
                nl, lab = cv2.connectedComponents((C | D[o]).astype(np.uint8), connectivity=8)
                keep = np.zeros(nl, bool); keep[np.unique(lab[D[o]])] = True; keep[0] = False
                A[o] = keep[lab] & C
            else:
                A[o] = np.zeros(hw, bool)
            st += [int(raw.sum()), int(C.sum()), int(A[o].sum()), 0, int(gf.sum())]
        cnt = np.zeros(hw, np.int32)
        for o in A: cnt += A[o]
        for o in objs:
            a = A[o] & (cnt == 1) if o in A else np.zeros(hw, bool)
            st[3] += int(a.sum())
            if not (gt == o).any(): continue
            rec = view["pairs"].get(str(o), {}).get(v)
            na = int(a.sum())
            if rec is None: rec = [0, 0, 0]
            if rec[2] != na: res["mism"] += 1
            if na and v == "mv":
                g = gt == o
                J0, F0 = jf_pair(g, D[o], bp, se); J1, F1 = jf_pair(g, D[o] | a, bp, se)
                res["maxd"] = max(res["maxd"], abs((J1 - J0) - rec[0]) + abs((F1 - F0) - rec[1]))
            res["n"] += 1
        res["stage"][v] = st.tolist()
    return res

if __name__ == "__main__":
    idx = json.load(open(DEV + "/index.json"))["scenes"]
    jobs = [(s, t, c) for s in sorted(idx) for t in idx[s]["frames"] for c in idx[s]["eval_cams"]]
    if len(sys.argv) > 1: jobs = [j for j in jobs if j[0] in sys.argv[1].split(",")]
    with Pool(14) as p: out = p.map(job, jobs)
    for v in VARS:
        st = np.sum([r["stage"][v] for r in out], 0)
        print(v, "G px %d | G&invis %d | &unclaimed %d (%.3f of raw) | &connected %d | &uncontested %d" % (st[4], st[0], st[1], st[1]/st[0], st[2], st[3]),
              "| removed by: claimed-by-some-D %.4f, not-connected %.4f, contested %.4f (of raw)" % ((st[0]-st[1])/st[0], (st[1]-st[2])/st[0], (st[2]-st[3])/st[0]))
    print("pair-rows", sum(r["n"] for r in out), "added-count mismatches", sum(r["mism"] for r in out), "max |dJ|+|dF| diff (mv)", max(r["maxd"] for r in out))
