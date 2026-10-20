"""DEV-A, predictions + inputs only (no target GT read): how often would the Fix A gate decide differently from reproj on
the round-1 memory candidates (k=3 neighbours of each eval cam)?  geoset_rep_xs_A round-1 masks + round1_conf.json.
A-v: ov3d (own share of lifted mask points by nearest reference-object point, r_max 0.2) >= 0.5; A-c: median nearest ref
centroid == own and ratio <= rho.  Abstain (< 20 lifted points, or object without a reference cloud) -> IoU rule (>= 0.7)."""
import json, os, sys
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points, scene_scale, to_reference
D = os.path.join(REPO, "stage1/Data"); DEV = os.path.join(D, "_dev/geoset_rep_xs_A")
NREF, NQ, NMIN, RMAX = 400, 3000, 20, 0.2
def sub(X, n):
    return X if len(X) <= n else X[np.linspace(0, len(X) - 1, n).astype(int)]
stats = defaultdict(Counter)
for s in sorted(os.listdir(DEV)):
    if not os.path.isdir(os.path.join(DEV, s)): continue
    meta = json.load(open(os.path.join(D, s, "meta.json"))); ref = meta["ref_cam"]
    for t in sorted(os.listdir(os.path.join(DEV, s))):
        fd = os.path.join(DEV, s, t); cfg = json.load(open(os.path.join(fd, "config.json")))
        conf = json.load(open(os.path.join(fd, "round1_conf.json")))
        C = os.path.join(D, s, "Cache/pi3_frames", t)
        z = np.load(os.path.join(C, f"{ref}.npz")); T_r = z["c2w"]; vr = (z["conf"] > CONF_MIN) & (z["depth"] > 0)
        Pr = to_reference(local_points(z["depth"], z["K"]), T_r, T_r); sc = scene_scale(Pr, vr); h, w = vr.shape
        lab = np.array(Image.open(os.path.join(D, s, "frames", t, "masks", f"{ref}.png")))
        labs = np.array(Image.fromarray(lab).resize((w, h), Image.NEAREST))
        objs = [int(o) for o in cfg["objects"]]
        R, cent = {}, {}
        for o in objs:
            sel = (labs == o) & vr
            if sel.sum() >= NMIN:
                R[o] = sub(Pr[sel] / sc, NREF); cent[o] = np.median(Pr[sel] / sc, 0)
        refpts = np.concatenate([R[o] for o in R]); reflab = np.concatenate([[o] * len(R[o]) for o in R])
        rn = (refpts ** 2).sum(1)
        cands = sorted({n for v in cfg["k3"].values() for n in v})
        for c in cands:
            zc = np.load(os.path.join(C, f"{c}.npz")); v = (zc["conf"] > CONF_MIN) & (zc["depth"] > 0)
            P = to_reference(local_points(zc["depth"], zc["K"]), zc["c2w"], T_r) / sc
            for o in objs:
                rc = conf[c][str(o)]
                present = rc["presence"] > 0
                ov = rc["overlap"]; iou_ok = rc["iou"] >= 0.7
                rep = present and (iou_ok if ov is None else ov >= 0.5)
                f = os.path.join(fd, "_r1", c, f"{o}.png")
                if not present:
                    stats[s]["absent"] += 1; continue
                m = np.array(Image.open(f)) > 127 if os.path.isfile(f) else None
                if m is None or not m.any():
                    stats[s]["present_empty"] += 1
                    av = ac = False; abst = True
                else:
                    ms = np.array(Image.fromarray(m.astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0
                    sel = ms & v
                    abst = sel.sum() < NMIN or o not in R
                    if abst:
                        av = ac = iou_ok
                    else:
                        Q = sub(P[sel], NQ)
                        d2 = (Q ** 2).sum(1)[:, None] + rn[None] - 2 * Q @ refpts.T
                        j = d2.argmin(1); dmin = np.sqrt(np.maximum(d2[np.arange(len(Q)), j], 0))
                        lab3 = np.where(dmin <= RMAX, reflab[j], 0)
                        av = (lab3 == o).mean() >= 0.5
                        mu = np.median(P[sel], 0)
                        dc = {k: np.linalg.norm(mu - cent[k]) for k in cent}
                        nearest = min(dc, key=dc.get)
                        other = min([dc[k] for k in dc if k != o], default=np.inf)
                        ac = nearest == o and dc[o] <= 1.0 * other
                        stats[s]["c_flag"] += int(nearest != o)
                        bc = np.bincount(lab3, minlength=max(objs) + 1); bc[0] = 0
                        stats[s]["v_flag"] += int(bc.sum() > 0 and int(bc.argmax()) != o)
                stats[s]["n"] += 1; stats[s]["abstain"] += int(abst)
                stats[s]["reproj"] += int(rep); stats[s]["A_v"] += int(av); stats[s]["A_c"] += int(ac)
                stats[s]["v_only"] += int(av and not rep); stats[s]["rep_only_v"] += int(rep and not av)
                stats[s]["c_only"] += int(ac and not rep); stats[s]["rep_only_c"] += int(rep and not ac)
    st = stats[s]
    print(f"{s:20s} present {st['n']:5d} abst {st['abstain']:4d} | pass reproj {st['reproj']/st['n']:.3f} A-v {st['A_v']/st['n']:.3f} A-c {st['A_c']/st['n']:.3f} | "
          f"A-v only {st['v_only']:4d} reproj-only(v) {st['rep_only_v']:4d} | A-c only {st['c_only']:4d} reproj-only(c) {st['rep_only_c']:4d} | flags v {st['v_flag']} c {st['c_flag']}", flush=True)
tot = Counter()
for st in stats.values(): tot.update(st)
print("ALL", dict(tot))
json.dump({k: dict(v) for k, v in stats.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixa_devA.json"), "w"), indent=1)
