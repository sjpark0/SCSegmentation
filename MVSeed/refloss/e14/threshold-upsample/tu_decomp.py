#!/usr/bin/env python3
"""Where do the pixels a global threshold shift adds go?  (bil2 = adopted upsampler; DEV-A; host)

Per eval view: keep masks at tau = 0 and at each tau in TAUS_D (tu_variants pipeline).  For every decoded object and
every pixel it gains (keep_tau & ~keep_0) or loses (keep_0 & ~keep_tau): was the pixel claimed by some other mask at
tau = 0 (claimed) or by none (unclaimed), and who owns it in the GT (own / another decoded object / a labelled object
that was not decoded / label 0).  Pooled over all decoded objects of all views, and over the scored good-seed pairs.

    stage1/.venv/bin/python tu_decomp.py --out decomp.json [--procs 14]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tu_variants as T  # noqa: E402

TAUS_D = (0.5, 1.0)


def full_keep(km, B, hw):
    out = np.zeros((B, *hw), bool)
    for b, (y0, y1, x0, x1, k) in km.items():
        out[b, y0:y1, x0:x1] = k
    return out


def view_job(args):
    scene, t, cam, good_objs = args
    d = T.E.load(T.DUMP, scene, t, cam)
    objs = [int(o) for o in d["objects"]]
    hw = (int(d["hw"][0]), int(d["hw"][1]))
    lr = d["low_res"].astype(np.float32)
    present = d["presence"] > 0
    gt = T.load_labelmap(os.path.join(T.DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    crops = T.up_bil2(lr, present, hw)
    arg = T.argmax_map(crops, hw)
    B = len(objs)
    K0 = full_keep(T.keep_masks(crops, arg, 0.0), B, hw)
    claim0 = K0.sum(0)
    seeded = np.isin(gt, objs)
    out = {}
    for tau in TAUS_D:
        K = full_keep(T.keep_masks(crops, arg, tau), B, hw)
        rec = {}
        for scope, sel in (("all", range(B)), ("good", [objs.index(o) for o in good_objs])):
            c = dict(add=0, add_unclaimed=0, add_claimed=0, rem=0)
            for gname in ("own", "seeded_other", "unseeded", "bg"):
                c[f"add_unclaimed_{gname}"] = 0
                c[f"add_claimed_{gname}"] = 0
                c[f"rem_{gname}"] = 0
            for b in sel:
                o = objs[b]
                add = K[b] & ~K0[b]
                rem = K0[b] & ~K[b]
                if not (add.any() or rem.any()):
                    continue
                own = gt == o
                so = seeded & ~own
                un = (gt != 0) & ~seeded
                bg = gt == 0
                unc = claim0 == 0
                c["add"] += int(add.sum())
                c["rem"] += int(rem.sum())
                for gname, g in (("own", own), ("seeded_other", so), ("unseeded", un), ("bg", bg)):
                    c[f"add_unclaimed_{gname}"] += int((add & unc & g).sum())
                    c[f"add_claimed_{gname}"] += int((add & ~unc & g).sum())
                    c[f"rem_{gname}"] += int((rem & g).sum())
                c["add_unclaimed"] += int((add & unc).sum())
                c["add_claimed"] += int((add & ~unc).sum())
            rec[scope] = c
        # view partition: pixels unclaimed at tau 0 that become claimed, and whether their single owner is right
        claim = K.sum(0)
        newly = (claim0 == 0) & (claim > 0)
        owner = np.where(K.any(0), np.array(objs)[K.argmax(0)], 0)
        rec["view"] = dict(unclaimed0=int((claim0 == 0).sum()), unclaimed0_gt_seeded=int(((claim0 == 0) & seeded).sum()),
                           newly_claimed=int(newly.sum()), newly_correct=int((newly & (owner == gt)).sum()),
                           newly_gt_seeded=int((newly & seeded).sum()), newly_multi=int((newly & (claim > 1)).sum()),
                           unclaimed_tau=int((claim == 0).sum()))
        out[f"{tau:g}"] = rec
    return dict(scene=scene, t=t, cam=cam, res=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    rows = json.load(open(T.PAIRS))["rows"]
    views = {}
    for r in rows:
        views.setdefault((r["scene"], int(r["t"]), r["cam"]), [])
        if r["JF"] >= 0.8:
            views[(r["scene"], int(r["t"]), r["cam"])].append(int(r["obj"]))
    jobs = [(s, t, c, g) for (s, t, c), g in sorted(views.items())]
    with Pool(a.procs) as pool:
        res = list(pool.imap_unordered(view_job, jobs, chunksize=1))
    tot = {}
    for r in res:
        for tau, rec in r["res"].items():
            for scope, c in rec.items():
                acc = tot.setdefault(tau, {}).setdefault(scope, {})
                for k, v in c.items():
                    acc[k] = acc.get(k, 0) + v
    json.dump(dict(total=tot, views=res, seconds=time.time() - t0), open(a.out, "w"), indent=1)
    for tau, rec in tot.items():
        print(f"tau {tau}")
        for scope, c in rec.items():
            if scope == "view":
                print(f"  view: unclaimed@0 {c['unclaimed0']} (GT decoded {c['unclaimed0_gt_seeded'] / c['unclaimed0']:.3f}); newly claimed "
                      f"{c['newly_claimed']} = {c['newly_claimed'] / c['unclaimed0']:.3f} of them; owner correct {c['newly_correct'] / c['newly_claimed']:.3f}; "
                      f"GT decoded {c['newly_gt_seeded'] / c['newly_claimed']:.3f}; still unclaimed {c['unclaimed_tau']}")
                continue
            A = max(c["add"], 1)
            print(f"  {scope}: added {c['add']} px (unclaimed@0 {c['add_unclaimed'] / A:.3f}, claimed@0 {c['add_claimed'] / A:.3f}); removed {c['rem']}")
            for g in ("own", "seeded_other", "unseeded", "bg"):
                print(f"     GT {g:13s}: added&unclaimed {c[f'add_unclaimed_{g}'] / A:.3f}  added&claimed {c[f'add_claimed_{g}'] / A:.3f}  "
                      f"removed {c[f'rem_{g}'] / max(c['rem'], 1):.3f}")


if __name__ == "__main__":
    main()
