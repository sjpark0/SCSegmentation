#!/usr/bin/env python3
"""S1-E14 multimask report (host): the whole pre-registered grid on DEV-A, paired scene-cluster bootstrap vs A0
(mvseg2d.paired.paired_scene_bootstrap, views "<t>/<cam>", 10 000 resamples, seed 0 = dev_compare), all / good (A0 J&F
>= 0.8) / failing pairs; the O4 ceiling; token diagnostics; the pre-registered pick.
    stage1/.venv/bin/python mm_report.py [--out <dir>]  -> <dir>/report_mm.txt, summary_mm.json
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, S1)
from mvseg2d.paired import paired_scene_bootstrap  # noqa: E402

OUT = os.path.join(S1, "geomem", "runs", "_e14_multimask")
PAIRS = os.path.join(S1, "geomem", "runs", "_e13_dev", "dev_pairs.json")
SEL = ("I4", "I3", "S4", "SF", "R4", "RF")
AGG = ("M4", "M3", "W4", "X4")
ARMS = SEL + AGG
SIZE_BINS = ((0, 2_000), (2_000, 10_000), (10_000, 100_000), (100_000, 10 ** 12))


def rows_of(rows, getter):
    out = []
    for r in rows:
        J, F = getter(r)
        out.append(dict(scene=r["scene"], view=r["view"], J=J, F=F, JF=(J + F) / 2))
    return out


def boot(base, cand, n=10_000):
    return {m: paired_scene_bootstrap(base, cand, key=m, n=n, seed=0, view_key="view") for m in ("J", "F", "JF")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    parts = sorted(glob.glob(os.path.join(a.out, "parts", "*.json")))
    recs = [json.load(open(p)) for p in parts]
    rows = [r for v in recs for r in v["rows"]]
    ref = {(r["scene"], int(r["t"]), r["cam"], int(r["obj"])): r for r in json.load(open(PAIRS))["rows"]}
    keys = {(r["scene"], r["t"], r["cam"], r["obj"]) for r in rows}
    assert len(rows) == len(keys) == len(ref) and keys == set(ref), (len(rows), len(keys), len(ref))
    for r in rows:
        k = (r["scene"], r["t"], r["cam"], r["obj"])
        assert r["base_J"] == ref[k]["J"] and r["base_F"] == ref[k]["F"], k
        r["base_JF"] = (r["base_J"] + r["base_F"]) / 2
    rows.sort(key=lambda r: (r["scene"], r["t"], r["cam"], r["obj"]))
    good = [r for r in rows if r["base_JF"] >= 0.8]
    fail = [r for r in rows if r["base_JF"] < 0.8]
    L = []
    S = dict(n_views=len(recs), n_pairs=len(rows), n_good=len(good), n_fail=len(fail), arms={})

    def p(s=""):
        L.append(s)
        print(s, flush=True)

    base_get = lambda r: (r["base_J"], r["base_F"])
    p(f"DEV-A multimask grid: {len(recs)} views, {len(rows)} pairs (good {len(good)}, failing {len(fail)})")
    for name, sub in (("all", rows), ("good", good), ("fail", fail)):
        b = rows_of(sub, base_get)
        p(f"  A0 {name:5s}: J {np.mean([x['J'] for x in b]):.4f}  F {np.mean([x['F'] for x in b]):.4f}  "
          f"J&F {np.mean([x['JF'] for x in b]):.4f} (pair mean)")
    p("")
    p("arm   subset  dJ&F    [95% CI]            dJ       dF       up>0.01 down>0.01  pair-mean dJ&F")

    def arm_get(arm):
        return lambda r: tuple(r["arms"][arm])

    def oracle_get(r):
        jfs = [(J + F) / 2 for J, F in r["oracle"]]
        return tuple(r["oracle"][int(np.argmax(jfs))])

    entries = [(a_, arm_get(a_)) for a_ in ARMS] + [("O4", oracle_get)]
    for arm, get in entries:
        S["arms"][arm] = {}
        for name, sub in (("all", rows), ("good", good), ("fail", fail)):
            b, c = rows_of(sub, base_get), rows_of(sub, get)
            bs = boot(b, c)
            d = np.array([y["JF"] - x["JF"] for x, y in zip(b, c)])
            ent = dict(n=len(sub), dJF=bs["JF"]["delta"], ci=list(bs["JF"]["ci"]), dJ=bs["J"]["delta"], dF=bs["F"]["delta"],
                       ciJ=list(bs["J"]["ci"]), ciF=list(bs["F"]["ci"]), up=int((d > 0.01).sum()), down=int((d < -0.01).sum()),
                       pair_mean_dJF=float(d.mean()), JF=float(np.mean([y["JF"] for y in c])))
            if name == "all":
                ent["per_scene"] = bs["JF"]["per_scene"]
            S["arms"][arm][name] = ent
            p(f"{arm:4s}  {name:5s}  {ent['dJF']:+.4f} [{ent['ci'][0]:+.4f}, {ent['ci'][1]:+.4f}]  {ent['dJ']:+.4f}  {ent['dF']:+.4f}  "
              f"{ent['up']:6d}  {ent['down']:6d}   {ent['pair_mean_dJF']:+.4f}")
    p("")

    # good seeds by GT size
    p("good seeds by GT px: pair-mean dJ&F per arm (bins <2k, 2k-10k, 10k-100k, >=100k)")
    S["good_by_size"] = {}
    for arm, get in entries:
        vals = []
        for lo, hi in SIZE_BINS:
            sub = [r for r in good if lo <= r["gt_px"] < hi]
            vals.append((len(sub), float(np.mean([sum(get(r)) / 2 - r["base_JF"] for r in sub]))))
        S["good_by_size"][arm] = vals
        p(f"  {arm:4s} " + "  ".join(f"n={n:4d} {v:+.4f}" for n, v in vals))
    p("")

    # switching and pixel changes
    objs = [o for v in recs for o in v["objects"] if o["present"]]
    p(f"present object-views: {len(objs)} (adopted fallback tokens: {sum(o['token'] > 0 for o in objs)})")
    S["switch"] = {}
    for arm in SEL:
        n_obj = sum(o["tokens"][arm] != o["token"] for o in objs)
        n_pair = sum(r["present"] and r["tokens"][arm] != r["token"] for r in rows)
        tk = np.bincount([o["tokens"][arm] for o in objs], minlength=4).tolist()
        S["switch"][arm] = dict(obj_views_switched=n_obj, frac=n_obj / len(objs), pairs_switched=n_pair, token_counts=tk)
        p(f"  {arm}: object-views switched {n_obj} ({n_obj / len(objs):.3f}); pairs switched {n_pair}; token counts {tk}")
    tot = defaultdict(lambda: defaultdict(int))
    for v in recs:
        for arm, st in v["arm_stats"].items():
            for k, x in st.items():
                tot[arm][k] += x
    S["arm_px"] = {arm: dict(d) for arm, d in tot.items()}
    for arm in ARMS:
        d = tot[arm]
        p(f"  {arm}: objects re-formed {d['n_obj_changed']}, pair masks changed {d['n_pair_mask_changed']}, "
          f"px added {d['px_added']:,} removed {d['px_removed']:,}")
    p("")

    # token diagnostics on the pairs (present objects): per-token quality (only this object changed), ranking signals
    pres = [r for r in rows if r["present"]]
    tokjf = np.array([[(J + F) / 2 for J, F in r["oracle"]] for r in pres])        # [n,4]
    basejf = np.array([r["base_JF"] for r in pres])
    best = tokjf.max(1)
    spread = tokjf.max(1) - tokjf.min(1)
    S["tokens"] = dict(n=len(pres), mean_jf_token=[float(x) for x in tokjf.mean(0)], base=float(basejf.mean()),
                       oracle=float(best.mean()), spread_gt_001=int((spread > 0.01).sum()),
                       spread_mean=float(spread.mean()))
    p(f"pairs with a present object: {len(pres)}; J&F with token k for that object only (others A0): "
      + ", ".join(f"t{k} {tokjf[:, k].mean():.4f}" for k in range(4)) + f"; A0 {basejf.mean():.4f}; oracle {best.mean():.4f}")
    p(f"  token J&F spread (max - min) > 0.01 on {int((spread > 0.01).sum())} pairs; mean spread {spread.mean():.4f}")
    gi = np.array([r["base_JF"] >= 0.8 for r in pres])
    p(f"  good seeds: " + ", ".join(f"t{k} {tokjf[gi, k].mean():.4f}" for k in range(4)) + f"; A0 {basejf[gi].mean():.4f}; oracle {best[gi].mean():.4f}")
    oracle_tok = tokjf.argmax(1)
    p(f"  oracle-best token counts (ties -> lowest): {np.bincount(oracle_tok, minlength=4).tolist()}")
    # how often each signal picks the oracle-best token and its regret (single-object, others A0)
    S["signals"] = {}
    sig = dict(iou4=lambda r: int(np.argmax(r["ious"])), stab4=lambda r: int(np.argmax(r["stab"])),
               agree4=lambda r: int(np.argmax(r["agree"])) if r["nP"] >= 64 else r["token"], adopted=lambda r: r["token"])
    for name, f in sig.items():
        ch = np.array([f(r) for r in pres])
        jfc = tokjf[np.arange(len(pres)), ch]
        hit = float(np.mean(np.isclose(jfc, best)))
        S["signals"][name] = dict(hit_best=hit, regret=float((best - jfc).mean()), mean_jf=float(jfc.mean()))
        p(f"  signal {name:7s}: picks a best token on {hit:.3f} of pairs; mean regret {float((best - jfc).mean()):.4f}; J&F {jfc.mean():.4f}")
    # does the signal's margin predict the J&F gain over token 0?  (Spearman of signal(k) - signal(0) vs J&F(k) - J&F(0), k=1..3)
    def spear(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    dj = (tokjf[:, 1:] - tokjf[:, :1]).ravel()
    for name, key in (("iou", "ious"), ("stab", "stab"), ("agree", "agree")):
        arr = np.array([r[key] for r in pres])
        ds = (arr[:, 1:] - arr[:, :1]).ravel()
        m = np.array([r["nP"] >= 64 for r in pres]).repeat(3) if name == "agree" else np.ones_like(ds, bool)
        S["signals"][f"spearman_{name}"] = spear(ds[m], dj[m])
        p(f"  Spearman(signal_k - signal_0, J&F_k - J&F_0), k=1..3: {name} {spear(ds[m], dj[m]):+.3f} (n={int(m.sum())})")
    pq = np.array([r["P_iou_gt_vis"] for r in pres if r["nP"] >= 64])
    p(f"  reprojected reference P: |P| >= 64 on {int(sum(r['nP'] >= 64 for r in pres))}/{len(pres)} pairs; IoU(P, GT) inside vis "
      f"mean {pq.mean():.3f}, median {np.median(pq):.3f}")
    S["P_quality"] = dict(n_valid=int(len(pq)), iou_gt_vis_mean=float(pq.mean()), iou_gt_vis_median=float(np.median(pq)))
    p("")

    # pre-registered pick
    cands = [(S["arms"][arm]["all"]["dJF"], arm) for arm in ARMS
             if S["arms"][arm]["all"]["ci"][0] > 0 and S["arms"][arm]["good"]["pair_mean_dJF"] >= 0]
    pick = max(cands)[1] if cands else None
    S["pick"] = pick
    S["qualifying"] = [a_ for _, a_ in sorted(cands, reverse=True)]
    p(f"pre-registered pick (largest all-pairs dJ&F with CI lo > 0 and good-seed dJ&F >= 0): {pick or 'none -- no multimask arm advances'}")
    with open(os.path.join(a.out, "report_mm.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(a.out, "summary_mm.json"), "w") as f:
        json.dump(S, f, indent=1)


if __name__ == "__main__":
    main()
