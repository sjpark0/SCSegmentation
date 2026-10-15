#!/usr/bin/env python3
"""S1-E14 zoom-xcl: host comparison of the DEV-A arms (dev_score outputs) -- numpy only.

Arms: x = the adopted geoset_rep_x (_e13_dev/dev_pairs.json), r2 = geoset_rep (no zoom, no xcl), z = geoset_rep_z2r
(zoom), zx = zoom -> xcl (primary), zxb = diagnostic.  Every comparison uses mvseg2d.dev_compare.compare (paired
scene-cluster bootstrap, views "<t>/<cam>", 10 000, seed 0) on all pairs, and the same bootstrap restricted to the good
(adopted x J&F >= 0.8) and failing (< 0.8) pairs; plus pair means, size bins, zoom verdict bins and the 2x2 interaction
(zx - z) - (x - r2) with the same scene bootstrap.
    PYTHONPATH=stage1 stage1/.venv/bin/python compare_arms.py --dir stage1/geomem/runs/_e14_zoom-xcl --out <json>
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from mvseg2d.dev_compare import compare, load
from mvseg2d.paired import paired_scene_bootstrap, scene_means

KEY = lambda r: (r["scene"], r["t"], r["cam"], r["obj"])  # noqa: E731
SIZE_BINS = ((0, 2_000, "<2k"), (2_000, 10_000, "2k-10k"), (10_000, 100_000, "10k-100k"), (100_000, 10 ** 12, ">=100k"))


def boot(rows_a, rows_b, key="JF"):
    v = paired_scene_bootstrap(rows_a, rows_b, key=key, n=10_000, seed=0, view_key="view")
    return dict(delta=v["delta"], ci=list(v["ci"]), n_scenes=v["n_scenes"])


def subset(rows, keys):
    return [r for r in rows if KEY(r) in keys]


def pair_mean(rows, f):
    return float(np.mean([f(r) for r in rows])) if rows else float("nan")


def comp(A, B, good, fail, zoomv):
    """A = base rows, B = candidate rows (same pair set)."""
    c = compare(A, B)
    kb = {KEY(r): r for r in A}
    kc = {KEY(r): r for r in B}
    out = dict(n_pairs=c["n_pairs"], all={m: dict(delta=c["paired"][m]["delta"], ci=c["paired"][m]["ci"]) for m in ("JF", "J", "F")},
               per_scene_JF=c["paired"]["JF"]["per_scene"])
    for name, ks in (("good", good), ("fail", fail)):
        a, b = subset(A, ks), subset(B, ks)
        d = [kc[k]["JF"] - kb[k]["JF"] for k in ks]
        out[name] = dict(n=len(ks), JF=boot(a, b, "JF"), J=boot(a, b, "J"), F=boot(a, b, "F"),
                         pair_mean_dJF=float(np.mean(d)), up=int(sum(x > 0.01 for x in d)), down=int(sum(x < -0.01 for x in d)),
                         changed=int(sum(x != 0 for x in d)),
                         base_JF=pair_mean(a, lambda r: r["JF"]), cand_JF=pair_mean(b, lambda r: r["JF"]),
                         base_1mrec=pair_mean(a, lambda r: 1 - r["recall"]), cand_1mrec=pair_mean(b, lambda r: 1 - r["recall"]),
                         base_1mprec=pair_mean([r for r in a if r["precision"] is not None], lambda r: 1 - r["precision"]),
                         cand_1mprec=pair_mean([r for r in b if r["precision"] is not None], lambda r: 1 - r["precision"]))
    d_all = {k: kc[k]["JF"] - kb[k]["JF"] for k in kb}
    out["pair_mean_dJF_all"] = float(np.mean(list(d_all.values())))
    out["changed_pairs"] = int(sum(v != 0 for v in d_all.values()))
    out["base_JF_hier"] = float(np.mean(list(scene_means(A, "JF", "view").values())))
    out["cand_JF_hier"] = float(np.mean(list(scene_means(B, "JF", "view").values())))
    out["base_JF_pairmean"] = pair_mean(A, lambda r: r["JF"])
    out["cand_JF_pairmean"] = pair_mean(B, lambda r: r["JF"])
    sb = {}
    for lo, hi, nm in SIZE_BINS:
        ks = [k for k in kb if lo <= kb[k]["gt_px"] < hi]
        kg = [k for k in ks if k in good]
        sb[nm] = dict(n=len(ks), dJF=float(np.mean([d_all[k] for k in ks])) if ks else None,
                      n_good=len(kg), dJF_good=float(np.mean([d_all[k] for k in kg])) if kg else None)
    out["size_bins"] = sb
    vb = defaultdict(list)
    for k in kb:
        vb[zoomv.get(k, "no-candidate")].append(d_all[k])
    out["zoom_verdict_bins"] = {v: dict(n=len(x), dJF=float(np.mean(x))) for v, x in sorted(vb.items())}
    return out


def interaction(r2, x, z, zx, good=None):
    """(zx - z) - (x - r2) on scene means (pair -> view -> scene), scene bootstrap 10 000 seed 0."""
    rows = [r2, x, z, zx]
    if good is not None:
        rows = [subset(R, good) for R in rows]
    m = [scene_means(R, "JF", "view") for R in rows]
    scenes = sorted(set.intersection(*[set(mm) for mm in m]))
    d = np.array([(m[3][s] - m[2][s]) - (m[1][s] - m[0][s]) for s in scenes])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, d.size, size=(10_000, d.size))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)], n_scenes=int(d.size))


def zoom_verdicts(root):
    """(scene, t, cam, obj) -> zoom verdict of that object-view (from zx_A/<scene>/<t>/zoom_conf.json)."""
    out = {}
    for s in sorted(os.listdir(root)):
        sd = os.path.join(root, s)
        if not os.path.isdir(sd):
            continue
        for t in sorted(os.listdir(sd)):
            p = os.path.join(sd, t, "zoom_conf.json")
            if not os.path.isfile(p):
                continue
            for cam, recs in json.load(open(p)).items():
                for o, rec in recs.items():
                    out[(s, int(t), cam, int(o))] = rec["verdict"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base", default=None, help="adopted x scoring (default <dir>/../_e13_dev/dev_pairs.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = a.base or os.path.join(os.path.dirname(os.path.abspath(a.dir)), "_e13_dev", "dev_pairs.json")
    arms = {"x": load(base)}
    for arm in ("r2", "z", "zx", "zxb"):
        p = os.path.join(a.dir, "scores", f"{arm}_A.json")
        if os.path.isfile(p):
            arms[arm] = load(p)
    kx = {KEY(r): r for r in arms["x"]}
    for arm, rows in arms.items():
        assert {KEY(r) for r in rows} == set(kx), arm
    good = {k for k, r in kx.items() if r["JF"] >= 0.8}
    fail = {k for k, r in kx.items() if r["JF"] < 0.8}
    zoomv = zoom_verdicts(os.path.join(a.dir, "zx_A"))
    pairs = [("x", "zx", "PRIMARY: zoom->xcl vs adopted x"), ("r2", "x", "xcl effect (no zoom)"), ("r2", "z", "zoom effect (no xcl, E10 C2)"),
             ("z", "zx", "xcl on top of zoom"), ("r2", "zx", "zoom->xcl vs geoset_rep"), ("x", "z", "zoom alone vs adopted x"),
             ("x", "zxb", "diag zxb vs adopted x"), ("zx", "zxb", "diag zxb vs zx")]
    res = dict(n_pairs=len(kx), n_good=len(good), n_fail=len(fail), base=base,
               zoom_verdict_counts={v: sum(1 for k in kx if zoomv.get(k, "no-candidate") == v)
                                    for v in sorted(set(zoomv.get(k, "no-candidate") for k in kx))},
               comparisons={})
    for a_, b_, desc in pairs:
        if a_ in arms and b_ in arms:
            res["comparisons"][f"{b_}-{a_}"] = dict(desc=desc, **comp(arms[a_], arms[b_], good, fail, zoomv))
    if all(k in arms for k in ("r2", "x", "z", "zx")):
        res["interaction_all"] = interaction(arms["r2"], arms["x"], arms["z"], arms["zx"])
        res["interaction_good"] = interaction(arms["r2"], arms["x"], arms["z"], arms["zx"], good)
        res["interaction_fail"] = interaction(arms["r2"], arms["x"], arms["z"], arms["zx"], fail)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    # ---- print
    print(f"pairs {res['n_pairs']} good {res['n_good']} fail {res['n_fail']}; zoom verdicts over pairs {res['zoom_verdict_counts']}")
    for name, c in res["comparisons"].items():
        A = c["all"]
        print(f"\n== {name}: {c['desc']}  (JF hier {c['base_JF_hier']:.4f} -> {c['cand_JF_hier']:.4f}; pair-mean {c['base_JF_pairmean']:.4f} -> {c['cand_JF_pairmean']:.4f})")
        print(f"  all  n={c['n_pairs']}  dJ&F {A['JF']['delta']:+.4f} [{A['JF']['ci'][0]:+.4f}, {A['JF']['ci'][1]:+.4f}]  "
              f"dJ {A['J']['delta']:+.4f} [{A['J']['ci'][0]:+.4f}, {A['J']['ci'][1]:+.4f}]  dF {A['F']['delta']:+.4f} [{A['F']['ci'][0]:+.4f}, {A['F']['ci'][1]:+.4f}]  "
              f"pair-mean {c['pair_mean_dJF_all']:+.4f}  changed {c['changed_pairs']}")
        for g in ("good", "fail"):
            G = c[g]
            print(f"  {g:4s} n={G['n']}  dJ&F {G['JF']['delta']:+.4f} [{G['JF']['ci'][0]:+.4f}, {G['JF']['ci'][1]:+.4f}]  "
                  f"dJ {G['J']['delta']:+.4f}  dF {G['F']['delta']:+.4f}  pair-mean {G['pair_mean_dJF']:+.4f}  up {G['up']} down {G['down']} changed {G['changed']}  "
                  f"1-rec {G['base_1mrec']:.4f}->{G['cand_1mrec']:.4f}  1-prec {G['base_1mprec']:.4f}->{G['cand_1mprec']:.4f}")
        print("  size " + "  ".join(f"{k}: n {v['n']} {v['dJF']:+.4f} (good {v['dJF_good'] if v['dJF_good'] is None else round(v['dJF_good'], 4)})" for k, v in c["size_bins"].items()))
        print("  zoom " + "  ".join(f"{k}: n {v['n']} {v['dJF']:+.4f}" for k, v in c["zoom_verdict_bins"].items()))
        worst = sorted(c["per_scene_JF"].items(), key=lambda kv: kv[1])[:3]
        print("  worst scenes " + ", ".join(f"{s} {d:+.4f}" for s, d in worst))
    for k in ("interaction_all", "interaction_good", "interaction_fail"):
        if k in res:
            print(f"{k}: {res[k]['delta']:+.4f} [{res[k]['ci'][0]:+.4f}, {res[k]['ci'][1]:+.4f}]")


if __name__ == "__main__":
    main()
