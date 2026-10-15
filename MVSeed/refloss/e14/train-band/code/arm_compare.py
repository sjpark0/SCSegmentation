#!/usr/bin/env python3
"""S1-E14 train-band: DEV-A comparison of trained arms with the adopted decode (host; never the keyframe).

Arms (score_arm.py outputs, same 3 300 DEV-A pairs; base = the adopted geoset_rep_x = C0):
  C0  geoset_rep_x       adopted (memory attention trained in the 2D repo 2662a84; its own sample/memory flow)
  C0' geoset_cut0_rep_x  E12 zero-dose retrain of the C0 recipe, current code (memory_rng flow)
  D1  geoset_dec_rep_x   E9 D1: C0 recipe + SAM mask decoder trained (flow = C0')
  D2  geoset_decb_rep_x  E9 D2: D1 + boundary-band BCE at W 1 (flow = D1)
Contrasts: paired scene-cluster bootstrap (mvseg2d.dev_compare.compare: views "<t>/<cam>", 10 000, seed 0).
Good seeds = base (C0) JF >= 0.8.  Boundary proxies on good seeds: J&F, +-2 px band oracle, 1-recall, 1-precision.
Decision flips per scored pair from the round-2 dumps (presence > 0, selected token) and round-1 memory-gate flips
(round1_conf.json: presence > 0 and iou >= tau 0.7) over all decoded non-reference cameras.
    stage1/.venv/bin/python arm_compare.py --out arm_compare.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
from mvseg2d.dev_compare import compare  # noqa: E402
from mvseg2d.paired import paired_scene_bootstrap  # noqa: E402

TB = os.path.join(S1, "geomem", "runs", "_e14_train-band")
DUMP = {"C0": os.path.join(S1, "geomem", "runs", "_e14_logitdump", "geoset_rep_x_A"),
        "C0'": os.path.join(TB, "geoset_cut0_rep_x_A"), "D1": os.path.join(TB, "geoset_dec_rep_x_A"),
        "D2": os.path.join(TB, "geoset_decb_rep_x_A")}
SCORE = {"C0": os.path.join(TB, "score_geoset_rep_x_A.json"), "C0'": os.path.join(TB, "score_geoset_cut0_rep_x_A.json"),
         "D1": os.path.join(TB, "score_geoset_dec_rep_x_A.json"), "D2": os.path.join(TB, "score_geoset_decb_rep_x_A.json")}
CONTRASTS = [("C0", "C0'"), ("C0", "D1"), ("C0", "D2"), ("C0'", "D1"), ("D1", "D2"), ("C0'", "D2")]
TAU = 0.7


def key(r):
    return (r["scene"], int(r["t"]), r["cam"], int(r["obj"]))


def load_rows(p):
    rows = json.load(open(p))["rows"]
    for r in rows:
        r.setdefault("view", f"{r['t']}/{r['cam']}")
    return rows


def decisions(root):
    """{(scene,t,cam,obj): (present, token)} for round 2 of eval cams; {(scene,t,cam,obj): conf} for round 1."""
    r2, r1 = {}, {}
    for scene in sorted(os.listdir(root)):
        sd = os.path.join(root, scene)
        if not os.path.isdir(sd):
            continue
        for td in sorted(os.listdir(sd)):
            ld = os.path.join(sd, td, "_logits")
            if not os.path.isdir(ld):
                continue
            t = int(td)
            for f in os.listdir(ld):
                if f.endswith(".npz") and not f.endswith("_mm.npz"):
                    with np.load(os.path.join(ld, f)) as z:
                        for o, p, tok, iou in zip(z["objects"], z["presence"], z["token"], z["ious"]):
                            r2[(scene, t, f[:-4], int(o))] = (bool(p > 0), int(tok), float(iou), float(p))
            rc = json.load(open(os.path.join(sd, td, "round1_conf.json")))
            for cam, objs in rc.items():
                for o, v in objs.items():
                    r1[(scene, t, cam, int(o))] = bool(v["presence"] > 0 and v["iou"] >= TAU)
    return r2, r1


def good_proxies(rows, good):
    rs = [r for r in rows if key(r) in good]
    rec = np.array([r["recall"] for r in rs])
    prec = np.array([r["precision"] if r["precision"] is not None else 0.0 for r in rs])
    return dict(n=len(rs), JF=float(np.mean([r["JF"] for r in rs])), J=float(np.mean([r["J"] for r in rs])),
                F=float(np.mean([r["F"] for r in rs])), or2=float(np.mean([r["or2"] for r in rs])),
                or2_gap=float(np.mean([r["or2"] - r["JF"] for r in rs])), one_minus_recall=float(1 - rec.mean()),
                one_minus_precision=float(1 - prec.mean()),
                fn_share_in2=float(sum(r["fn_in2"] for r in rs) / max(sum(r["fn_in2"] + r["fp_in2"] for r in rs), 1)),
                err_in2_per_band_px=float(sum(r["fn_in2"] + r["fp_in2"] for r in rs) / sum(r["band2"] for r in rs)),
                pred_over_gt_px=float(sum(r["seed_px"] for r in rs) / sum(r["gt_px"] for r in rs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = [k for k in SCORE if os.path.isfile(SCORE[k])]
    rows = {k: load_rows(SCORE[k]) for k in arms}
    base = {key(r): r for r in rows["C0"]}
    good = {k for k, r in base.items() if r["JF"] >= 0.8}
    res = dict(arms=arms, n_pairs=len(base), n_good=len(good), summary={}, contrasts={}, proxies={}, flips={})
    for k in arms:
        jf = np.array([r["JF"] for r in rows[k]])
        res["summary"][k] = dict(JF=float(jf.mean()), J=float(np.mean([r["J"] for r in rows[k]])),
                                 F=float(np.mean([r["F"] for r in rows[k]])), fail_J05=int(sum(r["J"] < 0.5 for r in rows[k])),
                                 empty=int(sum(r["seed_px"] == 0 for r in rows[k])), below08=int((jf < 0.8).sum()))
        res["proxies"][k] = good_proxies(rows[k], good)
    dec = {k: decisions(DUMP[k]) for k in arms if os.path.isdir(DUMP[k])}
    for a_, b_ in CONTRASTS:
        if a_ not in arms or b_ not in arms:
            continue
        A = {key(r): r for r in rows[a_]}
        B = {key(r): dict(r) for r in rows[b_]}
        for k in B:                      # kinds / failing split by the BASE C0 so every contrast splits the same pairs
            B[k]["kind"] = base[k].get("kind")
        Ar = [dict(A[k], kind=base[k].get("kind")) for k in sorted(A)]
        c = compare(Ar, [B[k] for k in sorted(A)])
        gk = sorted(good)
        cg = {m: paired_scene_bootstrap([A[k] for k in gk], [B[k] for k in gk], key=m, n=10_000, seed=0, view_key="view")
              for m in ("JF", "J", "F")}
        c["good"] = {m: dict(delta=v["delta"], ci=list(v["ci"])) for m, v in cg.items()}
        fk = sorted(set(base) - good)
        cf = paired_scene_bootstrap([A[k] for k in fk], [B[k] for k in fk], key="JF", n=10_000, seed=0, view_key="view")
        c["fail"] = dict(delta=cf["delta"], ci=list(cf["ci"]))
        for m in ("JF", "J", "F"):
            c["paired"][m].pop("per_scene", None)
        gk_ = [k for k in sorted(A) if k in good]
        c["pair_mean"] = dict(JF=float(np.mean([B[k]["JF"] - A[k]["JF"] for k in A])), J=float(np.mean([B[k]["J"] - A[k]["J"] for k in A])),
                              F=float(np.mean([B[k]["F"] - A[k]["F"] for k in A])),
                              JF_good=float(np.mean([B[k]["JF"] - A[k]["JF"] for k in gk_])),
                              JF_fail=float(np.mean([B[k]["JF"] - A[k]["JF"] for k in A if k not in good])),
                              up=int(sum(B[k]["JF"] - A[k]["JF"] > 0.01 for k in A)), down=int(sum(B[k]["JF"] - A[k]["JF"] < -0.01 for k in A)),
                              abs_lt_0005=float(np.mean([abs(B[k]["JF"] - A[k]["JF"]) < 0.005 for k in A])))
        c["per_scene_JF"] = {s: float(np.mean([B[k]["JF"] - A[k]["JF"] for k in A if k[0] == s])) for s in sorted({k[0] for k in A})}
        if a_ in dec and b_ in dec:
            (r2a, r1a), (r2b, r1b) = dec[a_], dec[b_]
            flip = {k for k in A if (r2a[k][0], r2a[k][1]) != (r2b[k][0], r2b[k][1])}
            pflip = {k for k in A if r2a[k][0] != r2b[k][0]}
            d = {k: B[k]["JF"] - A[k]["JF"] for k in A}
            N = len(A)
            same = [k for k in A if k not in flip]
            c["decisions"] = dict(
                n_r2_flip=len(flip), n_presence_flip=len(pflip), n_token_flip=len(flip - pflip),
                contrib_flip=float(sum(d[k] for k in flip) / N), contrib_same=float(sum(d[k] for k in same) / N),
                dJF_same_good=float(np.mean([d[k] for k in same if k in good])),
                n_same_good=sum(1 for k in same if k in good),
                r1_entries=len(set(r1a) & set(r1b)), r1_gate_flips=sum(r1a[k] != r1b[k] for k in set(r1a) & set(r1b)),
                r1_conf_a=float(np.mean([r1a[k] for k in r1a])), r1_conf_b=float(np.mean([r1b[k] for k in r1b])))
            sg = [k for k in same if k in good]
            if sg:
                cs = paired_scene_bootstrap([A[k] for k in sg], [B[k] for k in sg], key="JF", n=10_000, seed=0, view_key="view")
                c["decisions"]["same_good_ci"] = list(cs["ci"])
        res["contrasts"][f"{b_} - {a_}"] = c
    json.dump(res, open(a.out, "w"), indent=1)
    print("summary"); [print(f"  {k:4s}", {kk: round(v, 5) if isinstance(v, float) else v for kk, v in s.items()}) for k, s in res["summary"].items()]
    print("good-seed proxies"); [print(f"  {k:4s}", {kk: round(v, 5) if isinstance(v, float) else v for kk, v in s.items()}) for k, s in res["proxies"].items()]
    for nm, c in res["contrasts"].items():
        p = c["paired"]
        print(f"{nm:10s} dJF {p['JF']['delta']:+.4f} [{p['JF']['ci'][0]:+.4f},{p['JF']['ci'][1]:+.4f}] dJ {p['J']['delta']:+.4f} [{p['J']['ci'][0]:+.4f},{p['J']['ci'][1]:+.4f}] "
              f"dF {p['F']['delta']:+.4f} [{p['F']['ci'][0]:+.4f},{p['F']['ci'][1]:+.4f}] | good {c['good']['JF']['delta']:+.4f} [{c['good']['JF']['ci'][0]:+.4f},{c['good']['JF']['ci'][1]:+.4f}] "
              f"(J {c['good']['J']['delta']:+.4f} F {c['good']['F']['delta']:+.4f}) | fail {c['fail']['delta']:+.4f} [{c['fail']['ci'][0]:+.4f},{c['fail']['ci'][1]:+.4f}]")
        pm = c["pair_mean"]
        print(f"           pair-mean dJF {pm['JF']:+.4f} (J {pm['J']:+.4f} F {pm['F']:+.4f}) good {pm['JF_good']:+.4f} fail {pm['JF_fail']:+.4f} up {pm['up']} down {pm['down']} |d|<0.005 {pm['abs_lt_0005']:.3f}")
        ws = min(c["per_scene_JF"].items(), key=lambda kv: kv[1])
        print(f"           worst scene {ws[0]} {ws[1]:+.4f}; scenes up {sum(v > 0 for v in c['per_scene_JF'].values())}/17")
        if "decisions" in c:
            print("           decisions", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c["decisions"].items()})


if __name__ == "__main__":
    main()
