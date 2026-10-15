#!/usr/bin/env python3
"""S1-E14 train-band: the stability-fallback channel of the band arm on DEV-A (host; never the keyframe).

Round-2 final masks with the decoder's dynamic multimask fallback switched OFF (token 0 always: low_res = where(presence > 0,
token-0 logits, NO_OBJ_SCORE)), then the adopted path (two bilinear stages, threshold 0, xcl out).  Round 1 (memories) is
left as decoded (fallback on), so this is the output-side effect only.
  C0  token-0 logits from the logit dump (_e14_logitdump/geoset_rep_x_A <cam>_mm.npz mm_low_res[:, 0])
  D1 / D2  token-0 logits from the second pass (<preset>_A_t0 <cam>_t0.npz)
Also the D1 -> D2 blend of the token-0 logits (presence blended likewise) with token 0 forced, and per blend the share
of present objects whose token-0 stability (decoder _get_stability_scores: area(L > 0.05) / area(L > -0.05) at 288,
threshold 0.98) would have triggered the fallback.
    stage1/.venv/bin/python tok0.py --out tok0.json [--workers 10]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
sys.path.insert(0, os.path.join(S1, "geomem", "runs", "_e14_logitdump", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e14dump as E  # noqa: E402
from interp import score  # noqa: E402
from mvseg2d.io import load_labelmap  # noqa: E402
from mvseg2d.paired import paired_scene_bootstrap  # noqa: E402

DATA = os.path.join(S1, "Data")
TB = os.path.join(S1, "geomem", "runs", "_e14_train-band")
C0 = os.path.join(S1, "geomem", "runs", "_e14_logitdump", "geoset_rep_x_A")
T1, T2 = os.path.join(TB, "geoset_dec_rep_x_A_t0"), os.path.join(TB, "geoset_decb_rep_x_A_t0")
PAIRS = os.path.join(S1, "geomem", "runs", "_e13_dev", "dev_pairs.json")
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
DELTA, THRESH = 0.05, 0.98


def t0(root, scene, t, cam):
    z = E.load(root, scene, t, cam)
    p = os.path.join(root, scene, f"{t:06d}", "_logits", f"{cam}_t0.npz")
    with np.load(p) as y:
        z["t0"] = y["t0_low_res"].astype(np.float32)
    return z


def stability(low):
    ai = (low > DELTA).reshape(low.shape[0], -1).sum(1)
    au = (low > -DELTA).reshape(low.shape[0], -1).sum(1)
    return np.where(au > 0, ai / np.maximum(au, 1), 1.0)


def view_job(args):
    scene, t, cam, rows = args
    lab = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    z0 = E.load(C0, scene, t, cam, mm=True)
    hw = tuple(int(v) for v in z0["hw"])
    res, stab = {}, {}
    low = np.where((z0["presence"] > 0)[:, None, None], z0["mm_low_res"][:, 0].astype(np.float32), np.float32(E.NO_OBJ_SCORE))
    res["C0_tok0"] = score(E.masks_numpy(low.astype(np.float32), hw), z0["objects"], lab, rows)
    z1, z2 = t0(T1, scene, t, cam), t0(T2, scene, t, cam)
    assert list(z1["objects"]) == list(z2["objects"]) == list(z0["objects"])
    for a in ALPHAS:
        pa = (1 - a) * z1["presence"] + a * z2["presence"]
        la = ((1 - a) * z1["t0"] + a * z2["t0"]).astype(np.float32)
        present = pa > 0
        low = np.where(present[:, None, None], la, np.float32(E.NO_OBJ_SCORE)).astype(np.float32)
        res[f"tok0_a{a}"] = score(E.masks_numpy(low, hw), z1["objects"], lab, rows)
        st = stability(la)
        stab[f"a{a}"] = (int(present.sum()), int((present & (st < THRESH)).sum()))
    # sanity: the stored stability of D1 / D2 token 0 equals the host recomputation (present objects)
    chk = [float(np.max(np.abs(stability(z["t0"])[z["presence"] > 0] - z["stability"][z["presence"] > 0]), initial=0.0)) for z in (z1, z2)]
    return [(r["scene"], r["t"], r["cam"], r["obj"]) for r in rows], res, stab, chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    rows = json.load(open(PAIRS))["rows"]
    by = defaultdict(list)
    for r in rows:
        by[(r["scene"], int(r["t"]), r["cam"])].append(r)
    jobs = [(s, t, c, rs) for (s, t, c), rs in sorted(by.items())]
    keys, per, stab, chks = [], defaultdict(list), defaultdict(lambda: [0, 0]), []
    with Pool(a.workers) as pool:
        for ks, res, st, chk in pool.imap(view_job, jobs, chunksize=1):
            keys += ks
            chks.append(chk)
            for v, sc in res.items():
                per[v] += sc
            for v, (n, f) in st.items():
                stab[v][0] += n
                stab[v][1] += f
    base = {(r["scene"], int(r["t"]), r["cam"], int(r["obj"])): r["JF"] for r in rows}
    good = np.array([base[k] >= 0.8 for k in keys])
    mk = lambda v: [dict(scene=k[0], view=f"{k[1]}/{k[2]}", JF=(x[0] + x[1]) / 2, J=x[0], F=x[1]) for k, x in zip(keys, per[v])]
    # adopted-path scores (fallback on) of the three arms, from the score_arm outputs
    adopted = {}
    for nm, f in (("C0", "score_geoset_rep_x_A.json"), ("D1", "score_geoset_dec_rep_x_A.json"), ("D2", "score_geoset_decb_rep_x_A.json")):
        sr = {(r["scene"], int(r["t"]), r["cam"], int(r["obj"])): r for r in json.load(open(os.path.join(TB, f)))["rows"]}
        adopted[nm] = [dict(scene=k[0], view=f"{k[1]}/{k[2]}", JF=sr[k]["JF"], J=sr[k]["J"], F=sr[k]["F"]) for k in keys]
    out = dict(stability_check_max_abs=max(max(c) for c in chks), fallback_share={v: s[1] / max(s[0], 1) for v, s in stab.items()},
               variants={}, contrasts={})
    for v in per:
        arr = np.array(per[v])
        out["variants"][v] = dict(JF=float(arr[:, :2].mean()), J=float(arr[:, 0].mean()), F=float(arr[:, 1].mean()),
                                  JF_good=float(arr[good, :2].mean()))
    con = [("C0", adopted["C0"], "C0_tok0", mk("C0_tok0")), ("D1", adopted["D1"], "D1_tok0", mk("tok0_a0.0")),
           ("D2", adopted["D2"], "D2_tok0", mk("tok0_a1.0")), ("D1_tok0", mk("tok0_a0.0"), "D2_tok0", mk("tok0_a1.0")),
           ("C0_tok0", mk("C0_tok0"), "D1_tok0", mk("tok0_a0.0")), ("C0_tok0", mk("C0_tok0"), "D2_tok0", mk("tok0_a1.0"))]
    con += [("D1_tok0", mk("tok0_a0.0"), f"tok0_a{x}", mk(f"tok0_a{x}")) for x in ALPHAS[1:-1]]
    for na, A, nb, B in con:
        rec = {}
        for m in ("JF", "J", "F"):
            b = paired_scene_bootstrap(A, B, key=m, n=10_000, seed=0, view_key="view")
            rec[m] = dict(delta=b["delta"], ci=list(b["ci"]))
        bg = paired_scene_bootstrap([x for x, g in zip(A, good) if g], [x for x, g in zip(B, good) if g], key="JF", n=10_000, seed=0, view_key="view")
        rec["good"] = dict(delta=bg["delta"], ci=list(bg["ci"]))
        rec["pair_mean_JF"] = float(np.mean([y["JF"] - x["JF"] for x, y in zip(A, B)]))
        out["contrasts"][f"{nb} - {na}"] = rec
        print(f"{nb:12s} - {na:8s} dJF {rec['JF']['delta']:+.4f} [{rec['JF']['ci'][0]:+.4f},{rec['JF']['ci'][1]:+.4f}] dJ {rec['J']['delta']:+.4f} dF {rec['F']['delta']:+.4f} "
              f"| good {bg['delta']:+.4f} [{bg['ci'][0]:+.4f},{bg['ci'][1]:+.4f}] | pair-mean {rec['pair_mean_JF']:+.4f}")
    print("fallback share (token-0 stability < 0.98, present objects) by blend:", {k: round(v, 4) for k, v in out["fallback_share"].items()},
          "stability check", out["stability_check_max_abs"])
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
