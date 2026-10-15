#!/usr/bin/env python3
"""S1-E13 DEV report (host: stage1/.venv/bin/python dev_report.py <dev_pairs.json> [--uc-json out.json]).

Reads dev_score.py rows (DEV frames only; the keyframe is never read) and prints:
  (1) the dev set size; mean J / F / J&F over pairs, per scene, scene-mean; scene-cluster bootstrap 95 % CI (10 000, seed 0);
      round-1 J&F of the same pairs when present
  (2) failing pairs (J&F < 0.8) by the seedcat kind (MVSeed/refloss/seedcat.py rules)
  (3) the DEV UNDER-COVER population (task definition):  J&F < 0.8  AND  gt_invis_frac >= 0.5  AND  recall < 0.85
      gt_invis_frac = 1 - |GT & vis| / |GT|, vis = Pi3 reference visibility of MVSeed/refloss/e11/extent_vis.py (OCC_TOL 0.01)
      -> count, per scene, per (scene, cam, obj) across frames, missed-pixel class split (pixel-weighted / pair-mean),
         k=3 round-1 neighbour coverage of the missed pixels, share of the dev J&F deficit
  (4) for comparison, the seedcat-kind under-cover failing pairs (the keyframe's 22-pair definition) with the same split
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

CLS = ("vis", "outside", "occluded", "nodepth")
REF = {}


def idx_ref(r):
    return REF[r["scene"]]


def frac(a, b):
    return a / b if b else float("nan")


def boot_ci(rows, key="JF", n=10000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r[key])
    scenes = sorted(by)
    sums = np.array([np.sum(by[s]) for s in scenes])
    cnts = np.array([len(by[s]) for s in scenes])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scenes), size=(n, len(scenes)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def split(name, sel, all_rows):
    n = len(sel)
    print(f"== {name}: {n} pairs ({len({(r['scene'], r['cam'], r['obj']) for r in sel})} distinct scene/cam/obj, "
          f"{len({r['scene'] for r in sel})} scenes)")
    if not n:
        return {}
    M = sum(r["missed_px"] for r in sel)
    G = sum(r["gt_px"] for r in sel)
    deficit = sum(1 - r["JF"] for r in sel) / max(1e-12, sum(1 - r["JF"] for r in all_rows))
    print(f"   mean J&F {np.mean([r['JF'] for r in sel]):.4f}  mean recall {np.mean([r['recall'] for r in sel]):.3f}  "
          f"mean precision {np.nanmean([r['precision'] if r['precision'] is not None else np.nan for r in sel]):.3f}  "
          f"share of the dev J&F deficit sum(1-JF): {deficit:.3f}  missed px {M:,} of GT px {G:,}")
    out = {"n": n, "missed_px": M, "gt_px": G, "mean_JF": float(np.mean([r["JF"] for r in sel])), "deficit_share": deficit}
    print("   fraction of MISSED pixels                        pixel-weighted   pair-mean")
    for k, lab in (("vis", "(a)  visible from c_ini"), ("outside", "(b1) outside c_ini image / behind"),
                   ("occluded", "(b2) occluded for c_ini"), ("nodepth", "(b3) no valid Pi3 depth"),
                   ("on_refgt", "     (a) and on c_ini GT of same obj")):
        pw = frac(sum(r[f"missed_{k}"] for r in sel), M)
        pm = float(np.nanmean([frac(r[f"missed_{k}"], r["missed_px"]) for r in sel if r["missed_px"] > 0]))
        print(f"   {lab:<48s} {pw:7.3f}        {pm:7.3f}")
        out[f"missed_{k}"] = dict(pixel_weighted=pw, pair_mean=pm)
    nv = sum(r["missed_outside"] + r["missed_occluded"] + r["missed_nodepth"] for r in sel)
    out["missed_invisible"] = frac(nv, M)
    print(f"   (b) not visible total: pixel-weighted {frac(nv, M):.3f}")
    if "missed_nb_k3_r1" in sel[0]:
        pw = frac(sum(r["missed_nb_k3_r1"] for r in sel), M)
        pm = float(np.nanmean([frac(r["missed_nb_k3_r1"], r["missed_px"]) for r in sel if r["missed_px"] > 0]))
        print(f"   covered by the k=3 candidates' ROUND-1 masks (reprojected)   {pw:7.3f}        {pm:7.3f}")
        out["missed_nb_k3_r1"] = dict(pixel_weighted=pw, pair_mean=pm)
    big = max(sel, key=lambda r: r["missed_px"])
    print(f"   largest single pair: {big['scene']} {big['cam']} obj {big['obj']} ({big['label']}) t={big['t']} "
          f"= {frac(big['missed_px'], M):.2f} of the missed px")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--uc-json", default=None, help="write the under-cover population rows here")
    a = ap.parse_args()
    d = json.load(open(a.pairs))
    rows, meta = d["rows"], d["meta"]
    REF.update({s: v["ref_frame"] for s, v in json.load(open(meta["index"].replace("/host", "", 1)))["scenes"].items()})
    for r in rows:                                   # class counts partition the GT and the missed set
        assert sum(r[f"gt_{c}"] for c in CLS) == r["gt_px"] and sum(r[f"missed_{c}"] for c in CLS) == r["missed_px"], r
    print(f"meta: {meta}")
    frames = {(r["scene"], r["t"]) for r in rows}
    print(f"\n(1) DEV set: {len(rows)} pairs (GT non-empty) over {len(frames)} frames x 2 cams, {len({r['scene'] for r in rows})} scenes; "
          f"{len(d['empty_gt'])} (t, cam, obj) with empty GT excluded ({sum(e['seed_px'] > 0 for e in d['empty_gt'])} of them "
          f"with a non-empty prediction)")
    lo, hi = boot_ci(rows)
    print(f"   mean J {np.mean([r['J'] for r in rows]):.4f}  F {np.mean([r['F'] for r in rows]):.4f}  "
          f"J&F {np.mean([r['JF'] for r in rows]):.4f}  [scene-cluster bootstrap 95 % {lo:.4f}, {hi:.4f}]")
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r)
    print(f"   scene-mean J&F {np.mean([np.mean([r['JF'] for r in v]) for v in by.values()]):.4f}")
    if "JF_r1" in rows[0]:
        print(f"   round-1 J&F of the same pairs {np.mean([r['JF_r1'] for r in rows]):.4f} "
              f"(round 2 - round 1 = {np.mean([r['JF'] - r['JF_r1'] for r in rows]):+.4f})")
    print("   per scene: pairs  J&F  (failing < 0.8)")
    for s in sorted(by):
        v = by[s]
        print(f"     {s:20s} {len(v):4d}  {np.mean([r['JF'] for r in v]):.4f}  ({sum(r['JF'] < 0.8 for r in v)})")
    print("   by frame offset t - ref: " + ", ".join(
        f"+{o}: {np.mean([r['JF'] for r in rows if r['t'] - idx_ref(r) == o]):.4f}" for o in sorted({r['t'] - idx_ref(r) for r in rows})))
    fail = [r for r in rows if r["JF"] < 0.8]
    print(f"\n(2) failing (J&F < 0.8): {len(fail)} pairs = {len(fail) / len(rows):.3f}; deficit share "
          f"{sum(1 - r['JF'] for r in fail) / sum(1 - r['JF'] for r in rows):.3f}; by seedcat kind:")
    for k, n in Counter(r["kind"] for r in fail).most_common():
        sel = [r for r in fail if r["kind"] == k]
        print(f"     {k:28s} {n:4d}  mean J&F {np.mean([r['JF'] for r in sel]):.3f}")
    uc = [r for r in rows if r["JF"] < 0.8 and r["gt_invis_frac"] >= 0.5 and r["recall"] < 0.85]
    print("\n(3) DEV UNDER-COVER (J&F < 0.8, GT >= 50 % c_ini-invisible, recall < 0.85)")
    res = split("dev under-cover", uc, rows)
    print("   per scene:", ", ".join(f"{s} {n}" for s, n in sorted(Counter(r["scene"] for r in uc).items())))
    print("   seedcat kinds:", dict(Counter(r["kind"] for r in uc)))
    grp = defaultdict(list)
    for r in uc:
        grp[(r["scene"], r["cam"], r["obj"])].append(r)
    print("   by (scene, cam, obj) across the 5 frames: n_frames  mean J&F  mean GT-invisible  missed px  split vis/out/occ/nodepth  k3-r1 cover")
    for key, v in sorted(grp.items(), key=lambda kv: -sum(r["missed_px"] for r in kv[1])):
        M = sum(r["missed_px"] for r in v)
        sp = " / ".join(f"{frac(sum(r[f'missed_{c}'] for r in v), M):.2f}" for c in CLS)
        nb = frac(sum(r.get("missed_nb_k3_r1", 0) for r in v), M)
        print(f"     {key[0]:18s} {key[1]:12s} {key[2]:3d} {v[0]['label'] or '-':22s} {len(v)}  {np.mean([r['JF'] for r in v]):.3f}  "
              f"{np.mean([r['gt_invis_frac'] for r in v]):.2f}  {M:9,d}  {sp}  {nb:.2f}")
    if grp:
        top = max(grp, key=lambda k: sum(r["missed_px"] for r in grp[k]))
        split(f"dev under-cover without its largest group {top}", [r for r in uc if (r["scene"], r["cam"], r["obj"]) != top], rows)
    hp = [r for r in uc if r["precision"] is not None and r["precision"] >= 0.85]
    split("dev under-cover AND precision >= 0.85 (clean under-cover: the seed is right where it is)", hp, rows)
    split("dev under-cover AND (precision < 0.85 or empty seed): also on the wrong pixels", [r for r in uc if r not in hp], rows)
    print("   the same two subsets by seedcat kind:", dict(Counter(r["kind"] for r in hp)), "|",
          dict(Counter(r["kind"] for r in uc if r not in hp)))
    print("   dev under-cover by frame offset t - ref:", dict(sorted(Counter(r["t"] - idx_ref(r) for r in uc).items())))
    ucs = [r for r in fail if r["kind"].startswith("under-cover")]
    print("\n(4) comparison: seedcat-kind under-cover failing (recall < 0.85, precision >= 0.85, J&F < 0.8; the keyframe's 22-pair rule)")
    split("seedcat under-cover failing", ucs, rows)
    print(f"   overlap with (3): {len({id(r) for r in ucs} & {id(r) for r in uc})} pairs")
    if a.uc_json:
        json.dump(dict(definition="JF < 0.8 and gt_invis_frac >= 0.5 and recall < 0.85", summary=res, rows=uc), open(a.uc_json, "w"), indent=1)


if __name__ == "__main__":
    main()
