#!/usr/bin/env python3
"""S1-E5a: can a ground-truth-free signal tell a bad seed from a good one?  (docs/stage1-E5a-prereg.md)

Measurement only -- no gate, no verdict.  For every one of the same 796 rows as S1-E2/E4 the script
computes the registered signals on the RAW seed (MVSeed_e0_index) and reports how many of the 38
misplaced F0 pairs each one flags (hits) against how many of the 623 normal pairs it flags (false
positives), both at the registered operating point and at the threshold matched to the P5 area rule's
false-positive count.

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD 3dgs \\
        python MVSeed/geo/e5a_signals.py
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


warp = _load("MVSeed/geo/warp.py", "warp")
gate = _load("MVSeed/gate_det.py", "gate_det")

DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
CACHE = os.path.join(REPO, "MVSeed", "cache")
OUT = os.path.join(REPO, "MVSeed", "runs", "e5a_signals.json")
SEEDS, SEEDS_REV = "MVSeed_e0_index", "MVSeed_e1_reverse"
TAU_IOU = 0.5            # registered operating point for every IoU signal (plan section 4)
TAU_CTR = 1.0            # registered operating point for the centre-distance signals
P5_MIN_PROMPT = 64
P5_FRAC = 0.05
IOU_SIGNALS = ("S2a_iou", "S2b_iou", "S2max_iou", "S1_rev")
CTR_SIGNALS = ("S2a_ctr", "S2b_ctr")
# post-hoc diagnostics added after the registered readout (docs/stage1-E5a.md section 4): containment
# asks "is the seed INSIDE what geometry projected", which forgives a projection whose shape is poor.
DIAG_SIGNALS = ("S2a_cont", "S2b_cont", "S2max_cont")
SIGNALS = ("P5area",) + IOU_SIGNALS + CTR_SIGNALS
ALL_SIGNALS = SIGNALS + DIAG_SIGNALS
LOW_IS_FLAGGED = IOU_SIGNALS + DIAG_SIGNALS      # these flag LOW scores; CTR_SIGNALS flag HIGH ones


# ------------------------------------------------------------------ pure signal maths
def equivalent_diameter(area):
    """Diameter of the circle with the same area (0 for an empty mask)."""
    return 2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0


def centre_ratio(seed, proj):
    """Distance between the two centroids in units of the PROJECTION's equivalent diameter.

    inf when either mask is empty: a seed with no projection to agree with is flagged, which is the
    real operating condition of an arm whose scene failed to reconstruct (prereg section 2).
    """
    import numpy as np
    a, b = int(seed.sum()), int(proj.sum())
    if a == 0 or b == 0:
        return float("inf")
    sy, sx = np.nonzero(seed)
    py, px = np.nonzero(proj)
    d = float(np.hypot(sx.mean() - px.mean(), sy.mean() - py.mean()))
    return d / equivalent_diameter(b)


def p5_threshold(areas, min_area=P5_MIN_PROMPT, frac=P5_FRAC):
    """P5's per-view degeneracy threshold T(v) from that view's non-zero seed areas."""
    nz = sorted(a for a in areas if a > 0)
    if len(nz) < 3:
        return float(min_area)
    m = nz[len(nz) // 2] if len(nz) % 2 else (nz[len(nz) // 2 - 1] + nz[len(nz) // 2]) / 2.0
    return max(float(min_area), frac * m)


def p5_flag(seed_area, prompt_area, threshold, min_prompt=P5_MIN_PROMPT):
    """P5 area rule (prereg section 2): a healthy prompt whose seed collapsed in this view."""
    return bool(prompt_area >= min_prompt and seed_area < threshold)


def containment(seed, proj):
    """Fraction of the seed that falls inside the projection (0 when the seed is empty)."""
    a = int(seed.sum())
    return float((seed & proj).sum() / a) if a else 0.0


def flagged(row, name):
    """Is this row flagged by signal `name` at its registered operating point?"""
    v = row["signals"][name]
    if name == "P5area":
        return bool(v)
    if name in CTR_SIGNALS:
        return v is None or v > TAU_CTR          # None == inf (json has no infinity)
    assert name in LOW_IS_FLAGGED, name
    return v is None or v < TAU_IOU


# ------------------------------------------------------------------ readout (pure)
def _counts(rows, name):
    return sum(flagged(r, name) for r in rows)


def matched_threshold(pos, neg, name, target_fp, low=None):
    """Score threshold whose false-positive count on `neg` is <= target_fp, and the hits it gives.

    IoU-like signals flag LOW values, centre signals flag HIGH ones; both are swept over the observed
    scores so the comparison is at a matched false-positive count, not a matched threshold.
    `low` overrides the direction for a signal this module does not own (S1-E5b's S4_cos).
    """
    low = (name in LOW_IS_FLAGGED) if low is None else low
    def score(r):
        v = r["signals"][name]
        if v is None:
            return -1.0 if low else float("inf")       # missing projection: most-flagged end
        return v
    cand = sorted({score(r) for r in pos + neg if score(r) not in (float("inf"),)})
    best = None
    for t in cand + ([float("inf")] if not low else []):
        fp = sum(1 for r in neg if (score(r) < t if low else score(r) > t))
        if fp <= target_fp:
            hits = sum(1 for r in pos if (score(r) < t if low else score(r) > t))
            if best is None or hits > best["hits"]:
                best = {"threshold": None if t == float("inf") else round(t, 4), "hits": hits, "fp": fp}
    return best or {"threshold": None, "hits": 0, "fp": 0}


def summarize(rows):
    pos = [r for r in rows if r["set"] == "F0" and r.get("f0_kind") == "misplaced"]
    empty = [r for r in rows if r["set"] == "F0" and r.get("f0_kind") != "misplaced"]
    neg = [r for r in rows if r["set"] == "N0"]
    gt_empty = [r for r in rows if r["set"] == "gt_empty"]
    halluc = [r for r in gt_empty if r["seed_px"] > 0]
    s = {"n_rows": len(rows), "n_misplaced": len(pos), "n_empty_seed": len(empty), "n_normal": len(neg),
         "n_gt_empty": len(gt_empty), "n_gt_empty_nonempty_seed": len(halluc), "operating_point": {}}
    base_fp = _counts(neg, "P5area")
    for name in ALL_SIGNALS:
        s["operating_point"][name] = {
            "hits": _counts(pos, name), "n_pos": len(pos),
            "fp": _counts(neg, name), "n_neg": len(neg),
            "empty_seed_hits": _counts(empty, name),
            "gt_empty_hits": _counts(halluc, name), "n_gt_empty_nonempty": len(halluc),
            "hits_by_size": {b: [sum(1 for r in pos if r["size_bin"] == b and flagged(r, name)),
                                 sum(1 for r in pos if r["size_bin"] == b)] for b in gate.SIZE_BINS},
            "fp_by_size": {b: [sum(1 for r in neg if r["size_bin"] == b and flagged(r, name)),
                               sum(1 for r in neg if r["size_bin"] == b)] for b in gate.SIZE_BINS},
            "hits_by_rig": {g: [sum(1 for r in pos if r["rig"] == g and flagged(r, name)),
                                sum(1 for r in pos if r["rig"] == g)] for g in ("pinhole", "hemisphere")},
            "fp_by_rig": {g: [sum(1 for r in neg if r["rig"] == g and flagged(r, name)),
                              sum(1 for r in neg if r["rig"] == g)] for g in ("pinhole", "hemisphere")}}
    s["matched_fp"] = {"target_fp": base_fp,
                       **{name: matched_threshold(pos, neg, name, base_fp)
                          for name in ALL_SIGNALS if name != "P5area"}}
    combos = {"S2max_iou AND S1_rev": lambda r: flagged(r, "S2max_iou") and flagged(r, "S1_rev"),
              "S2b_ctr AND S1_rev": lambda r: flagged(r, "S2b_ctr") and flagged(r, "S1_rev"),
              "P5area OR S2max_iou": lambda r: flagged(r, "P5area") or flagged(r, "S2max_iou")}
    s["combinations"] = {k: {"hits": sum(1 for r in pos if f(r)), "fp": sum(1 for r in neg if f(r))}
                         for k, f in combos.items()}
    return s


# ------------------------------------------------------------------ data
def read_mask(path, shape=None):
    import cv2
    import numpy as np
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
    if m is None:
        return np.zeros(shape, bool) if shape else None
    if shape and m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def run_scene(scene, cfg, census, log):
    import cv2
    import numpy as np
    d = cfg[scene]
    start, folder = d["start_frame"], d["folder"]
    ref = census[scene]["c_ini"]
    targets = [c for c in census[scene]["cameras"] if c != ref]
    objects = [int(o) for o in census[scene]["seed_ids_c_ini"]]
    gt_ref = cv2.imread(os.path.join(DATA, folder, "Mask", ref, f"{start:06d}.png"), cv2.IMREAD_GRAYSCALE)
    shape = gt_ref.shape
    prompt_area = {o: int((gt_ref == o).sum()) for o in objects}
    rows = []
    for cam in targets:
        gt = cv2.imread(os.path.join(DATA, folder, "Mask", cam, f"{start:06d}.png"), cv2.IMREAD_GRAYSCALE)
        sdir = os.path.join(DATA, folder, SEEDS, cam, str(start))
        # P5's T(v): the median over the non-zero seed areas actually present in this view
        all_areas = []
        if os.path.isdir(sdir):
            for f in os.listdir(sdir):
                m = read_mask(os.path.join(sdir, f), shape)
                all_areas.append(int(m.sum()))
        T = p5_threshold(all_areas)
        for obj in objects:
            seed = read_mask(os.path.join(sdir, f"{obj}.png"), shape)
            rev = read_mask(os.path.join(DATA, folder, SEEDS_REV, cam, str(start), f"{obj}.png"), shape)
            pa = read_mask(os.path.join(CACHE, "proj_a", scene, cam, f"{obj}.png"), shape)
            pb = read_mask(os.path.join(CACHE, "proj_b", scene, cam, f"{obj}.png"), shape)
            ia, ib = warp.iou(seed, pa), warp.iou(seed, pb)
            ka, kb = containment(seed, pa), containment(seed, pb)
            ca, cb = centre_ratio(seed, pa), centre_ratio(seed, pb)
            inf2none = lambda x: None if math.isinf(x) else round(x, 4)
            rows.append(dict(scene=scene, cam=cam, obj=obj, ref_cam=ref,
                             gt_px=int((gt == obj).sum()), seed_px=int(seed.sum()),
                             prompt_px=prompt_area[obj], proj_a_px=int(pa.sum()), proj_b_px=int(pb.sum()),
                             p5_threshold=round(T, 1),
                             rig="hemisphere" if scene in gate.HEMISPHERE else "pinhole",
                             signals=dict(P5area=p5_flag(int(seed.sum()), prompt_area[obj], T),
                                          S2a_iou=round(ia, 4), S2b_iou=round(ib, 4),
                                          S2max_iou=round(max(ia, ib), 4),
                                          S1_rev=round(warp.iou(seed, rev), 4),
                                          S2a_ctr=inf2none(ca), S2b_ctr=inf2none(cb),
                                          S2a_cont=round(ka, 4), S2b_cont=round(kb, 4),
                                          S2max_cont=round(max(ka, kb), 4))))
    log(f"  {scene:16s} {len(rows)} rows")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    census = json.load(open(CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(census)
    f0, n0 = gate.membership(json.load(open(gate.spike.F0))["pairs"], json.load(open(gate.spike.N0))["pairs"])
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    rows = []
    for s in scenes:
        rows += [gate.tag_row(r, f0, n0) for r in run_scene(s, cfg, census, log)]
    summary = summarize(rows)
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    doc = {"schema": "MVSeed e5a_signals", "schema_version": 1,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head": head,
           "generated_by": "MVSeed/geo/e5a_signals.py", "argv": sys.argv,
           "prereg": "docs/stage1-E5a-prereg.md",
           "rules": {"seeds": SEEDS, "seeds_reverse": SEEDS_REV, "tau_iou": TAU_IOU, "tau_ctr": TAU_CTR,
                     "p5": {"min_prompt": P5_MIN_PROMPT, "frac": P5_FRAC},
                     "missing_projection": "treated as an empty mask -> IoU 0, centre ratio inf (flagged)"},
           "summary": summary, "rows": rows}
    json.dump(doc, open(args.out, "w"), indent=1)
    print_summary(summary)
    print(f"wrote {args.out}  ({len(rows)} rows, {time.time() - t0:.0f}s)")


def print_summary(s):
    print(f"\n[S1-E5a signals] rows {s['n_rows']}  misplaced {s['n_misplaced']}  empty-seed {s['n_empty_seed']}  "
          f"normal {s['n_normal']}  gt-empty {s['n_gt_empty']} (non-empty seed {s['n_gt_empty_nonempty_seed']})")
    print(f"  {'signal':12s} {'hits/38':>9s} {'fp/623':>9s}  {'empty':>5s} {'halluc':>7s}   hits by size / fp by size")
    for name in ALL_SIGNALS:
        o = s["operating_point"][name]
        hs = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in o["hits_by_size"].items())
        fs = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in o["fp_by_size"].items())
        print(f"  {name:12s} {o['hits']:4d}/{o['n_pos']:<4d} {o['fp']:4d}/{o['n_neg']:<4d}  "
              f"{o['empty_seed_hits']:5d} {o['gt_empty_hits']:3d}/{o['n_gt_empty_nonempty']:<3d}   {hs}  |  {fs}")
    m = s["matched_fp"]
    print(f"\n  matched to the P5 rule's {m['target_fp']} false positives:")
    for name in ALL_SIGNALS:
        if name == "P5area":
            continue
        v = m[name]
        print(f"    {name:12s} hits {v['hits']:3d}/{s['n_misplaced']}  fp {v['fp']:3d}  at threshold {v['threshold']}")
    print("  combinations:")
    for k, v in s["combinations"].items():
        print(f"    {k:24s} hits {v['hits']:3d}/{s['n_misplaced']}  fp {v['fp']:3d}")


if __name__ == "__main__":
    main()
