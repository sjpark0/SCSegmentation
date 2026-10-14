#!/usr/bin/env python3
"""Summarise train_vis.json and keyframe_vis.json (host: stage1/.venv/bin/python vis_report.py train_vis.json keyframe_vis.json).

Part 1 (training samples of the adopted recipe): fraction of the supervised target GT that is invisible from the
reference camera -- pixel-weighted and per-(sample, object) distribution, per scene, per object size (GT area / image),
per object type, per sample; the trainer-weighted version (each object of a sample weighted min(1, 16/n_obj), the
expected supervision mass under choose_objects); and the share of the target GT that the reference memory labels
directly (visible AND on the reference GT of the same object).
Part 2 (keyframe, 664 scored pairs): for each seed folder the recall of the adopted model on visible vs invisible GT
(pixel-weighted and pair-mean), by group (all / 22 under-cover failing / 59 failing / 605 good), per scene, per size,
and the false-positive split by visibility; per-pair table for the 22 under-cover pairs.
"""
import json
import sys
from collections import defaultdict

import numpy as np

MAX_OBJECTS = 16
SIZE_BINS = [(0, 1e-3, "<0.1%"), (1e-3, 1e-2, "0.1-1%"), (1e-2, 5e-2, "1-5%"), (5e-2, 2e-1, "5-20%"), (2e-1, 2.0, ">=20%")]
MIN_PX = 100            # pair-mean recall on a class is taken over pairs with >= MIN_PX GT pixels in that class


def load(path):
    src = open(path).read()
    return json.loads(src[src.index("{"):])


def frac(a, b):
    return a / b if b else float("nan")


def size_bin(r):
    f = r["gt_px"] / r["img_px"]
    for lo, hi, name in SIZE_BINS:
        if lo <= f < hi:
            return name
    return SIZE_BINS[-1][2]


def q(xs, w=None):
    xs = np.asarray(xs, float)
    if xs.size == 0:
        return "n/a"
    if w is None:
        p = np.percentile(xs, [10, 25, 50, 75, 90])
    else:
        w = np.asarray(w, float)
        o = np.argsort(xs)
        c = np.cumsum(w[o]) / w.sum()
        p = [xs[o][np.searchsorted(c, t / 100)] for t in (10, 25, 50, 75, 90)]
    return "p10/25/50/75/90 = " + " ".join(f"{v:.3f}" for v in p)


def train_group(name, sel, wkey=None):
    """sel: rows with gt_px > 0.  wkey: optional per-row weight (trainer weighting)."""
    if not sel:
        return
    w = np.array([r.get(wkey, 1.0) if wkey else 1.0 for r in sel])
    G = np.array([r["gt_px"] for r in sel], float)
    inv = np.array([r["gt_px"] - r["gt_vis"] for r in sel], float)
    out = np.array([r["gt_outside"] for r in sel], float)
    occ = np.array([r["gt_occluded"] for r in sel], float)
    nod = np.array([r["gt_nodepth"] for r in sel], float)
    ongt = np.array([r["gt_on_refgt"] for r in sel], float)
    onany = np.array([r["gt_on_refany"] for r in sel], float)
    f = inv / G
    print(f"== {name}: {len(sel)} (sample,object) rows with GT in the target; GT px {G.sum():,.0f}"
          + (f"; weights = min(1, 16/n_obj), effective rows {w.sum():,.1f}" if wkey else ""))
    print(f"   pixel-weighted:  invisible {np.sum(w * inv) / np.sum(w * G):.3f}  = outside {np.sum(w * out) / np.sum(w * G):.3f}"
          f" + occluded {np.sum(w * occ) / np.sum(w * G):.3f} + nodepth {np.sum(w * nod) / np.sum(w * G):.3f};"
          f"  visible {np.sum(w * (G - inv)) / np.sum(w * G):.3f} of which on ref GT same obj {np.sum(w * ongt) / np.sum(w * G):.3f},"
          f" on ref other label {np.sum(w * (onany - ongt)) / np.sum(w * G):.3f}, on ref background {np.sum(w * (G - inv - onany)) / np.sum(w * G):.3f}")
    print(f"   row-mean invisible fraction {np.average(f, weights=w):.3f} ({q(f, w)}); rows with invisible frac"
          f" == 0: {np.sum(w * (f == 0)) / w.sum():.3f}, >= 0.2: {np.sum(w * (f >= 0.2)) / w.sum():.3f}, >= 0.5: {np.sum(w * (f >= 0.5)) / w.sum():.3f},"
          f" >= 0.8: {np.sum(w * (f >= 0.8)) / w.sum():.3f}, == 1: {np.sum(w * (f == 1)) / w.sum():.3f}")
    ext = (G - ongt) / G          # share of the target GT NOT labelled by the reference memory (invisible or visible-but-not-on-ref-GT)
    print(f"   row-mean 'not on ref GT of same obj' fraction {np.average(ext, weights=w):.3f} ({q(ext, w)}); pixel-weighted {np.sum(w * (G - ongt)) / np.sum(w * G):.3f}")
    fo, fc, fn = out / G, occ / G, nod / G
    print(f"   rows with outside frac >= 0.5: {np.sum(w * (fo >= 0.5)) / w.sum():.3f} (GT px share {np.sum(w * G * (fo >= 0.5)) / np.sum(w * G):.3f}); "
          f"occluded frac >= 0.5: {np.sum(w * (fc >= 0.5)) / w.sum():.3f}; nodepth frac >= 0.5: {np.sum(w * (fn >= 0.5)) / w.sum():.3f}")
    bins = ((0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01))
    print("   rows by invisible-fraction bin [0,.2) [.2,.5) [.5,.8) [.8,1]: rows " + " ".join(f"{np.sum(w * ((f >= lo) & (f < hi))) / w.sum():.3f}" for lo, hi in bins)
          + "; GT px share " + " ".join(f"{np.sum(w * G * ((f >= lo) & (f < hi))) / np.sum(w * G):.3f}" for lo, hi in bins)
          + "; invisible px share " + " ".join(f"{np.sum(w * inv * ((f >= lo) & (f < hi))) / np.sum(w * inv):.3f}" for lo, hi in bins))


def train_table(rows, key, label, min_rows=1):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    print(f"\n   {label:<22s} {'rows':>6s} {'GT px':>13s} {'inv pw':>7s} {'out':>6s} {'occ':>6s} {'nod':>6s} {'onrefGT':>8s} {'inv rowmean':>11s} {'rows>=0.5':>9s} {'rows==0':>8s}")
    order = [k for k, _ in sorted(groups.items(), key=lambda kv: -sum(r['gt_px'] - r['gt_vis'] for r in kv[1]))] \
        if label != "size (GT/image)" else [b[2] for b in SIZE_BINS if b[2] in groups]
    for k in order:
        sel = groups[k]
        if len(sel) < min_rows:
            continue
        G = sum(r["gt_px"] for r in sel)
        inv = sum(r["gt_px"] - r["gt_vis"] for r in sel)
        f = np.array([(r["gt_px"] - r["gt_vis"]) / r["gt_px"] for r in sel])
        print(f"   {str(k):<22s} {len(sel):6d} {G:13,d} {frac(inv, G):7.3f} {frac(sum(r['gt_outside'] for r in sel), G):6.3f}"
              f" {frac(sum(r['gt_occluded'] for r in sel), G):6.3f} {frac(sum(r['gt_nodepth'] for r in sel), G):6.3f}"
              f" {frac(sum(r['gt_on_refgt'] for r in sel), G):8.3f} {f.mean():11.3f} {np.mean(f >= 0.5):9.3f} {np.mean(f == 0):8.3f}")


def part_train(d):
    rows, samples = d["rows"], d["samples"]
    nobj = {(s["scene"], s["t"], s["ref"], s["cam"]): s["n_obj"] for s in samples}
    for r in rows:
        r["w"] = min(1.0, MAX_OBJECTS / nobj[(r["scene"], r["t"], r["ref"], r["cam"])])
    present = [r for r in rows if r["gt_px"] > 0]
    absent = [r for r in rows if r["gt_px"] == 0]
    print(f"### PART 1 -- training samples of the adopted recipe (meta: {d['meta']})")
    print(f"samples {len(samples)} = 17 scenes x 11 frames (stride 2) x 6 ordered pairs of the 3 annotated cameras; "
          f"(sample,object) rows {len(rows)} (objects of the reference label map), of which {len(present)} have GT in the target "
          f"and {len(absent)} are absent from the target (presence-negative); samples with > {MAX_OBJECTS} objects: "
          f"{sum(s['n_obj'] > MAX_OBJECTS for s in samples)} / {len(samples)} (trainer draws {MAX_OBJECTS} uniformly); "
          f"objects labelled in the target but not in the reference map (never supervised): {sum(s['n_obj_tgt_only'] for s in samples)} over all samples")
    vis_img = np.array([s["vis_px"] / s["img_px"] for s in samples])
    print(f"target image visible from the reference (fraction of all pixels), sample-mean {vis_img.mean():.3f} ({q(vis_img)}); "
          f"outside {np.mean([s['outside_px'] / s['img_px'] for s in samples]):.3f} occluded {np.mean([s['occluded_px'] / s['img_px'] for s in samples]):.3f} "
          f"nodepth {np.mean([s['nodepth_px'] / s['img_px'] for s in samples]):.3f}\n")
    train_group("ALL rows (every object of the reference map, unweighted)", present)
    train_group("ALL rows, trainer-weighted", present, wkey="w")
    train_group("ALL rows, frame t = first labelled frame only (the keyframe's frame)", [r for r in present if r["t"] == min(x["t"] for x in rows if x["scene"] == r["scene"])])
    train_group("ALL rows, reference = the scene's keyframe reference camera (meta ref_cam)",
                [r for r in present if r["ref"] == d.get("ref_cams", {}).get(r["scene"], r["ref"])])
    # per sample
    per_s = defaultdict(lambda: [0, 0])
    for r in present:
        k = (r["scene"], r["t"], r["ref"], r["cam"])
        per_s[k][0] += r["gt_px"]
        per_s[k][1] += r["gt_px"] - r["gt_vis"]
    fs = np.array([v[1] / v[0] for v in per_s.values()])
    print(f"\n   per-SAMPLE invisible fraction (all objects of the sample pooled): mean {fs.mean():.3f} ({q(fs)}); "
          f"samples >= 0.5: {np.mean(fs >= 0.5):.3f}, >= 0.2: {np.mean(fs >= 0.2):.3f}")
    train_table(present, lambda r: r["scene"], "scene")
    train_table(present, size_bin, "size (GT/image)")
    train_table(present, lambda r: r["type"], "object type")
    train_table(present, lambda r: r["label"], "label (top by inv px)", min_rows=30)
    # the two camera roles: the 664 keyframe pairs use the scene's ref_cam as reference; in training every annotated camera is a reference
    print("\n   per (scene, reference camera): invisible fraction pixel-weighted / row-mean / rows")
    g = defaultdict(list)
    for r in present:
        g[(r["scene"], r["ref"])].append(r)
    for k in sorted(g):
        sel = g[k]
        G = sum(r["gt_px"] for r in sel)
        f = np.array([(r["gt_px"] - r["gt_vis"]) / r["gt_px"] for r in sel])
        print(f"      {k[0]:<20s} {k[1]:<12s} {frac(sum(r['gt_px'] - r['gt_vis'] for r in sel), G):.3f} / {f.mean():.3f} / {len(sel)}")


def kf_group(name, sel, seeds):
    if not sel:
        return
    G = sum(r["gt_px"] for r in sel)
    cls = {"vis": "gt_vis", "outside": "gt_outside", "occluded": "gt_occluded", "nodepth": "gt_nodepth", "on_refgt": "gt_on_refgt"}
    tot = {k: sum(r[v] for r in sel) for k, v in cls.items()}
    inv = G - tot["vis"]
    print(f"== {name}: {len(sel)} pairs, GT px {G:,} = visible {frac(tot['vis'], G):.3f} (on ref GT same obj {frac(tot['on_refgt'], G):.3f})"
          f" + invisible {frac(inv, G):.3f} (outside {frac(tot['outside'], G):.3f} occluded {frac(tot['occluded'], G):.3f} nodepth {frac(tot['nodepth'], G):.3f});"
          f" pair-mean invisible fraction {np.mean([(r['gt_px'] - r['gt_vis']) / r['gt_px'] for r in sel]):.3f}")
    print(f"   {'seeds':<28s} {'JF':>6s} | recall pixel-weighted: {'vis':>5s} {'invis':>5s} {'out':>5s} {'occ':>5s} {'nod':>5s} {'onrefGT':>7s} | pair-mean (>= {MIN_PX} px): {'vis':>5s} {'n':>4s} {'invis':>5s} {'n':>4s} | FP px: {'total':>9s} {'vis':>5s} {'invis':>5s} | pred px / GT px")
    for s in seeds:
        S = [r["seeds"][s] for r in sel]
        hv, hi = sum(x["hit_vis"] for x in S), sum(x["hit_px"] - x["hit_vis"] for x in S)
        ho, hc, hn, hg = (sum(x["hit_outside"] for x in S), sum(x["hit_occluded"] for x in S), sum(x["hit_nodepth"] for x in S),
                          sum(x["hit_on_refgt"] for x in S))
        fp = sum(x["fp_px"] for x in S)
        fpv = sum(x["fp_vis"] for x in S)
        pv = [x["hit_vis"] / r["gt_vis"] for x, r in zip(S, sel) if r["gt_vis"] >= MIN_PX]
        pi = [(x["hit_px"] - x["hit_vis"]) / (r["gt_px"] - r["gt_vis"]) for x, r in zip(S, sel) if r["gt_px"] - r["gt_vis"] >= MIN_PX]
        jf = np.mean([x["JF"] for x in S if x["JF"] is not None]) if any(x["JF"] is not None for x in S) else float("nan")
        print(f"   {s:<28s} {jf:6.4f} | {'':>22s} {frac(hv, tot['vis']):5.3f} {frac(hi, inv):5.3f} {frac(ho, tot['outside']):5.3f}"
              f" {frac(hc, tot['occluded']):5.3f} {frac(hn, tot['nodepth']):5.3f} {frac(hg, tot['on_refgt']):7.3f} |"
              f" {'':>21s} {np.mean(pv) if pv else float('nan'):5.3f} {len(pv):4d} {np.mean(pi) if pi else float('nan'):5.3f} {len(pi):4d} |"
              f" {fp:9,d} {frac(fpv, fp):5.3f} {frac(fp - fpv, fp):5.3f} | {frac(sum(x['pred_px'] for x in S), G):.3f}")
    print()


def part_keyframe(d):
    rows = d["rows"]
    seeds = d["meta"]["seeds"]
    print(f"\n\n### PART 2 -- keyframe, 664 scored pairs (meta: {d['meta']})")
    print(f"recall on a class = |pred & GT & class| / |GT & class|; groups follow seedfail.json (kinds and failing = JF < 0.8 of the e8 seeds)\n")
    uc = [r for r in rows if r["failing_e8"] and r["kind"].startswith("under-cover")]
    fail = [r for r in rows if r["failing_e8"]]
    good = [r for r in rows if not r["failing_e8"]]
    kf_group("all 664", rows, seeds)
    kf_group("under-cover failing (22)", uc, seeds)
    kf_group("under-cover failing without Dog camera_0004 tree (21)", [r for r in uc if not (r["scene"] == "Dog" and r["cam"] == "camera_0004" and r["obj"] == 3)], seeds)
    kf_group(f"all failing ({len(fail)})", fail, seeds)
    kf_group(f"good seeds ({len(good)})", good, seeds)
    for name in SIZE_BINS:
        kf_group(f"size {name[2]}", [r for r in rows if size_bin(r) == name[2]], seeds)
    for sc in sorted({r["scene"] for r in rows}):
        kf_group(f"scene {sc}", [r for r in rows if r["scene"] == sc], seeds)
    s0 = seeds[0]
    print(f"=== per under-cover failing pair, seeds = {s0} (sorted by stage-2 cost) ===")
    print("scene cam obj label | cost | GT px: invis frac (out/occ/nod) | recall vis / invis (out/occ/nod) | FP px (invis share) | JF e8 -> e11")
    for r in sorted(uc, key=lambda r: -r["cost"]):
        x = r["seeds"][s0]
        G, gv = r["gt_px"], r["gt_vis"]
        gi = G - gv
        print(f"{r['scene']} {r['cam']} {r['obj']} {r['label']} | {r['cost']:.3f} | {G:,}: {frac(gi, G):.2f} ({frac(r['gt_outside'], G):.2f}/{frac(r['gt_occluded'], G):.2f}/{frac(r['gt_nodepth'], G):.2f})"
              f" | {frac(x['hit_vis'], gv):.3f} / {frac(x['hit_px'] - x['hit_vis'], gi):.3f} ({frac(x['hit_outside'], r['gt_outside']):.2f}/{frac(x['hit_occluded'], r['gt_occluded']):.2f}/{frac(x['hit_nodepth'], r['gt_nodepth']):.2f})"
              f" | {x['fp_px']:,} ({frac(x['fp_px'] - x['fp_vis'], x['fp_px']):.2f}) | {r['JF_e8']:.3f} -> {x['JF']:.3f}")
    # the invisible-GT pixels the adopted model does NOT cover, over all 664 and over the 22, by class
    for name, sel in (("all 664", rows), ("under-cover 22", uc), ("good 605", good)):
        x = [r["seeds"][s0] for r in sel]
        miss_inv = sum(r["gt_px"] - r["gt_vis"] - (y["hit_px"] - y["hit_vis"]) for r, y in zip(sel, x))
        miss_vis = sum(r["gt_vis"] - y["hit_vis"] for r, y in zip(sel, x))
        by = {c: sum(r[f"gt_{c}"] - y[f"hit_{c}"] for r, y in zip(sel, x)) for c in ("outside", "occluded", "nodepth")}
        print(f"\n   {name}: missed GT px of {s0} = {miss_inv + miss_vis:,} of which invisible {miss_inv:,} ({frac(miss_inv, miss_inv + miss_vis):.3f})"
              f" = outside {by['outside']:,} ({frac(by['outside'], miss_inv + miss_vis):.3f}) + occluded {by['occluded']:,} ({frac(by['occluded'], miss_inv + miss_vis):.3f})"
              f" + nodepth {by['nodepth']:,} ({frac(by['nodepth'], miss_inv + miss_vis):.3f}); visible {miss_vis:,}")
    # recall as a function of the pair's own invisible fraction
    print(f"\n=== keyframe pairs binned by their invisible-GT fraction, seeds = {s0} (pair-mean recall over pairs with >= {MIN_PX} px in the class; J&F = mean) ===")
    print(f"   {'invis frac bin':<14s} {'pairs':>5s} {'GT px':>12s} {'rec vis pw':>10s} {'rec inv pw':>10s} {'rec vis pm':>10s} {'rec inv pm':>10s} {'JF':>6s} {'failing':>7s} {'under-cover':>11s}")
    for lo, hi in ((0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
        sel = [r for r in rows if lo <= (r["gt_px"] - r["gt_vis"]) / r["gt_px"] < hi]
        if not sel:
            continue
        x = [r["seeds"][s0] for r in sel]
        gv, gi = sum(r["gt_vis"] for r in sel), sum(r["gt_px"] - r["gt_vis"] for r in sel)
        pv = [y["hit_vis"] / r["gt_vis"] for r, y in zip(sel, x) if r["gt_vis"] >= MIN_PX]
        pi = [(y["hit_px"] - y["hit_vis"]) / (r["gt_px"] - r["gt_vis"]) for r, y in zip(sel, x) if r["gt_px"] - r["gt_vis"] >= MIN_PX]
        print(f"   [{lo:.1f}, {hi if hi <= 1 else 1.0:.1f}{')' if hi <= 1 else ']':<6s} {len(sel):5d} {sum(r['gt_px'] for r in sel):12,d}"
              f" {frac(sum(y['hit_vis'] for y in x), gv):10.3f} {frac(sum(y['hit_px'] - y['hit_vis'] for y in x), gi):10.3f}"
              f" {np.mean(pv) if pv else float('nan'):10.3f} {np.mean(pi) if pi else float('nan'):10.3f} {np.mean([y['JF'] for y in x]):6.3f}"
              f" {sum(r['failing_e8'] for r in sel):7d} {sum(r in uc for r in sel):11d}")


if __name__ == "__main__":
    tr = load(sys.argv[1])
    # the scenes' keyframe reference cameras (meta ref_cam), for the role split
    import os
    data = "/home/sjpark/Documents/SCSegmentation/stage1/Data"
    tr["ref_cams"] = {s: json.load(open(os.path.join(data, s, "meta.json")))["ref_cam"] for s in sorted(os.listdir(data))}
    part_train(tr)
    if len(sys.argv) > 2:
        part_keyframe(load(sys.argv[2]))
