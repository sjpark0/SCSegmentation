#!/usr/bin/env python3
"""Overlap oracle on the adopted stage-1 seeds (S1-E11 data analysis; host python, numpy/cv2, no GPU).

For every scored view (meta.mask_cams minus ref_cam) of the adopted seeds
stage1/Data/<scene>/Result/geoset_pred_ft_rep_e8/<cam>/<obj>.png (objects = ids of the reference label map,
exactly as stage1/mvseg2d/evaluate.py scores them) we measure how the independently decoded object masks overlap
and what re-assigning the overlapping pixels would do to the keyframe J&F.

An "overlap pixel" is a pixel covered by >= 2 predicted masks of the same view; its "contenders" are those masks.
Rules (each one touches ONLY overlap pixels; every other pixel of every mask is unchanged):

  oracle        the pixel stays only in the mask of the object the GT label map names there; when the GT names
                background or an object that is not a contender, the pixel is removed from every contender
  oracle_pick   as oracle, but when the GT object is not a contender the pixel is left as it is (all keep it)
  larger        the pixel goes to the contender with the largest predicted area
  smaller       ... the smallest predicted area
  centre        ... whose predicted-mask bounding-box centre is nearest to the pixel
  centroid      ... whose predicted-mask centroid (mean pixel position) is nearest to the pixel
  erase         the pixel is removed from every contender

Scoring is mvseg2d.metrics.jf_pair (DAVIS J, boundary F with the 0.8 % diagonal tolerance) on the 664 scored pairs
(gt_px > 0), aggregated pair -> camera -> scene -> mean over scenes (mvseg2d.paired.scene_means, the keyframe
headline) and as a plain pair mean; scene-cluster bootstrap (mvseg2d.paired.paired_scene_bootstrap, 10 000
resamples, seed 0) for the CI of the rule - baseline difference.  Subset gains ("restricted to the 14 spill pairs",
"the 605 good seeds") apply the rule to those pairs only and keep every other pair at its baseline value.

    stage1/.venv/bin/python overlap_oracle.py [--out overlap_oracle.json] [--procs 17]
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

REPO = "/home/sjpark/Documents/SCSegmentation"
S1 = os.path.join(REPO, "stage1")
sys.path.insert(0, S1)
from mvseg2d.evaluate import all_scenes, scored_views, view_dist  # noqa: E402
from mvseg2d.io import labelmap_objects, load_labelmap, load_meta, read_pred_mask, result_dir, size_bin  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402
from mvseg2d.paired import paired_scene_bootstrap, scene_means  # noqa: E402

DATA = os.path.join(S1, "Data")
METHOD = "geoset_pred_ft_rep_e8"
SEEDFAIL = os.path.join(REPO, "MVSeed", "runs", "seedfail.json")
KF_C0 = os.path.join(REPO, "MVSeed", "runs", "e10", "kf_geoset_pred_ft_rep_e8.json")
RULES = ("oracle", "oracle_pick", "larger", "smaller", "centre", "centroid", "erase")
KEEP_ALL, REMOVE_ALL = -2, -1


def _winners(C, gi, sizes, cy, cx, my, mx, yy, xx):
    """Winner index per overlap pixel for every rule.  C: contenders (B x n bool); gi: GT object index per pixel
    (-1 = background / not a decoded object); sizes: predicted areas; (cy, cx) bbox centres; (my, mx) centroids;
    (yy, xx) pixel coordinates."""
    B, n = C.shape
    ar = np.arange(n)
    gt_cont = (gi >= 0) & C[np.maximum(gi, 0), ar]
    out = {}
    out["oracle"] = np.where(gt_cont, gi, REMOVE_ALL)
    out["oracle_pick"] = np.where(gt_cont, gi, KEEP_ALL)
    s = sizes[:, None].astype(np.float64)
    out["larger"] = np.argmax(np.where(C, s, -1.0), axis=0)
    out["smaller"] = np.argmin(np.where(C, s, np.inf), axis=0)
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)
    d = (yy[None, :] - cy[:, None].astype(np.float32)) ** 2 + (xx[None, :] - cx[:, None].astype(np.float32)) ** 2
    out["centre"] = np.argmin(np.where(C, d, np.inf), axis=0)
    d = (yy[None, :] - my[:, None].astype(np.float32)) ** 2 + (xx[None, :] - mx[:, None].astype(np.float32)) ** 2
    out["centroid"] = np.argmin(np.where(C, d, np.inf), axis=0)
    out["erase"] = np.full(n, REMOVE_ALL)
    return out, gt_cont


def process_scene(scene):
    meta = load_meta(DATA, scene)
    ref_cam = meta["ref_cam"]
    ref_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{ref_cam}.png"))
    objects = labelmap_objects(ref_lab)
    B = len(objects)
    h, w = ref_lab.shape
    bp = bound_pix(h, w)
    se = disk(bp)
    rdir = result_dir(DATA, scene, METHOD)
    lut = np.full(256, -1, np.int64)
    lut[np.array(objects)] = np.arange(B)
    views, rows, pair_overlaps = [], [], []
    halluc = {r: 0 for r in ("base",) + RULES}
    for cam in scored_views(meta):
        gt_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{cam}.png"))
        assert gt_lab.shape == (h, w)
        P = np.zeros((B, h, w), bool)
        present = []
        for i, obj in enumerate(objects):
            p = os.path.join(rdir, cam, f"{obj}.png")
            if os.path.isfile(p):
                P[i] = read_pred_mask(p)
                present.append(obj)
        Pf = P.reshape(B, -1)
        sizes = Pf.sum(1)
        cnt = P.sum(0, dtype=np.int16)
        idx = np.flatnonzero((cnt >= 2).ravel())
        n_ov = int(idx.size)
        C = Pf[:, idx]
        gl = gt_lab.ravel()[idx]
        gi = lut[gl]
        yy, xx = idx // w, idx % w
        cy = np.zeros(B)
        cx = np.zeros(B)
        my = np.zeros(B)
        mx = np.zeros(B)
        for i in range(B):
            if sizes[i] == 0:
                continue
            r_ = np.flatnonzero(P[i].any(1))
            c_ = np.flatnonzero(P[i].any(0))
            cy[i], cx[i] = (r_[0] + r_[-1]) / 2.0, (c_[0] + c_[-1]) / 2.0
            ys, xs = np.nonzero(P[i])
            my[i], mx[i] = ys.mean(), xs.mean()
        win, gt_cont = _winners(C, gi, sizes, cy, cx, my, mx, yy, xx)
        # overlapping object pairs
        Ci = C.astype(np.int32)
        M = Ci @ Ci.T
        iu = np.triu_indices(B, 1)
        pair_px = M[iu]
        n_pairs = int((pair_px > 0).sum())
        for a, b, px in zip(iu[0], iu[1], pair_px):
            if px > 0:
                both = C[a] & C[b]
                pair_overlaps.append(dict(scene=scene, cam=cam, obj_a=objects[a], obj_b=objects[b], px=int(px),
                                          gt_a=int((both & (gi == a)).sum()), gt_b=int((both & (gi == b)).sum()),
                                          gt_bg=int((both & (gl == 0)).sum()),
                                          size_a=int(sizes[a]), size_b=int(sizes[b])))
        union_px = int((cnt >= 1).sum())
        acc = {rule: int(((wv == gi) & gt_cont).sum()) for rule, wv in win.items()}
        views.append(dict(rule_correct_px=acc,scene=scene, cam=cam, n_obj=B, n_present=len(present), n_nonempty=int((sizes > 0).sum()),
                          n_pairs_overlapping=n_pairs, n_pairs_possible=B * (B - 1) // 2, overlap_px=n_ov,
                          union_px=union_px, image_px=h * w, max_multiplicity=int(cnt.max()) if n_ov else int(cnt.max()),
                          ov_gt_contender=int(gt_cont.sum()), ov_gt_bg=int((gl == 0).sum()),
                          ov_gt_other=int(((gl != 0) & ~gt_cont).sum()),
                          ov_mult3plus=int((cnt >= 3).sum())))
        # new masks per rule (only overlap columns change)
        newc = {}
        for rule, wv in win.items():
            Qc = np.zeros_like(C)
            keep = wv == KEEP_ALL
            Qc[:, keep] = C[:, keep]
            sel = np.flatnonzero(wv >= 0)
            Qc[wv[sel], sel] = True
            newc[rule] = Qc
        vd = view_dist(meta, cam)
        for i, obj in enumerate(objects):
            gt = gt_lab == obj
            gt_px = int(gt.sum())
            pred = P[i]
            if gt_px == 0:
                if sizes[i] > 0:
                    halluc["base"] += 1
                for rule in RULES:
                    q = Pf[i].copy()
                    q[idx] = newc[rule][i]
                    if q.any():
                        halluc[rule] += 1
                continue
            J, F = jf_pair(gt, pred, bp, se)
            row = dict(scene=scene, cam=cam, obj=obj, gt_px=gt_px, pred_px=int(sizes[i]), view_dist=vd,
                       size_bin=size_bin(gt_px), J=float(J), F=float(F), JF=float(0.5 * (J + F)),
                       ov_px=int(C[i].sum()), ov_gt_mine=int((C[i] & (gi == i)).sum()),
                       ov_gt_other_contender=int((C[i] & gt_cont & (gi != i)).sum()),
                       ov_gt_bg=int((C[i] & (gl == 0)).sum()),
                       ov_gt_noncontender=int((C[i] & (gl != 0) & ~gt_cont).sum()),
                       gt_px_in_overlap=int(((gi == i)).sum()),  # GT pixels of this object lying in any overlap
                       gt_px_lost_to_others=int(((gi == i) & ~C[i]).sum()),
                       fp_px=int((pred & ~gt).sum()),                       # predicted pixels the GT gives to another object
                       fp_in_overlap=int((C[i] & (gi != i)).sum()),         # ... of which also claimed by another mask
                       fn_px=int((gt & ~pred).sum()),                       # GT pixels the prediction misses
                       fn_claimed_by_other=int((gt & ~pred & (cnt >= 1)).sum()))  # ... of which another mask covers
            for rule in RULES:
                q = Pf[i].copy()
                q[idx] = newc[rule][i]
                Jr, Fr = jf_pair(gt, q.reshape(h, w), bp, se)
                row[f"J_{rule}"] = float(Jr)
                row[f"F_{rule}"] = float(Fr)
                row[f"JF_{rule}"] = float(0.5 * (Jr + Fr))
            rows.append(row)
    return dict(scene=scene, views=views, rows=rows, pair_overlaps=pair_overlaps, halluc=halluc, n_obj=B, hw=[h, w])


def _rows_for(rows, rule):
    """Baseline-shaped rows whose J/F/JF are the rule's values."""
    if rule is None:
        return [dict(scene=r["scene"], cam=r["cam"], obj=r["obj"], J=r["J"], F=r["F"], JF=r["JF"]) for r in rows]
    return [dict(scene=r["scene"], cam=r["cam"], obj=r["obj"], J=r[f"J_{rule}"], F=r[f"F_{rule}"], JF=r[f"JF_{rule}"]) for r in rows]


def _headline(rows, key="JF"):
    return statistics.mean(scene_means(rows, key).values())


def _mixed(rows, rule, subset):
    """Rule applied only to the pairs in `subset` (set of (scene, cam, obj)); others at baseline."""
    out = []
    for r in rows:
        k = (r["scene"], r["cam"], r["obj"])
        if k in subset:
            out.append(dict(scene=r["scene"], cam=r["cam"], obj=r["obj"], J=r[f"J_{rule}"], F=r[f"F_{rule}"], JF=r[f"JF_{rule}"]))
        else:
            out.append(dict(scene=r["scene"], cam=r["cam"], obj=r["obj"], J=r["J"], F=r["F"], JF=r["JF"]))
    return out


def _mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlap_oracle.json"))
    ap.add_argument("--procs", type=int, default=17)
    a = ap.parse_args()
    scenes = all_scenes(DATA)
    with Pool(a.procs) as pool:
        per_scene = {d["scene"]: d for d in pool.imap_unordered(process_scene, scenes)}
    rows = [r for sc in scenes for r in per_scene[sc]["rows"]]
    views = [v for sc in scenes for v in per_scene[sc]["views"]]
    pair_overlaps = [p for sc in scenes for p in per_scene[sc]["pair_overlaps"]]
    key = lambda r: (r["scene"], r["cam"], r["obj"])

    # ---- sanity: baseline rows reproduce the C0 keyframe score file
    kf = json.load(open(KF_C0))
    k0 = {(r["scene"], r["cam"], r["obj"]): r for v in kf["scenes"].values() for r in v["rows"]}
    assert set(k0) == {key(r) for r in rows}, "pair set differs from kf_geoset_pred_ft_rep_e8.json"
    max_dj = max(abs(k0[key(r)]["J"] - r["J"]) for r in rows)
    max_df = max(abs(k0[key(r)]["F"] - r["F"]) for r in rows)

    # ---- seedfail anatomy (kinds, twin, cost)
    sf = {(r["scene"], r["cam"], r["obj"]): r for r in json.load(open(SEEDFAIL))["rows"]}
    for r in rows:
        s = sf[key(r)]
        r["kind"] = s["kind"]
        r["twin"] = s["twin"]
        r["cost"] = s["cost"]
        r["fail"] = s["JF"] < 0.8
    good = {key(r) for r in rows if not r["fail"]}
    fail = {key(r) for r in rows if r["fail"]}
    spill14 = {key(r) for r in rows if r["fail"] and r["kind"].startswith("spill")}
    spill_all = {key(r) for r in rows if r["kind"].startswith("spill")}
    wrong = {key(r) for r in rows if r["kind"] == "wrong-object"}
    kinds = sorted({r["kind"] for r in rows if r["fail"]})
    has_ov = {key(r) for r in rows if r["ov_px"] > 0}

    base_rows = _rows_for(rows, None)
    base_head = {k: _headline(base_rows, k) for k in ("J", "F", "JF")}
    base_pair = {k: _mean(r[k] for r in base_rows) for k in ("J", "F", "JF")}

    res = dict(method=METHOD, n_pairs=len(rows), n_views=len(views), n_scenes=len(scenes),
               sanity=dict(max_abs_dJ_vs_kf=max_dj, max_abs_dF_vs_kf=max_df),
               baseline=dict(headline=base_head, pair_mean=base_pair, hallucinations=sum(per_scene[s]["halluc"]["base"] for s in scenes)),
               rules={}, views=views, pair_overlaps=sorted(pair_overlaps, key=lambda p: -p["px"]))

    # ---- overlap statistics
    tot = dict(views=len(views), views_with_overlap=sum(v["overlap_px"] > 0 for v in views),
               pairs_overlapping=sum(v["n_pairs_overlapping"] for v in views),
               pairs_possible=sum(v["n_pairs_possible"] for v in views),
               overlap_px=sum(v["overlap_px"] for v in views), union_px=sum(v["union_px"] for v in views),
               ov_gt_contender=sum(v["ov_gt_contender"] for v in views), ov_gt_bg=sum(v["ov_gt_bg"] for v in views),
               ov_gt_other=sum(v["ov_gt_other"] for v in views), ov_mult3plus=sum(v["ov_mult3plus"] for v in views),
               scored_pairs_with_overlap=len(has_ov),
               scored_pairs_with_overlap_good=len(has_ov & good), scored_pairs_with_overlap_fail=len(has_ov & fail),
               scored_pairs_with_overlap_spill14=len(has_ov & spill14),
               overlap_px_in_scored_masks=sum(r["ov_px"] for r in rows),
               overlap_px_in_good=sum(r["ov_px"] for r in rows if key(r) in good),
               overlap_px_in_fail=sum(r["ov_px"] for r in rows if key(r) in fail),
               overlap_px_in_spill14=sum(r["ov_px"] for r in rows if key(r) in spill14))
    acc_tot = {rule: sum(v["rule_correct_px"][rule] for v in views) for rule in RULES}
    tot["rule_correct_px"] = acc_tot
    tot["rule_pixel_accuracy_on_gt_contender_px"] = {rule: acc_tot[rule] / max(1, tot["ov_gt_contender"]) for rule in RULES}
    tot["rule_pixel_accuracy_on_all_overlap_px"] = {rule: acc_tot[rule] / max(1, tot["overlap_px"]) for rule in RULES}
    fpd = {}
    for name, S in dict(all=set(k0), good605=good, fail59=fail, spill14=spill14, spill_kind_all=spill_all, wrong_object=wrong).items():
        sel = [r for r in rows if key(r) in S]
        fp, fpo, fn, fno = (sum(r[k] for r in sel) for k in ("fp_px", "fp_in_overlap", "fn_px", "fn_claimed_by_other"))
        fpd[name] = dict(n=len(sel), fp_px=fp, fp_in_overlap=fpo, fp_in_overlap_frac=fpo / max(1, fp),
                         fn_px=fn, fn_claimed_by_other=fno, fn_claimed_frac=fno / max(1, fn),
                         pairs_fp_in_overlap_frac_ge_half=sum(r["fp_in_overlap"] >= 0.5 * r["fp_px"] for r in sel if r["fp_px"] > 0))
    res["fp_decomposition"] = fpd
    res["overlap_totals"] = tot
    res["overlap_by_scene"] = {sc: dict(n_obj=per_scene[sc]["n_obj"], hw=per_scene[sc]["hw"],
                                        views=len(per_scene[sc]["views"]),
                                        pairs_overlapping=sum(v["n_pairs_overlapping"] for v in per_scene[sc]["views"]),
                                        pairs_possible=sum(v["n_pairs_possible"] for v in per_scene[sc]["views"]),
                                        overlap_px=sum(v["overlap_px"] for v in per_scene[sc]["views"]),
                                        union_px=sum(v["union_px"] for v in per_scene[sc]["views"]),
                                        ov_gt_contender=sum(v["ov_gt_contender"] for v in per_scene[sc]["views"]),
                                        ov_gt_bg=sum(v["ov_gt_bg"] for v in per_scene[sc]["views"]),
                                        ov_gt_other=sum(v["ov_gt_other"] for v in per_scene[sc]["views"]))
                               for sc in scenes}

    # ---- per rule
    subsets = dict(all=set(k0), good605=good, fail59=fail, spill14=spill14, spill_kind_all=spill_all, wrong_object=wrong,
                   has_overlap=has_ov, twin=({key(r) for r in rows if r["twin"]}), not_twin={key(r) for r in rows if not r["twin"]},
                   **{f"kind_{k}": {key(r) for r in rows if r["fail"] and r["kind"] == k} for k in kinds})
    for rule in RULES:
        rr = _rows_for(rows, rule)
        head = {k: _headline(rr, k) for k in ("J", "F", "JF")}
        pm = {k: _mean(r[k] for r in rr) for k in ("J", "F", "JF")}
        boot = {k: paired_scene_bootstrap(base_rows, rr, key=k) for k in ("J", "F", "JF")}
        d = {key(r): r[f"JF_{rule}"] - r["JF"] for r in rows}
        dJ = {key(r): r[f"J_{rule}"] - r["J"] for r in rows}
        dF = {key(r): r[f"F_{rule}"] - r["F"] for r in rows}
        worst = min(boot["JF"]["per_scene"].items(), key=lambda x: x[1])
        best = max(boot["JF"]["per_scene"].items(), key=lambda x: x[1])
        sub = {}
        for name, S in subsets.items():
            sel = [k for k in d if k in S]
            sub[name] = dict(n=len(sel), dJF_pair_mean=_mean(d[k] for k in sel), sum_dJF=float(sum(d[k] for k in sel)),
                             dJ_pair_mean=_mean(dJ[k] for k in sel), dF_pair_mean=_mean(dF[k] for k in sel),
                             up=int(sum(d[k] > 0.01 for k in sel)), down=int(sum(d[k] < -0.01 for k in sel)),
                             nonzero=int(sum(d[k] != 0 for k in sel)), min_d=float(min((d[k] for k in sel), default=float('nan'))), max_d=float(max((d[k] for k in sel), default=float('nan'))),
                             headline_gain_restricted=_headline(_mixed(rows, rule, S)) - base_head["JF"],
                             cost_covered=float(sum(sf[k]["cost"] for k in sel)))
        res["rules"][rule] = dict(headline=head, pair_mean=pm,
                                  delta_headline={k: head[k] - base_head[k] for k in head},
                                  delta_pair_mean={k: pm[k] - base_pair[k] for k in pm},
                                  bootstrap={k: dict(delta=boot[k]["delta"], ci=boot[k]["ci"]) for k in boot},
                                  per_scene_dJF=boot["JF"]["per_scene"], worst_scene=worst, best_scene=best,
                                  G1_ci_lower_gt_0=boot["JF"]["ci"][0] > 0, G2_no_scene_below_m0p01=worst[1] >= -0.01,
                                  hallucinations=sum(per_scene[s]["halluc"][rule] for s in scenes), subsets=sub)
    res["rows"] = rows
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)

    # ---- print
    print(f"{METHOD}: {len(rows)} scored pairs, {len(views)} scored views, {len(scenes)} scenes; sanity max|dJ| {max_dj:.2e} max|dF| {max_df:.2e}")
    print(f"baseline keyframe J&F headline {base_head['JF']:.5f} (J {base_head['J']:.5f} F {base_head['F']:.5f}); pair mean {base_pair['JF']:.5f}; hallucinations {res['baseline']['hallucinations']}")
    print("\n== overlap totals ==")
    for k, v in tot.items():
        print(f"  {k}: {v}")
    print(f"  overlap_px / union_px = {tot['overlap_px'] / tot['union_px']:.5f}; GT split of overlap px: contender {tot['ov_gt_contender'] / max(1, tot['overlap_px']):.3f} bg {tot['ov_gt_bg'] / max(1, tot['overlap_px']):.3f} other {tot['ov_gt_other'] / max(1, tot['overlap_px']):.3f}")
    print("  rule pixel accuracy on overlap px whose GT object is a contender: " + ", ".join(f"{r} {tot['rule_pixel_accuracy_on_gt_contender_px'][r]:.4f}" for r in RULES))
    print("\n== false-positive / false-negative decomposition (pixels summed over the subset) ==")
    for name, v in fpd.items():
        print(f"  {name:16s} n {v['n']:3d} FP {v['fp_px']:9d} of which in an overlap {v['fp_in_overlap']:8d} ({v['fp_in_overlap_frac']:.4f}); pairs with >=50% of FP in overlap {v['pairs_fp_in_overlap_frac_ge_half']}; "
              f"FN {v['fn_px']:9d} of which covered by another mask {v['fn_claimed_by_other']:8d} ({v['fn_claimed_frac']:.4f})")
    print("\n== per view ==")
    print("scene cam n_obj nonempty pairs_ov/possible overlap_px union_px ov/union gt_contender gt_bg gt_other mult>=3")
    for v in views:
        print(f"{v['scene']} {v['cam']} {v['n_obj']} {v['n_nonempty']} {v['n_pairs_overlapping']}/{v['n_pairs_possible']} {v['overlap_px']} {v['union_px']} "
              f"{v['overlap_px'] / max(1, v['union_px']):.4f} {v['ov_gt_contender']} {v['ov_gt_bg']} {v['ov_gt_other']} {v['ov_mult3plus']}")
    print("\n== biggest overlapping object pairs (top 25) ==")
    for p in res["pair_overlaps"][:25]:
        print(f"{p['scene']} {p['cam']} {p['obj_a']}&{p['obj_b']} px {p['px']} (gt: a {p['gt_a']} b {p['gt_b']} bg {p['gt_bg']}) sizes {p['size_a']}/{p['size_b']}")
    print("\n== rules (delta vs baseline) ==")
    print("rule | headline dJF [CI] | dJ | dF | pair-mean dJF | worst scene | up/down(+-0.01)/nonzero | halluc")
    for rule in RULES:
        R = res["rules"][rule]
        b = R["bootstrap"]["JF"]
        s = R["subsets"]["all"]
        print(f"{rule:12s} | {R['delta_headline']['JF']:+.5f} [{b['ci'][0]:+.5f}, {b['ci'][1]:+.5f}] | {R['delta_headline']['J']:+.5f} | {R['delta_headline']['F']:+.5f} | "
              f"{R['delta_pair_mean']['JF']:+.5f} | {R['worst_scene'][0]} {R['worst_scene'][1]:+.5f} | {s['up']}/{s['down']}/{s['nonzero']} | {R['hallucinations']}")
    print("\n== subset gains (rule applied to the subset only; headline = scene-mean J&F gain; pair = mean dJF over the subset) ==")
    names = ["good605", "fail59", "spill14", "spill_kind_all", "wrong_object", "has_overlap", "twin"] + [f"kind_{k}" for k in kinds]
    for rule in RULES:
        print(f"-- {rule}")
        for name in names:
            s = res["rules"][rule]["subsets"][name]
            print(f"   {name:28s} n {s['n']:3d} headline {s['headline_gain_restricted']:+.5f} pair {s['dJF_pair_mean']:+.5f} sum {s['sum_dJF']:+.4f} up/down/nz {s['up']}/{s['down']}/{s['nonzero']} min {s['min_d']:+.4f} max {s['max_d']:+.4f}")
    print("\n== failing spill / wrong-object pairs ==")
    print("scene cam obj kind JF pred_px fp_px fp_in_ov fn_px fn_by_other ov_px ov_gt_mine ov_gt_other_cont ov_gt_noncont | " + " ".join(RULES))
    for r in rows:
        k = key(r)
        if k in spill14 or k in wrong:
            print(f"{r['scene']} {r['cam']} {r['obj']} {r['kind']} {r['JF']:.3f} {r['pred_px']} {r['fp_px']} {r['fp_in_overlap']} {r['fn_px']} {r['fn_claimed_by_other']} {r['ov_px']} {r['ov_gt_mine']} {r['ov_gt_other_contender']} {r['ov_gt_noncontender']} | "
                  + " ".join(f"{r[f'JF_{ru}'] - r['JF']:+.3f}" for ru in RULES))
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
