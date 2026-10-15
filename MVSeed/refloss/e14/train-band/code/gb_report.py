#!/usr/bin/env python3
"""S1-E14 train-band: summary of grad_balance.py outputs (host).  -> gb_summary.json
Per arm: W_grad = rms ||g_base|| / rms ||g_band|| at the 288 low_res (the W that makes the two terms' gradients equal in
norm), its scene-cluster bootstrap CI (17 scenes, 2 000 resamples, seed 0), median per-object ratio, mean cosine; loss means;
band composition (FN / FP / confidently-wrong fractions of band pixels, band BCE share from confidently-wrong pixels);
by GT size; absent-predicted objects with non-empty GT (the loss spikes that carry no gradient).
    stage1/.venv/bin/python gb_report.py gb_<arm>.json ... --out gb_summary.json
"""
import argparse
import json
import os

import numpy as np

BINS = [(1, 2000), (2000, 10000), (10000, 100000), (100000, 10 ** 9)]


def wgrad(objs):
    nb = np.array([o["nb_low"] for o in objs]); nn = np.array([o["nn_low"] for o in objs])
    return float(np.sqrt((nb ** 2).mean()) / np.sqrt((nn ** 2).mean()))


def summarize(objs, rng):
    wb = [o for o in objs if o["present"] and o.get("nn_low")]
    pres_empty = [o for o in objs if o["present"] and o["gt_px"] == 0]
    absent_gt = [o for o in objs if not o["present"] and o["gt_px"] > 0]
    scenes = sorted({o["scene"] for o in wb})
    by = {s: [o for o in wb if o["scene"] == s] for s in scenes}
    boots = []
    for _ in range(2000):
        pick = rng.choice(len(scenes), len(scenes), replace=True)
        boots.append(wgrad([o for i in pick for o in by[scenes[i]]]))
    out = dict(n_objects=len(objs), n_present_with_band=len(wb), n_present_gt_empty=len(pres_empty), n_absent_gt_nonempty=len(absent_gt),
               W_grad=wgrad(wb), W_grad_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
               W_grad_median_obj=float(np.median([o["nb_low"] / o["nn_low"] for o in wb])),
               cos_low_mean=float(np.mean([o["cos_low"] for o in wb])), cos_full_mean=float(np.mean([o["cos_full"] for o in wb])),
               loss_focal20=float(np.mean([o["focal20"] for o in wb])), loss_dice=float(np.mean([o["dice"] for o in wb])),
               loss_band=float(np.mean([o["band"] for o in wb])), loss_band_median=float(np.median([o["band"] for o in wb])),
               W_loss=float(np.mean([o["focal20"] + o["dice"] for o in wb]) / np.mean([o["band"] for o in wb])),
               band_wrong_frac=float(np.mean([o["band_wrong_frac"] for o in wb])),
               band_fn_frac=float(np.mean([o["band_fn_frac"] for o in wb])), band_fp_frac=float(np.mean([o["band_fp_frac"] for o in wb])),
               band_confwrong_frac=float(np.mean([o["band_confwrong_frac"] for o in wb])),
               band_bce_from_confwrong=float(np.mean([o["band_bce_from_confwrong"] for o in wb])),
               absent_gt_band_mean=float(np.mean([o["band"] for o in absent_gt])) if absent_gt else None,
               absent_gt_focal20_mean=float(np.mean([o["focal20"] for o in absent_gt])) if absent_gt else None,
               by_size={})
    for lo, hi in BINS:
        s = [o for o in wb if lo <= o["gt_px"] < hi]
        if s:
            out["by_size"][f"{lo}-{hi}"] = dict(n=len(s), W_grad=wgrad(s), cos=float(np.mean([o["cos_low"] for o in s])),
                                               band=float(np.mean([o["band"] for o in s])),
                                               base=float(np.mean([o["focal20"] + o["dice"] for o in s])),
                                               fn_frac=float(np.mean([o["band_fn_frac"] for o in s])),
                                               fp_frac=float(np.mean([o["band_fp_frac"] for o in s])))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()
res = {}
for f in a.files:
    name = os.path.basename(f)[3:-5]
    res[name] = summarize(json.load(open(f))["objects"], np.random.default_rng(0))
    r = res[name]
    print(f"== {name}: W_grad {r['W_grad']:.4f} [{r['W_grad_ci'][0]:.4f},{r['W_grad_ci'][1]:.4f}] median-obj {r['W_grad_median_obj']:.4f} cos {r['cos_low_mean']:.3f}"
          f" | loss 20focal {r['loss_focal20']:.4f} dice {r['loss_dice']:.4f} band {r['loss_band']:.4f} (median {r['loss_band_median']:.4f}) W_loss {r['W_loss']:.4f}")
    print(f"   band px: wrong {r['band_wrong_frac']:.4f} (FN {r['band_fn_frac']:.4f} FP {r['band_fp_frac']:.4f}) conf-wrong {r['band_confwrong_frac']:.4f}, "
          f"band BCE from conf-wrong {r['band_bce_from_confwrong']:.3f}; absent-predicted with GT n={r['n_absent_gt_nonempty']} band {r['absent_gt_band_mean']} 20focal {r['absent_gt_focal20_mean']}")
    for b, v in r["by_size"].items():
        print(f"   {b:>16s} n {v['n']:4d} W_grad {v['W_grad']:.4f} cos {v['cos']:.3f} band {v['band']:.3f} base {v['base']:.3f} FN {v['fn_frac']:.4f} FP {v['fp_frac']:.4f}")
json.dump(res, open(a.out, "w"), indent=1)
