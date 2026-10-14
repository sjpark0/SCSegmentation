#!/usr/bin/env python3
"""S1-E10 judge (docs/stage1-E10-prereg.md sections 3-7), keyed by RESULT FOLDER (every arm shares the arm name
geoset_pred_ft with the baseline, so e9_judge's arm-name folders cannot be reused).

  --stage 1   keyframe scores (stage1/mvseg2d.evaluate JSONs in MVSeed/runs/e10/kf_<folder>.json): C1, C2, C3 against
              C0, scene-cluster bootstrap on J, F, J&F (mvseg2d.paired, 10 000, seed 0), gate G1/G2 (C1, C2 only),
              fitted/skipped, good/failing, size, twin and object-type splits, predictions P1-P7, watch values
              Z1 (counters), Z2 (round 1 untouched), Z5 (band containment, key sets), Z7 (cost).
              Run with stage1/.venv (numpy, cv2):  stage1/.venv/bin/python MVSeed/e10_judge.py --stage 1
  --stage 2   MUVOD basic/complete of the passing arms' in-process headline runs (Data/MVSeg/jf_e10.json) against
              the current headline, the E8 adoption rule, selection (tie -> C1), P8-P9, Z3/Z3b.  Also under the venv.
Both write MVSeed/runs/e10/e10_judge.json (stage 2 merges into stage 1's file).
"""
import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(REPO, "stage1")
OUT_DIR = os.path.join(REPO, "MVSeed", "runs", "e10")
OUT = os.path.join(OUT_DIR, "e10_judge.json")
D0 = "geoset_pred_ft_rep_e8"
D0_R1 = "geoset_pred_ft_r1_e8"
ARMS = [("C1", "geoset_pred_ft_rep_z2_e10", "geoset_rep_z2", "geoset_pred_ft_r1_z2_e10"),
        ("C2", "geoset_pred_ft_rep_z2r_e10", "geoset_rep_z2r", "geoset_pred_ft_r1_z2r_e10"),
        ("C3", "geoset_pred_ft_rep_gf_e10", None, None)]
GATED = ("C1", "C2")                           # the adoption set; C3 is a control
HEADLINE = "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs"
DEFENCE, TIE = -1.0, 0.05
SAFETY = -0.01                                 # G2: no scene below C0 - 0.01 (keyframe J&F)
SCENES = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "CBABasketball", "Carpark", "CoffeeMartini",
          "Dog", "Fencing", "FlameSteak", "Frog", "MATF", "MartialArts", "Painter", "PoznanStreet", "Welder")


def _load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _result(scene, folder):
    return os.path.join(S1, "Data", scene, "Result", folder)


def _mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


# ---------------------------------------------------------------- stage 1
def _pngs(d, skip=()):
    out = {}
    if not os.path.isdir(d):
        return out
    for cam in sorted(os.listdir(d)):
        if cam in skip or not os.path.isdir(os.path.join(d, cam)):
            continue
        for f in sorted(os.listdir(os.path.join(d, cam))):
            if f.endswith(".png"):
                out[(cam, f)] = os.path.join(d, cam, f)
    return out


def _bytes_equal(a, b):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def z2_round1(r1_folder, rep_folder):
    """Round 1 untouched: every PNG of Result/<r1_folder> byte-identical to Result/geoset_pred_ft_r1_e8 and the
    rep folder's round1_conf.json identical to C0's (scene counts)."""
    out = dict(scenes_ok=0, png_same=0, png_diff=0, key_mismatch=0, conf_same=0, missing=[])
    for sc in SCENES:
        a, b = _pngs(_result(sc, r1_folder)), _pngs(_result(sc, D0_R1))
        if not a:
            out["missing"].append(sc); continue
        ca, cb = os.path.join(_result(sc, rep_folder), "round1_conf.json"), os.path.join(_result(sc, D0), "round1_conf.json")
        if os.path.isfile(ca) and os.path.isfile(cb) and _bytes_equal(ca, cb):
            out["conf_same"] += 1
        if set(a) != set(b):
            out["key_mismatch"] += 1; continue
        same = sum(_bytes_equal(a[k], b[k]) for k in a)
        out["png_same"] += same; out["png_diff"] += len(a) - same
        if same == len(a):
            out["scenes_ok"] += 1
    out["ok"] = out["scenes_ok"] == 17 and out["conf_same"] == 17 and out["png_diff"] == 0 and out["key_mismatch"] == 0 and not out["missing"]
    return out


def z5_band(folder, ref_cams, r_of):
    """Band containment and key-set invariance against C0 (prereg Z5): per scene, the (cam, obj) PNG set equals C0's
    and every pixel that differs lies inside band_r(C0 mask)."""
    import cv2
    import numpy as np
    sys.path.insert(0, S1)
    from geomem import zoom
    out = dict(scenes_checked=0, key_mismatch=[], changed_pngs=0, outside_band_px=0, outside_band_pngs=0, missing=[])
    for sc in SCENES:
        a, b = _pngs(_result(sc, folder), skip=(ref_cams[sc],)), _pngs(_result(sc, D0), skip=(ref_cams[sc],))
        if not a:
            out["missing"].append(sc); continue
        out["scenes_checked"] += 1
        if set(a) != set(b):
            out["key_mismatch"].append(sc); continue
        r = r_of[sc]
        for k in a:
            if _bytes_equal(a[k], b[k]):
                continue
            out["changed_pngs"] += 1
            ma = cv2.imread(a[k], 0) > 127
            mb = cv2.imread(b[k], 0) > 127
            bad = int(((ma != mb) & ~zoom.band_mask(mb, r)).sum())
            if bad:
                out["outside_band_px"] += bad; out["outside_band_pngs"] += 1
    out["ok"] = not out["key_mismatch"] and out["outside_band_pngs"] == 0 and not out["missing"]
    return out


def _kind(r):
    """Failure type of a seed (MVSeed/refloss/seedcat.py rules, docs/stage1-seedfail.md section 2)."""
    if r["seed_px"] == 0 or (r["recall"] or 0) < 0.05:
        return "miss" if r["seed_px"] == 0 or r["gt_px"] < 50 else "wrong-object"
    if (r["precision"] or 0) < 0.3:
        return "wrong-object"
    if r["frac_other"] >= 0.2:
        return "spill-onto-other" + ("(twin)" if r["twin"] else "")
    if r["recall"] < 0.85 and r["precision"] >= 0.85:
        return "under-cover" + ("(beyond c_ini)" if r["vis_ratio"] > 1.2 else "")
    return "boundary/mixed"


def stage1():
    sys.path.insert(0, S1)
    from mvseg2d.paired import paired_scene_bootstrap, scene_means
    from mvseg2d.report import flatten
    from geomem import zoom

    def rows_of(name):
        rs = flatten(_load_json(os.path.join(OUT_DIR, f"kf_{name}.json")))
        for r in rs:
            r["JF"] = 0.5 * (r["J"] + r["F"])
        return rs

    def hallu(name):
        return sum(s["hallucinations"] for s in _load_json(os.path.join(OUT_DIR, f"kf_{name}.json"))["scenes"].values() if s["status"] == "ok")

    def summary(rs):
        m = {k: scene_means(rs, k) for k in ("J", "F", "JF")}
        return {k: statistics.mean(v.values()) for k, v in m.items()}

    def run_cfg(folder, sc):
        p = os.path.join(_result(sc, folder), "RUN.json")
        return _load_json(p).get("config", {}) if os.path.isfile(p) else {}

    key = lambda r: (r["scene"], r["cam"], r["obj"])
    r0 = rows_of(D0); k0 = {key(r): r for r in r0}
    good = {k for k, r in k0.items() if r["JF"] >= 0.8}
    ref_cams = {sc: _load_json(os.path.join(S1, "Data", sc, "meta.json"))["ref_cam"] for sc in SCENES}
    r_of = {sc: zoom.band_px(tuple(_load_json(os.path.join(S1, "Data", sc, "meta.json"))["image_size"])) for sc in SCENES}
    # seed anatomy of C0 (twin flag, object type, +-2 px band oracle) for the splits and P3
    sf0 = {(r["scene"], r["cam"], r["obj"]): r for r in _load_json(os.path.join(REPO, "MVSeed", "runs", "seedfail.json"))["rows"]}
    # fitted / skipped from the crop-zoom records (C1, else C2); a pair is fitted when its verdict is not skip_big
    fitted = None
    for rid, folder, tag, r1 in ARMS:
        if rid not in GATED:
            continue
        recs = {}
        ok = True
        for sc in SCENES:
            p = os.path.join(_result(sc, folder), "zoom_conf.json")
            if not os.path.isfile(p):
                ok = False; break
            for cam, objs in _load_json(p).items():
                for o, rec in objs.items():
                    recs[(sc, cam, int(o))] = rec
        if ok:
            fitted = {k for k in k0 if k in recs and recs[k]["verdict"] != "skip_big"}
            break
    res = {"C0": dict(name=D0, n=len(r0), **summary(r0), hallucinations=hallu(D0), fails=sum(r["J"] < 0.5 for r in r0),
                      good_seeds=len(good), fitted=None if fitted is None else len(fitted))}
    passing = []
    for rid, folder, tag, r1_folder in ARMS:
        p = os.path.join(OUT_DIR, f"kf_{folder}.json")
        if not os.path.isfile(p):
            print(f"{rid}: {p} missing (not decoded / scored yet)"); continue
        rs = rows_of(folder); kk = {key(r): r for r in rs}
        if set(kk) != set(k0):
            st_ = {sc: v["status"] for sc, v in _load_json(p)["scenes"].items() if v["status"] != "ok"}
            raise SystemExit(f"{rid}: pair set differs from C0 ({len(kk)} vs {len(k0)}); scenes not ok: {st_} -- fix the decode first")
        st = {k: paired_scene_bootstrap(r0, rs, key=k) for k in ("J", "F", "JF")}
        worst = min(st["JF"]["per_scene"].items(), key=lambda x: x[1])
        gate = dict(G1_mean_gt_0=st["JF"]["delta"] > 0, G1_ci_lower_gt_0=st["JF"]["ci"][0] > 0,
                    G2_no_scene_below=worst[1] >= SAFETY, worst_scene=worst[0], worst_delta=worst[1])
        gate["pass"] = gate["G1_mean_gt_0"] and gate["G1_ci_lower_gt_0"] and gate["G2_no_scene_below"]
        d = {k: kk[k]["JF"] - k0[k]["JF"] for k in k0}

        def split(sel):
            sel = list(sel)
            return dict(n=len(sel), dJF=_mean(d[k] for k in sel), dJ=_mean(kk[k]["J"] - k0[k]["J"] for k in sel),
                        dF=_mean(kk[k]["F"] - k0[k]["F"] for k in sel),
                        up=sum(d[k] > 0.01 for k in sel), down=sum(d[k] < -0.01 for k in sel), nonzero=sum(d[k] != 0 for k in sel))
        by_size = {b: split(k for k, r in k0.items() if r["size_bin"] == b) for b in ("<2000", "2000-10000", ">=10000")}
        by_size4 = {b: split(k for k, r in k0.items() if lo <= r["gt_px"] < hi) for b, lo, hi in
                    (("<2k", 0, 2000), ("2k-10k", 2000, 10000), ("10k-100k", 10000, 100000), (">=100k", 100000, 10 ** 12))}
        splits = dict(good=split(good), fail=split(k for k in k0 if k not in good),
                      fitted=None if fitted is None else split(fitted), skipped=None if fitted is None else split(k for k in k0 if k not in fitted),
                      fitted_good=None if fitted is None else split(fitted & good),
                      twin=split(k for k in k0 if sf0.get(k, {}).get("twin")), not_twin=split(k for k in k0 if not sf0.get(k, {}).get("twin")),
                      fitted_twin=None if fitted is None else split(k for k in fitted if sf0.get(k, {}).get("twin")),
                      **{f"vd_{b}": split(k for k, r in k0.items() if lo <= r["view_dist"] <= hi) for b, lo, hi in
                         (("<=1", 0, 1), ("2", 2, 2), ("3-4", 3, 4), (">=5", 5, 10 ** 6))},
                      **{f"otype_{t}": split(k for k, r in k0.items() if r["type"] == t) for t in ("dynamic", "static", "env")},
                      **{f"rig_{t}": split(k for k, r in k0.items() if r["rig"] == t) for t in ("pinhole", "hemisphere")})
        # P3 proxy: band-oracle gap (oracle_r2 - J&F) on the fitted good seeds, C0 (seedfail.json) vs this arm (sf_<folder>.json)
        sfp = os.path.join(OUT_DIR, f"sf_{folder}.json")
        band = kinds = None
        if os.path.isfile(sfp):
            sfa = {(r["scene"], r["cam"], r["obj"]): r for r in _load_json(sfp)}
            kinds = dict(C0={}, arm={})
            for k in k0:
                if k in sf0 and sf0[k]["JF"] < 0.8:
                    kinds["C0"][sf0[k].get("kind") or _kind(sf0[k])] = kinds["C0"].get(sf0[k].get("kind") or _kind(sf0[k]), 0) + 1
                if k in sfa and sfa[k]["JF"] < 0.8:
                    kinds["arm"][_kind(sfa[k])] = kinds["arm"].get(_kind(sfa[k]), 0) + 1
        if os.path.isfile(sfp) and fitted is not None:
            sel = [k for k in fitted & good if k in sfa and k in sf0]
            band = dict(n=len(sel), gap_C0=_mean(sf0[k]["oracle_r2"] - sf0[k]["JF"] for k in sel), gap=_mean(sfa[k]["oracle_r2"] - sfa[k]["JF"] for k in sel),
                        recall_C0=_mean(sf0[k]["recall"] for k in sel), recall=_mean(sfa[k]["recall"] for k in sel),
                        precision_C0=_mean(sf0[k]["precision"] for k in sel), precision=_mean(sfa[k]["precision"] for k in sel))
        # Z1 counters, Z7 cost (C1/C2: RUN.json of the zoomed run vs C0's masks_written)
        z1 = z7 = None
        if rid in GATED:
            z1 = dict(scenes=0, candidates_eq_masks_written=0, sums_ok=0, run=0, skipped=0, accepted=0, kept_absent=0, kept_iou=0, kept_empty=0, changed=0, px_added=0, px_removed=0)
            z7 = dict(seconds=0.0, zoom_seconds=0.0, gpu_mb_max=0, gpu_mb_max_C0=0)
            for sc in SCENES:
                c, c0 = run_cfg(folder, sc), run_cfg(D0, sc)
                if not c:
                    continue
                z1["scenes"] += 1
                z1["candidates_eq_masks_written"] += int(c.get("n_zoom_candidates") == c0.get("masks_written"))
                z1["sums_ok"] += int(c.get("n_zoom_run", -1) + c.get("n_zoom_skipped_large", -1) == c.get("n_zoom_candidates") and
                                     c.get("n_zoom_accepted", 0) + c.get("n_zoom_kept_absent", 0) + c.get("n_zoom_kept_iou", 0) + c.get("n_zoom_kept_empty", 0) == c.get("n_zoom_run"))
                for k2 in ("run", "accepted", "kept_absent", "kept_iou", "kept_empty", "changed"):
                    z1[k2] += int(c.get(f"n_zoom_{k2}", 0))
                z1["skipped"] += int(c.get("n_zoom_skipped_large", 0))
                z1["px_added"] += int(c.get("zoom_px_added", 0)); z1["px_removed"] += int(c.get("zoom_px_removed", 0))
                run = _load_json(os.path.join(_result(sc, folder), "RUN.json")); run0 = _load_json(os.path.join(_result(sc, D0), "RUN.json"))
                z7["seconds"] += float(run.get("seconds", 0)); z7["zoom_seconds"] += float(c.get("zoom_seconds", 0))
                z7["gpu_mb_max"] = max(z7["gpu_mb_max"], float(run.get("gpu_mb", 0) or 0)); z7["gpu_mb_max_C0"] = max(z7["gpu_mb_max_C0"], float(run0.get("gpu_mb", 0) or 0))
            z1["kept_iou_rate"] = z1["kept_iou"] / z1["run"] if z1["run"] else None
            z1["ok"] = z1["scenes"] == 17 and z1["candidates_eq_masks_written"] == 17 and z1["sums_ok"] == 17
            z7["per_crop"] = z7["zoom_seconds"] / z1["run"] if z1["run"] else None
            z7["ok"] = z1["scenes"] == 17 and z7["seconds"] <= 45 * 60 and z7["gpu_mb_max"] <= z7["gpu_mb_max_C0"] + 1500
        z2 = z2_round1(r1_folder, folder) if r1_folder else None
        z5 = z5_band(folder, ref_cams, r_of)
        if rid in GATED and fitted is not None:
            z5["skipped_delta_zero"] = splits["skipped"]["nonzero"] == 0
            z5["ok"] = z5["ok"] and z5["skipped_delta_zero"]
        r = dict(name=folder, tag=tag, **summary(rs), hallucinations=hallu(folder), fails=sum(x["J"] < 0.5 for x in rs),
                 paired={k: dict(delta=v["delta"], ci=list(v["ci"]), per_scene=v["per_scene"]) for k, v in st.items()},
                 gate=gate, splits=splits, by_size=by_size, by_size4=by_size4, band_proxy=band, failure_kinds=kinds, Z1=z1, Z2=z2, Z5=z5, Z7=z7)
        res[rid] = r
        if rid in GATED and gate["pass"]:
            passing.append(tag)
        print(f"{rid} {folder}: J {r['J']:.4f} F {r['F']:.4f} J&F {r['JF']:.4f} (C0 {res['C0']['JF']:.4f})  "
              f"ΔJ&F {st['JF']['delta']:+.4f} CI [{st['JF']['ci'][0]:+.4f}, {st['JF']['ci'][1]:+.4f}]  ΔJ {st['J']['delta']:+.4f} [{st['J']['ci'][0]:+.4f}, {st['J']['ci'][1]:+.4f}]"
              f"  ΔF {st['F']['delta']:+.4f} [{st['F']['ci'][0]:+.4f}, {st['F']['ci'][1]:+.4f}]  worst {worst[0]} {worst[1]:+.4f}  fails {res['C0']['fails']}→{r['fails']}  halluc {res['C0']['hallucinations']}→{r['hallucinations']}"
              f"  gate {'PASS' if gate['pass'] else 'FAIL'}" + ("" if rid in GATED else " (control, not gated)"))
        print(f"   good {splits['good']['dJF']:+.4f} (n={splits['good']['n']}, ↑{splits['good']['up']} ↓{splits['good']['down']})  fail {splits['fail']['dJF']:+.4f}"
              + ("" if fitted is None else f"  | fitted {splits['fitted']['dJF']:+.4f} (n={splits['fitted']['n']}, nonzero {splits['fitted']['nonzero']})  skipped {splits['skipped']['dJF']:+.4f} (n={splits['skipped']['n']}, nonzero {splits['skipped']['nonzero']})")
              + f"  | size {{{', '.join(f'{b}: {v['dJF']:+.4f}' for b, v in by_size4.items())}}}  twin {splits['twin']['dJF']:+.4f} fitted-twin {(splits['fitted_twin'] or {}).get('dJF')}  vd {{{', '.join(f'{b}: {splits[f'vd_{b}']['dJF']:+.4f}' for b in ('<=1', '2', '3-4', '>=5'))}}}"
              + ("" if band is None else f"  | band gap {band['gap_C0']:.4f}→{band['gap']:.4f} recall {band['recall_C0']:.4f}→{band['recall']:.4f} prec {band['precision_C0']:.4f}→{band['precision']:.4f}"))
        print(f"   kinds {kinds}\n   Z1 {z1}\n   Z2 {z2}\n   Z5 {z5}\n   Z7 {z7}")
    # predictions (section 4)
    pr = {}
    if "C1" in res:
        c1 = res["C1"]
        pr["P1"] = dict(delta=c1["paired"]["JF"]["delta"], ok=0.002 <= c1["paired"]["JF"]["delta"] <= 0.008 and
                        c1["paired"]["J"]["delta"] > c1["paired"]["F"]["delta"] and abs(c1["paired"]["F"]["delta"]) <= 0.002)
        s = c1["by_size"]
        pr["P2"] = dict(small=s["<2000"]["dJF"], mid=s["2000-10000"]["dJF"], large=s[">=10000"]["dJF"],
                        ok=s["<2000"]["dJF"] >= 0.010 and s["2000-10000"]["dJF"] >= 0.005 and -0.002 <= s[">=10000"]["dJF"] <= 0.005)
        b = c1["band_proxy"]
        pr["P3"] = None if b is None else dict(gap_shrink=b["gap_C0"] - b["gap"], ok=(b["gap_C0"] - b["gap"]) >= 0.003 and
                                                not ((b["recall"] - b["recall_C0"]) * (b["precision"] - b["precision_C0"]) < 0 and
                                                     abs(b["recall"] - b["recall_C0"]) > 0.01 and abs(b["precision"] - b["precision_C0"]) > 0.01))
    for rid in GATED:
        if rid in res:
            pr[f"P4_{rid}"] = dict(fails_delta=res[rid]["fails"] - res["C0"]["fails"], halluc_delta=res[rid]["hallucinations"] - res["C0"]["hallucinations"],
                                   ok=res[rid]["fails"] - res["C0"]["fails"] <= 1)
            tw = res[rid]["splits"]["fitted_twin"] or res[rid]["splits"]["twin"]
            pr[f"P6_{rid}"] = dict(worst=res[rid]["gate"]["worst_delta"], fitted_twin=tw["dJF"], n_twin=tw["n"],
                                   ok=res[rid]["gate"]["G2_no_scene_below"] and tw["dJF"] >= -0.005)
            for z in ("Z1", "Z2", "Z5", "Z7"):
                v = res[rid][z]
                pr[f"{z}_{rid}"] = bool(v and v.get("ok"))
    if "C1" in res and "C2" in res:
        # the remap path must be live: C1 and C2 differ somewhere (a silent no-op would make them byte-identical)
        n_diff = n_all = 0
        for sc in SCENES:
            a, b = _pngs(_result(sc, res["C1"]["name"])), _pngs(_result(sc, res["C2"]["name"]))
            for k in set(a) & set(b):
                n_all += 1; n_diff += int(not _bytes_equal(a[k], b[k]))
        pr["Z6b_remap_live"] = dict(pngs_differ=n_diff, pngs=n_all, ok=n_diff > 0)
    if "C1" in res and "C2" in res:
        dd = res["C2"]["paired"]["JF"]["delta"] - res["C1"]["paired"]["JF"]["delta"]
        pr["P5"] = dict(C2_minus_C1=dd, ok=0 <= dd <= 0.004)
    if "C3" in res:
        d3 = res["C3"]["paired"]["JF"]["delta"]
        pr["P7"] = dict(delta=d3, ok=-0.004 <= d3 <= 0.003 and ("C1" not in res or res["C1"]["paired"]["JF"]["delta"] - d3 >= 0.001))
        if "C1" in res and res["C3"]["splits"]["fitted"] is not None:      # same-population sensitivity: C3 refines every pair, C1 only the fitted ones
            pr["P7"]["fitted_C3"] = res["C3"]["splits"]["fitted"]["dJF"]; pr["P7"]["fitted_C1"] = res["C1"]["splits"]["fitted"]["dJF"]
            pr["P7"]["ok_fitted"] = res["C1"]["splits"]["fitted"]["dJF"] - res["C3"]["splits"]["fitted"]["dJF"] >= 0.001
        pr["Z5_C3"] = bool(res["C3"]["Z5"]["ok"])
    out = _load_json(OUT) if os.path.isfile(OUT) else {}
    out.update({"prereg": "docs/stage1-E10-prereg.md", "stage1": dict(arms=res, passing_tags=passing, predictions=pr)})
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"stage-1 gate: passing {passing}   predictions {json.dumps(pr)}")


# ---------------------------------------------------------------- stage 2
def stage2():
    sys.path.insert(0, os.path.join(REPO, "MVSeed"))
    import e7_judge as E7
    from e8_judge import ci_at
    from e9_judge import z3_digest
    out = _load_json(OUT) if os.path.isfile(OUT) else {}
    s1 = out.get("stage1")
    if s1 is None or "passing_tags" not in s1:
        raise SystemExit("stage-1 judgement missing (run --stage 1 first)")
    tags = s1["passing_tags"]
    if not tags:
        out["stage2"] = dict(runs={}, passed=[], adopted=None, adopted_tag=None, predictions={}, note="no arm passed the stage-1 gate")
        json.dump(out, open(OUT, "w"), indent=1)
        print("stage-2: no arm passed the stage-1 gate; nothing to judge (prereg section 5-1)"); return
    d = E7.load(["jf_e10.json"])
    res = {}
    for rid, folder, tag, _ in ARMS:
        if tag not in tags:
            continue
        m = f"SegMaskSam3XW1CGPS4MSd{tag}FsCs10kG2Rs"
        Rm = list(E7.rows(d, m)); Rb = list(E7.rows(d, HEADLINE))
        if not Rm or not Rb:
            print(f"{rid}: no rows for {m} or {HEADLINE} (not scored yet?)"); continue
        sm, sb = E7.scene_scores(Rm), E7.scene_scores(Rb)
        cm, cb = E7.scene_scores(list(E7.rows(d, m, "complete"))), E7.scene_scores(list(E7.rows(d, HEADLINE, "complete")))
        st, stc = E7.paired(sb, sm), E7.paired(cb, cm)
        worst = min(st["per_scene"].values())
        rule = dict(mean_gt_0=st["mean"] > 0, ci_lower_gt_0=st["ci"][0] > 0, defence=worst >= DEFENCE, worst=worst, complete_ge_0=stc["mean"] >= 0)
        rule["pass"] = all(rule[k] for k in ("mean_gt_0", "ci_lower_gt_0", "defence", "complete_ge_0"))
        z3 = z3_digest("geoset_pred_ft", m, result=folder)
        z3b = dict(same=0, different=0, missing=0)
        from PIL import Image
        import numpy as np
        for r in Rm:
            if r["ref"]: continue
            c = E7.CFG[r["scene"]]; f, s0 = c["folder"], c["start_frame"]
            a = os.path.join(_result(r["scene"], folder), r["cam"], f"{r['obj']}.png")
            b = os.path.join(E7.DATA, f, m, r["cam"], str(s0), f"{r['obj']}.png")
            if not os.path.exists(a) and not os.path.exists(b): continue
            if not os.path.exists(a) or not os.path.exists(b): z3b["missing"] += 1; continue
            ma = np.asarray(Image.open(a).convert("L")) > 127; mb = np.asarray(Image.open(b).convert("L")) > 127
            z3b["same" if ma.shape == mb.shape and bool((ma == mb).all()) else "different"] += 1
        ref0 = [E7.jf(r, 0) for r in Rm if r["ref"] and r["gt"][0]]
        p9 = all(x is not None and x > 0.999 for x in ref0)
        res[rid] = dict(method=m, tag=tag, base=HEADLINE, n_scenes=len(sm), basic=statistics.mean(sm.values()), base_basic=statistics.mean(sb.values()),
                        paired=st, ci_bonferroni3=ci_at(sb, sm, 1 - 0.05 / 3), complete=statistics.mean(cm.values()), base_complete=statistics.mean(cb.values()),
                        paired_complete=stc, rule=rule, Z3_mask_digest=z3, Z3b_frame0_pixels=z3b, P9_ref_frame0_is_gt=p9)
        print(f"{rid} {m}\n   basic {res[rid]['basic']:.3f} vs {res[rid]['base_basic']:.3f}  Δ {st['mean']:+.3f}  CI [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]"
              f"  (98.33% [{res[rid]['ci_bonferroni3'][0]:+.3f}, {res[rid]['ci_bonferroni3'][1]:+.3f}])  {st['wins']}↑/{st['losses']}↓  worst {st['worst'][1]} {st['worst'][0]:+.2f}"
              f"\n   complete {res[rid]['complete']:.3f} vs {res[rid]['base_complete']:.3f}  Δ {stc['mean']:+.3f}  rule {rule}\n   Z3 {z3}  Z3b {z3b}  P9 {p9}")
    passed = [r for r in GATED if r in res and res[r]["rule"]["pass"]]
    adopted = None
    if passed:
        best = max(res[r]["basic"] for r in passed)
        adopted = next(r for r in GATED if r in passed and res[r]["basic"] >= best - TIE)      # tie -> C1
    pr = {}
    for r in res:
        kf = s1.get("arms", {}).get(r, {}).get("paired", {}).get("JF", {}).get("delta")
        if kf:
            ratio = res[r]["paired"]["mean"] / (66.7 * kf)
            pr[f"P8_{r}"] = dict(ratio=ratio, ok=0.4 <= ratio <= 1.0)
        pr[f"P9_{r}"] = res[r]["P9_ref_frame0_is_gt"]
        pr[f"Z3_{r}"] = isinstance(res[r]["Z3_mask_digest"], dict) and res[r]["Z3_mask_digest"]["bad"] == [] and res[r]["Z3_mask_digest"]["ok"] == 17
    out["stage2"] = dict(runs=res, passed=passed, adopted=adopted, adopted_tag=res[adopted]["tag"] if adopted else None, predictions=pr)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nstage-2: passed {passed} -> adopted {adopted} ({out['stage2']['adopted_tag']})   predictions {pr}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    a = ap.parse_args()
    stage1() if a.stage == 1 else stage2()
