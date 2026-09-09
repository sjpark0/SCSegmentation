#!/usr/bin/env python3
"""Score first-frame seed masks against the ground truth (MVSeed, stage 1 only).

Same J as eval/eval_jf.py (mask IoU; empty prediction on empty GT scores 1.0) and the
same F (DAVIS boundary F, bound_th 0.008), read on ONE frame: the scene's start_frame.
The object set is MUVOD's basic set -- the ids visible in c_ini's first frame -- so the
numbers line up with the benchmark's basic evaluation.  The J/F code path is the
2026-09-09 original, untouched; everything below is bookkeeping around it.

Populations (docs/stage1-plan.md section 2).  Every (camera, object) row of the three
annotated cameras is scored, then split:

  gt_present   non-reference rows whose camera holds the object (664 today).  The
               denominator of every gate and the headline number.
  gt_empty     non-reference rows with an empty ground truth (132 today).  An empty
               prediction scores J=1 there, which is why they are kept OUT of the
               headline mean: a method that predicts nothing would earn 132 free
               points.  What matters on these rows is the hallucination count, a
               non-empty prediction on an empty ground truth.
  all          gt_present + gt_empty (796 today), the old default, reported alongside
               so the two conventions can be compared.
  ref          the c_ini rows.  A monitor only: the seed there is the prompt itself.

Row fields beyond J/F: gt_px, pred_px, size_bin (ground-truth area <2000 / 2000-10000
/ >=10000 px), rig (hemisphere for the four Google light-field scenes, pinhole for the
rest -- docs/muvod-protocol.md), fail (J<0.5), boundary_band (0.4<=J<0.6) and
view_distance, |view index of the camera - view index of c_ini| in the loaded camera
ring (perms, else start_cam..) -- the index eval/eval_jf.py records as meta.view_index
and eval/e4_judge.py bins by.

--baseline RUN pairs every gt_present row of a run with the same (scene, camera,
object) row of RUN and reports delta J = run - baseline: mean, scene-cluster bootstrap
95% CI and Wilcoxon p (eval/report_jf.py: seed 0, 10,000 resamples; exact Wilcoxon up
to 22 non-tied pairs, normal approximation above), wins/ties/losses, fail and band
counts on both sides, recovered (baseline J<0.5 -> run J>=0.5) and newly failed pairs.
Read on the whole population and on N (baseline J>=0.5) / F (baseline J<0.5), size
bins, rigs, view distances and scenes; plus the hallucination change on gt_empty and
the largest |delta| on the c_ini rows (must be 0 when the seed there is untouched).
The S1-E1 gate of docs/stage1-E0-prereg.md section 4 reads exactly these numbers; the
script prints them and does not judge.

Output JSON: {"meta": definitions, "runs": {run: {"summary", "scenes": {scene: {legacy
keys c_ini/n/n_nonref/J_all/F_all/J_nonref/fail_nonref, "rig", "rows"}}}}, "baseline":
same shape for RUN, "paired": {run: paired summary}}.

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \\
        python MVSeed/score_seeds.py --runs MVSeed_e0_index --out MVSeed/runs/e0_index.json
    ... --runs MVSeed_e1_reverse --baseline MVSeed_e0_index --out MVSeed/runs/e1_reverse.json
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from eval.eval_jf import db_eval_iou, db_eval_boundary, _disk, crop_box, BOUND_TH  # noqa: E402
from eval.report_jf import (boot_ci, wilcoxon_p, mean, TIE,  # noqa: E402
                            BOOT_SCHEME, WILCOXON_SCHEME)

DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")

# the Google Immersive Light Field rigs: camera numbers are not spatial order there
# (docs/muvod-protocol.md evidence B); every other scene is a pinhole array
HEMISPHERE = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder")
RIGS = ("pinhole", "hemisphere")
SIZE_BINS = ("<2000", "2000-10000", ">=10000")      # ground-truth px, plan section 2
FAIL_J = 0.5
BAND = (0.4, 0.6)                                    # boundary band: [0.4, 0.6)
POPULATIONS = ("gt_present", "all", "gt_empty", "ref")


def cam_name(c, prefix, prefix1):
    return f"{prefix}{c:0{prefix1}d}"


def score_scene(scene, run, cfg, basic, root=DATA):
    """[(camera, obj, J, F, gt_px, pred_px)] on the scene's start_frame."""
    d = cfg[scene]
    ds = os.path.join(root, d["folder"])
    start = d["start_frame"]
    rows = []
    for c in sorted(d["cam_list"]):
        cam = cam_name(c, d["prefix"], d["prefix1"])
        gt_path = os.path.join(ds, "Mask", cam, f"{start:06d}.png")
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            continue
        h, w = gt_img.shape[:2]
        bound_pix = int(math.ceil(BOUND_TH * np.linalg.norm((h, w))))
        se = _disk(bound_pix)
        for obj in sorted(basic):
            gm = gt_img == obj
            p = os.path.join(ds, run, cam, str(start), f"{obj}.png")
            pm = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
            if pm is None:
                pm = np.zeros((h, w), dtype=bool)
            else:
                if pm.shape[:2] != (h, w):
                    pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                pm = pm > 127
            box = crop_box(gm, pm, bound_pix + 2, (h, w))
            if box is None:
                J = F = 1.0
            else:
                y0, y1, x0, x1 = box
                J = db_eval_iou(gm[y0:y1, x0:x1], pm[y0:y1, x0:x1])
                F = db_eval_boundary(pm[y0:y1, x0:x1], gm[y0:y1, x0:x1], bound_pix, se)
            rows.append((cam, int(obj), float(J), float(F),
                         int(gm.sum()), int(pm.sum())))
    return rows


# ------------------------------------------------------------------ row fields
def rig_of(scene):
    return "hemisphere" if scene in HEMISPHERE else "pinhole"


def size_bin(gt_px):
    """Ground-truth area bin; None when there is no ground truth to bin."""
    if gt_px <= 0:
        return None
    return SIZE_BINS[0] if gt_px < 2000 else SIZE_BINS[1] if gt_px < 10000 else SIZE_BINS[2]


def view_index_map(d):
    """camera name -> index in the loaded camera ring (eval/e4_judge.view_index_map)."""
    perms = d.get("perms") or list(range(d["start_cam"], d["num_cam"] + d["start_cam"]))
    return {cam_name(x, d["prefix"], d["prefix1"]): i for i, x in enumerate(perms)}


def annotate(scene, rows, c_ini, vmap):
    """score_scene tuples -> row dicts with the section-2 fields."""
    out = []
    for cam, obj, J, F, gt_px, pred_px in rows:
        out.append(dict(scene=scene, cam=cam, obj=obj, ref=cam == c_ini, J=J, F=F,
                        gt_px=gt_px, pred_px=pred_px, size_bin=size_bin(gt_px),
                        rig=rig_of(scene), fail=J < FAIL_J,
                        boundary_band=BAND[0] <= J < BAND[1],
                        view_distance=abs(vmap[cam] - vmap[c_ini])))
    return out


def population(rows, name):
    if name == "ref":
        return [r for r in rows if r["ref"]]
    non = [r for r in rows if not r["ref"]]
    if name == "all":
        return non
    if name == "gt_present":
        return [r for r in non if r["gt_px"] > 0]
    if name == "gt_empty":
        return [r for r in non if r["gt_px"] == 0]
    raise ValueError(name)


# --------------------------------------------------------------------- summary
def pop_stats(rows):
    """n, mean J and F, fail and boundary-band counts (nan means when empty)."""
    return dict(n=len(rows), J=mean([r["J"] for r in rows]), F=mean([r["F"] for r in rows]),
                fail=sum(1 for r in rows if r["fail"]),
                band=sum(1 for r in rows if r["boundary_band"]))


def empty_stats(rows):
    """gt_empty rows: how many predictions are not empty (hallucinations)."""
    return dict(n=len(rows), halluc=sum(1 for r in rows if r["pred_px"] > 0))


def levels_of(rows, key, order=None):
    """The values of row field `key` that occur, in `order` when given, else sorted."""
    seen = {r[key] for r in rows}
    return [k for k in order if k in seen] if order else sorted(seen)


def breakdown(rows, key, order=None, stats=pop_stats):
    return {str(k): stats([r for r in rows if r[key] == k])
            for k in levels_of(rows, key, order)}


def summarize(rows):
    """One run: every population; gt_present and all by scene, size bin, rig, distance."""
    s = {}
    for name in ("gt_present", "all"):
        p = population(rows, name)
        by_scene = breakdown(p, "scene")
        s[name] = dict(pop_stats(p),
                       J_scene_mean=mean([v["J"] for v in by_scene.values()]),
                       by_scene=by_scene,
                       by_size=breakdown(p, "size_bin", SIZE_BINS),
                       by_rig=breakdown(p, "rig", RIGS),
                       by_view_distance=breakdown(p, "view_distance"))
    ge = population(rows, "gt_empty")
    s["gt_empty"] = dict(empty_stats(ge), by_scene=breakdown(ge, "scene", stats=empty_stats))
    s["ref"] = pop_stats(population(rows, "ref"))
    return s


# ---------------------------------------------------------------------- paired
def pair_rows(run_rows, base_rows):
    """[(run_row, base_row)] over the (scene, camera, object) keys both runs have."""
    base = {(r["scene"], r["cam"], r["obj"]): r for r in base_rows}
    out = []
    for r in run_rows:
        b = base.get((r["scene"], r["cam"], r["obj"]))
        if b is not None:
            out.append((r, b))
    return out


def delta_stats(pairs):
    """delta J = run - baseline over [(run_row, base_row)].

    Cluster bootstrap groups = one list of deltas per scene, scenes in sorted order
    (boot_ci is seeded, so the group order fixes the CI to the last digit)."""
    deltas = [r["J"] - b["J"] for r, b in pairs]
    groups = {}
    for (r, _), dl in zip(pairs, deltas):
        groups.setdefault(r["scene"], []).append(dl)
    lo, hi = boot_ci([groups[s] for s in sorted(groups)])
    p, how = wilcoxon_p(deltas)
    wins = sum(1 for x in deltas if x >= TIE)
    losses = sum(1 for x in deltas if x <= -TIE)
    return dict(n=len(pairs), n_scenes=len(groups), mean=mean(deltas), ci_lo=lo, ci_hi=hi,
                wilcoxon_p=p, wilcoxon=how,
                wins=wins, ties=len(deltas) - wins - losses, losses=losses,
                J_base=mean([b["J"] for _, b in pairs]), J_run=mean([r["J"] for r, _ in pairs]),
                fail_base=sum(1 for _, b in pairs if b["fail"]),
                fail_run=sum(1 for r, _ in pairs if r["fail"]),
                recovered=sum(1 for r, b in pairs if b["fail"] and not r["fail"]),
                new_fail=sum(1 for r, b in pairs if r["fail"] and not b["fail"]),
                band_base=sum(1 for _, b in pairs if b["boundary_band"]),
                band_run=sum(1 for r, _ in pairs if r["boundary_band"]))


def paired_summary(run_rows, base_rows):
    """Everything --baseline reports, gt_present pairs only (plus the two monitors)."""
    gp = population(run_rows, "gt_present")
    pairs = pair_rows(gp, population(base_rows, "gt_present"))

    def sub(keep):
        return delta_stats([p for p in pairs if keep(p)])

    def by(key, order=None):
        # the key is read off the baseline row; ground truth is shared so it is the same
        return {str(k): sub(lambda p, k=k: p[1][key] == k)
                for k in levels_of([b for _, b in pairs], key, order)}

    empty = pair_rows(population(run_rows, "gt_empty"), population(base_rows, "gt_empty"))
    ref = pair_rows(population(run_rows, "ref"), population(base_rows, "ref"))
    return dict(
        n_run=len(gp), n_paired=len(pairs),
        subsets={"gt_present": delta_stats(pairs),
                 "N": sub(lambda p: not p[1]["fail"]),
                 "F": sub(lambda p: p[1]["fail"])},
        by_size=by("size_bin", SIZE_BINS),
        by_rig=by("rig", RIGS),
        by_view_distance=by("view_distance"),
        by_scene=by("scene"),
        gt_empty=dict(n=len(empty),
                      halluc_base=sum(1 for _, b in empty if b["pred_px"] > 0),
                      halluc_run=sum(1 for r, _ in empty if r["pred_px"] > 0)),
        ref=dict(n=len(ref),
                 max_abs_delta=max((abs(r["J"] - b["J"]) for r, b in ref), default=0.0)))


# --------------------------------------------------------------------- printing
def f4(x, sign=False):
    if x != x:
        return "   -   " if sign else "  -   "
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "-"


def print_run(run, summary, n_scenes):
    print(f"== {run}: {n_scenes}장면 ==")
    print(f"  {'모집단':12}{'n':>5}  {'J':>6}  {'F':>6}  실패(J<0.5)   경계[0.4,0.6)")
    for name in ("gt_present", "all"):
        s = summary[name]
        print(f"  {name:12}{s['n']:>5}  {f4(s['J'])}  {f4(s['F'])}  "
              f"{s['fail']:>3} ({pct(s['fail'], s['n']):>5})   {s['band']:>3}")
    e = summary["gt_empty"]
    print(f"  {'gt_empty':12}{e['n']:>5}  환각(빈 정답에 비어 있지 않은 예측) {e['halluc']}")
    r = summary["ref"]
    print(f"  {'ref(c_ini)':12}{r['n']:>5}  {f4(r['J'])}  {f4(r['F'])}  {r['fail']:>3}")
    g = summary["gt_present"]
    print(f"  gt_present 세부 (장면 평균 J {f4(g['J_scene_mean'])})")
    for label, table in (("크기", g["by_size"]), ("리그", g["by_rig"]),
                         ("거리", g["by_view_distance"])):
        for k, v in table.items():
            print(f"    {label} {k:12}n {v['n']:>4}  J {f4(v['J'])}  "
                  f"실패 {v['fail']:>3} ({pct(v['fail'], v['n'])})  경계 {v['band']}")
    print("  장면별 gt_present (J 낮은 순)")
    for sc, v in sorted(g["by_scene"].items(), key=lambda kv: kv[1]["J"]):
        e = summary["gt_empty"]["by_scene"].get(sc, dict(n=0, halluc=0))
        print(f"    {sc:22}n {v['n']:>3}  J {v['J']:.3f}  실패 {v['fail']:>2}   "
              f"환각 {e['halluc']}/{e['n']}")


def print_paired(run, base, p):
    print(f"== ΔJ = {run} − {base} (gt_present 짝 {p['n_paired']}/{p['n_run']}) ==")
    print(f"  {'부분집합':18}{'n':>5}  {'평균Δ':>7}  {'클러스터 95% CI':>18}  "
          f"{'Wilcoxon p':>10}  상승/동점/하락   실패 전→후  회복  새실패")

    def line(label, s):
        if s["n"] == 0:
            print(f"  {label:18}{'0':>5}  (없음)")
            return
        print(f"  {label:18}{s['n']:>5}  {f4(s['mean'], True)}  "
              f"[{f4(s['ci_lo'], True)}, {f4(s['ci_hi'], True)}]  "
              f"{f4(s['wilcoxon_p']):>10}  {s['wins']:>3}/{s['ties']:>3}/{s['losses']:>3}      "
              f"{s['fail_base']:>3}→{s['fail_run']:<3}   {s['recovered']:>3}   {s['new_fail']:>3}"
              f"   ({s['wilcoxon']})")
    line("전체", p["subsets"]["gt_present"])
    line("N (기준 J≥0.5)", p["subsets"]["N"])
    line("F (기준 J<0.5)", p["subsets"]["F"])
    for label, table in (("크기", p["by_size"]), ("리그", p["by_rig"]),
                         ("거리", p["by_view_distance"])):
        for k, s in table.items():
            line(f"{label} {k}", s)
    print("  장면별 (평균Δ 낮은 순; 하락 > 0.005 표시)")
    for sc, s in sorted(p["by_scene"].items(), key=lambda kv: kv[1]["mean"]):
        flag = "   <-- 하락 > 0.005" if s["mean"] < -0.005 else ""
        print(f"    {sc:22}n {s['n']:>3}  평균Δ {f4(s['mean'], True)}  "
              f"J {s['J_base']:.3f}→{s['J_run']:.3f}  실패 {s['fail_base']}→{s['fail_run']}{flag}")
    e, r = p["gt_empty"], p["ref"]
    print(f"  gt_empty 환각 {e['halluc_base']} → {e['halluc_run']}  (n {e['n']})")
    print(f"  ref(c_ini) n {r['n']}  최대 |Δ| {r['max_abs_delta']:.4f}")


# ------------------------------------------------------------------------ main
def no_nan(o):
    """json.dump writes NaN, which strict readers reject; store null instead."""
    if isinstance(o, dict):
        return {k: no_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [no_nan(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def score_run(run, scenes, cfg, census, root):
    """{scene: per-scene entry} for one run folder; scenes without any row are dropped."""
    per_scene = {}
    for scene in scenes:
        if scene not in cfg:
            continue
        d = cfg[scene]
        basic = set(census[scene]["seed_ids_c_ini"])
        ci = census[scene]["c_ini"]
        if "c_ini" in d and cam_name(d["c_ini"], d["prefix"], d["prefix1"]) != ci:
            sys.exit(f"{scene}: c_ini differs between the config ({d['c_ini']}) and the "
                     f"census ({ci})")
        rows = score_scene(scene, run, cfg, basic, root)
        if not rows:
            continue
        rows = annotate(scene, rows, ci, view_index_map(d))
        non = population(rows, "all")
        per_scene[scene] = {
            # legacy keys of the 2026-09-09 layout (means over every row / every
            # non-reference row, gt_empty included); np.mean keeps them bit-identical
            "c_ini": ci, "n": len(rows), "n_nonref": len(non),
            "J_all": float(np.mean([r["J"] for r in rows])),
            "F_all": float(np.mean([r["F"] for r in rows])),
            "J_nonref": float(np.mean([r["J"] for r in non])) if non else None,
            "fail_nonref": sum(1 for r in non if r["fail"]),
            "rig": rig_of(scene),
            "rows": rows,
        }
    return per_scene


def all_rows(per_scene):
    return [r for v in per_scene.values() for r in v["rows"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="output folder names under each scene")
    ap.add_argument("--baseline", default=None,
                    help="run folder to pair every --runs entry against (delta J = run - baseline)")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--census", default=CENSUS, help="docs/raw/muvod_object_sets.json")
    args = ap.parse_args(argv)
    cfg = json.load(open(args.config, encoding="utf-8"))
    census = json.load(open(args.census))
    scenes = args.scenes or sorted(census)
    out = {"meta": dict(populations=POPULATIONS, size_bins=SIZE_BINS, hemisphere=HEMISPHERE,
                        fail_j=FAIL_J, band=BAND, boot=BOOT_SCHEME, wilcoxon=WILCOXON_SCHEME),
           "runs": {}}
    base_rows = None
    if args.baseline:
        per_scene = score_run(args.baseline, scenes, cfg, census, args.data_root)
        if not per_scene:
            sys.exit(f"--baseline {args.baseline}: no scene has a row")
        base_rows = all_rows(per_scene)
        out["baseline"] = {"run": args.baseline, "summary": summarize(base_rows),
                           "scenes": per_scene}
        out["paired"] = {}
    for run in args.runs:
        per_scene = score_run(run, scenes, cfg, census, args.data_root)
        rows = all_rows(per_scene)
        summary = summarize(rows)
        out["runs"][run] = {"summary": summary, "scenes": per_scene}
        print_run(run, summary, len(per_scene))
        if base_rows is not None:
            paired = paired_summary(rows, base_rows)
            out["paired"][run] = paired
            print_paired(run, args.baseline, paired)
        print()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(no_nan(out), f, indent=1)
        print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
