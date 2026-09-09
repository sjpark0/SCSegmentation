#!/usr/bin/env python3
"""S1-E0': is the reference camera's frame-0 ceiling (J 0.951) a resolution round trip?

Why.  Prompting c_ini with its own ground truth and reading what the mainline writes at
frame 0 gives J 0.951 over the 398 reference pairs (docs/stage1-headroom.md section 1).
Every seed generator of docs/stage1-plan.md is finally read on that frame 0, so if the
missing 0.049 is lost in the mask plumbing rather than in the model it is a ceiling no
generator can pass, and the seed-J gates have to be read against it (plan section 5,
S1-E0'; rule in docs/stage1-E0-prereg.md section 4).  This script rebuilds the plumbing
on CPU, with no model, and measures the round trip on the same 398 pairs.

The chain, read from the frozen packages (line numbers of 2026-09-10; runMVSeg.py is
cited by method because it is being edited).  A c_ini ground-truth mask passes through
the tracker TWICE before it is written: stage 1 (the cross-view session,
runMVSeg.py AddReferenceMask / PropagateAcrossViews) stores its own re-prediction of the
reference view in masks_spatial, and stage 2 (TrackForward) feeds that back as a mask
prompt and writes frame 0 from the result (`out_binary_masks > 0`, the PNG loop).  Both
stages run the same code -- the stage-1 package's SCSam3TrackerPredictor.py /
SCSam3VideoInference.py differ from the NewMem files cited below only in comments
(diffed function by function) -- so one round trip is:

  1. add_new_mask (SCSam3TrackerPredictorNewMem.py:386-517).  The prompt is resized to
     input_mask_size = 1152 (:407-413; sam3_tracker_base.py:130-134: 288 x 4, not 1008)
     and sent through track_step -> _use_mask_as_output (sam3_tracker_base.py:961-967,
     :388-434).  That is the only place the SAM decoder runs for a mask prompt, and it
     runs for the object pointer alone (:410-415); the low-res logits it also returns
     are dropped at :482 (`pred_masks = None`).  What is kept is the prompt itself at
     video resolution, thresholded at 0.5 and mapped to +-NO_OBJ_SCORE = +-1024
     (:421-431, :483-485; sam3_tracker_base.py:23).  The prompt already is at video
     resolution (ground-truth PNG and JPEG sizes agree on all 17 scenes), so this step
     is lossless; the 1152 copy never reaches the output.
  2. propagate_in_video_preflight (:728) -> _consolidate_temp_output_across_obj with
     consolidate_at_video_res=False (:754-759): the +-1024 video-res mask is resized to
     low_res_mask_size = 288 x 288, bilinear, antialias=True, align_corners=False
     (:574-575, :648-656).  THE lossy step: the aspect ratio is not kept, one cell is
     6.7 x 3.75 px at 1920x1080 and 8.9 x 6.7 px at 2560x1920.  The memory encoder that
     follows (:667-681, 1008 upsample + non-overlap) feeds frames >= 1 only.
  3. propagate_in_video (:905-908): a frame that carries a prompt is not re-inferred;
     the stored 288 logits are yielded as low_res_masks (:943-946; the video-res half
     of _get_orig_video_res_output, :520-545, is discarded by the caller).  Tracker-
     level non-overlap never fires: non_overlap_masks_for_output=False
     (build_scsam3.py:506, :566) and every object sits in its own tracker state of
     batch size 1 (SCSam3VideoInferenceNewMem.py:1976; sam3_tracker_base.py:1115-1122
     returns a batch of one unchanged).
  4. _propogate_tracker_one_frame_local_gpu_multiple (SCSam3VideoInferenceNewMem.py:
     1041-1121; sam3_video_base.py:1098 for stage 1): fill_holes_in_mask_scores with
     max_area = fill_hole_area = 0 (SCSam3Video.py:18, :37) returns its input
     (sam3_tracker_utils.py:383-384); so does the call at :2054-2060.
  5. _convert_low_res_mask_to_video_res (:2199-2228, called at :1402): 288 -> (H, W)
     bilinear, align_corners=False, no antialias, then > 0.
  6. _postprocess_output (:435-527): empty masks are dropped (:471), then the object-
     wise non-overlap (:513 -> SCSam3TrackerPredictorNewMem.py:2040-2075).  Every mask-
     prompted object carries the same tracker score 10.0 (sam3_tracker_base.py:419-421,
     stored at :1386), so torch.argmax's first-maximal-index rule hands a contested
     pixel to the object that sorts first (:444, ascending object id).

So frame 0 of c_ini = roundtrip(roundtrip(ground truth)) with roundtrip = steps 2, 5, 6,
and the seed that `MVSeed/run_seed.py --order index` writes for c_ini is one roundtrip.
Nothing is left out: no step of the mask path needs the model.  What a CPU rebuild
cannot promise is CUDA's float32 interpolation bit for bit -- a pixel can flip only
where the upsampled logit lands within rounding of 0 -- and --compare-run measures that
against the mainline's PNGs directly.  Verified 2026-09-10 against SegMaskSam3XW1CGPS4M:
367 of the 398 c_ini frame-0 PNGs are pixel-identical to the rebuild, the other 31
differ by 1-4 px each (44 px in all), and the mean J agrees to 1e-5.  Step 6 turned out
to be a no-op on this input: linear resampling of disjoint masks sums to at most 1, so
no pixel crosses 0 for two objects at once (J_resample == J_rt1 on every row).

Output: per (scene, object) rows and a summary -- mean / median / min / fails (J<0.5)
of J after one round trip (rt1, the raw seed), two (rt2, frame 0) and resample-only
(steps 2+5 without the overlap rule, to isolate the rule's share); by size bin (<2000 /
2000-10000 / >=10000 ground-truth px, plan section 2), by rig (hemisphere:
docs/muvod-protocol.md), by scene.  With --compare-run: the measured frame-0 J from
that folder's PNGs and the agreement (IoU, differing pixels) between the rebuilt and
the written frame 0.  The pre-registered reading (prereg section 4) is taken from the
rt2 mean and stored as "reading".  J is eval/eval_jf.py's (mask IoU).

    docker run --rm -v /:/host -w /host$PWD scsam3 python MVSeed/ceiling_roundtrip.py
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from eval.eval_jf import db_eval_iou  # noqa: E402

DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
OUT = os.path.join(REPO, "MVSeed", "runs", "ceiling_roundtrip.json")
COMPARE_RUN = "SegMaskSam3XW1CGPS4M"     # the run the 0.951 was read from (stage1-headroom)

LOW_RES = 288                 # sam3_tracker_base.py:130  1008 // 14 * 4
NO_OBJ_SCORE = 1024.0         # sam3_tracker_base.py:23, sign flipped for the foreground
HEMISPHERE = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder")   # muvod-protocol.md
SIZE_BINS = ("<2000", "2000-10000", ">=10000")      # ground-truth px, plan section 2
FAIL_J = 0.5
CEILING = 0.951               # stage1-headroom.md section 1, 398 reference pairs
ARTEFACT_BAND = 0.02          # prereg section 4: |mean rt2 - CEILING| <= band -> artefact
OTHER_MIN = 0.98              # prereg section 4: mean rt2 >= this -> other cause
J_KEYS = ("J_resample", "J_rt1", "J_rt2")
COMPARE_KEYS = ("J_measured", "agree")


def cam_name(c, prefix, prefix1):
    return f"{prefix}{c:0{prefix1}d}"


def rig_of(scene):
    return "hemisphere" if scene in HEMISPHERE else "pinhole"


def size_bin(gt_px):
    return SIZE_BINS[0] if gt_px < 2000 else SIZE_BINS[1] if gt_px < 10000 else SIZE_BINS[2]


# ---------------------------------------------------------------- the round trip
def downsample(mask, low_res=LOW_RES):
    """Step 2.  bool (H, W) -> (1, 1, low_res, low_res) float32 logits: the +-1024 mask
    of SCSam3TrackerPredictorNewMem.py:483-485 through the interpolate of :649-655."""
    m = torch.as_tensor(np.ascontiguousarray(mask, dtype=bool))
    logits = torch.where(m, torch.tensor(NO_OBJ_SCORE), torch.tensor(-NO_OBJ_SCORE))
    return F.interpolate(logits[None, None], size=(low_res, low_res), mode="bilinear",
                         align_corners=False, antialias=True)


def upsample(logits, hw):
    """Step 5.  SCSam3VideoInferenceNewMem.py:2220-2228 -> bool (H, W)."""
    video = F.interpolate(logits, size=(int(hw[0]), int(hw[1])), mode="bilinear",
                          align_corners=False)
    return (video[0, 0] > 0.0).numpy()


def resample(mask, low_res=LOW_RES):
    """Steps 2 + 5 on one object, no overlap rule."""
    return upsample(downsample(mask, low_res), mask.shape)


def resolve_overlaps(masks):
    """Step 6 with equal tracker scores: in the given order (ascending object id, the
    only order the chain uses) a pixel already claimed is removed from every later mask."""
    out, claimed = [], None
    for m in masks:
        if claimed is None:
            claimed = np.zeros_like(m)
        out.append(m & ~claimed)
        claimed = claimed | m
    return out


def roundtrip(masks, low_res=LOW_RES):
    """One pass of the tracker over a set of masks: what stage 1 stores for c_ini, and
    what stage 2 writes at frame 0 when applied to that once more."""
    return resolve_overlaps([resample(m, low_res) for m in masks])


# ------------------------------------------------------------------- one scene
def load_mask(path, hw):
    """A mainline PNG as bool, the way MVSeed/score_seeds.py reads one (missing = empty)."""
    pm = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
    if pm is None:
        return np.zeros(hw, dtype=bool)
    if pm.shape[:2] != tuple(hw):
        pm = cv2.resize(pm, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return pm > 127


def scene_rows(scene, cfg, census, compare_run=None, low_res=LOW_RES, root=DATA):
    """Rows for every basic object of the scene's c_ini on its start_frame."""
    d = cfg[scene]
    ds = os.path.join(root, d["folder"])
    start = d["start_frame"]
    c_ini = census[scene]["c_ini"]
    assert c_ini == cam_name(d["c_ini"], d["prefix"], d["prefix1"]), (scene, c_ini)
    gt_path = os.path.join(ds, "Mask", c_ini, f"{start:06d}.png")
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(gt_path)
    objs = sorted(int(o) for o in census[scene]["seed_ids_c_ini"])
    gts = [gt == o for o in objs]
    free = [resample(g, low_res) for g in gts]      # steps 2 + 5
    rt1 = resolve_overlaps(free)                     # + step 6: the stage-1 seed
    rt2 = roundtrip(rt1, low_res)                    # the stage-2 frame 0

    cmp_dir = os.path.join(ds, compare_run, c_ini, str(start)) if compare_run else None
    compare = bool(cmp_dir and os.path.isdir(cmp_dir))
    rows = []
    for i, o in enumerate(objs):
        row = dict(scene=scene, cam=c_ini, obj=o, gt_px=int(gts[i].sum()),
                   size_bin=size_bin(int(gts[i].sum())), rig=rig_of(scene),
                   J_resample=float(db_eval_iou(gts[i], free[i])),
                   J_rt1=float(db_eval_iou(gts[i], rt1[i])),
                   J_rt2=float(db_eval_iou(gts[i], rt2[i])),
                   rt1_px=int(rt1[i].sum()), rt2_px=int(rt2[i].sum()))
        if compare:
            pm = load_mask(os.path.join(cmp_dir, f"{o}.png"), gt.shape)
            row.update(J_measured=float(db_eval_iou(gts[i], pm)), measured_px=int(pm.sum()),
                       agree=float(db_eval_iou(pm, rt2[i])), diff_px=int((rt2[i] != pm).sum()))
        rows.append(row)
    return rows


# --------------------------------------------------------------------- summary
def stats(vals):
    if not vals:
        return None
    return dict(n=len(vals), mean=float(np.mean(vals)), median=float(np.median(vals)),
                min=float(np.min(vals)), fail=int(sum(v < FAIL_J for v in vals)))


def block(rows, keys):
    return {k: stats([r[k] for r in rows if r.get(k) is not None]) for k in keys}


def summarize(rows):
    keys = J_KEYS + tuple(k for k in COMPARE_KEYS if any(r.get(k) is not None for r in rows))
    out = {"n": len(rows), "all": block(rows, keys),
           "by_size": {b: block([r for r in rows if r["size_bin"] == b], keys) for b in SIZE_BINS},
           "by_rig": {g: block([r for r in rows if r["rig"] == g], keys)
                      for g in ("pinhole", "hemisphere")},
           "by_scene": {s: block([r for r in rows if r["scene"] == s], keys)
                        for s in sorted({r["scene"] for r in rows})}}
    cmp_rows = [r for r in rows if r.get("diff_px") is not None]
    if cmp_rows:
        out["compare"] = dict(n=len(cmp_rows),
                              diff_px_total=int(sum(r["diff_px"] for r in cmp_rows)),
                              pairs_identical=int(sum(r["diff_px"] == 0 for r in cmp_rows)),
                              max_diff_px=int(max(r["diff_px"] for r in cmp_rows)),
                              mean_delta_measured_minus_rt2=float(np.mean(
                                  [r["J_measured"] - r["J_rt2"] for r in cmp_rows])))
    return out


def reading(mean_rt2):
    """docs/stage1-E0-prereg.md section 4, applied to the two-round-trip mean."""
    if abs(mean_rt2 - CEILING) <= ARTEFACT_BAND:
        return "artefact"
    if mean_rt2 >= OTHER_MIN:
        return "other"
    if mean_rt2 > CEILING:
        return "partial"
    return "below_band"      # the rebuild loses more than the mainline did: chain mismatch


def fmt(s, key="mean"):
    return "   -  " if s is None else f"{s[key]:.4f}"


def print_report(summary, rows, compare_run):
    keys = [k for k in J_KEYS + COMPARE_KEYS if summary["all"].get(k)]
    head = "".join(f"{k:>12}" for k in keys)
    print(f"{'n=' + str(summary['n']):24}{head}")
    for label, blk in [("all", summary["all"])] + \
            [(f"  size {b}", summary["by_size"][b]) for b in SIZE_BINS] + \
            [(f"  rig {g}", summary["by_rig"][g]) for g in ("pinhole", "hemisphere")]:
        n = next((blk[k]["n"] for k in keys if blk.get(k)), 0)
        print(f"{label:18}n={n:<5}" + "".join(f"{fmt(blk.get(k)):>12}" for k in keys))
    print(f"{'median':24}" + "".join(f"{fmt(summary['all'].get(k), 'median'):>12}" for k in keys))
    print(f"{'min':24}" + "".join(f"{fmt(summary['all'].get(k), 'min'):>12}" for k in keys))
    print(f"{'fail J<0.5':24}" + "".join(
        f"{('   -  ' if summary['all'].get(k) is None else summary['all'][k]['fail']):>12}"
        for k in keys))
    print("장면별 (n, rt1, rt2" + (", measured, agree)" if "agree" in keys else ")"))
    for s, blk in sorted(summary["by_scene"].items(), key=lambda x: x[1]["J_rt2"]["mean"]):
        line = f"    {s:22}n={blk['J_rt2']['n']:<4}{blk['J_rt1']['mean']:.3f}  {blk['J_rt2']['mean']:.3f}"
        if blk.get("agree"):
            line += f"  {blk['J_measured']['mean']:.3f}  {blk['agree']['mean']:.4f}"
        print(line)
    if summary.get("compare"):
        c = summary["compare"]
        print(f"재현 vs {compare_run} 프레임 0: {c['pairs_identical']}/{c['n']} 쌍 픽셀 동일, "
              f"다른 픽셀 합 {c['diff_px_total']} (최대 {c['max_diff_px']}), "
              f"측정 J − rt2 J 평균 {c['mean_delta_measured_minus_rt2']:+.5f}")
    m = summary["all"]["J_rt2"]["mean"]
    print(f"왕복 J 평균: rt1 {summary['all']['J_rt1']['mean']:.4f}, rt2 {m:.4f} vs 천장 {CEILING} "
          f"(차이 {m - CEILING:+.4f}) → 사전 등록 판독: {reading(m)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=None, help="default: the 17 census scenes")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--compare-run", default=COMPARE_RUN,
                    help="mainline folder whose c_ini frame 0 is compared; '' to skip")
    ap.add_argument("--low-res", type=int, default=LOW_RES,
                    help="what-if grid; anything but 288 is not the chain and skips the compare")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--census", default=CENSUS)
    ap.add_argument("--data-root", default=DATA)
    args = ap.parse_args(argv)

    cfg = json.load(open(args.config, encoding="utf-8"))
    census = json.load(open(args.census))
    scenes = args.scenes or sorted(census)
    compare_run = args.compare_run if (args.compare_run and args.low_res == LOW_RES) else None
    rows = []
    for scene in scenes:
        if scene not in cfg or scene not in census:
            sys.exit(f"{scene!r} is not in both {args.config} and {args.census}")
        rows.extend(scene_rows(scene, cfg, census, compare_run, args.low_res, args.data_root))
        print(f"  {scene:22}{len(rows):>4} rows", flush=True)
    summary = summarize(rows)
    print_report(summary, rows, compare_run)
    out = {"meta": dict(low_res=args.low_res, no_obj_score=NO_OBJ_SCORE, compare_run=compare_run,
                        ceiling=CEILING, artefact_band=ARTEFACT_BAND, other_min=OTHER_MIN,
                        J="eval/eval_jf.py db_eval_iou (mask IoU)",
                        J_resample="steps 2+5: antialiased bilinear down to low_res^2, bilinear up, >0",
                        J_rt1="one round trip (+ overlap rule, lowest id wins) = the stage-1 seed",
                        J_rt2="two round trips = the mainline's frame 0 for c_ini",
                        J_measured=f"J of {compare_run}'s c_ini frame-0 PNG against the ground truth",
                        agree=f"IoU between the rebuilt rt2 mask and {compare_run}'s PNG",
                        size_bins=SIZE_BINS, hemisphere=HEMISPHERE, argv=sys.argv[1:]),
           "reading": reading(summary["all"]["J_rt2"]["mean"]),
           "summary": summary, "rows": rows}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
