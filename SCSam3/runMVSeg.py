#!/usr/bin/env python3
"""Run one SCSam3 demo algorithm over an MVSeg dataset and write per-object masks.

The output layout is the one Data/MVSeg/eval_jf.py expects, so J&F can be
computed straight afterwards:

    <dataset>/<out>/<cam>/<frame>/<object id>.png      binary, 0 or 255

What the run does, matching the SAM 2 baseline in
SCSam2/demo/sam2_demoVideoNew_maskSingleInputMVSeg.py so the numbers stay
comparable:

  1. open one temporal session per camera over that camera's JPEG folder,
  2. build one cross-view session from frame `start_frame` of every camera,
  3. pick the reference camera as the annotated camera whose ground truth at
     `start_frame` holds the most objects,
  4. feed that camera's ground-truth masks in as the only prompt,
  5. propagate across views, then track forward in time in each view.

Only step 5 needs a GPU.  Nothing here writes outside the dataset directory.

    python runMVSeg.py Blocks --algo OneStageNew
    python runMVSeg.py Blocks --algo OneStage --out SegMaskSam3OneStage
    python runMVSeg.py Welder --algo MVOpt --xview-window 0   # -> SegMaskSam3XW0, closure
    python runMVSeg.py Fencing --algo MVOpt --xview-window 1 --xview-mode B   # -> SegMaskSam3XW1B, closure cone
    python runMVSeg.py Fencing --algo MVOpt --xview-window 1 --xview-gate --xview-ptr   # -> SegMaskSam3XW1GP
    python runMVSeg.py Fencing --algo MVOpt --xview-window 0 --ref-cam muvod --seeds-from MVSeed_control
                                                                  # -> SegMaskSam3XW0MSdcontrol, stage-1 supply
    python runMVSeg.py Fencing --algo MVOpt --xview-window 0 --ref-cam muvod --frame0-seed
                                                                  # -> SegMaskSam3XW0MFs, seed written as frame 0
    python runMVSeg.py Fencing --algo MVOpt --xview-window 0 --ref-cam muvod --frame0-seed --crop-small 10000
                                                                  # -> SegMaskSam3XW0MFsCs10kG2, small objects crop-tracked
    python runMVSeg.py Fencing --algo MVOpt --xview-window 1 --xview-mode C --xview-gate --xview-ptr \
        --xview-tpos-shift 4 --ref-cam muvod --stage1 geoset_rep_xs --frame0-seed --crop-small 10000 --ref-snap
                                                                  # -> SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs, the
                                                                  #    headline in one process (stage 1 in memory; S1-E14
                                                                  #    adopted geoset_rep_xs; geoset_rep_x = S1-E11,
                                                                  #    geoset_rep = the S1-E8 preset)

Both demos carry their own copy of the model code, so the algorithm is selected
by putting its directory first on sys.path.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
# let the model infer the cross-view start from the prompted view
SPATIAL_START_IMPLICIT = os.environ.get("SPATIAL_START_IMPLICIT", "1") == "1"
ALGOS = {"OneStage": "demoSCSam3OneStage",
         "OneStageNew": "demoSCSam3OneStageNew",
         "MVOpt": "demoSCSam3MVOpt"}
DEFAULT_OUT = {"OneStage": "SegMaskSam3OneStage",
               "OneStageNew": "SegMaskSam3OneStageNew",
               "MVOpt": "SegMaskSam3MVOpt"}
# MVOpt is the development target. As of 2026-09-04 demoSCSam3OneStageNew holds
# a byte-identical frozen snapshot of it (see that folder's FROZEN.md); the
# unpatched original that produced the published J&F numbers is at git tag
# baseline-onestagenew. Both share the cross-view memory design, so both need
# every view tracked.
NEEDS_ALL_VIEWS = ("OneStageNew", "MVOpt")
# Cross-view (XW) lineage: hygiene fixes + window, MVOpt only (OneStage has no
# cross-view memory, OneStageNew is a frozen snapshot).  ROADMAP Phase 2.
XVIEW_ALGOS = ("MVOpt",)
XVIEW_MAX_WINDOW = 6                    # maskmem_tpos_enc has 7 rows; row 6 is the cond row
XVIEW_ENV = ("SCSAM3_XVIEW_WINDOW", "SCSAM3_XVIEW_HYGIENE")
XVIEW_OUT_PREFIX = "SegMaskSam3XW"      # every XW output folder starts with this; legacy never does
ONESTAGE_CLOSURE_OUT = "SegMaskSam3OneStageC"   # experiment row 2
# Phase 3 / P4 neighbourhood variants (docs/phase3-neighbourhood.md).  "A" is today's
# gather and keeps the bare XW{W} name; the others get XW{W}{letter}.  E runs two passes
# per frame (the generators, then a recompute request).  closure growth: how many
# sessions per tracked frame the dependency cone of a scored view widens by (x W).
XVIEW_MODES = ("A", "B", "C", "D", "E")
XVIEW_LEGACY_MODE = "A"
XVIEW_TWO_PASS = ("E",)
XVIEW_CLOSURE_GROWTH = {"A": 0, "B": 1, "C": 1, "D": 0, "E": 2}
# Phase 3 / P12 conditioning knobs on the neighbour tokens (docs/phase3-conditioning.md):
# G gate, P pointer, S tpos row shift.  Folder SegMaskSam3XW{W}{mode}{G}{P}{S<s>} (A has no
# mode letter: XW1G, XW1P, XW1GP, XW1S2, XW1S4).  Bound: neighbour v-W adds temporal row
# W-1+s <= 5 (row 6 is the cond row), so W + s <= XVIEW_MAX_WINDOW.  Shift 0 is "no knob":
# the flag does not accept it (a shift-0 run would resolve to the bare XW{W} folder).  S
# moves the neighbour inside the non-cond rows only; REPORT P12's cond-row (row 6) and
# mean-of-rows variants are out of scope for this flag.
XVIEW_TPOS_SHIFTS = tuple(range(1, XVIEW_MAX_WINDOW))      # 1..5
# Stage-1 supply (docs/stage1-plan.md section 3 "2단계 공급", section 6): --seeds-from swaps
# the cross-view seeds for the PNGs of a <scene>/MVSeed_<tag>/ folder in the slot P5 uses.
# MVOpt only: it is the development target and the package that carries seeds_from.py
# (the frozen packages gain no files).  Folder suffix Sd<tag>, in the Rp position.
SEEDS_FROM_ALGOS = ("MVOpt",)
# Frame-0 seed write (docs/stage1-E0.md section 3, section 6 item 3): the start_frame PNG
# is the tracker's re-prediction of the seed, a second 288-grid round trip that caps
# frame-0 J at 0.951 whatever the seed was.  --frame0-seed writes the seed itself
# (masks_spatial after --seeds-from) for that one frame; tracker state and every later
# frame are untouched.  Folder suffix Fs, after Sd<tag>.  The rule is seeds_from.select_frame0.
FRAME0_SUFFIX = "Fs"
# Small-object crop tracking (S2-R2, docs/stage2-R2-prereg.md section 1; the R1 rule of
# SCSam3/crop_track.py run inside the mainline): after the write loop, every scored
# (camera, object) whose prompt is smaller than --crop-small pixels is tracked again on
# the temporal predictor inside a square window around the prompt and, when the gate
# keeps it, its frames start+1 .. start+num_frame-1 are overwritten with the crop result.
# Folder suffix Cs<max/1000>k + G1/G2 (gate v1/v2; none: no letter), after Fs.  MVOpt
# only (the temporal predictor's request shape and masks_spatial are its).  Excludes
# --xview-mode E (the two-pass loop owns the written frames).
CROP_SMALL_ALGOS = ("MVOpt",)
CROP_GATES = ("none", "v1", "v2")               # == crop_track.GATES
CROP_GATE_DEFAULT = "v2"
CROP_SCALE_DEFAULT = 4.0
CROP_MIN_SIDE_DEFAULT = 256
CROP_GATE_SUFFIX = {"none": "", "v1": "G1", "v2": "G2"}
# Stage-1 fine-tuned memory attention in the stage-2 tracker (S2-R5, docs/stage2-R5-prereg.md):
# --ft-encoder <2D arm> loads the scene's own-fold checkpoint's memory_attention into the
# temporal tracker's encoder after the spatial model is retired and before TrackForward, and
# restores the original weights before the crop pass.  Folder suffix Ft<tag>, after Fs and
# before Cs.  MVOpt only; excludes the two-pass loop (--xview-mode E).  SCSam3/ft_encoder.py.
FT_ENCODER_ALGOS = ("MVOpt",)
# In-process stage 1 (docs/pipeline-inprocess.md): --stage1 <tag> runs the stage-1 GeoSet-2 decoder
# (stage1/, preset = a registered MVSeed_<tag> set) inside this process after build_runner and before
# the stage-2 model is built, frees it, and puts its masks into the --seeds-from slot from memory.  Same
# seeds as --seeds-from MVSeed_<tag>, so the same folder suffix Sd<tag>.  SCSam3/stage1_supply.py.
STAGE1_ALGOS = ("MVOpt",)
# S2-R6 reference-camera static snap in-process (docs/stage2-R6.md): --ref-snap applies the rule to
# the run's own output after the crop pass.  Folder suffix Rs, last.  Needs --ref-cam muvod (the rule
# is defined on c_ini, whose frame 0 is the prompt).  SCSam3/ref_snap.py.
REF_SNAP_SUFFIX = "Rs"
# S2-R7 (docs/stage2-R7-prereg.md): --ref-snap-mem also writes each online S2-R6 decision into the c_ini
# tracker memory (the frame-0 prompt re-encoded on frame t, in place).  Suffix Rm, just before Rs.
REF_SNAP_MEM_SUFFIX = "Rm"
# --save-stages: also write the intermediate results to <dataset>/<out>_stages/ (stage1/ = stage 1 in
# its CLI layout, seeds/ = the MVSeed_ folder of the seeds, stage2/ = the output before --ref-snap).
STAGES_SUFFIX = "_stages"
_SEEDS_FROM = None
_CROP_TRACK = None
_FT_ENCODER = None
_STAGE1_SUPPLY = None
_REF_SNAP = None


def _load_by_path(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage1_supply_module():
    """SCSam3/stage1_supply.py, loaded by path (torch-free at import; resolve_run needs its tag rule)."""
    global _STAGE1_SUPPLY
    if _STAGE1_SUPPLY is None:
        _STAGE1_SUPPLY = _load_by_path("stage1_supply", os.path.join(HERE, "stage1_supply.py"))
    return _STAGE1_SUPPLY


def ref_snap_module():
    global _REF_SNAP
    if _REF_SNAP is None:
        _REF_SNAP = _load_by_path("ref_snap", os.path.join(HERE, "ref_snap.py"))
    return _REF_SNAP


def seeds_from_module():
    """demoSCSam3MVOpt/seeds_from.py, loaded by path.  resolve_run needs its folder rule
    before build_runner has put any package on sys.path, and SCSam3/ itself must never
    go there (it shadows the installed sam3, see build_runner)."""
    global _SEEDS_FROM
    if _SEEDS_FROM is None:
        import importlib.util
        path = os.path.join(HERE, ALGOS["MVOpt"], "seeds_from.py")
        spec = importlib.util.spec_from_file_location("seeds_from", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SEEDS_FROM = mod
    return _SEEDS_FROM


def crop_track_module():
    """SCSam3/crop_track.py, loaded by path (as seeds_from_module: SCSam3/ never goes on
    sys.path).  Only the crop pass needs it, so a run without --crop-small never loads it."""
    global _CROP_TRACK
    if _CROP_TRACK is None:
        import importlib.util
        path = os.path.join(HERE, "crop_track.py")
        spec = importlib.util.spec_from_file_location("crop_track", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CROP_TRACK = mod
    return _CROP_TRACK


def ft_encoder_module():
    """SCSam3/ft_encoder.py, loaded by path (as crop_track_module).  resolve_run needs its
    folder rule, so it is torch-free at import."""
    global _FT_ENCODER
    if _FT_ENCODER is None:
        import importlib.util
        path = os.path.join(HERE, "ft_encoder.py")
        spec = importlib.util.spec_from_file_location("ft_encoder", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FT_ENCODER = mod
    return _FT_ENCODER


def _scale_tag(scale):
    """4.0 -> "4", 2.5 -> "2p5": a folder name may not carry a dot."""
    t = f"{float(scale):g}"
    return t.replace(".", "p")


def crop_suffix(max_px, gate, scale=CROP_SCALE_DEFAULT, min_side=CROP_MIN_SIDE_DEFAULT):
    """--crop-small folder marker: Cs<max_px/1000>k when max_px is a multiple of 1000
    (Cs2k, Cs10k), else Cs<max_px> (Cs2500), so two areas never share a name; then the
    window shape when it is not the default -- S<scale> and Ms<min_side> (S2GS2Ms128) --
    and finally G1 / G2 for gate v1 / v2 and nothing for none.  Two different windows must
    never share a folder name (S2-R4): the scale and the floor set the window just as the
    area sets which pairs are targets.  At the defaults the marker is unchanged, so every
    folder written before S2-R4 keeps its name.  Distinct from Cr<..> (crop_track.py's
    assembled folders) and from the G/G2 folders of crop_guard.py."""
    if int(max_px) <= 0:
        raise ValueError(f"max_px must be positive, got {max_px}")
    if gate not in CROP_GATES:
        raise ValueError(f"gate must be one of {CROP_GATES}, got {gate!r}")
    if float(scale) <= 0 or int(min_side) <= 0:
        raise ValueError(f"scale and min_side must be positive, got {scale}, {min_side}")
    n = f"{int(max_px) // 1000}k" if int(max_px) % 1000 == 0 else f"{int(max_px)}"
    shape = ""
    if float(scale) != float(CROP_SCALE_DEFAULT):
        shape += f"S{_scale_tag(scale)}"
    if int(min_side) != int(CROP_MIN_SIDE_DEFAULT):
        shape += f"Ms{int(min_side)}"
    return f"Cs{n}{shape}" + CROP_GATE_SUFFIX[gate]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset name as it appears in MVSeg.json")
    ap.add_argument("--algo", choices=sorted(ALGOS), default="OneStageNew")
    ap.add_argument("--out", default=None,
                    help="output folder name under the dataset (default: per algo)")
    ap.add_argument("--config", default=os.path.join(HERE, "demo", "MVSeg.json"))
    ap.add_argument("--data-root", default=os.path.join(HERE, "..", "Data", "MVSeg"))
    ap.add_argument("--device", default=None, help="cuda (default), cpu or mps")
    ap.add_argument("--track-cams", choices=["written", "all", "closure"], default=None,
                    help="open a temporal session for only the scored cameras "
                         "(written), for every camera (all), or for cameras "
                         "0..max(scored index) (closure). Default: written for "
                         "OneStage, all for OneStageNew/MVOpt, closure for XW runs. "
                         "closure on MVOpt requires --xview-window/--xview-hygiene: "
                         "the legacy gather wraps negative indices (operations.md 함정 1).")
    ap.add_argument("--xview-window", type=int, choices=range(0, XVIEW_MAX_WINDOW + 1),
                    metavar="W", default=None,
                    help="XW lineage: cross-view window W (0 = no neighbour memory), "
                         "hygiene fixes on, output SegMaskSam3XW{W} (SegMaskSam3XW{W}all "
                         "with --track-cams all), closure by default. MVOpt only. "
                         "Reading XW results: under closure+hygiene camera 0 has no "
                         "neighbour for any W, so its inputs are identical across "
                         "windows; an nb=0 delta between XW runs other than exactly 0 "
                         "signals nondeterminism or a cross-session leak, not a "
                         "control effect.")
    ap.add_argument("--xview-hygiene", action="store_true",
                    help="XW lineage with the legacy window (W=4): same as --xview-window 4")
    ap.add_argument("--xview-mode", choices=XVIEW_MODES, default=None,
                    help="XW lineage neighbourhood: A = lower views at t (default, folder "
                         "SegMaskSam3XW{W}); B = both sides at t-1; C = lower at t, upper at "
                         "t-1; D = lower at t-1; E = pass 1 as B, then recompute t from all "
                         "neighbours' pass-1 t. Folder SegMaskSam3XW{W}{mode}. Needs "
                         "--xview-window W >= 1. closure = the dependency cone "
                         "(A/D: max(scored)+1; B/C: +W per tracked frame; E: +2W).")
    ap.add_argument("--xview-gate", action="store_true",
                    help="P12 G: use a neighbour's memory token (and, with --xview-ptr, its "
                         "pointer) only if the neighbour's output passes the score test "
                         "frame_filter applies to the own memories (eff_iou_score > 0.01; "
                         "an entry carrying no score is kept). Folder suffix G.")
    ap.add_argument("--xview-ptr", action="store_true",
                    help="P12 P: append the neighbours' object pointers (4 tokens each) after "
                         "the own pointers, at the temporal position their memory token "
                         "aliases (v-k -> k frames ago). Folder suffix P.")
    ap.add_argument("--xview-tpos-shift", type=int, choices=XVIEW_TPOS_SHIFTS, metavar="S",
                    default=None,
                    help="P12 S: neighbour v-k uses temporal row k-1+S instead of k-1 (and "
                         "pointer position k+S with --xview-ptr). 1..5, W + S <= 6. Folder "
                         "suffix S<S>.")
    ap.add_argument("--ref-cam", default=None, metavar="C",
                    help="seed from this camera instead of the pick_reference rule "
                         "(the annotated camera with the largest object id at start_frame, "
                         "ties going to the first entry of cam_list). C is a camera number "
                         "from cam_list, 'center' for the middle entry of sorted(cam_list), "
                         "or 'muvod' for the dataset's c_ini in MVSeg.json (MUVOD's initial "
                         "camera, read off the published rig geometry). An auto-named output "
                         "folder gains a suffix -- M for muvod, R<rank> otherwise -- so two "
                         "references never share a folder.")
    ap.add_argument("--repair-seeds", action="store_true",
                    help="P5: after the cross-view pass, detect degenerate first-frame seeds "
                         "without ground truth and re-propagate each from a donor view in a "
                         "fresh spatial session (docs/phase5-seed-repair-prereg.md). Only "
                         "the flagged (view, obj) seeds change. Folder suffix Rp. Needs a "
                         "package with a cross-view pass (MVOpt/OneStageNew).")
    ap.add_argument("--seeds-from", default=None, metavar="MVSeed_TAG",
                    help="stage-1 supply (docs/stage1-plan.md section 3): after the cross-view "
                         "pass, replace the seed of every tracked (view, obj) that has a PNG "
                         "<dataset>/MVSeed_TAG/<cam>/<start_frame>/<obj>.png; the rest keep "
                         "their G0 seed. The folder name must start with MVSeed_ and its "
                         "SEED_MANIFEST.json, when present, must cover every scored camera. "
                         "Folder suffix Sd<TAG>; an explicit --out must carry it too. Excludes "
                         "--repair-seeds. MVOpt only.")
    ap.add_argument("--frame0-seed", action="store_true",
                    help="write the seed itself (masks_spatial, after --seeds-from) as the "
                         "start_frame PNG of every scored view instead of the tracker's "
                         "re-prediction of it (a second 288-grid round trip, docs/stage1-E0.md "
                         "section 3). An object the seed lacks keeps the tracker output; the "
                         "tracker state and every later frame are unchanged. Folder suffix Fs "
                         "(after Sd<TAG>); an explicit --out must carry it too. Excludes "
                         "--xview-mode E (its seed frame is written by the two-pass loop).")
    ap.add_argument("--crop-small", type=int, default=0, metavar="MAXPX",
                    help="S2-R2 (docs/stage2-R2-prereg.md): after the write loop, track every "
                         "scored (camera, object) whose prompt has 0 < area < MAXPX pixels "
                         "again inside a square window around the prompt (crop_track.py's "
                         "rule: side = clamp(ceil(scale * max(box w, h)), min-side, min(H, W)), "
                         "the num_frame frames, frame-0 mask prompt, forward) on the temporal "
                         "predictor, and overwrite its frames start+1 .. start+num_frame-1 "
                         "with the crop result when --crop-gate keeps it. The prompt is the "
                         "ground truth (== obj) on c_ini and masks_spatial (after --seeds-from) "
                         "elsewhere. Frame start is never touched; a reverted pair keeps the "
                         "mainline masks. 0 = off (default; same code path and bytes as before). "
                         "Folder suffix Cs<MAXPX/1000>k + G1/G2 (after Fs); an explicit --out "
                         "must carry it too. MVOpt only; excludes --xview-mode E.")
    ap.add_argument("--crop-gate", choices=CROP_GATES, default=None,
                    help=f"gate of --crop-small (default {CROP_GATE_DEFAULT}): none keeps every "
                         "target; v1 reverts a pair whose crop mask at frame start+1 has "
                         "IoU < 0.05 with the prompt; v2 also reverts when zoom = 1008 / side "
                         "< 1.2 or the mask lies on a window edge (not an image edge) in >= 10 "
                         "of the written frames (docs/stage2-R1c-prereg.md). Needs --crop-small.")
    ap.add_argument("--crop-scale", type=float, default=None, metavar="S",
                    help=f"--crop-small window: side = ceil(S * max(box w, h)) (default "
                         f"{CROP_SCALE_DEFAULT:g}). Needs --crop-small.")
    ap.add_argument("--crop-min-side", type=int, default=None, metavar="PX",
                    help=f"--crop-small window: side is at least PX (default "
                         f"{CROP_MIN_SIDE_DEFAULT}; capped at min(H, W)). Needs --crop-small.")
    ap.add_argument("--ft-encoder", default=None, metavar="ARM",
                    help="S2-R5 (docs/stage2-R5-prereg.md): run the temporal tracker with the "
                         "memory attention stage 1 fine-tuned (stage1/ geomem arm ARM, "
                         "the scene's own-fold checkpoint; memory_attention only, no 3D bias); "
                         "the original weights come back before the crop pass. Folder suffix "
                         "Ft<tag> (after Fs); an explicit --out must carry it too. MVOpt only.")
    ap.add_argument("--ft-runs", default=None, metavar="DIR",
                    help="--ft-encoder: the 2D runs folder holding <ARM>/fold<k>/ckpt.pt "
                         "(default stage1/geomem/runs in this repo)")
    ap.add_argument("--ft-folds", default=None, metavar="JSON",
                    help="--ft-encoder: the scene -> fold table (default "
                         "stage1/geomem/folds.json)")
    ap.add_argument("--stage1", default=None, metavar="TAG",
                    help="run stage 1 in this process (docs/pipeline-inprocess.md): the stage-1 GeoSet-2 "
                         "preset TAG (geoset_rep_xs = the S1-E14 headline, geoset_rep_x = S1-E11, geoset_rep = S1-E8, geoset_abs, "
                         "geoalign_rep, geoset_rep_z2/z2r, geoset_rep_xm) decodes the "
                         "reference frame from stage1/Data and its fold checkpoint, and its masks replace "
                         "the seeds exactly as --seeds-from MVSeed_TAG would, without a file. Folder "
                         "suffix Sd<TAG> (the same seeds); excludes --seeds-from and --repair-seeds. MVOpt only.")
    ap.add_argument("--ref-snap", action="store_true",
                    help="S2-R6 (docs/stage2-R6.md): after the crop pass, replace a reference-camera "
                         "object's frame-t PNG by its frame-0 PNG when IoU >= 0.8 and the image change "
                         "around it <= 4 (SCSam3/ref_snap.py). A post-process designed on the test GT: "
                         "report with LOSO +0.387. Folder suffix Rs (last). Needs --ref-cam muvod.")
    ap.add_argument("--ref-snap-mem", action="store_true",
                    help="S2-R7 (docs/stage2-R7-prereg.md): on each c_ini frame t the S2-R6 rule snaps "
                         "(decided online on the PNG just written; c_ini crop targets excluded), also "
                         "re-encode the frame-0 prompt into that frame's tracker memory, in place, before "
                         "the next camera is advanced. Needs --ref-snap and --ref-cam muvod; MVOpt, one GPU, "
                         "not two-pass. Folder suffix Rm, just before Rs.")
    ap.add_argument("--save-stages", action="store_true",
                    help="also write the intermediate results to <dataset>/<out>_stages/: stage1/ "
                         "(--stage1 outputs in the stage-1 CLI layout), seeds/ (an MVSeed_ folder of the "
                         "seeds) and stage2/ (the output before --ref-snap, hard links)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rerun even if the output folder already holds masks")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the plan, load no model")
    return ap.parse_args()


def load_config(path, name):
    with open(path) as fh:
        cfg = json.load(fh)
    if name not in cfg:
        sys.exit(f"{name!r} is not in {path}. Available: {', '.join(sorted(cfg))}")
    d = cfg[name]
    perms = d.get("perms")
    if perms is None:
        perms = list(range(d["start_cam"], d["num_cam"] + d["start_cam"]))
    return {
        "folder": d["folder"],
        "start_frame": d["start_frame"],
        "num_frame": d["num_frame"],
        "cam_list": d["cam_list"],
        "perms": perms,
        "prefix": d["prefix"],
        "prefix1": d["prefix1"],
        "c_ini": d.get("c_ini"),
    }


def cam_name(c, prefix, prefix1):
    return f"{prefix}{c:0{prefix1}d}"


def pick_reference(ds_dir, c, cv2, np):
    """The annotated camera whose ground truth at start_frame holds most objects.

    Same rule as the SAM 2 baseline: more objects in the reference frame means
    more of them get a prompt at all.
    """
    best_cam, best_n = None, -1
    for cam in c["cam_list"]:
        n = max_object_id(ds_dir, c, cam, cv2, np)
        if n > best_n:
            best_cam, best_n = cam, n
    return best_cam, best_n


def max_object_id(ds_dir, c, cam, cv2, np):
    """Largest object id in one camera's ground truth at start_frame.  Objects 1..n
    are the ones the run prompts, so this is the count the reference frame implies."""
    p = os.path.join(ds_dir, "Mask", cam_name(cam, c["prefix"], c["prefix1"]),
                     f"{c['start_frame']:06d}.png")
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"missing ground truth: {p}")
    return int(np.max(img))


class ViewAreas:
    """Per-view mask areas, recorded for later analysis.  Purely additive: nothing here
    is read back by the run, so the masks written to disk are unchanged.

    Motivated by 2026-09-09: the harm P5 propagated came from views that are NOT scored
    (Blocks cam8, CBABasketball v05), whose masks are therefore never written.  Any
    future "is this neighbour healthy" test needs their trajectories, and re-running the
    whole benchmark to get them costs a GPU sweep.  Recording the areas costs kilobytes.

      seed[view][obj]           area of the cross-view pass's first-frame mask (every
                                loaded view, not just the scored ones)
      tracked[view][frame][obj] area during temporal tracking, for every tracked view
    """

    def __init__(self):
        self.seed = {}
        self.tracked = {}

    def record_seed(self, masks_spatial):
        self.seed = {int(v): {int(o): int(m.sum().item()) for o, m in masks.items()}
                     for v, masks in masks_spatial.items()}

    def record_frame(self, view, frame_idx, obj_ids, masks):
        """masks: iterable aligned with obj_ids; each a bool tensor/array."""
        per = self.tracked.setdefault(int(view), {}).setdefault(int(frame_idx), {})
        for obj_id, m in zip(obj_ids, masks):
            n = m.sum()
            per[int(obj_id)] = int(n.item() if hasattr(n, "item") else n)

    def as_record(self):
        return {"seed": {str(v): {str(o): a for o, a in d.items()}
                         for v, d in sorted(self.seed.items())},
                "tracked": {str(v): {str(f): {str(o): a for o, a in objs.items()}
                                     for f, objs in sorted(fr.items())}
                            for v, fr in sorted(self.tracked.items())},
                "n_views_seed": len(self.seed), "n_views_tracked": len(self.tracked)}


def resolve_ref_cam(spec, cam_list, c_ini=None):
    """--ref-cam -> (camera number, folder suffix).  (None, None) when the flag is
    absent, which means main() falls back to pick_reference.  Pure: no image reads, so
    resolve_run can name the folder before the ground truth is opened.

    muvod  the dataset's c_ini from MVSeg.json -- MUVOD's initial camera, read off the
           published rig geometry per scene (docs/muvod-protocol.md).  Suffix M, the
           same for every scene, so one method name spans the benchmark.
    center the middle entry of sorted(cam_list).  Suffix R<rank>.
    <n>    that camera number.  Suffix R<rank>.
    """
    if spec is None:
        return None, None
    ordered = sorted(cam_list)
    if spec == "muvod":
        if c_ini is None:
            sys.exit("--ref-cam muvod needs a \"c_ini\" entry for this dataset in MVSeg.json")
        if c_ini not in ordered:
            sys.exit(f"c_ini {c_ini} is not in this dataset's cam_list {ordered}")
        return c_ini, "M"
    if spec == "center":
        cam = ordered[(len(ordered) - 1) // 2]     # 3 cameras -> the middle one
    else:
        try:
            cam = int(spec)
        except ValueError:
            sys.exit("--ref-cam takes a camera number from cam_list, 'center' or "
                     f"'muvod', not {spec!r}")
        if cam not in ordered:
            sys.exit(f"--ref-cam {cam} is not in this dataset's cam_list {ordered}")
    return cam, f"R{ordered.index(cam)}"


def closure_reach(mode, window, max_scored, n_cams, num_frame):
    """Number of sessions 0..reach-1 that give the scored views exactly their `all`
    outputs.  A/D read lower views only: max_scored+1.  B/C read (v+k, t-1), k <= W,
    whose memory came from (v+2k, t-2), ... down to the seed: the cone widens by W per
    tracked frame.  E has two links per frame (pass 2 reads pass-1 t of v+-W, which read
    t-1 of v+-2W): 2W.  num_frame-1 frames are tracked after the seed."""
    growth = XVIEW_CLOSURE_GROWTH[mode]
    if growth == 0:
        return max_scored + 1
    if num_frame is None:
        sys.exit(f"--track-cams closure with --xview-mode {mode} needs the dataset's num_frame")
    return min(n_cams, max_scored + 1 + growth * window * (num_frame - 1))


def resolve_run(args, cam_names, written_cams, num_frame=None, ref_suffix=None):
    """Everything main() decides from the flags: output folder, track mode, session
    list and the tracker kwargs.  Legacy invocations resolve exactly as before this
    function existed; every new combination is explicit or refused."""
    scored_idx = [i for i, n in enumerate(cam_names) if n in written_cams]
    xview_on = args.xview_window is not None or args.xview_hygiene
    mode = getattr(args, "xview_mode", None)          # None -> "A" (the Phase 2 gather)
    if mode is not None and not xview_on:
        sys.exit("--xview-mode needs an XW flag (--xview-window W, or --xview-hygiene for W=4)")
    gate = bool(getattr(args, "xview_gate", False))
    ptr = bool(getattr(args, "xview_ptr", False))
    shift = getattr(args, "xview_tpos_shift", None) or 0     # None / 0 -> no S knob
    knobs_on = gate or ptr or bool(shift)
    if knobs_on and not xview_on:
        sys.exit("--xview-gate/--xview-ptr/--xview-tpos-shift need an XW flag "
                 "(--xview-window W, or --xview-hygiene for W=4)")
    if xview_on and args.algo not in XVIEW_ALGOS:
        sys.exit(f"--xview-window/--xview-hygiene are wired into {XVIEW_ALGOS} only "
                 f"(OneStage has no cross-view memory, OneStageNew is frozen)")
    if not xview_on:
        stray = [k for k in XVIEW_ENV if os.environ.get(k, "").strip()]
        if stray:
            sys.exit(f"{stray} set in the environment but no --xview flag given: the runner "
                     "takes the cross-view configuration from the command line only "
                     "(unset the variable, or pass --xview-window)")
    window = None
    xview_kwargs = {}
    lineage = "legacy"
    mode_eff = XVIEW_LEGACY_MODE
    knob_suffix = ""
    if xview_on:
        window = 4 if args.xview_window is None else args.xview_window
        xview_kwargs = dict(cross_view_window=window, cross_view_hygiene=True)
        if mode is not None:                          # an explicit letter reaches the tracker
            xview_kwargs["cross_view_mode"] = mode
            mode_eff = mode
        if mode_eff != XVIEW_LEGACY_MODE and window == 0:
            sys.exit(f"--xview-mode {mode_eff} with --xview-window 0 reads no neighbour; W=0 is "
                     "the control XW0 (drop --xview-mode)")
        if knobs_on and window == 0:
            sys.exit("--xview-gate/--xview-ptr/--xview-tpos-shift act on neighbour tokens and "
                     "--xview-window 0 has none (W=0 is the control XW0)")
        if shift and window + shift > XVIEW_MAX_WINDOW:
            sys.exit(f"--xview-window {window} + --xview-tpos-shift {shift} > {XVIEW_MAX_WINDOW}: "
                     f"neighbour v-{window} would use temporal row {window - 1 + shift}, but row "
                     f"{XVIEW_MAX_WINDOW} is the cond-frame row (W + S <= {XVIEW_MAX_WINDOW})")
        if gate:                                      # only a given knob reaches the tracker
            xview_kwargs["cross_view_gate"] = True
        if ptr:
            xview_kwargs["cross_view_ptr"] = True
        if shift:
            xview_kwargs["cross_view_tpos_shift"] = shift
        knob_suffix = ("G" if gate else "") + ("P" if ptr else "") + (f"S{shift}" if shift else "")
        lineage = f"XW{window}" + ("" if mode_eff == XVIEW_LEGACY_MODE else mode_eff) + knob_suffix

    if args.track_cams is not None:
        track_mode = args.track_cams
    elif xview_on:
        track_mode = "closure"
    else:
        track_mode = "all" if args.algo in NEEDS_ALL_VIEWS else "written"
    if track_mode == "closure" and args.algo in NEEDS_ALL_VIEWS and not xview_on:
        sys.exit("--track-cams closure on a NEEDS_ALL_VIEWS package needs --xview-window "
                 "or --xview-hygiene: the legacy gather wraps negative indices, so closure "
                 "would be a different model (operations.md 함정 1, REPORT C1/C2)")
    if xview_on and track_mode == "written":
        sys.exit("XW runs define neighbours by camera index; --track-cams written would "
                 "renumber them (operations.md 함정 1). Use closure (default) or all.")
    # Session list, computed per mode: the legacy modes never look at max(scored_idx),
    # so an empty cam_list resolves exactly as it always did (all -> every camera,
    # written -> no session).  Closure is the only mode that needs a scored camera.
    if track_mode == "all":
        track_idx = list(range(len(cam_names)))
    elif track_mode == "written":
        track_idx = scored_idx
    else:
        if not scored_idx:
            sys.exit("--track-cams closure needs at least one scored camera (cam_list) "
                     f"among the loaded cameras, but none of {written_cams} is in "
                     f"{cam_names}")
        track_idx = list(range(closure_reach(mode_eff, window, max(scored_idx), len(cam_names), num_frame)))

    if args.out:
        out_name = args.out
    elif xview_on:
        # closure is the XW default and keeps the bare name; an `all` run is a
        # different session set (experiment row 6 compares the two), so it gets its own
        out_name = (f"{XVIEW_OUT_PREFIX}{window}" + ("" if mode_eff == XVIEW_LEGACY_MODE else mode_eff)
                    + knob_suffix + ("all" if track_mode == "all" else ""))
    elif track_mode == "closure":            # OneStage only: the guard above excludes the rest
        out_name = ONESTAGE_CLOSURE_OUT
    else:
        out_name = DEFAULT_OUT[args.algo]
    if not args.out and ref_suffix:
        # a different seed camera is a different run: never let it land on the folder
        # the default reference wrote
        out_name += ref_suffix
    repair = bool(getattr(args, "repair_seeds", False))
    if repair and args.algo not in NEEDS_ALL_VIEWS:
        sys.exit(f"--repair-seeds needs a package with a cross-view pass "
                 f"({', '.join(NEEDS_ALL_VIEWS)}), not {args.algo}")
    if not args.out and repair:
        out_name += "Rp"
    seeds_from = getattr(args, "seeds_from", None)
    stage1 = getattr(args, "stage1", None)
    if stage1 is not None:
        if seeds_from is not None:
            sys.exit("--stage1 and --seeds-from both supply the seeds: pass one of them")
        if repair:
            sys.exit("--stage1 and --repair-seeds both rewrite the first-frame seeds and "
                     "are different treatments: pass one of them")
        if args.algo not in STAGE1_ALGOS:
            sys.exit(f"--stage1 is wired into {STAGE1_ALGOS} only, not {args.algo}")
        try:
            seed_suffix = stage1_supply_module().folder_suffix(stage1)
        except ValueError as exc:
            sys.exit(f"--stage1: {exc}")
        if not args.out:
            out_name += seed_suffix
        elif seed_suffix not in out_name:
            sys.exit(f"refusing to write a --stage1 run into {out_name}: the name must "
                     f"carry {seed_suffix} (a control folder has no seed marker)")
    if seeds_from is not None:
        if repair:
            sys.exit("--seeds-from and --repair-seeds both rewrite the first-frame seeds and "
                     "are different treatments: pass one of them")
        if args.algo not in SEEDS_FROM_ALGOS:
            sys.exit(f"--seeds-from is wired into {SEEDS_FROM_ALGOS} only, not {args.algo}")
        try:
            seed_suffix = seeds_from_module().folder_suffix(seeds_from)
        except ValueError as exc:
            sys.exit(f"--seeds-from: {exc}")
        # Rp position: the treatment must never land on the folder its control wrote,
        # so even a user-chosen --out has to carry the marker
        if not args.out:
            out_name += seed_suffix
        elif seed_suffix not in out_name:
            sys.exit(f"refusing to write a --seeds-from run into {out_name}: the name must "
                     f"carry {seed_suffix} (a control folder has no seed marker)")
    two_pass = bool(xview_on and mode_eff in XVIEW_TWO_PASS)
    frame0_seed = bool(getattr(args, "frame0_seed", False))
    if frame0_seed:
        if two_pass:
            sys.exit(f"--frame0-seed and --xview-mode {mode_eff} do not combine: the two-pass "
                     "loop writes its own seed frame (run_two_pass)")
        # after Sd<tag>, same rule: a frame-0 seed run must never land on the folder its
        # control (the tracker's re-prediction) wrote, so an explicit --out carries Fs too
        if not args.out:
            out_name += FRAME0_SUFFIX
        elif FRAME0_SUFFIX not in out_name:
            sys.exit(f"refusing to write a --frame0-seed run into {out_name}: the name must "
                     f"carry {FRAME0_SUFFIX} (a control folder has no frame-0 marker)")
    # --ft-encoder: after Fs, before Cs (Sd -> Fs -> Ft -> Cs); same never-overwrite rule
    ft_encoder = getattr(args, "ft_encoder", None)
    if ft_encoder is not None:
        if args.algo not in FT_ENCODER_ALGOS:
            sys.exit(f"--ft-encoder is wired into {FT_ENCODER_ALGOS} only, not {args.algo}")
        if two_pass:
            sys.exit(f"--ft-encoder and --xview-mode {mode_eff} do not combine: the swap and the "
                     "restore before the crop pass are the one-pass loop's")
        try:
            ft_suffix = ft_encoder_module().folder_suffix(ft_encoder)
        except ValueError as exc:
            sys.exit(f"--ft-encoder: {exc}")
        if not args.out:
            out_name += ft_suffix
        elif ft_suffix not in out_name:
            sys.exit(f"refusing to write an --ft-encoder run into {out_name}: the name must "
                     f"carry {ft_suffix} (a control folder has no fine-tuned-encoder marker)")
    elif getattr(args, "ft_runs", None) is not None or getattr(args, "ft_folds", None) is not None:
        sys.exit("--ft-runs / --ft-folds need --ft-encoder ARM")
    # --crop-small: after Fs (Rp/Sd -> Fs -> Cs).  crop = None when off, so main() and the
    # manifest see one value; the settings are resolved here so the folder name, the run
    # and the manifest all carry the same numbers.
    crop = None
    crop_px = getattr(args, "crop_small", 0) or 0
    crop_gate = getattr(args, "crop_gate", None)
    crop_scale = getattr(args, "crop_scale", None)
    crop_min_side = getattr(args, "crop_min_side", None)
    if crop_px < 0:
        sys.exit(f"--crop-small takes a pixel area (0 = off), not {crop_px}")
    if not crop_px:
        stray = [f for f, v in (("--crop-gate", crop_gate), ("--crop-scale", crop_scale),
                                ("--crop-min-side", crop_min_side)) if v is not None]
        if stray:
            sys.exit(f"{stray} need --crop-small MAXPX (0 = off: the crop pass does not run)")
    else:
        if args.algo not in CROP_SMALL_ALGOS:
            sys.exit(f"--crop-small is wired into {CROP_SMALL_ALGOS} only, not {args.algo}")
        if two_pass:
            sys.exit(f"--crop-small and --xview-mode {mode_eff} do not combine: the two-pass "
                     "loop owns the written frames (run_two_pass)")
        # local names must not shadow `gate` (the P12 xview gate above)
        c_gate = CROP_GATE_DEFAULT if crop_gate is None else crop_gate
        if c_gate not in CROP_GATES:
            sys.exit(f"--crop-gate must be one of {CROP_GATES}, not {c_gate!r}")
        c_scale = CROP_SCALE_DEFAULT if crop_scale is None else float(crop_scale)
        c_min_side = CROP_MIN_SIDE_DEFAULT if crop_min_side is None else int(crop_min_side)
        if c_scale <= 0 or c_min_side <= 0:
            sys.exit("--crop-scale and --crop-min-side must be positive")
        c_suffix = crop_suffix(crop_px, c_gate, c_scale, c_min_side)
        # same rule as Sd/Fs: a crop run must never land on the folder its control wrote
        if not args.out:
            out_name += c_suffix
        elif c_suffix not in out_name:
            sys.exit(f"refusing to write a --crop-small run into {out_name}: the name must "
                     f"carry {c_suffix} (a control folder has no crop marker)")
        crop = dict(max_px=int(crop_px), gate=c_gate, scale=c_scale, min_side=c_min_side,
                    suffix=c_suffix)
    ref_snap = bool(getattr(args, "ref_snap", False))
    ref_snap_mem = bool(getattr(args, "ref_snap_mem", False))
    if ref_snap_mem:
        if not ref_snap:
            sys.exit("--ref-snap-mem feeds the --ref-snap decisions into the memory: pass --ref-snap")
        if getattr(args, "algo", None) != "MVOpt":
            sys.exit("--ref-snap-mem is MVOpt only")
        if two_pass:
            sys.exit(f"--ref-snap-mem and --xview-mode {mode_eff} do not combine (two-pass loop)")
    if ref_snap:
        if getattr(args, "ref_cam", None) != "muvod":
            sys.exit("--ref-snap is defined on MUVOD's c_ini (its frame 0 is the prompt): "
                     "pass --ref-cam muvod")
        if not args.out:
            out_name += (REF_SNAP_MEM_SUFFIX if ref_snap_mem else "") + REF_SNAP_SUFFIX
        elif REF_SNAP_SUFFIX not in out_name:
            sys.exit(f"refusing to write a --ref-snap run into {out_name}: the name must "
                     f"carry {REF_SNAP_SUFFIX}")
        elif ref_snap_mem and REF_SNAP_MEM_SUFFIX + REF_SNAP_SUFFIX not in out_name:
            sys.exit(f"refusing to write a --ref-snap-mem run into {out_name}: the name must "
                     f"carry {REF_SNAP_MEM_SUFFIX + REF_SNAP_SUFFIX}")
    if xview_on and out_name in (*DEFAULT_OUT.values(), ONESTAGE_CLOSURE_OUT):
        sys.exit(f"refusing to write an XW run into {out_name}: that folder is the published "
                 "legacy lineage (pass --out SegMaskSam3XW...)")
    if not xview_on and out_name.startswith(XVIEW_OUT_PREFIX):
        sys.exit(f"refusing to write a legacy run into {out_name}: folder names starting "
                 f"with {XVIEW_OUT_PREFIX} are the XW lineage (pass --xview-window, or "
                 "another --out)")
    return dict(out_name=out_name, repair_seeds=repair, seeds_from=seeds_from, stage1=stage1,
                ref_snap=ref_snap, ref_snap_mem=ref_snap_mem, save_stages=bool(getattr(args, "save_stages", False)),
                track_mode=track_mode, track_idx=track_idx,
                scored_idx=scored_idx, xview_on=xview_on, xview_window=window,
                xview_kwargs=xview_kwargs, lineage=lineage,
                xview_mode=mode, xview_mode_eff=(mode_eff if xview_on else None),
                two_pass=two_pass, frame0_seed=frame0_seed, crop=crop, ft_encoder=ft_encoder,
                closure_reach=(len(track_idx) if track_mode == "closure" else None),
                xview_gate=gate, xview_ptr=ptr, xview_tpos_shift=shift)


def ref_snap_mem_targets(gt0, c_ini_cam, obj_ids, crop, ct):
    """S2-R7 T (docs/stage2-R7-prereg.md section 4): the c_ini crop targets exactly as CropSmall selects them --
    crop_track.select_targets_from with the c_ini prompt = the frame-start GT == obj, the run's max_px / scale /
    min_side -- over the prompted objects; [] when the crop pass is off."""
    if not crop:
        return []
    targets, _, _ = ct.select_targets_from([c_ini_cam], [int(o) for o in obj_ids],
                                           lambda cam, obj: (gt0, obj, obj, ct.PROMPT_GT), crop["max_px"],
                                           scale=crop["scale"], min_side=crop["min_side"])
    return sorted(int(x["obj"]) for x in targets)


def inject_ref_memory(predictor, session_id, t, objs_masks0, torch):
    """S2-R7 (docs/stage2-R7-prereg.md section 4): for each (obj, frame-0 prompt mask) the c_ini session snapped
    at frame t, re-encode the prompt on frame t's cached image features (the frame-0 prompt's route: video res ->
    288 bilinear antialias -> x20-10 -> 1008 bilinear, is_mask_from_pts=True) with the stored entry's object
    score, and copy it in place into that object's non_cond_frame_outputs[t]['maskmem_features'] (the per-object
    slices share its storage).  obj_ptr, pos enc, scores, pred_masks and every cond entry stay as they are.
    -> one record per object: obj, cache_hit, obj_score, rel_change (or error)."""
    import numpy as np
    import torch.nn.functional as F
    model = predictor.model
    tracker = model.tracker
    state = predictor._ALL_INFERENCE_STATES[session_id]["state"]
    recs = []
    with torch.inference_mode():
        for obj, m0 in objs_masks0:
            rec = {"obj": int(obj)}
            tss = model._get_tracker_inference_states_by_obj_ids(state, [int(obj)])
            e = tss[0]["output_dict"]["non_cond_frame_outputs"].get(t) if tss else None
            if e is None:
                rec["error"] = "no tracker state" if not tss else "no non-cond entry"
                recs.append(rec)
                continue
            ts = tss[0]
            rec["cache_hit"] = t in ts["cached_features"]
            image, _, feats, _, sizes = tracker._get_image_feature(ts, t, 1)
            dev = feats[-1].device
            m0t = torch.from_numpy(np.asarray(m0, dtype=np.float32))[None, None].to(dev)
            m288 = F.interpolate(m0t, size=(tracker.low_res_mask_size, tracker.low_res_mask_size),
                                 mode="bilinear", align_corners=False, antialias=True) * 20.0 - 10.0
            hr = F.interpolate(m288, size=(tracker.image_size, tracker.image_size), mode="bilinear",
                               align_corners=False)
            score = e["object_score_logits"].to(dev)
            mf, _ = tracker._encode_new_memory(image=image, current_vision_feats=feats, feat_sizes=sizes,
                                               pred_masks_high_res=hr, object_score_logits=score,
                                               is_mask_from_pts=True)
            stored = e["maskmem_features"]
            mf = mf.to(stored.dtype).to(stored.device)
            rec["obj_score"] = float(score.float().mean())
            rec["rel_change"] = float((mf.float() - stored.float()).norm() / stored.float().norm())
            stored.copy_(mf)
            recs.append(rec)
    return recs


def build_runner(algo):
    """Import the selected demo's model code and extend it for MVSeg."""
    algo_dir = os.path.join(HERE, ALGOS[algo])

    # SCSam3/sam3 is the upstream checkout: a directory with no __init__.py.
    # While this file's own directory is on sys.path it shadows the installed
    # sam3 package as a namespace package, whose __file__ is None, and the
    # model builder dies looking up the tokenizer inside it.  The demos do not
    # hit this because they run from inside their own folder.
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != HERE]
    sys.path.insert(0, algo_dir)
    os.chdir(algo_dir)          # the demo modules resolve some paths relatively

    import torch
    from SCSam3Video import SCSam3Video
    from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU

    class MVSegVideo(SCSam3Video):
        """SCSam3Video with the pieces MVSeg needs.

        The base class only knows how to load a folder of MP4s and always
        prompts on frame 0; MVSeg is folders of JPEGs and starts at an
        arbitrary frame.  LoadVideo_Folder_MVSeg and AddMaskSingle in the base
        class still target the old SAM 2 predictor API and would raise, so the
        two methods below replace them rather than calling into them.
        """

        @property
        def spatial(self):
            # OneStageNew keeps a separate predictor for the cross-view pass
            return getattr(self, "predictor_spatial", self.predictor)

        def LoadCameraFolders(self, video_root, cam_names, start_frame, track_idx):
            """Frame readers for every camera, temporal sessions for `track_idx`.

            Every camera contributes its `start_frame` to the cross-view pass,
            but only the cameras we actually score need a tracking session, and
            a session per camera is what runs the GPU out of memory on the
            45-camera scenes.  The readers are lazy, so the cameras we skip cost
            one decoded frame each.
            """
            self.track_views = list(track_idx)
            self.start_frame = start_frame          # RepairSeeds re-opens the same frames
            track_set = set(self.track_views)
            for i, name in enumerate(cam_names):
                folder = os.path.join(video_root, name)
                if not os.path.isdir(folder):
                    raise FileNotFoundError(folder)
                cpu_image, height, width = load_video_frames(video_path=folder)
                image = AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)
                self.images.append(image)
                self.cpu_images.append(cpu_image)
                if i not in track_set:
                    continue
                response = self.predictor.handle_request(
                    request=dict(type="start_session", images=image,
                                 orig_height=height, orig_width=width))
                session_id = response["session_id"]
                self.session_ids.append(session_id)
                self.predictor.handle_request(
                    request=dict(type="reset_session", session_id=session_id))

            self.numImage = len(cam_names)
            self.video_height, self.video_width = height, width

            # the cross-view "video" is frame `start_frame` of every camera
            frames = [self.images[i][start_frame] for i in range(self.numImage)]
            response = self.spatial.handle_request(
                request=dict(type="start_session", images=frames,
                             orig_height=height, orig_width=width))
            self.session_id_statial = response["session_id"]
            self.spatial.handle_request(
                request=dict(type="reset_session",
                             session_id=self.session_id_statial))

        def AddReferenceMask(self, view_index, mask, obj_id):
            """One binary ground-truth mask as the prompt for one object."""
            response = self.spatial.handle_request(
                request=dict(type="add_prompt",
                             session_id=self.session_id_statial,
                             frame_index=view_index,
                             mask=torch.tensor(mask, dtype=torch.float32),
                             obj_id=obj_id))
            self.obj_ids = response["outputs"]["out_obj_ids"].tolist()

        def PropagateAcrossViews(self, view_index):
            """Both directions, so views on either side of the reference are filled.

            The base class passes `start_frame_idx`, which the predictor does
            not read - it looks for `start_frame_index` - so the start view is
            spelled out here.
            """
            request = dict(type="propagate_in_video",
                           session_id=self.session_id_statial,
                           propagation_direction="both")
            if not SPATIAL_START_IMPLICIT:
                request["start_frame_index"] = view_index
            responses = self.spatial.handle_stream_request(request=request)
            for response in responses:
                out = response["outputs"]
                self.masks_spatial[response["frame_index"]] = {
                    obj_id: (out["out_binary_masks"][i] > 0.0)
                    for i, obj_id in enumerate(out["out_obj_ids"].tolist())
                }

        def RepairSeeds(self, ref_index, ref_gt, n_obj, max_rounds=None):
            """P5 (docs/phase5-seed-repair-prereg.md section 1).

            Detect degenerate cross-view seeds without ground truth, re-propagate each
            from a donor view in a fresh spatial session, and replace only the flagged
            (view, obj) entries of masks_spatial.  Must run after PropagateAcrossViews
            and before RetireSpatialPredictor (tests/test_seed_repair_prereq.py pins the
            order: after retirement this would silently repair nothing).
            """
            import seed_repair
            rounds = seed_repair.MAX_ROUNDS if max_rounds is None else max_rounds
            ref_areas = {obj: int((ref_gt == obj).sum()) for obj in range(1, n_obj + 1)}
            # Amendment (prereg section 8): only the tracked views are examined -- their
            # seeds are the only ones TrackForward consumes -- and only tracked views
            # other than the reference may donate.  On a 46-camera hemisphere the far
            # views hold blobs and blanks that poison any per-view statistic.
            views = sorted(self.track_views)
            donors = [v for v in views if v != ref_index]
            stats = {"params": dict(min_px=seed_repair.MIN_PX, rel_frac=seed_repair.REL_FRAC,
                                    donor_lo=seed_repair.DONOR_LO, donor_hi=seed_repair.DONOR_HI,
                                    window=seed_repair.WINDOW, fallback=seed_repair.FALLBACK,
                                    max_rounds=rounds),
                     "ref_view": ref_index, "ref_areas": ref_areas,
                     "views": views, "donors": donors, "n_obj": n_obj, "rounds": []}
            used = {}
            for rnd in range(1, rounds + 1):
                areas = self._seed_areas()
                repairs, unrep = seed_repair.plan(areas, ref_areas, ref_index, views=views,
                                                  donors=donors, used=used)
                rec = {"round": rnd, "flagged": len(repairs) + len(unrep),
                       "repairs": seed_repair.as_records(repairs),
                       "unrepairable": seed_repair.as_records(unrep), "after": []}
                stats["rounds"].append(rec)
                if not repairs:
                    break
                new = self._repropagate(ref_index, ref_gt, n_obj, repairs)
                for r in repairs:
                    # a donor counts as tried whether or not the session produced a
                    # mask for the pair (prereg 1.3 step 5: "a donor not yet used");
                    # otherwise round 2 repeats round 1 for pairs the model will not
                    # output at that view at all
                    used.setdefault((r.view, r.obj), set()).add(r.donor)
                    mask = new.get(r.view, {}).get(r.obj)
                    if mask is None:
                        rec["after"].append(dict(view=r.view, obj=r.obj, area=None))
                        continue
                    self.masks_spatial[r.view][r.obj] = mask
                    rec["after"].append(dict(view=r.view, obj=r.obj,
                                             area=int(mask.sum().item())))
            stats["used_donors"] = [dict(view=v, obj=o, donors=sorted(d))
                                    for (v, o), d in sorted(used.items())]
            return stats

        def LoadSeedsFrom(self, folder, start_frame, cam_names, track_idx):
            """Stage-1 supply (docs/stage1-plan.md section 3 "2단계 공급").

            Replace the seed of every tracked (view, obj) that has a PNG in the MVSeed_
            folder; the rest keep the G0 seed PropagateAcrossViews just produced, so
            obj_ids and the fallback come from the model as always.  Same slot as
            RepairSeeds -- after PropagateAcrossViews, before RetireSpatialPredictor,
            pinned by tests/test_seed_repair_prereq.py -- not because the spatial model
            is needed (it is not) but because record_seed and TrackForward read
            masks_spatial right after, and one slot keeps "seed" meaning one thing.
            The PNGs come back as the bool (H, W) arrays the pass leaves behind, so
            nothing downstream sees a different type.
            """
            import cv2
            sf = seeds_from_module()
            report = sf.apply_seed_folder(
                self.masks_spatial, folder, start_frame, cam_names, track_idx, self.obj_ids,
                shape=(self.video_height, self.video_width),
                read_png=lambda p: cv2.imread(p, cv2.IMREAD_GRAYSCALE))
            report.update(sf.folder_provenance(folder))
            return report

        def LoadSeeds(self, seeds, cam_names, track_idx):
            """--stage1: LoadSeedsFrom for seeds held in memory ({cam: {obj: bool (H, W)}}), same
            slot, same rules (seeds_from.apply_seeds)."""
            return seeds_from_module().apply_seeds(
                self.masks_spatial, seeds, cam_names, track_idx, self.obj_ids,
                shape=(self.video_height, self.video_width))

        def _seed_areas(self):
            return {view: {obj: int(m.sum().item()) for obj, m in masks.items()}
                    for view, masks in self.masks_spatial.items()}

        def _repropagate(self, ref_index, ref_gt, n_obj, repairs):
            """One fresh spatial session for this round.  EVERY object is prompted at
            the reference exactly as in PropagateAcrossViews (so the per-frame overlap
            competition is the one TrackForward will see -- a session holding only the
            flagged objects let a repaired seed grab another object's pixels), plus the
            donor seed of each flagged object as a second conditioning frame.  Both
            directions, implicit start.  Returns {view: {obj: mask}} for every yielded
            view; the caller commits only the flagged pairs."""
            # The pass-1 session is dead weight by now: masks_spatial holds fresh bool
            # tensors, not views into its state.  Close it BEFORE opening the repair
            # session -- two N-view sessions side by side doubled host RAM and got the
            # 17-scene sweep killed on AlexaMeadeExhibit (45 views).  RetireSpatial-
            # Predictor tolerates the None it finds afterwards.
            old_sid = getattr(self, "session_id_statial", None)
            if old_sid is not None:
                self.spatial.handle_request(request=dict(type="close_session",
                                                         session_id=old_sid))
                self.session_id_statial = None
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            frames = [self.images[i][self.start_frame] for i in range(self.numImage)]
            response = self.spatial.handle_request(
                request=dict(type="start_session", images=frames,
                             orig_height=self.video_height, orig_width=self.video_width))
            sid = response["session_id"]
            self.spatial.handle_request(request=dict(type="reset_session", session_id=sid))
            try:
                for obj in range(1, n_obj + 1):          # same set as AddReferenceMask
                    self.spatial.handle_request(request=dict(
                        type="add_prompt", session_id=sid, frame_index=ref_index,
                        mask=torch.tensor((ref_gt == obj).astype("float32"),
                                          dtype=torch.float32),
                        obj_id=obj))
                for donor, obj in sorted({(r.donor, r.obj) for r in repairs}):
                    self.spatial.handle_request(request=dict(
                        type="add_prompt", session_id=sid, frame_index=donor,
                        mask=torch.tensor(self.masks_spatial[donor][obj],
                                          dtype=torch.float32),
                        obj_id=obj))
                request = dict(type="propagate_in_video", session_id=sid,
                               propagation_direction="both")
                if not SPATIAL_START_IMPLICIT:
                    request["start_frame_index"] = ref_index
                out = {}
                for response in self.spatial.handle_stream_request(request=request):
                    o = response["outputs"]
                    out[response["frame_index"]] = {
                        obj_id: (o["out_binary_masks"][i] > 0.0)
                        for i, obj_id in enumerate(o["out_obj_ids"].tolist())}
                return out
            finally:
                self.spatial.handle_request(request=dict(type="close_session", session_id=sid))

        def InjectRefMemory(self, j, t, objs_masks0):
            """S2-R7: inject_ref_memory on c_ini session j (docs/stage2-R7-prereg.md section 4)."""
            return inject_ref_memory(self.predictor, self.session_ids[j], t, objs_masks0, torch)

        def TrackForward(self, start_frame, num_frame):
            """Seed each view with its propagated mask, then track forward only.

            Forward only, and capped at num_frame: MVSeg annotates
            start_frame .. start_frame + num_frame - 1, and letting it run
            backward would track hundreds of frames nobody scores.
            """
            zero = torch.zeros((self.video_height, self.video_width),
                               dtype=torch.float32)
            for j, view in enumerate(self.track_views):
                for obj_id in self.obj_ids:
                    mask = self.masks_spatial[view].get(obj_id)
                    self.predictor.handle_request(
                        request=dict(type="add_prompt",
                                     session_id=self.session_ids[j],
                                     frame_index=start_frame,
                                     mask=(torch.tensor(mask, dtype=torch.float32)
                                           if mask is not None else zero),
                                     obj_id=obj_id))

            self.tracking_result = [None] * len(self.track_views)
            for m in range(len(self.track_views)):
                request = dict(type="propagate_in_video",
                               propagation_direction="forward",
                               start_frame_index=start_frame,
                               max_frame_num_to_track=num_frame)
                # OneStageNew's tracker reads every session to share memory
                # across views; OneStage tracks one session at a time.
                # Test the flag, not the attribute: the cross-view model is
                # retired before this point, so its presence says nothing.
                # The attribute check is the fallback for packages that do not
                # set the flag. Since 2026-09-04 both OneStageNew and MVOpt set
                # it; OneStage sets neither and takes the single-session branch,
                # because its __init__ never builds predictor_spatial at all.
                if getattr(self, "uses_spatial_predictor",
                           getattr(self, "predictor_spatial", None) is not None):
                    request["session_ids"] = self.session_ids
                    request["spatial_idx"] = m
                else:
                    request["session_id"] = self.session_ids[m]
                self.tracking_result[m] = self.predictor.handle_stream_request(
                    request=request)

        def RecomputeFrame(self, frame_idx, output_for=None):
            """XW mode E pass 2: recompute frame_idx in every session from the pass-1
            memories (compute all, then commit, inside the predictor).  Returns the
            per-session outputs list; None where `output_for` excludes the session."""
            response = self.predictor.handle_request(
                request=dict(type="recompute_frame", session_ids=self.session_ids,
                             frame_index=frame_idx, output_for=output_for))
            return response["outputs"]

        def CropSmall(self, ds_dir, out_dir, cam_names, written_cams, c_ini_cam, start_frame,
                      num_frame, max_px, gate, scale, min_side, cv2, np):
            """S2-R2 (docs/stage2-R2-prereg.md section 1): crop-track the small objects of
            the scored cameras and overwrite their frames start+1 .. start+num_frame-1.

            Runs after the write loop: every mainline mask is on disk and the temporal
            sessions have yielded their last frame.  Nothing here reads a neighbour memory
            and no crop session writes into a mainline session (the R1 definition: the
            crop result does not flow into the neighbour memories either -- they are done).
            The target rule, window, tracking loop and paste-back are crop_track.py's; the
            gate is crop_guard.py's, as crop_track.gate_verdict:
              prompt   c_ini: the ground truth (== obj); other cameras:
                       masks_spatial[view][obj] as TrackForward consumed it (after
                       --seeds-from) -- the frame-0 PNG of a --frame0-seed run
              target   0 < prompt area < max_px, over self.obj_ids
              window   side = clamp(ceil(scale * max(box w, h)), min_side, min(H, W)), square
                       centred on the prompt box, pushed inside the image; the same window
                       for the num_frame frames start .. start+num_frame-1, read from the
                       original JPGs
              session  one per target on self.predictor: start_session over the crops,
                       reset, frame-0 mask prompt, forward propagation capped at num_frame,
                       close_session; session_ids=[sid] so the cross-view gather sees no
                       neighbour
              gate     none / v1 (IoU(crop frame 1, prompt) < 0.05) / v2 (v1, zoom < 1.2,
                       window-edge frames >= 10); a reverted pair keeps the mainline PNGs
              write    kept pairs: frames start+1 .. start+num_frame-1 as the crop mask
                       pasted into a full canvas (0 outside the window), the mainline's
                       PNG encoding; frame start is never touched
            Returns the "crop_small" block of MANIFEST.json (plain JSON types).
            """
            ct = crop_track_module()
            t_start = time.time()
            H, W = int(self.video_height), int(self.video_width)
            views = {name: i for i, name in enumerate(cam_names)}
            cams = [cam for cam in written_cams if cam in views]
            objs = [int(o) for o in self.obj_ids]
            gt = None
            if c_ini_cam is not None and c_ini_cam in cams:
                gt_path = os.path.join(ds_dir, "Mask", c_ini_cam, f"{start_frame:06d}.png")
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    raise FileNotFoundError(gt_path)
                if gt.shape != (H, W):
                    raise ValueError(f"{gt_path}: {gt.shape[1]}x{gt.shape[0]} but the video "
                                     f"is {W}x{H}")
            seeds = {}                                 # (cam, obj) -> bool (H, W)

            def prompt(cam, obj):
                if gt is not None and cam == c_ini_cam:
                    return gt, obj, obj, ct.PROMPT_GT
                m = self.masks_spatial.get(views[cam], {}).get(obj)
                if m is None:
                    return None
                m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                m = np.squeeze(m).astype(bool)
                if m.shape != (H, W):
                    raise ValueError(f"{cam} obj {obj}: seed is {m.shape}, the video is {(H, W)}")
                seeds[(cam, obj)] = m
                return m.astype(np.uint8), 1, 1, ct.PROMPT_SEED

            targets, skipped, _ = ct.select_targets_from(cams, objs, prompt, max_px,
                                                         scale=scale, min_side=min_side)
            print(f"crop small     {len(targets)} targets of {len(cams) * len(objs)} (cam, obj) "
                  f"pairs (0 < prompt px < {max_px}); gate {gate}, scale {scale:g}, "
                  f"min side {min_side}", flush=True)
            frames, frames_cam = None, None
            n_written = 0
            for t in targets:
                t0 = time.time()
                cam, obj = t["cam"], t["obj"]
                cx0, cy0, side = t["crop"][0], t["crop"][1], t["side"]
                if frames_cam != cam:
                    frames = ct.read_frames(cv2, os.path.join(ds_dir, "Video", cam),
                                            start_frame, num_frame)
                    frames_cam = cam
                    if any(f.shape[:2] != (H, W) for f in frames):
                        raise ValueError(f"{cam}: video frames are not {W}x{H}")
                full = (gt == obj) if (gt is not None and cam == c_ini_cam) else seeds[(cam, obj)]
                prompt_crop = full[cy0:cy0 + side, cx0:cx0 + side]
                crops = [np.ascontiguousarray(f[cy0:cy0 + side, cx0:cx0 + side]) for f in frames]
                masks = ct.track_crop(self.predictor, torch, np, AsyncVideoFrameCPUToGPU, crops,
                                      prompt_crop, obj, num_frame,
                                      multi_session=bool(getattr(self, "uses_spatial_predictor",
                                                                 False)))
                canvases, areas, empty, touching = {}, [], 0, 0
                for k in range(num_frame):
                    m = masks.get(k)
                    if m is None:
                        m = np.zeros((side, side), dtype=bool)
                    a = int(m.sum())
                    areas.append(a)
                    if k == 0:
                        continue                       # frame start is never touched
                    empty += int(a == 0)
                    touching += int(ct.touches_window_edge(m, cx0, cy0, H, W))
                    canvases[k] = ct.paste_back(m, cx0, cy0, H, W)
                iou_f1 = ct.mask_iou(canvases[1], full) if 1 in canvases else 1.0
                kept, why = ct.gate_verdict(gate, iou_f1, t["zoom"], touching)
                # PNGs the mainline never wrote (its tracker dropped the object there):
                # a kept pair writes them, the folder gains that many files
                no_png = sum(1 for k in canvases
                             if not os.path.isfile(os.path.join(out_dir, cam, f"{start_frame + k:d}",
                                                                f"{obj:d}.png")))
                if kept:
                    for k, canvas in sorted(canvases.items()):
                        folder = os.path.join(out_dir, cam, f"{start_frame + k:d}")
                        os.makedirs(folder, exist_ok=True)
                        cv2.imwrite(os.path.join(folder, f"{obj:d}.png"),
                                    canvas.astype(np.uint8) * 255)
                        n_written += 1
                t.update({"iou_f1_seed": round(iou_f1, 6), "frames_empty": empty,
                          "frames_touching_border": touching, "frames_yielded": len(masks),
                          "frames_no_mainline_png": no_png, "areas": areas,
                          "kept": bool(kept), "why": why,
                          "seconds": round(time.time() - t0, 2)})
                print(f"  {cam:<12} obj {obj:>3}  {t['prompt']:<4} {t['seed_px']:>6} px  side "
                      f"{side:>4}  zoom {t['zoom']:.2f}  iou f1 {iou_f1:.3f}  empty "
                      f"{empty}/{num_frame - 1}  border {touching}/{num_frame - 1}  "
                      f"{'kept' if kept else 'reverted (' + str(why) + ')'}  "
                      f"{t['seconds']:.1f} s", flush=True)
            frames = None
            reverted = [t for t in targets if not t["kept"]]
            return {"max_px": int(max_px), "gate": gate, "scale": float(scale),
                    "min_side": int(min_side), "image_size": ct.IMAGE_SIZE,
                    "tau": ct.GATE_TAU if gate != "none" else None,
                    "zoom_min": ct.GATE_ZOOM_MIN if gate == "v2" else None,
                    "border_max": ct.GATE_BORDER_MAX if gate == "v2" else None,
                    "prereg": "docs/stage2-R2-prereg.md",
                    "rule": {"target": "0 < prompt px < max_px, over the prompted objects of "
                                       "the scored cameras",
                             "prompt": "c_ini: ground truth == obj; other cameras: "
                                       "masks_spatial[view][obj] (after --seeds-from)",
                             "side": "clamp(ceil(scale * max(box w, box h)), min_side, "
                                     "min(H, W)), square centred on the box, pushed inside "
                                     "the image",
                             "box": "[x0, y0, x1, y1], x1/y1 exclusive; crop = [x0, y0]",
                             "gate": "none: keep all; v1: revert when iou_f1_seed < tau; "
                                     "v2: v1, or zoom < zoom_min, or "
                                     "frames_touching_border >= border_max",
                             "frames_replaced": [int(start_frame) + 1,
                                                 int(start_frame) + int(num_frame) - 1],
                             "frames_touching_border": "written frames whose mask lies on a "
                                                       "window edge that is not an image edge",
                             "frames_no_mainline_png": "written frames for which the mainline "
                                                       "wrote no PNG (its tracker dropped the "
                                                       "object; scored as empty)"},
                    "c_ini": c_ini_cam, "cameras": cams, "objects": objs,
                    "n_targets": len(targets), "n_kept": len(targets) - len(reverted),
                    "n_reverted": len(reverted),
                    "reverted": [[t["cam"], t["obj"], t["why"]] for t in reverted],
                    "targets": targets, "skipped": skipped, "n_png_written": n_written,
                    "seconds": round(time.time() - t_start, 1)}

    return MVSegVideo, torch


def run_two_pass(sc, start_frame, num_frame, cam_names, written_cams, out_dir, cv2, np,
                 areas=None):
    """XW mode E write loop.  Pass 1 = one next() per session in camera order (exactly the
    lockstep of main's loop; those outputs are provisional and are dropped).  Pass 2 =
    one recompute request for the frame, issued after every session yielded it and
    before any session is advanced (the tracker needs feature_cache[t], which the
    next() for t+1 pops).  The seed frame is prompted, never recomputed: its pass-1
    response is written as is.  The PNG writing mirrors main's loop line for line.

    `areas` (instrumentation only) records the PASS-1 areas of every session: pass 2 is
    requested for the scored sessions alone (`output_for=scored_j`), so the unscored
    views have no pass-2 output to record.  Mode E's `view_areas.tracked` therefore holds
    provisional pass-1 areas, unlike every other mode; `two_pass` in the same manifest
    says which one you are reading."""
    scored_j = [j for j, view in enumerate(sc.track_views) if cam_names[view] in written_cams]
    for _ in range(num_frame):
        frame_idx, seed_outputs = None, {}
        for j in range(len(sc.track_views)):
            response = next(sc.tracking_result[j])                     # pass 1
            if frame_idx is None:
                frame_idx = response["frame_index"]
            assert response["frame_index"] == frame_idx, (j, response["frame_index"], frame_idx)
            if areas is not None:
                o1 = response["outputs"]
                areas.record_frame(sc.track_views[j], frame_idx, o1["out_obj_ids"].tolist(),
                                   [o1["out_binary_masks"][i] > 0.0
                                    for i in range(len(o1["out_obj_ids"]))])
            if frame_idx == start_frame and j in scored_j:
                seed_outputs[j] = response["outputs"]
        if frame_idx == start_frame:
            outputs = seed_outputs
        else:
            outputs = sc.RecomputeFrame(frame_idx, output_for=scored_j)  # pass 2, all sessions
        for j in scored_j:
            out = outputs[j]
            folder = os.path.join(out_dir, cam_names[sc.track_views[j]], f"{frame_idx:d}")
            os.makedirs(folder, exist_ok=True)
            for i, obj_id in enumerate(out["out_obj_ids"].tolist()):
                mask = (out["out_binary_masks"][i] > 0.0)
                mask = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
                cv2.imwrite(os.path.join(folder, f"{obj_id:d}.png"),
                            mask.squeeze().astype(np.uint8) * 255)
        print(f"  frame {frame_idx} written", flush=True)


def xview_stats_summary(stats, gate):
    """JSON-clean summary of the tracker's P12 counters (xview_gather.new_xview_stats):
    for the log line and MANIFEST.json (under "provenance").  None when nothing was
    recorded (no knob on).  A cell is one (session, frame); its counts are summed over
    the per-object calls."""
    if not stats or not stats.get("calls"):
        return None
    per_view, cells, cells_fail = {}, 0, 0
    for v in sorted(stats["per_view"]):
        frames = stats["per_view"][v]
        frames_fail = sorted((t, f) for t, (_, f) in frames.items() if f)
        cells += len(frames)
        cells_fail += len(frames_fail)
        per_view[str(v)] = {"seen": sum(s for s, _ in frames.values()),
                            "fail": sum(f for _, f in frames.values()),
                            "frames_fail": [[int(t), int(f)] for t, f in frames_fail]}
    return {"gate": bool(gate), "calls": int(stats["calls"]),
            "nb_seen": int(stats["nb_seen"]), "nb_fail": int(stats["nb_fail"]),
            "nb_dropped": int(stats["nb_fail"]) if gate else 0,
            "cells": cells, "cells_fail": cells_fail, "per_view": per_view}


def main():
    args = parse_args()
    c = load_config(args.config, args.dataset)
    data_root = os.path.abspath(args.data_root)
    ds_dir = os.path.join(data_root, c["folder"])
    video_root = os.path.join(ds_dir, "Video")
    cam_names = [cam_name(x, c["prefix"], c["prefix1"]) for x in c["perms"]]
    written_cams = [cam_name(x, c["prefix"], c["prefix1"]) for x in c["cam_list"]]
    forced_ref, ref_suffix = resolve_ref_cam(args.ref_cam, c["cam_list"], c.get("c_ini"))
    run = resolve_run(args, cam_names, written_cams, num_frame=c["num_frame"],
                      ref_suffix=ref_suffix)
    out_name = run["out_name"]

    missing = [n for n in cam_names if not os.path.isdir(os.path.join(video_root, n))]
    if missing:
        sys.exit(f"missing camera folders under {video_root}: {missing}")

    # --seeds-from: refuse a bad folder here, before a model is loaded (pure file checks)
    seed_dir, seed_pre = None, None
    if run["seeds_from"]:
        seed_dir = os.path.join(ds_dir, run["seeds_from"])
        try:
            seed_pre = seeds_from_module().preflight(
                seed_dir, c["start_frame"], cam_names, written_cams,
                expect={"dataset": args.dataset, "start_frame": c["start_frame"]})
        except ValueError as exc:
            sys.exit(f"--seeds-from {run['seeds_from']}: {exc}")

    # --stage1: its inputs, checkpoint and keyframe checked before a model is loaded
    stage1_pre, c_ini_cam = None, (cam_name(c["c_ini"], c["prefix"], c["prefix1"])
                                   if c.get("c_ini") is not None else None)
    if run["stage1"]:
        try:
            stage1_pre = stage1_supply_module().preflight(run["stage1"], args.dataset,
                                                          c["start_frame"], c_ini_cam)
        except ValueError as exc:
            sys.exit(f"--stage1 {run['stage1']}: {exc}")
    stages_dir = os.path.join(ds_dir, out_name + STAGES_SUFFIX) if run["save_stages"] else None

    # --ft-encoder: the scene's own-fold checkpoint, found before a model is loaded
    ft_info = None
    if run["ft_encoder"]:
        fm = ft_encoder_module()
        try:
            ft_info = fm.resolve(run["ft_encoder"], c["folder"],
                                 runs_dir=args.ft_runs or fm.DEFAULT_RUNS,
                                 folds_path=args.ft_folds or fm.DEFAULT_FOLDS)
        except ValueError as exc:
            sys.exit(f"--ft-encoder: {exc}")

    import cv2
    import numpy as np

    if forced_ref is None:
        ref_cam, n_obj = pick_reference(ds_dir, c, cv2, np)
        ref_rule = "maxid"
    else:
        ref_cam = forced_ref
        n_obj = max_object_id(ds_dir, c, ref_cam, cv2, np)
        ref_rule = args.ref_cam if args.ref_cam in ("center", "muvod") else "explicit"
    ref_index = c["perms"].index(ref_cam)

    print(f"dataset        {args.dataset}  ({ds_dir})")
    print(f"algorithm      {args.algo}  ({ALGOS[args.algo]})")
    print(f"output         {os.path.join(ds_dir, out_name)}")
    print(f"cameras        {len(cam_names)} loaded, {len(written_cams)} written "
          f"{written_cams}")
    print(f"frames         {c['start_frame']} .. "
          f"{c['start_frame'] + c['num_frame'] - 1}  ({c['num_frame']})")
    print(f"reference      {cam_name(ref_cam, c['prefix'], c['prefix1'])} "
          f"(view index {ref_index}), {n_obj} objects prompted, rule {ref_rule}", flush=True)
    print(f"cross-view     lineage {run['lineage']}, track {run['track_mode']} "
          f"({len(run['track_idx'])} sessions)", flush=True)
    print(f"seed repair    {'on (Rp, max 2 rounds)' if run['repair_seeds'] else 'off'}",
          flush=True)
    print(f"frame0 seed    {'on (Fs: start_frame PNG = the seed, not its re-prediction)' if run['frame0_seed'] else 'off'}",
          flush=True)
    print("crop small     " + (f"on ({run['crop']['suffix']}: prompt < {run['crop']['max_px']} px, "
                               f"gate {run['crop']['gate']}, scale {run['crop']['scale']:g}, "
                               f"min side {run['crop']['min_side']})" if run["crop"] else "off"),
          flush=True)
    print("ft encoder     " + (f"on ({run['ft_encoder']} fold{ft_info['fold']}: {ft_info['path']}, "
                               f"sha256 {ft_info['sha256'][:12]}; restored before the crop pass)"
                               if ft_info else "off"), flush=True)
    if stage1_pre is not None:
        print(f"stage 1        in process: {run['stage1']} = {stage1_pre['arm']}, gate {stage1_pre['gate']}, "
              f"fold{stage1_pre['fold']} ({stage1_pre['checkpoint']}, sha256 "
              f"{str(stage1_pre['ckpt_sha256'])[:12]})", flush=True)
    print(f"ref snap       {'on (Rs: S2-R6 rule on c_ini after the crop pass)' if run['ref_snap'] else 'off'}",
          flush=True)
    if stages_dir:
        print(f"save stages    {stages_dir}", flush=True)
    if seed_pre is not None:
        n_png = sum(len(p) for p in seed_pre["pngs"].values())
        cover = ("manifest covers " + ", ".join(seed_pre["written"]) if seed_pre["written"]
                 else "no manifest")
        print(f"seeds from     {seed_dir}  ({n_png} PNG in {len(seed_pre['pngs'])} cameras, "
              f"{cover})", flush=True)
        if seed_pre["warning"]:
            print(f"               warning: {seed_pre['warning']}", flush=True)

    out_dir = os.path.join(ds_dir, out_name)
    if not args.overwrite and os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(f"{out_dir} already exists and is not empty; pass --overwrite")
    if args.dry_run:
        print("dry run, stopping before the model is built")
        return

    if stages_dir and os.path.isdir(stages_dir):
        if not args.overwrite:
            sys.exit(f"{stages_dir} already exists; pass --overwrite")
        import shutil
        shutil.rmtree(stages_dir)

    MVSegVideo, torch = build_runner(args.algo)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device         {device}", flush=True)

    # --stage1: after build_runner (its sys.path surgery is what lets the SAM 3 package import) and
    # before the stage-2 model exists; run_stage1 frees the stage-1 model before returning
    stage1_seeds, stage1_record = None, None
    if run["stage1"]:
        video_cams = sorted(d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d)))
        first = sorted(os.listdir(os.path.join(video_root, video_cams[0])))[0]
        hw = cv2.imread(os.path.join(video_root, video_cams[0], first), cv2.IMREAD_GRAYSCALE).shape
        try:
            stage1_seeds, stage1_record = stage1_supply_module().run_stage1(
                run["stage1"], args.dataset, video_cams, hw, c_ini_cam, c["start_frame"], np,
                save_dir=stages_dir, cv2=cv2, log=lambda m: print(m, flush=True))
        except ValueError as exc:
            sys.exit(f"--stage1 {run['stage1']}: {exc}")

    track_mode, track_idx = run["track_mode"], run["track_idx"]

    sc = MVSegVideo(device, **run["xview_kwargs"])
    # Read back what the model actually holds (catches an env override or a broken hop).
    tracker = getattr(getattr(sc.predictor, "model", None), "tracker", None)
    eff_window = getattr(tracker, "cross_view_window", None)
    eff_hygiene = getattr(tracker, "cross_view_hygiene", None)
    eff_mode = getattr(tracker, "cross_view_mode", None)
    eff_gate = getattr(tracker, "cross_view_gate", None)
    eff_ptr = getattr(tracker, "cross_view_ptr", None)
    eff_shift = getattr(tracker, "cross_view_tpos_shift", None)
    got = (eff_window, eff_hygiene, eff_mode, eff_gate, eff_ptr, eff_shift)
    if args.algo in XVIEW_ALGOS:
        want = ((run["xview_window"] if run["xview_on"] else 4), run["xview_on"],
                (run["xview_mode_eff"] if run["xview_on"] else XVIEW_LEGACY_MODE),
                run["xview_gate"], run["xview_ptr"], run["xview_tpos_shift"])
        if got != want:
            sys.exit(f"tracker holds cross_view={got} but the command line asked for {want}")
    print(f"cross-view     window {eff_window}, hygiene {eff_hygiene}, mode {eff_mode}, "
          f"gate {eff_gate}, ptr {eff_ptr}, tpos-shift {eff_shift}", flush=True)
    sc.LoadCameraFolders(video_root, cam_names, c["start_frame"], track_idx)
    print(f"sessions ready ({sc.numImage} cameras loaded, "
          f"{len(track_idx)} tracked [{track_mode}], "
          f"{sc.video_width}x{sc.video_height})", flush=True)

    ref_gt = cv2.imread(
        os.path.join(ds_dir, "Mask", cam_name(ref_cam, c["prefix"], c["prefix1"]),
                     f"{c['start_frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
    for obj_id in range(1, n_obj + 1):
        sc.AddReferenceMask(ref_index, (ref_gt == obj_id).astype("float32"), obj_id)
    print(f"prompted       {len(sc.obj_ids)} objects on view {ref_index}", flush=True)

    sc.PropagateAcrossViews(ref_index)
    print(f"cross-view     done, {len(sc.masks_spatial)} views have masks", flush=True)

    areas = ViewAreas()

    # P5.  Before RetireSpatialPredictor: the repair re-opens the spatial model.
    repair_stats = None
    if run["repair_seeds"]:
        repair_stats = sc.RepairSeeds(ref_index, ref_gt, n_obj)
        for rec in repair_stats["rounds"]:
            print(f"seed repair    round {rec['round']}: {rec['flagged']} flagged, "
                  f"{len(rec['repairs'])} repaired, {len(rec['unrepairable'])} unrepairable",
                  flush=True)
            for r, a in zip(rec["repairs"], rec["after"]):
                print(f"               view {r['view']} obj {r['obj']}: {r['area']} px "
                      f"(T={r['threshold']:.0f}) <- donor view {r['donor']} "
                      f"({r['donor_area']} px) -> {a['area']} px", flush=True)
            for u in rec["unrepairable"]:
                print(f"               view {u['view']} obj {u['obj']}: {u['area']} px "
                      f"(T={u['threshold']:.0f}) no donor", flush=True)

    # Stage-1 supply, same slot (docs/stage1-plan.md section 3): masks_spatial must be
    # final before record_seed and TrackForward.  Excludes --repair-seeds (resolve_run).
    seeds_stats = None
    if run["seeds_from"] or run["stage1"]:
        try:
            if run["stage1"]:
                seeds_stats = sc.LoadSeeds(stage1_seeds, cam_names, track_idx)
                seeds_stats.update(stage1_record)
                del stage1_seeds
            else:
                seeds_stats = sc.LoadSeedsFrom(seed_dir, c["start_frame"], cam_names, track_idx)
        except ValueError as exc:
            sys.exit(f"--{'stage1' if run['stage1'] else 'seeds-from'} "
                     f"{run['stage1'] or run['seeds_from']}: {exc}")
        print(f"seeds from     {seeds_stats['replaced']} (view, obj) seeds replaced in views "
              f"{sorted(int(v) for v in seeds_stats['replaced_views'])}, "
              f"{sum(len(o) for o in seeds_stats['changed_views'].values())} differ from G0 "
              f"({seeds_stats['added']} added, {seeds_stats['emptied']} emptied); "
              f"{len(seeds_stats['untracked_views'])} untracked views skipped; "
              + (f"mask digest {seeds_stats['mask_digest'][:12]}" if run["stage1"]
                 else f"png digest {seeds_stats['png_digest'][:12]}"), flush=True)

    # The cross-view model is dead weight from here on: ~3.2 GiB of parameters,
    # the N-view pseudo-video, its feature cache and its per-view tracker
    # memories.  Retiring it does not touch `masks_spatial`, which TrackForward
    # still reads.  The autocast cache pins a bf16 copy of every weight it ever
    # cast (the tracker enters torch.autocast permanently and never exits, so
    # ATen never clears it) and keys those entries on a weak ref to the now
    # dead fp32 parameters, so without this clear ~1.6 GiB stays stranded.
    # Clearing is value-neutral: entries are a deterministic `param.to(bf16)`
    # of weights that never change during inference, so they are recomputed
    # bit-for-bit.  Once here, not per frame: inside a frame the whole live
    # weight set is cached anyway, so per-frame clearing lowers no peak.
    # Instrumentation only (2026-09-09): the seeds every loaded view starts from,
    # including the views nobody scores.  Read after any repair, so it reflects what
    # TrackForward actually consumes.
    areas.record_seed(sc.masks_spatial)

    if hasattr(sc, "RetireSpatialPredictor"):
        sc.RetireSpatialPredictor()
        torch.clear_autocast_cache()
        print("spatial model  retired", flush=True)

    # --ft-encoder (S2-R5): the swap happens before TrackForward, so no memory-attention forward
    # of the temporal tracker has run on the original weights yet; the autocast clear drops any
    # bf16 copy keyed on the (in-place updated) parameters all the same.
    ft_stats, ft_original, ft_enc = None, None, None
    if ft_info is not None:
        fm = ft_encoder_module()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            sys.exit("--ft-encoder: more than one GPU is visible; the predictor's worker "
                     "processes would keep the original weights (run with one GPU)")
        ft_enc = getattr(getattr(tracker, "transformer", None), "encoder", None)
        if ft_enc is None:
            sys.exit("--ft-encoder: the temporal tracker has no transformer.encoder")
        ck = torch.load(ft_info["path"], map_location="cpu")
        try:
            train_scenes = fm.check_checkpoint(ck, ft_info)
        except ValueError as exc:
            sys.exit(f"--ft-encoder: {exc}")
        digest_ckpt = fm.state_digest(ck["memory_attention"], torch)
        # functional probe: what the encoder's linears output through the live autocast cache.
        # Run on the original weights first, which fills the cache with their bf16 copies --
        # exactly the state a missing clear would leave -- so the probe after the swap fails
        # unless the forwards really use the loaded weights.
        probe_orig, probe_dtypes = fm.probe_digest(ft_enc, torch)
        try:
            ft_original = fm.swap_in(ft_enc, ck["memory_attention"], torch)
        except ValueError as exc:
            sys.exit(f"--ft-encoder: {exc}")
        torch.clear_autocast_cache()
        digest_orig = fm.state_digest(ft_original, torch)
        digest_loaded = fm.state_digest(ft_enc.state_dict(), torch)
        if digest_loaded != digest_ckpt or digest_loaded == digest_orig:
            sys.exit(f"--ft-encoder: encoder digest after the load {digest_loaded[:12]} "
                     f"(checkpoint {digest_ckpt[:12]}, original {digest_orig[:12]})")
        probe_loaded, _ = fm.probe_digest(ft_enc, torch)
        if probe_loaded == probe_orig:
            sys.exit("--ft-encoder: the encoder's outputs did not change after the load (stale "
                     "autocast cache?)")
        ft_stats = {"prereg": "docs/stage2-R5-prereg.md", "arm": ft_info["arm"],
                    "fold": ft_info["fold"], "checkpoint": ft_info["path"],
                    "checkpoint_sha256": ft_info["sha256"], "folds": ft_info["folds_path"],
                    "folds_sha256": ft_info["folds_sha256"], "train_scenes": train_scenes,
                    "n_tensors": len(ck["memory_attention"]),
                    "digest_original": digest_orig, "digest_checkpoint": digest_ckpt,
                    "digest_loaded": digest_loaded,
                    "probe_original": probe_orig, "probe_loaded": probe_loaded,
                    "probe_dtypes": probe_dtypes,
                    "restored_before_crop": False, "digest_restored": None,
                    "probe_restored": None}
        del ck
        print(f"ft encoder     loaded {ft_stats['n_tensors']} tensors from fold{ft_info['fold']} "
              f"(digest {digest_orig[:12]} -> {digest_loaded[:12]}; probe {probe_orig[:12]} -> "
              f"{probe_loaded[:12]} {probe_dtypes}; trained on {len(train_scenes)} other scenes)",
              flush=True)

    if run["ref_snap_mem"] and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        sys.exit("--ref-snap-mem: more than one GPU is visible; the predictor's worker processes would "
                 "hold other copies of the c_ini memory (run with one GPU)")
    sc.TrackForward(c["start_frame"], c["num_frame"])

    frame0_stats = None
    if run["two_pass"]:
        run_two_pass(sc, c["start_frame"], c["num_frame"], cam_names, written_cams, out_dir,
                     cv2, np, areas=areas)
    else:
        # --frame0-seed: {view: {obj: "seed" | "tracker"}} for the manifest; None when off
        frame0_sources = {} if run["frame0_seed"] else None
        # --ref-snap-mem (S2-R7): the online S2-R6 decision on c_ini, c_ini crop targets T excluded
        mem_snap, mem_T, mem_recs, mem_secs = None, [], [], 0.0
        if run["ref_snap_mem"]:
            if c_ini_cam is None or c_ini_cam not in written_cams or cam_names[ref_index] != c_ini_cam:
                sys.exit("--ref-snap-mem: the reference camera must be MUVOD's c_ini and a written camera")
            if run["crop"]:
                gt_path = os.path.join(ds_dir, "Mask", c_ini_cam, f"{c['start_frame']:06d}.png")
                gt0 = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt0 is None:
                    raise FileNotFoundError(gt_path)
                mem_T = ref_snap_mem_targets(gt0, c_ini_cam, sc.obj_ids, run["crop"], crop_track_module())
            mem_snap = ref_snap_module().OnlineSnap(out_dir, os.path.join(video_root, c_ini_cam), c_ini_cam,
                                                    c["start_frame"], mem_T, np, cv2)
        for _ in range(c["num_frame"]):
            for j, view in enumerate(sc.track_views):
                response = next(sc.tracking_result[j])
                out = response["outputs"]
                obj_ids = out["out_obj_ids"].tolist()
                masks = [out["out_binary_masks"][i] > 0.0 for i in range(len(obj_ids))]
                # instrumentation: every tracked view, scored or not, always the tracker's
                # own output (the frame-0 seed swap below is a write-time substitution)
                areas.record_frame(view, response["frame_index"], obj_ids, masks)
                if cam_names[view] not in written_cams:
                    continue
                frame_idx = response["frame_index"]
                if frame0_sources is not None and frame_idx == c["start_frame"]:
                    picked = seeds_from_module().select_frame0(
                        sc.masks_spatial.get(view), obj_ids, masks)
                    masks = [m for m, _ in picked]
                    frame0_sources[view] = {o: src for o, (_, src) in zip(obj_ids, picked)}
                folder = os.path.join(out_dir, cam_names[view], f"{frame_idx:d}")
                os.makedirs(folder, exist_ok=True)
                written_np = {}
                for obj_id, mask in zip(obj_ids, masks):
                    mask = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
                    cv2.imwrite(os.path.join(folder, f"{obj_id:d}.png"),
                                mask.squeeze().astype(np.uint8) * 255)
                    if mem_snap is not None:
                        written_np[int(obj_id)] = mask.squeeze().astype(bool)
                if mem_snap is not None and view == ref_index and frame_idx > c["start_frame"]:
                    t_mem = time.time()
                    snapped = mem_snap.decide_frame(frame_idx, written_np)
                    if snapped:
                        for rec in sc.InjectRefMemory(j, frame_idx, [(o, mem_snap.frame0(o)) for o in snapped]):
                            rec["t"] = int(frame_idx)
                            mem_recs.append(rec)
                    mem_secs += time.time() - t_mem
            print(f"  frame {response['frame_index']} written", flush=True)
        if frame0_sources is not None:
            frame0_stats = seeds_from_module().frame0_provenance(frame0_sources)
            fell_back = {v: d["tracker_objs"] for v, d in frame0_stats["views"].items()
                         if d["tracker_objs"]}
            print(f"frame0 seed    {frame0_stats['seed']} (view, obj) PNGs written from the seed "
                  f"in {frame0_stats['n_views']} views, {frame0_stats['tracker']} from the "
                  f"tracker (no seed){': ' + str(fell_back) if fell_back else ''}", flush=True)

    # --crop-small (S2-R2): after every mainline PNG is on disk, before the counters and
    # the manifest.  None when off, so the manifest carries one value either way.
    crop_stats = None
    if run["crop"]:
        cs = run["crop"]
        if ft_original is not None:
            # S2-R5: the crop pass runs on the original weights (prereg section 2), so its
            # output must equal the base run's byte for byte (watch value Z2)
            fm = ft_encoder_module()
            fm.restore(ft_enc, ft_original, torch)
            torch.clear_autocast_cache()
            ft_stats["digest_restored"] = fm.state_digest(ft_enc.state_dict(), torch)
            ft_stats["probe_restored"], _ = fm.probe_digest(ft_enc, torch)
            if (ft_stats["digest_restored"] != ft_stats["digest_original"]
                    or ft_stats["probe_restored"] != ft_stats["probe_original"]):
                sys.exit(f"--ft-encoder: restore failed before the crop pass (digest "
                         f"{ft_stats['digest_restored'][:12]}, original "
                         f"{ft_stats['digest_original'][:12]}; probe "
                         f"{ft_stats['probe_restored'][:12]}, original "
                         f"{ft_stats['probe_original'][:12]})")
            ft_stats["restored_before_crop"] = True
            print(f"ft encoder     original weights restored before the crop pass "
                  f"(digest {ft_stats['digest_restored'][:12]})", flush=True)
        c_ini_cam = (cam_name(c["c_ini"], c["prefix"], c["prefix1"])
                     if c.get("c_ini") is not None else None)
        crop_stats = sc.CropSmall(ds_dir, out_dir, cam_names, written_cams, c_ini_cam,
                                  c["start_frame"], c["num_frame"], cs["max_px"], cs["gate"],
                                  cs["scale"], cs["min_side"], cv2, np)
        if run["ref_snap_mem"]:
            t_crop = sorted(int(x["obj"]) for x in crop_stats["targets"] if x["cam"] == c_ini_cam)
            if t_crop != mem_T:
                sys.exit(f"--ref-snap-mem: the excluded set {mem_T} is not the crop pass's c_ini targets {t_crop}")
        print(f"crop small     {crop_stats['n_kept']} kept, {crop_stats['n_reverted']} reverted "
              f"of {crop_stats['n_targets']} targets; {crop_stats['n_png_written']} PNG "
              f"overwritten ({crop_stats['seconds']} s)"
              + (": " + ", ".join(f"{cam}/{obj} ({why})" for cam, obj, why in crop_stats["reverted"])
                 if crop_stats["reverted"] else ""), flush=True)

    # --ref-snap (S2-R6): the last change to the masks, after the crop pass
    snap_stats = None
    if run["ref_snap"]:
        snap_stats = ref_snap_module().snap_in_place(
            out_dir, os.path.join(video_root, c_ini_cam), c_ini_cam, c["start_frame"], np, cv2,
            keep_dir=os.path.join(stages_dir, "stage2") if stages_dir else None)
        print(f"ref snap       {snap_stats['snapped']} / {snap_stats['objects_frames']} (object, frame) "
              f"PNGs of {c_ini_cam} set to the frame-0 prompt", flush=True)
    elif stages_dir:
        import shutil
        shutil.copytree(out_dir, os.path.join(stages_dir, "stage2"), copy_function=os.link)
    # --ref-snap-mem (S2-R7): what went into the c_ini memory (null when off)
    mem_stats = None
    if run["ref_snap_mem"]:
        rsm = ref_snap_module()
        ok = [r for r in mem_recs if "error" not in r]
        rel = sorted(r["rel_change"] for r in ok)
        injected = sorted(f"{c_ini_cam}/{r['t']}/{r['obj']}.png" for r in ok)
        snap_minus_T = sorted(x for x in snap_stats["snapped_files"]
                              if int(x.split("/")[-1].split(".")[0]) not in set(mem_T))
        mem_stats = {"prereg": "docs/stage2-R7-prereg.md",
                     "rule": {"tau": rsm.TAU, "delta": rsm.DELTA, "dilate": rsm.DILATE},
                     "encoding": "frame-0 PNG -> 288 bilinear antialias -> x20-10 -> 1008 bilinear, "
                                 "is_mask_from_pts=True, stored object score; maskmem_features copied in place",
                     "excluded_crop_targets": mem_T, "injected": injected, "n_injected": len(injected),
                     "injected_equals_snap_minus_targets": injected == snap_minus_T,
                     "errors": [r for r in mem_recs if "error" in r],
                     "cache_misses": sum(1 for r in ok if not r["cache_hit"]),
                     "obj_score_min": min((r["obj_score"] for r in ok), default=None),
                     "rel_change": ({"median": rel[len(rel) // 2], "p90": rel[int(0.9 * (len(rel) - 1))],
                                     "max": rel[-1]} if rel else None),
                     "seconds": round(mem_secs, 3)}
        print(f"ref snap mem   {len(injected)} (object, frame) memories of {c_ini_cam} set to the frame-0 "
              f"prompt ({len(mem_T)} crop targets excluded; equals the PNG snaps minus targets: "
              f"{mem_stats['injected_equals_snap_minus_targets']}; cache misses {mem_stats['cache_misses']}; "
              f"errors {len(mem_stats['errors'])}; {mem_stats['seconds']} s)", flush=True)

    # P12 counters (filled only while a knob is on): one summary line, and the manifest.
    xstats = xview_stats_summary(getattr(tracker, "xview_stats", None), eff_gate)
    if xstats is not None:
        print(f"cross-view     neighbour entries seen {xstats['nb_seen']}, failing admission "
              f"{xstats['nb_fail']} (dropped {xstats['nb_dropped']}), cells with a failure "
              f"{xstats['cells_fail']}/{xstats['cells']}", flush=True)

    print(f"done -> {out_dir}", flush=True)

    # Provenance manifest, <out_dir>/MANIFEST.json (ROADMAP Phase 1, REPORT.md
    # P13).  Purely additive: every mask is already on disk, and a failure
    # here must never fail a run.  --dry-run returned before the model was
    # built, so it never reaches this point.  The repo root goes on sys.path
    # for this one import only; the sys.path surgery in build_runner stays as
    # it is.  Bytecode writing is off so a root process in the container does
    # not leave a root-owned .pyc under eval/.
    try:
        repo_root = os.path.dirname(HERE)
        dont_write = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, repo_root)
        try:
            from eval.manifest import write_run_manifest
        finally:
            sys.path.remove(repo_root)
            sys.dont_write_bytecode = dont_write
        manifest = write_run_manifest(
            out_dir, dataset=c["folder"], method=out_name, algo=args.algo,
            package_dir=os.path.join(HERE, ALGOS[args.algo]), track_cams=track_mode,
            repo=repo_root, torch=torch,
            extra={"dataset_config_key": args.dataset,
                   "reference": cam_name(ref_cam, c["prefix"], c["prefix1"]),
                   "reference_view_index": ref_index,
                   "n_objects_prompted": n_obj,
                   "reference_rule": ref_rule,
                   "reference_suffix": ref_suffix,
                   "start_frame": c["start_frame"], "num_frame": c["num_frame"],
                   "device": device,
                   "lineage": run["lineage"],
                   "xview_window": eff_window, "xview_hygiene": eff_hygiene,
                   "track_cams_requested": args.track_cams,
                   "track_idx": run["track_idx"], "n_sessions": len(run["track_idx"]),
                   "scored_view_idx": run["scored_idx"],
                   "xview_mode": eff_mode, "two_pass": run["two_pass"],
                   "closure_reach": run["closure_reach"],
                   "xview_gate": eff_gate, "xview_ptr": eff_ptr, "xview_tpos_shift": eff_shift,
                   "xview_gate_stats": xstats,
                   "seed_repair": repair_stats,
                   "seeds_from": seeds_stats,
                   "frame0_seed": run["frame0_seed"], "frame0_seed_stats": frame0_stats,
                   "crop_small": crop_stats,
                   "ft_encoder": ft_stats,
                   "stage1": run["stage1"], "ref_snap": snap_stats, "ref_snap_mem": mem_stats,
                   "save_stages": stages_dir,
                   "view_areas": areas.as_record()})
        print(f"manifest -> {manifest}", flush=True)
    except Exception as exc:  # bookkeeping only: never let it fail the run
        print(f"manifest not written ({type(exc).__name__}: {exc})", flush=True)


if __name__ == "__main__":
    main()
