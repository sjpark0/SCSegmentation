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

Both demos carry their own copy of the model code, so the algorithm is selected
by putting its directory first on sys.path.
"""
import argparse
import json
import os
import sys

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
_SEEDS_FROM = None


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
    if xview_on and out_name in (*DEFAULT_OUT.values(), ONESTAGE_CLOSURE_OUT):
        sys.exit(f"refusing to write an XW run into {out_name}: that folder is the published "
                 "legacy lineage (pass --out SegMaskSam3XW...)")
    if not xview_on and out_name.startswith(XVIEW_OUT_PREFIX):
        sys.exit(f"refusing to write a legacy run into {out_name}: folder names starting "
                 f"with {XVIEW_OUT_PREFIX} are the XW lineage (pass --xview-window, or "
                 "another --out)")
    return dict(out_name=out_name, repair_seeds=repair, seeds_from=seeds_from,
                track_mode=track_mode, track_idx=track_idx,
                scored_idx=scored_idx, xview_on=xview_on, xview_window=window,
                xview_kwargs=xview_kwargs, lineage=lineage,
                xview_mode=mode, xview_mode_eff=(mode_eff if xview_on else None),
                two_pass=bool(xview_on and mode_eff in XVIEW_TWO_PASS),
                closure_reach=(len(track_idx) if track_mode == "closure" else None),
                xview_gate=gate, xview_ptr=ptr, xview_tpos_shift=shift)


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

    MVSegVideo, torch = build_runner(args.algo)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device         {device}", flush=True)

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
    if run["seeds_from"]:
        try:
            seeds_stats = sc.LoadSeedsFrom(seed_dir, c["start_frame"], cam_names, track_idx)
        except ValueError as exc:
            sys.exit(f"--seeds-from {run['seeds_from']}: {exc}")
        print(f"seeds from     {seeds_stats['replaced']} (view, obj) seeds replaced in views "
              f"{sorted(int(v) for v in seeds_stats['replaced_views'])}, "
              f"{sum(len(o) for o in seeds_stats['changed_views'].values())} differ from G0 "
              f"({seeds_stats['added']} added, {seeds_stats['emptied']} emptied); "
              f"{len(seeds_stats['untracked_views'])} untracked views skipped; "
              f"png digest {seeds_stats['png_digest'][:12]}", flush=True)

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

    sc.TrackForward(c["start_frame"], c["num_frame"])

    if run["two_pass"]:
        run_two_pass(sc, c["start_frame"], c["num_frame"], cam_names, written_cams, out_dir,
                     cv2, np, areas=areas)
    else:
        for _ in range(c["num_frame"]):
            for j, view in enumerate(sc.track_views):
                response = next(sc.tracking_result[j])
                out = response["outputs"]
                # instrumentation: every tracked view, scored or not
                areas.record_frame(view, response["frame_index"],
                                   out["out_obj_ids"].tolist(),
                                   [out["out_binary_masks"][i] > 0.0
                                    for i in range(len(out["out_obj_ids"]))])
                if cam_names[view] not in written_cams:
                    continue
                frame_idx = response["frame_index"]
                folder = os.path.join(out_dir, cam_names[view], f"{frame_idx:d}")
                os.makedirs(folder, exist_ok=True)
                for i, obj_id in enumerate(out["out_obj_ids"].tolist()):
                    mask = (out["out_binary_masks"][i] > 0.0)
                    mask = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
                    cv2.imwrite(os.path.join(folder, f"{obj_id:d}.png"),
                                mask.squeeze().astype(np.uint8) * 255)
            print(f"  frame {response['frame_index']} written", flush=True)

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
                   "view_areas": areas.as_record()})
        print(f"manifest -> {manifest}", flush=True)
    except Exception as exc:  # bookkeeping only: never let it fail the run
        print(f"manifest not written ({type(exc).__name__}: {exc})", flush=True)


if __name__ == "__main__":
    main()
