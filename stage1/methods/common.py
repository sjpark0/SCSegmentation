"""Shared plumbing for every method: the one input they may read and the one output format.

Importable from the host venv and from every container (no torch at import time).  A method is

    python methods/<name>/run.py --scene <name|all> --data Data --out <method> [--overwrite] [...]

and implements `fn(scene: Scene, args) -> config dict`, handed to `run_all`, which copies the
reference camera's ground truth, times the call, writes RUN.json (with the traceback on failure)
and moves on to the next scene.
"""
import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIBLING = os.path.dirname(REPO)          # the main repo: stage 1 lives in <main>/stage1 (ported from SCSegmentation2D)
SAM3_PKG = os.path.join(SIBLING, "SCSam3", "demoSCSam3MVOpt")
SAM3_HF_CACHE = os.path.join(SIBLING, "SCSam3", "hf_cache")

if REPO not in sys.path:
    sys.path.insert(0, REPO)
from mvseg2d.io import (labelmap_objects, labelmap_to_mask, load_labelmap, load_meta,  # noqa: E402
                        result_dir, write_mask)


@dataclass
class Scene:
    name: str
    cams: list
    ref_cam: str
    mask_cams: list
    hw: tuple
    image_paths: dict
    ref_labels: np.ndarray
    objects: list
    frame: int = None        # None = the benchmark's reference frame; t = Data/<scene>/frames/<t>/
    external_fold: int = None  # a scene outside the benchmark (not in folds.json): the held-out fold it is decoded with
    ring_order: list = None    # meta.json "ring_order": the cameras in physical rig order when the index order is not it (CROOM)


def frame_dir(data, scene, frame=None):
    """Folder holding images/ and masks/: the scene root for the reference frame, frames/<t>/ otherwise."""
    root = os.path.join(data, scene)
    return root if frame is None else os.path.join(root, "frames", f"{int(frame):06d}")


def load_scene(data, scene, frame=None):
    meta = load_meta(data, scene)
    d = frame_dir(data, scene, frame)
    ref_labels = load_labelmap(os.path.join(d, "masks", f"{meta['ref_cam']}.png"))
    return Scene(name=scene, cams=list(meta["cams"]), ref_cam=meta["ref_cam"],
                 mask_cams=list(meta["mask_cams"]), hw=tuple(ref_labels.shape),
                 image_paths={c: os.path.join(d, "images", f"{c}.jpg") for c in meta["cams"]},
                 ref_labels=ref_labels, objects=labelmap_objects(ref_labels), frame=frame,
                 external_fold=meta.get("external_fold"), ring_order=meta.get("ring_order"))


def all_scenes(data):
    return sorted(d for d in os.listdir(data) if os.path.isfile(os.path.join(data, d, "meta.json")))


def parse_args(description, extra=None, argv=None):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="scene name or 'all'")
    ap.add_argument("--data", default=os.path.join(REPO, "Data"))
    ap.add_argument("--out", required=True, help="method folder name under Data/<scene>/Result/")
    ap.add_argument("--overwrite", action="store_true", help="re-run scenes that already have a RUN.json")
    if extra:
        extra(ap)
    return ap.parse_args(argv)


def write_result(data, scene_name, method, cam, obj_id, mask_bool):
    write_mask(os.path.join(result_dir(data, scene_name, method), cam, f"{int(obj_id)}.png"), mask_bool)


def copy_reference(scene, data, method):
    for o in scene.objects:
        write_result(data, scene.name, method, scene.ref_cam, o, labelmap_to_mask(scene.ref_labels, o))


def _git_hash():
    """Short hash of this repo, with "-dirty" when the tree has uncommitted changes."""
    try:
        h = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return None


def gpu_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 2 ** 20)
    except Exception:
        pass
    return None


class RunTimer:
    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self.t0 = time.time()
        self.seconds = None
        self.gpu_mb = None
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        self.gpu_mb = gpu_peak_mb()
        return False


def write_run(data, scene_name, method, seconds, gpu_mb, config, error=None, status="done"):
    """RUN.json.  `status` is "running" while the method works and "done" afterwards; the scorer
    treats a missing RUN.json or status "running" as an incomplete run (excluded, footnoted)."""
    rdir = result_dir(data, scene_name, method)
    os.makedirs(rdir, exist_ok=True)
    run = {"method": method, "scene": scene_name, "status": status, "seconds": seconds, "gpu_mb": gpu_mb,
           "config": config, "git": _git_hash(), "date": _dt.datetime.now().isoformat(timespec="seconds"),
           "error": error}
    with open(os.path.join(rdir, "RUN.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
    return run


def run_all(args, fn, aux_outs=()):
    """Run `fn(scene, args)` per scene into `args.out`.  `aux_outs` are further result folders the method
    writes itself (e.g. `<out>_r1` of a two-round method): they are cleared and marked 'running' before fn,
    and get an error record if fn fails, so no stale PNG or stale 'done' RUN.json can survive a re-run."""
    scenes = all_scenes(args.data) if args.scene == "all" else [args.scene]
    for name in scenes:
        rdir = result_dir(args.data, name, args.out)
        if os.path.isfile(os.path.join(rdir, "RUN.json")) and not getattr(args, "overwrite", False):
            print(f"{name:20s} skip (RUN.json exists; --overwrite to redo)", flush=True)
            continue
        scene = load_scene(args.data, name)
        # a clean folder: no PNG from an earlier run may survive into this one
        for out in [args.out, *aux_outs]:
            d = result_dir(args.data, name, out)
            if os.path.isdir(d):
                shutil.rmtree(d)
            write_run(args.data, name, out, None, None, None, status="running")
        copy_reference(scene, args.data, args.out)
        fatal = None
        with RunTimer() as t:
            try:
                config = fn(scene, args)
                error = None
            except Exception:
                config = None
                error = traceback.format_exc()
            except BaseException as e:            # KeyboardInterrupt, SystemExit: record, then re-raise
                config = None
                error = traceback.format_exc()
                fatal = e
        write_run(args.data, name, args.out, t.seconds, t.gpu_mb, config, error)
        if error:
            for out in aux_outs:
                write_run(args.data, name, out, t.seconds, t.gpu_mb, None, error)
        status = "ERROR " + error.strip().splitlines()[-1] if error else "ok"
        print(f"{name:20s} {t.seconds:7.1f}s  {status}", flush=True)
        if fatal is not None:
            raise fatal
