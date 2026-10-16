#!/usr/bin/env python3
"""S2-R6H (docs/stage2-R6H-prereg.md section 2/8): the FROZEN S2-R6 rule applied unchanged to MUVOD's annotated
NON-reference cameras of an existing GT-prompted stage-2 output folder.

For each scene: every camera folder of <method> is copied as hard links (MANIFEST.json is never linked) into
<method>Rh, then SCSam3/ref_snap.snap_in_place (TAU 0.8, DELTA 4, DILATE 31; unlink + fresh copy of the frame-0
PNG, so the source keeps its bytes) runs on each annotated camera other than c_ini.  c_ini and unannotated cameras
stay hard links.  A fresh MANIFEST.json records the source, its content digest and MANIFEST sha256, the rule, the
script and ref_snap.py sha256, git HEAD and the per-camera snap lists.  Host python (cv2, numpy):

    PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python MVSeed/s2r6h_snap.py --method SegMaskSam3XW1CGPS4MSdgtFsCs10kG2
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data", "MVSeg")
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json"), encoding="utf-8"))
CEN = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
SUFFIX = "Rh"
REF_SNAP = os.path.join(REPO, "SCSam3", "ref_snap.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def heldout_cams(scene):
    """The annotated cameras other than c_ini (docs/raw/muvod_object_sets.json `cameras`)."""
    return [c for c in CEN[scene]["cameras"] if c != CEN[scene]["c_ini"]]


def snap_scene(job):
    scene, method, overwrite = job
    import cv2
    import numpy as np
    rs = _load("ref_snap", REF_SNAP)
    f, start = CFG[scene]["folder"], CFG[scene]["start_frame"]
    src, dst = os.path.join(DATA, f, method), os.path.join(DATA, f, method + SUFFIX)
    if os.path.exists(dst):
        if not overwrite:
            raise SystemExit(f"{dst} exists (pass --overwrite)")
        shutil.rmtree(dst)
    os.makedirs(dst)
    for cam in sorted(os.listdir(src)):
        if os.path.isdir(os.path.join(src, cam)):
            shutil.copytree(os.path.join(src, cam), os.path.join(dst, cam), copy_function=os.link)
    per_cam = {}
    for cam in heldout_cams(scene):
        if not os.path.isdir(os.path.join(dst, cam)):
            raise SystemExit(f"{scene}: held-out camera {cam} missing in {src}")
        st = rs.snap_in_place(dst, os.path.join(DATA, f, "Video", cam), cam, start, np, cv2)
        per_cam[cam] = st
    return scene, dict(c_ini=CEN[scene]["c_ini"], heldout=heldout_cams(scene), start_frame=start, per_cam=per_cam,
                       snapped=sum(s["snapped"] for s in per_cam.values()),
                       objects_frames=sum(s["objects_frames"] for s in per_cam.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True)
    ap.add_argument("--scenes", nargs="*", default=sorted(CEN))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(REPO, "eval"))
    import manifest as M
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = len(subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.splitlines())
    src_info = {}
    for s in a.scenes:
        src = os.path.join(DATA, CFG[s]["folder"], a.method)
        if not os.path.isdir(src):
            raise SystemExit(f"{src} missing")
        man = os.path.join(src, "MANIFEST.json")
        src_info[s] = dict(content_digest=M.content_digest(src)[0], manifest_sha256=sha256(man) if os.path.isfile(man) else None)
    with Pool(min(a.jobs, len(a.scenes))) as pool:
        res = dict(pool.map(snap_scene, [(s, a.method, a.overwrite) for s in a.scenes]))
    me, rsh = sha256(os.path.abspath(__file__)), sha256(REF_SNAP)
    rs = _load("ref_snap", REF_SNAP)
    for s in a.scenes:
        r = res[s]
        man = dict(schema="MVSeed s2r6h_snap", schema_version=1, prereg="docs/stage2-R6H-prereg.md", dataset=s,
                   source_method=a.method, method=a.method + SUFFIX, source=src_info[s],
                   rule=dict(tau=rs.TAU, delta=rs.DELTA, dilate=rs.DILATE), ref_snap_sha256=rsh, script_sha256=me,
                   git_head=head, git_dirty=dirty, written_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **r)
        with open(os.path.join(DATA, CFG[s]["folder"], a.method + SUFFIX, "MANIFEST.json"), "w") as fh:
            json.dump(man, fh, indent=1)
        print(f"{s:20s} held-out {','.join(r['heldout']):24s} snapped {r['snapped']:5d} / {r['objects_frames']:5d}", flush=True)
    print(f"total snapped {sum(r['snapped'] for r in res.values())} / {sum(r['objects_frames'] for r in res.values())}")


if __name__ == "__main__":
    main()
