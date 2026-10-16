#!/usr/bin/env python3
"""Full-length CROOM T06_flag_ganggang scenes at native 3840x2160 for the long runs (user request 2026-10-16:
"not the paper's conditions -- every frame, from frame 0, results on the NAS").

The only ground truth is source frame 900 (labelme polygons Data/Set9/images/NN.json, people p1..p6, c_ini cam_06), so
the take 0..9937 is covered by two runs of the same pipeline from that prompt:
    F (forward)   T06_flag_ganggang_F  Video/cam_KK/<f:06d>.jpg = source frame f, f = 0..9937; start_frame 900, num_frame 9038
    B (backward)  T06_flag_ganggang_B  Video/cam_KK/<i:06d>.jpg = source frame 900 - i, i = 0..900; start_frame 0, num_frame 901
B is the same forward tracker on time-reversed frames; merge_back renames its output folders i -> 900 - i.

    stage    copy CROOM/JPG/T06_flag_ganggang/<PERMS[K]>/<f>.jpg -> <root>/src/cam_KK/<f:06d>.jpg (byte copies, threaded,
             resumable by size; the runs read local copies so a NAS stall cannot block a multi-day run)
    scenes   build F and B: relative symlinks into <root>/src, 4K ground truth (plenoptic/build_dataset.rasterize at
             scale 1.0), objects_labels.json, SOURCE.json
    stage1   stage1/Data/<key>/ inputs for --stage1 (meta.json with external_fold / basic_objects, images, masks)
    merge_back <B method folder> <F method folder> : link B's frames 0..899 into F's tree under source frame numbers
Host python (numpy, PIL):  PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python plenoptic/build_full.py stage|scenes|stage1
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_JPG = os.path.join(REPO, "PlenopticServer1", "CROOM", "JPG", "T06_flag_ganggang")
ROOT = os.path.join(REPO, "Data", "PlenopticFull")
FRAME_GT, LAST = 900, 9937
SCENE_F, SCENE_B = "T06_flag_ganggang_F", "T06_flag_ganggang_B"
EXTERNAL_FOLD = 2        # fixed before any T06 run: the fold with the largest training set (14 scenes), lowest index


def _bd():
    spec = importlib.util.spec_from_file_location("bd", os.path.join(REPO, "plenoptic", "build_dataset.py"))
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    bd.SCALE, bd.W, bd.H = 1.0, bd.SRC_W, bd.SRC_H          # native resolution
    return bd


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def stage(root, threads):
    bd = _bd()
    jobs = []
    for k, jpg in enumerate(bd.PERMS):
        d = os.path.join(root, "src", bd.cam_name(k))
        os.makedirs(d, exist_ok=True)
        for f in range(0, LAST + 1):
            jobs.append((os.path.join(SRC_JPG, str(jpg), f"{f}.jpg"), os.path.join(d, f"{f:06d}.jpg")))

    def one(job):
        s, t = job
        if os.path.isfile(t) and os.path.getsize(t) == os.path.getsize(s):
            return 0
        tmp = t + ".part"
        shutil.copyfile(s, tmp)
        os.replace(tmp, t)
        return 1
    done = 0
    with ThreadPoolExecutor(threads) as ex:
        for i, n in enumerate(ex.map(one, jobs, chunksize=64)):
            done += n
            if i % 20000 == 0:
                print(f"stage {i}/{len(jobs)} ({done} copied)", flush=True)
    print(f"stage done: {len(jobs)} files, {done} copied this pass", flush=True)


def _gt(bd):
    out = {}
    for k, jpg in enumerate(bd.PERMS):
        nn = f"{bd.PNUM[jpg]:02d}"
        planes = bd.rasterize(os.path.join(bd.SRC_LAB, f"{nn}.json"))
        gt = np.zeros((bd.H, bd.W), dtype=np.uint8)
        for i, p in enumerate(bd.PEOPLE, start=1):
            gt[planes[p]] = i                                    # later ids overwrite earlier on overlap (as the paper)
        out[bd.cam_name(k)] = (gt, dict(jpg_folder=jpg, P=nn, areas={p: int(planes[p].sum()) for p in bd.PEOPLE}))
    return out


def scenes(root):
    bd = _bd()
    gts = _gt(bd)
    for key, frames, start, num in ((SCENE_F, list(range(0, LAST + 1)), FRAME_GT, LAST + 1 - FRAME_GT),
                                    (SCENE_B, [FRAME_GT - i for i in range(FRAME_GT + 1)], 0, FRAME_GT + 1)):
        sd = os.path.join(root, key)
        stats, digests = {}, {}
        for k in range(len(bd.PERMS)):
            cam = bd.cam_name(k)
            vd = os.path.join(sd, "Video", cam)
            os.makedirs(vd, exist_ok=True)
            for i, f in enumerate(frames):
                dst = os.path.join(vd, f"{i:06d}.jpg")
                if not os.path.lexists(dst):
                    os.symlink(os.path.relpath(os.path.join(root, "src", cam, f"{f:06d}.jpg"), vd), dst)
            md = os.path.join(sd, "Mask", cam)
            os.makedirs(md, exist_ok=True)
            out = os.path.join(md, f"{start:06d}.png")
            Image.fromarray(gts[cam][0], mode="L").save(out)
            digests[cam], stats[cam] = _sha(out), gts[cam][1]
        json.dump({str(i): {"label": "person", "type": "dynamic"} for i in range(1, 7)},
                  open(os.path.join(sd, "Mask", "objects_labels.json"), "w"), indent=1)
        src = dict(scene=key, frames=SRC_JPG, labels=bd.SRC_LAB, gt_source_frame=FRAME_GT,
                   direction="forward" if key == SCENE_F else "backward (time-reversed)",
                   file_index_to_source_frame=("f -> f (folder name = source frame)" if key == SCENE_F else "i -> 900 - i"),
                   start_frame=start, num_frame=num, ring_order_jpg_folders=bd.PERMS, resolution=[bd.W, bd.H], scale=1.0,
                   cameras=stats, gt_sha256=digests,
                   note="GT only at source frame 900 (labels p1..p6 -> ids 1..6); c_ini cam_06 as in docs/plenoptic-P-prereg.md")
        json.dump(src, open(os.path.join(sd, "SOURCE.json"), "w"), indent=1)
        print(f"{key}: {len(frames)} frames x {len(bd.PERMS)} cams linked, GT at index {start}", flush=True)


def stage1_inputs(root):
    """stage1/Data/<key>/ for --stage1 (both directions: the same frame-900 images and prompt)."""
    bd = _bd()
    for key, ref in ((SCENE_F, FRAME_GT), (SCENE_B, 0)):
        sd = os.path.join(root, key)
        d = os.path.join(REPO, "stage1", "Data", key)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        os.makedirs(os.path.join(d, "masks"), exist_ok=True)
        cams = [bd.cam_name(k) for k in range(len(bd.PERMS))]
        for k, cam in enumerate(cams):                     # source frame 900 straight from the NAS (byte copies)
            shutil.copyfile(os.path.join(SRC_JPG, str(bd.PERMS[k]), f"{FRAME_GT}.jpg"), os.path.join(d, "images", f"{cam}.jpg"))
        shutil.copyfile(os.path.join(sd, "Mask", "cam_06", f"{ref:06d}.png"), os.path.join(d, "masks", "cam_06.png"))
        shutil.copyfile(os.path.join(sd, "Mask", "objects_labels.json"), os.path.join(d, "objects_labels.json"))
        meta = dict(dataset=key, ref_frame=ref, ref_cam="cam_06", cams=cams, mask_cams=["cam_06"],
                    image_size=[bd.W, bd.H], basic_objects=[1, 2, 3, 4, 5, 6], external_fold=EXTERNAL_FOLD,
                    source=dict(frames=SRC_JPG, gt_source_frame=FRAME_GT, scene=sd))
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(f"stage1/Data/{key}: {len(cams)} images, mask cam_06, ref_frame {ref}, external fold {EXTERNAL_FOLD}", flush=True)


def merge_back(b_dir, f_dir):
    """Hard-link B's frames i = 1..900 (source 900 - i) into F's tree as source frame folders 0..899."""
    n = 0
    for cam in sorted(os.listdir(b_dir)):
        cb = os.path.join(b_dir, cam)
        if not os.path.isdir(cb):
            continue
        for i in range(1, FRAME_GT + 1):
            src = os.path.join(cb, str(i))
            if not os.path.isdir(src):
                continue
            dst = os.path.join(f_dir, cam, str(FRAME_GT - i))
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(src):
                t = os.path.join(dst, fn)
                if not os.path.exists(t):
                    os.link(os.path.join(src, fn), t)
                    n += 1
    print(f"merge_back: {n} PNGs linked into {f_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("stage", "scenes", "stage1", "merge_back"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "stage":
        stage(a.root, a.threads)
    elif a.cmd == "scenes":
        scenes(a.root)
    elif a.cmd == "stage1":
        stage1_inputs(a.root)
    else:
        merge_back(*a.args)


if __name__ == "__main__":
    main()
