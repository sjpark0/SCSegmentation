#!/usr/bin/env python3
"""SAM 2.1 pseudo-video propagation baseline on the stage-1 DEV sets (iccv2027-plan.md WP2 #1, the G0-style control
with SAM 2.1 instead of SAM 3): the cameras of one frame are stacked as a video, the reference camera's GT masks are the
prompts, SAM 2.1 propagates in both directions; the eval cameras' masks are written in the DEV layout of
methods.geomem.dev_decode (Data/_dev/<name>/<scene>/<t:06d>/<cam>/<obj>.png + index.json + DONE), so
methods.geomem.dev_score scores it unchanged.

    scripts/docker_run.sh scsam2 python -m baselines.sam21_dev --order index [--split A] [--ref-rot 0|1|2] [--scenes Dog]
        -> Data/_dev/sam21_<order>_<split>[_rot<r>]

Orders (registered before any TEST run; both reported):
  index  camera-number order (what SCSam3/runMVSeg.py and MVSeed/run_seed.py --order index do)
  nn     greedy nearest-neighbour chain from the reference: the next camera is the unvisited one closest (Pi3 camera
         centre, Cache/pi3_frames/<t>) to the previous one -- adjacent pseudo-frames stay similar
The prompt is at the reference's position; propagation runs forward and backward from it (SAM 2's own rule).
Image scsam2 = SCSam2/Dockerfile (torch 2.6 cu118, SAM 2 1.0 at /opt/sam2); weights SCSam2/models/sam2.1_hiera_large.pt
through the /host mount.  Per frame: a temp dir of symlinks "<pos>.jpg" (SAM 2 reads integer-named JPEGs).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

S1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, S1)
from geomem import devset  # noqa: E402
from geomem.cache import pi3_frame_cache  # noqa: E402
from methods import common  # noqa: E402
from mvseg2d.io import labelmap_objects, load_labelmap  # noqa: E402

DATA = os.path.join(S1, "Data")
MAIN = os.path.dirname(S1)
CKPT = os.path.join(MAIN, "SCSam2", "models", "sam2.1_hiera_large.pt")
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
ORDERS = ("index", "nn")


def cam_number(c):
    d = "".join(ch for ch in c if ch.isdigit())
    return int(d) if d else 0


def order_index(cams, ref_cam):
    return sorted(cams, key=lambda c: (cam_number(c), c))


def order_nn(cams, ref_cam, centres):
    """Greedy nearest-neighbour chain starting at the reference (ties by camera number)."""
    left = [c for c in cams if c != ref_cam]
    out, cur = [ref_cam], ref_cam
    while left:
        nxt = min(left, key=lambda c: (float(np.linalg.norm(centres[c] - centres[cur])), cam_number(c), c))
        out.append(nxt)
        left.remove(nxt)
        cur = nxt
    return out


def view_order(order, cams, ref_cam, centres=None):
    if order == "index":
        return order_index(cams, ref_cam)
    if order == "nn":
        return order_nn(cams, ref_cam, centres)
    raise ValueError(order)


def centres_of(data, scene, t, cams):
    d = pi3_frame_cache(data, scene, t)
    return {c: np.load(os.path.join(d, f"{c}.npz"))["c2w"][:3, 3].astype(np.float64) for c in cams}


def frame_scene(scene_name, t, basic, rot=0):
    """As dev_decode.frame_scene: frame t, the input camera of rotation `rot`, its GT restricted to the basic set."""
    sc = common.load_scene(DATA, scene_name, frame=t)
    ref = devset.ref_cam(DATA, scene_name, rot)
    lab = sc.ref_labels if ref == sc.ref_cam else load_labelmap(os.path.join(common.frame_dir(DATA, scene_name, t), "masks", f"{ref}.png"))
    lab = lab.copy()
    lab[~np.isin(lab, basic)] = 0
    return sc.name, list(sc.cams), ref, sc.image_paths, lab, labelmap_objects(lab)


def write_mask(path, m):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (np.asarray(m) > 0).astype(np.uint8) * 255)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def decode_frame(predictor, cams, ref, image_paths, lab, objects, order, torch):
    """-> ({cam: {obj: bool HxW}}, record): the pseudo-video in `order`, the GT masks prompted at the reference's position,
    forward + backward propagation."""
    pos = {c: i for i, c in enumerate(order)}
    tmp = tempfile.mkdtemp(prefix="sam21_")
    try:
        for i, c in enumerate(order):
            os.symlink(os.path.abspath(image_paths[c]), os.path.join(tmp, f"{i}.jpg"))
        state = predictor.init_state(video_path=tmp, offload_video_to_cpu=False, async_loading_frames=False)
        rp = pos[ref]
        for o in objects:
            predictor.add_new_mask(state, rp, int(o), torch.from_numpy(lab == o))
        masks = {}
        for reverse in (False, True):
            for fi, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=rp, reverse=reverse):
                cam = order[fi]
                masks[cam] = {int(o): (logits[i, 0] > 0.0).cpu().numpy() for i, o in enumerate(obj_ids)}
        predictor.reset_state(state)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return masks, dict(pseudo_video=order, ref_pos=rp, n_frames=len(order))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", choices=ORDERS, required=True)
    ap.add_argument("--split", default="A", choices=devset.SPLITS)
    ap.add_argument("--ref-rot", type=int, default=0, choices=devset.ROTS, dest="ref_rot")
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--cfg", default=CFG)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.split != "A":
        ap.error("DEV-B is kept for registrations' confirmations: baselines validate their harness on DEV-A / DEV-A' only")
    name = f"sam21_{a.order}_{a.split}" + (f"_rot{a.ref_rot}" if a.ref_rot else "")
    root = a.root or os.path.join(DATA, "_dev", name)
    dev = devset.dev_scenes()
    scenes = dev if a.scenes == "all" else a.scenes.split(",")
    bad = [s for s in scenes if s not in dev]
    if bad:
        ap.error(f"--scenes {bad}: not DEV scenes")
    os.makedirs(root, exist_ok=True)
    idx = devset.index(DATA, dev, a.split, a.ref_rot)
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(dict(preset=name, baseline="sam2.1_hiera_large", order=a.order, ckpt=os.path.relpath(a.ckpt, MAIN),
                       ckpt_sha256=sha256(a.ckpt), cfg=a.cfg, **idx), f, indent=1)
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(a.cfg, a.ckpt, device="cuda")
    n_done = 0
    for s in scenes:
        basic = idx["scenes"][s]["basic"]
        for t in idx["scenes"][s]["frames"]:
            d = os.path.join(root, s, f"{t:06d}")
            if os.path.isfile(os.path.join(d, "DONE")) and not a.overwrite:
                continue
            t0 = time.time()
            scene, cams, ref, image_paths, lab, objects = frame_scene(s, t, basic, a.ref_rot)
            evals = devset.eval_cams(DATA, s, a.ref_rot)
            assert ref not in evals, (s, t, ref, evals)
            centres = centres_of(DATA, s, t, cams) if a.order == "nn" else None
            order = view_order(a.order, cams, ref, centres)
            assert sorted(order) == sorted(cams) and order.count(ref) == 1
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, rec = decode_frame(predictor, cams, ref, image_paths, lab, objects, order, torch)
            if os.path.isdir(d):
                shutil.rmtree(d)
            for o in objects:                                   # the reference: its (restricted) GT, as dev_decode
                write_mask(os.path.join(d, ref, f"{o}.png"), lab == o)
            n = 0
            for cam in evals:
                for o, m in masks.get(cam, {}).items():
                    write_mask(os.path.join(d, cam, f"{int(o)}.png"), m)
                    n += 1
            cfg = dict(baseline="sam2.1_hiera_large", order=a.order, frame=t, ref_cam=ref, eval_cams=evals, objects=objects,
                       seconds=time.time() - t0, masks_written=n, ref_rot=int(a.ref_rot), **rec)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f, indent=1)
            open(os.path.join(d, "DONE"), "w").close()
            n_done += 1
            print(f"{s:20s} t={t:4d} objs {len(objects):2d} cams {len(cams):2d} order {a.order} {cfg['seconds']:6.1f}s", flush=True)
    print(f"done {n_done} frames -> {root}")


if __name__ == "__main__":
    main()
