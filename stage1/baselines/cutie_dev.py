#!/usr/bin/env python3
"""Cutie (CVPR 2024, hkchengrex/Cutie, MIT) pseudo-video baseline on the stage-1 DEV sets (iccv2027-plan.md WP2 #6):
the cameras of one frame as a video, the reference camera's GT as the first-frame mask, Cutie's memory tracks the rest.
Cutie is forward-only, so the pseudo-video is walked twice from the reference's position with a fresh memory each
time: forward (ref -> later positions) and backward (ref -> earlier positions), exactly the two directions SAM 2 runs.
Output in the DEV layout of methods.geomem.dev_decode (scored by methods.geomem.dev_score); orders / frame loading /
writers shared with baselines.sam21_dev.

    scripts/docker_run.sh-style run inside scsam2 with PYTHONPATH=<repo>:<repo>/baselines/third_party/Cutie:
      python -m baselines.cutie_dev --order index [--ref-rot 0|1|2] [--scenes Dog]   -> Data/_dev/cutie_<order>_A[_rot<r>]

Model: cutie-base-mega.pth (baselines/weights/, release v1.0), eval_config with max_internal_size 480 (the scripting
demo's default; -1 would run at 1920 px), mem_every / top_k / max_mem_frames as in eval_config.  Code pinned by the
clone's commit (baselines/third_party/Cutie, recorded in index.json).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

S1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, S1)
from baselines.sam21_dev import DATA, MAIN, ORDERS, centres_of, frame_scene, sha256, view_order, write_mask  # noqa: E402
from geomem import devset  # noqa: E402

CUTIE_DIR = os.path.join(S1, "baselines", "third_party", "Cutie")
WEIGHTS = os.path.join(S1, "baselines", "weights", "cutie-base-mega.pth")
MAX_INTERNAL_SIZE = 480


def cutie_commit():
    try:
        return subprocess.check_output(["git", "-C", CUTIE_DIR, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_model(weights):
    """cutie.utils.get_default_model without its download step and with an absolute config dir."""
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from cutie.inference.utils.args_utils import get_dataset_cfg
    from cutie.model.cutie import CUTIE
    sys.path.insert(0, CUTIE_DIR)
    with initialize_config_dir(version_base="1.3.2", config_dir=os.path.join(CUTIE_DIR, "cutie", "config"), job_name="eval_config"):
        cfg = compose(config_name="eval_config")
    with open_dict(cfg):
        cfg["weights"] = weights
    get_dataset_cfg(cfg)
    model = CUTIE(cfg).cuda().eval()
    model.load_weights(torch.load(weights, map_location="cuda"))
    return model


def track(model, image_paths, seq, lab, objects, torch):
    """One direction: seq[0] is the reference (memorised with its mask), the rest are tracked in order.
    -> {cam: {obj: bool HxW}} for seq[1:]."""
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from cutie.inference.inference_core import InferenceCore
    proc = InferenceCore(model, cfg=model.cfg)
    proc.max_internal_size = MAX_INTERNAL_SIZE
    out = {}
    mask0 = torch.from_numpy(lab.astype(np.int64)).cuda()
    for i, cam in enumerate(seq):
        img = to_tensor(Image.open(image_paths[cam]).convert("RGB")).cuda().float()
        if i == 0:
            prob = proc.step(img, mask0, objects=list(objects))
            continue
        prob = proc.step(img)
        m = proc.output_prob_to_mask(prob).cpu().numpy()
        out[cam] = {int(o): m == o for o in objects}
    return out


def decode_frame(model, cams, ref, image_paths, lab, objects, order, torch):
    pos = order.index(ref)
    masks = {}
    masks.update(track(model, image_paths, order[pos:], lab, objects, torch))                 # forward
    masks.update(track(model, image_paths, order[pos::-1], lab, objects, torch))              # backward (ref first)
    return masks, dict(pseudo_video=order, ref_pos=pos, n_frames=len(order))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", choices=ORDERS, required=True)
    ap.add_argument("--split", default="A", choices=devset.SPLITS)
    ap.add_argument("--ref-rot", type=int, default=0, choices=devset.ROTS, dest="ref_rot")
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--root", default=None)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.split != "A":
        ap.error("DEV-B is kept for registrations' confirmations: baselines validate their harness on DEV-A / DEV-A' only")
    name = f"cutie_{a.order}_{a.split}" + (f"_rot{a.ref_rot}" if a.ref_rot else "")
    root = a.root or os.path.join(DATA, "_dev", name)
    dev = devset.dev_scenes()
    scenes = dev if a.scenes == "all" else a.scenes.split(",")
    bad = [s for s in scenes if s not in dev]
    if bad:
        ap.error(f"--scenes {bad}: not DEV scenes")
    os.makedirs(root, exist_ok=True)
    idx = devset.index(DATA, dev, a.split, a.ref_rot)
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(dict(preset=name, baseline="cutie-base-mega", order=a.order, weights=os.path.relpath(a.weights, MAIN),
                       weights_sha256=sha256(a.weights), cutie_commit=cutie_commit(), max_internal_size=MAX_INTERNAL_SIZE, **idx),
                  f, indent=1)
    import torch
    model = load_model(a.weights)
    knobs = {k: model.cfg.get(k) for k in ("mem_every", "max_mem_frames", "top_k", "stagger_updates", "use_long_term")}
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
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                masks, rec = decode_frame(model, cams, ref, image_paths, lab, objects, order, torch)
            if os.path.isdir(d):
                shutil.rmtree(d)
            for o in objects:
                write_mask(os.path.join(d, ref, f"{o}.png"), lab == o)
            n = 0
            for cam in evals:
                for o, m in masks.get(cam, {}).items():
                    write_mask(os.path.join(d, cam, f"{int(o)}.png"), m)
                    n += 1
            cfg = dict(baseline="cutie-base-mega", order=a.order, frame=t, ref_cam=ref, eval_cams=evals, objects=objects,
                       seconds=time.time() - t0, masks_written=n, ref_rot=int(a.ref_rot), max_internal_size=MAX_INTERNAL_SIZE,
                       **knobs, **rec)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f, indent=1)
            open(os.path.join(d, "DONE"), "w").close()
            n_done += 1
            print(f"{s:20s} t={t:4d} objs {len(objects):2d} cams {len(cams):2d} order {a.order} {cfg['seconds']:6.1f}s", flush=True)
    print(f"done {n_done} frames -> {root}")


if __name__ == "__main__":
    main()
