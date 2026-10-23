#!/usr/bin/env python3
"""SAM-V (gong208/SAM-V, arXiv 2609.25490; Apache-2.0 code, CC BY-NC weights) on the stage-1 DEV sets
(iccv2027-plan.md WP2 #2): the prompt-conditioned single-instance mode -- N frames of one scene + point prompts on one
frame -> that instance's mask on every frame.  Our input is the reference camera's GT mask, so (as the plan registers)
each object is prompted with NUM_POINTS = 10 points sampled uniformly from its reference mask (the benchmark's own
rule, benchmarks/compare_baseline_sam2.py build_sample, seed 0), one forward per object; the SAM image embeddings are
computed once per frame and reused across objects (sam_feats_precomputed), VGGT runs per forward.

Frame set: SAM-V's released evaluation uses 16 frames per sample, and VGGT-1B's aggregator scales with N, so a frame
is decoded with the reference, the eval cameras and the cameras nearest to the reference (Pi3 centre) up to
MAX_FRAMES = 16 (all cameras when the scene has at most 16).  The eval cameras are always in the set.
Images are resized to 1024 x 1024 for the model (its TARGET_SIZE); logits are resized back to the original size.

    docker run ... scsamv with PYTHONPATH=<repo>:<SAM-V>:<SAM-V>/submodules/sam-hq:<SAM-V>/submodules/vggt
      python -m baselines.samv_dev [--ref-rot 0|1|2] [--scenes Dog]   -> Data/_dev/samv_A[_rot<r>]
Weights: baselines/weights/samv/{sam_vit_h_4b8939.pth, vggt_model.pt, sam_v_stage2.pth}; code commit recorded.
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
from baselines.sam21_dev import DATA, MAIN, cam_number, centres_of, frame_scene, sha256, write_mask  # noqa: E402
from geomem import devset  # noqa: E402

SAMV_DIR = os.path.join(S1, "baselines", "third_party", "SAM-V")
WDIR = os.path.join(S1, "baselines", "weights", "samv")
W_SAM, W_VGGT, W_SAMV = (os.path.join(WDIR, n) for n in ("sam_vit_h_4b8939.pth", "vggt_model.pt", "sam_v_stage2.pth"))
NUM_POINTS, PROMPT_SEED, MAX_FRAMES, TARGET = 10, 0, 16, 1024


def samv_commit():
    try:
        return subprocess.check_output(["git", "-C", SAMV_DIR, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def frame_set(cams, ref, evals, centres, max_frames=MAX_FRAMES):
    """ref + evals + the other cameras nearest to the reference, at most max_frames; order = reference first, then by
    distance from the reference (ties by camera number)."""
    others = sorted((float(np.linalg.norm(centres[c] - centres[ref])), cam_number(c), c) for c in cams if c != ref)
    chosen = [ref] + list(evals)
    for _, _, c in others:
        if len(chosen) >= max_frames:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


def prompt_points(mask1024, num_points, seed, torch):
    """The benchmark's build_sample rule: num_points uniform foreground pixels of the 1024 x 1024 mask (with
    replacement when fewer), labels 1.  -> ([K,2] float (x, y), [K] long)."""
    torch.manual_seed(seed)
    ys, xs = torch.where(mask1024)
    if xs.numel() == 0:
        return None, None
    sel = torch.randint(0, xs.numel(), (num_points,)) if xs.numel() < num_points else torch.randperm(xs.numel())[:num_points]
    return torch.stack([xs[sel], ys[sel]], dim=1).float(), torch.ones(num_points, dtype=torch.long)


def load_model(torch):
    sys.path[:0] = [SAMV_DIR, os.path.join(SAMV_DIR, "submodules", "sam-hq"), os.path.join(SAMV_DIR, "submodules", "vggt")]
    from model.sam_vggt_model import SamVGGT
    from utils.checkpoint import is_partial_checkpoint, load_partial_checkpoint
    model = SamVGGT(sam_model_type="vit_h", sam_checkpoint=W_SAM, vggt_checkpoint=W_VGGT, device="cuda")
    ck = torch.load(W_SAMV, map_location="cuda")
    assert is_partial_checkpoint(ck), "sam_v_stage2.pth is expected to be a released (partial) checkpoint"
    load_partial_checkpoint(model, ck)
    return model.cuda().eval()


def decode_frame(model, cams_used, ref, image_paths, lab, objects, hw, torch):
    """-> {cam: {obj: bool HxW}} for every camera of cams_used except the reference."""
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    H, W = hw
    imgs = torch.stack([to_tensor(Image.open(image_paths[c]).convert("RGB")) for c in cams_used]).float()   # [N,3,H,W] in [0,1]
    imgs = F.interpolate(imgs, size=(TARGET, TARGET), mode="bilinear", align_corners=False) * 255.0        # the benchmark feeds 0-255
    sam_pre = imgs.unsqueeze(0).cuda()                                                                       # [1,N,3,1024,1024]
    lab1024 = F.interpolate(torch.from_numpy(lab.astype(np.float32))[None, None], size=(TARGET, TARGET), mode="nearest")[0, 0].long()
    out = {c: {} for c in cams_used if c != ref}
    N = len(cams_used)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        sam_feats, _ = model.encode_sam_batched(model.sam.preprocess(sam_pre))                                # [1,N,256,64,64]
        for o in objects:
            pts, lbl = prompt_points(lab1024 == int(o), NUM_POINTS, PROMPT_SEED, torch)
            if pts is None:
                continue
            res = model.forward(sam_pre=sam_pre, sam_feats_precomputed=sam_feats, point_coords_list=[pts.cuda()],
                                point_labels_list=[lbl.cuda()], point_frame_indices_list=[torch.zeros(NUM_POINTS, dtype=torch.long).cuda()],
                                multimask_output=False, visualize=False)
            lg = res["low_res_logits"][:, 0:1]                                                              # [1,1,h,w*N]
            h, w = lg.shape[2], lg.shape[3] // N
            per = lg[0, 0].reshape(h, N, w).permute(1, 0, 2).unsqueeze(1).float()                            # [N,1,h,w]
            per = F.interpolate(per, size=(H, W), mode="bilinear", align_corners=False)[:, 0]                # original size
            m = (per > 0).cpu().numpy()
            for i, c in enumerate(cams_used):
                if c != ref:
                    out[c][int(o)] = m[i]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="A", choices=devset.SPLITS)
    ap.add_argument("--ref-rot", type=int, default=0, choices=devset.ROTS, dest="ref_rot")
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--root", default=None)
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES, dest="max_frames")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.split != "A":
        ap.error("DEV-B is kept for registrations' confirmations: baselines validate their harness on DEV-A / DEV-A' only")
    name = f"samv_{a.split}" + (f"_rot{a.ref_rot}" if a.ref_rot else "")
    root = a.root or os.path.join(DATA, "_dev", name)
    dev = devset.dev_scenes()
    scenes = dev if a.scenes == "all" else a.scenes.split(",")
    bad = [s for s in scenes if s not in dev]
    if bad:
        ap.error(f"--scenes {bad}: not DEV scenes")
    os.makedirs(root, exist_ok=True)
    idx = devset.index(DATA, dev, a.split, a.ref_rot)
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(dict(preset=name, baseline="sam-v stage2 (vit_h + VGGT-1B)", num_points=NUM_POINTS, prompt_seed=PROMPT_SEED,
                       max_frames=a.max_frames, target_size=TARGET, samv_commit=samv_commit(),
                       weights={os.path.basename(p): sha256(p) for p in (W_SAM, W_VGGT, W_SAMV)}, **idx), f, indent=1)
    import torch
    model = load_model(torch)
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
            used = frame_set(cams, ref, evals, centres_of(DATA, s, t, cams), a.max_frames)
            masks = decode_frame(model, used, ref, image_paths, lab, objects, lab.shape, torch)
            if os.path.isdir(d):
                shutil.rmtree(d)
            for o in objects:
                write_mask(os.path.join(d, ref, f"{o}.png"), lab == o)
            n = 0
            for cam in evals:
                for o, m in masks.get(cam, {}).items():
                    write_mask(os.path.join(d, cam, f"{int(o)}.png"), m)
                    n += 1
            cfg = dict(baseline="sam-v stage2", frame=t, ref_cam=ref, eval_cams=evals, objects=objects, frames_used=used,
                       num_points=NUM_POINTS, prompt_seed=PROMPT_SEED, seconds=time.time() - t0, masks_written=n, ref_rot=int(a.ref_rot))
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f, indent=1)
            open(os.path.join(d, "DONE"), "w").close()
            n_done += 1
            print(f"{s:20s} t={t:4d} objs {len(objects):2d} frames {len(used):2d}/{len(cams):2d} {cfg['seconds']:6.1f}s", flush=True)
    print(f"done {n_done} frames -> {root}")


if __name__ == "__main__":
    main()
