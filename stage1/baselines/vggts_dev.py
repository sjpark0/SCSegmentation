#!/usr/bin/env python3
"""VGGT-Segmentor (buaa-colalab/VGGT-S, CVPR 2026; Apache-2.0) on the stage-1 DEV sets (iccv2027-plan.md WP2 #3):
pairwise cross-view segmentation -- source = the reference camera's image + one object's GT mask, target = another
camera's image -> that object's mask in the target.  There is no released inference script, so this is a wrapper over
the repo's own pieces exactly as its validation loop uses them (src/main_exp/hybrid.py val_epoch):
  1. source / target images resized to 518 x 518 (INTER_AREA), the source mask to 518 x 518 (nearest) -- the
     repo's uncropped "ego target" path (ego_tar); the exo-side object crop (exo_src / exo_tar, src_R 2, meta) is
     for small egocentric objects and is NOT used here (MUVOD objects are large; recorded in index.json);
  2. 5 k-means query points on the source mask (vggt.sample_num), VGGT-1B aggregator + track head on the pair with
     `feat=True` (the README's track-head patch) -> feature maps and the tracked target points;
  3. the predictor (main_exp.pth) with refine_time 1 (two passes, the second fed the first's low-res mask);
  4. the 518 x 518 binary prediction resized back to the original size (nearest).
Targets are batched (BATCH pairs per forward).  Weights: third_party/VGGT-S/third_party/vggt_main/model.pt (VGGT-1B,
the same file as baselines/weights/samv/vggt_model.pt) and third_party/VGGT-S/official_ckpts/main_exp.pth.

    docker run ... scsamv with PYTHONPATH=<repo>:<VGGT-S>/src:<VGGT-S>/third_party/vggt_main
      python -m baselines.vggts_dev [--ref-rot 0|1|2] [--scenes Dog]   -> Data/_dev/vggts_A[_rot<r>]
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
from baselines.sam21_dev import DATA, MAIN, frame_scene, sha256, write_mask  # noqa: E402
from geomem import devset  # noqa: E402

VS_DIR = os.path.join(S1, "baselines", "third_party", "VGGT-S")
VGGT_CKPT = os.path.join(VS_DIR, "third_party", "vggt_main", "model.pt")
VS_CKPT = os.path.join(VS_DIR, "official_ckpts", "main_exp.pth")
IMG, SA_TOKENS, SAMPLE_NUM, REFINE_TIME, TAR_SQUARE, BATCH = 518, 37, 5, 1, 704, 8


def vs_commit():
    try:
        return subprocess.check_output(["git", "-C", VS_DIR, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_models(torch):
    sys.path[:0] = [os.path.join(VS_DIR, "src"), os.path.join(VS_DIR, "third_party", "vggt_main")]
    from vggt.models.vggt import VGGT
    from model.predictor import Predictor
    vggt = VGGT()
    vggt.load_state_dict(torch.load(VGGT_CKPT, map_location="cpu"))
    vggt = vggt.cuda().eval()
    predictor = Predictor(img_size=IMG, sa_tokens=SA_TOKENS, resume_ckpt=VS_CKPT).cuda().eval()
    return vggt, predictor


def prep(img_bgr_path, cv2):
    img = cv2.cvtColor(cv2.imread(img_bgr_path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)          # uint8 [518,518,3]


def predict_pairs(vggt, predictor, src_img, src_mask, tar_imgs, torch):
    """src_img uint8 [518,518,3], src_mask uint8 {0,1} [518,518], tar_imgs uint8 [B,518,518,3]
    -> bool [B,518,518] (the repo's val_epoch for a batch of B source-target pairs)."""
    from utils import mask_sampling, sample_feat_at_points
    B = len(tar_imgs)
    src_b = np.repeat(src_img[None], B, axis=0)
    mask_b = np.repeat(src_mask[None], B, axis=0).astype(np.uint8)
    src_points = torch.as_tensor(mask_sampling(mask_b[:, None], n_samples=SAMPLE_NUM)).float().squeeze(1).cuda()   # [B,N,2]
    pair = torch.stack([torch.from_numpy(src_b / 255.0).permute(0, 3, 1, 2).float(),
                        torch.from_numpy(tar_imgs / 255.0).permute(0, 3, 1, 2).float()], dim=1).cuda().contiguous()  # [B,2,3,H,W]
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        tokens, patch_start = vggt.aggregator(pair)
        coords, _, _, feat = vggt.track_head(tokens, images=pair, patch_start_idx=patch_start, query_points=src_points, feat=True)
        feat = feat.float()
        tar_points = coords[-1][:, 1].float()                                   # the tracked points in the target [B,N,2]
        src_feat = sample_feat_at_points(feat[:, 0], src_points, IMG, IMG).float()
        tar_in = None
        for _ in range(REFINE_TIME + 1):
            pred, low, _, _ = predictor(feat_map_b=feat, src_points_b=src_points, tar_points_b=tar_points, src_mask_b=mask_b,
                                        src_points_feat_b=src_feat, tar_mask_b=tar_in, ret_logits=False, meta=[None] * B,
                                        tar_square=TAR_SQUARE)
            tar_in = low.float()
    return pred[:, 0].bool().cpu().numpy()


def decode_frame(vggt, predictor, cams, ref, image_paths, lab, objects, hw, torch, cv2):
    H, W = hw
    src = prep(image_paths[ref], cv2)
    targets = [c for c in cams if c != ref]
    tar_imgs = np.stack([prep(image_paths[c], cv2) for c in targets])
    out = {c: {} for c in targets}
    for o in objects:
        m518 = cv2.resize((lab == o).astype(np.uint8), (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        if m518.sum() == 0:
            continue
        for i in range(0, len(targets), BATCH):
            pred = predict_pairs(vggt, predictor, src, m518, tar_imgs[i:i + BATCH], torch)
            for j, c in enumerate(targets[i:i + BATCH]):
                out[c][int(o)] = cv2.resize(pred[j].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="A", choices=devset.SPLITS)
    ap.add_argument("--ref-rot", type=int, default=0, choices=devset.ROTS, dest="ref_rot")
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--root", default=None)
    ap.add_argument("--eval-only", action="store_true", dest="eval_only",
                    help="decode the eval cameras only (what dev_score reads); default decodes every camera")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.split != "A":
        ap.error("DEV-B is kept for registrations' confirmations: baselines validate their harness on DEV-A / DEV-A' only")
    name = f"vggts_{a.split}" + (f"_rot{a.ref_rot}" if a.ref_rot else "")
    root = a.root or os.path.join(DATA, "_dev", name)
    dev = devset.dev_scenes()
    scenes = dev if a.scenes == "all" else a.scenes.split(",")
    bad = [s for s in scenes if s not in dev]
    if bad:
        ap.error(f"--scenes {bad}: not DEV scenes")
    os.makedirs(root, exist_ok=True)
    idx = devset.index(DATA, dev, a.split, a.ref_rot)
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(dict(preset=name, baseline="vggt-s main_exp (VGGT-1B + predictor)", img_size=IMG, sample_num=SAMPLE_NUM,
                       refine_time=REFINE_TIME, crop="none (ego_tar-style full-frame resize for source and target)",
                       vggts_commit=vs_commit(), weights={os.path.basename(p): sha256(p) for p in (VGGT_CKPT, VS_CKPT)}, **idx),
                  f, indent=1)
    import cv2
    import torch
    vggt, predictor = load_models(torch)
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
            use = [ref] + list(evals) if a.eval_only else cams
            masks = decode_frame(vggt, predictor, use, ref, image_paths, lab, objects, lab.shape, torch, cv2)
            if os.path.isdir(d):
                shutil.rmtree(d)
            for o in objects:
                write_mask(os.path.join(d, ref, f"{o}.png"), lab == o)
            n = 0
            for cam in evals:
                for o, m in masks.get(cam, {}).items():
                    write_mask(os.path.join(d, cam, f"{int(o)}.png"), m)
                    n += 1
            cfg = dict(baseline="vggt-s main_exp", frame=t, ref_cam=ref, eval_cams=evals, objects=objects, decoded_cams=use,
                       seconds=time.time() - t0, masks_written=n, ref_rot=int(a.ref_rot), img_size=IMG, sample_num=SAMPLE_NUM,
                       refine_time=REFINE_TIME)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f, indent=1)
            open(os.path.join(d, "DONE"), "w").close()
            n_done += 1
            print(f"{s:20s} t={t:4d} objs {len(objects):2d} cams {len(use):2d} {cfg['seconds']:6.1f}s", flush=True)
    print(f"done {n_done} frames -> {root}")


if __name__ == "__main__":
    main()
