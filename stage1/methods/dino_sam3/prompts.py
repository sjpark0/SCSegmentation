#!/usr/bin/env python3
"""dino_sam3 stage A: DINOv2 correspondence -> candidate positive points per (camera, object).

    (run via scripts/run_dino_sam3.sh; this stage runs in scgeo:dino)
    python methods/dino_sam3/prompts.py --scene <name|all> --data Data

The reference camera's mask of an object becomes a prototype (mean of the DINOv2 patch tokens whose
patch centre lies inside the mask); its cosine against every patch token of a target view gives a
similarity map whose top-K local maxima are the candidates.  Writes
Data/<scene>/Cache/dino/prompts.json = {cam: {obj: [{"point": [x, y], "cos": c}, ...]}} in original
pixel coordinates, candidates sorted by cosine descending.  Constants were fixed in
../SCSegmentation/docs/stage1-E3-prereg.md before this benchmark existed and are not tuned here.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from methods import common  # noqa: E402
from mvseg2d.io import cache_dir, labelmap_to_mask  # noqa: E402

MODEL_ID = "facebook/dinov2-large"   # Apache-2.0
PATCH = 14
LONG_SIDE = 1036                     # 74 patches on the long side (E3 prereg)
K_CANDIDATES = 5                     # E3 prereg
NMS_PATCHES = 3                      # E3 prereg: non-maximum suppression radius, in patches


def resized_hw(h, w, long_side=LONG_SIDE, patch=PATCH):
    """Aspect-preserving size whose sides are multiples of the patch size."""
    scale = long_side / max(h, w)
    rh = max(patch, int(h * scale) // patch * patch)
    rw = max(patch, int(w * scale) // patch * patch)
    return rh, rw


def peaks(sim_full, radius_px, k=K_CANDIDATES):
    """Top-k local maxima of the full-resolution similarity map, greedy NMS with a disc."""
    import numpy as np
    s = sim_full.astype("float32").copy()
    h, w = s.shape
    yy, xx = np.ogrid[:h, :w]
    out = []
    for _ in range(k):
        idx = int(np.argmax(s))
        y, x = divmod(idx, w)
        v = float(s[y, x])
        if not np.isfinite(v) or v <= -1.0:
            break
        out.append({"point": [int(x), int(y)], "cos": round(v, 5)})
        s[(yy - y) ** 2 + (xx - x) ** 2 <= radius_px ** 2] = -2.0
    return out


def patch_tokens(model, image_rgb, rh, rw, device="cuda"):
    """L2-normalised patch tokens of one image as (rows, cols, C)."""
    import cv2
    import numpy as np
    import torch
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    img = cv2.resize(image_rgb, (rw, rh), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    x = torch.from_numpy(((img - mean) / std).transpose(2, 0, 1))[None].to(device)
    with torch.no_grad():
        out = model(pixel_values=x).last_hidden_state[0, 1:]        # drop CLS
    out = torch.nn.functional.normalize(out.float(), dim=-1)
    return out.reshape(rh // PATCH, rw // PATCH, -1)


def prototype(tokens, mask, rh, rw):
    """Mean of the tokens whose patch centre is inside the mask; bilinear fallback for tiny masks."""
    import cv2
    import numpy as np
    import torch
    gr, gc = tokens.shape[0], tokens.shape[1]
    small = cv2.resize(mask.astype("uint8"), (rw, rh), interpolation=cv2.INTER_NEAREST)
    rows = (np.arange(gr) * PATCH + PATCH // 2).clip(0, rh - 1)
    cols = (np.arange(gc) * PATCH + PATCH // 2).clip(0, rw - 1)
    inside = small[rows[:, None], cols[None, :]] > 0
    if inside.any():
        v = tokens[torch.from_numpy(inside).to(tokens.device)].mean(0)
    else:
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return None
        cy = float(ys.mean()) / mask.shape[0] * gr - 0.5
        cx = float(xs.mean()) / mask.shape[1] * gc - 0.5
        grid = torch.tensor([[[[cx / max(gc - 1, 1) * 2 - 1, cy / max(gr - 1, 1) * 2 - 1]]]],
                            dtype=torch.float32, device=tokens.device)
        v = torch.nn.functional.grid_sample(tokens.permute(2, 0, 1)[None], grid,
                                            align_corners=True)[0, :, 0, 0]
    return torch.nn.functional.normalize(v, dim=-1)


def run_scene(scene, out_dir, model):
    import cv2
    H, W = scene.hw
    rh, rw = resized_hw(H, W)

    def frame(cam):
        return cv2.cvtColor(cv2.imread(scene.image_paths[cam]), cv2.COLOR_BGR2RGB)

    ref_tok = patch_tokens(model, frame(scene.ref_cam), rh, rw)
    protos = {o: prototype(ref_tok, labelmap_to_mask(scene.ref_labels, o), rh, rw) for o in scene.objects}
    radius_px = NMS_PATCHES * PATCH * max(H, W) / max(rh, rw)
    out = {}
    for cam in scene.cams:
        if cam == scene.ref_cam:
            continue
        tok = patch_tokens(model, frame(cam), rh, rw)
        flat = tok.reshape(-1, tok.shape[-1])
        per_obj = {}
        for o in scene.objects:
            if protos[o] is None:
                per_obj[str(o)] = []
                continue
            sim = (flat @ protos[o]).reshape(tok.shape[0], tok.shape[1]).cpu().numpy()
            full = cv2.resize(sim, (W, H), interpolation=cv2.INTER_LINEAR)
            per_obj[str(o)] = peaks(full, radius_px)
        out[cam] = per_obj
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prompts.json"), "w") as f:
        json.dump({"model": MODEL_ID, "long_side": LONG_SIDE, "k": K_CANDIDATES, "nms_patches": NMS_PATCHES,
                   "ref_cam": scene.ref_cam, "grid": [rw // PATCH, rh // PATCH], "cams": out}, f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default=os.path.join(common.REPO, "Data"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID).eval().cuda()
    scenes = common.all_scenes(a.data) if a.scene == "all" else [a.scene]
    for name in scenes:
        out_dir = cache_dir(a.data, name, "dino")
        if os.path.isfile(os.path.join(out_dir, "prompts.json")) and not a.overwrite:
            print(f"{name:20s} skip (prompts.json exists)", flush=True)
            continue
        scene = common.load_scene(a.data, name)
        run_scene(scene, out_dir, model)
        torch.cuda.empty_cache()
        print(f"{name:20s} prompts for {len(scene.cams) - 1} cams x {len(scene.objects)} objects", flush=True)


if __name__ == "__main__":
    main()
