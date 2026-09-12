#!/usr/bin/env python3
"""S1-E3 pass 1: DINOv2 correspondence -> candidate points.  (docs/stage1-E3-prereg.md section 2)

For every scene the reference camera's ground-truth mask of an object becomes a prototype
descriptor (the mean of the DINOv2 patch tokens whose patch centre lies inside the mask); the
cosine of that prototype against every patch token of a target view gives a similarity map, and
its top-K local maxima become K positive-point candidates.  Only LOCATIONS are produced here --
the segmentation is SAM 3's job in pass 2 (MVSeed/corr/e3_segment.py), which is why the patch
resolution that killed S1-E4b is survivable: a point inside the object is enough.

    docker run --rm --gpus all --user $(id -u):$(id -g) -v /:/host -w /host$PWD scgeo:dino \\
        python MVSeed/corr/dino_prompt.py [--scenes A,B]

Writes MVSeed/runs/e3_prompts.json: {scene: {cam: {obj: [{"point": [x, y], "cos": c}, ...]}}}
with points in ORIGINAL image pixel coordinates.  Runs in `scgeo:dino` (transformers + baked
DINOv2 weights); no other image in this repo has transformers.
"""
import argparse
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
OUT = os.path.join(REPO, "MVSeed", "runs", "e3_prompts.json")

MODEL_ID = "facebook/dinov2-large"   # Apache-2.0; DINOv3 excluded (gated licence, plan section 3)
PATCH = 14
LONG_SIDE = 1036                     # 74 patches on the long side (registered)
K_CANDIDATES = 5                     # registered
NMS_PATCHES = 3                      # registered: non-maximum suppression radius, in patches


def resized_hw(h, w, long_side=LONG_SIDE, patch=PATCH):
    """Aspect-preserving size whose sides are multiples of the patch size."""
    scale = long_side / max(h, w)
    rh = max(patch, int(h * scale) // patch * patch)
    rw = max(patch, int(w * scale) // patch * patch)
    return rh, rw


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


def peaks(sim_full, radius_px, k=K_CANDIDATES):
    """Top-k local maxima of the full-resolution similarity map, greedy NMS with a disc."""
    import numpy as np
    s = sim_full.copy()
    h, w = s.shape
    yy, xx = np.ogrid[:h, :w]
    out = []
    for _ in range(k):
        idx = int(np.argmax(s))
        y, x = divmod(idx, w)
        v = float(s[y, x])
        if not np.isfinite(v) or v <= -1.0:
            break
        out.append((x, y, v))
        s[(yy - y) ** 2 + (xx - x) ** 2 <= radius_px ** 2] = -2.0
    return out


def run_scene(scene, cfg, census, model, log=print):
    import cv2
    d = cfg[scene]
    start, folder = d["start_frame"], d["folder"]
    ref = census[scene]["c_ini"]
    targets = [c for c in census[scene]["cameras"] if c != ref]
    objects = [int(o) for o in census[scene]["seed_ids_c_ini"]]
    ds = os.path.join(DATA, folder)

    def frame(cam):
        p = os.path.join(ds, "Video", cam, f"{start:06d}.jpg")
        img = cv2.imread(p)
        assert img is not None, p
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ref_img = frame(ref)
    H, W = ref_img.shape[:2]
    rh, rw = resized_hw(H, W)
    gt = cv2.imread(os.path.join(ds, "Mask", ref, f"{start:06d}.png"), cv2.IMREAD_GRAYSCALE)
    assert gt is not None and gt.shape == (H, W)
    ref_tok = patch_tokens(model, ref_img, rh, rw)
    protos = {o: prototype(ref_tok, gt == o, rh, rw) for o in objects}
    radius_px = NMS_PATCHES * PATCH * max(H, W) / max(rh, rw)
    out = {}
    for cam in targets:
        img = frame(cam)
        th, tw = img.shape[:2]
        trh, trw = resized_hw(th, tw)
        tok = patch_tokens(model, img, trh, trw)
        flat = tok.reshape(-1, tok.shape[-1])
        per_obj = {}
        for o in objects:
            if protos[o] is None:
                per_obj[str(o)] = []
                continue
            sim = (flat @ protos[o]).reshape(tok.shape[0], tok.shape[1]).cpu().numpy()
            full = cv2.resize(sim, (tw, th), interpolation=cv2.INTER_LINEAR)
            per_obj[str(o)] = [dict(point=[int(x), int(y)], cos=round(float(v), 5))
                               for x, y, v in peaks(full, radius_px)]
        out[cam] = per_obj
        log(f"  {scene:16s} {cam:14s} {tw}x{th} -> grid {trw // PATCH}x{trh // PATCH}  "
            f"top cos median {sorted(c['cos'] for v in per_obj.values() for c in v[:1])[len(per_obj) // 2] if per_obj else None}")
    return dict(ref_cam=ref, size=[W, H], grid=[rw // PATCH, rh // PATCH], cams=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    import torch
    from transformers import AutoModel
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    census = json.load(open(CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(census)
    model = AutoModel.from_pretrained(MODEL_ID).cuda().eval()
    doc = {"schema": "MVSeed e3_prompts", "schema_version": 1, "prereg": "docs/stage1-E3-prereg.md",
           "rules": dict(model=MODEL_ID, long_side=LONG_SIDE, patch=PATCH, k=K_CANDIDATES,
                         nms_patches=NMS_PATCHES,
                         prototype="mean of patch tokens whose centre is inside the reference GT mask, "
                                   "bilinear sample at the mask centroid when none is",
                         points="original-resolution pixel coordinates of the similarity peaks"),
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "scenes": {}}
    t0 = time.time()
    with torch.inference_mode():
        for s in scenes:
            doc["scenes"][s] = run_scene(s, cfg, census, model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(doc, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}  ({len(doc['scenes'])} scenes, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
