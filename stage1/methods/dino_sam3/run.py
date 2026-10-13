#!/usr/bin/env python3
"""dino_sam3 stage B: segment the best DINOv2 candidate point of each (camera, object) with SAM 3.

    scripts/run_dino_sam3.sh <scene|all> [--out NAME]

Reads Data/<scene>/Cache/dino/prompts.json (stage A, methods/dino_sam3/prompts.py).  The first
candidate (highest cosine) with cos >= TAU_SIM becomes one positive point; the mask SAM 3 returns
for it is the prediction.  No candidate above the threshold -> empty prediction.  Runs in `scsam3`.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from methods import common  # noqa: E402
from mvseg2d.io import cache_dir  # noqa: E402

TAU_SIM = 0.5    # E3 prereg: a candidate exists only at cosine >= this
FILL_HOLE_AREA = 0   # recorded in RUN.json; see methods/sam3_single.py


def pick(cands, tau=TAU_SIM):
    """The first candidate at or above `tau` (candidates arrive sorted by cosine), or None."""
    for c in cands:
        if c["cos"] >= tau:
            return c
    return None


def segment_scene(scene, args):
    from methods import sam3_single
    model = segment_scene.model
    if model is None:
        model = sam3_single.load_model(FILL_HOLE_AREA)
        segment_scene.model = model
    p = os.path.join(cache_dir(args.data, scene.name, "dino"), "prompts.json")
    with open(p) as f:
        prompts = json.load(f)
    written = none = 0
    for cam in scene.cams:
        if cam == scene.ref_cam:
            continue
        frame = None
        for o in scene.objects:
            c = pick(prompts["cams"].get(cam, {}).get(str(o), []))
            if c is None:
                none += 1
                continue
            if frame is None:
                frame, h, w = sam3_single.load_frame(scene.image_paths[cam])
                assert (h, w) == scene.hw
            m = sam3_single.segment(model, frame, h, w, [tuple(c["point"])], [1])
            if m.any():
                common.write_result(args.data, scene.name, args.out, cam, o, m)
                written += 1
        del frame
    import torch
    torch.cuda.empty_cache()
    return {"tau_sim": TAU_SIM, "k": prompts["k"], "long_side": prompts["long_side"], "model": prompts["model"],
            "nms_patches": prompts["nms_patches"], "fill_hole_area": FILL_HOLE_AREA, "masks_written": written, "no_candidate": none}


segment_scene.model = None

if __name__ == "__main__":
    common.run_all(common.parse_args(__doc__), segment_scene)
