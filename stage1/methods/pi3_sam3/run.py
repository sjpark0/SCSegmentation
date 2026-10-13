#!/usr/bin/env python3
"""pi3_sam3: refine another method's reprojected masks with SAM 3 point+box prompts.

    scripts/run_pi3_sam3.sh <scene|all> [--source pi3_reproj] [--out NAME]

For every non-reference camera and object, the source mask (default Result/pi3_reproj) gives the
position: its bounding box (labels 2, 3) and one positive point at the deepest interior pixel
(distance-transform argmax, label 1).  SAM 3 then draws the boundary on the full-resolution image.
An empty or missing source mask yields an empty prediction.  Runs inside `scsam3`.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from methods import common  # noqa: E402
from mvseg2d.io import read_pred_mask, result_dir  # noqa: E402

PROMPT_RULE = "bbox+dt_center"
FILL_HOLE_AREA = 0   # recorded in RUN.json; see methods/sam3_single.py


def prompts_from_mask(m):
    """None for an empty mask; else ([(x, y) tl, (x, y) br, (x, y) interior], [2, 3, 1])."""
    import cv2
    m = np.asarray(m, dtype=bool)
    if not m.any():
        return None
    rows = np.flatnonzero(m.any(1))
    cols = np.flatnonzero(m.any(0))
    tl = (int(cols[0]), int(rows[0]))
    br = (int(cols[-1]), int(rows[-1]))
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return [tl, br, (int(x), int(y))], [2, 3, 1]


def refine(scene, args):
    from methods import sam3_single
    src = result_dir(args.data, scene.name, args.source)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"source results missing: {src}")
    run_json = os.path.join(src, "RUN.json")
    if os.path.isfile(run_json):
        with open(run_json) as f:
            err = json.load(f).get("error")
        if err:
            raise RuntimeError(f"source {args.source} failed on {scene.name}: {err.strip().splitlines()[-1]}")
    model = refine.model
    if model is None:
        model = sam3_single.load_model(FILL_HOLE_AREA)
        refine.model = model
    written = empty = 0
    for cam in scene.cams:
        if cam == scene.ref_cam:
            continue
        frame = None
        for o in scene.objects:
            p = os.path.join(src, cam, f"{o}.png")
            pr = prompts_from_mask(read_pred_mask(p)) if os.path.isfile(p) else None
            if pr is None:
                empty += 1
                continue
            if frame is None:
                frame, h, w = sam3_single.load_frame(scene.image_paths[cam])
                assert (h, w) == scene.hw
            m = sam3_single.segment(model, frame, h, w, *pr)
            if m.any():
                common.write_result(args.data, scene.name, args.out, cam, o, m)
                written += 1
        del frame
    import torch
    torch.cuda.empty_cache()
    return {"source": args.source, "prompt_rule": PROMPT_RULE, "fill_hole_area": FILL_HOLE_AREA, "masks_written": written,
            "empty_source": empty}


refine.model = None


def _extra(ap):
    ap.add_argument("--source", default="pi3_reproj", help="Result folder whose masks give the prompts")


if __name__ == "__main__":
    common.run_all(common.parse_args(__doc__, _extra), refine)
