#!/usr/bin/env python3
"""pi3_reproj: lift the reference camera's masks through Pi3X pointmaps and reproject them.

    scripts/run_pi3_reproj.sh <scene|all> [--pixel-limit N] [--out NAME] [--skip-geometry]

Stage A (GPU, geometry.py) caches depth/intrinsics/pose per camera at the Pi3 work resolution.
Stage B (numpy) warps each object's reference mask into every other view as the union of the
backward warp (target pixels whose surface point lands on the reference mask) and the forward
splat (reference mask pixels splatted into the target), both with the 1 % occlusion test of
mvseg2d/geo/warp.py, then upsamples to the original resolution by nearest sampling.
Runs in `scgeo:pi3`.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from methods import common  # noqa: E402
from methods.pi3_reproj.geometry import PIXEL_LIMIT, run_pi3  # noqa: E402
from mvseg2d.geo import warp  # noqa: E402
from mvseg2d.geo.cameras import from_c2w  # noqa: E402
from mvseg2d.io import cache_dir, labelmap_to_mask  # noqa: E402


def load_geometry(cache):
    """Cache/pi3 -> ({cam: dict(cam=Pinhole, pose=Pose, depth)}, manifest)."""
    with open(os.path.join(cache, "MANIFEST.json")) as f:
        manifest = json.load(f)
    geo = {}
    for c in manifest["cams"]:
        z = np.load(os.path.join(cache, f"{c}.npz"))
        cam, pose = from_c2w(z["K"], z["c2w"])
        geo[c] = dict(cam=cam, pose=pose, depth=z["depth"])
    return geo, manifest


def downsample_labels(ref_labels, work_hw):
    """Reference label map at the work resolution (nearest sampling at pixel centres)."""
    import cv2
    return cv2.resize(ref_labels, (work_hw[1], work_hw[0]), interpolation=cv2.INTER_NEAREST)


def upsample_mask(mask_work, orig_hw):
    import cv2
    return cv2.resize(mask_work.astype(np.uint8), (orig_hw[1], orig_hw[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def reproject(scene, geo, manifest, write):
    """Warp every object into every non-reference camera; `write(cam, obj, mask_bool)` per non-empty mask."""
    work_hw = tuple(manifest["work_hw"])
    ref = geo[scene.ref_cam]
    ref_lab = downsample_labels(scene.ref_labels, work_hw)
    written = 0
    for c in scene.cams:
        if c == scene.ref_cam:
            continue
        tgt = geo[c]
        corr_b = warp.backward_correspondence(ref["cam"], ref["pose"], ref["depth"],
                                              tgt["cam"], tgt["pose"], tgt["depth"])
        corr_f = warp.forward_correspondence(ref["cam"], ref["pose"], ref["depth"],
                                             tgt["cam"], tgt["pose"], tgt["depth"])
        for o in scene.objects:
            rm = ref_lab == o
            if not rm.any():
                continue
            m = warp.backward_from(corr_b, rm) | warp.forward_from(corr_f, rm)
            if m.any():
                write(c, o, upsample_mask(m, scene.hw))
                written += 1
    return written


def run(scene, args):
    cache = cache_dir(args.data, scene.name, "pi3" if args.pixel_limit == PIXEL_LIMIT else f"pi3_{args.pixel_limit}")
    if not (args.skip_geometry and os.path.isfile(os.path.join(cache, "MANIFEST.json"))):
        run_pi3(scene, cache, args.pixel_limit)
    geo, manifest = load_geometry(cache)
    written = reproject(scene, geo, manifest,
                        lambda cam, o, m: common.write_result(args.data, scene.name, args.out, cam, o, m))
    return {"pixel_limit": args.pixel_limit, "work_hw": manifest["work_hw"], "occ_tol": warp.OCC_TOL,
            "close_kernel": warp.CLOSE_KERNEL, "conf_min": manifest["conf_min"], "model": manifest["model"],
            "warp": "backward|forward", "masks_written": written, "cache": cache}


def _extra(ap):
    ap.add_argument("--pixel-limit", type=int, default=PIXEL_LIMIT)
    ap.add_argument("--skip-geometry", action="store_true", help="reuse Cache/pi3 if present")


if __name__ == "__main__":
    common.run_all(common.parse_args(__doc__, _extra), run)
