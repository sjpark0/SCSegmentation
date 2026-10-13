#!/usr/bin/env python3
"""Check method: copies the ground truth of every annotated camera.  The scorer must give J=F=1.

This is the only method allowed to read masks/<cam>.png for cameras other than the reference.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from methods import common  # noqa: E402
from mvseg2d.io import labelmap_to_mask, load_labelmap  # noqa: E402


def oracle(scene, args):
    for cam in scene.mask_cams:
        if cam == scene.ref_cam:
            continue
        lab = load_labelmap(os.path.join(args.data, scene.name, "masks", f"{cam}.png"))
        for o in scene.objects:
            m = labelmap_to_mask(lab, o)
            if m.any():
                common.write_result(args.data, scene.name, args.out, cam, o, m)
    return {"kind": "oracle"}


if __name__ == "__main__":
    common.run_all(common.parse_args(__doc__), oracle)
