"""A two-scene fake Data/ tree plus several Result/<method> folders exercising the scorer's rules.

S1   pinhole, cams v0..v4, ref v2, annotated v0 v2 v4; objects 1 and 2 in every annotated view
     except object 2 absent from v4; objects_labels.json lists only id 1.
Dog  hemisphere, cams camera_0001..0004, ref camera_0002, annotated 0002 0003 0004; objects 1, 2.
"""
import json
import os

import cv2
import numpy as np
import pytest

H, W = 40, 60


def _labelmap(obj2=True):
    lab = np.zeros((H, W), np.uint8)
    lab[5:15, 5:25] = 1            # 200 px
    if obj2:
        lab[20:35, 30:55] = 2      # 375 px
    return lab


def _write_scene(root, scene, cams, ref, mask_cams, obj2_missing_in=()):
    d = os.path.join(root, scene)
    os.makedirs(os.path.join(d, "images"))
    os.makedirs(os.path.join(d, "masks"))
    for c in cams:
        cv2.imwrite(os.path.join(d, "images", f"{c}.jpg"), np.zeros((H, W, 3), np.uint8))
    for c in mask_cams:
        cv2.imwrite(os.path.join(d, "masks", f"{c}.png"), _labelmap(obj2=c not in obj2_missing_in))
    json.dump({"1": {"label": "person", "type": "dynamic"}},
              open(os.path.join(d, "objects_labels.json"), "w"))
    json.dump({"dataset": scene, "ref_frame": 0, "ref_cam": ref, "cams": cams,
               "mask_cams": mask_cams, "image_size": [W, H], "source": "fake"},
              open(os.path.join(d, "meta.json"), "w"))


def _run_json(root, scene, method, error=None):
    r = os.path.join(root, scene, "Result", method)
    os.makedirs(r, exist_ok=True)
    json.dump({"method": method, "scene": scene, "seconds": 1.0, "gpu_mb": None,
               "config": {}, "git": None, "date": "2026-10-06", "error": error},
              open(os.path.join(r, "RUN.json"), "w"))


def _png(root, scene, method, cam, obj, mask):
    p = os.path.join(root, scene, "Result", method, cam, f"{obj}.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, mask.astype(np.uint8) * 255)


@pytest.fixture
def fake_data(tmp_path):
    root = str(tmp_path / "Data")
    _write_scene(root, "S1", ["v0", "v1", "v2", "v3", "v4"], "v2", ["v0", "v2", "v4"],
                 obj2_missing_in=("v4",))
    _write_scene(root, "Dog", ["camera_0001", "camera_0002", "camera_0003", "camera_0004"],
                 "camera_0002", ["camera_0002", "camera_0003", "camera_0004"])
    # oracle: GT copied for every annotated camera (reference included), plus a stray folder v9
    for scene, mask_cams in (("S1", ["v0", "v2", "v4"]),
                             ("Dog", ["camera_0002", "camera_0003", "camera_0004"])):
        _run_json(root, scene, "oracle")
        for c in mask_cams:
            lab = cv2.imread(os.path.join(root, scene, "masks", f"{c}.png"), cv2.IMREAD_GRAYSCALE)
            for o in (1, 2):
                if (lab == o).any():
                    _png(root, scene, "oracle", c, o, lab == o)
    _png(root, "S1", "oracle", "v9", 1, _labelmap() == 1)
    # empty: RUN.json only
    _run_json(root, "S1", "empty")
    # halluc: like oracle on S1 but also predicts object 2 on v4 where GT has none
    _run_json(root, "S1", "halluc")
    for c in ("v0", "v4"):
        _png(root, "S1", "halluc", c, 1, _labelmap() == 1)
    _png(root, "S1", "halluc", "v0", 2, _labelmap() == 2)
    _png(root, "S1", "halluc", "v4", 2, _labelmap() == 2)
    # broken: RUN.json with an error
    _run_json(root, "S1", "broken", error="Traceback: boom")
    # wrongsize: a PNG at half resolution
    _run_json(root, "S1", "wrongsize")
    _png(root, "S1", "wrongsize", "v0", 1, np.ones((H // 2, W // 2), bool))
    return root
