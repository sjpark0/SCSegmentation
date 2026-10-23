"""Baseline harness (iccv2027-plan.md WP2 #1): the SAM 2.1 pseudo-video orders are pure functions of the inputs."""
import numpy as np
import pytest

from baselines import sam21_dev as B


def test_orders():
    cams = ["cam9", "cam0", "cam4", "cam2"]
    assert B.order_index(cams, "cam4") == ["cam0", "cam2", "cam4", "cam9"]
    c = {"cam0": np.array([0.0, 0, 0]), "cam2": np.array([2.0, 0, 0]), "cam4": np.array([1.0, 0, 0]), "cam9": np.array([9.0, 0, 0])}
    assert B.order_nn(cams, "cam4", c) == ["cam4", "cam0", "cam2", "cam9"]   # from cam4 (x=1): cam0 (1) < cam2 (1)? tie -> number
    c["cam0"] = np.array([-0.5, 0, 0])
    assert B.order_nn(cams, "cam4", c) == ["cam4", "cam2", "cam0", "cam9"]   # cam2 at 1.0 beats cam0 at 1.5; then nearest to cam2
    assert B.view_order("index", cams, "cam4") == B.order_index(cams, "cam4")
    with pytest.raises(ValueError):
        B.view_order("random", cams, "cam4")
    assert B.cam_number("camera_0012") == 12 and B.cam_number("cam0") == 0


def test_frame_scene_restricts_to_basic():
    name, cams, ref, paths, lab, objects = B.frame_scene("Dog", 4, [1, 2], 0)
    assert name == "Dog" and ref == "camera_0002" and set(objects) <= {1, 2} and lab.dtype == np.uint8
    assert all(p.endswith(".jpg") for p in paths.values()) and len(paths) == len(cams)
