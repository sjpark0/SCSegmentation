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


def test_cutie_harness_pieces():
    from baselines import cutie_dev as C
    assert C.MAX_INTERNAL_SIZE == 480 and C.WEIGHTS.endswith("cutie-base-mega.pth")
    assert C.view_order is B.view_order and C.frame_scene is B.frame_scene      # shared orders / loading
    assert C.cutie_commit() is None or len(C.cutie_commit()) >= 7


def test_samv_frame_set():
    from baselines import samv_dev as V
    cams = [f"c{i}" for i in range(20)]
    centres = {c: np.array([float(i), 0, 0]) for i, c in enumerate(cams)}
    used = V.frame_set(cams, "c10", ["c0", "c19"], centres, max_frames=6)
    assert used[:3] == ["c10", "c0", "c19"] and len(used) == 6 and used[3:] == ["c9", "c11", "c8"]   # nearest to the reference
    assert V.frame_set(cams[:5], "c2", ["c0", "c4"], centres, max_frames=16) == ["c2", "c0", "c4", "c1", "c3"]
    assert V.NUM_POINTS == 10 and V.MAX_FRAMES == 16 and V.TARGET == 1024


def test_vggts_wrapper_pieces():
    import cv2
    from baselines import vggts_dev as G
    assert (G.IMG, G.SA_TOKENS, G.SAMPLE_NUM, G.REFINE_TIME, G.TAR_SQUARE) == (518, 37, 5, 1, 704)   # hybrid.yaml / val_epoch
    name, cams, ref, paths, lab, objects = G.frame_scene("Dog", 4, [1, 2], 0)
    img = G.prep(paths[ref], cv2)
    assert img.shape == (518, 518, 3) and img.dtype == np.uint8
    assert G.vs_commit() is None or len(G.vs_commit()) >= 7
