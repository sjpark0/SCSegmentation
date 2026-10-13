"""load_sample on tiny fake caches (CPU)."""
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.cache import feat_cache_path, pi3_frame_cache, pos_enc_path  # noqa: E402
from geomem.data import Sample, load_sample  # noqa: E402


def _scene(root, hw=(40, 60)):
    s = os.path.join(root, "S1")
    cams, mask_cams = ["v0", "v1", "v2"], ["v0", "v2"]
    d = os.path.join(s, "frames", "000003")
    os.makedirs(os.path.join(d, "images")); os.makedirs(os.path.join(d, "masks"))
    for c in cams:
        cv2.imwrite(os.path.join(d, "images", f"{c}.jpg"), np.zeros((*hw, 3), np.uint8))
    for c in mask_cams:
        lab = np.zeros(hw, np.uint8); lab[5:15, 5:25] = 1; lab[20:30, 30:50] = 2
        cv2.imwrite(os.path.join(d, "masks", f"{c}.png"), lab)
    json.dump({"dataset": "S1", "ref_frame": 0, "ref_cam": "v0", "cams": cams, "mask_cams": mask_cams,
               "image_size": [hw[1], hw[0]], "train_frames": [3]}, open(os.path.join(s, "meta.json"), "w"))
    for c in mask_cams:
        p = feat_cache_path(root, "S1", 3, c); os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save({"image": torch.zeros(3, 16, 16, dtype=torch.half),
                    "backbone_fpn": [torch.zeros(1, 4, 8, 8, dtype=torch.half), torch.zeros(1, 4, 4, 4, dtype=torch.half),
                                     torch.zeros(1, 8, 2, 2, dtype=torch.half)], "hw": hw}, p)
    torch.save([torch.zeros(1, 8, 8, 8, dtype=torch.half), torch.zeros(1, 8, 4, 4, dtype=torch.half),
                torch.zeros(1, 8, 2, 2, dtype=torch.half)], pos_enc_path(root, "S1"))
    cache = pi3_frame_cache(root, "S1", 3); os.makedirs(cache)
    K = np.array([[50.0, 0, 30], [0, 50.0, 20], [0, 0, 1]])
    for c in cams:
        np.savez(os.path.join(cache, f"{c}.npz"), depth=np.full(hw, 2.0, np.float32), conf=np.ones(hw, np.float32), K=K, c2w=np.eye(4))
    json.dump({"cams": cams, "work_hw": list(hw)}, open(os.path.join(cache, "MANIFEST.json"), "w"))
    return root


def test_load_sample_shapes(tmp_path):
    root = _scene(str(tmp_path))
    b = load_sample(root, Sample("S1", 3, "v0", "v2"), [1, 2], device="cpu")
    assert b["ref_masks"].shape == (2, 40, 60) and b["tgt_masks"].shape == (2, 40, 60)
    assert b["ref_image"].shape == (1, 3, 16, 16) and b["ref_image"].dtype == torch.float32
    assert len(b["tgt_backbone"]["backbone_fpn"]) == 3 and len(b["tgt_backbone"]["vision_pos_enc"]) == 3
    assert b["p_ref"].shape == (5184, 3) and b["valid_tgt"].shape == (5184,) and b["valid_tgt"].all()
    assert b["hw"] == (40, 60) and b["objects"] == [1, 2]


# ---------------------------------------------------------------- GeoFuse spec 2.5: per-frame level-72 feature cache
def _view_caches(root, cams, t=3):
    s = os.path.join(root, "S1")
    os.makedirs(s, exist_ok=True)
    json.dump({"dataset": "S1", "ref_frame": 0, "ref_cam": cams[0], "cams": cams, "mask_cams": cams[:2],
               "image_size": [40, 60], "train_frames": [t]}, open(os.path.join(s, "meta.json"), "w"))
    for i, c in enumerate(cams):
        p = feat_cache_path(root, "S1", t, c)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save({"image": torch.zeros(3, 8, 8, dtype=torch.float16),
                    "backbone_fpn": [torch.zeros(1, 32, 8, 8, dtype=torch.float16), torch.zeros(1, 64, 8, 8, dtype=torch.float16),
                                     torch.full((1, 256, 72, 72), float(i), dtype=torch.float16)],
                    "hw": (40, 60)}, p)
    return t


def test_build_level72_from_view_caches(tmp_path):
    from geomem.cache import level72_path
    from geomem.cache_feats import build_level72
    from geomem.data import load_level72
    root = str(tmp_path); cams = ["v0", "v1", "v2"]
    t = _view_caches(root, cams)
    build_level72(root, "S1", t)
    assert os.path.isfile(level72_path(root, "S1", t))
    got_cams, feats = load_level72(root, "S1", t, "cpu")
    assert got_cams == cams and feats.shape == (3, 256, 72, 72) and feats.dtype == torch.float16
    assert feats[2].float().mean().item() == 2.0


def test_level72_cams_match_meta(tmp_path):
    import pytest
    from geomem.cache import level72_path
    from geomem.data import load_level72
    root = str(tmp_path); cams = ["v0", "v1", "v2"]
    t = _view_caches(root, cams)
    p = level72_path(root, "S1", t); os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"cams": ["v2", "v1", "v0"], "feats": torch.zeros(3, 256, 72, 72, dtype=torch.float16)}, p)
    with pytest.raises(AssertionError):
        load_level72(root, "S1", t, "cpu")


def test_level72_missing_names_file(tmp_path):
    import pytest
    from geomem.data import load_level72
    root = str(tmp_path); _view_caches(root, ["v0", "v1"])
    with pytest.raises(FileNotFoundError) as e:
        load_level72(root, "S1", 7, "cpu")
    assert "000007.pt" in str(e.value)
