import json
import os

import cv2
import numpy as np

import prepare_data


def _fake_mvseg(root, start=5, n_frames=3, cams=("v0", "v1"), mask_cams=("v1",)):
    ds = os.path.join(root, "Toy")
    for c in cams:
        os.makedirs(os.path.join(ds, "Video", c))
        for t in range(start, start + n_frames + 2):            # a couple of extra video frames
            cv2.imwrite(os.path.join(ds, "Video", c, f"{t:06d}.jpg"), np.zeros((8, 12, 3), np.uint8))
    for c in mask_cams:
        os.makedirs(os.path.join(ds, "Mask", c))
        for t in range(start, start + n_frames):
            lab = np.zeros((8, 12), np.uint8); lab[2:5, 3:9] = 1
            cv2.imwrite(os.path.join(ds, "Mask", c, f"{t:06d}.png"), lab)
    json.dump({"1": {"label": "x", "type": "static"}}, open(os.path.join(ds, "Mask", "objects_labels.json"), "w"))
    os.makedirs(os.path.join(ds, "MVSeed_gt"))
    json.dump({"start_frame": start, "c_ini": "v1"}, open(os.path.join(ds, "MVSeed_gt", "SEED_MANIFEST.json"), "w"))
    return ds


def test_prepare_train_frames_copies_labelled_frames(tmp_path):
    ds = _fake_mvseg(str(tmp_path / "src"))
    out = str(tmp_path / "Data" / "Toy")
    meta = prepare_data.prepare(ds, out, train_frames=True)
    assert meta["train_frames"] == [5, 6, 7]
    assert os.path.isfile(os.path.join(out, "frames", "000005", "images", "v0.jpg"))
    assert os.path.isfile(os.path.join(out, "frames", "000007", "images", "v1.jpg"))
    assert os.path.isfile(os.path.join(out, "frames", "000007", "masks", "v1.png"))
    assert not os.path.exists(os.path.join(out, "frames", "000008"))
    # the benchmark inputs are unchanged
    assert os.path.isfile(os.path.join(out, "images", "v0.jpg")) and os.path.isfile(os.path.join(out, "masks", "v1.png"))


def test_prepare_without_flag_has_no_frames(tmp_path):
    ds = _fake_mvseg(str(tmp_path / "src"))
    out = str(tmp_path / "Data" / "Toy")
    meta = prepare_data.prepare(ds, out)
    assert "train_frames" not in meta and not os.path.exists(os.path.join(out, "frames"))
