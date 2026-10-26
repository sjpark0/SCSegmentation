"""CROOM ring ground truth: label-map I/O (annot/labels.py), scene builder and ring scorer (scripts/croom/)."""
import importlib.util
import json
import os

import numpy as np
import pytest
from PIL import Image

from annot.labels import as_label, is_2d_png, read_label

S1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(S1, "scripts", "croom", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_as_label_rgba_channel0_and_mismatch():
    rgba = np.zeros((4, 5, 4), np.uint8)
    rgba[1:3, 1:3, :3] = 2
    rgba[..., 3] = 1                                  # the alpha garbage of the pre-fix server
    lab = as_label(rgba)
    assert lab.shape == (4, 5) and lab.dtype == np.uint8 and (lab[1:3, 1:3] == 2).all() and lab.sum() == 8
    bad = rgba.copy()
    bad[0, 0, 1] = 7
    with pytest.raises(ValueError):
        as_label(bad)
    assert as_label(np.ones((3, 3), np.uint8)).shape == (3, 3)


def _instant(tmp_path, n_cams=8, H=24, W=32):
    """A fake annotated instant: object 1 a square in every camera, object 2 only in cameras 0..3 (absent from 4..7)."""
    root = tmp_path / "CROOM"
    inst = root / "T09_fake_out" / "000100"
    (inst / "images").mkdir(parents=True)
    (inst / "annot" / "labels").mkdir(parents=True)
    for k in range(n_cams):
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(inst / "images" / f"{k}.jpg")
        lab = np.zeros((H, W), np.uint8)
        lab[2:10, 2:10] = 1
        if k < 4:
            lab[12:20, 12:28] = 2
        Image.fromarray(lab).save(inst / "annot" / "labels" / f"{k}.png")
    json.dump(dict(take="T09_fake_out", frame=100, cams=[str(k) for k in range(n_cams)], image_sha256={}),
              open(inst / "index.json", "w"))
    json.dump({"1": dict(name="p1", color=[1, 2, 3]), "2": dict(name="chair", color=[4, 5, 6])},
              open(inst / "annot" / "objects.json", "w"))
    return root


def test_build_scene_and_score(tmp_path):
    B, S = _load("build_scene"), _load("score")
    root = _instant(tmp_path)
    data = tmp_path / "Data"
    data.mkdir()
    out, meta = B.build("T09_fake_out", 100, 0, 2, str(root), str(data))
    assert os.path.basename(out) == "CROOM_T09_000100_r00" == meta["dataset"]
    assert meta["ref_cam"] == "cam_00" and meta["cams"][7] == "cam_07" and meta["basic_objects"] == [1, 2]
    assert meta["external_fold"] == 2 and meta["mask_cams"] == ["cam_00"]
    assert os.path.islink(os.path.join(out, "images", "cam_03.jpg")) and Image.open(os.path.join(out, "images", "cam_03.jpg")).size == (32, 24)
    assert is_2d_png(os.path.join(out, "masks", "cam_00.png")) and (read_label(os.path.join(out, "masks", "cam_00.png")) == 2).sum() == 128
    labels = json.load(open(os.path.join(out, "objects_labels.json")))
    assert labels["2"]["label"] == "chair"
    # a reference camera without object 2: it is not a basic object there
    out5, meta5 = B.build("T09_fake_out", 100, 5, 2, str(root), str(data))
    assert meta5["basic_objects"] == [1] and meta5["annot"]["objects_absent_in_ref"] == [2]
    with pytest.raises(SystemExit):
        B.build("T09_fake_out", 100, 5, 2, str(root), str(data))
    # predictions for the r00 scene: object 1 exact in every camera; object 2 predicted on object 1's pixels in cam 2
    # (a swap), empty in cam 3
    res = data / "CROOM_T09_000100_r00" / "Result" / "croom_X"
    for k in range(1, 8):
        (res / f"cam_{k:02d}").mkdir(parents=True)
        m = np.zeros((24, 32), np.uint8)
        m[2:10, 2:10] = 255
        Image.fromarray(m).save(res / f"cam_{k:02d}" / "1.png")
        if k == 2:
            Image.fromarray(m).save(res / f"cam_{k:02d}" / "2.png")
        if k == 1:
            m2 = np.zeros((24, 32), np.uint8)
            m2[12:20, 12:28] = 255
            Image.fromarray(m2).save(res / f"cam_{k:02d}" / "2.png")
    cams = meta["cams"]
    GTF, GTQ = S.load_gt(str(root / "T09_fake_out" / "000100" / "annot"), cams, 2)
    rows = S.score_dir(str(res), cams, "cam_00", [1, 2], GTF, GTQ, 2)
    assert len(rows) == 7 + 3                        # object 1 in 7 non-ref cams, object 2 in cams 1..3
    assert S.ring_distance(7, 0, 8) == 1 and S.ring_distance(4, 0, 8) == 4
    assert meta.get("ring_order") is None                 # 8 synthetic cameras: not the 32-camera CROOM rig
    assert len(B.RING_ORDER) == 32 and B.RING_ORDER[8] == "cam_16" and B.RING_ORDER[24] == "cam_08"
    # ring order: with cam_04 placed next to the reference, its pairs get distance 1 instead of 4
    order = ["cam_00", "cam_04", "cam_01", "cam_02", "cam_03", "cam_05", "cam_06", "cam_07"]
    rows_o = S.score_dir(str(res), cams, "cam_00", [1, 2], GTF, GTQ, 2, ring_order=order)
    d4 = {r["cam"]: r["d"] for r in rows_o}
    assert d4["cam_04"] == 1 and d4["cam_01"] == 2 and d4["cam_07"] == 1
    assert {r["d"] for r in rows if r["cam"] == "cam_07"} == {1}
    by = {(r["cam"], r["p"]): r for r in rows}
    assert by[("cam_01", 1)]["J"] == 1.0 and by[("cam_01", 2)]["J"] == 1.0
    assert by[("cam_02", 2)]["covers"] == 1 and by[("cam_02", 2)]["J"] == 0.0
    assert by[("cam_03", 2)]["covers"] == 0
    s = S.summarise(rows, 8)
    assert s["n_pairs"] == 10 and s["wrong_all"] == 1 and s["empty_all"] == 1 and s["swap_pattern"] == ["2->1 x1"]
    assert s["J_by_obj"]["1"] == 1.0 and s["J_by_d"]["4"] == 1.0
