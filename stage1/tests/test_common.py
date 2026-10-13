import argparse
import json
import os

import numpy as np

from methods import common


def test_load_scene_fields(fake_data):
    sc = common.load_scene(fake_data, "S1")
    assert sc.name == "S1" and sc.ref_cam == "v2" and sc.cams == ["v0", "v1", "v2", "v3", "v4"]
    assert sc.hw == (40, 60) and sc.objects == [1, 2]
    assert sc.image_paths["v3"].endswith(os.path.join("S1", "images", "v3.jpg"))
    assert sc.ref_labels.dtype == np.uint8 and (sc.ref_labels == 2).sum() == 375


def test_all_scenes(fake_data):
    assert common.all_scenes(fake_data) == ["Dog", "S1"]


def test_write_result_and_copy_reference(fake_data):
    sc = common.load_scene(fake_data, "S1")
    common.copy_reference(sc, fake_data, "m")
    common.write_result(fake_data, "S1", "m", "v0", 1, np.ones((40, 60), bool))
    assert os.path.isfile(os.path.join(fake_data, "S1", "Result", "m", "v2", "1.png"))
    assert os.path.isfile(os.path.join(fake_data, "S1", "Result", "m", "v2", "2.png"))
    assert os.path.isfile(os.path.join(fake_data, "S1", "Result", "m", "v0", "1.png"))


def test_write_run_schema(fake_data):
    common.write_run(fake_data, "S1", "m", 2.5, None, {"k": 1})
    run = json.load(open(os.path.join(fake_data, "S1", "Result", "m", "RUN.json")))
    assert run["method"] == "m" and run["scene"] == "S1" and run["seconds"] == 2.5
    assert run["config"] == {"k": 1} and run["error"] is None and "date" in run and "git" in run


def test_run_all_continues_after_exception(fake_data):
    calls = []

    def fn(scene, args):
        calls.append(scene.name)
        if scene.name == "Dog":
            raise RuntimeError("boom")
        return {"ok": True}

    args = argparse.Namespace(scene="all", data=fake_data, out="m", overwrite=True)
    common.run_all(args, fn)
    assert calls == ["Dog", "S1"]
    dog = json.load(open(os.path.join(fake_data, "Dog", "Result", "m", "RUN.json")))
    s1 = json.load(open(os.path.join(fake_data, "S1", "Result", "m", "RUN.json")))
    assert "boom" in dog["error"] and s1["error"] is None and s1["config"] == {"ok": True}
    # reference camera copied for both scenes before fn ran
    assert os.path.isfile(os.path.join(fake_data, "Dog", "Result", "m", "camera_0002", "1.png"))


def test_run_all_clears_stale_results_before_rerun(fake_data):
    """A PNG left by a previous run must not survive into a re-run that no longer predicts it."""
    stale = os.path.join(fake_data, "S1", "Result", "m", "v0", "2.png")
    common.write_result(fake_data, "S1", "m", "v0", 2, np.ones((40, 60), bool))
    common.write_run(fake_data, "S1", "m", 1.0, None, {})
    args = argparse.Namespace(scene="S1", data=fake_data, out="m", overwrite=True)
    common.run_all(args, lambda scene, a: {"ok": True})
    assert not os.path.exists(stale)
    assert os.path.isfile(os.path.join(fake_data, "S1", "Result", "m", "v2", "1.png"))   # reference re-copied


def test_run_all_marks_interrupted_run(fake_data):
    """KeyboardInterrupt (or any BaseException) leaves a RUN.json that says the run did not finish."""
    def fn(scene, args):
        raise KeyboardInterrupt
    args = argparse.Namespace(scene="S1", data=fake_data, out="m", overwrite=True)
    import pytest
    with pytest.raises(KeyboardInterrupt):
        common.run_all(args, fn)
    run = json.load(open(os.path.join(fake_data, "S1", "Result", "m", "RUN.json")))
    assert run["error"] and "KeyboardInterrupt" in run["error"]


def test_run_all_writes_running_marker_first(fake_data):
    seen = {}

    def fn(scene, args):
        seen["run"] = json.load(open(os.path.join(fake_data, scene.name, "Result", "m", "RUN.json")))
        return {}
    args = argparse.Namespace(scene="S1", data=fake_data, out="m", overwrite=True)
    common.run_all(args, fn)
    assert seen["run"]["status"] == "running"
    assert json.load(open(os.path.join(fake_data, "S1", "Result", "m", "RUN.json")))["status"] == "done"


def test_git_hash_marks_dirty_tree(monkeypatch):
    monkeypatch.setattr(common.subprocess, "check_output",
                        lambda cmd, **kw: "abc1234\n" if "rev-parse" in cmd else " M file.py\n")
    assert common._git_hash() == "abc1234-dirty"
    monkeypatch.setattr(common.subprocess, "check_output",
                        lambda cmd, **kw: "abc1234\n" if "rev-parse" in cmd else "")
    assert common._git_hash() == "abc1234"


def test_load_scene_frame(fake_data):
    """With frame=t the scene reads frames/<t>/ instead of the reference-frame folders."""
    import cv2
    import numpy as np
    d = os.path.join(fake_data, "S1", "frames", "000003")
    os.makedirs(os.path.join(d, "images")); os.makedirs(os.path.join(d, "masks"))
    for c in ["v0", "v1", "v2", "v3", "v4"]:
        cv2.imwrite(os.path.join(d, "images", f"{c}.jpg"), np.zeros((40, 60, 3), np.uint8))
    lab = np.zeros((40, 60), np.uint8); lab[1:3, 1:3] = 5
    for c in ["v0", "v2", "v4"]:
        cv2.imwrite(os.path.join(d, "masks", f"{c}.png"), lab)
    sc = common.load_scene(fake_data, "S1", frame=3)
    assert sc.frame == 3 and sc.objects == [5]
    assert sc.image_paths["v1"].endswith(os.path.join("frames", "000003", "images", "v1.jpg"))
    assert common.load_scene(fake_data, "S1").frame is None


def test_run_all_clears_and_marks_aux_outputs(fake_data):
    """Auxiliary result folders (e.g. `<out>_r1` of a two-round method) get the same treatment as `out`:
    cleared before fn runs, a 'running' marker first, and an error record if fn fails."""
    stale = os.path.join(fake_data, "S1", "Result", "m_r1", "v0", "2.png")
    common.write_result(fake_data, "S1", "m_r1", "v0", 2, np.ones((40, 60), bool))
    common.write_run(fake_data, "S1", "m_r1", 1.0, None, {})
    seen = {}

    def fn(scene, args):
        seen["stale"] = os.path.exists(stale)
        seen["run"] = json.load(open(os.path.join(fake_data, "S1", "Result", "m_r1", "RUN.json")))
        raise RuntimeError("boom")

    args = argparse.Namespace(scene="S1", data=fake_data, out="m", overwrite=True)
    common.run_all(args, fn, aux_outs=["m_r1"])
    assert seen["stale"] is False and seen["run"]["status"] == "running"
    aux = json.load(open(os.path.join(fake_data, "S1", "Result", "m_r1", "RUN.json")))
    assert aux["error"] and "boom" in aux["error"]
