import argparse
import json
import os

import pytest

from methods import common
from methods.geomem.run import assert_no_leak, checkpoint_for

FOLDS = {"S1": 0, "Dog": 1}


def _runs(tmp_path, arm="geomem", folds=(0, 1), scenes_by_fold=None):
    runs = str(tmp_path / "runs")
    for k in folds:
        d = os.path.join(runs, arm, f"fold{k}"); os.makedirs(d)
        open(os.path.join(d, "ckpt.pt"), "wb").close()
        json.dump({"arm": arm, "fold": k, "scenes": (scenes_by_fold or {}).get(k, []), "ckpt_sha256": "0" * 64, "git": "abc"},
                  open(os.path.join(d, "config.json"), "w"))
    return runs


def test_checkpoint_for_uses_scene_fold(tmp_path):
    runs = _runs(tmp_path)
    assert checkpoint_for("Dog", "geomem", runs, FOLDS) == os.path.join(runs, "geomem", "fold1", "ckpt.pt")
    assert checkpoint_for("S1", "geomem", runs, FOLDS) == os.path.join(runs, "geomem", "fold0", "ckpt.pt")


def test_checkpoint_for_missing_mentions_fold(tmp_path):
    runs = _runs(tmp_path, folds=(0,))
    with pytest.raises(FileNotFoundError, match="fold1"):
        checkpoint_for("Dog", "geomem", runs, FOLDS)


def test_run_refuses_leaking_checkpoint():
    with pytest.raises(RuntimeError, match="S1"):
        assert_no_leak({"scenes": ["S1", "Barn"], "fold": 0}, "S1")
    assert_no_leak({"scenes": ["Barn"], "fold": 0}, "S1")       # fine


def test_run_missing_checkpoint_is_error(fake_data, tmp_path):
    runs = _runs(tmp_path, folds=(0,))

    def fn(scene, args):
        checkpoint_for(scene.name, "geomem", runs, FOLDS)
        return {}
    args = argparse.Namespace(scene="Dog", data=fake_data, out="geomem", overwrite=True)
    common.run_all(args, fn)
    run = json.load(open(os.path.join(fake_data, "Dog", "Result", "geomem", "RUN.json")))
    assert run["error"] and "fold1" in run["error"]


def test_select_neighbours_k_nearest_filtered():
    import numpy as np
    from methods.geomem.run import select_neighbours
    centres = {"c0": np.array([0, 0, 0.]), "c1": np.array([1, 0, 0.]), "c2": np.array([2, 0, 0.]),
               "c3": np.array([3, 0, 0.]), "c4": np.array([9, 0, 0.])}
    conf = {"c0": True, "c1": True, "c2": False, "c3": True, "c4": True}
    assert select_neighbours("c0", centres, conf, k=3) == ["c1", "c3"]          # c2 nearest-3 but not confident
    assert select_neighbours("c0", centres, {c: False for c in centres}, k=3) == []
    assert "c0" not in select_neighbours("c0", centres, conf, k=5)



def test_hub_stats_uniform_and_peaked():
    import numpy as np
    from methods.geomem.run import hub_stats
    uni = np.full((2, 3, 4), 0.25)
    s = hub_stats(uni, ref_index=0)
    assert abs(s["entropy_mean"] - 1.0) < 1e-6 and abs(s["ref_frac_mean"] - 0.25) < 1e-6 and abs(s["top3_frac_mean"] - 0.75) < 1e-6
    peak = np.zeros((2, 3, 4)); peak[..., 0] = 1.0
    s = hub_stats(peak, ref_index=0)
    assert s["entropy_mean"] == 0.0 and s["ref_frac_mean"] == 1.0 and s["top3_frac_mean"] == 1.0


# ---------------------------------------------------------------- GeoSet-2: object gate helpers and --r1-out
def test_select_candidates_k_nearest_unfiltered():
    import numpy as np
    from methods.geomem.run import select_candidates
    centres = {"c0": np.array([0, 0, 0.]), "c1": np.array([1, 0, 0.]), "c2": np.array([2, 0, 0.]),
               "c3": np.array([3, 0, 0.]), "c4": np.array([9, 0, 0.])}
    assert select_candidates("c0", centres, k=3) == ["c1", "c2", "c3"]
    assert "c0" not in select_candidates("c0", centres, k=5)


def test_object_conf_rule():
    torch = pytest.importorskip("torch")
    from methods.geomem.run import object_conf
    conf = object_conf(torch.tensor([1.0, -1.0, 1.0]), torch.tensor([0.9, 0.9, 0.4]), tau=0.7)
    assert conf.tolist() == [True, False, False]


def test_r1_out_none_keeps_existing_r1(fake_data, tmp_path):
    from methods.geomem.run import r1_aux_outs
    r1 = os.path.join(fake_data, "S1", "Result", "x_r1")
    os.makedirs(r1)
    with open(os.path.join(r1, "RUN.json"), "w") as f:
        json.dump({"status": "done"}, f)
    assert r1_aux_outs("x", "none") == [] and r1_aux_outs("x", None) == ["x_r1"] and r1_aux_outs("x", "y") == ["y"]
    args = argparse.Namespace(scene="S1", data=fake_data, out="x_obj", overwrite=True)
    common.run_all(args, lambda scene, a: {}, aux_outs=r1_aux_outs("x_obj", "none"))
    assert os.path.isfile(os.path.join(r1, "RUN.json"))
    assert os.path.isfile(os.path.join(fake_data, "S1", "Result", "x_obj", "RUN.json"))


def test_obj_gate_refuses_the_arm_name_as_out():
    from methods.geomem.run import check_gate_out
    with pytest.raises(SystemExit):
        check_gate_out("geoset_pred_ft", "obj", "geoset_pred_ft")        # would overwrite the view-gate result
    check_gate_out("geoset_pred_ft", "obj", "geoset_pred_ft_obj")
    check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft")


# ---------------------------------------------------------------- GeoSet-2 §8: gate signals
def test_object_conf_mode_rules():
    import numpy as np
    from methods.geomem.run import object_conf_mode
    presence = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    ious = np.array([0.9, 0.9, 0.4, 0.4, 0.9])
    overlap = np.array([0.9, 0.9, 0.9, 0.2, np.nan])        # nan: too few valid reprojected pixels
    assert object_conf_mode("obj", presence, ious, overlap).tolist() == [True, False, False, False, True]
    assert object_conf_mode("absent", presence, ious, overlap).tolist() == [True, False, True, True, True]
    # reproj: present and overlap >= 0.5; nan falls back to the IoU rule (object 4: iou 0.9 -> True)
    assert object_conf_mode("reproj", presence, ious, overlap).tolist() == [True, False, True, False, True]
    assert object_conf_mode("reproj", presence, ious, np.array([0.9, 0.9, 0.9, 0.2, np.nan]) * 0 + np.nan).tolist() \
        == [True, False, False, False, True]


# ---------------------------------------------------------------- GeoAlign spec 4: aligned memory neighbours
def test_aligned_views_exclude_reference():
    import numpy as np
    from methods.geomem.run import aligned_neighbours
    centres = {f"c{i}": np.array([float(i), 0, 0]) for i in range(5)}
    assert aligned_neighbours("c0", list(centres), centres, k=3) == ["c1", "c2", "c3"]


def test_aligned_neighbours_small_scene():
    import numpy as np
    from methods.geomem.run import aligned_neighbours
    centres = {f"c{i}": np.array([float(i), 0, 0]) for i in range(3)}
    assert aligned_neighbours("c1", list(centres), centres, k=3) == ["c0", "c2"]


def test_aligned_neighbours_exclude_target():
    """Review follow-up: a target must not be its own aligned memory source; the next nearest camera fills in."""
    import numpy as np
    from methods.geomem.run import aligned_neighbours
    centres = {f"c{i}": np.array([float(i), 0, 0]) for i in range(6)}
    assert aligned_neighbours("c0", list(centres), centres, k=3, exclude="c2") == ["c1", "c3", "c4"]
    assert aligned_neighbours("c0", list(centres), centres, k=3, exclude="c5") == ["c1", "c2", "c3"]
    assert aligned_neighbours("c0", ["c0", "c1", "c2"], centres, k=3, exclude="c2") == ["c1"]
