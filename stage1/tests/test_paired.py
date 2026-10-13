import numpy as np

from mvseg2d.paired import paired_scene_bootstrap, scene_means


def _rows(method, vals):
    out = []
    for scene, cams in vals.items():
        for cam, js in cams.items():
            for i, j in enumerate(js):
                out.append({"method": method, "scene": scene, "cam": cam, "obj": i + 1, "J": j, "F": j})
    return out


def test_scene_means_pair_view_scene():
    rows = _rows("a", {"S": {"c1": [1.0, 0.0], "c2": [1.0]}})          # view means 0.5 and 1.0 -> scene 0.75
    assert scene_means(rows, "J") == {"S": 0.75}


def test_paired_bootstrap_identical_methods_is_zero():
    a = _rows("a", {"S1": {"c": [0.5, 0.7]}, "S2": {"c": [0.9]}, "S3": {"c": [0.2]}})
    b = _rows("b", {"S1": {"c": [0.5, 0.7]}, "S2": {"c": [0.9]}, "S3": {"c": [0.2]}})
    r = paired_scene_bootstrap(a, b, "J", n=1000, seed=0)
    assert r["delta"] == 0.0 and r["ci"] == (0.0, 0.0) and r["n_scenes"] == 3


def test_paired_bootstrap_constant_shift():
    a = _rows("a", {f"S{i}": {"c": [0.5]} for i in range(6)})
    b = _rows("b", {f"S{i}": {"c": [0.6]} for i in range(6)})
    r = paired_scene_bootstrap(a, b, "J", n=500, seed=0)
    assert abs(r["delta"] - 0.1) < 1e-9 and abs(r["ci"][0] - 0.1) < 1e-9 and abs(r["ci"][1] - 0.1) < 1e-9


def test_paired_bootstrap_only_common_scenes():
    a = _rows("a", {"S1": {"c": [0.5]}, "S2": {"c": [0.5]}})
    b = _rows("b", {"S1": {"c": [0.7]}})
    r = paired_scene_bootstrap(a, b, "J", n=100, seed=0)
    assert r["n_scenes"] == 1 and abs(r["delta"] - 0.2) < 1e-9
