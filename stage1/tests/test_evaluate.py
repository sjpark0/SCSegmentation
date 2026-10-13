import pytest

from mvseg2d.evaluate import evaluate_method, evaluate_scene, scored_views, view_dist
from mvseg2d.io import SIZE_BINS, ShapeMismatch, load_meta


def test_scored_views_and_view_dist(fake_data):
    meta = load_meta(fake_data, "S1")
    assert scored_views(meta) == ["v0", "v4"]
    assert view_dist(meta, "v0") == 2 and view_dist(meta, "v4") == 2


def test_oracle_like_results_score_one(fake_data):
    r = evaluate_method(fake_data, "oracle", ["S1", "Dog"])
    rows = [row for s in r["scenes"].values() for row in s["rows"]]
    assert rows and all(row["J"] == 1.0 and row["F"] == 1.0 for row in rows)


def test_missing_png_counts_as_empty(fake_data):
    s = evaluate_scene(fake_data, "S1", "empty")
    assert s["status"] == "ok"
    assert s["rows"] and all(row["J"] == 0.0 and row["pred_px"] == 0 for row in s["rows"])


def test_gt_empty_pair_excluded_and_hallucination_counted(fake_data):
    s = evaluate_scene(fake_data, "S1", "halluc")
    assert not any(row["cam"] == "v4" and row["obj"] == 2 for row in s["rows"])
    assert s["hallucinations"] == 1


def test_error_run_json_marks_scene(fake_data):
    s = evaluate_scene(fake_data, "S1", "broken")
    assert s["status"] == "error" and s["rows"] == [] and "boom" in s["error"]


def test_missing_result_folder(fake_data):
    assert evaluate_scene(fake_data, "Dog", "nosuch")["status"] == "missing"


def test_shape_mismatch_rejected(fake_data):
    with pytest.raises(ShapeMismatch):
        evaluate_scene(fake_data, "S1", "wrongsize")


def test_reference_camera_never_scored(fake_data):
    s = evaluate_scene(fake_data, "S1", "oracle")
    assert {row["cam"] for row in s["rows"]} == {"v0", "v4"}


def test_unknown_label_and_extra_camera_warn(fake_data):
    s = evaluate_scene(fake_data, "S1", "oracle")
    assert any("v9" in w for w in s["warnings"])
    assert any(row["obj"] == 2 and row["label"] == "unknown" for row in s["rows"])
    assert any(row["obj"] == 1 and row["label"] == "person" for row in s["rows"])


def test_row_fields(fake_data):
    row = evaluate_scene(fake_data, "Dog", "oracle")["rows"][0]
    assert row["rig"] == "hemisphere" and row["size_bin"] in SIZE_BINS and row["view_dist"] >= 1
    assert row["gt_px"] > 0 and row["method"] == "oracle" and row["scene"] == "Dog"


def test_missing_run_json_is_incomplete(fake_data):
    """A result folder with PNGs but no RUN.json (killed before finishing) must not score as ok."""
    import os
    os.remove(os.path.join(fake_data, "Dog", "Result", "oracle", "RUN.json"))
    s = evaluate_scene(fake_data, "Dog", "oracle")
    assert s["status"] == "incomplete" and s["rows"] == []


def test_running_run_json_is_incomplete(fake_data):
    import json, os
    p = os.path.join(fake_data, "S1", "Result", "empty", "RUN.json")
    run = json.load(open(p)); run["status"] = "running"; json.dump(run, open(p, "w"))
    assert evaluate_scene(fake_data, "S1", "empty")["status"] == "incomplete"


def test_view_dist_uses_rig_order_not_lexicographic():
    cams = sorted(f"v{i}" for i in range(12))                 # meta.json's lexicographic listing: v0, v1, v10, v11, v2, ...
    meta = {"cams": cams, "ref_cam": "v2"}
    assert view_dist(meta, "v10") == 8 and view_dist(meta, "v0") == 2 and view_dist(meta, "v3") == 1
