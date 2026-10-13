from mvseg2d.report import flatten, headline, tables


def _row(method, scene, cam, obj, J, F=None, rig="pinhole", gt_px=500, vd=1):
    return {"method": method, "scene": scene, "cam": cam, "obj": obj, "label": "x", "type": "static",
            "rig": rig, "gt_px": gt_px, "pred_px": 1, "view_dist": vd,
            "size_bin": "<2000" if gt_px < 2000 else ("2000-10000" if gt_px < 10000 else ">=10000"),
            "J": J, "F": J if F is None else F}


def _scene(scene, rows, status="ok", error=None, halluc=0, seconds=3.0, gpu_mb=1000):
    return {"scene": scene, "method": rows[0]["method"] if rows else "m", "status": status,
            "error": error, "rows": rows, "hallucinations": halluc, "warnings": [],
            "run": {"seconds": seconds, "gpu_mb": gpu_mb} if status != "missing" else None}


SCORE = {"method": "m", "scenes": {
    # scene A: one view with J=1 and J=0 (view mean 0.5); scene B: one pair J=1
    "A": _scene("A", [_row("m", "A", "c1", 1, 1.0), _row("m", "A", "c1", 2, 0.0, gt_px=20000, vd=4)]),
    "B": _scene("B", [_row("m", "B", "c1", 1, 1.0, rig="hemisphere", gt_px=5000, vd=2)]),
}}
SCORE_ERR = {"method": "m", "scenes": {
    "A": _scene("A", [_row("m", "A", "c1", 1, 0.5)]),
    "S_err": _scene("S_err", [], status="error", error="Traceback: boom"),
}}


def test_flatten_skips_error_scenes():
    assert len(flatten(SCORE)) == 3 and len(flatten(SCORE_ERR)) == 1


def test_headline_scene_mean_not_pair_mean():
    h = headline(flatten(SCORE))
    assert abs(h["m"]["J"] - 0.75) < 1e-9 and h["m"]["n_scenes"] == 2


def test_error_scene_excluded_and_footnoted():
    md = tables([SCORE_ERR])
    assert "S_err" in md and "boom" in md
    assert "0.500" in md


def test_size_rig_and_viewdist_tables_present():
    md = tables([SCORE])
    assert "<2000" in md and ">=10000" in md
    assert "hemisphere" in md and "pinhole" in md
    assert ">=4" in md and "2-3" in md
    assert "GPU" in md


def test_incomplete_scene_footnoted_with_last_error_line():
    sc = _scene("S_inc", [], status="incomplete", error=None)
    err = _scene("S_err", [], status="error", error="Traceback (most recent call last):\n  ...\nRuntimeError: CUDA out of memory")
    md = tables([{"method": "m", "scenes": {"S_inc": sc, "S_err": err}}])
    assert "S_inc" in md and "incomplete" in md
    assert "CUDA out of memory" in md and "most recent call" not in md
