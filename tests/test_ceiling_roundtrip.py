"""S1-E0': MVSeed/ceiling_roundtrip.py, the CPU rebuild of the mask round trip.

The two halves of the round trip are checked against the REAL package code on bare
instances -- the tracker's _consolidate_temp_output_across_obj (the 288 downsample) and
the inference wrapper's _convert_low_res_mask_to_video_res (the upsample + threshold) --
so the script is tied to the chain it claims to reproduce, not to a re-typing of it.
The overlap rule, the size bins, the pre-registered reading and the CLI shape are
goldens on synthetic masks; the one data test runs the smallest scene when the data is
present.  Container only (torch; the package halves also need sam3).
"""
import importlib.util
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import REPO  # noqa: E402

SCRIPT = os.path.join(REPO, "MVSeed", "ceiling_roundtrip.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("ceiling_roundtrip", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load_script()
HW = (1080, 1920)


def rect(hw, y0, y1, x0, x1):
    m = np.zeros(hw, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def iou(a, b):
    return cr.db_eval_iou(a, b)


# ------------------------------------------------------------ the two halves
def test_downsample_is_the_trackers_consolidation(cpu_tensors):
    """Step 2 on the real _consolidate_temp_output_across_obj: a bare tracker, one
    object whose temp output carries the +-1024 video-res mask add_new_mask stores."""
    pytest.importorskip("sam3")
    from harness import bare_tracker
    tr = bare_tracker(use_sel=False)
    tr.low_res_mask_size = cr.LOW_RES
    mask = rect((360, 640), 100, 250, 200, 500)
    video_res = torch.where(torch.as_tensor(mask), torch.tensor(cr.NO_OBJ_SCORE),
                            torch.tensor(-cr.NO_OBJ_SCORE))[None, None]
    assert video_res.dtype == torch.float32
    assert video_res[0, 0, 150, 300] == cr.NO_OBJ_SCORE and video_res[0, 0, 0, 0] == -cr.NO_OBJ_SCORE
    out = {"pred_masks": None,                       # add_new_mask :482 stores None here
           "pred_masks_video_res": video_res, "obj_ptr": torch.zeros(1, tr.hidden_dim),
           "object_score_logits": torch.tensor([[10.0]])}
    state = {"video_height": 360, "video_width": 640, "storage_device": torch.device("cpu"),
             "device": torch.device("cpu"), "obj_idx_to_id": {0: 1},
             "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {0: out},
                                              "non_cond_frame_outputs": {}}},
             "output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}}}
    cons = tr._consolidate_temp_output_across_obj(state, 0, is_cond=True, run_mem_encoder=False)
    ours = cr.downsample(mask)
    assert cons["pred_masks"].shape == ours.shape == (1, 1, cr.LOW_RES, cr.LOW_RES)
    assert torch.equal(cons["pred_masks"], ours)


def test_upsample_is_the_wrappers_conversion():
    """Step 5 on the real _convert_low_res_mask_to_video_res (uses no attribute of self)."""
    pytest.importorskip("sam3")
    import SCSam3VideoInferenceNewMem as W
    cls = W.SCSam3VideoInferenceWithInstanceInteractivityNewMem
    lo = cr.downsample(rect((360, 640), 100, 250, 200, 500))
    theirs = cls._convert_low_res_mask_to_video_res(cls.__new__(cls), lo[0, 0],
                                                    {"orig_height": 360, "orig_width": 640})
    assert theirs.shape == (1, 360, 640) and theirs.dtype == torch.bool
    assert np.array_equal(theirs[0].numpy(), cr.upsample(lo, (360, 640)))


# ------------------------------------------------------------ resample goldens
def test_resample_extremes_and_grid():
    full = np.ones(HW, dtype=bool)
    assert cr.resample(full).all()
    assert not cr.resample(np.zeros(HW, dtype=bool)).any()
    big = rect(HW, 200, 600, 400, 800)
    assert iou(big, cr.resample(big)) > 0.98
    dot = rect(HW, 500, 501, 900, 901)            # far below one 288-grid cell
    assert not cr.resample(dot).any()
    assert cr.downsample(dot).shape == (1, 1, 288, 288)


def test_resample_is_worse_at_a_coarser_grid_and_better_at_a_finer_one():
    m = rect(HW, 300, 340, 600, 660)                # 40 x 60 px: a few cells wide
    j = {lr: iou(m, cr.resample(m, lr)) for lr in (144, 288, 576)}
    assert j[144] < j[288] < j[576]


# -------------------------------------------------------------- overlap rule
def test_resolve_overlaps_lowest_id_keeps_the_pixel():
    a = rect(HW, 0, 100, 0, 100)
    b = rect(HW, 50, 150, 50, 150)
    ra, rb = cr.resolve_overlaps([a, b])
    assert np.array_equal(ra, a)                        # first in order: untouched
    assert np.array_equal(rb, b & ~a)                   # later: loses the contested pixels
    assert not (ra & rb).any()
    rb2, ra2 = cr.resolve_overlaps([b, a])              # order is the rule
    assert np.array_equal(rb2, b) and np.array_equal(ra2, a & ~b)
    c = rect(HW, 500, 600, 500, 600)
    assert all(np.array_equal(x, y) for x, y in zip(cr.resolve_overlaps([a, c]), [a, c]))


def test_roundtrip_of_touching_objects_stays_disjoint_and_near_fixed():
    left = rect(HW, 200, 600, 400, 700)
    right = rect(HW, 200, 600, 700, 1000)            # shares the boundary column
    # linear resampling of disjoint masks sums to <= 1, so at most one crosses 0.5:
    # the overlap rule is a no-op on ground truth (the 17-scene run: 0 of 398 rows)
    free = [cr.resample(left), cr.resample(right)]
    assert not (free[0] & free[1]).any()
    rt1 = cr.roundtrip([left, right])
    assert np.array_equal(rt1[0], free[0]) and np.array_equal(rt1[1], free[1])
    rt2 = cr.roundtrip(rt1)
    # the second pass moves fewer pixels than the first
    assert (rt2[0] != rt1[0]).sum() <= (rt1[0] != left).sum()
    assert (rt2[1] != rt1[1]).sum() <= (rt1[1] != right).sum()


# ------------------------------------------------------ bins, reading, summary
def test_size_bin_and_rig():
    assert [cr.size_bin(x) for x in (1, 1999, 2000, 9999, 10000)] == \
        ["<2000", "<2000", "2000-10000", "2000-10000", ">=10000"]
    assert cr.rig_of("Welder") == "hemisphere" and cr.rig_of("Fencing") == "pinhole"


def test_reading_follows_the_prereg_rule():
    assert cr.reading(0.951) == "artefact"
    assert cr.reading(0.970) == "artefact" and cr.reading(0.932) == "artefact"
    assert cr.reading(0.975) == "partial"
    assert cr.reading(0.98) == "other" and cr.reading(0.999) == "other"
    assert cr.reading(0.90) == "below_band"


def test_summarize_shape():
    rows = [dict(scene="S", cam="c", obj=1, gt_px=500, size_bin="<2000", rig="pinhole",
                 J_resample=0.4, J_rt1=0.4, J_rt2=0.3, rt1_px=1, rt2_px=1,
                 J_measured=0.3, measured_px=1, agree=1.0, diff_px=0),
            dict(scene="S", cam="c", obj=2, gt_px=50000, size_bin=">=10000", rig="pinhole",
                 J_resample=0.99, J_rt1=0.99, J_rt2=0.98, rt1_px=1, rt2_px=1,
                 J_measured=0.97, measured_px=1, agree=0.99, diff_px=7)]
    s = cr.summarize(rows)
    assert s["n"] == 2 and s["all"]["J_rt2"]["fail"] == 1
    assert s["all"]["J_rt2"]["mean"] == pytest.approx(0.64)
    assert s["by_size"]["2000-10000"]["J_rt2"] is None
    assert s["by_size"]["<2000"]["J_rt2"]["n"] == 1
    assert s["by_rig"]["hemisphere"]["J_rt1"] is None
    assert s["compare"] == dict(n=2, diff_px_total=7, pairs_identical=1, max_diff_px=7,
                                mean_delta_measured_minus_rt2=pytest.approx(-0.005))
    plain = cr.summarize([{k: v for k, v in r.items() if k not in
                           ("J_measured", "measured_px", "agree", "diff_px")} for r in rows])
    assert "compare" not in plain and "agree" not in plain["all"]


# --------------------------------------------------------------- data (if any)
def test_cli_on_the_smallest_scene(tmp_path, capsys):
    gt = os.path.join(REPO, "Data", "MVSeg", "Frog", "Mask", "v7", "000180.png")
    if not os.path.isfile(gt):
        pytest.skip("Frog ground truth not present")
    out = tmp_path / "ceiling.json"
    res = cr.main(["--scenes", "Frog", "--out", str(out), "--compare-run", ""])
    assert res["meta"]["compare_run"] is None and "agree" not in res["summary"]["all"]
    assert [r["obj"] for r in res["rows"]] == [1, 2, 3, 4, 5]
    for r in res["rows"]:
        assert 0.0 <= r["J_rt2"] <= 1.0 and 0.0 <= r["J_rt1"] <= 1.0
        assert r["J_resample"] == r["J_rt1"]     # disjoint inputs never overlap after resampling
        assert r["gt_px"] > 0 and r["rig"] == "pinhole" and r["size_bin"] == ">=10000"
    saved = json.load(open(out))
    assert saved["reading"] == cr.reading(saved["summary"]["all"]["J_rt2"]["mean"])
    assert "사전 등록 판독" in capsys.readouterr().out
