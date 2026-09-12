"""S1-E3 (G-corr): window sizing, peak NMS, and that the gate is E2's gate with E2's constants."""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("numpy")


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dino = _load("MVSeed/corr/dino_prompt.py", "dino_prompt")
e3 = _load("MVSeed/corr/e3_segment.py", "e3_segment")
gate = e3.gate


def test_resized_hw_keeps_aspect_and_patch_multiples():
    for h, w in ((1080, 1920), (1920, 1080), (2012, 2684), (720, 720)):
        rh, rw = dino.resized_hw(h, w)
        assert rh % dino.PATCH == 0 and rw % dino.PATCH == 0
        assert max(rh, rw) <= dino.LONG_SIDE
        assert max(rh, rw) > dino.LONG_SIDE - dino.PATCH      # the long side really is ~1036
        assert abs(rh / rw - h / w) < 0.03                    # aspect preserved to within one patch


def test_registered_constants():
    assert (dino.MODEL_ID, dino.LONG_SIDE, dino.PATCH) == ("facebook/dinov2-large", 1036, 14)
    assert (dino.K_CANDIDATES, dino.NMS_PATCHES) == (5, 3)
    assert e3.TAU_SIM == 0.5 and e3.IOU_EXIST == 0.5


def test_peaks_suppress_neighbours_and_rank_by_value():
    np = pytest.importorskip("numpy")
    s = np.zeros((40, 40), "float32")
    s[10, 10] = 0.9
    s[10, 12] = 0.8          # inside the suppression disc of the first peak
    s[30, 30] = 0.7          # far away
    got = dino.peaks(s, radius_px=5, k=5)
    assert [(x, y) for x, y, _ in got][:2] == [(10, 10), (30, 30)]
    assert got[0][2] > got[1][2]
    assert all(v <= 1.0 for _, _, v in got)


def test_peaks_respect_k():
    np = pytest.importorskip("numpy")
    s = np.random.RandomState(0).rand(60, 60).astype("float32")
    assert len(dino.peaks(s, radius_px=2, k=3)) == 3


def _row(scene, cam, obj, gt_px, best_iou, rank=1, n_cand=5, rig="pinhole", tag=None):
    cands = [dict(point=[1, 1], cos=0.9, px=100, iou=best_iou if i + 1 == rank else 0.0, kept=True)
             for i in range(n_cand)]
    r = dict(scene=scene, cam=cam, obj=obj, ref_cam="c0", gt_px=gt_px, rig=rig, candidates=cands,
             n_cand=n_cand, exist=bool(best_iou >= e3.IOU_EXIST), best_iou=best_iou,
             best_rank=rank if n_cand else None, top1_hit=bool(rank == 1 and best_iou >= e3.IOU_EXIST),
             top1_cos=0.9 if n_cand else None)
    if tag:
        r["set"] = tag
        r["size_bin"] = (None if gt_px == 0 else gate.SIZE_BINS[0] if gt_px < 2000
                         else gate.SIZE_BINS[1] if gt_px < 10000 else gate.SIZE_BINS[2])
    return r


def _population(exist_ge2k, halluc, n_cand_empty=5):
    rows = []
    for i in range(19):                       # F0 & >= 2k: 16 pinhole + 3 hemisphere, as registered
        rows.append(_row("S", f"c{i}", i, 5000, 0.9 if i < exist_ge2k else 0.1,
                         rig="pinhole" if i < 16 else "hemisphere", tag="F0"))
    for i in range(22):
        rows.append(_row("S", f"d{i}", i, 500, 0.1, tag="F0"))
    for i in range(623):
        rows.append(_row("S", f"n{i}", i, 20000, 0.9, tag="N0"))
    for i in range(132):
        rows.append(_row("S", f"e{i}", i, 0, 0.0,
                         n_cand=n_cand_empty if i < halluc else 0, tag="gt_empty"))
    for r in rows:
        r.setdefault("f0_kind", "misplaced")
    return rows


def test_denominators_and_gates_match_the_registered_numbers():
    s = e3.summarize(_population(exist_ge2k=11, halluc=13))
    assert s["denominators"]["match"] is True
    assert s["denominators"]["measured"] == gate.REGISTERED
    assert s["gates"]["gate1_exist_ge2k"]["pass"] is True       # 11/19 >= 0.55
    assert s["gates"]["gate2_rank"]["pass"] is True
    assert s["gates"]["gate3_halluc"]["pass"] is True           # 13/132 <= 0.10
    assert s["gates"]["verdict"] == "proceed with G-corr"


def test_gate1_is_the_same_integer_comparison_as_e2():
    assert e3.summarize(_population(10, 0))["gates"]["gate1_exist_ge2k"]["pass"] is False
    assert e3.summarize(_population(11, 0))["gates"]["gate1_exist_ge2k"]["pass"] is True
    # the pinhole sub-gate is the registered 8 of 16
    assert e3.summarize(_population(8, 0))["gates"]["gate1p_pinhole"]["pass"] is True
    assert e3.summarize(_population(7, 0))["gates"]["gate1p_pinhole"]["pass"] is False


def test_gate3_boundary_is_ten_percent_of_132():
    assert e3.summarize(_population(11, 13))["gates"]["gate3_halluc"]["pass"] is True
    assert e3.summarize(_population(11, 14))["gates"]["gate3_halluc"]["pass"] is False


def test_failed_gates_are_named_in_the_verdict():
    v = e3.summarize(_population(3, 132))["gates"]["verdict"]
    assert v.startswith("discard G-corr") and "gate 1 existence" in v and "gate 3 hallucination" in v


def test_tau_curve_is_monotone_in_hallucination():
    rows = _population(11, 132)
    c = e3.tau_curve(rows, taus=(0.5, 0.95))
    assert c["0.5"]["halluc"] >= c["0.95"]["halluc"]


def test_prereg_registers_these_constants():
    t = open(os.path.join(REPO, "docs", "stage1-E3-prereg.md"), encoding="utf-8").read()
    assert "facebook/dinov2-large" in t and "1,036 px" in t
    assert "K = 5" in t and "3 패치" in t and "τ_sim = 0.5" in t
    assert "19쌍 중 정답과 IoU ≥ 0.5인 후보가 있는 쌍 ≥ 11" in t
