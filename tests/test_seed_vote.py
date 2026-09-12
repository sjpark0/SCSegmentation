"""S1-E6: the registered selection rule (docs/stage1-plan.md section 4) as a pure function."""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("numpy")
spec = importlib.util.spec_from_file_location("seed_vote", os.path.join(REPO, "MVSeed", "seed_vote.py"))
sv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv)


def _row(rig="pinhole", s2_source="Ggeo_a", **over):
    base = {n: dict(px=100, s4=0.90, iou_g0=0.0, agree_max=0.0, n_agree=0, s2=0.0, overlap_other=0.0)
            for n in sv.NAMES}
    base["G0"].update(px=100, s4=0.95, iou_g0=1.0, s2=0.0)
    for n in sv.GEO_SELF:
        base[n]["s2"] = None
    for n, kv in over.items():
        base[n].update(kv)
    return dict(scene="S", cam="c1", obj=1, ref_cam="c0", rig=rig, s2_source=s2_source, streams=base)


def test_registered_constants():
    assert sv.TAU == 0.5 and sv.OVERLAP_TAU == 0.5
    assert sv.EMPTY_MIN_AGREE == {"pinhole": 2, "hemisphere": 3}
    assert sv.NAMES[0] == "G0" and sv.SEED_TAG.startswith("MVSeed_")
    assert sv.CAND == ["G1", "Gdet", "Gcorr", "Ggeo_a", "Ggeo_b", "Ggeo_c"]


def test_default_is_G0_when_nothing_qualifies():
    assert sv.choose(_row())[0] is None


def test_all_three_conditions_are_required():
    ok = dict(iou_g0=0.1, n_agree=1, s4=0.99)          # (i) differs, (ii) agrees, (iii) better S4
    assert sv.choose(_row(G1=ok))[0] == "G1"
    assert sv.choose(_row(G1={**ok, "iou_g0": 0.6}))[0] is None    # (i) too similar to G0
    assert sv.choose(_row(G1={**ok, "n_agree": 0}))[0] is None     # (ii) no agreement, no S2
    assert sv.choose(_row(G1={**ok, "s4": 0.90}))[0] is None       # (iii) S4 not above G0


def test_s2_can_replace_agreement_but_not_for_the_projection_itself():
    ok = dict(iou_g0=0.1, n_agree=0, s4=0.99)
    assert sv.choose(_row(Gdet={**ok, "s2": 0.7}))[0] == "Gdet"     # geometry backs a non-geometry arm
    # Ggeo_a IS the projection: its s2 is None, so it must agree with another family
    assert sv.choose(_row(Ggeo_a={**ok, "s2": None}))[0] is None
    assert sv.choose(_row(Ggeo_a={**ok, "s2": None, "n_agree": 1}))[0] == "Ggeo_a"


def test_overlap_with_another_objects_seed_is_rejected():
    ok = dict(iou_g0=0.1, n_agree=1, s4=0.99)
    assert sv.choose(_row(G1={**ok, "overlap_other": 0.51}))[0] is None
    assert sv.choose(_row(G1={**ok, "overlap_other": 0.50}))[0] == "G1"   # strictly greater rejects


def test_empty_G0_needs_two_agreements_and_geometry_on_pinhole():
    e = {"G0": dict(px=0, s4=None, iou_g0=0.0)}
    ok = dict(iou_g0=0.0, n_agree=2, s2=0.7, s4=0.9)
    assert sv.choose(_row(**e, G1=ok))[0] == "G1"
    assert sv.choose(_row(**e, G1={**ok, "n_agree": 1}))[0] is None
    assert sv.choose(_row(**e, G1={**ok, "s2": 0.4}))[0] is None
    # S4 above G0 is not required when G0 is empty (there is no G0 pointer to beat)
    assert sv.choose(_row(**e, G1={**ok, "s4": 0.1}))[0] == "G1"


def test_empty_G0_on_hemisphere_needs_three_agreements_and_no_geometry():
    e = {"G0": dict(px=0, s4=None, iou_g0=0.0)}
    ok = dict(iou_g0=0.0, n_agree=3, s2=0.0, s4=0.9)
    assert sv.choose(_row(rig="hemisphere", **e, G1=ok))[0] == "G1"
    assert sv.choose(_row(rig="hemisphere", **e, G1={**ok, "n_agree": 2}))[0] is None


def test_empty_candidates_are_never_picked():
    ok = dict(px=0, iou_g0=0.0, n_agree=3, s4=0.99, s2=0.9)
    assert sv.choose(_row(G1=ok))[0] is None


def test_tie_break_is_highest_S4_then_stream_order():
    ok = dict(iou_g0=0.1, n_agree=1)
    assert sv.choose(_row(G1={**ok, "s4": 0.97}, Gdet={**ok, "s4": 0.99}))[0] == "Gdet"
    assert sv.choose(_row(G1={**ok, "s4": 0.99}, Gdet={**ok, "s4": 0.99}))[0] == "G1"   # order wins ties


def test_summarize_counts_replacements_and_empty_fills():
    votes = [dict(scene="S", cam="c", obj=1, rig="pinhole", g0_px=100, pick="G1", why="", pick_px=5),
             dict(scene="S", cam="c", obj=2, rig="pinhole", g0_px=0, pick="Gdet", why="", pick_px=7),
             dict(scene="S", cam="c", obj=3, rig="pinhole", g0_px=50, pick=None, why="", pick_px=50)]
    s = sv.summarize(votes)
    assert (s["n_rows"], s["replaced"], s["kept"], s["empty_g0_filled"]) == (3, 2, 1, 1)
    assert s["by_stream"] == {"G1": 1, "Gdet": 1}


def test_prereg_registers_the_tie_break_and_the_geometry_self_rule():
    t = open(os.path.join(REPO, "docs", "stage1-E6-prereg.md"), encoding="utf-8").read()
    assert "S4가 가장 높은 것" in t and "G1 → Gdet → Gcorr → Ggeo_a → Ggeo_b → Ggeo_c" in t
    assert "자신은 S2로 조건을 만족할 수 없습니다" in t
