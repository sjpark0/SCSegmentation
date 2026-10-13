import sys
import types


def test_load_model_disables_fill_hole_area(monkeypatch):
    """The raw builder defaults fill_hole_area to 16, whose Triton connected-components kernel fails on
    images larger than ~2048 px wide; the baseline (sam3_prop) runs with 0, so the single-frame model must too."""
    fake = types.ModuleType("build_scsam3")
    fake.build_scsam3_video_model = lambda: types.SimpleNamespace(fill_hole_area=16)
    monkeypatch.setitem(sys.modules, "build_scsam3", fake)
    from methods import sam3_single
    model = sam3_single.load_model()
    assert model.fill_hole_area == 0


def test_pi3_sam3_and_dino_sam3_record_fill_hole_area():
    """Both single-frame arms must write the model setting into RUN.json's config (review I3)."""
    from methods.pi3_sam3 import run as a
    from methods.dino_sam3 import run as b
    assert a.FILL_HOLE_AREA == 0 and b.FILL_HOLE_AREA == 0
