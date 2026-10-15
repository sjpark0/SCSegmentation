"""S1-E11 (docs/stage1-E11-prereg.md): registries, constants and plumbing of the in-view exclusivity arms (host)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from methods.geomem import run as R  # noqa: E402
from methods.geomem import stage1 as S  # noqa: E402


def test_constants_as_registered():
    assert R.XCL_MODES == ("none", "out", "outmem", "out_nocap", "outs")
    assert R.XCL_TAU_KEEP == 0.5 and R.XCL_MEM_CLAMP == -10.0
    assert R.XCL_SHIFT == 0.5                                               # S1-E14 S (docs/stage1-E14-prereg.md 2.2)


def test_presets_exact():
    assert S.PRESETS["geoset_rep_x"] == dict(arm="geoset_pred_ft", gate="reproj", xcl="out")
    assert S.PRESETS["geoset_rep_xm"] == dict(arm="geoset_pred_ft", gate="reproj", xcl="outmem")
    assert "xcl" not in S.PRESETS["geoset_rep"] and "xcl" not in S.PRESETS["geoset_rep_z2r"]
    assert all("out_nocap" != p.get("xcl") for p in S.PRESETS.values())      # D has no preset: never in-process, never swept
    # S1-E14: exactly one preset uses `outs`, the adopted one with the shifted threshold; the adopted preset untouched
    assert S.PRESETS["geoset_rep_xs"] == dict(arm="geoset_pred_ft", gate="reproj", xcl="outs")
    assert [t for t, p in S.PRESETS.items() if p.get("xcl") == "outs"] == ["geoset_rep_xs"]
    assert list(S.PRESETS)[-1] == "geoset_rep_xs"


def test_check_gate_out_refuses_the_arms_own_folder_with_xcl():
    R.check_gate_out("geoset_pred_ft", "reproj", "geoset_pred_ft_rep_x_e11", "pred", None, "out")
    R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", None, "none")
    R.check_gate_out("geoset_pred_ft", "reproj", "geoset_pred_ft_rep_xs_e14", "pred", None, "outs")
    for mode in ("out", "outmem", "out_nocap", "outs"):
        with pytest.raises(SystemExit):
            R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", None, mode)


def test_check_preflight_reports_xcl(monkeypatch, tmp_path):
    (tmp_path / "X").mkdir(); (tmp_path / "X" / "meta.json").write_text("{}")
    ck = tmp_path / "runs" / "geoset_pred_ft" / "fold0"; ck.mkdir(parents=True)
    (ck / "config.json").write_text('{"fold": 0, "scenes": [], "ckpt_sha256": "x"}')
    monkeypatch.setattr(R, "checkpoint_for", lambda scene, arm, runs, folds: str(ck / "ckpt.pt"))
    import geomem.folds
    monkeypatch.setattr(geomem.folds, "load_folds", lambda: {})
    assert S.check_preflight("geoset_rep_xm", "X", data=str(tmp_path), runs=str(tmp_path / "runs"))["xcl"] == "outmem"
    assert S.check_preflight("geoset_rep", "X", data=str(tmp_path), runs=str(tmp_path / "runs"))["xcl"] == "none"
    assert S.check_preflight("geoset_rep_xs", "X", data=str(tmp_path), runs=str(tmp_path / "runs"))["xcl"] == "outs"
