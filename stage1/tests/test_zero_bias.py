"""Ring diagnosis D3 (stage1/docs/ring-diagnosis-2026-10-20.md §2.6): the opt-in `--zero-bias` decode flag (host)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from methods import common  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem import stage1 as S  # noqa: E402


def _args(*extra):
    return common.parse_args(R.__doc__, R._extra, argv=["--scene", "X", "--out", "y", *extra])


def test_flag_is_off_by_default_and_opt_in():
    assert _args().zero_bias is False
    assert _args("--zero-bias").zero_bias is True


def test_check_gate_out_refuses_the_arms_own_folder_with_zero_bias():
    with pytest.raises(SystemExit):
        R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", None, "none", True)
    R.check_gate_out("geoset_pred_ft", "reproj", "d3_bias0", "pred", None, "outs", True)
    # the default (flag off) behaves as before: the arm's own name stays allowed for the default settings
    R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", None, "none")
    R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", None, "none", False)


def test_no_preset_and_no_in_process_default_zeroes_the_bias():
    # Stage1.run builds its args from the presets + a fixed base: neither may switch the diagnostic on
    assert all("zero_bias" not in p for p in S.PRESETS.values())
