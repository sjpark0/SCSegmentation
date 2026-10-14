"""S1-E9 (docs/stage1-E9-prereg.md): the two decoder arms are GeoSet-2 arms with the predicted-memory recipe,
flagged for decoder training (D1, D2) and the band loss (D2 only); presets map them to the reproj gate."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem import arms  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem.stage1 import PRESETS  # noqa: E402

# geomem.train imports torch (through geomem.losses); its flag table is checked in tests_torch/test_e9_decoder.py


def test_dec_arms_membership():
    assert arms.DEC_ARMS == ("geoset_dec_ft", "geoset_decb_ft") and arms.BAND_ARMS == ("geoset_decb_ft",)
    for a in arms.DEC_ARMS:
        assert a in arms.SET2_ARMS and a in arms.NO_EMBED_ARMS and a in arms.PRED_MEM_ARMS
        assert a in R.ARMS and a in R.SET_ARMS
        assert arms.base_arm(a) == a
    assert "geoset_pred_ft" in arms.PRED_MEM_ARMS and "geoset_only_ft" not in arms.PRED_MEM_ARMS


def test_presets_for_e9():
    assert PRESETS["geoset_dec_rep"] == dict(arm="geoset_dec_ft", gate="reproj")
    assert PRESETS["geoset_decb_rep"] == dict(arm="geoset_decb_ft", gate="reproj")
    assert PRESETS["geoset_rep"] == dict(arm="geoset_pred_ft", gate="reproj")        # the adopted preset is untouched


def test_init_modules_cover_both_checkpoint_keys():
    assert set(R.INIT_MODULES) == {"memory_attention", "mask_decoder"}
