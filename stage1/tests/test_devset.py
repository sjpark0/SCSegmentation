"""The standard DEV set (geomem/devset.py, docs/stage1-dev-protocol.md): splits, keyframe exclusion, the E13 frames,
and the DEV view key of the paired bootstrap.  Host."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from geomem import devset  # noqa: E402
from mvseg2d.paired import paired_scene_bootstrap, scene_means  # noqa: E402

DATA = os.path.join(ROOT, "Data")
needs_data = pytest.mark.skipif(not os.path.isfile(os.path.join(DATA, "Fencing", "meta.json")), reason="needs stage1/Data")
SCENES = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "CBABasketball", "Carpark", "CoffeeMartini",
          "Dog", "Fencing", "FlameSteak", "Frog", "MATF", "MartialArts", "Painter", "PoznanStreet", "Welder")


@needs_data
@pytest.mark.parametrize("scene", SCENES)
def test_splits_partition_the_nonref_train_frames(scene):
    ref = int(json.load(open(os.path.join(DATA, scene, "meta.json")))["ref_frame"])
    a, b = devset.dev_frames(DATA, scene, "A"), devset.dev_frames(DATA, scene, "B")
    nonref = devset.nonref_train_frames(DATA, scene)
    assert ref not in a and ref not in b and ref not in nonref
    assert not set(a) & set(b) and sorted(a + b) == sorted(nonref)
    assert len(a) == 5 and len(b) == 15
    assert a == [ref + 4, ref + 8, ref + 12, ref + 16, ref + 20]


@needs_data
def test_dev_a_equals_the_e13_dev_frames():
    p = os.path.join(ROOT, "geomem", "runs", "_e13_dev", "index.json")
    if not os.path.isfile(p):
        pytest.skip("no S1-E13 DEV index")
    e13 = json.load(open(p))["scenes"]
    for s in SCENES:
        assert devset.dev_frames(DATA, s, "A") == e13[s]["frames"]
        assert devset.eval_cams(DATA, s) == e13[s]["eval_cams"] and devset.basic_set(s) == e13[s]["basic"]


def test_paired_view_key():
    rows_a = [dict(scene="S", cam="c", t=1, view="1/c", JF=0.5), dict(scene="S", cam="c", t=2, view="2/c", JF=0.5),
              dict(scene="S", cam="c", t=2, view="2/c", JF=0.5)]
    rows_b = [dict(r, JF=1.0 if r["t"] == 1 else 0.5) for r in rows_a]
    assert scene_means(rows_b, "JF")["S"] == pytest.approx(2 / 3)                     # by camera: one view of 3 pairs
    assert scene_means(rows_b, "JF", view_key="view")["S"] == pytest.approx(0.75)     # by (t, cam): two views
    assert paired_scene_bootstrap(rows_a, rows_b, "JF", view_key="view")["delta"] == pytest.approx(0.25)
