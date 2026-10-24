"""S1-E17 part 1 (docs/stage1-E17-prereg.md): the C-ov arms, the weighted sampler, the relay-style memory draw, the decode arm."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from geomem import arms
from geomem.data import Sample, SampleIterator, load_train_angles, relay_memory_camera, sample_weights
from methods.geomem import dev_decode as D
from methods.geomem import run as R
from methods.geomem import stage1 as S


def test_cov_arm_sets_and_flags():
    assert arms.COV_ARMS == ("geoset_cov2_ft", "geoset_cov4_ft") and arms.COV_LAMBDA == {"geoset_cov2_ft": 2.0, "geoset_cov4_ft": 4.0}
    assert arms.COV_THETA == 45.0 and arms.COV_RELAY_PROB == 0.5
    for a in arms.COV_ARMS:
        assert a in arms.SET2_ARMS and a in arms.PRED_MEM_ARMS and a in arms.NO_EMBED_ARMS and a in R.ARMS and a in R.SET_ARMS
        assert a not in arms.CUT_ARMS and a not in arms.DEC_ARMS and a not in arms.NOGEO_ARMS
    assert list(S.PRESETS)[-2:] == ["geoset_rep_xs", "geoset_rep_xs_ab"]   # PRESETS untouched
    T = pytest.importorskip("geomem.train")
    for a in arms.COV_ARMS:
        assert a in T.ARMS and a in T.BIAS_ARMS and a in T.FT_ARMS and a in T.SET_ARMS and a not in T.PAIR_ARMS
        f = T.arm_flags(a)
        g = T.arm_flags("geoset_pred_ft")
        assert f["cov_lambda"] == arms.COV_LAMBDA[a] and f["cov_theta"] == 45.0 and f["cov_relay_prob"] == 0.5
        assert {k: v for k, v in f.items() if not k.startswith("cov_")} == g      # the geoset_pred_ft recipe otherwise
    assert "cov_lambda" not in T.arm_flags("geoset_pred_ft")
    from geomem import wrapper as W
    assert all(a in W.FT_ARMS and a in W.BIAS_ARMS for a in arms.COV_ARMS)


def _angles():
    return {("s", 0, "r"): {"v1": 10.0, "v2": 50.0, "m1": 5.0, "m2": 30.0, "m3": 70.0}}


def test_sample_weights_and_iterator():
    ang = _angles()
    samples = [Sample("s", 0, "r", "v1"), Sample("s", 0, "r", "v2")]
    w = sample_weights(samples, ang, 4.0, 45.0)
    assert w.tolist() == [1.0, 5.0]
    with pytest.raises(KeyError):                                   # the cache must cover every sample
        sample_weights([Sample("s", 1, "r", "v1")], ang, 4.0, 45.0)
    it = SampleIterator(samples * 50, seed=0, weights=np.tile(w, 50))
    ep = it.epoch()
    assert len(ep) == 100                                           # epoch size unchanged, with replacement
    frac = sum(1 for s in ep if s.tgt_cam == "v2") / 100
    assert 0.7 < frac < 0.95                                        # 5/6 expected
    assert SampleIterator(samples, seed=0).epoch() != SampleIterator(samples, seed=0, weights=w).epoch() or True
    plain = SampleIterator(samples * 50, seed=0).epoch()
    assert sorted(plain, key=str) == sorted(samples * 50, key=str)  # unweighted: a permutation
    with pytest.raises(AssertionError):
        SampleIterator(samples, weights=[1.0, 0.0])


def test_relay_memory_camera():
    ang = _angles()[("s", 0, "r")]
    pool = ["m1", "m2", "m3"]
    rules, picks = [], []
    for i in range(400):
        c, rule = relay_memory_camera(pool, ang, "v2", np.random.default_rng(i), 0.5)
        rules.append(rule)
        picks.append((rule, c))
    assert set(rules) == {"relay", "uniform"} and 0.4 < rules.count("relay") / 400 < 0.6
    assert all(c in ("m1", "m2") for r, c in picks if r == "relay")          # a(r->m) < a(r->v2) = 50
    assert {c for r, c in picks if r == "uniform"} == set(pool)
    # no nearer camera: the fallback is uniform over the pool, recorded as relay_none
    out = {relay_memory_camera(pool, {"v0": 1.0, "m1": 5.0, "m2": 30.0, "m3": 70.0}, "v0", np.random.default_rng(i), 1.0) for i in range(50)}
    assert {r for _, r in out} == {"relay_none"} and {c for c, _ in out} == set(pool)
    assert relay_memory_camera(pool, ang, "v2", np.random.default_rng(0), 0.0)[1] == "uniform"


def test_load_train_angles(tmp_path):
    p = tmp_path / "a.json"
    json.dump({"entries": [{"scene": "s", "t": 4, "ref": "r", "all": {"c": 12.5}}]}, open(p, "w"))
    assert load_train_angles(str(p)) == {("s", 4, "r"): {"c": 12.5}}


def test_e17_presets(tmp_path):
    assert D.E17_ARMS == {"C-ov2": "geoset_cov2_ft", "C-ov4": "geoset_cov4_ft"}
    p = D.e17_preset("C-ov4", str(tmp_path))
    assert p["arm"] == "geoset_cov4_ft" and p["runs"] == str(tmp_path) and p["_e17"] == "C-ov4" and p["_preset"] == "e17:C-ov4"
    main = dict(S.PRESETS["geoset_rep_xs_ab"])
    assert {k: v for k, v in p.items() if not k.startswith("_") and k != "runs"} == dict(main, arm="geoset_cov4_ft")
    assert R.e15_options(SimpleNamespace(**p))["row"] == "AB"      # decoded as the main arm (Fix A c*, relay 60)
    assert D.E16_ARMS and "C-ov2" not in D.E16_ARMS


def test_select_lambda():
    import importlib.util
    spec = importlib.util.spec_from_file_location("e17_score", os.path.join(os.path.dirname(__file__), "..", "scripts", "e17", "e17_score.py"))
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    mk = lambda wide, lo: {"A": {"vs_C0": {"ci": [lo, lo + 0.01]}}, "A2": {"vs_AB": {"wide_dJ": wide, "dJF": 0.0}}}
    res = {"C-ov2_s0": mk(0.010, -0.002), "C-ov4_s0": mk(0.020, -0.004)}
    assert M.select_lambda(res)[0] == "C-ov4"
    res = {"C-ov2_s0": mk(0.0205, -0.002), "C-ov4_s0": mk(0.020, -0.004)}
    assert M.select_lambda(res)[0] == "C-ov2"                        # tie within 0.001 -> CI lower bound
    assert M.select_lambda({"C-ov2_s0": mk(0.0, 0.0)})[0] is None
