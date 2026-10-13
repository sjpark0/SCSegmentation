"""GPU: two-round inference on Blocks with the untrained geoset code path."""
import argparse
import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods import common  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks")),
                                reason="needs GPU and Data/Blocks")


def test_view_confident_rule():
    from methods.geomem.run import view_confident
    assert view_confident(torch.tensor([1.0, -1.0]), torch.tensor([0.9, 0.1]), tau=0.7) is True
    assert view_confident(torch.tensor([1.0, 1.0]), torch.tensor([0.9, 0.4]), tau=0.7) is False
    assert view_confident(torch.tensor([-1.0, -1.0]), torch.tensor([0.9, 0.9]), tau=0.7) is False   # nothing present


def _pngs(d, skip=()):
    out = {}
    for cam in sorted(os.listdir(d)):
        if cam in skip:
            continue
        if os.path.isdir(os.path.join(d, cam)):
            for f in sorted(os.listdir(os.path.join(d, cam))):
                out[(cam, f)] = cv2.imread(os.path.join(d, cam, f), 0)
    return out


def test_round2_without_neighbours_equals_round1():
    from methods.geomem import run as R
    scene = common.load_scene(DATA, "Blocks")
    args = argparse.Namespace(data=DATA, out="dbg_geoset", arm="geoset_ft", untrained=True, runs="geomem/runs",
                              tau=1.01, k=3)
    import shutil
    for d in ("dbg_geoset", "dbg_geoset_r1"):
        shutil.rmtree(os.path.join(DATA, "Blocks", "Result", d), ignore_errors=True)
    cfg = R.decode_scene(scene, args)
    assert cfg["two_round"] is True and cfg["views_with_ref_only_frac"] == 1.0 and cfg["mem_per_view_mean"] == 1.0
    # the reference camera's GT is copied by run_all for args.out and by decode_scene for _r1; compare targets only
    r1 = _pngs(os.path.join(DATA, "Blocks", "Result", "dbg_geoset_r1"), skip=(scene.ref_cam,))
    r2 = _pngs(os.path.join(DATA, "Blocks", "Result", "dbg_geoset"), skip=(scene.ref_cam,))
    assert r1 and set(r1) == set(r2)
    assert all(np.array_equal(r1[k], r2[k]) for k in r1)


def test_round2_with_neighbours_uses_memory_sets():
    from methods.geomem import run as R
    scene = common.load_scene(DATA, "Blocks")
    args = argparse.Namespace(data=DATA, out="dbg_geoset2", arm="geoset_ft", untrained=True, runs="geomem/runs",
                              tau=0.0, k=3)
    cfg = R.decode_scene(scene, args)
    assert cfg["mem_per_view_mean"] > 1.0 and cfg["views_with_ref_only_frac"] < 1.0
    assert os.path.isdir(os.path.join(DATA, "Blocks", "Result", "dbg_geoset2", "cam0"))


# ---------------------------------------------------------------- GeoSet-2: --gate obj
def _args(out, **kw):
    base = dict(data=DATA, out=out, arm="geoset_ft", untrained=True, runs="geomem/runs", tau=0.7, k=3, gate="view",
                r1_out=None, round2="pred")
    base.update(kw)
    return argparse.Namespace(**base)


def _clean(*outs):
    import shutil
    for d in outs:
        shutil.rmtree(os.path.join(DATA, "Blocks", "Result", d), ignore_errors=True)


def test_object_gate_drops_unconfident_view():
    import json
    from methods.geomem import run as R
    scene = common.load_scene(DATA, "Blocks")
    _clean("dbg_geoset_obj")
    cfg = R.decode_scene(scene, _args("dbg_geoset_obj", gate="obj", tau=1.01, r1_out="none"))
    centres = R.camera_centres(os.path.join(DATA, "Blocks", "Cache", "pi3"), scene.cams)
    n_cand = sum(len([c for c in R.select_candidates(cam, centres, 3) if c != scene.ref_cam])
                 for cam in scene.cams if cam != scene.ref_cam)
    assert cfg["gate"] == "obj" and cfg["mem_per_view_mean"] == 1.0 and cfg["views_with_ref_only_frac"] == 1.0
    assert cfg["views_dropped"] == n_cand > 0 and cfg["gated_pairs_frac"] == 1.0
    assert not os.path.isdir(os.path.join(DATA, "Blocks", "Result", "dbg_geoset_obj_r1"))
    conf = json.load(open(os.path.join(DATA, "Blocks", "Result", "dbg_geoset_obj", "round1_conf.json")))
    assert set(conf) == set(scene.cams) - {scene.ref_cam}
    assert all(set(v) == {str(o) for o in scene.objects} and all({"iou", "presence"} <= set(x) for x in v.values())
               for v in conf.values())


def test_object_gate_tau0_accounting_matches_round1_record():
    """With tau=0 only ABSENT objects are gated (spec 2.1), so the gate statistics follow from round1_conf.json."""
    import json
    from methods.geomem import run as R
    scene = common.load_scene(DATA, "Blocks")
    _clean("dbg_g_view", "dbg_g_view_r1", "dbg_g_obj")
    cfg_v = R.decode_scene(scene, _args("dbg_g_view", gate="view", tau=0.0))
    cfg_o = R.decode_scene(scene, _args("dbg_g_obj", gate="obj", tau=0.0, r1_out="none"))
    assert os.path.isdir(os.path.join(DATA, "Blocks", "Result", "dbg_g_view_r1"))
    rec = json.load(open(os.path.join(DATA, "Blocks", "Result", "dbg_g_obj", "round1_conf.json")))
    centres = R.camera_centres(os.path.join(DATA, "Blocks", "Cache", "pi3"), scene.cams)
    absent = {c: [v["presence"] <= 0 for v in rec[c].values()] for c in rec}
    n_pairs = n_gated = dropped = 0
    for cam in scene.cams:
        if cam == scene.ref_cam:
            continue
        for c in [x for x in R.select_candidates(cam, centres, 3) if x != scene.ref_cam]:
            n_pairs += len(absent[c]); n_gated += sum(absent[c]); dropped += all(absent[c])
    assert abs(cfg_o["gated_pairs_frac"] - n_gated / n_pairs) < 1e-9 and cfg_o["views_dropped"] == dropped
    # a view with nothing present is excluded by both gates, so the set sizes agree
    assert cfg_o["mem_per_view_mean"] == cfg_v["mem_per_view_mean"]


def test_absent_and_reproj_gates_smoke():
    import json
    from methods.geomem import run as R
    scene = common.load_scene(DATA, "Blocks")
    _clean("dbg_g_abs", "dbg_g_rep", "dbg_g_obj2")
    cfg_o = R.decode_scene(scene, _args("dbg_g_obj2", gate="obj", r1_out="none"))
    cfg_a = R.decode_scene(scene, _args("dbg_g_abs", gate="absent", r1_out="none"))
    cfg_r = R.decode_scene(scene, _args("dbg_g_rep", gate="reproj", r1_out="none"))
    assert cfg_a["gate"] == "absent" and cfg_r["gate"] == "reproj"
    assert cfg_a["gated_pairs_frac"] <= cfg_o["gated_pairs_frac"]        # absent-only gates a subset of obj
    rec = json.load(open(os.path.join(DATA, "Blocks", "Result", "dbg_g_rep", "round1_conf.json")))
    vals = [v for cam in rec.values() for v in cam.values()]
    assert all({"iou", "presence", "overlap", "n_valid"} <= set(v) for v in vals)
    assert any(v["n_valid"] > 0 and v["overlap"] is not None for v in vals)
    assert os.path.isdir(os.path.join(DATA, "Blocks", "Result", "dbg_g_rep", "cam0"))


# ---------------------------------------------------------------- GeoFuse spec 4: inference with fusion
def test_geofuse_untrained_equals_geoset_untrained():
    """With g = 0 the fused arm must reproduce the A-3 arm's masks in both rounds, and record fusion statistics."""
    from methods.geomem import run as R
    scene = common.load_scene(DATA, "Blocks")
    _clean("dbg_gs", "dbg_gs_r1", "dbg_gf", "dbg_gf_r1")
    R.decode_scene(scene, _args("dbg_gs", arm="geoset_pred_ft", gate="reproj", r1_out="dbg_gs_r1"))
    cfg = R.decode_scene(scene, _args("dbg_gf", arm="geofuse_ft", gate="reproj", r1_out="dbg_gf_r1"))
    for a, b in (("dbg_gs_r1", "dbg_gf_r1"), ("dbg_gs", "dbg_gf")):
        pa = _pngs(os.path.join(DATA, "Blocks", "Result", a), skip=(scene.ref_cam,))
        pb = _pngs(os.path.join(DATA, "Blocks", "Result", b), skip=(scene.ref_cam,))
        assert pa and set(pa) == set(pb) and all(np.array_equal(pa[k], pb[k]) for k in pa), a
    for k in ("fuse_self_mass", "fuse_view_entropy", "fuse_valid_views_mean", "valid_align_frac"):
        assert k in cfg and 0.0 <= cfg[k] <= (10.0 if k == "fuse_valid_views_mean" else 1.0), k
    assert cfg["valid_align_frac"] > 0.1 and cfg["fuse_valid_views_mean"] > 1.0


# ---------------------------------------------------------------- GeoAlign spec 4: shared aligned memory views
def test_geoalign_untrained_round1_and_round2():
    from methods.geomem import run as R
    scene = common.load_scene(DATA, "Blocks")
    _clean("dbg_ga", "dbg_ga_r1", "dbg_gp", "dbg_gp_r1")
    cfg = R.decode_scene(scene, _args("dbg_ga", arm="geoalign_ft", gate="reproj", r1_out="dbg_ga_r1"))
    R.decode_scene(scene, _args("dbg_gp", arm="geoset_pred_ft", gate="reproj", r1_out="dbg_gp_r1"))
    assert cfg["align_views"] == 3 and 0.0 < cfg["align_valid_mean"] <= 1.0
    assert len(cfg["align_pool"]) == 4 and scene.ref_cam not in cfg["align_pool"] and cfg["align_targets_substituted"] == 3
    assert cfg["mem_per_view_mean"] >= 4.0                                  # c_ini + 3 aligned (+ predicted views)
    a = _pngs(os.path.join(DATA, "Blocks", "Result", "dbg_ga_r1"), skip=(scene.ref_cam,))
    b = _pngs(os.path.join(DATA, "Blocks", "Result", "dbg_gp_r1"), skip=(scene.ref_cam,))
    # the memory differs, so the round-1 masks differ: a key present in one run only (an object predicted empty by the
    # other, no PNG written) or a differing PNG
    assert a and b and (set(a) != set(b) or any(not np.array_equal(a[k], b[k]) for k in a))
    assert os.path.isdir(os.path.join(DATA, "Blocks", "Result", "dbg_ga", "cam0"))


# ---------------------------------------------------------------- GeoProj: prediction-free round 2 (projected GT masks)
@pytest.mark.parametrize("mode", ["proj_bin", "proj_soft"])
def test_round2_projected_masks_smoke(mode):
    import json
    from methods.geomem import run as R
    scene = common.load_scene(DATA, "Blocks")
    _clean(f"dbg_{mode}")
    cfg = R.decode_scene(scene, _args(f"dbg_{mode}", arm="geoset_pred_ft", gate="absent", r1_out="none", round2=mode))
    assert cfg["round2"] == mode and cfg["mem_per_view_mean"] > 1.0
    assert 0.0 < cfg["proj_vis_mean"] <= 1.0 and cfg["proj_objects_visible_frac"] > 0.0
    assert os.path.isdir(os.path.join(DATA, "Blocks", "Result", f"dbg_{mode}", "cam0"))
    rec = json.load(open(os.path.join(DATA, "Blocks", "Result", f"dbg_{mode}", "round1_conf.json")))
    assert rec                                                                     # round 1 still runs and is recorded
