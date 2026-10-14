"""S1-E12 (docs/stage1-E12-prereg.md): the memory-token drop through MemorySet.gate_mask (CPU), its effect inside
decode_view and train_one_step (GPU, Blocks), the arm flags, the 20-step training smokes of the two new arms against
geoset_pred_ft (rng-stream identity, step-1 loss identity of the zero-dose arm), and the adopted decode path's byte
identity after the change."""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from geomem.wrapper import MemorySet, MemoryView  # noqa: E402

DATA = os.path.join(REPO, "Data")
L = 5184
gpu_blocks = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks", "Cache", "sam3_feats")),
                                reason="needs GPU and the Blocks caches")


@pytest.fixture(autouse=True)
def _free_gpu_between_tests():
    yield
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- gate_mask with a drop (CPU)
def _view(B, conf=None, name="v", drop=None):
    z = torch.zeros(L, B, 64)
    return MemoryView(feats=z, pos=z.clone(), obj_ptr_tokens=torch.zeros(4, B, 64), obj_ptr_pos=torch.zeros(4, B, 64),
                      object_score_logits=torch.zeros(B, 1), p=torch.zeros(L, 3), valid=torch.ones(L, dtype=torch.bool),
                      name=name, conf=conf, drop=drop)


def test_gate_mask_unchanged_without_drop():
    B = 3
    assert MemorySet([_view(B, name="ref")]).gate_mask(torch.float32, "cpu") is None
    assert MemorySet([_view(B, name="ref", drop=torch.zeros(B, L, dtype=torch.bool))]).gate_mask(torch.float32, "cpu") is None
    conf = torch.tensor([True, False, True])
    a = MemorySet([_view(B, name="ref"), _view(B, conf)]).gate_mask(torch.float32, "cpu")
    b = MemorySet([_view(B, name="ref", drop=torch.zeros(B, L, dtype=torch.bool)), _view(B, conf)]).gate_mask(torch.float32, "cpu")
    assert torch.equal(a, b)


def test_gate_mask_drop_columns_rows_and_conf():
    B, V = 3, 2
    drop = torch.zeros(B, L, dtype=torch.bool)
    drop[1, 100:200] = True
    drop[2, :10] = True
    conf = torch.tensor([True, False, True])
    m = MemorySet([_view(B, name="ref", drop=drop), _view(B, conf, "v1")]).gate_mask(torch.float32, "cpu")
    inf = float("-inf")
    assert m.shape == (B, 1, 1, V * L + 4 * V)
    assert (m[0, 0, 0, :L] == 0).all()                                            # object 0 has no drop
    assert (m[1, 0, 0, 100:200] == inf).all() and (m[1, 0, 0, :100] == 0).all() and (m[1, 0, 0, 200:L] == 0).all()
    assert (m[2, 0, 0, :10] == inf).all() and (m[2, 0, 0, 10:L] == 0).all()
    assert (m[:, 0, 0, V * L:V * L + 4] == 0).all()                               # reference pointer columns never dropped
    assert (m[1, 0, 0, L:2 * L] == inf).all() and (m[1, 0, 0, V * L + 4:] == inf).all()   # conf-gating of view 1 intact
    assert (m[0, 0, 0, L:] == 0).all()


def test_subset_slices_drop_on_the_object_axis():
    B = 4
    drop = torch.zeros(B, L, dtype=torch.bool)
    drop[2, 7] = True
    ms = MemorySet([_view(B, name="ref", drop=drop)]).subset(torch.tensor([2, 3]))
    assert ms.views[0].drop.shape == (2, L) and ms.views[0].drop[0, 7] and not ms.views[0].drop[1].any()


def test_object_cells_grid():
    from geomem.sam3_io import object_cells
    m = torch.zeros(1, 1080, 1920, dtype=torch.bool)
    m[0, :540, :960] = True                                                      # the top-left quarter
    c = object_cells(m)
    assert c.shape == (1, 72, 72) and c[0, :36, :36].all() and not c[0, 36:].any() and not c[0, :, 36:].any()


def test_cut_arm_flags_are_the_pred_recipe_plus_cut():
    from geomem.train import ARMS, BIAS_ARMS, CUT_KEEP, CUT_MIN_CELLS, FT_ARMS, SET_ARMS, arm_flags
    base = arm_flags("geoset_pred_ft")
    assert "cut_prob" not in base
    for a, p in (("geoset_cut0_ft", 0.0), ("geoset_cutall_ft", 1.0)):
        assert a in ARMS and a in BIAS_ARMS and a in FT_ARMS and a in SET_ARMS
        f = arm_flags(a)
        assert f["extra_mem_pred"] is True and f["extra_mem_prob"] == 0.5 and f["train_decoder"] is False
        assert f["cut_prob"] == p and f["cut_keep"] == list(CUT_KEEP) and f["cut_min_cells"] == CUT_MIN_CELLS
        assert {k: v for k, v in f.items() if not k.startswith("cut_")} == base


# ---------------------------------------------------------------- GPU: drop inside decode_view (Blocks)
from tests_torch.test_wrapper import setup  # noqa: E402,F401


def _bias_wrapper(s):
    from geomem.bias import GeoBias
    from geomem.embed import GeoEmbed
    from geomem.wrapper import GeoMemStar
    return GeoMemStar(s["model"], GeoEmbed().cuda(), geo_bias=GeoBias().cuda(), use_geometry=False, use_bias=True)


@gpu_blocks
def test_drop_changes_only_the_dropped_object(setup):
    from geomem.data import fov_cut
    from geomem.sam3_io import object_cells
    import numpy as np
    s = setup
    g = _bias_wrapper(s)
    bb0, p0, v0 = s["views"]["cam0"]
    B = s["masks"].shape[0]
    cells = object_cells(s["masks"]).cpu().numpy()
    big = int(np.argmax(cells.reshape(B, -1).sum(1)))
    assert cells[big].sum() >= 20
    drop, recs = fov_cut(cells[big:big + 1], np.random.default_rng(0))
    assert recs[0]["dropped"] > 0
    d = torch.zeros(B, L, dtype=torch.bool, device="cuda")
    d[big] = torch.from_numpy(drop[0]).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
        plain = g.decode_view(bb0, B, ref, p0, v0)["logits_1008"]
        cut = g.decode_view(bb0, B, MemorySet([replace(ref.views[0], drop=d)]), p0, v0)["logits_1008"]
        zero = g.decode_view(bb0, B, MemorySet([replace(ref.views[0], drop=torch.zeros_like(d))]), p0, v0)["logits_1008"]
    others = [i for i in range(B) if i != big]
    assert torch.isfinite(cut).all()
    assert torch.equal(cut[others], plain[others]), "a per-object drop leaked into other objects"
    assert not torch.equal(cut[big], plain[big]), "the drop had no effect"
    assert torch.equal(zero, plain)
    g.close()


# ---------------------------------------------------------------- GPU: train_one_step with a drop
def _pred_batch(arm="geoset_pred_ft"):
    import numpy as np
    from geomem.data import Sample, choose_objects, load_sample, objects_for
    from geomem.train import build_optimizer, build_wrapper
    from methods import sam3_single
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = sam3_single.load_model()
    w = build_wrapper(model, arm); w.freeze(arm); opt = build_optimizer(w, arm)
    s = Sample("Blocks", 0, "cam4", "cam0")
    b = load_sample(DATA, s, choose_objects(objects_for(s, DATA), 16, rng), "cuda", extra_mem_cam="cam2", extra_mem_gt=False)
    return w, opt, b, rng


def _drop_for(b, seed=0):
    import numpy as np
    from geomem.data import fov_cut
    from geomem.sam3_io import object_cells
    drop, recs = fov_cut(object_cells(b["ref_masks"]).cpu().numpy(), np.random.default_rng(seed))
    assert drop.any()
    return torch.from_numpy(drop).cuda()


@gpu_blocks
def test_train_step_with_drop_reaches_both_reads(monkeypatch):
    from geomem import train as T
    w, opt, b, rng = _pred_batch("geoset_cutall_ft")
    b["ref_drop"] = _drop_for(b)
    seen = []
    orig = w.decode_view

    def spy(backbone_out, B, memory, *a, **kw):
        seen.append(memory.views[0].drop)
        return orig(backbone_out, B, memory, *a, **kw)
    monkeypatch.setattr(w, "decode_view", spy)
    monkeypatch.setattr(T, "PRED_MEM_TAU", 0.0)                 # every present object kept -> the second memory is used
    opt.zero_grad(set_to_none=True)
    terms = T.train_one_step(w, b, "geoset_cutall_ft", rng)
    assert len(seen) == 2 and all(d is not None and torch.equal(d, b["ref_drop"]) for d in seen)
    assert math.isfinite(terms["loss"]) and terms["n_mem"] in (1, 2)
    assert w.geo_bias.w.grad is not None and w.geo_bias.w.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in w.tracker.transformer.encoder.parameters() if p.requires_grad)
    w.close()


@gpu_blocks
def test_train_forward_identity_zero_dose():
    """Step-1 loss terms are bitwise identical: geoset_pred_ft vs geoset_cut0_ft on the same batch, and drop None vs an
    all-False drop.  Gradients and later steps are NOT compared (backward is nondeterministic on this GPU)."""
    from geomem import train as T
    out = {}
    for name, arm, zero_drop in (("pred", "geoset_pred_ft", False), ("cut0", "geoset_cut0_ft", False), ("zero", "geoset_pred_ft", True)):
        w, opt, b, rng = _pred_batch(arm)
        if zero_drop:
            b["ref_drop"] = torch.zeros(b["ref_masks"].shape[0], L, dtype=torch.bool, device="cuda")
        opt.zero_grad(set_to_none=True)
        out[name] = T.train_one_step(w, b, arm, rng)
        w.close()
        del w, opt, b
        torch.cuda.empty_cache()
    keys = ("loss", "focal", "dice", "iou", "presence", "n_mem")
    assert all(out["pred"][k] == out["cut0"][k] == out["zero"][k] for k in keys), out


# ---------------------------------------------------------------- GPU: the adopted decode path unchanged
@gpu_blocks
def test_adopted_decode_byte_identical_on_blocks():
    from methods import common
    from methods.geomem import run as R
    e11 = os.path.join(DATA, "Blocks", "Result", "geoset_pred_ft_rep_x_e11")
    if not os.path.isdir(e11):
        pytest.skip("no E11 output on Blocks")
    scene = common.load_scene(DATA, "Blocks")
    out = "dbg_e12_adopted"
    for n in (out, out + "_r1"):
        shutil.rmtree(os.path.join(DATA, "Blocks", "Result", n), ignore_errors=True)
    from methods import sam3_single
    R.decode_scene.model = sam3_single.load_model()
    R.decode_scene.init_state = R.snapshot_init(R.decode_scene.model)
    try:
        args = argparse.Namespace(data=DATA, out=out, r1_out=out + "_r1", arm="geoset_pred_ft", gate="reproj", untrained=False,
                                  runs=os.path.join(REPO, "geomem", "runs"), tau=0.7, k=3, round2="pred", xcl="out")
        R.decode_scene(scene, args)
    finally:
        R.decode_scene.model, R.decode_scene.init_state = None, None

    def pngs(d):
        return {(c, f): open(os.path.join(d, c, f), "rb").read() for c in sorted(os.listdir(d)) if c != scene.ref_cam
                and os.path.isdir(os.path.join(d, c)) for f in sorted(os.listdir(os.path.join(d, c))) if f.endswith(".png")}
    a, b = pngs(os.path.join(DATA, "Blocks", "Result", out)), pngs(e11)
    assert a and set(a) == set(b) and all(a[k] == b[k] for k in a)
    for name in ("round1_conf.json", "xcl_conf.json"):
        assert open(os.path.join(DATA, "Blocks", "Result", out, name), "rb").read() == open(os.path.join(e11, name), "rb").read()
