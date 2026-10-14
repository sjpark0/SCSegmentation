"""S1-E9 (docs/stage1-E9-prereg.md): mask-decoder fine-tuning arms — band loss, freezing, optimizer groups, the
decoder receiving gradient in the real loop, the checkpoint carrying the decoder, and the per-scene restore
of the initial weights (watch value Z2)."""
import csv
import json
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data")
sys.path.insert(0, REPO)
from geomem.losses import R_BAND, W_BAND, band_bce, boundary_band, supervised_loss  # noqa: E402

gpu_blocks = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks", "Cache", "sam3_feats")),
                                reason="needs GPU and the Blocks caches")


# ---------------------------------------------------------------- band loss (CPU)
def _rect(B, h, w, y0, y1, x0, x1):
    m = torch.zeros(B, 1, h, w); m[:, :, y0:y1, x0:x1] = 1.0; return m


def test_boundary_band_is_the_symmetric_ring():
    t = _rect(1, 40, 40, 10, 30, 10, 30)                       # 20x20 square
    band = boundary_band(t, r=2)
    assert band.dtype == torch.bool and band.shape == t.shape
    assert bool(band[0, 0, 10, 10]) and bool(band[0, 0, 8, 10]) and bool(band[0, 0, 11, 11])     # on, outside, inside the edge
    assert not bool(band[0, 0, 20, 20]) and not bool(band[0, 0, 5, 5])                            # deep inside / far outside
    assert int(band.sum()) == 24 * 24 - 16 * 16                                                   # dilate(2) minus erode(2)
    assert int(boundary_band(torch.zeros(1, 1, 8, 8), r=2).sum()) == 0                            # absent object: no band


def test_band_bce_perfect_near_zero_and_shifted_large():
    t = _rect(2, 48, 48, 12, 36, 12, 36)
    good = torch.where(t > 0.5, 10.0, -10.0)
    assert band_bce(good, t, r=R_BAND) < 1e-3
    shifted = torch.where(_rect(2, 48, 48, 12, 36, 16, 40) > 0.5, 10.0, -10.0)
    assert band_bce(shifted, t, r=R_BAND) > 1.0
    # objects without a band (absent) contribute 0, not NaN
    t2 = t.clone(); t2[1] = 0
    assert torch.isfinite(band_bce(good, t2)) and band_bce(good, t2) < 1e-3


def test_supervised_loss_band_flag():
    out = {"logits_1008": torch.randn(2, 1, 64, 64, requires_grad=True), "ious": torch.tensor([0.5, 0.5], requires_grad=True),
           "presence": torch.tensor([0.0, 0.0], requires_grad=True)}
    tgt = torch.zeros(2, 40, 50, dtype=torch.bool); tgt[0, 10:30, 10:40] = True
    l0, t0 = supervised_loss(out, tgt)
    l1, t1 = supervised_loss(out, tgt, band=True)
    assert t0["band"] == 0.0 and t1["band"] > 0
    assert abs(float((l1 - l0).detach()) - W_BAND * t1["band"]) < 1e-5
    l1.backward()
    assert torch.isfinite(out["logits_1008"].grad).all()


def test_dec_arm_flags_are_the_pred_recipe_plus_decoder():
    from geomem.train import ARMS, BIAS_ARMS, FT_ARMS, SET_ARMS, arm_flags
    base = arm_flags("geoset_pred_ft")
    assert base["train_decoder"] is False and base["band_loss"] is False
    for a, band in (("geoset_dec_ft", False), ("geoset_decb_ft", True)):
        assert a in ARMS and a in BIAS_ARMS and a in FT_ARMS and a in SET_ARMS
        f = arm_flags(a)
        assert f["train_decoder"] is True and f["band_loss"] is band
        assert {k: v for k, v in f.items() if k not in ("train_decoder", "band_loss")} == \
               {k: v for k, v in base.items() if k not in ("train_decoder", "band_loss")}
    assert arm_flags("geoset_only_ft")["train_decoder"] is False and arm_flags("geofuse_ft")["train_decoder"] is False


# ---------------------------------------------------------------- freezing / optimizer / restore (GPU model)
@pytest.fixture(scope="module")
def model():
    pytest.importorskip("sam3")
    if not torch.cuda.is_available():
        pytest.skip("GPU")
    from methods import sam3_single
    return sam3_single.load_model()


def _n(params):
    return sum(p.numel() for p in params)


def test_freeze_trains_decoder_without_conv_s(model):
    from geomem.train import build_optimizer, build_wrapper
    from geomem.wrapper import DECODER_FROZEN, decoder_parameters
    w = build_wrapper(model, "geoset_dec_ft")
    names = w.freeze("geoset_dec_ft")
    dec = [n for n in names if n.startswith("tracker.sam_mask_decoder.")]
    assert dec and not any(n.split("tracker.sam_mask_decoder.")[1].startswith(DECODER_FROZEN) for n in dec)
    assert any("output_upscaling" in n for n in dec) and any("output_hypernetworks_mlps" in n for n in dec)
    assert all(not p.requires_grad for n, p in model.tracker.sam_mask_decoder.named_parameters() if n.startswith(DECODER_FROZEN))
    assert all(not p.requires_grad for p in model.tracker.obj_ptr_proj.parameters())
    n_dec = _n(p for _, p in decoder_parameters(model.tracker))
    assert n_dec > 1_000_000
    assert _n(w.trainable_parameters()) == 5_922_321 + n_dec                        # Z1: the geoset_pred_ft budget + decoder
    opt = build_optimizer(w, "geoset_dec_ft")
    groups = {g["name"]: g for g in opt.param_groups}
    assert groups["mask_decoder"]["lr"] == 1e-5 and _n(groups["mask_decoder"]["params"]) == n_dec
    # the adopted arm is untouched: no decoder group, decoder frozen
    w.close()
    w0 = build_wrapper(model, "geoset_pred_ft"); names0 = w0.freeze("geoset_pred_ft")
    assert not any(n.startswith("tracker.sam_mask_decoder.") for n in names0)
    assert all(not p.requires_grad for p in model.tracker.sam_mask_decoder.parameters())
    assert "mask_decoder" not in {g["name"] for g in build_optimizer(w0, "geoset_pred_ft").param_groups}
    w0.close()


def test_restore_init_puts_decoder_and_attention_back(model):
    from methods.geomem import run as R
    init = R.snapshot_init(model)
    dec_p = next(p for n, p in model.tracker.sam_mask_decoder.named_parameters() if "output_upscaling" in n)
    attn_p = next(iter(model.tracker.transformer.encoder.parameters()))
    before = (dec_p.detach().clone(), attn_p.detach().clone())
    with torch.no_grad():
        dec_p.add_(1.0); attn_p.add_(1.0)
    R.restore_init(model, init, {"memory_attention": None})          # a checkpoint without the decoder: decoder untouched
    assert torch.equal(attn_p, before[1]) and not torch.equal(dec_p, before[0])
    R.restore_init(model, init, {"memory_attention": None, "mask_decoder": None})
    assert torch.equal(dec_p, before[0]) and torch.equal(attn_p, before[1])
    with pytest.raises(AssertionError):
        R.restore_init(model, None, {"mask_decoder": None})


@gpu_blocks
def test_decoder_receives_gradient_in_the_real_loop(model):
    import numpy as np
    from geomem import train as T
    from geomem.data import Sample, choose_objects, load_sample, objects_for
    from geomem.train import build_optimizer, build_wrapper
    from geomem.wrapper import decoder_parameters
    torch.manual_seed(0); rng = np.random.default_rng(0)
    for arm in ("geoset_dec_ft", "geoset_decb_ft"):
        w = build_wrapper(model, arm); w.freeze(arm); opt = build_optimizer(w, arm)
        s = Sample("Blocks", 0, "cam4", "cam0")
        b = load_sample(DATA, s, choose_objects(objects_for(s, DATA), 16, rng), "cuda", extra_mem_cam="cam2", extra_mem_gt=False)
        opt.zero_grad(set_to_none=True)
        terms = T.train_one_step(w, b, arm, rng)
        assert terms["loss"] == terms["loss"]
        assert (terms["band"] > 0) == (arm == "geoset_decb_ft")
        dec = [p for _, p in decoder_parameters(model.tracker)]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec), f"{arm}: decoder got no gradient"
        assert all(p.grad is None for n, p in model.tracker.sam_mask_decoder.named_parameters() if n.startswith(("conv_s0", "conv_s1")))
        assert w.geo_bias.w.grad is not None and w.geo_bias.w.grad.abs().sum() > 0
        opt.zero_grad(set_to_none=True)
        w.close()


@gpu_blocks
@pytest.mark.parametrize("arm", ["geoset_dec_ft", "geoset_decb_ft"])
def test_train_smoke_dec_arms(tmp_path, arm):
    cmd = [sys.executable, "-m", "geomem.train", "--arm", arm, "--fold", "0", "--max-steps", "20",
           "--scenes", "Blocks", "--out", str(tmp_path), "--data", DATA]
    subprocess.run(cmd, check=True, cwd=REPO)
    d = os.path.join(tmp_path, arm, "fold0")
    rows = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
    cfg = json.load(open(os.path.join(d, "config.json")))
    ck = torch.load(os.path.join(d, "ckpt.pt"), map_location="cpu")
    assert len(rows) == 20 and all(float(r["loss"]) == float(r["loss"]) for r in rows)
    assert cfg["train_decoder"] is True and cfg["band_loss"] == (arm == "geoset_decb_ft")
    assert cfg["extra_mem_pred"] is True and cfg["extra_mem_prob"] == 0.5 and cfg["use_geometry"] is False
    assert cfg["trainable_params"] > 5_922_321 + 1_000_000 and cfg["lr"]["mask_decoder"] == 1e-5
    assert not any("conv_s" in n for n in cfg["trainable_names"])
    assert "mask_decoder" in ck and "memory_attention" in ck and "geo_bias" in ck
    assert any(k.startswith("conv_s0") for k in ck["mask_decoder"])                   # the full decoder state (exact load)
    assert {int(r["n_mem"]) for r in rows} == {1, 2}
    if arm == "geoset_decb_ft":
        assert all(float(r["band"]) > 0 for r in rows)
    else:
        assert all(float(r["band"]) == 0.0 for r in rows)
