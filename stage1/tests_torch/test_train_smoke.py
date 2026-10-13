"""20-step training smoke on Blocks (GPU, needs the Blocks caches)."""
import csv
import math
import json
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks", "Cache", "sam3_feats")),
                                reason="needs GPU and the Blocks caches")


@pytest.fixture(autouse=True)
def _free_gpu_between_tests():
    """In-process tests load a model each; release it before the next subprocess training (48 GB GPU shared)."""
    yield
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run(tmp, arm, steps):
    cmd = [sys.executable, "-m", "geomem.train", "--arm", arm, "--fold", "0", "--max-steps", str(steps),
           "--scenes", "Blocks", "--out", tmp, "--data", DATA,
           "--init-from-runs", os.path.join(REPO, "geomem", "runs")]      # warm start of the GeoQuery arms
    subprocess.run(cmd, check=True, cwd=REPO)
    d = os.path.join(tmp, arm, "fold0")
    rows = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
    cfg = json.load(open(os.path.join(d, "config.json")))
    ck = torch.load(os.path.join(d, "ckpt.pt"), map_location="cpu")
    return rows, cfg, ck


def test_train_smoke_20_steps(tmp_path):
    rows, cfg, ck = _run(str(tmp_path), "geomem", 20)
    assert len(rows) == 20 and rows[0].keys() >= {"step", "epoch", "loss", "focal", "dice", "iou", "presence", "ssl", "lr", "seconds"}
    assert all(float(r["loss"]) == float(r["loss"]) for r in rows)          # finite (not NaN)
    assert ck["geo_embed"]["key_head.weight"].abs().sum() > 0                # heads trained away from zero
    assert cfg["trainable_params"] < 200_000 and cfg["arm"] == "geomem" and cfg["fold"] == 0
    assert "Blocks" in cfg["scenes"] and len(cfg["ckpt_sha256"]) == 64


def test_train_ft_arm_trains_memory_attention(tmp_path):
    rows, cfg, ck = _run(str(tmp_path), "geomem_ft", 2)
    assert cfg["trainable_params"] > 1_000_000 and "memory_attention" in ck


def test_train_ssl_arm_logs_ssl_term(tmp_path):
    rows, cfg, ck = _run(str(tmp_path), "geomem_ssl", 2)
    assert all(float(r["ssl"]) == float(r["ssl"]) and float(r["ssl"]) > 0 for r in rows)


def test_embedding_mlp_receives_gradient_in_the_real_loop():
    """Review follow-up: in the real loop the MLP and e_none never trained.  Cause: the SAM 3 tracker enters a
    process-wide autocast context in __init__ (SCSam3TrackerPredictor.py:52) that never exits, so autocast's
    weight-cast cache is never cleared and every forward reuses the bf16 casts made at the FIRST forward (zero
    heads).  After the fix, the MLP must get a non-zero gradient from step 2 on and the forward must use the
    updated head weights."""
    import numpy as np
    sys.path.insert(0, REPO)
    from geomem.data import Sample, choose_objects, load_sample, objects_for
    from geomem.embed import GeoEmbed
    from geomem.train import build_optimizer, train_one_step
    from geomem.wrapper import GeoMemStar
    from methods import sam3_single
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = sam3_single.load_model()
    assert not torch.is_autocast_cache_enabled(), "load_model must disable the autocast weight cache"
    geo = GeoEmbed().cuda(); w = GeoMemStar(model, geo); w.freeze("geomem"); opt = build_optimizer(w, "geomem")
    s = Sample("Blocks", 0, "cam4", "cam0")
    b = load_sample(DATA, s, choose_objects(objects_for(s, DATA), 16, rng), "cuda")
    for step in range(1, 4):
        opt.zero_grad(set_to_none=True)
        train_one_step(w, b, "geomem", rng)
        if step >= 2:
            assert geo.mlp[0].weight.grad.abs().max().item() > 0, f"step {step}: MLP got no gradient"
        torch.nn.utils.clip_grad_norm_(w.trainable_parameters(), 1.0); opt.step()
    # the forward must see the updated heads: key output is no longer identically zero
    with torch.autocast("cuda", dtype=torch.bfloat16):
        k, q = geo(b["p_ref"], b["valid_ref"])
    assert k.float().abs().max().item() > 0 and q.float().abs().max().item() > 0


@pytest.mark.parametrize("arm", ["geobias_ft", "geobias_norope_ft", "geobias_only_ft", "geoset_ft"])
def test_train_smoke_new_arms(tmp_path, arm):
    steps = 20 if arm == "geoset_ft" else 3
    rows, cfg, ck = _run(str(tmp_path), arm, steps)
    assert cfg["use_bias"] is True and cfg["no_rope"] == (arm == "geobias_norope_ft")
    assert cfg["use_geometry"] == (arm != "geobias_only_ft")
    assert "geo_bias" in ck and ck["geo_bias"]["w"].abs().sum() > 0           # bias trained away from zero
    assert "memory_attention" in ck
    if arm == "geobias_only_ft":
        assert ck["geo_embed"]["key_head.weight"].abs().sum() == 0              # embedding untouched
    if arm == "geoset_ft":
        n_mem = {int(r["n_mem"]) for r in rows}
        assert n_mem == {1, 2}, n_mem
        assert cfg["extra_mem_prob"] == 0.5
    else:
        assert all(int(r["n_mem"]) == 1 for r in rows)


GBO_CKPT = os.path.join(REPO, "geomem", "runs", "geobias_only_ft", "fold0", "ckpt.pt")
needs_gbo = pytest.mark.skipif(not os.path.isfile(GBO_CKPT), reason="needs the geobias_only_ft fold0 checkpoint (init_from)")


@needs_gbo
@pytest.mark.parametrize("arm", ["geoquery_ft", "geoquery_nobias_ft", "geobias_only_ft2"])
def test_train_smoke_query_arms(tmp_path, arm):
    rows, cfg, ck = _run(str(tmp_path), arm, 3)
    assert len(rows) == 3 and all(float(r["loss"]) == float(r["loss"]) for r in rows)
    assert len(cfg["init_from_sha256"]) == 64 and cfg["use_geometry"] is False and cfg["use_bias"] is True
    assert "memory_attention" in ck and "geo_bias" in ck
    assert all(int(r["n_targets"]) == 2 for r in rows)
    if arm.startswith("geoquery"):
        assert "query_hub" in ck and int(rows[0]["n_views"]) == 8 and cfg["use_hub"] is True
        assert ("obj_bias.w" in ck["query_hub"]) == (arm == "geoquery_ft")
        assert cfg["use_objbias"] == (arm == "geoquery_ft")
    else:
        assert "query_hub" not in ck and int(rows[0]["n_views"]) == 0 and cfg["use_hub"] is False


@needs_gbo
def test_query_hub_receives_gradient_in_the_real_loop():
    import numpy as np
    from geomem.data import PairSample, choose_objects, hub_views, load_sample_pair, objects_for_pair, pair_targets
    from geomem.train import arm_flags, build_optimizer, build_wrapper, load_init_from, train_pair_step
    from methods import sam3_single
    from mvseg2d.io import load_meta
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = sam3_single.load_model()
    w = build_wrapper(model, "geoquery_ft"); w.freeze("geoquery_ft"); load_init_from(w, GBO_CKPT)
    opt = build_optimizer(w, "geoquery_ft")
    s = PairSample("Blocks", 0, "cam4")
    cams = hub_views(load_meta(DATA, "Blocks")["cams"], s.ref_cam, pair_targets(DATA, s), 5, rng)
    b = load_sample_pair(DATA, s, choose_objects(objects_for_pair(s, DATA), 16, rng), cams, "cuda")
    # step 1: only proj_q / type_embed can receive gradient (the gates sit behind the zero proj_q);
    # step 2: proj_q has moved, so the gates and the hub internals get gradient
    for step in (1, 2):
        opt.zero_grad(set_to_none=True)
        terms = train_pair_step(w, b, arm_flags("geoquery_ft"))
        assert terms["n_views"] == 8 and terms["n_targets"] == 2 and terms["loss"] == terms["loss"]
        assert w.hub.proj_q.weight.grad.abs().max().item() > 0
        if step == 2:
            g = w.hub.layers[0].g_cross.grad
            assert g is not None and g.abs().item() > 0, "the hub's cross-attention gate got no gradient after one update"
        torch.nn.utils.clip_grad_norm_(w.trainable_parameters(), 1.0)
        opt.step()
    w.close()


@needs_gbo
def test_train_smoke_ptr_arm(tmp_path):
    rows, cfg, ck = _run(str(tmp_path), "geoquery_ptr_ft", 3)
    assert cfg["inject"] == "ptr" and cfg["g_cross_init"] == 0.1 and cfg["objbias_from_geobias"] is True and cfg["use_hub"] is True
    hub = ck["query_hub"]
    assert "ptr_proj.weight" in hub and hub["ptr_proj.weight"].abs().sum() > 0            # the pointer path trained
    assert hub["obj_bias.w"].abs().sum() > 0.5                                            # started from A's GeoBias kernel
    assert all(int(r["n_views"]) == 8 and int(r["n_targets"]) == 2 for r in rows)


# ---------------------------------------------------------------- GeoSet-2 arms
@pytest.mark.parametrize("arm", ["geoset_only_ft", "geoset_pred_ft"])
def test_train_smoke_set2_arms(tmp_path, arm):
    rows, cfg, ck = _run(str(tmp_path), arm, 20)
    assert cfg["use_bias"] is True and cfg["use_geometry"] is False and cfg["no_rope"] is False
    assert cfg["extra_mem_prob"] == 0.5 and cfg["extra_mem_pred"] == (arm == "geoset_pred_ft")
    assert "geo_bias" in ck and "memory_attention" in ck
    assert not any(n.startswith("geo_embed") for n in cfg["trainable_names"])
    assert ck["geo_embed"]["key_head.weight"].abs().sum() == 0
    n_mem = {int(r["n_mem"]) for r in rows}
    assert n_mem == {1, 2}, n_mem
    two = [r for r in rows if int(r["n_mem"]) == 2]
    if arm == "geoset_pred_ft":
        assert all(0.0 <= float(r["gated_frac"]) <= 1.0 for r in two)
    else:
        assert all(r["gated_frac"] == "" for r in rows)


def _pred_batch():
    import numpy as np
    sys.path.insert(0, REPO)
    from geomem.data import Sample, choose_objects, load_sample, objects_for
    from geomem.train import build_optimizer, build_wrapper
    from methods import sam3_single
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = sam3_single.load_model()
    w = build_wrapper(model, "geoset_pred_ft"); w.freeze("geoset_pred_ft"); opt = build_optimizer(w, "geoset_pred_ft")
    s = Sample("Blocks", 0, "cam4", "cam0")
    b = load_sample(DATA, s, choose_objects(objects_for(s, DATA), 16, rng), "cuda", extra_mem_cam="cam2", extra_mem_gt=False)
    assert "mem_masks" not in b
    return w, opt, b, rng


def test_pred_memory_step_all_gated(monkeypatch):
    from geomem import train as T
    w, opt, b, rng = _pred_batch()
    monkeypatch.setattr(T, "PRED_MEM_TAU", 1.01)             # no predicted IoU reaches 1.01 -> every object gated
    opt.zero_grad(set_to_none=True)
    terms = T.train_one_step(w, b, "geoset_pred_ft", rng)
    assert terms["n_mem"] == 1 and terms["gated_frac"] is None and math.isfinite(terms["loss"])
    w.close()


def test_pred_memory_no_grad_into_prediction(monkeypatch):
    from geomem import train as T
    w, opt, b, rng = _pred_batch()
    seen = {}
    orig = w.encode_memory_from_prediction

    def spy(backbone_out, B, logits_1008, presence, obj_ptr, p_v, valid_v, name, **kw):
        seen["requires_grad"] = logits_1008.requires_grad
        seen["kw"] = dict(kw)
        return orig(backbone_out, B, logits_1008, presence, obj_ptr, p_v, valid_v, name, **kw)
    monkeypatch.setattr(w, "encode_memory_from_prediction", spy)
    monkeypatch.setattr(T, "PRED_MEM_TAU", 0.0)              # every present object kept -> two memories
    opt.zero_grad(set_to_none=True)
    terms = T.train_one_step(w, b, "geoset_pred_ft", rng)
    assert terms["n_mem"] == 2 and 0.0 <= terms["gated_frac"] < 1.0
    assert seen["requires_grad"] is False and seen["kw"]["tau"] == 0.0
    assert w.geo_bias.w.grad is not None and w.geo_bias.w.grad.abs().sum() > 0
    attn = [p for p in w.tracker.transformer.encoder.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in attn)
    w.close()


def test_geofuse_build_and_freeze():
    from geomem.fusion import ViewFusion
    from geomem.train import build_optimizer, build_wrapper
    from methods import sam3_single
    w = build_wrapper(sam3_single.load_model(), "geofuse_ft")
    assert isinstance(w.fusion, ViewFusion)
    names = w.freeze("geofuse_ft")
    assert any(n.startswith("fusion.") for n in names) and any(n.startswith("geo_bias.") for n in names)
    assert all(not n.startswith("geo_embed.") for n in names)
    opt = build_optimizer(w, "geofuse_ft")
    groups = {g["name"]: g["lr"] for g in opt.param_groups}
    assert groups["geo_embed"] == 1e-3 and groups["memory_attention"] == 1e-5
    fusion_ids = {id(p) for p in w.fusion.parameters()}
    assert fusion_ids <= {id(p) for g in opt.param_groups if g["name"] == "geo_embed" for p in g["params"]}
    w.close()


# ---------------------------------------------------------------- GeoFuse (spec 3)
def _ensure_blocks_level72():
    sys.path.insert(0, REPO)
    from geomem.cache import level72_path, train_frames
    from geomem.cache_feats import build_level72
    for t in train_frames(DATA, "Blocks")[::2]:
        if not os.path.isfile(level72_path(DATA, "Blocks", t)):
            build_level72(DATA, "Blocks", t)


def test_train_smoke_geofuse(tmp_path):
    _ensure_blocks_level72()
    rows, cfg, ck = _run(str(tmp_path), "geofuse_ft", 20)
    assert cfg["use_fusion"] is True and cfg["extra_mem_pred"] is True and cfg["use_geometry"] is False
    assert "fusion" in ck and "geo_bias" in ck and "memory_attention" in ck
    assert {int(r["n_mem"]) for r in rows} == {1, 2}
    assert all(r["g"] != "" and r["self_mass"] != "" for r in rows)
    assert all(0.0 <= float(r["self_mass"]) <= 1.0 for r in rows)
    assert abs(float(ck["fusion"]["g"].item()) - float(rows[-1]["g"])) < 1e-3


def test_fusion_receives_gradient_in_the_real_loop(monkeypatch):
    import numpy as np
    _ensure_blocks_level72()
    from geomem import train as T
    from geomem.data import Sample, choose_objects, load_sample, objects_for
    from geomem.train import build_optimizer, build_wrapper
    from methods import sam3_single
    torch.manual_seed(0); rng = np.random.default_rng(0)
    w = build_wrapper(sam3_single.load_model(), "geofuse_ft"); w.freeze("geofuse_ft"); opt = build_optimizer(w, "geofuse_ft")
    s = Sample("Blocks", 0, "cam4", "cam0")
    b = load_sample(DATA, s, choose_objects(objects_for(s, DATA), 16, rng), "cuda", extra_mem_cam="cam2", extra_mem_gt=False,
                    aligned=True)
    assert len(b["views"]) == 10 and b["views"][0]["feat"].shape == (256, 72, 72) and "scale" in b and "c2w_ref" in b
    seen = {}
    orig = w.encode_memory_from_prediction

    def spy(backbone_out, B, logits_1008, presence, obj_ptr, p_v, valid_v, name, **kw):
        seen["requires_grad"] = logits_1008.requires_grad
        return orig(backbone_out, B, logits_1008, presence, obj_ptr, p_v, valid_v, name, **kw)
    monkeypatch.setattr(w, "encode_memory_from_prediction", spy)
    monkeypatch.setattr(T, "PRED_MEM_TAU", 0.0)
    opt.zero_grad(set_to_none=True)
    terms = T.train_one_step(w, b, "geofuse_ft", rng)
    assert terms["n_mem"] == 2 and seen["requires_grad"] is False
    assert terms["g"] == 0.0 and 0.0 <= terms["self_mass"] <= 1.0
    assert w.fusion.g.grad is not None and w.fusion.g.grad.abs().item() > 0
    assert w.fusion.o_proj.weight.grad is None or w.fusion.o_proj.weight.grad.abs().sum() == 0   # g = 0 blocks the rest
    assert w.geo_bias.w.grad is not None and w.geo_bias.w.grad.abs().sum() > 0
    w.close()


# ---------------------------------------------------------------- GeoAlign (spec 3)
def test_train_smoke_geoalign(tmp_path):
    _ensure_blocks_level72()
    rows, cfg, ck = _run(str(tmp_path), "geoalign_ft", 20)
    assert cfg["extra_mem_aligned"] is True and cfg["extra_mem_pred"] is False and cfg["use_fusion"] is False
    assert cfg["use_geometry"] is False and cfg["use_bias"] is True and cfg["trainable_params"] == 5922321
    assert "fusion" not in ck and "geo_bias" in ck and "memory_attention" in ck
    assert {int(r["n_mem"]) for r in rows} == {1, 2}
    two = [r for r in rows if int(r["n_mem"]) == 2]
    assert all(0.0 <= float(r["align_valid"]) <= 1.0 for r in two) and all(r["align_valid"] == "" for r in rows if int(r["n_mem"]) == 1)


def test_geoalign_step_gradients():
    import numpy as np
    _ensure_blocks_level72()
    from geomem import train as T
    from geomem.data import Sample, choose_objects, load_sample, objects_for
    from geomem.train import build_optimizer, build_wrapper
    from methods import sam3_single
    torch.manual_seed(0); rng = np.random.default_rng(0)
    w = build_wrapper(sam3_single.load_model(), "geoalign_ft"); w.freeze("geoalign_ft"); opt = build_optimizer(w, "geoalign_ft")
    s = Sample("Blocks", 0, "cam4", "cam0")
    b = load_sample(DATA, s, choose_objects(objects_for(s, DATA), 16, rng), "cuda", extra_mem_cam="cam2", extra_mem_gt=False,
                    aligned_mem=True)
    assert "mem_backbone" not in b and "mem_view" in b and b["mem_view"]["feat"].shape == (256, 72, 72) and b["mem_cam"] == "cam2"
    opt.zero_grad(set_to_none=True)
    terms = T.train_one_step(w, b, "geoalign_ft", rng)
    assert terms["n_mem"] == 2 and math.isfinite(terms["loss"]) and 0.0 < terms["align_valid"] <= 1.0
    assert w.geo_bias.w.grad is not None and w.geo_bias.w.grad.abs().sum() > 0
    attn = [p for p in w.tracker.transformer.encoder.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in attn)
    w.close()


def test_train_smoke_geofuse2(tmp_path):
    _ensure_blocks_level72()
    rows, cfg, ck = _run(str(tmp_path), "geofuse2_ft", 20)
    assert cfg["use_fusion"] is True and cfg["fuse_two_sided"] is True and cfg["fuse_exclude_self"] is True
    assert "fusion" in ck and {int(r["n_mem"]) for r in rows} == {1, 2}
    assert all(r["g"] != "" and r["self_mass"] != "" for r in rows)
