"""GeoFuse spec 2.2: ViewFusion — per-cell cross-view attention with a zero-init gate (CPU)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.fusion import ViewFusion  # noqa: E402

L, D, N = 5184, 256, 3
KEYS = {"self_mass", "view_entropy", "valid_views_mean"}


def _inputs(seed=0, valid=None):
    g = torch.Generator().manual_seed(seed)
    F_t = torch.randn(L, D, generator=g)
    W = torch.randn(N, L, D, generator=g)
    if valid is None:
        valid = torch.rand(N, L, generator=g) > 0.3
    return F_t, W, valid


def test_fusion_zero_gate_is_identity():
    fu = ViewFusion()
    F_t, W, valid = _inputs()
    out, stats = fu(F_t, W, valid)
    assert torch.equal(out, F_t) and KEYS <= set(stats)


def test_fusion_gate_changes_output():
    fu = ViewFusion()
    with torch.no_grad():
        fu.g.fill_(0.5)
    F_t, W, valid = _inputs()
    out, _ = fu(F_t, W, valid)
    assert not torch.equal(out, F_t) and torch.isfinite(out).all()


def test_fusion_all_invalid_returns_self():
    fu = ViewFusion()
    with torch.no_grad():
        fu.g.fill_(1.0)
    F_t, W, _ = _inputs()
    valid = torch.zeros(N, L, dtype=torch.bool)
    out, stats = fu(F_t, W, valid)
    expected = F_t + fu.o_proj(fu.v_proj(F_t))                  # attention over the self token alone
    assert torch.isfinite(out).all() and torch.allclose(out, expected, atol=1e-5)
    assert stats["self_mass"] == 1.0 and stats["view_entropy"] == 0.0 and stats["valid_views_mean"] == 0.0


def test_fusion_stats_ranges():
    fu = ViewFusion()
    F_t, W, valid = _inputs(seed=1)
    _, stats = fu(F_t, W, valid)
    assert 0.0 <= stats["self_mass"] <= 1.0 and 0.0 <= stats["view_entropy"] <= 1.0
    assert abs(stats["valid_views_mean"] - valid.float().sum(0).mean().item()) < 1e-6


def test_fusion_param_count():
    assert sum(p.numel() for p in ViewFusion().parameters()) == 4 * (256 * 256 + 256) + 1


def test_fusion_gate_receives_gradient():
    fu = ViewFusion()
    F_t, W, valid = _inputs(seed=2)
    out, _ = fu(F_t, W, valid)
    out.sum().backward()
    assert fu.g.grad is not None and fu.g.grad.abs().item() > 0


# ---------------------------------------------------------------- GPU: fusion inside decode_view (Blocks) + arm registration
import pytest  # noqa: E402
from tests_torch.test_wrapper import setup  # noqa: E402,F401

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "Blocks")), reason="needs GPU and Data/Blocks")


def _fuse_wrapper(s):
    from geomem.bias import GeoBias
    from geomem.embed import GeoEmbed
    from geomem.wrapper import GeoMemStar
    return GeoMemStar(s["model"], GeoEmbed().cuda(), geo_bias=GeoBias().cuda(), fusion=ViewFusion().cuda(),
                      use_geometry=False, use_bias=True)


def _aligned_for(s, cam):
    """Align {cam4 (reference), cam9} onto `cam`'s grid using the Blocks Pi3 cache and the setup's backbone maps."""
    from geomem.align import align_views, reference_frame, view_geometry
    from geomem.pointmap import view_grid
    cache = os.path.join("Data", "Blocks", "Cache", "pi3")
    scale, c2w_ref = reference_frame(cache, s["scene"].ref_cam, "cuda")
    feats = {"cam4": s["ref_bb"]["backbone_fpn"][2][0], "cam9": s["views"]["cam9"][0]["backbone_fpn"][2][0]}
    views = [dict(feat=feats[c], **view_geometry(cache, c, "cuda")) for c in ("cam4", "cam9")]
    bb, p, v = s["views"][cam]
    g = view_grid(cache, cam, s["scene"].ref_cam)
    return align_views(p, v, scale, c2w_ref, views)


@needs_gpu
def test_decode_view_with_aligned_runs(setup):
    s = setup
    g = _fuse_wrapper(s)
    bb0, p0, v0 = s["views"]["cam0"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
        B = s["masks"].shape[0]
        aligned = _aligned_for(s, "cam0")
        assert aligned[1].any(), "cam4/cam9 must see some cam0 cells"
        plain = g.decode_view(bb0, B, ref, p0, v0)["logits_1008"]
        fused = g.decode_view(bb0, B, ref, p0, v0, aligned=aligned)["logits_1008"]
    assert torch.equal(plain, fused)                                       # g = 0 -> byte-identical
    assert KEYS <= set(g.last_fusion_stats) and 0.0 <= g.last_fusion_stats["self_mass"] <= 1.0


@needs_gpu
def test_decode_view_gate_changes_output(setup):
    s = setup
    g = _fuse_wrapper(s)
    with torch.no_grad():
        g.fusion.g.fill_(0.5)
    bb0, p0, v0 = s["views"]["cam0"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
        B = s["masks"].shape[0]
        plain = g.decode_view(bb0, B, ref, p0, v0)["logits_1008"].float()
        fused = g.decode_view(bb0, B, ref, p0, v0, aligned=_aligned_for(s, "cam0"))["logits_1008"].float()
    assert not torch.equal(plain, fused) and torch.isfinite(fused).all()


def test_geofuse_arm_flags_and_freeze():
    from geomem.train import arm_flags
    base = arm_flags("geoset_pred_ft")
    flags = arm_flags("geofuse_ft")
    assert flags == dict(base, use_fusion=True) and base["use_fusion"] is False


# ---------------------------------------------------------------- GeoFuse2 (discriminating rerun): two-sided validity, own view excluded
def test_geofuse2_arm_flags():
    from geomem.train import arm_flags
    f1, f2 = arm_flags("geofuse_ft"), arm_flags("geofuse2_ft")
    assert f1["fuse_two_sided"] is False and f1["fuse_exclude_self"] is False
    assert f2 == dict(f1, fuse_two_sided=True, fuse_exclude_self=True)


def test_fusion_view_subset_excludes_self():
    from geomem.align import fusion_views
    views = [dict(feat=torch.zeros(1), name=c) for c in ("a", "b", "c")]
    assert [v["name"] for v in fusion_views(views, ["a", "b", "c"], exclude="b")] == ["a", "c"]
    assert [v["name"] for v in fusion_views(views, ["a", "b", "c"], exclude=None)] == ["a", "b", "c"]
    assert [v["name"] for v in fusion_views(views, ["a", "b", "c"], exclude="zzz")] == ["a", "b", "c"]
