"""GeoFuse spec 2.1: align other views' 72x72 features to the target grid (CPU, synthetic geometry)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.align import align_views, project_to_view, view_geometry  # noqa: E402

G = 72
Z = 2.0


def _identity_view(feat=None, depth=None):
    """A view whose work resolution IS the 72x72 grid with K = I: pixel (u, v) == cell (col, row)."""
    if feat is None:
        feat = torch.arange(G * G, dtype=torch.float32).reshape(1, G, G).expand(256, -1, -1).clone()
    if depth is None:
        depth = torch.full((G, G), Z)
    return {"feat": feat, "depth": depth, "K": torch.eye(3), "c2w": torch.eye(4)}


def _target():
    """Target cells = the identity view's pixels at depth Z, expressed in the reference frame (= world, c2w_ref = I)."""
    v, u = torch.meshgrid(torch.arange(G, dtype=torch.float32), torch.arange(G, dtype=torch.float32), indexing="ij")
    p = torch.stack([u * Z, v * Z, torch.full_like(u, Z)], dim=-1).reshape(-1, 3)
    return p, torch.ones(G * G, dtype=torch.bool), 1.0, torch.eye(4)


def test_project_to_view_identity():
    p, _, _, c2w_ref = _target()
    u, v, z = project_to_view(p, c2w_ref, torch.eye(4), torch.eye(3))
    assert torch.allclose(u, p[:, 0] / Z) and torch.allclose(v, p[:, 1] / Z) and torch.allclose(z, torch.full_like(z, Z))


def test_align_self_view_is_identity():
    p, valid_t, scale, c2w_ref = _target()
    view = _identity_view()
    W, valid = align_views(p, valid_t, scale, c2w_ref, [view])
    assert W.shape == (1, G * G, 256) and valid.shape == (1, G * G) and W.dtype == torch.float32
    assert valid.all()
    assert torch.allclose(W[0, :, 0], torch.arange(G * G, dtype=torch.float32), atol=1e-4)


def test_align_occluded_cells_invalid():
    p, valid_t, scale, c2w_ref = _target()
    depth = torch.full((G, G), Z)
    depth[:, :36] = 1.0                                        # the view sees something nearer on its left half
    W, valid = align_views(p, valid_t, scale, c2w_ref, [_identity_view(depth=depth)])
    vg = valid[0].reshape(G, G)
    assert not vg[:, :36].any() and vg[:, 36:].all()


def test_align_out_of_bounds_invalid():
    p, valid_t, scale, c2w_ref = _target()
    c2w = torch.eye(4)
    c2w[0, 3] = 100.0 * Z                                      # camera moved so every point lands 100 px outside
    view = _identity_view(); view["c2w"] = c2w
    W, valid = align_views(p, valid_t, scale, c2w_ref, [view])
    assert not valid.any() and (W == 0).all()


def test_align_invalid_target_cell_invalid_everywhere():
    p, valid_t, scale, c2w_ref = _target()
    valid_t = valid_t.clone(); valid_t[0] = False
    _, valid = align_views(p, valid_t, scale, c2w_ref, [_identity_view(), _identity_view()])
    assert valid.shape[0] == 2 and not valid[:, 0].any() and valid[:, 1].all()


def test_align_accepts_half_features():
    p, valid_t, scale, c2w_ref = _target()
    feat = torch.rand(256, G, G)
    W32, _ = align_views(p, valid_t, scale, c2w_ref, [_identity_view(feat=feat)])
    W16, _ = align_views(p, valid_t, scale, c2w_ref, [_identity_view(feat=feat.half())])
    assert W16.dtype == torch.float32 and (W32 - W16).abs().max().item() < 1e-2


def test_view_geometry_missing_names_file(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        view_geometry(str(tmp_path), "camera_0099", "cpu")
    assert "camera_0099.npz" in str(e.value)


def test_view_geometry_reads_npz(tmp_path):
    np.savez(tmp_path / "cam1.npz", depth=np.ones((4, 6), np.float32), conf=np.ones((4, 6), np.float32),
             K=np.eye(3, dtype=np.float32), c2w=np.eye(4, dtype=np.float32))
    g = view_geometry(str(tmp_path), "cam1", "cpu")
    assert g["depth"].shape == (4, 6) and g["K"].shape == (3, 3) and g["c2w"].shape == (4, 4)


# ---------------------------------------------------------------- real geometry (Blocks, level-72 cache): the convention check
# the synthetic tests cannot make (K = I, c2w = I would pass a transposed or mis-inverted projection just the same)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_L72 = os.path.join(REPO, "Data", "Blocks", "Cache", "sam3_feats72")


@pytest.mark.skipif(not os.path.isdir(_L72), reason="needs Data/Blocks and its level-72 cache")
def test_align_real_blocks_self_identity_and_cross_similarity():
    import json
    from geomem.align import reference_frame
    from geomem.cache import pi3_frame_cache
    from geomem.data import load_level72
    from geomem.embed import grid_tensors
    from geomem.pointmap import view_grid
    data = os.path.join(REPO, "Data")
    meta = json.load(open(os.path.join(data, "Blocks", "meta.json")))
    ref, t = meta["ref_cam"], meta["train_frames"][0]
    cache = pi3_frame_cache(data, "Blocks", t)
    cams, feats = load_level72(data, "Blocks", t, "cpu")
    views = [dict(feat=feats[i], **view_geometry(cache, c, "cpu")) for i, c in enumerate(cams)]
    scale, c2w_ref = reference_frame(cache, ref, "cpu")
    cos = torch.nn.functional.cosine_similarity
    for ti in (0, 4):
        p, v = grid_tensors(view_grid(cache, cams[ti], ref), "cpu")
        W, valid = align_views(p, v, scale, c2w_ref, views)
        own = feats[ti].float().reshape(256, -1).T
        assert valid[ti].float().mean() > 0.9
        assert cos(W[ti][valid[ti]], own[valid[ti]], dim=-1).mean() > 0.99          # a view aligned onto itself
        others = [j for j in range(len(cams)) if j != ti and valid[j].sum() > 200]
        cross = torch.stack([cos(W[j][valid[j]], own[valid[j]], dim=-1).mean() for j in others]).mean()
        rnd = torch.stack([cos(W[j][valid[j]], own[torch.randperm(5184)][valid[j]], dim=-1).mean() for j in others]).mean()
        assert cross > 0.4 and cross > 3 * rnd, (cross, rnd)                          # measured: ~0.55 vs ~0.12


# ---------------------------------------------------------------- review follow-up: two-sided depth consistency of valid alignments
def test_depth_consistency_two_sided():
    from geomem.align import alignment_depth_ratio
    p, valid_t, scale, c2w_ref = _target()                       # points at z = 2
    near = _identity_view(depth=torch.full((G, G), 1.0))         # the view's surface is NEARER than the point (occluded)
    far = _identity_view(depth=torch.full((G, G), 4.0))          # the view's surface is FARTHER: one-sided rule calls it valid
    same = _identity_view()
    ratio = alignment_depth_ratio(p, scale, c2w_ref, [near, far, same])
    assert ratio.shape == (3, G * G)
    assert torch.allclose(ratio[0], torch.full((G * G,), 2.0)) and torch.allclose(ratio[1], torch.full((G * G,), 0.5))
    assert torch.allclose(ratio[2], torch.ones(G * G))
    _, valid = align_views(p, valid_t, scale, c2w_ref, [near, far, same])
    assert not valid[0].any() and valid[1].all() and valid[2].all()        # the one-sided rule accepts the "far" view
    consistent = ((ratio - 1).abs() <= 0.05) & valid
    assert not consistent[1].any() and consistent[2].all()                 # the two-sided statistic does not


# ---------------------------------------------------------------- GeoAlign spec 2.1: two-sided validity and hole filling
def test_two_sided_rejects_points_in_front():
    from geomem.align import TWO_SIDED_TOL
    p, valid_t, scale, c2w_ref = _target()                       # points at z = 2
    far = _identity_view(depth=torch.full((G, G), 4.0))          # the point floats in FRONT of the view's surface (ratio 0.5)
    same = _identity_view()
    _, one_sided = align_views(p, valid_t, scale, c2w_ref, [far, same])
    _, two_sided = align_views(p, valid_t, scale, c2w_ref, [far, same], two_sided=True)
    assert one_sided[0].all() and one_sided[1].all()
    assert not two_sided[0].any() and two_sided[1].all()
    assert TWO_SIDED_TOL == 0.05


def test_fill_aligned_mixes_by_validity():
    from geomem.align import fill_aligned
    W = torch.ones(G * G, 8); F_ref = torch.full((G * G, 8), 2.0, dtype=torch.float16)
    valid = torch.zeros(G * G, dtype=torch.bool); valid[::2] = True
    out = fill_aligned(W, valid, F_ref)
    assert out.dtype == torch.float16 and (out[::2] == 1).all() and (out[1::2] == 2).all()


def test_fill_aligned_all_invalid_is_reference():
    from geomem.align import fill_aligned
    W = torch.randn(G * G, 8); F_ref = torch.randn(G * G, 8)
    out = fill_aligned(W, torch.zeros(G * G, dtype=torch.bool), F_ref)
    assert torch.equal(out, F_ref)


# ---------------------------------------------------------------- GeoAlign spec 2.1: aligned memory views (GPU, Blocks)
from tests_torch.test_wrapper import setup  # noqa: E402,F401

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(REPO, "Data", "Blocks")),
                               reason="needs GPU and Data/Blocks")


def _bias_wrapper(s):
    from geomem.bias import GeoBias
    from geomem.embed import GeoEmbed
    from geomem.wrapper import GeoMemStar
    return GeoMemStar(s["model"], GeoEmbed().cuda(), geo_bias=GeoBias().cuda(), use_geometry=False, use_bias=True)


def _aligned_on_ref(s, cam):
    """Align `cam`'s level-72 map (from the setup's backbones) onto the reference grid, two-sided."""
    from geomem.align import reference_frame
    cache = os.path.join(REPO, "Data", "Blocks", "Cache", "pi3")
    scale, c2w_ref = reference_frame(cache, s["scene"].ref_cam, "cuda")
    feat = (s["ref_bb"] if cam == s["scene"].ref_cam else s["views"][cam][0])["backbone_fpn"][2][0]
    view = dict(feat=feat, **view_geometry(cache, cam, "cuda"))
    W, valid = align_views(s["p_ref"], s["v_ref"], scale, c2w_ref, [view], two_sided=True)
    return W[0], valid[0]


@needs_gpu
def test_encode_aligned_exact_map_is_byte_identical(setup):
    """Plumbing: feeding the reference's own level-72 map through encode_aligned reproduces the reference memory exactly."""
    s = setup
    g = _bias_wrapper(s)
    f2 = s["ref_bb"]["backbone_fpn"][2]
    W = f2[0].reshape(f2.shape[1], -1).T
    valid = torch.ones(W.shape[0], dtype=torch.bool, device=W.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"]).views[0]
        mine = g.encode_aligned(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"], W, valid, "self")
    for k in ("feats", "obj_ptr_tokens", "object_score_logits"):
        assert torch.equal(getattr(mine, k), getattr(ref, k)), k
    assert mine.name == "self" and mine.conf is None


@needs_gpu
def test_encode_aligned_self_view_is_near_identity(setup):
    """Geometry: the reference view aligned onto its own grid (area-averaged cell points re-projected, bilinear) is
    near-identical, not byte-identical — the memory it yields is within bf16 noise of the reference memory."""
    s = setup
    g = _bias_wrapper(s)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"]).views[0]
        W, valid = _aligned_on_ref(s, s["scene"].ref_cam)
        assert valid.float().mean() > 0.9
        mine = g.encode_aligned(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"], W, valid, "self")
    a, b = mine.feats.float(), ref.feats.float()
    assert torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1).mean() > 0.99
    assert ((a - b).abs().max() / b.abs().max()).item() < 0.1
    assert torch.equal(mine.object_score_logits > 0, ref.object_score_logits > 0)


@needs_gpu
def test_encode_aligned_other_view_differs_and_decodes(setup):
    s = setup
    g = _bias_wrapper(s)
    B = s["masks"].shape[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
        W, valid = _aligned_on_ref(s, "cam9")
        assert valid.float().mean() > 0.3
        v9 = g.encode_aligned(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"], W, valid, "cam9")
        assert not torch.equal(v9.feats, ref.views[0].feats) and torch.isfinite(v9.feats.float()).all()
        bb0, p0, v0 = s["views"]["cam0"]
        out = g.decode_view(bb0, B, ref.with_view(v9), p0, v0)
    assert out["logits_1008"].shape == (B, 1, 1008, 1008) and torch.isfinite(out["logits_1008"].float()).all()


@needs_gpu
def test_encode_aligned_does_not_mutate_backbone(setup):
    s = setup
    g = _bias_wrapper(s)
    before = [f.clone() for f in s["ref_bb"]["backbone_fpn"]]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        W, valid = _aligned_on_ref(s, "cam9")
        g.encode_aligned(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"], W, valid, "cam9")
    assert all(torch.equal(a, b) for a, b in zip(before, s["ref_bb"]["backbone_fpn"]))
