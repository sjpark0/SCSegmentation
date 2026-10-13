"""GeoSet-2 object-level gate: MemoryView.conf -> MemorySet.gate_mask (CPU), and its effect in decode_view (GPU)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.wrapper import MemorySet, MemoryView  # noqa: E402

L = 5184


def _view(B, conf=None, name="v"):
    z = torch.zeros(L, B, 64)
    return MemoryView(feats=z, pos=z.clone(), obj_ptr_tokens=torch.zeros(4, B, 64), obj_ptr_pos=torch.zeros(4, B, 64),
                      object_score_logits=torch.zeros(B, 1), p=torch.zeros(L, 3), valid=torch.ones(L, dtype=torch.bool),
                      name=name, conf=conf)


def _layout():
    B = 3
    ref = _view(B, None, "ref")
    v1 = _view(B, torch.tensor([True, False, True]), "v1")
    v2 = _view(B, torch.tensor([False, False, True]), "v2")
    return MemorySet([ref, v1, v2]), B


def test_gate_mask_none_when_all_confident():
    B = 2
    ms = MemorySet([_view(B), _view(B, torch.ones(B, dtype=torch.bool))])
    assert ms.gate_mask(torch.float32, "cpu") is None


def test_gate_mask_shape_and_columns():
    ms, B = _layout()
    m = ms.gate_mask(torch.float32, "cpu")
    V = 3
    assert m.shape == (B, 1, 1, V * L + 4 * V)
    inf = float("-inf")
    row1 = m[1, 0, 0]                                   # object 1: unconfident in v1 AND v2
    assert (row1[L:3 * L] == inf).all() and (row1[V * L + 4:] == inf).all()
    assert (row1[:L] == 0).all() and (row1[V * L:V * L + 4] == 0).all()
    row0 = m[0, 0, 0]
    assert (row0[2 * L:3 * L] == inf).all() and (row0[V * L + 8:] == inf).all()
    assert (row0[L:2 * L] == 0).all() and (row0[V * L + 4:V * L + 8] == 0).all()
    assert (m[2, 0, 0] == 0).all()


def test_gate_mask_reference_columns_open():
    ms, B = _layout()
    m = ms.gate_mask(torch.float32, "cpu")
    V = 3
    assert (m[:, 0, 0, :L] == 0).all() and (m[:, 0, 0, V * L:V * L + 4] == 0).all()
    assert (m > float("-inf")).any(-1).all()


def test_gate_mask_reference_conf_must_be_all_true():
    B = 2
    ms = MemorySet([_view(B, torch.tensor([True, False]), "ref"), _view(B)])
    with pytest.raises(AssertionError):
        ms.gate_mask(torch.float32, "cpu")


def test_n_gated_counts_false_entries():
    ms, _ = _layout()
    assert ms.n_gated() == 3


# ---------------------------------------------------------------- GPU: the gate inside decode_view (Blocks)
from tests_torch.test_wrapper import setup  # noqa: E402,F401  (module fixture: model, Blocks reference, cam0/cam9 views)

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "Blocks")), reason="needs GPU and Data/Blocks")


def _bias_wrapper(s):
    from geomem.bias import GeoBias
    from geomem.embed import GeoEmbed
    from geomem.wrapper import GeoMemStar
    return GeoMemStar(s["model"], GeoEmbed().cuda(), geo_bias=GeoBias().cuda(), use_geometry=False, use_bias=True)


def _two_views(s, g, tau=None):
    """ref + cam9's round-1 prediction (conf from tau); returns (ref set, cam9 view, B)."""
    B = s["masks"].shape[0]
    ref = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
    bb9, p9, v9 = s["views"]["cam9"]
    r1 = g.decode_view(bb9, B, ref, p9, v9)
    v = g.encode_memory_from_prediction(bb9, B, r1["logits_1008"], r1["presence"], r1["obj_ptr"], p9, v9, "cam9",
                                        ious=r1["ious"], tau=tau)
    return ref, v, B


def _with_conf(view, conf):
    from dataclasses import replace
    return replace(view, conf=conf)


@needs_gpu
def test_two_view_decode_byte_identical_with_conf_none(setup):
    s = setup
    g = _bias_wrapper(s)
    bb0, p0, v0 = s["views"]["cam0"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref, v9, B = _two_views(s, g)
        assert v9.conf is None
        out_a = g.decode_view(bb0, B, ref.with_view(v9), p0, v0)["logits_1008"]
        out_b = g.decode_view(bb0, B, ref.with_view(_with_conf(v9, torch.ones(B, dtype=torch.bool, device="cuda"))),
                              p0, v0)["logits_1008"]
    assert torch.equal(out_a, out_b)


@needs_gpu
def test_gated_object_matches_reference_only_decode(setup):
    s = setup
    g = _bias_wrapper(s)
    bb0, p0, v0 = s["views"]["cam0"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref, v9, B = _two_views(s, g)
        conf = torch.ones(B, dtype=torch.bool, device="cuda")
        conf[0] = False
        full = g.decode_view(bb0, B, ref.with_view(v9), p0, v0)["logits_1008"].float()
        gated = g.decode_view(bb0, B, ref.with_view(_with_conf(v9, conf)), p0, v0)["logits_1008"].float()
        only = g.decode_view(bb0, B, ref, p0, v0)["logits_1008"].float()
    assert torch.isfinite(gated).all()
    a, b = gated[0], only[0]
    assert ((a > 0) == (b > 0)).float().mean().item() > 0.9999
    assert ((a - b).abs().max() / b.abs().max()).item() < 0.05
    assert torch.equal(gated[1:], full[1:])


@needs_gpu
def test_conf_from_prediction_rule(setup):
    s = setup
    g = _bias_wrapper(s)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref, v9, B = _two_views(s, g, tau=0.7)
        bb9, p9, v9g = s["views"]["cam9"]
        r1 = g.decode_view(bb9, B, ref, p9, v9g)
        v = g.encode_memory_from_prediction(bb9, B, r1["logits_1008"], torch.tensor([1.0, -1.0, 1.0] + [1.0] * (B - 3), device="cuda"),
                                            r1["obj_ptr"], p9, v9g, "cam9",
                                            ious=torch.tensor([0.9, 0.9, 0.4] + [0.9] * (B - 3), device="cuda"), tau=0.7)
    assert v.conf.tolist()[:3] == [True, False, False] and v9.conf is not None


def test_subset_slices_every_object_field():
    ms, B = _layout()
    sub = ms.subset(torch.tensor([0, 2]))
    assert len(sub.views) == 3 and sub.views[0].conf is None
    assert sub.views[1].feats.shape == (L, 2, 64) and sub.views[1].obj_ptr_tokens.shape == (4, 2, 64)
    assert sub.views[1].object_score_logits.shape == (2, 1) and sub.views[1].conf.tolist() == [True, True]
    assert sub.views[2].conf.tolist() == [False, True] and torch.equal(sub.views[2].p, ms.views[2].p)


@needs_gpu
def test_decode_view_chunked_equals_full(setup):
    """Object-chunked decoding (OOM guard for gated sets with many objects) must give the same masks."""
    s = setup
    g = _bias_wrapper(s)
    bb0, p0, v0 = s["views"]["cam0"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref, v9, B = _two_views(s, g, tau=0.7)
        assert B > 8 and not bool(v9.conf.all()), "Blocks must have >8 objects with at least one gated"
        mem = ref.with_view(v9)
        full = g.decode_view(bb0, B, mem, p0, v0)
        chunked = g.decode_view(bb0, B, mem, p0, v0, chunk=8)
    # bf16 kernels differ with the batch size (~1 % of the memory-attention output, gate or not); masks must agree
    # and the heads of PRESENT objects must agree.  An absent object's IoU may swing (its logits are NO_OBJ_SCORE).
    for k in ("logits_1008", "ious", "presence", "obj_ptr"):
        assert full[k].shape == chunked[k].shape, k
    present = (full["presence"] > 0) & (chunked["presence"] > 0)
    assert present.sum() >= 8
    a, b = full["logits_1008"].float(), chunked["logits_1008"].float()
    assert (((a > 0) == (b > 0)).float().mean().item()) > 0.9999
    assert ((a - b).abs().max() / a.abs().max()).item() < 0.05
    assert (full["ious"][present] - chunked["ious"][present]).abs().max().item() < 0.02
    assert ((full["presence"] > 0) == (chunked["presence"] > 0)).all()
