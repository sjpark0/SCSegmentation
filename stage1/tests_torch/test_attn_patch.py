"""GPU tests on Blocks: the cross-attention patch must be invisible with no/zero bias and effective with one."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem import sam3_io  # noqa: E402
from geomem.attn_patch import attn_mask, patch_cross_attention, unpatch_cross_attention  # noqa: E402
from geomem.embed import GeoEmbed, grid_tensors  # noqa: E402
from geomem.pointmap import view_grid  # noqa: E402
from geomem.wrapper import GeoMemStar  # noqa: E402
from methods import common, sam3_single  # noqa: E402
from mvseg2d.io import labelmap_to_mask  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks")),
                                reason="needs GPU and Data/Blocks")


@pytest.fixture(scope="module")
def s():
    model = sam3_single.load_model()
    scene = common.load_scene(DATA, "Blocks")
    cache = os.path.join(DATA, "Blocks", "Cache", "pi3")
    ref_img, h, w = sam3_io.load_image(scene.image_paths[scene.ref_cam])
    ref_bb = sam3_io.backbone_features(model, ref_img)
    masks = torch.from_numpy(np.stack([labelmap_to_mask(scene.ref_labels, o) for o in scene.objects])).cuda()
    p_ref, v_ref = grid_tensors(view_grid(cache, scene.ref_cam, scene.ref_cam), "cuda")
    img, _, _ = sam3_io.load_image(scene.image_paths["cam0"])
    bb = sam3_io.backbone_features(model, img)
    p_v, v_v = grid_tensors(view_grid(cache, "cam0", scene.ref_cam), "cuda")
    return dict(model=model, scene=scene, ref_img=ref_img, ref_bb=ref_bb, masks=masks, p_ref=p_ref, v_ref=v_ref,
                bb=bb, p_v=p_v, v_v=v_v)


def _decode(s, autocast=True):
    """Always under bf16 autocast: the tracker's own process-wide autocast is on and its cached
    backbone features are bf16, so an fp32 pass is not a configuration the model supports."""
    g = GeoMemStar(s["model"], GeoEmbed().cuda(), use_geometry=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
        return g.decode_view(s["bb"], s["masks"].shape[0], ref, s["p_v"], s["v_v"])["logits_1008"].float()


def test_patch_with_none_mask_is_byte_identical(s):
    before = _decode(s)
    patch_cross_attention(s["model"].tracker)
    patch_cross_attention(s["model"].tracker)                      # idempotent
    assert torch.equal(before, _decode(s))
    unpatch_cross_attention(s["model"].tracker)


def test_zero_mask_is_byte_identical(s):
    before = _decode(s)
    tr = s["model"].tracker
    patch_cross_attention(tr)
    Lk = 5184 + 4
    with attn_mask(tr, torch.zeros(1, 1, 5184, Lk, device="cuda")):
        after = _decode(s)
    unpatch_cross_attention(tr)
    # A present attn_mask makes SDPA pick a non-flash kernel, so bf16 results differ in the last bits,
    # amplified through 4 attention layers and the decoder (observed max |Δ| 0.68 on logits of magnitude
    # up to ~20, i.e. < 5 % relative); the decision (sign) must be unchanged almost everywhere.
    assert ((before > 0) == (after > 0)).float().mean().item() > 0.9999
    assert (before - after).abs().max().item() / before.abs().max().item() < 0.05


def test_patched_attention_runs_under_autocast_with_bias(s):
    tr = s["model"].tracker
    patch_cross_attention(tr)
    before = _decode(s, autocast=True)
    torch.manual_seed(0)
    mask = torch.randn(1, 1, 5184, 5184 + 4, device="cuda") * 2.0
    mask[..., -4:] = 0
    with attn_mask(tr, mask):
        after = _decode(s, autocast=True)
    unpatch_cross_attention(tr)
    assert torch.isfinite(after).all() and not torch.equal(before, after)


def test_no_rope_changes_output(s):
    tr = s["model"].tracker
    patch_cross_attention(tr)
    with_rope = _decode(s)
    unpatch_cross_attention(tr)
    patch_cross_attention(tr, no_rope=True)
    without = _decode(s)
    unpatch_cross_attention(tr)
    assert torch.isfinite(without).all() and not torch.equal(with_rope, without)


def test_unpatch_restores(s):
    tr = s["model"].tracker
    before = _decode(s)
    patch_cross_attention(tr, no_rope=True)
    unpatch_cross_attention(tr)
    assert torch.equal(before, _decode(s))
    assert not getattr(tr, "_geo_patched", False)
