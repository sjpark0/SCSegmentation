"""GPU tests on Blocks: the star wrapper over the real SAM 3 tracker (spec 3.1-3.5)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem import sam3_io  # noqa: E402
from geomem.embed import GeoEmbed, grid_tensors  # noqa: E402
from geomem.pointmap import view_grid  # noqa: E402
from geomem.wrapper import GeoMemStar  # noqa: E402
from methods import common, sam3_single  # noqa: E402
from mvseg2d.io import labelmap_to_mask  # noqa: E402
from mvseg2d.metrics import db_eval_iou  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks")),
                                reason="needs GPU and Data/Blocks")


@pytest.fixture(scope="module")
def setup():
    model = sam3_single.load_model()
    scene = common.load_scene(DATA, "Blocks")
    cache = os.path.join(DATA, "Blocks", "Cache", "pi3")
    ref_img, h, w = sam3_io.load_image(scene.image_paths[scene.ref_cam])
    ref_bb = sam3_io.backbone_features(model, ref_img)
    masks = torch.from_numpy(np.stack([labelmap_to_mask(scene.ref_labels, o) for o in scene.objects])).cuda()
    p_ref, v_ref = grid_tensors(view_grid(cache, scene.ref_cam, scene.ref_cam), "cuda")
    views = {}
    for cam in ("cam0", "cam9"):
        img, _, _ = sam3_io.load_image(scene.image_paths[cam])
        views[cam] = (sam3_io.backbone_features(model, img), *grid_tensors(view_grid(cache, cam, scene.ref_cam), "cuda"))
    return dict(model=model, scene=scene, ref_img=ref_img, ref_bb=ref_bb, masks=masks, p_ref=p_ref, v_ref=v_ref,
                views=views, hw=(h, w))


def _decode(s, use_geometry):
    g = GeoMemStar(s["model"], GeoEmbed().cuda(), use_geometry=use_geometry)
    with torch.no_grad():
        ref = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
        out = {}
        for cam, (bb, p, v) in s["views"].items():
            out[cam] = g.decode_view(bb, s["masks"].shape[0], ref, p, v)
    return out


def test_star_untrained_equals_no_geometry(setup):
    a = _decode(setup, True)
    b = _decode(setup, False)
    for cam in a:
        assert torch.equal(a[cam]["logits_1008"], b[cam]["logits_1008"])


def test_decode_all_objects_batch(setup):
    out = _decode(setup, True)["cam0"]
    B = setup["masks"].shape[0]
    assert B == 16
    assert out["logits_1008"].shape == (B, 1, 1008, 1008) and out["ious"].shape == (B,) and out["presence"].shape == (B,)


def test_star_blocks_quality_smoke(setup):
    out = _decode(setup, True)
    js = []
    for cam in ("cam0", "cam9"):
        gt = common.load_scene(DATA, "Blocks")  # reference labels only; load the target labels directly
        lab = __import__("cv2").imread(os.path.join(DATA, "Blocks", "masks", f"{cam}.png"), 0)
        pred = sam3_io.logits_to_original(out[cam]["logits_1008"], setup["hw"]) > 0
        for i, o in enumerate(setup["scene"].objects):
            g = lab == o
            if g.any():
                js.append(db_eval_iou(pred[i, 0].cpu().numpy(), g))
    mean_j = float(np.mean(js))
    print(f"\nstar untrained Blocks cam0/cam9 mean J = {mean_j:.3f} over {len(js)} pairs")
    assert mean_j >= 0.5


@pytest.mark.skipif(not os.path.isfile(os.path.join(DATA, "..", "geomem", "runs", "geomem_ft", "fold0", "ckpt.pt")),
                    reason="needs a trained geomem_ft fold0 checkpoint")
def test_ft_checkpoint_load_order_does_not_matter():
    """Regression: with the SAM 3 tracker's process-wide autocast context, autocast cached the bf16 casts of
    the attention weights made at the first forward, so loading a checkpoint before vs after that forward gave
    different (and, before, collapsed) results.  With the cache disabled both orders must give the same logits
    and a non-empty prediction (fresh subprocess so no earlier forward exists)."""
    import subprocess
    code = r'''
import os, sys, copy, torch, numpy as np
sys.path.insert(0, os.getcwd())
from methods import common, sam3_single
from methods.geomem.run import load_checkpoint_into
from geomem.embed import GeoEmbed, grid_tensors
from geomem.pointmap import view_grid
from geomem.wrapper import GeoMemStar
from geomem import sam3_io
from mvseg2d.io import labelmap_to_mask
order = sys.argv[1]
model = sam3_single.load_model(); geo = GeoEmbed().cuda()
w = GeoMemStar(model, geo, use_geometry=True)
scene = common.load_scene("Data", "Blocks"); cache = os.path.join("Data", "Blocks", "Cache", "pi3")
masks = torch.from_numpy(np.stack([labelmap_to_mask(scene.ref_labels, o) for o in scene.objects])).cuda()
def decode():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        img, h, wd = sam3_io.load_image(scene.image_paths[scene.ref_cam]); bb = sam3_io.backbone_features(model, img)
        p, v = grid_tensors(view_grid(cache, scene.ref_cam, scene.ref_cam), "cuda")
        ref = w.encode_reference(img, bb, masks, p, v)
        img2, _, _ = sam3_io.load_image(scene.image_paths["cam0"]); bb2 = sam3_io.backbone_features(model, img2)
        p2, v2 = grid_tensors(view_grid(cache, "cam0", scene.ref_cam), "cuda")
        return w.decode_view(bb2, masks.shape[0], ref, p2, v2)["logits_1008"].float()
if order == "after":
    decode()                                                            # a forward with the original weights first
load_checkpoint_into(w, geo, "geomem/runs/geomem_ft/fold0/ckpt.pt")
out = decode()
print("PX", (out > 0).float().mean().item(), "SUM", out.abs().sum().item())
'''
    res = {}
    for order in ("before", "after"):
        r = subprocess.run([sys.executable, "-c", code, order], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        line = [l for l in r.stdout.splitlines() if l.startswith("PX")]
        assert line, r.stdout[-500:] + r.stderr[-2000:]
        _, px, _, tot = line[0].split()
        res[order] = (float(px), float(tot))
    assert res["before"][0] > 0.01 and res["after"][0] > 0.01, res
    assert abs(res["before"][1] - res["after"][1]) / max(res["after"][1], 1) < 1e-3, res


# ---------------------------------------------------------------- GeoMem-2: MemorySet
def _setup2(setup):
    """Round-1 prediction of cam9 as a second memory view, plus its grid."""
    from geomem.wrapper import MemorySet  # noqa: F401
    s = setup
    cache = os.path.join(DATA, "Blocks", "Cache", "pi3")
    bb9, p9, v9 = s["views"]["cam9"]
    return cache, bb9, p9, v9


def test_memoryset_single_view_equals_refmemory_path(setup):
    """The untrained single-view path must still reproduce the committed star result on disk."""
    import cv2
    s = setup
    g = GeoMemStar(s["model"], GeoEmbed().cuda(), use_geometry=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        mem = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
        assert len(mem.views) == 1 and mem.prompt()[2] == 4
        bb, p, v = s["views"]["cam0"]
        out = g.decode_view(bb, s["masks"].shape[0], mem, p, v)
    pred = (sam3_io.logits_to_original(out["logits_1008"], s["hw"]) > 0)[:, 0].cpu().numpy()
    star_png = os.path.join(DATA, "Blocks", "Result", "star", "cam0", "1.png")
    if not os.path.isfile(star_png):
        pytest.skip("no star result on disk")
    star = cv2.imread(star_png, 0) > 127
    i = s["scene"].objects.index(1)
    assert (pred[i] == star).mean() > 0.9999
    assert "obj_ptr" in out and out["obj_ptr"].shape == (s["masks"].shape[0], 256)


def test_memoryset_two_views_shapes(setup):
    s = setup
    B = s["masks"].shape[0]
    g = GeoMemStar(s["model"], GeoEmbed().cuda(), use_geometry=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        mem = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
        bb9, p9, v9 = s["views"]["cam9"]
        r1 = g.decode_view(bb9, B, mem, p9, v9)
        view9 = g.encode_memory_from_prediction(bb9, B, r1["logits_1008"], r1["presence"], r1["obj_ptr"], p9, v9, "cam9")
        mem2 = mem.with_view(view9)
        prompt, prompt_pos, n_ptr = mem2.prompt()
        assert prompt.shape[0] == 2 * 5184 + 8 and prompt_pos.shape == prompt.shape and n_ptr == 8
        # layout the bias and `num_k_exclude_rope` rely on: view blocks in order, then ALL pointer tokens last
        assert torch.equal(prompt[:5184], mem.views[0].feats)
        assert torch.equal(prompt[5184:2 * 5184], view9.feats)
        assert torch.equal(prompt[-8:], torch.cat([mem.views[0].obj_ptr_tokens, view9.obj_ptr_tokens], 0))
        assert torch.equal(prompt_pos[-8:], torch.cat([mem.views[0].obj_ptr_pos, view9.obj_ptr_pos], 0))
        assert len(mem2.grids()) == 2
        bb0, p0, v0 = s["views"]["cam0"]
        out = g.decode_view(bb0, B, mem2, p0, v0)
    assert out["logits_1008"].shape == (B, 1, 1008, 1008) and torch.isfinite(out["logits_1008"].float()).all()


def test_bias_zero_two_views_equals_no_bias(setup):
    from geomem.bias import GeoBias
    s = setup
    B = s["masks"].shape[0]
    outs = {}
    for use_bias in (False, True):
        g = GeoMemStar(s["model"], GeoEmbed().cuda(), geo_bias=GeoBias().cuda(), use_geometry=True, use_bias=use_bias)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mem = g.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
            bb9, p9, v9 = s["views"]["cam9"]
            r1 = g.decode_view(bb9, B, mem, p9, v9)
            mem2 = mem.with_view(g.encode_memory_from_prediction(bb9, B, r1["logits_1008"], r1["presence"], r1["obj_ptr"], p9, v9, "cam9"))
            bb0, p0, v0 = s["views"]["cam0"]
            outs[use_bias] = g.decode_view(bb0, B, mem2, p0, v0)["logits_1008"].float()
        g.close()
    # kernel dispatch differs once an attn_mask is present (see tests_torch/test_attn_patch.py)
    assert ((outs[True] > 0) == (outs[False] > 0)).float().mean().item() > 0.9999
    assert (outs[True] - outs[False]).abs().max().item() / outs[False].abs().max().item() < 0.05
