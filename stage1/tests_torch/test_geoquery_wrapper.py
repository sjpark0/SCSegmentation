"""GPU tests on Blocks: GeoQuery = geobias_only path + object-query hub (GeoQuery spec 2.4)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem import sam3_io  # noqa: E402
from geomem.bias import GeoBias  # noqa: E402
from geomem.embed import GeoEmbed, grid_tensors  # noqa: E402
from geomem.pointmap import view_grid  # noqa: E402
from geomem.query_hub import QueryHub  # noqa: E402
from geomem.wrapper import GeoMemStar, GeoQuery, base_arm  # noqa: E402
from methods import common, sam3_single  # noqa: E402
from mvseg2d.io import labelmap_to_mask  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks")),
                                reason="needs GPU and Data/Blocks")
CAMS = ("cam0", "cam9")


@pytest.fixture(scope="module")
def setup():
    model = sam3_single.load_model()
    scene = common.load_scene(DATA, "Blocks")
    cache = os.path.join(DATA, "Blocks", "Cache", "pi3")
    ref_img, h, w = sam3_io.load_image(scene.image_paths[scene.ref_cam])
    ref_bb = sam3_io.backbone_features(model, ref_img)
    masks_np = np.stack([labelmap_to_mask(scene.ref_labels, o) for o in scene.objects])
    vg = view_grid(cache, scene.ref_cam, scene.ref_cam)
    p_ref, v_ref = grid_tensors(vg, "cuda")
    views, grids = {}, [(p_ref, v_ref)]
    for cam in CAMS:
        img, _, _ = sam3_io.load_image(scene.image_paths[cam])
        p, v = grid_tensors(view_grid(cache, cam, scene.ref_cam), "cuda")
        views[cam] = (sam3_io.backbone_features(model, img), p, v)
        grids.append((p, v))
    return dict(model=model, ref_img=ref_img, ref_bb=ref_bb, masks=torch.from_numpy(masks_np).cuda(), masks_np=masks_np,
                p_ref=p_ref, v_ref=v_ref, p_ref_np=np.asarray(vg["p"], np.float32).reshape(-1, 3),
                v_ref_np=np.asarray(vg["valid"], bool).reshape(-1), views=views,
                p_views=torch.stack([g[0] for g in grids]), valid_views=torch.stack([g[1] for g in grids]))


def _sign_agree(a, b):
    return ((a > 0) == (b > 0)).float().mean().item()


def _star_decode(s, cam):
    star = GeoMemStar(s["model"], GeoEmbed().cuda(), GeoBias().cuda(), use_geometry=False, use_bias=True)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = star.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
            bb, p, v = s["views"][cam]
            return star.decode_view(bb, s["masks"].shape[0], ref, p, v)
    finally:
        star.close()


def _star_ref(s):
    star = GeoMemStar(s["model"], GeoEmbed().cuda(), GeoBias().cuda(), use_geometry=False, use_bias=True)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return star.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
    finally:
        star.close()


def _query_decode(s, cam, perturb=False, use_query=True):
    gq = GeoQuery(s["model"], GeoEmbed().cuda(), GeoBias().cuda(), QueryHub().cuda(), use_geometry=False, use_bias=True)
    if perturb:                                   # only the q path; type_embed stays 0
        with torch.no_grad():
            gq.hub.proj_q.weight.normal_(0, 0.05)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = gq.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
            bbs = [s["ref_bb"]] + [s["views"][c][0] for c in CAMS]
            hub = gq.run_hub(ref, bbs, s["p_views"], s["valid_views"], s["masks_np"], s["p_ref_np"], s["v_ref_np"])
            bb, p, v = s["views"][cam]
            out = gq.decode_view(bb, s["masks"].shape[0], ref, p, v, query=hub["q"] if use_query else None)
            return out, hub
    finally:
        gq.close()


def test_geoquery_zero_gates_matches_geobias_only(setup):
    s = setup
    B = s["masks"].shape[0]
    out_q, hub = _query_decode(s, "cam0")
    out_a = _star_decode(s, "cam0")
    assert _sign_agree(out_q["logits_1008"], out_a["logits_1008"]) > 0.9999
    assert hub["mass"].shape == (2, B, 3) and torch.allclose(hub["mass"].sum(-1), torch.ones(2, B).cuda(), atol=1e-3)
    assert len(hub["cloud_sizes"]) == B and max(hub["cloud_sizes"]) <= 256 and max(hub["cloud_sizes"]) > 0
    assert out_q["logits_1008"].shape == (B, 1, 1008, 1008) and out_q["obj_ptr"].shape == (B, 256)


def test_geoquery_query_changes_output(setup):
    """With a non-zero proj_q (type_embed zero) the query must reach the decoder: output differs from the no-query
    path, and two different queries give different outputs."""
    s = setup
    out_q, hub = _query_decode(s, "cam0", perturb=True)
    out_a = _star_decode(s, "cam0")
    assert _sign_agree(out_q["logits_1008"], out_a["logits_1008"]) < 0.9999 or not torch.allclose(out_q["ious"], out_a["ious"])
    gq = GeoQuery(s["model"], GeoEmbed().cuda(), GeoBias().cuda(), QueryHub().cuda(), use_geometry=False, use_bias=True)
    try:
        with torch.no_grad():
            gq.hub.proj_q.weight.normal_(0, 0.05)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = gq.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
            bb, p, v = s["views"]["cam0"]
            B = s["masks"].shape[0]
            q1 = hub["q"]
            q2 = q1 + torch.randn_like(q1)
            o1 = gq.decode_view(bb, B, ref, p, v, query=q1)
            o2 = gq.decode_view(bb, B, ref, p, v, query=q2)
        assert not torch.equal(o1["logits_1008"], o2["logits_1008"])
    finally:
        gq.close()


def test_geoquery_decode_without_query_is_parent_path(setup):
    s = setup
    out, _ = _query_decode(s, "cam9", use_query=False)
    out_a = _star_decode(s, "cam9")
    assert torch.equal(out["logits_1008"], out_a["logits_1008"])


def test_freeze_new_arms_and_base_arm(setup):
    s = setup
    assert base_arm("geoquery_ft") == base_arm("geobias_only_ft2") == "geobias_only_ft" and base_arm("geomem_ft") == "geomem_ft"
    gq = GeoQuery(s["model"], GeoEmbed().cuda(), GeoBias().cuda(), QueryHub().cuda(), use_geometry=False, use_bias=True)
    try:
        names = gq.freeze("geoquery_ft")
        assert any(n.startswith("hub.") for n in names) and any("transformer.encoder" in n for n in names)
        assert not any(n.startswith("geo_embed") for n in names) and any(n.startswith("geo_bias") for n in names)
        gq2 = GeoQuery(s["model"], GeoEmbed().cuda(), GeoBias().cuda(), None, use_geometry=False, use_bias=True)
        assert not any(n.startswith("hub.") for n in gq2.freeze("geobias_only_ft2"))
    finally:
        gq.close()


def _ptr_decode(s, cam, perturb=False):
    gq = GeoQuery(s["model"], GeoEmbed().cuda(), GeoBias().cuda(), QueryHub(ptr=True, g_cross_init=0.1).cuda(),
                  use_geometry=False, use_bias=True, inject="ptr")
    if perturb:
        with torch.no_grad():
            gq.hub.ptr_proj.weight.normal_(0, 0.05)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = gq.encode_reference(s["ref_img"], s["ref_bb"], s["masks"], s["p_ref"], s["v_ref"])
            bbs = [s["ref_bb"]] + [s["views"][c][0] for c in CAMS]
            hub = gq.run_hub(ref, bbs, s["p_views"], s["valid_views"], s["masks_np"], s["p_ref_np"], s["v_ref_np"])
            bb, p, v = s["views"][cam]
            return gq.decode_view(bb, s["masks"].shape[0], ref, p, v, query=hub["q"]), hub
    finally:
        gq.close()


def test_geoquery_ptr_zero_init_is_byte_identical_to_geobias_only(setup):
    """Pointer injection with a zero ptr_proj adds 0 to the reference pointer tokens: same tokens, same path, same bytes
    — even though g_cross=0.1 makes the hub output differ from obj_ptr."""
    s = setup
    out_q, hub = _ptr_decode(s, "cam0")
    out_a = _star_decode(s, "cam0")
    assert torch.equal(out_q["logits_1008"], out_a["logits_1008"]) and torch.equal(out_q["ious"], out_a["ious"])
    assert not torch.equal(hub["q"].float(), GeoQuery.obj_ptr_of(_star_ref(s)).float())   # the hub really moved q


def test_geoquery_ptr_query_changes_output(setup):
    s = setup
    out_q, _ = _ptr_decode(s, "cam0", perturb=True)
    out_a = _star_decode(s, "cam0")
    assert not torch.equal(out_q["logits_1008"], out_a["logits_1008"])
