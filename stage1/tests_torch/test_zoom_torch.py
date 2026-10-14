"""GPU (scsam3): S1-E10 crop-zoom pass -- full-window identities (prereg Z6), the zoom=None path byte-identical to
the adopted output (Z4 at Blocks scale), and a zoomed decode of Blocks whose changes stay inside the band (Z5).
Needs Data/Blocks, the geoset_pred_ft checkpoints (geomem/runs) and Data/Blocks/Result/geoset_pred_ft_rep_e8."""
import argparse
import hashlib
import os
import shutil
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods import common  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data")
RUNS = os.path.join(REPO, "geomem", "runs")
E8 = os.path.join(DATA, "Blocks", "Result", "geoset_pred_ft_rep_e8")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks")) or not os.path.isdir(E8)
    or not os.path.isdir(os.path.join(RUNS, "geoset_pred_ft")),
    reason="needs GPU, Data/Blocks, geoset_pred_ft checkpoints and the _e8 result folder")


@pytest.fixture(scope="module")
def model():
    from methods import sam3_single
    from methods.geomem import run as R
    if R.decode_scene.model is None:
        R.decode_scene.model = sam3_single.load_model()
        R.decode_scene.init_state = R.snapshot_init(R.decode_scene.model)
    yield R.decode_scene.model
    R.decode_scene.model = None
    R.decode_scene.init_state = None
    torch.cuda.empty_cache()


def _args(out, **kw):
    base = dict(data=DATA, out=out, r1_out=out + "_r1", arm="geoset_pred_ft", gate="reproj", untrained=False, runs=RUNS,
                tau=0.7, k=3, round2="pred")
    base.update(kw)
    return argparse.Namespace(**base)


def _pngs(d, skip=()):
    out = {}
    for cam in sorted(os.listdir(d)):
        if cam in skip or not os.path.isdir(os.path.join(d, cam)):
            continue
        for f in sorted(os.listdir(os.path.join(d, cam))):
            if f.endswith(".png"):
                out[(cam, f)] = open(os.path.join(d, cam, f), "rb").read()
    return out


def _rm(*names):
    for n in names:
        shutil.rmtree(os.path.join(DATA, "Blocks", "Result", n), ignore_errors=True)


def test_crop_loader_full_image_equals_load_image():
    from geomem import sam3_io
    scene = common.load_scene(DATA, "Blocks")
    p = scene.image_paths[scene.ref_cam]
    a, h, w = sam3_io.load_image(p)
    b = sam3_io.load_image_array(cv2.imread(p))
    assert a.shape == b.shape == (1, 3, 1008, 1008) and torch.equal(a, b)
    c = sam3_io.load_image_array(np.ascontiguousarray(cv2.imread(p)[h // 4: h // 4 + h // 2, w // 4: w // 4 + w // 2]))
    assert c.shape == (1, 3, 1008, 1008) and not torch.equal(a, c)


def test_remapped_rope_table_full_window_equals_the_layers_table(model):
    from geomem import attn_patch, zoom
    scene = common.load_scene(DATA, "Blocks")
    tr = model.tracker
    t_x, t_y = zoom.cell_positions((0, scene.hw[0], 0, scene.hw[1]), scene.hw)
    table = attn_patch.axial_cis_at(tr, t_x, t_y)
    for layer in tr.transformer.encoder.layers:
        attn = layer.cross_attn_image
        ref = attn.compute_cis(end_x=72, end_y=72, device=table.device)
        assert table.shape == ref.shape and torch.equal(table, ref)


def test_window_grid_full_window_equals_view_grid_on_blocks():
    from geomem import pointmap
    from geomem.embed import grid_tensors
    from mvseg2d.io import cache_dir
    scene = common.load_scene(DATA, "Blocks")
    cache = cache_dir(DATA, "Blocks", "pi3")
    cam = [c for c in scene.cams if c != scene.ref_cam][0]
    vg = pointmap.view_grid(cache, cam, scene.ref_cam)
    wg = pointmap.window_grid(pointmap.ref_frame_points(cache, cam, scene.ref_cam), (0, scene.hw[0], 0, scene.hw[1]), scene.hw)
    p1, v1 = grid_tensors(vg, "cuda"); p2, v2 = grid_tensors(wg, "cuda")
    assert torch.equal(p1, p2) and torch.equal(v1, v2)


def test_decode_view_with_full_window_rope_table_is_identical(model):
    """The remap path with the integer lattice reproduces the unchanged path bit for bit (same ops, same table)."""
    from geomem import attn_patch, sam3_io, zoom
    from geomem.embed import grid_tensors
    from geomem.pointmap import view_grid
    from geomem.train import build_wrapper
    from geomem.folds import load_folds
    from methods.geomem import run as R
    from mvseg2d.io import cache_dir, labelmap_to_mask
    scene = common.load_scene(DATA, "Blocks")
    cache = cache_dir(DATA, "Blocks", "pi3")
    wrapper = build_wrapper(model, "geoset_pred_ft")
    ck = None
    try:
        ck = R.load_checkpoint_into(wrapper, wrapper.geo_embed, R.checkpoint_for("Blocks", "geoset_pred_ft", RUNS, load_folds()))
        wrapper.eval()
        masks = torch.from_numpy(np.stack([labelmap_to_mask(scene.ref_labels, o) for o in scene.objects])).cuda()
        cam = [c for c in scene.cams if c != scene.ref_cam][0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref_img, _, _ = sam3_io.load_image(scene.image_paths[scene.ref_cam])
            ref_bb = sam3_io.backbone_features(model, ref_img)
            p_ref, v_ref = grid_tensors(view_grid(cache, scene.ref_cam, scene.ref_cam), "cuda")
            ref = wrapper.encode_reference(ref_img, ref_bb, masks, p_ref, v_ref)
            img, _, _ = sam3_io.load_image(scene.image_paths[cam])
            bb = sam3_io.backbone_features(model, img)
            p_v, v_v = grid_tensors(view_grid(cache, cam, scene.ref_cam), "cuda")
            mem = ref.subset(torch.tensor([0], device="cuda"))
            a = wrapper.decode_view(bb, 1, mem, p_v, v_v)
            t_x, t_y = zoom.cell_positions((0, scene.hw[0], 0, scene.hw[1]), scene.hw)
            table = attn_patch.axial_cis_at(model.tracker, t_x, t_y)
            b = wrapper.decode_view(bb, 1, mem, p_v, v_v, rope_q=table)
            c = wrapper.decode_view(bb, 1, mem, p_v, v_v)
        assert torch.equal(a["logits_1008"], c["logits_1008"]), "the plain path itself is not deterministic here"
        for k in ("logits_1008", "ious", "presence"):
            assert torch.equal(a[k], b[k]), k
        assert model.tracker._geo_rope_q is None
        # the remap branch is live: a table for a genuine 2x window (half-spaced, offset positions) changes the output
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_x2, t_y2 = zoom.cell_positions(zoom.window_for((scene.hw[0] // 2 - 20, scene.hw[0] // 2 + 20,
                                                              scene.hw[1] // 2 - 20, scene.hw[1] // 2 + 20), scene.hw), scene.hw)
            e = wrapper.decode_view(bb, 1, mem, p_v, v_v, rope_q=attn_patch.axial_cis_at(model.tracker, t_x2, t_y2))
        assert not torch.equal(a["logits_1008"], e["logits_1008"]), "a different query RoPE table left the output unchanged (dead branch?)"
        assert model.tracker._geo_rope_q is None
    finally:
        if ck is not None:
            R.restore_init(model, R.decode_scene.init_state, ck)
        wrapper.close()


def test_zoom_none_is_byte_identical_to_the_adopted_output(model):
    from methods.geomem import run as R
    scene = common.load_scene(DATA, "Blocks")
    _rm("dbg_z0", "dbg_z0_r1")
    cfg = R.decode_scene(scene, _args("dbg_z0"))
    assert "zoom" not in cfg
    a = _pngs(os.path.join(DATA, "Blocks", "Result", "dbg_z0"), skip=(scene.ref_cam,))
    b = _pngs(E8, skip=(scene.ref_cam,))
    assert a and set(a) == set(b) and all(a[k] == b[k] for k in a)


@pytest.mark.parametrize("rope", ["grid", "remap"])
def test_zoomed_decode_changes_only_the_band(model, rope):
    from geomem import zoom
    from methods.geomem import run as R
    scene = common.load_scene(DATA, "Blocks")
    out = f"dbg_z2_{rope}"
    _rm(out, out + "_r1")
    cfg = R.decode_scene(scene, _args(out, zoom=2, zoom_rope=rope))
    r = zoom.band_px(scene.hw)
    assert cfg["zoom"] == 2 and cfg["zoom_rope"] == rope and cfg["band_px"] == r
    n = {k: cfg[f"n_zoom_{k}"] for k in ("candidates", "run", "skipped_large", "kept_absent", "kept_iou", "kept_empty", "accepted", "changed")}
    assert n["candidates"] == cfg["masks_written"]       # a refined mask is never emptied, an empty one never created
    assert n["run"] + n["skipped_large"] == n["candidates"]
    assert n["accepted"] + n["kept_absent"] + n["kept_iou"] + n["kept_empty"] == n["run"]
    assert n["run"] > 0, "Blocks should have at least one crop"
    print(f"\n[{rope}] crops {n['run']} skipped {n['skipped_large']} accepted {n['accepted']} changed {n['changed']} "
          f"+{cfg['zoom_px_added']} -{cfg['zoom_px_removed']} px  {cfg['zoom_seconds'] / max(n['run'], 1):.3f} s/crop")
    d = os.path.join(DATA, "Blocks", "Result", out)
    recs = __import__("json").load(open(os.path.join(d, "zoom_conf.json")))
    assert sum(len(v) for v in recs.values()) == n["candidates"]
    # round 1 untouched: r1 folder and round1_conf.json byte-identical to the adopted run's
    assert _pngs(d + "_r1", skip=(scene.ref_cam,)) == _pngs(os.path.join(DATA, "Blocks", "Result", "geoset_pred_ft_r1_e8"), skip=(scene.ref_cam,))
    assert open(os.path.join(d, "round1_conf.json"), "rb").read() == open(os.path.join(E8, "round1_conf.json"), "rb").read()
    # Z5: same PNG set as the adopted output, every differing pixel inside band_r(adopted mask)
    a = _pngs(d, skip=(scene.ref_cam,)); b = _pngs(E8, skip=(scene.ref_cam,))
    assert set(a) == set(b)
    changed = 0
    for k in a:
        if a[k] == b[k]:
            continue
        changed += 1
        ma = cv2.imread(os.path.join(d, *k), 0) > 127
        mb = cv2.imread(os.path.join(E8, *k), 0) > 127
        diff = ma != mb
        assert diff[~zoom.band_mask(mb, r)].sum() == 0, k
    assert changed == n["changed"]
