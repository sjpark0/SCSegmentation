"""GeoSet-2 §8 reprojection gate: overlap of a neighbour's predicted mask with the reference GT, on synthetic geometry (CPU)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.geomem.run import reproj_overlap  # noqa: E402

H, W = 40, 60


def _identity_geometry(depth=1.0):
    """The view IS the reference camera: every pixel reprojects onto itself (K = [1,0,0;0,1,0;0,0,1], z = depth)."""
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    pts = torch.stack([u * depth, v * depth, torch.full_like(u, depth)], dim=-1)      # x = u·z, y = v·z -> u = x/z
    K = torch.eye(3)
    return pts, torch.ones(H, W, dtype=torch.bool), torch.full((H, W), depth), K


def _box(r0, r1, c0, c1):
    m = torch.zeros(H, W, dtype=torch.bool)
    m[r0:r1, c0:c1] = True
    return m


def test_overlap_exact_superset_and_shifted():
    pts, valid, depth_ref, K = _identity_geometry()
    gt = _box(10, 20, 10, 30)                                   # 10x20 = 200 px
    pred = torch.stack([gt, _box(8, 22, 8, 32), _box(10, 20, 40, 60)])   # exact, coarse superset (14x24=336), shifted
    ov, n = reproj_overlap(pred, pts, valid, depth_ref, K, gt[None].expand(3, -1, -1))
    assert n.tolist() == [200, 336, 200]
    assert abs(ov[0].item() - 1.0) < 1e-6
    assert abs(ov[1].item() - 200 / 336) < 1e-6                 # coarse but correctly placed -> passes 0.5
    assert ov[2].item() == 0.0


def test_overlap_excludes_occluded_and_invalid_pixels():
    pts, valid, depth_ref, K = _identity_geometry()
    gt = _box(10, 20, 10, 40)                                   # 10 x 30 = 300 px
    valid = valid.clone(); valid[:, :20] = False                # left part of the view has no Pi3 point
    depth_ref = depth_ref.clone(); depth_ref[15:, :] = 0.5      # the reference sees something nearer in the lower half
    ov, n = reproj_overlap(gt[None], pts, valid, depth_ref, K, gt[None])
    assert n.item() == 5 * 20                                   # rows 10..14 x cols 20..39 survive (>= MIN_VALID)
    assert abs(ov.item() - 1.0) < 1e-6


def test_overlap_nan_below_min_valid():
    pts, valid, depth_ref, K = _identity_geometry()
    tiny = _box(10, 12, 10, 12)                                 # 4 px < MIN_VALID
    ov, n = reproj_overlap(tiny[None], pts, valid, depth_ref, K, tiny[None])
    assert n.item() == 4 and torch.isnan(ov).all()


# ---------------------------------------------------------------- GeoProj: reference GT masks projected into a neighbour view
def test_project_reference_masks_identity_and_occlusion():
    from methods.geomem.run import project_reference_masks
    pts, valid, depth_ref, K = _identity_geometry()
    gt = torch.stack([_box(10, 20, 10, 40), _box(25, 35, 5, 15)])                 # two objects
    masks, vis = project_reference_masks(gt, pts, valid, depth_ref, K)
    assert masks.shape == gt.shape and masks.dtype == torch.bool and vis.shape == (H, W)
    assert torch.equal(masks, gt) and vis.all()                                   # identity geometry: the mask projects onto itself
    depth_ref = depth_ref.clone(); depth_ref[15:, :] = 0.5                          # the reference sees something nearer below row 15
    masks2, vis2 = project_reference_masks(gt, pts, valid, depth_ref, K)
    assert not vis2[15:].any() and vis2[:15].all()
    assert torch.equal(masks2[0], gt[0] & vis2) and not masks2[1].any()           # object 2 lies entirely in the occluded rows
