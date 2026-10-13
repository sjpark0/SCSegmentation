"""GeoFuse spec 2.1: align other views' 72x72 backbone features to a target view's grid.

A target cell carries a 3D point in the REFERENCE frame (pointmap.view_grid, divided by the scene
scale).  Multiplying the scale back gives Pi3's metric reference frame; `inv(c2w_v) @ c2w_ref` moves it
into view v's camera, K_v (pixel-index intrinsics at the Pi3 work resolution) projects it, and the
view's 72x72 feature map is sampled bilinearly at the normalised coordinate — the grid is the whole
image stretched (pointmap.pool_grid), so the normalised work-resolution coordinate is the grid coordinate.
A sample is valid when the target cell is valid, the point is in front of the camera and inside the
image, and it is not occluded (z <= depth_v(u, v) * (1 + OCC_TOL); same rule as warp_torch).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from geomem.pointmap import view_grid
from geomem.warp_torch import OCC_TOL

TWO_SIDED_TOL = 0.05   # GeoAlign spec 2.1: |z / depth_v - 1| <= tol -> the point lies ON the view's surface


def project_to_view(P, c2w_ref, c2w_v, K_v):
    """P [M,3] metric points in the reference frame -> (u [M], v [M], z [M]) pixel-index coords in view v."""
    T = torch.linalg.inv(c2w_v.double()) @ c2w_ref.double()
    X = P.double() @ T[:3, :3].T + T[:3, 3]
    z = X[:, 2]
    zs = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
    u = K_v[0, 0].double() * X[:, 0] / zs + K_v[0, 2].double()
    v = K_v[1, 1].double() * X[:, 1] / zs + K_v[1, 2].double()
    return u.float(), v.float(), z.float()


def align_views(p_tgt, valid_tgt, scale, c2w_ref, views, tol=OCC_TOL, two_sided=False):
    """p_tgt [L,3] (reference frame / scale), valid_tgt [L] bool, views: list of dict(feat [C,72,72] (any float dtype),
    depth [h,w], K [3,3], c2w [4,4]) -> (W [N,L,C] float32 sampled features, valid [N,L] bool).
    `two_sided` (GeoAlign spec 2.1) also rejects points floating in FRONT of the view's surface: |z/depth - 1| <= TWO_SIDED_TOL."""
    device = p_tgt.device
    P = p_tgt.float() * float(scale)
    Ws, Vs = [], []
    for view in views:
        depth = view["depth"]
        h, w = depth.shape
        u, v, z = project_to_view(P, c2w_ref.to(device), view["c2w"].to(device), view["K"].to(device))
        inb = (z > 0) & (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5) & torch.isfinite(u) & torch.isfinite(v)
        col = u.round().clamp(0, w - 1).long()
        row = v.round().clamp(0, h - 1).long()
        d = depth.to(device)[row, col]
        valid = valid_tgt.to(device) & inb & (d > 0) & (z <= d * (1.0 + tol))
        if two_sided:
            valid = valid & ((z / d.clamp(min=1e-9) - 1.0).abs() <= TWO_SIDED_TOL)
        gx = (u + 0.5) / w * 2 - 1
        gy = (v + 0.5) / h * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
        feat = view["feat"].to(device).float()[None]                                      # [1,C,72,72]
        sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, 0]  # [C,L]
        sampled = sampled.T * valid[:, None]
        Ws.append(sampled)
        Vs.append(valid)
    return torch.stack(Ws), torch.stack(Vs)


def fill_aligned(W_v, valid_v, F_ref):
    """GeoAlign spec 2.1: aligned features where valid, the reference's own features elsewhere; dtype of F_ref."""
    return torch.where(valid_v[:, None].to(F_ref.device), W_v.to(F_ref.device, F_ref.dtype), F_ref)


def fusion_views(views, cams, exclude=None):
    """The view list for fusion without camera `exclude` (GeoFuse2: a target never fuses its own aligned features)."""
    return [v for v, c in zip(views, cams) if c != exclude]


def alignment_depth_ratio(p_tgt, scale, c2w_ref, views):
    """z / depth_v(u, v) of every target point in every view, [N,L] (inf where the view has no depth).  1 means the point
    lies ON the view's surface; < 1 means it floats in FRONT of it (the one-sided validity rule of `align_views` accepts
    this although the sampled feature then belongs to a farther surface); > 1 means occluded.  Review follow-up: the
    two-sided statistic |ratio - 1| <= tol over valid tokens is the alignment-quality measure, not the valid fraction."""
    device = p_tgt.device
    P = p_tgt.float() * float(scale)
    out = []
    for view in views:
        depth = view["depth"].to(device)
        h, w = depth.shape
        u, v, z = project_to_view(P, c2w_ref.to(device), view["c2w"].to(device), view["K"].to(device))
        col = u.round().clamp(0, w - 1).long()
        row = v.round().clamp(0, h - 1).long()
        d = depth[row, col]
        out.append(torch.where(d > 0, z / d.clamp(min=1e-9), torch.full_like(z, float("inf"))))
    return torch.stack(out)


def view_geometry(cache_dir, cam, device):
    """Pi3 geometry of one view from <cache_dir>/<cam>.npz -> dict(depth [h,w], K [3,3], c2w [4,4]) float32 tensors."""
    path = os.path.join(cache_dir, f"{cam}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no Pi3 geometry for view {cam}: {path}")
    z = np.load(path)
    return {k: torch.from_numpy(np.asarray(z[k], np.float32)).to(device) for k in ("depth", "K", "c2w")}


def reference_frame(cache_dir, ref_cam, device="cpu"):
    """(scene scale, c2w_ref) of the reference camera — the frame every view grid is expressed in."""
    scale = float(view_grid(cache_dir, ref_cam, ref_cam)["scale"])
    return scale, view_geometry(cache_dir, ref_cam, device)["c2w"]
