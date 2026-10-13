"""Differentiable reprojection of a view's pixels into the reference camera (for the consistency loss).

Points are given per pixel of view u, already in the REFERENCE camera frame (pointmap.to_reference at
the original image resolution).  `reproject_coords` projects them with the reference intrinsics and
returns normalised grid_sample coordinates plus a validity mask (in front of the camera, inside the
image, and not hidden behind the reference surface by more than `tol`).
"""
import torch
import torch.nn.functional as F

OCC_TOL = 0.01


def reproject_coords(p_u, K_ref, hw_ref, depth_ref=None, tol=OCC_TOL):
    """p_u [h,w,3] (ref frame), K_ref 3x3 pixel-index intrinsics, hw_ref -> (grid [1,h,w,2] in [-1,1], valid [h,w] bool)."""
    h_ref, w_ref = hw_ref
    z = p_u[..., 2]
    zs = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
    u = K_ref[0, 0] * p_u[..., 0] / zs + K_ref[0, 2]
    v = K_ref[1, 1] * p_u[..., 1] / zs + K_ref[1, 2]
    valid = (z > 0) & (u >= -0.5) & (u <= w_ref - 0.5) & (v >= -0.5) & (v <= h_ref - 0.5) & torch.isfinite(u) & torch.isfinite(v)
    if depth_ref is not None:
        col = u.round().clamp(0, w_ref - 1).long()
        row = v.round().clamp(0, h_ref - 1).long()
        d = depth_ref[row, col]
        valid = valid & ((d <= 0) | (z <= d * (1.0 + tol)))
    gx = (u + 0.5) / w_ref * 2 - 1          # align_corners=False: pixel-index u sits at (u+0.5)/w
    gy = (v + 0.5) / h_ref * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    return grid, valid


def sample_reference(ref_maps, grid):
    """ref_maps [B,1,h_ref,w_ref] float -> values at the grid positions [B,1,h,w] (bilinear, zeros outside)."""
    B = ref_maps.shape[0]
    return F.grid_sample(ref_maps, grid.expand(B, -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=False)
