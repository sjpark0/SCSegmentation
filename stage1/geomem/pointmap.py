"""Pi3 pointmap cache -> per-view 3D positions on the tracker's 72x72 feature grid (spec 3.2).

Conventions: depth and K are at the Pi3 work resolution with pixel-INDEX principal point (the
cache written by methods/pi3_reproj/geometry.py); c2w are camera-to-world.  Every view of a scene is
expressed in the REFERENCE camera's frame and divided by one scene scale (median distance of the
reference view's valid points from its origin), so all views share one coordinate system.  The
72x72 pooling stretches the full image to the grid, as SAM 3's 1008x1008 resize does.
"""
import json
import os

import cv2
import numpy as np

CONF_MIN = 0.1
VALID_RATIO = 0.25
GRID = 72


def local_points(depth, K):
    """[h,w] depth + 3x3 K (pixel-index coords) -> [h,w,3] camera-frame points; zero where depth == 0."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    z = depth.astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    p = np.stack([x, y, z], axis=-1)
    p[depth <= 0] = 0.0
    return p


def to_reference(points_local_v, c2w_v, c2w_ref):
    """Camera-frame points of view v -> the reference camera's frame: inv(c2w_ref) @ c2w_v @ p."""
    T = np.linalg.inv(np.asarray(c2w_ref, np.float64)) @ np.asarray(c2w_v, np.float64)
    p = np.asarray(points_local_v, np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


def scene_scale(points_ref, valid):
    d = np.linalg.norm(points_ref[valid], axis=-1)
    return float(max(np.median(d), 1e-6)) if d.size else 1.0


def pool_grid(points, valid, grid=GRID):
    """Area-average of the points over valid pixels on a grid x grid lattice (full image stretched).

    Returns (p [grid,grid,3] float32, cell_valid [grid,grid] bool); invalid cells are zero."""
    vf = valid.astype(np.float32)
    num = cv2.resize(points.astype(np.float32) * vf[..., None], (grid, grid), interpolation=cv2.INTER_AREA)
    den = cv2.resize(vf, (grid, grid), interpolation=cv2.INTER_AREA)
    cell_valid = den >= VALID_RATIO
    p = np.zeros((grid, grid, 3), np.float32)
    p[cell_valid] = num[cell_valid] / den[cell_valid][:, None]
    return p, cell_valid


def _load(cache_dir, cam):
    z = np.load(os.path.join(cache_dir, f"{cam}.npz"))
    return z["depth"], z["conf"], z["K"], z["c2w"]


def ref_frame_points(cache_dir, cam, ref_cam):
    """One view's full work-resolution point field in the reference frame, scale-normalised:
    (p [h_w,w_w,3] float64, valid [h_w,w_w] bool, scale) -- exactly what `view_grid` pools."""
    d_ref, c_ref, K_ref, c2w_ref = _load(cache_dir, ref_cam)
    valid_ref = (c_ref > CONF_MIN) & (d_ref > 0)
    p_ref = to_reference(local_points(d_ref, K_ref), c2w_ref, c2w_ref)
    scale = scene_scale(p_ref, valid_ref)
    if cam == ref_cam:
        p, valid = p_ref, valid_ref
    else:
        d, c, K, c2w = _load(cache_dir, cam)
        valid = (c > CONF_MIN) & (d > 0)
        p = to_reference(local_points(d, K), c2w, c2w_ref)
    return p / scale, valid, scale


def view_grid(cache_dir, cam, ref_cam, grid=GRID):
    """One view's pooled 3D grid in the reference frame, scale-normalised: dict(p, valid, scale)."""
    p, valid, scale = ref_frame_points(cache_dir, cam, ref_cam)
    g, cell_valid = pool_grid(p, valid, grid)
    return {"p": g.astype(np.float32), "valid": cell_valid, "scale": scale}


def window_to_work(window, orig_hw, work_hw):
    """A window (y0, y1, x0, x1), exclusive, in original pixels -> the same window in the work resolution (rounded)."""
    y0, y1, x0, x1 = window
    H, W = orig_hw
    h, w = work_hw
    wy0, wy1 = int(round(y0 * h / H)), int(round(y1 * h / H))
    wx0, wx1 = int(round(x0 * w / W)), int(round(x1 * w / W))
    assert wy1 > wy0 and wx1 > wx0, (window, orig_hw, work_hw)
    return wy0, wy1, wx0, wx1


def window_grid(src, window, orig_hw, grid=GRID):
    """S1-E10: the pooled grid of a window of the view (src = ref_frame_points(...)): the window's work-resolution
    slice of the same point field, pooled to grid x grid.  The full-image window reproduces `view_grid` exactly."""
    p, valid, scale = src
    wy0, wy1, wx0, wx1 = window_to_work(window, orig_hw, p.shape[:2])
    g, cell_valid = pool_grid(p[wy0:wy1, wx0:wx1], valid[wy0:wy1, wx0:wx1], grid)
    return {"p": g.astype(np.float32), "valid": cell_valid, "scale": scale}


def manifest(cache_dir):
    with open(os.path.join(cache_dir, "MANIFEST.json")) as f:
        return json.load(f)
