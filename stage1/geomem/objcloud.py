"""Object point clouds from the reference mask and token-to-cloud distances (GeoQuery spec 2.2).

`mask_coverage` and `object_clouds` are numpy/OpenCV (host-testable); `cloud_distances` imports torch
lazily.  Cell order is row-major over the 72x72 grid, like `geomem.embed.grid_tensors`.
"""
import cv2
import numpy as np

GRID = 72
MAX_POINTS = 256
COVERAGE_MIN = 0.5


def mask_coverage(masks_bool, grid=GRID):
    """[B,h,w] bool -> [B,grid,grid] float32 fraction of each cell covered (area resampling)."""
    out = np.empty((masks_bool.shape[0], grid, grid), np.float32)
    for i, m in enumerate(masks_bool):
        out[i] = cv2.resize(m.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    return out


def object_clouds(masks_bool, p_ref, valid_ref, max_points=MAX_POINTS, seed=0):
    """Per object: the 3D points of reference cells with coverage >= COVERAGE_MIN and valid geometry,
    capped at `max_points` by a seeded choice without replacement.  An object may get 0 points."""
    cov = mask_coverage(masks_bool, int(np.sqrt(p_ref.shape[0]))).reshape(masks_bool.shape[0], -1)
    rng = np.random.default_rng(seed)
    clouds = []
    for c in cov:
        idx = np.flatnonzero((c >= COVERAGE_MIN) & valid_ref)
        if idx.size > max_points:
            idx = idx[np.sort(rng.choice(idx.size, size=max_points, replace=False))]
        clouds.append(np.asarray(p_ref[idx], np.float32).reshape(-1, 3))
    return clouds


def cloud_distances(p_views, valid_views, clouds):
    """p_views [N,5184,3], valid_views [N,5184] bool (torch), clouds list of [n_m,3] numpy ->
    [M, N*5184] float32 min distance from each token's point to the object's cloud; inf where the token is
    invalid or the cloud is empty."""
    import torch
    pts = p_views.reshape(-1, 3).float()
    valid = valid_views.reshape(-1)
    out = torch.full((len(clouds), pts.shape[0]), float("inf"), dtype=torch.float32, device=pts.device)
    for m, cloud in enumerate(clouds):
        if len(cloud) == 0:
            continue
        c = torch.as_tensor(np.asarray(cloud, np.float32), device=pts.device)
        d = torch.cdist(pts, c).min(dim=1).values
        out[m] = torch.where(valid, d, out[m])
    return out
