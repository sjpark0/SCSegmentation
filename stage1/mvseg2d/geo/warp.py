#!/usr/bin/env python3
"""Mask warping between two COLMAP-registered views through dense depth (S1-E4a).

Pure numpy; no COLMAP binary, no torch.  Two directions, both with an occlusion test:

  backward  for every TARGET pixel with a valid depth, unproject it, project it into the reference
            view and read the reference mask there.  Produces a hole-free mask wherever the target
            depth exists, and is the natural "which target pixels are the object" question.
  forward   for every REFERENCE pixel inside the mask with a valid depth, unproject and splat it into
            the target.  Fills target regions whose depth is missing, at the price of splat holes
            (closed with a fixed morphological kernel).

A sample is rejected when the point it carries is hidden in the other view, i.e. its depth there
exceeds that view's own depth by more than `occ_tol` (relative).  The registered seed of S1-E4a is
the union of the two (docs/stage1-E4a-prereg.md section 2).

All image coordinates are COLMAP's: the centre of pixel (col, row) is (col + 0.5, row + 0.5).
Depth maps and intrinsics belong to the UNDISTORTED images written by `colmap image_undistorter`;
masks belong to the ORIGINAL images, so every mask read/write goes through the two pixel maps built
by `undistorted_to_original_map` / `original_to_undistorted_map`.
"""
import numpy as np

OCC_TOL = 0.01          # a sample is occluded when its depth exceeds the local depth by > 1 %
CLOSE_KERNEL = 5        # forward splat: morphological closing side, in undistorted pixels


def _grid(h, w):
    """Pixel-centre coordinate grids (u = column + 0.5, v = row + 0.5)."""
    v, u = np.meshgrid(np.arange(h, dtype=np.float64) + 0.5,
                       np.arange(w, dtype=np.float64) + 0.5, indexing="ij")
    return u, v


def undistorted_to_original_map(cam_und, cam_orig, h, w):
    """For every undistorted pixel, the ORIGINAL pixel it came from -> (u_orig, v_orig) float arrays."""
    u, v = _grid(h, w)
    x = (u - cam_und.cx) / cam_und.fx
    y = (v - cam_und.cy) / cam_und.fy
    if cam_orig.has_distortion:
        r2 = x * x + y * y
        s = 1.0 + cam_orig.k1 * r2 + cam_orig.k2 * r2 * r2
        x, y = x * s, y * s
    return cam_orig.fx * x + cam_orig.cx, cam_orig.fy * y + cam_orig.cy


def original_to_undistorted_map(cam_orig, cam_und, h, w, iters=20):
    """For every ORIGINAL pixel, the undistorted pixel it maps to -> (u_und, v_und) float arrays."""
    u, v = _grid(h, w)
    xd = (u - cam_orig.cx) / cam_orig.fx
    yd = (v - cam_orig.cy) / cam_orig.fy
    if cam_orig.has_distortion:
        x, y = xd.copy(), yd.copy()
        for _ in range(iters):
            r2 = x * x + y * y
            s = 1.0 + cam_orig.k1 * r2 + cam_orig.k2 * r2 * r2
            x, y = xd / s, yd / s
    else:
        x, y = xd, yd
    return cam_und.fx * x + cam_und.cx, cam_und.fy * y + cam_und.cy


def sample_nearest(img, u, v):
    """Nearest-neighbour sample of `img` at continuous pixel centres (u, v); out of bounds -> 0/False."""
    h, w = img.shape[:2]
    col = np.floor(u).astype(np.int64)
    row = np.floor(v).astype(np.int64)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    out = np.zeros(u.shape, dtype=img.dtype)
    out[ok] = img[row[ok], col[ok]]
    return out, ok


def pose_matrices(image):
    """(R, t) of a COLMAP image: X_cam = R X_world + t."""
    return np.asarray(image.R, dtype=np.float64), np.asarray(image.tvec, dtype=np.float64)


def unproject(cam, depth, mask=None):
    """Valid pixels of a depth map -> (points in the camera frame [n, 3], their pixel indices [n])."""
    h, w = depth.shape
    u, v = _grid(h, w)
    ok = depth > 0
    if mask is not None:
        ok &= mask
    idx = np.flatnonzero(ok)
    d = depth.reshape(-1)[idx]
    x = (u.reshape(-1)[idx] - cam.cx) / cam.fx
    y = (v.reshape(-1)[idx] - cam.cy) / cam.fy
    return np.stack([x * d, y * d, d], axis=1), idx


def transfer(points_cam, R_from, t_from, R_to, t_to):
    """Camera-frame points of one view -> the other view's camera frame."""
    world = (points_cam - t_from) @ R_from                       # R_from^T (X - t), row-vector form
    return world @ R_to.T + t_to


def project(cam, points_cam):
    """Camera-frame points -> (u, v, z) in that camera; z <= 0 means behind the camera."""
    z = points_cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return cam.fx * points_cam[:, 0] / safe + cam.cx, cam.fy * points_cam[:, 1] / safe + cam.cy, z


def visible(cam, depth, u, v, z, occ_tol=OCC_TOL):
    """True where a sample at (u, v, z) is in front of / on the surface that view already sees."""
    d, inside = sample_nearest(depth, u, v)
    return inside & (z > 0) & ((d <= 0) | (z <= d * (1.0 + occ_tol)))


def backward_correspondence(ref_cam, ref_img, ref_depth, tgt_cam, tgt_img, tgt_depth, occ_tol=OCC_TOL):
    """Where every valid TARGET pixel lands in the reference view.  Object-independent: computed once
    per camera pair and then reused for every object (`backward_from`)."""
    pts, idx = unproject(tgt_cam, tgt_depth)
    if idx.size == 0:
        return {"idx": idx, "u": idx.astype(np.float64), "v": idx.astype(np.float64),
                "vis": np.zeros(0, bool), "shape": tgt_depth.shape}
    R_t, t_t = pose_matrices(tgt_img)
    R_r, t_r = pose_matrices(ref_img)
    u, v, z = project(ref_cam, transfer(pts, R_t, t_t, R_r, t_r))
    return {"idx": idx, "u": u, "v": v,
            "vis": visible(ref_cam, ref_depth, u, v, z, occ_tol), "shape": tgt_depth.shape}


def backward_from(corr, ref_mask_und):
    """One object's backward-warped target mask from a `backward_correspondence`."""
    out = np.zeros(corr["shape"], dtype=bool)
    if corr["idx"].size == 0:
        return out
    lab, _ = sample_nearest(ref_mask_und, corr["u"], corr["v"])
    flat = out.reshape(-1)
    flat[corr["idx"][corr["vis"] & lab]] = True
    return flat.reshape(out.shape)


def forward_correspondence(ref_cam, ref_img, ref_depth, tgt_cam, tgt_img, tgt_depth, occ_tol=OCC_TOL):
    """Where every valid REFERENCE pixel lands in the target view (object-independent, see `forward_from`)."""
    pts, idx = unproject(ref_cam, ref_depth)
    h, w = tgt_depth.shape
    if idx.size == 0:
        return {"src": idx, "col": idx, "row": idx, "shape": (h, w), "ref_shape": ref_depth.shape}
    R_r, t_r = pose_matrices(ref_img)
    R_t, t_t = pose_matrices(tgt_img)
    u, v, z = project(tgt_cam, transfer(pts, R_r, t_r, R_t, t_t))
    keep = visible(tgt_cam, tgt_depth, u, v, z, occ_tol)
    col = np.floor(u).astype(np.int64)
    row = np.floor(v).astype(np.int64)
    keep &= (col >= 0) & (col < w) & (row >= 0) & (row < h)
    return {"src": idx[keep], "col": col[keep], "row": row[keep], "shape": (h, w), "ref_shape": ref_depth.shape}


def forward_from(corr, ref_mask_und, close_kernel=CLOSE_KERNEL):
    """One object's forward-splatted target mask from a `forward_correspondence`, morphologically closed."""
    out = np.zeros(corr["shape"], dtype=bool)
    if corr["src"].size == 0:
        return out
    sel = ref_mask_und.reshape(-1)[corr["src"]]
    if not sel.any():
        return out
    out[corr["row"][sel], corr["col"][sel]] = True
    if close_kernel:
        import cv2
        k = np.ones((close_kernel, close_kernel), np.uint8)
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return out


def backward_warp(ref_mask_und, ref_cam, ref_img, ref_depth, tgt_cam, tgt_img, tgt_depth, occ_tol=OCC_TOL):
    """Target pixels whose own surface point falls on the reference mask and is visible there."""
    return backward_from(backward_correspondence(ref_cam, ref_img, ref_depth, tgt_cam, tgt_img,
                                                 tgt_depth, occ_tol), ref_mask_und)


def forward_warp(ref_mask_und, ref_cam, ref_img, ref_depth, tgt_cam, tgt_img, tgt_depth,
                 occ_tol=OCC_TOL, close_kernel=CLOSE_KERNEL):
    """Reference mask pixels splatted into the target, then morphologically closed."""
    return forward_from(forward_correspondence(ref_cam, ref_img, ref_depth, tgt_cam, tgt_img,
                                               tgt_depth, occ_tol), ref_mask_und, close_kernel)


def to_original(mask_und, cam_orig, cam_und, h, w, cache=None):
    """An undistorted-frame mask -> the original image frame (nearest sampling through the camera maps)."""
    if cache is None:
        cache = original_to_undistorted_map(cam_orig, cam_und, h, w)
    u, v = cache
    out, _ = sample_nearest(mask_und, u, v)
    return out.astype(bool)


def iou(a, b):
    """IoU of two boolean masks; two empty masks count as 1.0 (nothing to disagree on)."""
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def centre_offset(pred, gt):
    """Distance between the centroids, in units of the ground truth's equivalent-circle diameter."""
    if not gt.any() or not pred.any():
        return None
    gy, gx = np.nonzero(gt)
    py, px = np.nonzero(pred)
    d = float(np.hypot(px.mean() - gx.mean(), py.mean() - gy.mean()))
    return d / (2.0 * (gt.sum() / np.pi) ** 0.5)
