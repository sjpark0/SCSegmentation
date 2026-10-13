"""Minimal camera containers with the attributes mvseg2d/geo/warp.py reads.

warp.py was written against COLMAP's Camera/Image; here a camera is a distortion-free pinhole and a
pose holds the WORLD-TO-CAMERA rotation R and translation t (X_cam = R X_world + t).  Pixel centres
are at (col + 0.5, row + 0.5), COLMAP's convention, which is why `from_c2w` adds 0.5 to a principal
point fitted in pixel-index coordinates (as Pi3's `recover_intrinsic_from_rays_d` does).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Pinhole:
    fx: float
    fy: float
    cx: float
    cy: float
    has_distortion: bool = False
    k1: float = 0.0
    k2: float = 0.0


@dataclass
class Pose:
    R: np.ndarray
    tvec: np.ndarray


def from_c2w(K, c2w):
    """(3x3 intrinsics in pixel-index coords, 4x4 camera-to-world) -> (Pinhole, Pose)."""
    K = np.asarray(K, dtype=np.float64)
    c2w = np.asarray(c2w, dtype=np.float64)
    cam = Pinhole(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]) + 0.5, float(K[1, 2]) + 0.5)
    R_c2w, t_c2w = c2w[:3, :3], c2w[:3, 3]
    R = R_c2w.T
    return cam, Pose(R, -R @ t_c2w)


def scale(cam, sx, sy):
    """Intrinsics of the same camera after resizing the image by (sx, sy)."""
    return Pinhole(cam.fx * sx, cam.fy * sy, cam.cx * sx, cam.cy * sy)
