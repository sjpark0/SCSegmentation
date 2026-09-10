#!/usr/bin/env python3
"""COLMAP model / depth-map readers and the pinhole+radial camera maps (S1-E4, docs/stage1-E4a-prereg.md).

Pure functions on plain python / numpy: no torch, no COLMAP binary.  Everything here is exercised by
tests/test_colmap_io.py; the geometry conventions follow COLMAP exactly:

  * `images.txt` stores the WORLD-TO-CAMERA rotation as a quaternion (qw, qx, qy, qz) and translation t,
    so a world point X maps to the camera frame as  X_cam = R X + t  and the camera centre is -R^T t.
  * pixel coordinates are continuous with the pixel CENTRE at (col + 0.5, row + 0.5) (COLMAP convention),
    which is what `image_undistorter` and `patch_match_stereo` assume.
  * SIMPLE_RADIAL params are (f, cx, cy, k): normalised (x, y) are distorted as x*(1 + k r^2).
    PINHOLE params are (fx, fy, cx, cy) and have no distortion.
"""
import os
import struct

CAMERA_MODELS = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
}


class Camera:
    """One COLMAP camera: intrinsics plus the distortion of its model."""

    def __init__(self, camera_id, model, width, height, params):
        assert model in CAMERA_MODELS, f"unsupported camera model {model}"
        self.id, self.model, self.width, self.height = int(camera_id), model, int(width), int(height)
        self.params = [float(p) for p in params]
        assert len(self.params) == len(CAMERA_MODELS[model]), (model, self.params)
        p = dict(zip(CAMERA_MODELS[model], self.params))
        self.fx = p.get("fx", p.get("f"))
        self.fy = p.get("fy", p.get("f"))
        self.cx, self.cy = p["cx"], p["cy"]
        self.k1 = p.get("k", p.get("k1", 0.0))
        self.k2 = p.get("k2", 0.0)

    @property
    def has_distortion(self):
        return self.k1 != 0.0 or self.k2 != 0.0

    def distort(self, x, y):
        """Normalised undistorted (x, y) -> normalised distorted, i.e. what the sensor actually saw."""
        if not self.has_distortion:
            return x, y
        r2 = x * x + y * y
        s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return x * s, y * s

    def undistort(self, xd, yd, iters=20, tol=1e-10):
        """Inverse of `distort` by fixed-point iteration (COLMAP uses Newton; both converge for small k)."""
        if not self.has_distortion:
            return xd, yd
        x, y = xd, yd
        for _ in range(iters):
            r2 = x * x + y * y
            s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            nx, ny = xd / s, yd / s
            if abs(nx - x) < tol and abs(ny - y) < tol:
                x, y = nx, ny
                break
            x, y = nx, ny
        return x, y

    def pixel_to_normalized(self, u, v):
        """Pixel (u, v) -> normalised distorted coordinates (no undistortion applied)."""
        return (u - self.cx) / self.fx, (v - self.cy) / self.fy

    def normalized_to_pixel(self, x, y):
        return self.fx * x + self.cx, self.fy * y + self.cy


class Image:
    """One registered COLMAP image: name, camera id and the world-to-camera pose."""

    def __init__(self, image_id, qvec, tvec, camera_id, name):
        self.id, self.camera_id, self.name = int(image_id), int(camera_id), name
        self.qvec = [float(q) for q in qvec]
        self.tvec = [float(t) for t in tvec]

    @property
    def R(self):
        return qvec_to_rotmat(self.qvec)


def qvec_to_rotmat(q):
    """COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation as a list of lists (world-to-camera)."""
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]]


def read_cameras_txt(path):
    cams = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        c = Camera(f[0], f[1], f[2], f[3], f[4:])
        cams[c.id] = c
    return cams


def read_images_txt(path):
    """`images.txt` -> {name: Image}.  Only the pose lines are read (the 2D point lines are skipped)."""
    out, expect_pose = {}, True
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if expect_pose:
            f = line.split()
            im = Image(f[0], f[1:5], f[5:8], f[8], " ".join(f[9:]))
            out[im.name] = im
        expect_pose = not expect_pose
    return out


def read_model_txt(folder):
    """(cameras, images) of a COLMAP TXT model folder."""
    return (read_cameras_txt(os.path.join(folder, "cameras.txt")),
            read_images_txt(os.path.join(folder, "images.txt")))


def read_colmap_array(path):
    """COLMAP dense binary array ('<w>&<h>&<c>&' header + row-major float32) -> (list of floats, w, h, c)."""
    with open(path, "rb") as fh:
        head = b""
        amp = 0
        while amp < 3:
            ch = fh.read(1)
            if not ch:
                raise ValueError(f"truncated header in {path}")
            head += ch
            if ch == b"&":
                amp += 1
        w, h, c = (int(x) for x in head.decode().split("&")[:3])
        data = fh.read(w * h * c * 4)
    assert len(data) == w * h * c * 4, (path, len(data), w * h * c * 4)
    return list(struct.unpack(f"<{w * h * c}f", data)), w, h, c


def read_depth_map(path):
    """COLMAP depth map -> (numpy float32 [h, w], with 0 meaning 'no depth')."""
    import numpy as np
    vals, w, h, c = read_colmap_array(path)
    assert c == 1, (path, c)
    return np.asarray(vals, dtype="float32").reshape(h, w)
