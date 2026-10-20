"""S1-E15 test helper (not a test module): a synthetic Pi3 cache -- a ring of pinhole cameras around two spheres inside a
large background sphere, ray-cast depth and labels -- in the layout of Data/<scene>/Cache/pi3/<cam>.npz.

Original resolution 84 x 112 (1008 = 12 * 84 = 9 * 112, so a nearest-upsampled mask survives the decoder's bilinear
1008 -> original resize exactly), work resolution 42 x 56.  Cameras cam00..cam{n-1} sit at ring angles 360 * i / n
around the origin (cam00 = the reference); OpenCV axes (x right, y down = world +y, z forward)."""
import json
import os

import numpy as np

H, W = 84, 112
h, w = 42, 56
F_WORK = 44.0
SPHERES = ((1, (-0.45, -0.4, 0.0), 0.32), (2, (0.45, 0.4, 0.0), 0.32))   # never occlude each other
BG_RADIUS = 6.0
RING = 3.0
# the A-c* constants of prediction P1 (prereg §5.2), pinned for the rule tests: they test the statistic, not the Step-1
# selection, so the Step-1 commit (id3d.CSTAR = c_best) leaves them unchanged
P1_CSTAR = dict(beta=1.0, kappa=1.25, sigma=0.05, rho_env=0.5, b_max=0.01, gamma=float("inf"), phi_rig=float("inf"))


def cam_name(i):
    return f"cam{i:02d}"


def c2w_at(phi_deg, radius=RING, height=0.0):
    p = np.radians(phi_deg)
    C = np.array([radius * np.sin(p), height, -radius * np.cos(p)])
    z = -C / np.linalg.norm(C)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = C
    return T


def K_at(hh, ww, f):
    return np.array([[f, 0, (ww - 1) / 2.0], [0, f, (hh - 1) / 2.0], [0, 0, 1.0]])


def raycast(c2w, hh, ww, f, spheres=SPHERES):
    """-> (z-depth [hh,ww] float32, labels [hh,ww] uint8); background = the far side of the BG sphere (label 0)."""
    K = K_at(hh, ww, f)
    v, u = np.mgrid[0:hh, 0:ww].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / f, (v - K[1, 2]) / f, np.ones_like(u)], -1) @ c2w[:3, :3].T
    C = c2w[:3, 3]
    a = (d * d).sum(-1)
    b = 2 * (d * C).sum(-1)
    t_bg = (-b + np.sqrt(b * b - 4 * a * ((C * C).sum() - BG_RADIUS ** 2))) / (2 * a)
    depth, lab = t_bg.copy(), np.zeros((hh, ww), np.uint8)
    for o, S, r in spheres:
        oc = C - np.asarray(S)
        bb = 2 * (d * oc).sum(-1)
        disc = bb * bb - 4 * a * ((oc * oc).sum() - r * r)
        t = np.where(disc >= 0, (-bb - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
        hit = (t > 0) & (t < depth)
        depth = np.where(hit, t, depth)
        lab = np.where(hit, o, lab).astype(np.uint8)
    return depth.astype(np.float32), lab


def write_scene(root, name="Syn", n=12, extra=None):
    """Data/<name>/{meta.json, Cache/pi3/<cam>.npz}; extra = {cam: (phi, height)} cameras off the ring.
    -> dict(data, cache, cams, ref, labels {cam: original-res labels}, work_labels, centres)."""
    data = os.path.join(root, "Data")
    cache = os.path.join(data, name, "Cache", "pi3")
    os.makedirs(cache, exist_ok=True)
    poses = {cam_name(i): (360.0 * i / n, 0.0) for i in range(n)}
    poses.update(extra or {})
    labels, work_labels, centres = {}, {}, {}
    for c, (phi, hgt) in poses.items():
        T = c2w_at(phi, height=hgt)
        depth, lw = raycast(T, h, w, F_WORK)
        _, lo = raycast(T, H, W, 2 * F_WORK)
        np.savez(os.path.join(cache, f"{c}.npz"), depth=depth, conf=np.ones_like(depth), K=K_at(h, w, F_WORK), c2w=T)
        labels[c], work_labels[c], centres[c] = lo, lw, T[:3, 3]
    cams = list(poses)
    json.dump({"dataset": name, "ref_frame": 0, "ref_cam": cams[0], "cams": cams, "mask_cams": cams,
               "image_size": [W, H], "source": "synthetic"}, open(os.path.join(data, name, "meta.json"), "w"))
    json.dump({"ref_cam": cams[0]}, open(os.path.join(cache, "MANIFEST.json"), "w"))
    return dict(data=data, cache=cache, cams=cams, ref=cams[0], labels=labels, work_labels=work_labels, centres=centres)
