#!/usr/bin/env python3
"""Replace the library background of the cart take with an outdoor panorama (docs/cart.md §6; user 2026-10-21: show why
object segmentation is needed by putting only the segmented objects into a new background, "야외도서관 컨셉으로").

    bg_cart.py geom    <bg dir>                                  floor plane, metric scale, table centre -> geom.json
    bg_cart.py preview <bg dir> --pano P [--yaw D ...] [--cams ...] [--frame 120] [--masks DIR] [--tag T]
    bg_cart.py yaws    <bg dir> --pano P [--cam 5]                 one camera at 8 panorama yaws (contact sheet)
    bg_cart.py comp    <bg dir> --pano P --yaw D --fg <export fg dir> --out DIR [--workers 8]   every frame + videos
    bg_cart.py tonemap <hdr> <tonemapped jpg> <out jpg>             16K HDR in the look of the 8K tonemapped JPG
    bg_cart.py static  <bg dir> --name N --pick X Y --prompts P,.. [--back-m --half-m --height-m]
                                                    SAM 3 masks of a static background object kept with the objects
                                                    (scsam3 container, GPU)

Geometry.  Cameras: COLMAP cameras.txt / images.txt of the take (<bg>/colmap/, copied from the NAS
GSProject/GSData/library_261014/cart/png/).  Floor: RANSAC plane through the COLMAP points
(<bg>/colmap/points3D_multipleview.ply, from SCSegGaussians data4dgs/cart_f100_199) whose normal is within 20 deg of
the cameras' mean image-down axis, below every camera.  Metric scale: the table top (height mode of the points that fall
inside the table mask in most cameras) is taken to be TABLE_M high.  Table centre: those points' centroid on the floor.

Background.  The panorama is a ground-projected dome (as Blender's / three.js' "ground projected" environment): the
panorama's own camera stands H_CAP_M above the floor over the table centre, its ground is the floor plane out to a
sphere of radius R_DOME_M, beyond that the sphere.  Each camera ray hits the floor or the sphere at X; the pixel takes
the panorama colour in direction X - (panorama camera).  Cameras are static, so each camera's background is one image.

Contact shadows.  Overcast sky: the ground darkens by the sky fraction the objects hide.  Per frame, a visual hull of the
foreground masks on a ground grid (GRID_M around the table centre, HEIGHTS_M layers; a voxel's occupancy rises linearly
from 0 to 1 as the share of the cameras seeing it that put it in the foreground goes from HULL_LO to 1, so an object
missing from a few cameras still casts a weaker shadow), each layer blurred by the form factor of a horizontal
patch at its height, h^2 / (pi (r^2 + h^2)^2), blocked = max over layers, then 1 - (1 - blocked)^SHADOW_GAMMA (the
hull of thin legs and of objects missing from some cameras is too small, and the camera arc of ~85 deg smears it away
from the cameras, so the physical value is too weak where it shows), ground x (1 - SHADOW * blocked).

Composite: alpha = the union of the object masks, closed (5 px ellipse), eroded 1 px and blurred (EDGE_SIGMA); out = alpha src +
(1 - alpha) background x shadow.  Host python (stage1/.venv: cv2, numpy).
"""
import argparse
import glob
import json
import math
import os

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CART = os.path.join(REPO, "Data", "Cart")
PROBE = os.path.join(CART, "cart_f120_probe", "SegMaskSam3XW1CGPS4MFsCs10kG2")
W, H, NCAM, REF = 1920, 1080, 24, 5
TABLE_M = 0.74          # table-top height (m) that fixes the metric scale
H_CAP_M = 1.7           # height of the panorama camera above its ground (m)
R_DOME_M = 40.0         # dome radius (m)
GRID_M, GRID_RES_M = 5.0, 0.025
HEIGHTS_M = (0.02, 0.08, 0.2, 0.4, 0.65, 0.9, 1.2, 1.6)
HULL_LO, HULL_MIN_VIEWS = 0.85, 6
SHADOW, SHADOW_GAMMA = 0.85, 2.5
EDGE_SIGMA = 0.8
RING = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]


# ---------------------------------------------------------------------------------------------------------- geometry
def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_cams(colmap):
    """{cam index: (K, R, t)} from COLMAP text files; image names c000-f000000.webp .. c023-..."""
    intr = {}
    for line in open(os.path.join(colmap, "cameras.txt")):
        s = line.split()
        if s and not line.startswith("#"):
            assert s[1] == "SIMPLE_PINHOLE" and (int(s[2]), int(s[3])) == (W, H), s
            f, cx, cy = map(float, s[4:7])
            intr[int(s[0])] = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    cams = {}
    for line in open(os.path.join(colmap, "images.txt")):
        s = line.split()
        if len(s) == 10 and not line.startswith("#"):
            cams[int(s[9][1:4])] = (intr[int(s[8])], _rot(list(map(float, s[1:5]))), np.array(list(map(float, s[5:8]))))
    assert sorted(cams) == list(range(NCAM)), sorted(cams)
    return cams


def load_ply(path):
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        n = int([l for l in head.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        a = np.frombuffer(f.read(n * dt.itemsize), dt)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)


def project(cams, ci, X):
    """pixel (u, v) and depth of world points X (N, 3) in camera ci."""
    K, R, t = cams[ci]
    Xc = X @ R.T + t
    z = Xc[:, 2]
    uv = Xc @ K.T
    return uv[:, 0] / np.maximum(z, 1e-9), uv[:, 1] / np.maximum(z, 1e-9), z


def cmd_geom(a):
    cams = load_cams(os.path.join(a.bg, "colmap"))
    P = load_ply(os.path.join(a.bg, "colmap", "points3D_multipleview.ply"))
    Cs = np.array([-cams[i][1].T @ cams[i][2] for i in range(NCAM)])
    down = np.array([cams[i][1][1] for i in range(NCAM)]).mean(0)
    down /= np.linalg.norm(down)
    rng = np.random.default_rng(0)
    best, cos20 = (0, None), math.cos(math.radians(20))
    for _ in range(20000):
        i = rng.choice(len(P), 3, replace=False)
        n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]])
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n) * np.sign(n @ down)
        d = n @ P[i[0]]
        if n @ down < cos20 or (Cs @ n).max() >= d:       # floor: near-horizontal and below every camera
            continue
        c = int((np.abs(P @ n - d) < 0.03).sum())
        if c > best[0]:
            best = (c, (n, d))
    n, d = best[1]
    Q = P[np.abs(P @ n - d) < 0.03]
    mu = Q.mean(0)
    n = np.linalg.svd(Q - mu)[2][-1]
    n *= np.sign(n @ down)
    d = float(n @ mu)
    height = lambda X: d - X @ n                             # noqa: E731  (units above the floor)
    # table top: points inside the table mask (id 7) in >= 8 of these cameras, height mode
    hits = []
    for ci in (2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20):
        m = cv2.imread(os.path.join(a.masks, f"cam{ci:04d}", "0", "7.png"), cv2.IMREAD_GRAYSCALE)
        u, v, z = project(cams, ci, P)
        ok = (z > 0.1) & (u >= 0) & (u < W - 0.5) & (v >= 0) & (v < H - 0.5)
        idx = np.where(ok)[0]
        hits.append(idx[m[v[idx].astype(int), u[idx].astype(int)] > 0])
    uq, cnt = np.unique(np.concatenate(hits), return_counts=True)
    tp = P[uq[cnt >= 8]]
    th = height(tp)
    hist, edges = np.histogram(th, bins=np.arange(0.6, 2.0, 0.01))
    mode = edges[hist.argmax()] + 0.005
    band = np.abs(th - mode) < 0.04
    top = float(np.median(th[band]))
    scale = TABLE_M / top                                    # metres per unit
    centre = tp[band].mean(0)
    centre = centre + (height(centre[None])[0]) * n          # drop to the floor
    up = -n
    f5 = cams[REF][1][2]
    e1 = f5 - (f5 @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, up)                                    # right of the reference camera's view
    assert cams[REF][1][0] @ e2 > 0.9, cams[REF][1][0] @ e2
    g = dict(floor_n=n.tolist(), floor_d=d, floor_inliers=best[0], scale_m_per_unit=scale, table_top_units=top,
             table_points=int(band.sum()), centre=centre.tolist(), up=up.tolist(), e1=e1.tolist(), e2=e2.tolist(),
             cam_height_m=[round(float(height(Cs[i:i + 1])[0] * scale), 3) for i in range(NCAM)],
             cam_dist_m=[round(float(np.linalg.norm((Cs[i] - centre) - ((Cs[i] - centre) @ up) * up) * scale), 3)
                         for i in range(NCAM)],
             TABLE_M=TABLE_M, masks=a.masks)
    json.dump(g, open(os.path.join(a.bg, "geom.json"), "w"), indent=1)
    print(json.dumps({k: g[k] for k in ("floor_inliers", "scale_m_per_unit", "table_top_units", "table_points")}))
    print("camera heights (m)", g["cam_height_m"])
    print("camera distance to table centre (m)", g["cam_dist_m"])


# -------------------------------------------------------------------------------------------------------- background
class Scene:
    def __init__(self, bg):
        self.cams = load_cams(os.path.join(bg, "colmap"))
        g = json.load(open(os.path.join(bg, "geom.json")))
        self.g = g
        self.n, self.d, self.s = np.array(g["floor_n"]), g["floor_d"], g["scale_m_per_unit"]
        self.up, self.e1, self.e2 = (np.array(g[k]) for k in ("up", "e1", "e2"))
        self.O = np.array(g["centre"])
        self.Op = self.O + (H_CAP_M / self.s) * self.up
        self.Rd = R_DOME_M / self.s

    def rays(self, ci, step=1):
        K, R, t = self.cams[ci]
        ys, xs = np.mgrid[0:H:step, 0:W:step].astype(np.float64)
        v = np.stack([(xs + 0.5 * step - K[0, 2]) / K[0, 0], (ys + 0.5 * step - K[1, 2]) / K[1, 1], np.ones_like(xs)], -1)
        v = v @ R                                             # R^T v for row vectors
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        return -R.T @ t, v

    def background_maps(self, ci, pano_wh, yaw_deg, step=1):
        """remap maps into the panorama + ground coordinates (metres along e1, e2 from the table centre; nan = sky/far)."""
        c, v = self.rays(ci, step)
        vn = v @ self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = (self.d - c @ self.n) / vn
        X = c + np.where(vn[..., None] > 0, tg[..., None], np.inf) * v
        oc = c - self.Op
        b = v @ oc
        ts = -b + np.sqrt(b * b - (oc @ oc - self.Rd ** 2))
        ground = (vn > 0) & (np.linalg.norm(X - self.Op, axis=-1) <= self.Rd)
        X = np.where(ground[..., None], X, c + ts[..., None] * v)
        w = X - self.Op
        w /= np.linalg.norm(w, axis=-1, keepdims=True)
        az = np.arctan2(w @ self.e2, w @ self.e1) + math.radians(yaw_deg)
        lat = np.arcsin(np.clip(w @ self.up, -1, 1))
        pw, ph = pano_wh
        mx = ((0.5 + az / (2 * math.pi)) % 1.0) * pw - 0.5
        my = (0.5 - lat / math.pi) * ph - 0.5
        rel = (X - self.O) * self.s
        gx = np.where(ground, rel @ self.e1, np.nan)
        gy = np.where(ground, rel @ self.e2, np.nan)
        return mx.astype(np.float32), my.astype(np.float32), gx.astype(np.float32), gy.astype(np.float32)


def render_bg(pano, mx, my):
    return cv2.remap(pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# ----------------------------------------------------------------------------------------------------------- shadows
class Hull:
    def __init__(self, sc):
        self.sc = sc
        r = np.arange(-GRID_M, GRID_M + 1e-9, GRID_RES_M)
        self.N = len(r)
        gy, gx = np.meshgrid(r, r, indexing="ij")                # row = e2 (gy), col = e1 (gx)
        self.idx, self.valid = [], []
        for h in HEIGHTS_M:
            X = (sc.O + (gx[..., None] * sc.e1 + gy[..., None] * sc.e2 + h * sc.up) / sc.s).reshape(-1, 3)
            ii, vv = [], []
            for ci in range(NCAM):
                u, v, z = project(sc.cams, ci, X)
                ok = (z > 0) & (u >= 0) & (u < W - 0.5) & (v >= 0) & (v < H - 0.5)
                ii.append(np.where(ok, np.round(v).clip(0, H - 1).astype(np.int32) * W + np.round(u).clip(0, W - 1).astype(np.int32), 0))
                vv.append(ok)
            self.idx.append(np.stack(ii))
            self.valid.append(np.stack(vv))
        self.kern = []
        for h in HEIGHTS_M:
            rad = max(2, int(math.ceil(3 * h / GRID_RES_M)))
            yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1] * GRID_RES_M
            k = h * h / (math.pi * (xx * xx + yy * yy + h * h) ** 2) * GRID_RES_M ** 2
            self.kern.append(k.astype(np.float32))

    def blocked(self, fg):
        """fg: list of 24 boolean masks (H, W) -> blocked sky fraction on the grid (N, N)."""
        flat = np.stack([m.reshape(-1) for m in fg])
        out = np.zeros((self.N, self.N), np.float32)
        self.occ = []
        for k in range(len(HEIGHTS_M)):
            inside = np.take_along_axis(flat, self.idx[k], axis=1) & self.valid[k]
            nv = self.valid[k].sum(0)
            frac = inside.sum(0) / np.maximum(nv, 1)
            occ = (np.clip((frac - HULL_LO) / (1 - HULL_LO), 0, 1) * (nv >= HULL_MIN_VIEWS)).reshape(self.N, self.N)
            occ = occ.astype(np.float32)
            self.occ.append(occ)
            out = np.maximum(out, cv2.filter2D(occ, -1, self.kern[k], borderType=cv2.BORDER_CONSTANT))
        return 1 - (1 - np.clip(out, 0, 1)) ** SHADOW_GAMMA


def shadow_in_view(blocked, gx, gy):
    mx = (gx + GRID_M) / GRID_RES_M
    my = (gy + GRID_M) / GRID_RES_M
    b = cv2.remap(blocked, np.nan_to_num(mx, nan=-10).astype(np.float32), np.nan_to_num(my, nan=-10).astype(np.float32),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return 1.0 - SHADOW * b


# --------------------------------------------------------------------------------------------------------- composite
def load_fg(masks, ci, i, static=()):
    """union of the run's object masks (<masks>/camNNNN/<i>/<id>.png) and the static layers (static masks, per camera)."""
    fs = glob.glob(os.path.join(masks, f"cam{ci:04d}", str(i), "*.png"))
    m = np.zeros((H, W), bool)
    for f in fs:
        m |= cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 0
    for st in static:
        m |= st[ci]
    return m


def load_static(bg, names):
    return [[cv2.imread(os.path.join(bg, "static", nm, f"cam{ci:04d}.png"), cv2.IMREAD_GRAYSCALE) > 0
             for ci in range(NCAM)] for nm in names]


def alpha_of(fg):
    """close the 1-2 px slivers between neighbouring objects' masks (the new background showed through them as
    coloured lines), then erode 1 px and blur the outer edge"""
    m = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    a = cv2.erode(m, np.ones((3, 3), np.uint8)).astype(np.float32) / 255
    return cv2.GaussianBlur(a, (0, 0), EDGE_SIGMA)[..., None]


def composite(src, bg, alpha, shade):
    return np.clip(alpha * src.astype(np.float32) + (1 - alpha) * bg.astype(np.float32) * shade[..., None], 0, 255
                   ).astype(np.uint8)


def load_pano(path):
    p = cv2.imread(path, cv2.IMREAD_COLOR)
    assert p is not None and p.shape[1] == 2 * p.shape[0], path
    return p


def cmd_yaws(a):
    sc = Scene(a.bg)
    pano = load_pano(a.pano)
    src = cv2.imread(os.path.join(CART, "src", f"cam{a.cam:04d}", f"{a.frame:06d}.jpg"))
    fg = load_fg(a.masks, a.cam, 0, load_static(a.bg, a.static))
    al = alpha_of(fg)
    tiles = []
    for yaw in range(0, 360, 45):
        mx, my, gx, gy = sc.background_maps(a.cam, pano.shape[1::-1], yaw)
        out = composite(src, render_bg(pano, mx, my), al, np.ones((H, W), np.float32))
        t = cv2.resize(out, (W // 3, H // 3), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"yaw {yaw}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(t)
    sheet = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(2)])
    os.makedirs(os.path.join(a.bg, "preview"), exist_ok=True)
    out = os.path.join(a.bg, "preview", f"yaws_{os.path.basename(a.pano).split('_tm')[0]}_cam{a.cam:04d}.jpg")
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out)


def cmd_preview(a):
    sc = Scene(a.bg)
    pano = load_pano(a.pano)
    name = os.path.basename(a.pano).split("_tm")[0].split("_16k")[0]
    fgs = [load_fg(a.masks, ci, 0, load_static(a.bg, a.static)) for ci in range(NCAM)]
    hull = Hull(sc)
    blocked = hull.blocked(fgs)
    od = os.path.join(a.bg, "preview", f"{name}{a.tag}")
    os.makedirs(od, exist_ok=True)
    cv2.imwrite(os.path.join(od, "blocked_grid.png"), (255 * (1 - blocked)).astype(np.uint8))
    tiles = {}
    for ci in range(NCAM):
        src = cv2.imread(os.path.join(CART, "src", f"cam{ci:04d}", f"{a.frame:06d}.jpg"))
        mx, my, gx, gy = sc.background_maps(ci, pano.shape[1::-1], a.yaw)
        out = composite(src, render_bg(pano, mx, my), alpha_of(fgs[ci]), shadow_in_view(blocked, gx, gy))
        if ci in a.cams:
            cv2.imwrite(os.path.join(od, f"cam{ci:04d}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles[ci] = cv2.resize(out, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(tiles[ci], f"cam{ci:04d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    mosaic = np.vstack([np.hstack([tiles[c] for c in RING[r * 6:(r + 1) * 6]]) for r in range(4)])
    cv2.imwrite(os.path.join(od, "all24.jpg"), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(od)


def cmd_static(a):
    """A static object left out of the run but kept in the composite (the chrysanthemum pot in front of the table -- without
    it the table base it hides would be cut off -- and, user 2026-10-21, the empty chair at the front corner of the
    table).  Reference camera: the SAM 3 text instance under --pick.  Its 3D centre: on the ray through the mask
    centroid, --back-m deeper than the floor point under the mask's lowest pixel (the front of the object).  A 3D box
    (that centre dropped to the floor, +-half-m along e1 / e2, 0..height-m up) is projected into every camera; the
    camera's mask is the text instance whose box overlaps the projected box most (IoU >= 0.3, ties to the one holding the
    projected centre), else a box prompt.  One mask per camera, used for every frame ->
    <bg>/static/<name>/camNNNN.png (0/255), all24.jpg, static.json."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import seed_cart as SC
    sc = Scene(a.bg)
    sam = SC.Sam3(conf=0.3, interactive=True)
    prompts = a.prompts.split(",")

    def texts(img):
        out = []
        for pr in prompts:
            m, s = sam.text(img, pr)
            out += [(float(si), mi, pr) for mi, si in zip(m, s)]
        return out

    ref = SC.load(f"cam{REF:04d}", a.frame)
    held = [c for c in texts(ref) if c[1][int(a.pick[1]), int(a.pick[0])]]
    assert held, "no text instance under --pick"
    m0 = max(held, key=lambda c: c[0])[1]
    ys, xs = np.nonzero(m0)
    yb = ys.max()
    xb = float(np.median(xs[ys >= yb - 2]))
    K, R, t = sc.cams[REF]
    c = -R.T @ t
    v = R.T @ np.array([(xb + 0.5 - K[0, 2]) / K[0, 0], (yb + 0.5 - K[1, 2]) / K[1, 1], 1.0])
    G = c + (sc.d - c @ sc.n) / (v @ sc.n) * v
    zc = (R @ G + t)[2] + a.back_m / sc.s
    P = R.T @ (np.array([(xs.mean() + 0.5 - K[0, 2]) / K[0, 0], (ys.mean() + 0.5 - K[1, 2]) / K[1, 1], 1.0]) * zc - t)
    Pf = P + (sc.d - P @ sc.n) * sc.n
    hw = a.half_m / sc.s
    corners = np.array([Pf + (i * hw) * sc.e1 + (j * hw) * sc.e2 + k * (a.height_m / sc.s) * sc.up
                        for i in (-1, 1) for j in (-1, 1) for k in (0, 1)])
    od = os.path.join(a.bg, "static", a.name)
    os.makedirs(od, exist_ok=True)
    rep, tiles = {}, {}
    for ci in range(NCAM):
        u, vv, z = project(sc.cams, ci, P[None])
        cu, cv_, cz = project(sc.cams, ci, corners)
        box = [float(np.clip(cu.min(), 0, W - 1)), float(np.clip(cv_.min(), 0, H - 1)),
               float(np.clip(cu.max(), 0, W - 1)), float(np.clip(cv_.max(), 0, H - 1))]
        img = SC.load(f"cam{ci:04d}", a.frame)
        m = np.zeros((H, W), bool)
        r = dict(centre=[round(float(u[0]), 1), round(float(vv[0]), 1)], box=[round(x) for x in box])
        if (cz > 0).all() and box[2] - box[0] > 4 and box[3] - box[1] > 4:
            cands = []
            for sc1, mi, pr in texts(img):
                iou = SC.box_iou(SC.bbox(mi), box)
                inside = 0 <= u[0] < W and 0 <= vv[0] < H and bool(mi[int(vv[0]), int(u[0])])
                if iou >= 0.3:
                    cands.append((iou + 0.2 * inside, sc1, mi, pr, iou))
            if cands:
                _, s1, m, how, iou = max(cands, key=lambda q: q[0])
            else:
                m, s1 = sam.inst(img, box=box)
                how, iou = "box", SC.box_iou(SC.bbox(m), box)
            r.update(score=round(float(s1), 3), by=how, iou=round(float(iou), 3), px=int(m.sum()))
        rep[f"cam{ci:04d}"] = r
        cv2.imwrite(os.path.join(od, f"cam{ci:04d}.png"), m.astype(np.uint8) * 255)
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        bgr[m] = (0.45 * bgr[m] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.rectangle(bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 3)
        tiles[ci] = cv2.resize(bgr, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(tiles[ci], f"cam{ci:04d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(od, "all24.jpg"),
                np.vstack([np.hstack([tiles[c] for c in RING[r * 6:(r + 1) * 6]]) for r in range(4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(dict(name=a.name, frame=a.frame, pick_ref=a.pick, prompts=prompts, centre_3d=P.tolist(),
                   half_m=a.half_m, height_m=a.height_m, back_m=a.back_m, cams=rep),
              open(os.path.join(od, "static.json"), "w"), indent=1)
    print(json.dumps(rep))


def cmd_tonemap(a):
    """16K HDR -> 8-bit panorama in the look of Poly Haven's own tonemapped JPG (8K): per-channel curves fitted on a
    2048x1024 downsample (median JPG value per bin of log2 radiance, made monotone), applied to the full HDR in strips.
    Run it in a container: the full float image is 1.6 GB (docs/cart.md §6, the 2026-10-21 session kill)."""
    hdr = cv2.imread(a.hdr, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    ref = cv2.imread(a.ref)
    small = cv2.resize(hdr, (2048, 1024), interpolation=cv2.INTER_AREA)
    refs = cv2.resize(ref, (2048, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
    curves, rep = [], {}
    for ch in range(3):
        x = np.log2(small[..., ch].ravel() + 1e-6)
        y = refs[..., ch].ravel()
        lo, hi = np.percentile(x, [0.01, 99.99])
        k = np.clip(((x - lo) / (hi - lo) * 256).astype(int), 0, 255)
        order = np.lexsort((y, k))
        ks, ys = k[order], y[order]
        starts = np.searchsorted(ks, np.arange(256))
        ends = np.searchsorted(ks, np.arange(256), side="right")
        med = np.array([ys[(s0 + e0) // 2] if e0 > s0 else np.nan for s0, e0 in zip(starts, ends)])
        c = lo + (np.arange(256) + 0.5) / 256 * (hi - lo)
        ok = ~np.isnan(med)
        med = np.maximum.accumulate(np.interp(c, c[ok], med[ok]))
        curves.append((c, med))
        rep[ch] = round(float(np.abs(np.interp(x, c, med) - y).mean()), 2)
    out = np.empty(hdr.shape, np.uint8)
    for r0 in range(0, hdr.shape[0], 512):
        blk = hdr[r0:r0 + 512]
        for ch, (c, med) in enumerate(curves):
            out[r0:r0 + 512, :, ch] = np.clip(np.interp(np.log2(blk[..., ch] + 1e-6), c, med) + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(a.out, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(json.dumps(dict(hdr=a.hdr, shape=list(hdr.shape), fit_mae_per_channel=rep, out=a.out)))


# ------------------------------------------------------------------------------------------------- full composite
C_ = {}


def _plates(a):
    """per-camera background plate (PNG) and ground coordinates (npy) -> <out>/plates, computed once"""
    pd = os.path.join(a.out, "plates")
    os.makedirs(pd, exist_ok=True)
    if all(os.path.exists(os.path.join(pd, f"cam{c:04d}_gxy.npy")) for c in range(NCAM)):
        return pd
    sc = Scene(a.bg)
    pano = load_pano(a.pano)
    for ci in range(NCAM):
        mx, my, gx, gy = sc.background_maps(ci, pano.shape[1::-1], a.yaw)
        cv2.imwrite(os.path.join(pd, f"cam{ci:04d}.png"), render_bg(pano, mx, my))
        np.save(os.path.join(pd, f"cam{ci:04d}_gxy.npy"), np.stack([gx, gy]))
    return pd


def _init(a, pd):
    C_["a"] = a
    C_["hull"] = Hull(Scene(a.bg))
    C_["bg"] = [cv2.imread(os.path.join(pd, f"cam{c:04d}.png")) for c in range(NCAM)]
    gxy = [np.load(os.path.join(pd, f"cam{c:04d}_gxy.npy")) for c in range(NCAM)]
    C_["map"] = [(np.nan_to_num((g[0] + GRID_M) / GRID_RES_M, nan=-10).astype(np.float32),
                  np.nan_to_num((g[1] + GRID_M) / GRID_RES_M, nan=-10).astype(np.float32)) for g in gxy]


def _frame(s):
    a = C_["a"]
    outs = [os.path.join(a.out, "frames", f"cam{c:04d}", f"{s:06d}.jpg") for c in range(NCAM)]
    if all(os.path.exists(o) for o in outs):
        return s, None
    fgs = [cv2.imread(os.path.join(a.fg, f"cam{c:04d}", f"{s:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127 for c in range(NCAM)]
    blocked = C_["hull"].blocked(fgs)
    for c in range(NCAM):
        src = cv2.imread(os.path.join(CART, "src", f"cam{c:04d}", f"{s:06d}.jpg"))
        b = cv2.remap(blocked, *C_["map"][c], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out = composite(src, C_["bg"][c], alpha_of(fgs[c]), 1.0 - SHADOW * b)
        tmp = outs[c] + ".part.jpg"
        cv2.imwrite(tmp, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        os.replace(tmp, outs[c])
    return s, float(blocked.max())


def _writer(path, w, h, crf=18):
    import subprocess
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import render_cart as RC
    return subprocess.Popen([RC.ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                             f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)


def _video(job):
    a = C_["a"]
    kind, arg = job
    fr = lambda c, s: cv2.imread(os.path.join(a.out, "frames", f"cam{c:04d}", f"{s:06d}.jpg"))   # noqa: E731
    if kind == "cam":
        p = _writer(os.path.join(a.out, "cams", f"cam{arg:04d}.mp4"), W, H)
        for s in a.src_frames:
            p.stdin.write(fr(arg, s).tobytes())
    elif kind == "compare":                    # original | composite, one camera
        p = _writer(os.path.join(a.out, f"compare_cam{arg:04d}.mp4"), W, H // 2)
        for s in a.src_frames:
            o = cv2.imread(os.path.join(CART, "src", f"cam{arg:04d}", f"{s:06d}.jpg"))
            row = np.hstack([cv2.resize(o, (W // 2, H // 2), interpolation=cv2.INTER_AREA),
                             cv2.resize(fr(arg, s), (W // 2, H // 2), interpolation=cv2.INTER_AREA)])
            p.stdin.write(row.tobytes())
    else:                                      # mosaic part [lo, hi)
        lo, hi = arg
        p = _writer(os.path.join(a.out, f".mosaic_{lo:06d}.mp4"), 6 * 640, 4 * 360, crf=23)
        for s in a.src_frames[lo:hi]:
            tiles = []
            for c in RING:
                t = cv2.resize(fr(c, s), (640, 360), interpolation=cv2.INTER_AREA)
                cv2.putText(t, f"cam{c:04d}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                tiles.append(t)
            p.stdin.write(np.vstack([np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(4)]).tobytes())
    p.stdin.close()
    p.wait()
    return job


def cmd_comp(a):
    """All frames: <out>/frames/camNNNN/<src frame>.jpg (q92), cams/camNNNN.mp4, mosaic.mp4 (ring order 6x4),
    compare_cam0005.mp4 (original | composite).  fg = the export's fg PNGs (run objects + static chair and pot).
    Resumable (finished frames are skipped).  Run in a container (docs/cart.md §6)."""
    import subprocess
    import sys
    from multiprocessing import Pool
    a.src_frames = sorted(int(f[:-4]) for f in os.listdir(os.path.join(a.fg, "cam0005")) if f.endswith(".png"))
    if a.limit:
        a.src_frames = a.src_frames[:a.limit]
    for c in range(NCAM):
        os.makedirs(os.path.join(a.out, "frames", f"cam{c:04d}"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "cams"), exist_ok=True)
    pd = _plates(a)
    print("plates", pd, "frames", len(a.src_frames), flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a, pd)) as pool:
        n = 0
        for s, bmax in pool.imap_unordered(_frame, a.src_frames, chunksize=4):
            n += 1
            if n % 100 == 0:
                print(f"frames {n}/{len(a.src_frames)}", flush=True)
        if a.no_video:
            return
        step = -(-len(a.src_frames) // 8)
        jobs = ([("cam", c) for c in range(NCAM)] + [("compare", REF)]
                + [("mosaic", (lo, lo + step)) for lo in range(0, len(a.src_frames), step)])
        for j in pool.imap_unordered(_video, jobs):
            print("video", j, flush=True)
    parts = sorted(f for f in os.listdir(a.out) if f.startswith(".mosaic_"))
    lst = os.path.join(a.out, ".mosaic.txt")
    open(lst, "w").write("".join(f"file '{f}'\n" for f in parts))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import render_cart as RC
    subprocess.run([RC.ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    os.path.join(a.out, "mosaic.mp4")], check=True)
    for f in parts + [".mosaic.txt"]:
        os.remove(os.path.join(a.out, f))
    json.dump(dict(pano=os.path.abspath(a.pano), yaw=a.yaw, fg=os.path.abspath(a.fg), frames=[a.src_frames[0], a.src_frames[-1]],
                   n=len(a.src_frames), H_CAP_M=H_CAP_M, R_DOME_M=R_DOME_M, SHADOW=SHADOW, SHADOW_GAMMA=SHADOW_GAMMA,
                   HULL_LO=HULL_LO, EDGE_SIGMA=EDGE_SIGMA, geom=json.load(open(os.path.join(a.bg, "geom.json")))),
              open(os.path.join(a.out, "comp.json"), "w"), indent=1)
    print("done", a.out, flush=True)


def main():
    global SHADOW, HULL_LO, SHADOW_GAMMA
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("geom", "yaws", "preview", "static"):
        p = sp.add_parser(nm)
        p.add_argument("bg")
        p.add_argument("--masks", default=PROBE)
        p.add_argument("--frame", type=int, default=120)
        if nm in ("yaws", "preview"):
            p.add_argument("--pano", required=True)
            p.add_argument("--static", nargs="*", default=["chair_front", "pot"], help="static layers kept with the objects")
        if nm == "static":
            p.add_argument("--name", required=True)
            p.add_argument("--pick", type=float, nargs=2, required=True)
            p.add_argument("--prompts", default="potted plant,flower pot,yellow flowers")
            p.add_argument("--back-m", type=float, default=0.2)
            p.add_argument("--half-m", type=float, default=0.3)
            p.add_argument("--height-m", type=float, default=0.85)
        if nm == "yaws":
            p.add_argument("--cam", type=int, default=REF)
        if nm == "preview":
            p.add_argument("--yaw", type=float, default=0.0)
            p.add_argument("--cams", type=int, nargs="*", default=[0, 5, 9, 17, 21])
            p.add_argument("--tag", default="")
            p.add_argument("--shadow", type=float, default=SHADOW)
            p.add_argument("--hull-lo", type=float, default=HULL_LO)
            p.add_argument("--gamma", type=float, default=SHADOW_GAMMA)
    p = sp.add_parser("comp")
    p.add_argument("bg")
    p.add_argument("--pano", required=True)
    p.add_argument("--yaw", type=float, required=True)
    p.add_argument("--fg", required=True, help="export fg dir: camNNNN/<src frame>.png")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--no-video", action="store_true")
    p = sp.add_parser("tonemap")
    p.add_argument("hdr")
    p.add_argument("ref", help="the panorama's own tonemapped JPG")
    p.add_argument("out")
    a = ap.parse_args()
    SHADOW, HULL_LO = getattr(a, "shadow", SHADOW), getattr(a, "hull_lo", HULL_LO)
    SHADOW_GAMMA = getattr(a, "gamma", SHADOW_GAMMA)
    {"geom": cmd_geom, "yaws": cmd_yaws, "preview": cmd_preview, "static": cmd_static, "tonemap": cmd_tonemap,
     "comp": cmd_comp}[a.cmd](a)


if __name__ == "__main__":
    main()
