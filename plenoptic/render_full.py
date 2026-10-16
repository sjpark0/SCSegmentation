#!/usr/bin/env python3
"""Viewing products of a full-length CROOM run (docs/plenoptic-F.md): per-camera overlay MP4s, an 8x4 ring mosaic MP4,
a camera x time contact sheet and an area timeline (area / frame-900 seed area per camera and person).  Host python
(stage1/.venv: cv2, numpy) + the host ffmpeg.

    render_full.py <scene_dir> <method> <out_dir> [--first 0] [--last 9937] [--workers 24]

Frames: <scene_dir>/Video/<cam>/<t:06d>.jpg decoded at 1/4 scale (960x540); masks: <scene_dir>/<method>/<cam>/<t>/<obj>.png
(the merged forward + backward tree, folder = source frame).  Areas: MANIFEST.json (forward, source frames 900..) and
MANIFEST_backward.json (backward index i = source frame 900 - i).
"""
import argparse
import json
import os
import subprocess
import time
from multiprocessing import Pool

import cv2
import numpy as np

COLS = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180]],
                np.float32)[:, ::-1]                                              # BGR, as plenoptic/make_figs.py
FPS, W4, H4 = 59.94, 960, 540
A = None


def ffmpeg(path, w, h, crf=20):
    return subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)


def overlay(cam, t):
    img = cv2.imread(os.path.join(A.scene, "Video", cam, f"{t:06d}.jpg"), cv2.IMREAD_REDUCED_COLOR_4)
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    d = os.path.join(A.scene, A.method, cam, str(t))
    for o in range(1, 7):
        p = os.path.join(d, f"{o}.png")
        if not os.path.exists(p):
            continue
        m = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST) > 127
        out[m] = 0.45 * out[m] + 0.55 * COLS[o]
    out = out.astype(np.uint8)
    cv2.putText(out, f"{cam}  frame {t}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def one_cam(cam):
    t0 = time.time()
    p = ffmpeg(os.path.join(A.out, "cams", f"{cam}.mp4"), W4, H4)
    small = []
    for t in range(A.first, A.last + 1):
        f = overlay(cam, t)
        p.stdin.write(f.tobytes())
        if (t - A.first) % A.sheet_every == 0:
            small.append(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA))
    p.stdin.close()
    p.wait()
    np.save(os.path.join(A.out, "cams", f"{cam}_sheet.npy"), np.stack(small))
    return cam, round(time.time() - t0, 1)


def mosaic(cams):
    caps = [cv2.VideoCapture(os.path.join(A.out, "cams", f"{c}.mp4")) for c in cams]
    p = ffmpeg(os.path.join(A.out, "ring_mosaic.mp4"), 8 * 480, 4 * 270, crf=23)
    for t in range(A.first, A.last + 1):
        tiles = []
        for cap in caps:
            ok, f = cap.read()
            tiles.append(cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA) if ok else np.zeros((270, 480, 3), np.uint8))
        g = np.vstack([np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(4)])
        cv2.putText(g, f"source frame {t}  ({t / FPS:.1f} s; prompt at frame 900)", (10, 4 * 270 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        p.stdin.write(g.tobytes())
    p.stdin.close()
    p.wait()


def sheet(cams):
    sm = [np.load(os.path.join(A.out, "cams", f"{c}_sheet.npy")) for c in cams]
    n = min(s.shape[0] for s in sm)
    cv2.imwrite(os.path.join(A.out, "contact_sheet.jpg"), np.vstack([np.hstack([s[i] for i in range(n)]) for s in sm]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(dict(rows=cams, cols=[A.first + i * A.sheet_every for i in range(n)]),
              open(os.path.join(A.out, "contact_sheet.json"), "w"))


def timeline(cams):
    """rows = (view, person), columns = source frames; colour = area / seed area (0 black .. 1.5 white)."""
    mdir = os.path.join(A.scene, A.method)
    fwd = json.load(open(os.path.join(mdir, "MANIFEST.json")))["provenance"]["view_areas"]
    bp = os.path.join(mdir, "MANIFEST_backward.json")
    bwd = json.load(open(bp))["provenance"]["view_areas"] if os.path.isfile(bp) else None
    seed, tr = fwd["seed"], fwd["tracked"]
    frames = list(range(A.first, A.last + 1))
    rows, labels = [], []
    for v in sorted(tr, key=int):
        for o in range(1, 7):
            a0 = seed.get(v, {}).get(str(o), 0)
            r = []
            for t in frames:
                if t >= 900:
                    a = tr[v].get(str(t), {}).get(str(o), 0)
                else:
                    a = (bwd["tracked"].get(v, {}).get(str(900 - t), {}).get(str(o), 0)) if bwd else 0
                r.append(a / a0 if a0 else 0.0)
            rows.append(r)
            labels.append(f"v{v} p{o}")
    img = (np.clip(np.array(rows, np.float32), 0, 1.5) / 1.5 * 255).astype(np.uint8)
    img = cv2.resize(img, (min(4000, len(frames)), len(rows) * 4), interpolation=cv2.INTER_AREA)
    img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    x900 = int((900 - A.first) / max(1, len(frames) - 1) * (img.shape[1] - 1))
    cv2.line(img, (x900, 0), (x900, img.shape[0] - 1), (255, 255, 255), 1)
    cv2.imwrite(os.path.join(A.out, "area_timeline.png"), img)
    json.dump(dict(rows=labels, first=A.first, last=A.last, scale="area / frame-900 seed area, clipped at 1.5",
                   prompt_frame=900), open(os.path.join(A.out, "area_timeline.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("method")
    ap.add_argument("out")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=9937)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sheet-every", type=int, default=600)
    ap.add_argument("--what", default="cams,mosaic,sheet,timeline")
    A = ap.parse_args()
    os.makedirs(os.path.join(A.out, "cams"), exist_ok=True)
    what = A.what.split(",")
    cams = sorted(c for c in os.listdir(os.path.join(A.scene, A.method)) if c.startswith("cam_"))
    rec, t0 = {}, time.time()
    if "cams" in what:
        with Pool(A.workers) as pool:
            rec["cams"] = pool.map(one_cam, cams)
        rec["cams_seconds"] = round(time.time() - t0, 1)
    for name, fn in (("mosaic", mosaic), ("sheet", sheet), ("timeline", timeline)):
        if name in what:
            t1 = time.time()
            fn(cams)
            rec[f"{name}_seconds"] = round(time.time() - t1, 1)
    json.dump(rec, open(os.path.join(A.out, "render_record.json"), "w"), indent=1)
    print(json.dumps(rec)[:2000])
