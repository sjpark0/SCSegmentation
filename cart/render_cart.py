#!/usr/bin/env python3
"""Overlay videos of a cart run (docs/cart.md) from the exported label PNGs (export_cart.py), 30 fps H.264:

    render_cart.py <scene dir> <export dir> <out dir> [--start 120] [--workers 12]

  <out>/cams/camNNNN.mp4   1920x1080, every source frame, objects tinted in fixed colours with outlines + legend
  <out>/mosaic.mp4         all 24 cameras in ring order as a 6x4 grid of 640x360 tiles + legend (3840x1484)
Frames: <scene>/Video/camNNNN/<i:06d>.jpg (i = source frame - start).  ffmpeg: $FFMPEG, PATH, or the static build in
Data/PlenopticFull/tools/ffmpeg or Data/Cart/tools/ffmpeg.  Host python (stage1/.venv: cv2, numpy).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_cart as R  # noqa: E402  (colours, legend)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RING = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
A = None
LUT = np.zeros((256, 3), np.float32)
for _o, _c in R.COLS.items():
    LUT[_o] = _c[::-1]


def ffmpeg_exe():
    for c in (os.environ.get("FFMPEG"), shutil.which("ffmpeg"), os.path.join(REPO, "Data", "Cart", "tools", "ffmpeg"),
              os.path.join(REPO, "Data", "PlenopticFull", "tools", "ffmpeg")):
        if c and os.access(c, os.X_OK):
            return c
    raise SystemExit("no ffmpeg")


def writer(path, w, h, crf=20):
    return subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                             f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)


def frame(cam, i, alpha=0.45):
    img = cv2.imread(os.path.join(A.scene, "Video", cam, f"{i:06d}.jpg"))
    lab = cv2.imread(os.path.join(A.export, "labels", cam, f"{A.start + i:06d}.png"), cv2.IMREAD_GRAYSCALE)
    out = img.astype(np.float32)
    m = lab > 0
    out[m] = (1 - alpha) * out[m] + alpha * LUT[lab[m]]
    k = np.ones((3, 3), np.uint8)
    edge = (cv2.dilate(lab, k) != cv2.erode(lab, k)) & (cv2.dilate(lab, k) > 0)
    out[edge] = LUT[cv2.dilate(lab, k)[edge]]
    return out.astype(np.uint8)


def one_cam(cam):
    leg = R.legend(A.labels, 1920)
    p = writer(os.path.join(A.out, "cams", f"{cam}.mp4"), 1920, 1080 + leg.shape[0])
    for i in range(A.n):
        f = frame(cam, i)
        cv2.putText(f, f"{cam}  source frame {A.start + i}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        p.stdin.write(np.vstack([f, leg]).tobytes())
    p.stdin.close()
    p.wait()
    return cam


def mosaic_part(args):
    lo, hi = args
    leg = R.legend(A.labels, 6 * 640)
    p = writer(os.path.join(A.out, f".mosaic_{lo:06d}.mp4"), 6 * 640, 4 * 360 + leg.shape[0], crf=23)
    for i in range(lo, hi):
        tiles = []
        for c in RING:
            cam = f"cam{c:04d}"
            t = cv2.resize(frame(cam, i), (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(t, cam, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(t)
        g = np.vstack([np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(4)])
        cv2.putText(g, f"source frame {A.start + i}", (10, 4 * 360 - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        p.stdin.write(np.vstack([g, leg]).tobytes())
    p.stdin.close()
    p.wait()
    return lo


def main():
    global A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("export")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--what", default="cams,mosaic")
    A = ap.parse_args()
    A.labels = json.load(open(os.path.join(A.scene, "Mask", "objects_labels.json")))
    A.n = len(os.listdir(os.path.join(A.export, "labels", "cam0005")))
    os.makedirs(os.path.join(A.out, "cams"), exist_ok=True)
    print("ffmpeg:", ffmpeg_exe(), "frames:", A.n, flush=True)
    with Pool(A.workers) as pool:
        if "cams" in A.what:
            pool.map(one_cam, [f"cam{c:04d}" for c in range(24)])
        if "mosaic" in A.what:
            step = max(1, -(-A.n // A.workers))
            parts = sorted(pool.map(mosaic_part, [(lo, min(A.n, lo + step)) for lo in range(0, A.n, step)]))
            lst = os.path.join(A.out, ".mosaic_parts.txt")
            with open(lst, "w") as f:
                f.writelines(f"file '{os.path.join(A.out, f'.mosaic_{lo:06d}.mp4')}'\n" for lo in parts)
            subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                            os.path.join(A.out, "mosaic.mp4")], check=True)
            for lo in parts:
                os.remove(os.path.join(A.out, f".mosaic_{lo:06d}.mp4"))
            os.remove(lst)
    print("done", A.out, flush=True)


if __name__ == "__main__":
    main()
