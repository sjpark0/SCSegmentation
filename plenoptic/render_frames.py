#!/usr/bin/env python3
"""Per-frame overlay images of the full-length CROOM T06 runs at the native 3840x2160 (user request 2026-10-20: "원본
사이즈로 … 나스에 저장"), straight from the NAS: source JPEGs CROOM/JPG/T06_flag_ganggang/<PERMS[K]>/<t>.jpg and the
masks SegResult/T06_flag_ganggang_F/<method>/cam_KK/<t>/<o>.png, written to
SegResult/T06_flag_ganggang_F/viz/<method>/frames/cam_KK/<t:06d>.jpg (JPEG quality 90; the same tint as the overlay
videos of plenoptic/render_full.py).  Both methods in one pass, so each source frame is read once.  Resumable (an
existing output is skipped).  Host python (stage1/.venv: cv2, numpy).

    render_frames.py [--workers 8] [--cams 0-31] [--first 0] [--last 9937]
"""
import argparse
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JPG = os.path.join(REPO, "PlenopticServer1", "CROOM", "JPG", "T06_flag_ganggang")
RES = os.path.join(REPO, "PlenopticServer1", "CROOM", "SegResult", "T06_flag_ganggang_F")
METHODS = ["SegMaskSam3XW1CGPS4MFsCs10kG2", "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs"]
PERMS = list(range(8)) + list(range(16, 32)) + list(range(8, 16))       # ring order cam_KK -> JPG folder (build_dataset)
COLS = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180]],
                np.float32)[:, ::-1]                                       # BGR, as plenoptic/render_full.py
A = None


def job(args):
    k, frames = args
    cam = f"cam_{k:02d}"
    outs = {m: os.path.join(RES, "viz", m, "frames", cam) for m in METHODS}
    for d in outs.values():
        os.makedirs(d, exist_ok=True)
    n = 0
    for t in frames:
        dst = {m: os.path.join(outs[m], f"{t:06d}.jpg") for m in METHODS}
        todo = [m for m in METHODS if not os.path.exists(dst[m])]
        if not todo:
            continue
        img = cv2.imread(os.path.join(JPG, str(PERMS[k]), f"{t}.jpg"))
        if img is None:
            raise IOError(f"missing source {cam} {t}")
        base = img.astype(np.float32)
        for m in todo:
            out = base.copy()
            d = os.path.join(RES, m, cam, str(t))
            for o in range(1, 7):
                p = os.path.join(d, f"{o}.png")
                if os.path.exists(p):
                    mk = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
                    out[mk] = 0.45 * out[mk] + 0.55 * COLS[o]
            tmp = dst[m] + ".part.jpg"
            if not cv2.imwrite(tmp, out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise IOError(f"write failed {tmp}")
            os.replace(tmp, dst[m])
            n += 1
    return k, frames[0], n


def main():
    global A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cams", default="0-31")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=9937)
    ap.add_argument("--chunk", type=int, default=250)
    A = ap.parse_args()
    lo, hi = map(int, A.cams.split("-"))
    jobs = [(k, list(range(f, min(A.last + 1, f + A.chunk)))) for f in range(A.first, A.last + 1, A.chunk)
            for k in range(lo, hi + 1)]
    t0, done, written = time.time(), 0, 0
    with Pool(A.workers) as pool:
        for k, f0, n in pool.imap_unordered(job, jobs):
            done += 1
            written += n
            print(f"{time.strftime('%F %T')} job {done}/{len(jobs)} cam_{k:02d} from {f0}: {n} written "
                  f"(total {written}, {written / max(1e-9, time.time() - t0):.1f}/s)", flush=True)
    print(f"done: {written} images written", flush=True)


if __name__ == "__main__":
    main()
