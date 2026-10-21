#!/usr/bin/env python3
"""Per-frame label / foreground images and an SCSegGaussians-style label volume from a cart run (docs/cart.md).

    export_cart.py <scene dir> <method> <out dir> [--start 120] [--npy <dir>] [--static <objects.json>] [--workers 16]

  <out>/labels/camNNNN/<src frame:06d>.png   8-bit grey, pixel = object id (0 background, 1..17); where two objects'
                                              masks overlap, the later id in PAINT wins (the order the reference prompt
                                              was painted: table, cart, sign, laptops, chairs, people, books)
  <out>/fg/camNNNN/<src frame:06d>.png       8-bit grey 0/255, union of all objects (kept when the background is replaced)
  <npy>/labels.npy                            uint8 (frames, 24 cameras cam0000..cam0023, H, W): the same ids, and 255
                                              (ignore) on a BAND_PX band around every boundary between two labels
                                              (background included) -- the convention of SCSegGaussians'
                                              seg/labels/seg_labels_cart_f100_2699 (ignore 255, band_px 2)
  <npy>/frames.npy, <npy>/objects.json        source frame numbers; ids, names, sources, missing PNG counts
--static: objects that do not move and were segmented once, at source frame 120 (user 2026-10-21: the empty chair at
  the front; the chrysanthemum pot): {"<id>": {"label", "dir"}} with one 0/255 mask per camera in <dir>/camNNNN.png
  (dir relative to the JSON), used for every frame.  Painted after the run's objects and winning over them -- at frame
  120 they are in front of the table and the chair behind them -- except over MOVERS, the objects that walk or roll
  through the scene and can pass in front of them.
The per-object binary masks the pipeline writes (<method>/camNNNN/<i>/<id>.png, 0/255) stay as they are.
Host python (stage1/.venv: cv2, numpy).
"""
import argparse
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

PAINT = [7, 6, 13, 11, 12, 14, 15, 16, 17, 1, 2, 3, 4, 5, 8, 9, 10]
MOVERS = (1, 5, 6, 13)          # cart puller, curly hair (leaves at ~1200 and comes back), book cart, sign on the cart
IGNORE, BAND_PX = 255, 2
A = None


def load_static(path):
    """[(id, label, [24 boolean masks])] from a --static JSON"""
    if not path:
        return []
    spec = json.load(open(path))
    out = []
    for k in sorted(spec, key=int):
        d = os.path.join(os.path.dirname(os.path.abspath(path)), spec[k]["dir"])
        out.append((int(k), spec[k]["label"], [cv2.imread(os.path.join(d, f"cam{c:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127
                                                for c in range(24)]))
    return out


def label_map(src, i, statics=(), cam_i=None):
    lab, overlap, missing = None, 0, []
    for o in PAINT:
        p = os.path.join(src, str(i), f"{o}.png")
        if not os.path.exists(p):
            missing.append(o)
            continue
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
        if lab is None:
            lab = np.zeros(m.shape, np.uint8)
        overlap += int((m & (lab > 0)).sum())
        lab[m] = o
    if lab is None:
        lab = np.zeros((1080, 1920), np.uint8)
    for o, _, masks in statics:
        m = masks[cam_i] & ~np.isin(lab, MOVERS)
        overlap += int((m & (lab > 0)).sum())
        lab[m] = o
    return lab, overlap, missing


def banded(lab):
    k = np.ones((2 * BAND_PX + 1, 2 * BAND_PX + 1), np.uint8)
    out = lab.copy()
    out[cv2.dilate(lab, k) != cv2.erode(lab, k)] = IGNORE
    return out


def job(args):
    cam_i, cam, frames = args
    src = os.path.join(A.scene, A.method, cam)
    ld, fd = os.path.join(A.out, "labels", cam), os.path.join(A.out, "fg", cam)
    os.makedirs(ld, exist_ok=True)
    os.makedirs(fd, exist_ok=True)
    vol = np.load(os.path.join(A.npy, "labels.npy"), mmap_mode="r+") if A.npy else None
    st = {"overlap_px": 0, "fg_px": 0, "missing": {}}
    statics = load_static(A.static)
    for i in frames:
        lab, ov, miss = label_map(src, i, statics, cam_i)
        name = f"{A.start + i:06d}.png"
        cv2.imwrite(os.path.join(ld, name), lab)
        cv2.imwrite(os.path.join(fd, name), (lab > 0).astype(np.uint8) * 255)
        if vol is not None:
            vol[i, cam_i] = banded(lab)
        st["overlap_px"] += ov
        st["fg_px"] += int((lab > 0).sum())
        for o in miss:
            st["missing"][o] = st["missing"].get(o, 0) + 1
    if vol is not None:
        vol.flush()
    return cam, st


def main():
    global A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("method")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--npy")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--static", help="JSON of objects segmented once at frame 120 and kept for every frame")
    ap.add_argument("--limit", type=int, help="first N frames only (testing)")
    A = ap.parse_args()
    root = os.path.join(A.scene, A.method)
    cams = sorted(c for c in os.listdir(root) if c.startswith("cam"))
    nframe = len([t for t in os.listdir(os.path.join(root, cams[0])) if t.isdigit()])
    if A.limit:
        nframe = min(nframe, A.limit)
    if A.npy:
        os.makedirs(A.npy, exist_ok=True)
        np.lib.format.open_memmap(os.path.join(A.npy, "labels.npy"), mode="w+", dtype=np.uint8,
                                  shape=(nframe, len(cams), 1080, 1920)).flush()
        np.save(os.path.join(A.npy, "frames.npy"), np.arange(A.start, A.start + nframe, dtype=np.int64))
    jobs = [(k, c, list(range(f, min(nframe, f + A.chunk)))) for k, c in enumerate(cams) for f in range(0, nframe, A.chunk)]
    with Pool(A.workers) as pool:
        res = pool.map(job, jobs)
    per_cam = {}
    for cam, st in res:
        r = per_cam.setdefault(cam, {"overlap_px": 0, "fg_px": 0, "missing": {}})
        r["overlap_px"] += st["overlap_px"]
        r["fg_px"] += st["fg_px"]
        for o, n in st["missing"].items():
            r["missing"][o] = r["missing"].get(o, 0) + n
    labels = json.load(open(os.path.join(A.scene, "Mask", "objects_labels.json")))
    objects = {k: v["label"] for k, v in labels.items()}
    static = json.load(open(A.static)) if A.static else {}
    objects.update({k: v["label"] for k, v in static.items()})
    miss = {str(o): sum(r["missing"].get(o, 0) for r in per_cam.values()) for o in sorted(PAINT)}
    meta = dict(num_objects=len(objects), objects=objects,
                resolution_hw=[1080, 1920], ignore=IGNORE, band_px=BAND_PX, frame_range=[A.start, A.start + nframe - 1],
                cams=cams, axes="labels[frame index, camera index, y, x]; frame index i = source frame frames[i]",
                paint_order=PAINT + [int(k) for k in sorted(static, key=int)], static_objects=static,
                static_rule=f"static objects (one mask per camera, every frame) painted last, except over {list(MOVERS)}",
                sources=[os.path.abspath(root)] + ([os.path.abspath(A.static)] if A.static else []),
                padded_background_frames=None,
                missing_png_cam_frames=miss, note="docs/cart.md; per-frame PNGs without the ignore band in <out>/labels")
    json.dump(dict(meta, per_cam=per_cam), open(os.path.join(A.out, "export.json"), "w"), indent=1)
    if A.npy:
        json.dump(meta, open(os.path.join(A.npy, "objects.json"), "w"), indent=1)
    print(f"{len(cams)} cams x {nframe} frames; missing PNGs per object {miss}")


if __name__ == "__main__":
    main()
