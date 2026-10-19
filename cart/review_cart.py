#!/usr/bin/env python3
"""Review sheets for one frame of a cart run (docs/cart.md): the reference camera at full size and a 6x4 mosaic of all
24 cameras (ring order), each object id in a fixed colour with a legend, plus a table of (camera, object) pairs with no
mask.  Host python (stage1/.venv: cv2, numpy).

    review_cart.py <scene dir> <method> <frame index> <out dir> [--ref cam0005]
"""
import argparse
import json
import os

import cv2
import numpy as np

COLS = {1: (230, 25, 75), 2: (60, 180, 75), 3: (255, 225, 25), 4: (0, 130, 200), 5: (245, 130, 48), 6: (145, 30, 180),
        7: (70, 240, 240), 8: (240, 50, 230), 9: (210, 245, 60), 10: (250, 190, 212), 11: (0, 128, 128),
        12: (170, 110, 40), 13: (255, 250, 200), 14: (128, 0, 0), 15: (170, 255, 195), 16: (128, 128, 0),
        17: (255, 215, 180), 18: (0, 0, 128), 19: (128, 128, 128), 20: (255, 255, 255)}


def paint(img, cam_dir, t, labels, alpha=0.5, outline=True):
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    present = []
    for o in sorted(labels, key=int):
        p = os.path.join(cam_dir, str(t), f"{o}.png")
        if not os.path.exists(p):
            continue
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = m > 127
        if not m.any():
            continue
        present.append(int(o))
        c = np.array(COLS[int(o)][::-1], np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * c
        if outline:
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnt, -1, tuple(float(v) for v in COLS[int(o)][::-1]), 2)
    return out.astype(np.uint8), present


def legend(labels, width):
    bar = np.full((44, width, 3), 30, np.uint8)
    x = 10
    for o in sorted(labels, key=int):
        txt = f"{o} {labels[o]['label']}"
        cv2.rectangle(bar, (x, 10), (x + 24, 34), COLS[int(o)][::-1], -1)
        cv2.putText(bar, txt, (x + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        x += 40 + 14 * len(txt)
    return bar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("method")
    ap.add_argument("frame", type=int)
    ap.add_argument("out")
    ap.add_argument("--ref", default="cam0005")
    ap.add_argument("--order", default="0,1,2,3,4,5,6,7,8,9,10,11,23,22,21,20,19,18,17,16,15,14,13,12")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    labels = json.load(open(os.path.join(a.scene, "Mask", "objects_labels.json")))
    cams = [f"cam{int(c):04d}" for c in a.order.split(",")]
    tiles, missing = [], {}
    for cam in cams:
        img = cv2.imread(os.path.join(a.scene, "Video", cam, f"{a.frame:06d}.jpg"))
        ov, present = paint(img, os.path.join(a.scene, a.method, cam), a.frame, labels)
        missing[cam] = [int(o) for o in labels if int(o) not in present]
        if cam == a.ref:
            cv2.imwrite(os.path.join(a.out, f"ref_{cam}{a.tag}.jpg"), np.vstack([ov, legend(labels, ov.shape[1])]),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        t = cv2.resize(ov, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(t, cam + (" (ref)" if cam == a.ref else ""), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if missing[cam]:
            cv2.putText(t, "no mask: " + ",".join(map(str, missing[cam])), (8, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        tiles.append(t)
    grid = np.vstack([np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(4)])
    cv2.imwrite(os.path.join(a.out, f"mosaic{a.tag}.jpg"), np.vstack([grid, legend(labels, grid.shape[1])]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(dict(frame=a.frame, missing=missing), open(os.path.join(a.out, f"missing{a.tag}.json"), "w"), indent=1)
    print(json.dumps({c: m for c, m in missing.items() if m}))


if __name__ == "__main__":
    main()
