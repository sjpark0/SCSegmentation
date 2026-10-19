#!/usr/bin/env python3
"""Library 'cart' take (NAS GSProject/GSData/library_261014/cart, 24 cameras in two arcs of 12, 1920x1080, frames
0..2699) for the SCSegmentation pipeline (user request 2026-10-20: segment the SCSegGaussians objects -- five people and
the book cart -- plus the table and any book a person is reading, starting at the first frame where every object is in
view, reviewed by the user before the run).  docs/cart.md.

    stage   PNG -> JPEG (quality 95, native size) into <root>/src/camNNNN/<f:06d>.jpg, threaded, resumable
    scene   <root>/<key>/Video/camNNNN/<i:06d>.jpg -> src frame start+i (relative symlinks), SOURCE.json
Host python (cv2):  PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python cart/build_cart.py stage
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "PlenopticServer2", "GSProject", "GSData", "library_261014", "cart", "png")
ROOT = os.path.join(REPO, "Data", "Cart")
NCAM, NFRAME, QUALITY = 24, 2700, 95


def cams():
    return [f"cam{c:04d}" for c in range(NCAM)]


def stage(root, threads):
    import cv2
    jobs = []
    for cam in cams():
        d = os.path.join(root, "src", cam)
        os.makedirs(d, exist_ok=True)
        jobs += [(os.path.join(SRC, cam, f"{f:06d}.png"), os.path.join(d, f"{f:06d}.jpg")) for f in range(NFRAME)]

    def one(job):
        s, t = job
        if os.path.isfile(t):
            return 0
        im = cv2.imread(s, cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (1080, 1920):
            raise IOError(f"bad frame {s}")
        tmp = t + ".part.jpg"
        if not cv2.imwrite(tmp, im, [cv2.IMWRITE_JPEG_QUALITY, QUALITY]):
            raise IOError(f"write failed {tmp}")
        os.replace(tmp, t)
        return 1
    done = 0
    with ThreadPoolExecutor(threads) as ex:
        for i, n in enumerate(ex.map(one, jobs, chunksize=16)):
            done += n
            if i % 5000 == 0:
                print(f"stage {i}/{len(jobs)} ({done} converted)", flush=True)
    print(f"stage done: {len(jobs)} frames, {done} converted this pass", flush=True)


def scene(root, key, start, num):
    sd = os.path.join(root, key)
    for cam in cams():
        vd = os.path.join(sd, "Video", cam)
        os.makedirs(vd, exist_ok=True)
        for i in range(num):
            dst = os.path.join(vd, f"{i:06d}.jpg")
            if not os.path.lexists(dst):
                os.symlink(os.path.relpath(os.path.join(root, "src", cam, f"{start + i:06d}.jpg"), vd), dst)
    json.dump(dict(scene=key, frames=SRC, file_index_to_source_frame=f"i -> {start} + i", start_source_frame=start,
                   num_frame=num, cameras=cams(), resolution=[1920, 1080], jpeg_quality=QUALITY,
                   rig="two arcs of 12 (cam0000..0011 lower, cam0012..0023 upper), COLMAP cameras.txt/images.txt "
                       "beside the PNGs"),
              open(os.path.join(sd, "SOURCE.json"), "w"), indent=1)
    print(f"{key}: {num} frames x {NCAM} cams linked (source {start}..{start + num - 1})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("stage", "scene"))
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--key")
    ap.add_argument("--start", type=int)
    ap.add_argument("--num", type=int)
    a = ap.parse_args()
    if a.cmd == "stage":
        stage(a.root, a.threads)
    else:
        scene(a.root, a.key, a.start, a.num)


if __name__ == "__main__":
    main()
