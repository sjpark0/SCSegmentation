"""Contact sheet of one annotated instant: every camera (preview size) with the label map overlaid and object ids written
at each object's centroid.  Ground-truth only (no predictions).

    python stage1/annot/contact_sheet.py --root <CROOM root> --take T03_walking_rsp_out --frame 4181 --out <jpg> [--cols 8]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annot.labels import read_label  # noqa: E402


def sheet(root, take, frame, cols=8, tile_w=480, alpha=0.45, only=None):
    d = os.path.join(root, take, f"{int(frame):06d}")
    objs = json.load(open(os.path.join(d, "annot", "objects.json")))
    cams = [f[:-4] for f in os.listdir(os.path.join(d, "images")) if f.endswith(".jpg")]
    cams.sort(key=lambda c: (0, int(c)) if c.isdigit() else (1, c))
    if only is not None:
        cams = [c for c in cams if c in only]
    tiles, rows = [], []
    for c in cams:
        im = Image.open(os.path.join(d, "images", f"{c}.jpg")).convert("RGB")
        W, H = im.size
        s = tile_w / W
        im = im.resize((tile_w, round(H * s)), Image.BILINEAR)
        lp = os.path.join(d, "annot", "labels", f"{c}.png")
        ids = []
        if os.path.isfile(lp):
            lab = read_label(lp)
            lab_s = np.array(Image.fromarray(lab).resize(im.size, Image.NEAREST))
            arr = np.asarray(im).astype(np.float32)
            for oid in np.unique(lab_s):
                if not oid:
                    continue
                col = np.array(objs[str(oid)]["color"], np.float32)
                m = lab_s == oid
                arr[m] = arr[m] * (1 - alpha) + col * alpha
                ids.append(int(oid))
            im = Image.fromarray(arr.astype(np.uint8))
            dr = ImageDraw.Draw(im)
            for oid in ids:
                ys, xs = np.nonzero(lab_s == oid)
                cx, cy = int(xs.mean()), int(ys.mean())
                dr.rectangle([cx - 9, cy - 9, cx + 9, cy + 9], fill=(0, 0, 0))
                dr.text((cx - 4, cy - 6), str(oid), fill=(255, 255, 255))
            tag = f"{c}  {ids if ids else 'EMPTY'}"
        else:
            tag = f"{c}  (no label)"
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 8 + 7 * len(tag), 14], fill=(0, 0, 0))
        dr.text((4, 2), tag, fill=(255, 255, 0))
        tiles.append(im)
        rows.append((c, ids))
    tw, th = tiles[0].size
    n_rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * tw, n_rows * th), (40, 40, 40))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * tw, (i // cols) * th))
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--take", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--tile", type=int, default=480)
    ap.add_argument("--cams", default=None, help="comma list of camera names (default all)")
    a = ap.parse_args()
    out, rows = sheet(a.root, a.take, a.frame, a.cols, a.tile, only=a.cams.split(",") if a.cams else None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.save(a.out, quality=88)
    for c, ids in rows:
        print(c, ids)
    print("wrote", a.out, out.size)


if __name__ == "__main__":
    main()
