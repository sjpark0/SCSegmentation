"""Label-map I/O shared by the annotation server, check.py and the CROOM scene builder / scorer.

A label map is a 2-D uint8 PNG (0 background, id = object).  Until 2026-10-26 the server stored the browser's RGBA
canvas PNG as-is (R = G = B = id, A = the first saved id); `as_label` keeps channel 0 of such an array after checking
that the colour channels agree, and annot/fix_rgba.py rewrote the stored files (docs/croom-annotation.md 5).
"""
import hashlib

import numpy as np
from PIL import Image


def as_label(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if not ((arr[..., 0] == arr[..., 1]).all() and (arr[..., 0] == arr[..., 2]).all()):
            raise ValueError("colour channels of the label PNG differ")
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def read_label(path):
    return as_label(np.array(Image.open(path)))


def is_2d_png(path):
    return Image.open(path).mode == "L"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
