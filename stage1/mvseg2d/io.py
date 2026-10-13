"""Reading and writing the benchmark's files: meta.json, label maps, binary prediction PNGs.

Conventions (spec 2.1-2.2): a label map is an 8-bit grey PNG whose pixel value is the object id
(0 = background).  A prediction is an 8-bit grey PNG with values exactly 0 or 255, at the image's
resolution.  Nothing here resizes.
"""
import json
import os
import re

import cv2
import numpy as np

HEMISPHERE = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder")
SIZE_BINS = ("<2000", "2000-10000", ">=10000")


class BadMaskValues(ValueError):
    """A prediction PNG holds values other than 0 and 255."""


class ShapeMismatch(ValueError):
    """A prediction PNG's shape differs from the ground truth's."""


def natural_key(name):
    """Sort key that orders the digit runs of a camera name numerically: v2 < v10, S1_CAM_2 < S1_CAM_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def camera_order(cams):
    """The rig's camera order.  meta.json lists cams lexicographically (v0, v1, v10, ..., v2), which is not the
    physical order in six scenes (Barn, Breakfast, Frog, MATF, MartialArts, Painter); every use of camera
    *position* (view distance, pseudo-video chain) goes through this."""
    return sorted(cams, key=natural_key)


def size_bin(area_px):
    if area_px < 2000:
        return SIZE_BINS[0]
    if area_px < 10000:
        return SIZE_BINS[1]
    return SIZE_BINS[2]


def rig_of(scene):
    return "hemisphere" if scene in HEMISPHERE else "pinhole"


def scene_dir(data, scene):
    return os.path.join(data, scene)


def result_dir(data, scene, method):
    return os.path.join(data, scene, "Result", method)


def cache_dir(data, scene, provider):
    return os.path.join(data, scene, "Cache", provider)


def load_meta(data, scene):
    with open(os.path.join(data, scene, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def load_labels(data, scene):
    p = os.path.join(data, scene, "objects_labels.json")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def _read_grey(path, strict=False):
    """8-bit grey array.  A colour image is accepted only when its colour channels agree (a grey
    image saved as RGB); with `strict`, anything else raises BadMaskValues instead of silently
    taking channel 0.  An alpha channel is ignored.  Palette PNGs are expanded to colour by cv2,
    so a coloured palette also fails the channel check."""
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    if a.ndim == 3:
        rgb = a[..., :3]
        if strict and not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
            raise BadMaskValues(f"{path}: coloured image; a mask must be grey (equal channels) with values 0/255")
        a = rgb[..., 0]
    return a


def load_labelmap(path):
    return _read_grey(path).astype(np.uint8)


def labelmap_objects(lab):
    ids = np.unique(lab).tolist()
    return sorted(int(i) for i in ids if i != 0)


def labelmap_to_mask(lab, obj_id):
    return lab == obj_id


def read_pred_mask(path):
    a = _read_grey(path, strict=True)
    bad = np.unique(a[(a != 0) & (a != 255)])
    if bad.size:
        raise BadMaskValues(f"{path}: values {bad[:5].tolist()} are not 0/255")
    return a == 255


def write_mask(path, mask_bool):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = cv2.imwrite(path, np.asarray(mask_bool, dtype=np.uint8) * 255)
    if not ok:
        raise IOError(f"could not write {path}")
