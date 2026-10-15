"""The standard stage-1 DEV set (docs/stage1-dev-protocol.md; adopted 2026-10-15 after S1-E13).

The scored keyframe (each scene's meta.json `ref_frame`, the 664 pairs of MVSeed/runs/seedfail.json) is the TEST set:
nothing is chosen or tuned on it.  Design and exploration use DEV frames of the same scenes:

    DEV-A  "explore"  the scene's train frames other than the ref frame, [3::4]  -> ref+4, +8, +12, +16, +20 (5 / scene)
    DEV-B  "confirm"  the remaining 15 non-ref train frames (ref+1, 2, 3, 5, 6, 7, 9, ..., 19), spent once per
                      registration to confirm a DEV-A finding before the keyframe gate

Views = (frame t, camera) for the 2 annotated non-reference cameras (meta mask_cams minus ref_cam); input = the reference
camera c_ini (meta ref_cam) at frame t with its GT restricted to the scene's basic object set (docs/raw/muvod_object_sets.json
seed_ids_c_ini); a pair = (scene, t, cam, obj) with the object present in c_ini at t and non-empty GT in cam (as the keyframe's
gt_px > 0 rule).  Every DEV scene is decoded with its own held-out fold checkpoint, so a DEV frame is never in the training
data of the model that decodes it.  Torch-free (host tests import it).
"""
import json
import os

from geomem.cache import train_frames
from mvseg2d.io import load_meta

SPLITS = ("A", "B")
OFFSET_A, STRIDE_A = 3, 4            # DEV-A = non-ref train frames [3::4]
OBJSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "docs", "raw", "muvod_object_sets.json")


def nonref_train_frames(data, scene):
    ref = int(load_meta(data, scene)["ref_frame"])
    return [int(t) for t in train_frames(data, scene) if int(t) != ref]


def dev_frames(data, scene, split="A"):
    assert split in SPLITS, split
    nonref = nonref_train_frames(data, scene)
    a = nonref[OFFSET_A::STRIDE_A]
    return a if split == "A" else [t for t in nonref if t not in a]


def eval_cams(data, scene):
    m = load_meta(data, scene)
    return [c for c in m["mask_cams"] if c != m["ref_cam"]]


def basic_set(scene, objsets=OBJSETS):
    with open(objsets) as f:
        return [int(o) for o in json.load(f)[scene]["seed_ids_c_ini"]]


def index(data, scenes, split="A"):
    """The DEV definition written next to every DEV output (index.json)."""
    out = {}
    for s in scenes:
        m = load_meta(data, s)
        out[s] = dict(ref_frame=int(m["ref_frame"]), ref_cam=m["ref_cam"], frames=dev_frames(data, s, split),
                      eval_cams=eval_cams(data, s), basic=basic_set(s))
    return dict(split=split, rule=("non-ref train frames [3::4]" if split == "A" else "non-ref train frames minus [3::4]"),
                scenes=out)
