"""Cache locations for training (spec 4.1) and what is still missing."""
import os

from methods.common import load_scene
from mvseg2d.io import load_meta


def feat_cache_path(data, scene, t, cam):
    return os.path.join(data, scene, "Cache", "sam3_feats", f"{int(t):06d}", f"{cam}.pt")


def pos_enc_path(data, scene):
    """The shared positional encodings of a scene's cached views (identical for every 1008x1008 input)."""
    return os.path.join(data, scene, "Cache", "sam3_feats", "pos_enc.pt")


def level72_path(data, scene, t):
    """GeoFuse spec 2.5: every camera's 72x72 backbone level of one frame, stacked ({"cams", "feats" fp16 [N,256,72,72]})."""
    return os.path.join(data, scene, "Cache", "sam3_feats72", f"{int(t):06d}.pt")


def pi3_frame_cache(data, scene, t):
    return os.path.join(data, scene, "Cache", "pi3_frames", f"{int(t):06d}")


def train_frames(data, scene):
    meta = load_meta(data, scene)
    if "train_frames" not in meta:
        raise RuntimeError(f"{scene}: meta.json has no train_frames; run prepare_data.py --train-frames")
    return [int(t) for t in meta["train_frames"]]


def missing_feats(data, scene, all_views=False, frame_stride=1):
    """(t, cam) whose backbone-feature cache does not exist yet: the annotated cameras of every labelled
    frame, or with `all_views` every camera of every `frame_stride`-th labelled frame (GeoQuery spec 3)."""
    meta = load_meta(data, scene)
    cams = meta["cams"] if all_views else meta["mask_cams"]
    frames = train_frames(data, scene)[::max(1, int(frame_stride))]
    return [(t, c) for t in frames for c in cams if not os.path.isfile(feat_cache_path(data, scene, t, c))]


def missing_pi3_frames(data, scene):
    return [t for t in train_frames(data, scene)
            if not os.path.isfile(os.path.join(pi3_frame_cache(data, scene, t), "MANIFEST.json"))]
