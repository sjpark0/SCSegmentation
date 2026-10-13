#!/usr/bin/env python3
"""Cache the SAM 3 backbone features of the annotated cameras for every training frame (spec 4.1).

    scripts/docker_run.sh scsam3 python -m geomem.cache_feats --scene <name|all> [--data Data] [--all-views] [--frame-stride k]

Each Data/<scene>/Cache/sam3_feats/<t>/<cam>.pt holds {"image": fp16 [3,1008,1008], "backbone_fpn":
[fp16 maps], "hw": (h, w)}.  The positional encodings depend only on the 1008x1008 input size, so
they are identical for every view and stored once per scene in Cache/sam3_feats/pos_enc.pt
(verified equal to the current view's at save time).  Existing files are skipped.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.cache import feat_cache_path, level72_path, missing_feats, pos_enc_path, train_frames  # noqa: E402
from methods import common  # noqa: E402
from mvseg2d.io import load_meta  # noqa: E402


def cache_view(model, path, img_path, pos_enc_path):
    import torch
    from geomem import sam3_io
    image, h, w = sam3_io.load_image(img_path)
    bb = sam3_io.backbone_features(model, image)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pos = [p.half().cpu() for p in bb["vision_pos_enc"]]
    if os.path.isfile(pos_enc_path):
        ref = torch.load(pos_enc_path)
        assert all(torch.equal(a, b) for a, b in zip(ref, pos)), "positional encodings differ between views"
    else:
        os.makedirs(os.path.dirname(pos_enc_path), exist_ok=True)
        torch.save(pos, pos_enc_path)
    torch.save({"image": image[0].half().cpu(),
                "backbone_fpn": [f.half().cpu() for f in bb["backbone_fpn"]],
                "hw": (h, w)}, path)


def build_level72(data, scene, t):
    """Stack every camera's cached `backbone_fpn[2]` (fp16 [256,72,72]) of frame t into Cache/sam3_feats72/<t>.pt
    (GeoFuse spec 2.5).  No backbone is recomputed; a missing view cache raises FileNotFoundError naming it."""
    import torch
    cams = list(load_meta(data, scene)["cams"])
    feats = []
    for cam in cams:
        path = feat_cache_path(data, scene, t, cam)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no cached backbone features for {cam}: {path} (run geomem.cache_feats --all-views)")
        feats.append(torch.load(path, map_location="cpu")["backbone_fpn"][2][0].half())
    out = level72_path(data, scene, t)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"cams": cams, "feats": torch.stack(feats)}, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default=os.path.join(common.REPO, "Data"))
    ap.add_argument("--all-views", action="store_true", help="every camera (GeoQuery hub), not only the annotated ones")
    ap.add_argument("--frame-stride", type=int, default=1, help="only every k-th labelled frame")
    ap.add_argument("--level72", action="store_true",
                    help="only build Cache/sam3_feats72/<t>.pt from the existing view caches (GeoFuse); no GPU")
    a = ap.parse_args()
    from methods import sam3_single
    model = None
    scenes = common.all_scenes(a.data) if a.scene == "all" else [a.scene]
    if a.level72:
        for name in scenes:
            t0, n = time.time(), 0
            for t in train_frames(a.data, name)[::max(1, a.frame_stride)]:
                if not os.path.isfile(level72_path(a.data, name, t)):
                    build_level72(a.data, name, t)
                    n += 1
            print(f"{name:20s} level72 {n} frames in {time.time() - t0:.0f}s", flush=True)
        return
    for name in scenes:
        todo = missing_feats(a.data, name, all_views=a.all_views, frame_stride=a.frame_stride)
        if not todo:
            print(f"{name:20s} cached", flush=True)
            continue
        if model is None:
            model = sam3_single.load_model()
        t0 = time.time()
        for t, cam in todo:
            scene = common.load_scene(a.data, name, frame=t)
            cache_view(model, feat_cache_path(a.data, name, t, cam), scene.image_paths[cam],
                       pos_enc_path(a.data, name))
        print(f"{name:20s} {len(todo)} views in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
