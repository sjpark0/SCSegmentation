#!/usr/bin/env python3
"""Collect the reference frame (image + GT mask) of every MVSeg dataset.

Source layout (../SCSegmentation/Data/MVSeg/<dataset>/):
    Video/<cam>/<frame>.jpg               all cameras, all frames
    Mask/<cam>/<frame>.png                GT, greyscale, pixel value = object id (3 annotated cams)
    Mask/objects_labels.json              object id -> {label, type}
    MVSeed_gt/SEED_MANIFEST.json          start_frame (reference frame) and c_ini (reference cam)

Output layout (./Data/<dataset>/):
    images/<cam>.jpg                      reference frame of every camera
    frames/<t>/{images,masks}/            (--train-frames) every labelled frame, same layout
    masks/<cam>.png                       GT mask of the reference frame (annotated cams only)
    objects_labels.json
    meta.json                             ref_frame, ref_cam, cams, mask_cams, image size
"""
import argparse, json, os, shutil
from PIL import Image

SRC_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "Data", "MVSeg")
DST_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")


def labelled_frames(mask_dir, mask_cams):
    """Frame indices that have a label map in the first annotated camera (sorted ints)."""
    names = os.listdir(os.path.join(mask_dir, mask_cams[0]))
    return sorted(int(os.path.splitext(f)[0]) for f in names if f.endswith(".png"))


def copy_frame(video, mask, cams, mask_cams, t, out_dir):
    """images of every camera and label maps of the annotated cameras for frame t -> out_dir/{images,masks}/."""
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    for c in cams:
        src = os.path.join(video, c, f"{t:06d}.jpg")
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        shutil.copy2(src, os.path.join(out_dir, "images", f"{c}.jpg"))
    for c in mask_cams:
        src = os.path.join(mask, c, f"{t:06d}.png")
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        shutil.copy2(src, os.path.join(out_dir, "masks", f"{c}.png"))


def prepare(ds_dir, out_dir, train_frames=False):
    name = os.path.basename(ds_dir)
    manifest = json.load(open(os.path.join(ds_dir, "MVSeed_gt", "SEED_MANIFEST.json")))
    ref = int(manifest["start_frame"])
    ref_cam = manifest["c_ini"]
    video = os.path.join(ds_dir, "Video")
    mask = os.path.join(ds_dir, "Mask")
    cams = sorted(c for c in os.listdir(video) if os.path.isdir(os.path.join(video, c)))
    mask_cams = sorted(c for c in os.listdir(mask) if os.path.isdir(os.path.join(mask, c)))

    img_out = os.path.join(out_dir, "images")
    msk_out = os.path.join(out_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    size = None
    for c in cams:
        src = os.path.join(video, c, f"{ref:06d}.jpg")
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        shutil.copy2(src, os.path.join(img_out, f"{c}.jpg"))
        if size is None:
            size = Image.open(src).size
    for c in mask_cams:
        src = os.path.join(mask, c, f"{ref:06d}.png")
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        shutil.copy2(src, os.path.join(msk_out, f"{c}.png"))
    shutil.copy2(os.path.join(mask, "objects_labels.json"),
                 os.path.join(out_dir, "objects_labels.json"))

    frames = None
    if train_frames:
        frames = labelled_frames(mask, mask_cams)
        for t in frames:
            copy_frame(video, mask, cams, mask_cams, t, os.path.join(out_dir, "frames", f"{t:06d}"))

    meta = {
        "dataset": name,
        "ref_frame": ref,
        "ref_cam": ref_cam,
        "cams": cams,
        "mask_cams": mask_cams,
        "image_size": list(size),
        "source": os.path.abspath(ds_dir),
    }
    if frames is not None:
        meta["train_frames"] = frames
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dst", default=DST_DEFAULT)
    ap.add_argument("--datasets", nargs="*", help="subset of dataset names (default: all with SEED_MANIFEST)")
    ap.add_argument("--train-frames", action="store_true",
                    help="also copy every labelled frame to Data/<scene>/frames/<t>/{images,masks} (training data)")
    a = ap.parse_args()

    names = a.datasets or sorted(
        d for d in os.listdir(a.src)
        if os.path.isfile(os.path.join(a.src, d, "MVSeed_gt", "SEED_MANIFEST.json")))
    for d in names:
        m = prepare(os.path.join(a.src, d), os.path.join(a.dst, d), train_frames=a.train_frames)
        print(f"{d:20s} ref_frame={m['ref_frame']:3d} ref_cam={m['ref_cam']:12s} "
              f"cams={len(m['cams']):2d} mask_cams={m['mask_cams']}")


if __name__ == "__main__":
    main()
