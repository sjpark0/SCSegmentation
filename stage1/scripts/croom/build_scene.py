#!/usr/bin/env python3
"""Turn one annotated CROOM instant (annot/prep_croom.py layout, labels by annot/server.py) into a stage-1 scene folder
that methods/*/run.py decode like T06_flag_ganggang_F (docs/croom-annotation.md 6):

    python scripts/croom/build_scene.py --take T04_sidestep_out --frame 1000 --ref 0 [--fold 2] [--overwrite]

Data/CROOM_<T>_<frame:06d>_r<ref:02d>/
  images/cam_KK.jpg        relative symlinks to Data/_annot/CROOM/<take>/<frame>/images/<K>.jpg (input only)
  masks/cam_RR.png         the reference camera's label map (the only ground truth a method sees)
  meta.json                dataset, ref_frame, ref_cam, cams, mask_cams, image_size, basic_objects, external_fold,
                           source "croom_annot", annot {take, frame, ref_label_sha256, index_sha256, objects_sha256}
  objects_labels.json      {id: {"label": name, "type": "dynamic"}}
The other cameras' label maps stay in the annotation folder; scripts/croom/score.py reads them from there.
`basic_objects` = the objects present in the reference camera (an object absent there has no prompt and is not scored).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, S1)
from annot.labels import read_label, sha256  # noqa: E402

ANNOT_ROOT = os.path.join(S1, "Data", "_annot", "CROOM")


# Physical ring order of the T03/T04/T05/T07 camera numbering (input only, 2026-10-26): ORB + RANSAC homography inliers
# between every T06 camera (whose numbering follows the ring, S3-P) and every T03 #4181 / T04 #1000 camera give a one-to-one
# map: cams 0-7 -> T06 0-7, cams 16-31 -> T06 8-23, cams 8-15 -> T06 24-31.  So in these takes the index order is NOT the
# ring order; ring distance, pseudo-video order and the ring-shape diagnostics use this list (docs/croom-annotation.md 5).
RING_ORDER = [f"cam_{k:02d}" for k in list(range(0, 8)) + list(range(16, 32)) + list(range(8, 16))]


def scene_name(take, frame, ref):
    return f"CROOM_{take.split('_')[0]}_{int(frame):06d}_r{int(ref):02d}"


def cam_name(k):
    return f"cam_{int(k):02d}"


def build(take, frame, ref, fold, root=ANNOT_ROOT, data=os.path.join(S1, "Data"), overwrite=False):
    inst = os.path.join(root, take, f"{int(frame):06d}")
    index = json.load(open(os.path.join(inst, "index.json")))
    objects = json.load(open(os.path.join(inst, "annot", "objects.json")))
    cams = sorted(index["cams"], key=int)
    assert str(ref) in cams, (ref, cams)
    name = scene_name(take, frame, ref)
    out = os.path.join(data, name)
    if os.path.isdir(out):
        if not overwrite:
            raise SystemExit(f"{out} exists (--overwrite to rebuild; Result/ and Cache/ are removed too)")
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "images"))
    os.makedirs(os.path.join(out, "masks"))
    rel = os.path.relpath(os.path.join(inst, "images"), os.path.join(out, "images"))
    for k in cams:
        os.symlink(os.path.join(rel, f"{k}.jpg"), os.path.join(out, "images", f"{cam_name(k)}.jpg"))
    ref_label_path = os.path.join(inst, "annot", "labels", f"{ref}.png")
    lab = read_label(ref_label_path)
    W, H = Image.open(os.path.join(inst, "images", f"{ref}.jpg")).size
    assert lab.shape == (H, W), (lab.shape, (H, W))
    Image.fromarray(lab).save(os.path.join(out, "masks", f"{cam_name(ref)}.png"), compress_level=3)
    present = sorted(int(i) for i in np.unique(lab) if i)
    assert all(str(i) in objects for i in present), (present, list(objects))
    meta = dict(dataset=name, ref_frame=int(frame), ref_cam=cam_name(ref), cams=[cam_name(k) for k in cams],
                mask_cams=[cam_name(ref)], image_size=[W, H], basic_objects=present, external_fold=fold,
                ring_order=list(RING_ORDER) if set(RING_ORDER) == {cam_name(k) for k in cams} else None,
                source="croom_annot",
                annot=dict(take=take, frame=int(frame), ref=str(ref), objects=objects,
                           objects_absent_in_ref=sorted(int(i) for i in objects if int(i) not in present),
                           ref_label_sha256=sha256(ref_label_path), index_sha256=sha256(os.path.join(inst, "index.json")),
                           objects_sha256=sha256(os.path.join(inst, "annot", "objects.json"))))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    json.dump({str(i): dict(label=objects[str(i)]["name"], type="dynamic") for i in present},
              open(os.path.join(out, "objects_labels.json"), "w"), indent=1, ensure_ascii=False)
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--ref", type=int, required=True, help="reference camera index")
    ap.add_argument("--fold", type=int, default=2, help="external_fold: the held-out fold whose checkpoint decodes it")
    ap.add_argument("--root", default=ANNOT_ROOT)
    ap.add_argument("--data", default=os.path.join(S1, "Data"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out, meta = build(a.take, a.frame, a.ref, a.fold, a.root, a.data, a.overwrite)
    print(f"{out}: {len(meta['cams'])} cams, ref {meta['ref_cam']}, objects {meta['basic_objects']}, "
          f"absent in ref {meta['annot']['objects_absent_in_ref']}, fold {meta['external_fold']}")


if __name__ == "__main__":
    main()
