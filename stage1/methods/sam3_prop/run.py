#!/usr/bin/env python3
"""sam3_prop: treat the reference frame's N camera images as a video and propagate the reference
camera's ground-truth masks through it with SAM 3 (the stage-1 baseline of ../SCSegmentation).

    scripts/run_sam3_prop.sh <scene|all> [--order index|ref_outward] [--out NAME]

The pseudo-video is built as a folder of integer-named symlinks (the SAM 3 loader sorts by
int(stem)), the reference position is prompted with one mask per object, and propagation runs in
both directions.  Protocol as in ../SCSegmentation/MVSeed/run_seed.py; `start_frame_index` is
never passed.  Runs inside the `scsam3` image.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from methods import common  # noqa: E402
from mvseg2d.io import cache_dir, camera_order, labelmap_to_mask  # noqa: E402

ORDERS = ("index", "ref_outward")


def view_order(order, n, ref_idx):
    """A permutation of range(n): the sequence the pseudo-video is built in."""
    if order == "index":
        return list(range(n))
    if order == "ref_outward":
        seq, lo, hi = [ref_idx], ref_idx - 1, ref_idx + 1
        while lo >= 0 or hi < n:
            if lo >= 0:
                seq.append(lo)
                lo -= 1
            if hi < n:
                seq.append(hi)
                hi += 1
        return seq
    raise ValueError(order)


def build_pseudo_video(scene, cams, order_seq, folder):
    """Symlink images/<cam>.jpg as <folder>/%06d.jpg in `order_seq` (indices into `cams`, the rig order), replacing
    whatever was there."""
    os.makedirs(folder, exist_ok=True)
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))
    for pos, view_idx in enumerate(order_seq):
        target = os.path.realpath(scene.image_paths[cams[view_idx]])
        # relative so the link also resolves when the tree is mounted under /host inside a container
        os.symlink(os.path.relpath(target, os.path.realpath(folder)),
                   os.path.join(folder, f"{pos:06d}.jpg"))


def propagate(scene, args):
    import numpy as np
    import torch
    sys.path.insert(0, common.SAM3_PKG)
    from build_scsam3 import build_scsam3_video_predictor
    from io_utils import AsyncVideoFrameCPUToGPU, load_video_frames

    cams = scene.ring_order or camera_order(scene.cams)   # the rig's order (meta ring_order, else natural index order)
    n = len(cams)
    ref_idx = cams.index(scene.ref_cam)
    seq = view_order(args.order, n, ref_idx)
    pos_of_view = {v: p for p, v in enumerate(seq)}
    folder = os.path.join(cache_dir(args.data, scene.name, "pseudo_video"), args.order)
    build_pseudo_video(scene, cams, seq, folder)

    predictor = propagate.predictor
    if predictor is None:
        predictor = build_scsam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
        predictor.model.fill_hole_area = 0
        propagate.predictor = predictor

    cpu_image, height, width = load_video_frames(video_path=folder)
    assert (height, width) == scene.hw, ((height, width), scene.hw)
    frames = AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)
    images = [frames[i] for i in range(n)]

    response = predictor.handle_request(request=dict(type="start_session", images=images,
                                                     orig_height=height, orig_width=width))
    sid = response["session_id"]
    predictor.handle_request(request=dict(type="reset_session", session_id=sid))
    for obj_id in scene.objects:
        predictor.handle_request(request=dict(
            type="add_prompt", session_id=sid, frame_index=pos_of_view[ref_idx],
            mask=torch.tensor(labelmap_to_mask(scene.ref_labels, obj_id).astype("float32")),
            obj_id=int(obj_id)))
    responses = predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=sid, propagation_direction="both"))
    written = 0
    for r in responses:
        view_idx = seq[r["frame_index"]]
        if view_idx == ref_idx:
            continue                       # reference camera: ground truth already copied
        o = r["outputs"]
        for i, obj_id in enumerate(o["out_obj_ids"].tolist()):
            m = o["out_binary_masks"][i] > 0.0
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            m = m.squeeze().astype(bool)
            if m.any():
                common.write_result(args.data, scene.name, args.out, cams[view_idx], obj_id, m)
                written += 1
    try:
        predictor.handle_request(request=dict(type="close_session", session_id=sid))
    except Exception:
        pass
    torch.cuda.empty_cache()
    return {"order": args.order, "sequence": seq, "prompt_position": pos_of_view[ref_idx],
            "fill_hole_area": 0, "n_objects": len(scene.objects), "masks_written": written}


propagate.predictor = None


def _extra(ap):
    ap.add_argument("--order", choices=ORDERS, default="index")


if __name__ == "__main__":
    common.run_all(common.parse_args(__doc__, _extra), propagate)
