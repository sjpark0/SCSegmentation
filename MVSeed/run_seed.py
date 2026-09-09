#!/usr/bin/env python3
"""Stage 1 only: propagate the reference camera's ground-truth masks to every view.

The control (`--order index`) is exactly what SCSam3/runMVSeg.py does today -- take
frame `start_frame` of every camera in camera-number order, treat that stack as a video,
prompt the reference position with the ground truth, propagate both directions -- with
the temporal stage removed.  Every other `--order` reorders that pseudo-video and maps
the result back, so the two are comparable frame for frame.

Writes a PNG for EVERY view, not only the three annotated cameras, to
Data/MVSeg/<scene>/<--out>/<camera>/<start_frame>/<obj>.png, and refuses any --out that
does not start with MVSeed_ (MVSeed/README.md rule R2).  Every view because the folder
is what the mainline `--seeds-from` hook will load in place of its own stage 1
(docs/stage1-plan.md section 6), and that needs a seed for each tracked view;
score_seeds.py reads just the annotated three.  SEED_MANIFEST.json beside the PNGs says
what was written (`written_views`, per-view `coverage`) next to what is scored
(`scored_views`), so a consumer can check coverage without listing the folder;
build_manifest() is pure so that schema is tested without torch.

    docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
      --user $(id -u):$(id -g) -v /:/host -w /host$PWD \
      -e HF_HOME=/host$PWD/SCSam3/hf_cache -e HF_HUB_OFFLINE=1 scsam3 \
      python MVSeed/run_seed.py Fencing --order index --out MVSeed_control
    # then, on the annotated cameras only:
    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \
      python MVSeed/score_seeds.py --runs MVSeed_control

HF_HOME is required when running as yourself: the weights live in the image at
/root/.cache/huggingface (mode 700, root-owned), so a --user run cannot read them and
would try to download from a gated repo.  The mainline scripts run as root and never
hit this.  SCSam3/hf_cache holds the same snapshot on the host.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "SCSam3", "demoSCSam3MVOpt")
DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
OUT_PREFIX = "MVSeed_"
ORDERS = ("index", "reverse", "ref_outward")
GENERATOR = "video"     # the pseudo-video arm (G0/G1 in stage1-plan section 3)


def cam_name(c, prefix, prefix1):
    return f"{prefix}{c:0{prefix1}d}"


def view_order(order, n, ref_idx):
    """A permutation of range(n): the sequence the pseudo-video is built in.

    index        0,1,..,n-1                     the control (today's behaviour)
    reverse      n-1,..,0                       sanity arm: propagation is not symmetric
    ref_outward  ref, ref-1, ref+1, ref-2, ...  nearest-first from the reference
    """
    if order == "index":
        return list(range(n))
    if order == "reverse":
        return list(range(n - 1, -1, -1))
    if order == "ref_outward":
        seq, lo, hi = [ref_idx], ref_idx - 1, ref_idx + 1
        while lo >= 0 or hi < n:
            if lo >= 0:
                seq.append(lo); lo -= 1
            if hi < n:
                seq.append(hi); hi += 1
        return seq
    raise ValueError(order)


def build_manifest(dataset, order, sequence, cams, ref_idx, start_frame, n_objects,
                   scored, areas, argv):
    """SEED_MANIFEST.json as a dict.  Pure (plain ints, strs, lists) so it is testable
    without torch, and so every generator writes the same schema.

    `areas` is {camera: {obj_id: pixel count}} for the views a PNG was written for, in
    view-index order; `written_views` and `coverage` are derived from it rather than
    from `cams` so the manifest never claims a PNG that is not on disk.  `coverage`
    lists, per view, the objects that have a PNG there.  An empty mask still counts:
    it is the generator's statement that the object is absent from that view, and
    `areas` (0) tells it apart from a real one.  This generator gives every view every
    prompted object; the per-view generators of stage1-plan section 6 may not, and the
    mainline `--seeds-from` hook reads `coverage` to refuse a folder that does not
    cover the annotated cameras.
    """
    sequence = list(sequence)
    return {"dataset": dataset, "generator": GENERATOR,
            "order": order, "sequence": sequence,
            "reference": cams[ref_idx], "reference_view_index": ref_idx,
            "prompt_position": sequence.index(ref_idx), "start_frame": start_frame,
            "n_views": len(cams), "n_objects": n_objects,
            "written_views": list(areas), "scored_views": list(scored),
            "coverage": {cam: sorted(int(o) for o in per) for cam, per in areas.items()},
            "areas": areas, "argv": list(argv)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--order", choices=ORDERS, default="index")
    ap.add_argument("--out", required=True, help=f"output folder; must start with {OUT_PREFIX}")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.out.startswith(OUT_PREFIX):
        sys.exit(f"MVSeed R2: --out must start with {OUT_PREFIX!r}, got {args.out!r}")

    cfg = json.load(open(args.config, encoding="utf-8"))
    if args.dataset not in cfg:
        sys.exit(f"{args.dataset!r} is not in {args.config}")
    d = cfg[args.dataset]
    perms = d.get("perms") or list(range(d["start_cam"], d["num_cam"] + d["start_cam"]))
    cams = [cam_name(x, d["prefix"], d["prefix1"]) for x in perms]
    start = d["start_frame"]
    ds_dir = os.path.join(args.data_root, d["folder"])
    if "c_ini" not in d:
        sys.exit(f"{args.dataset}: no c_ini in the config")
    ref_idx = perms.index(d["c_ini"])
    ref_cam = cams[ref_idx]
    order = view_order(args.order, len(cams), ref_idx)
    assert sorted(order) == list(range(len(cams)))
    out_dir = os.path.join(ds_dir, args.out)
    scored = [cam_name(x, d["prefix"], d["prefix1"]) for x in d["cam_list"]]

    print(f"dataset        {args.dataset}  ({ds_dir})")
    print(f"views          {len(cams)}  frame {start}")
    print(f"reference      {ref_cam} (view index {ref_idx})")
    print(f"order          {args.order}: {order[:8]}{' ...' if len(order) > 8 else ''}")
    print(f"output         {out_dir}  (all {len(cams)} views; scored {scored})")
    if args.dry_run:
        return
    if os.path.isdir(out_dir) and not args.overwrite:
        sys.exit(f"{out_dir} exists; pass --overwrite")

    sys.path.insert(0, PKG)
    import cv2
    import numpy as np
    import torch
    from build_scsam3 import build_scsam3_video_predictor
    from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU

    ref_gt = cv2.imread(os.path.join(ds_dir, "Mask", ref_cam, f"{start:06d}.png"),
                        cv2.IMREAD_GRAYSCALE)
    if ref_gt is None:
        sys.exit(f"missing ground truth for {ref_cam}")
    n_obj = int(np.max(ref_gt))
    print(f"objects        {n_obj} prompted on {ref_cam}", flush=True)

    predictor = build_scsam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
    predictor.model.fill_hole_area = 0

    # one decoded frame per camera, then the pseudo-video in `order`
    frames_by_view, height, width = {}, None, None
    for i, name in enumerate(cams):
        folder = os.path.join(ds_dir, "Video", name)
        if not os.path.isdir(folder):
            sys.exit(f"missing camera folder: {folder}")
        cpu_image, height, width = load_video_frames(video_path=folder)
        image = AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)
        frames_by_view[i] = image[start]
    frames = [frames_by_view[v] for v in order]
    pos_of_view = {v: p for p, v in enumerate(order)}

    response = predictor.handle_request(
        request=dict(type="start_session", images=frames,
                     orig_height=height, orig_width=width))
    sid = response["session_id"]
    predictor.handle_request(request=dict(type="reset_session", session_id=sid))
    for obj_id in range(1, n_obj + 1):
        predictor.handle_request(request=dict(
            type="add_prompt", session_id=sid, frame_index=pos_of_view[ref_idx],
            mask=torch.tensor((ref_gt == obj_id).astype("float32"), dtype=torch.float32),
            obj_id=obj_id))
    print(f"prompted       position {pos_of_view[ref_idx]} of the pseudo-video", flush=True)

    # implicit start: never pass start_frame_index (investigations-closed 4)
    responses = predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=sid, propagation_direction="both"))
    masks = {}
    for r in responses:
        o = r["outputs"]
        masks[order[r["frame_index"]]] = {
            obj_id: (o["out_binary_masks"][i] > 0.0)
            for i, obj_id in enumerate(o["out_obj_ids"].tolist())}
    print(f"propagated     {len(masks)} views have masks", flush=True)

    # every view to disk, in view-index order (the scored three are a subset)
    areas = {}
    for v, per in sorted(masks.items()):
        areas[cams[v]] = {int(o): int(m.sum().item()) for o, m in per.items()}
        folder = os.path.join(out_dir, cams[v], str(start))
        os.makedirs(folder, exist_ok=True)
        for obj_id, m in per.items():
            a = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            cv2.imwrite(os.path.join(folder, f"{obj_id:d}.png"),
                        a.squeeze().astype(np.uint8) * 255)
    manifest = build_manifest(args.dataset, args.order, order, cams, ref_idx, start, n_obj,
                              scored, areas, sys.argv)
    with open(os.path.join(out_dir, "SEED_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"done -> {out_dir}  ({len(areas)} views written)", flush=True)


if __name__ == "__main__":
    main()
