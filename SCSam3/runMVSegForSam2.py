#!/usr/bin/env python3
"""Run the demoSCSam3ForSam2New variant over an MVSeg dataset.

That demo keeps the SAM 2-style predictor API, so `AddMaskSingle` and
`InitializeSegmentation` work as they do in the SAM 2 baseline.  Its
`LoadVideo_Folder_MVSeg` does not - it calls init_state(video_path=...), a
signature the predictor no longer has - so the loader is replaced below with a
folder version of the class's own `LoadVideo_File`.

Everything else follows SCSam2/demo/sam2_demoVideoNew_maskSingleInputMVSeg.py:
same reference-camera rule, same single mask seed, same propagation order.

    python runMVSegForSam2.py FlameSteak
    python runMVSegForSam2.py CoffeeMartini --out SegMaskSam3ForSam2New

Writes <dataset>/<out>/<cam>/<frame>/<object id>.png, the layout eval_jf.py reads.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
ALGO_DIR = os.path.join(HERE, "demoSCSam3ForSam2New")
DEFAULT_OUT = "SegMaskSam3ForSam2New"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--config", default=os.path.join(HERE, "demo", "MVSeg.json"))
    ap.add_argument("--data-root", default=os.path.join(HERE, "..", "Data", "MVSeg"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_config(path, name):
    cfg = json.load(open(path))
    if name not in cfg:
        sys.exit(f"{name!r} is not in {path}")
    d = cfg[name]
    perms = d.get("perms") or list(range(d["start_cam"],
                                         d["num_cam"] + d["start_cam"]))
    return {"folder": d["folder"], "start_frame": d["start_frame"],
            "num_frame": d["num_frame"], "cam_list": d["cam_list"],
            "perms": perms, "prefix": d["prefix"], "prefix1": d["prefix1"]}


def cam_name(c, prefix, prefix1):
    return f"{prefix}{c:0{prefix1}d}"


def main():
    args = parse_args()
    c = load_config(args.config, args.dataset)
    ds_dir = os.path.join(os.path.abspath(args.data_root), c["folder"])
    out_dir = os.path.join(ds_dir, args.out)
    written = [cam_name(x, c["prefix"], c["prefix1"]) for x in c["cam_list"]]

    import cv2
    import numpy as np

    # reference camera: the annotated one whose ground truth at start_frame
    # carries the highest object id - the SAM 2 baseline's rule
    best_cam, best_n = None, -1
    for cam in c["cam_list"]:
        p = os.path.join(ds_dir, "Mask", cam_name(cam, c["prefix"], c["prefix1"]),
                         f"{c['start_frame']:06d}.png")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"missing ground truth: {p}")
        if int(np.max(img)) > best_n:
            best_cam, best_n = cam, int(np.max(img))
    ref_index = c["perms"].index(best_cam)

    print(f"dataset        {args.dataset}  ({ds_dir})")
    print(f"algorithm      ForSam2New")
    print(f"output         {out_dir}")
    print(f"cameras        {len(c['perms'])} loaded, {len(written)} written {written}")
    print(f"frames         {c['start_frame']} .. "
          f"{c['start_frame'] + c['num_frame'] - 1}  ({c['num_frame']})")
    print(f"reference      {cam_name(best_cam, c['prefix'], c['prefix1'])} "
          f"(view index {ref_index}), {best_n} objects", flush=True)

    if not args.overwrite and os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(f"{out_dir} exists and is not empty; pass --overwrite")
    if args.dry_run:
        print("dry run, stopping before the model is built")
        return

    # SCSam3/sam3 is the upstream checkout with no __init__.py; with this
    # script's directory on sys.path it shadows the installed sam3 package.
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != HERE]
    sys.path.insert(0, ALGO_DIR)
    os.chdir(ALGO_DIR)

    import torch
    from SCSam3Video import SCSam3Video
    from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU

    class MVSegForSam2(SCSam3Video):
        def LoadCameraFolders(self, video_root, cam_names, start_frame):
            """LoadVideo_File, but over per-camera JPEG folders.

            The class ships a LoadVideo_Folder_MVSeg, but it calls
            init_state(video_path=...), a signature this predictor no longer
            has, so it raises before it loads anything.  This mirrors
            LoadVideo_File instead, and seeds the cross-view state from
            `start_frame` of each camera rather than frame 0.
            """
            for name in cam_names:
                folder = os.path.join(video_root, name)
                if not os.path.isdir(folder):
                    raise FileNotFoundError(folder)
                cpu_image, self.video_height, self.video_width = load_video_frames(
                    video_path=folder)
                image = AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)
                state = self.predictor.init_state(
                    images=image, video_height=self.video_height,
                    video_width=self.video_width, num_frames=len(cpu_image),
                    offload_video_to_cpu=True, offload_state_to_cpu=True)
                self.predictor.reset_state(state)
                self.inference_state.append(state)
                self.images.append(image)
                self.cpu_images.append(cpu_image)

            self.numImage = len(cam_names)
            imgs = [self.images[i][start_frame] for i in range(self.numImage)]
            self.inference_state_spatial = self.predictor_spatial.init_state(
                images=imgs, video_height=self.video_height,
                video_width=self.video_width, num_frames=self.numImage,
                offload_video_to_cpu=True, offload_state_to_cpu=True)
            self.predictor_spatial.reset_state(self.inference_state_spatial)

        def RunNaiveTracking(self, frame_idx, reverse=False):
            """As the base method, but tolerating a view that lost an object.

            The base indexes masks_spatial[m][obj_id] directly and raises when
            cross-view propagation dropped an object in some view; an empty mask
            there is what the OneStageNew demo does.
            """
            from itertools import chain
            zero = None
            for m in range(self.numImage):
                for obj_id in self.obj_ids:
                    mask = self.masks_spatial[m].get(obj_id)
                    if mask is None:
                        if zero is None:
                            zero = torch.zeros((self.video_height, self.video_width),
                                               dtype=torch.float32)
                        mask = zero
                    else:
                        mask = mask[0, ...]
                    self.predictor.add_new_mask(
                        inference_state=self.inference_state[m],
                        frame_idx=frame_idx, obj_id=obj_id, mask=mask)

            self.tracking_result = [
                self.predictor.propagate_in_video(
                    self.inference_state, spatial_idx=m, start_frame_idx=frame_idx,
                    max_frame_num_to_track=240, reverse=False,
                    propagate_preflight=True)
                for m in range(self.numImage)]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device         {device}", flush=True)

    sc = MVSegForSam2(device)
    cam_names = [cam_name(x, c["prefix"], c["prefix1"]) for x in c["perms"]]
    sc.LoadCameraFolders(os.path.join(ds_dir, "Video"), cam_names, c["start_frame"])
    print(f"states ready   ({sc.numImage} cameras)", flush=True)

    ref_gt = cv2.imread(
        os.path.join(ds_dir, "Mask", cam_name(best_cam, c["prefix"], c["prefix1"]),
                     f"{c['start_frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
    # this variant's add_new_mask asserts a 2-D torch tensor, where the SAM 2
    # one took a numpy array; AddMaskSingle divides by 255 on the way in
    for i in range(best_n):
        seed = torch.from_numpy(((ref_gt == (i + 1)) * 255).astype(np.uint8))
        sc.AddMaskSingle(ref_index, seed, i + 1)
    print(f"prompted       {len(sc.obj_ids)} objects on view {ref_index}", flush=True)

    # views before the reference need the backward pass, as in the SAM 2 script
    if ref_index != 0:
        sc.InitializeSegmentation(refCamID=ref_index, reverse=True)
    sc.InitializeSegmentation(refCamID=ref_index)
    print(f"cross-view     done, {len(sc.masks_spatial)} views have masks", flush=True)

    sc.RunNaiveTracking(c["start_frame"])

    for _ in range(c["num_frame"]):
        for m in range(sc.numImage):
            frame_idx, out_obj_ids, _, out_mask_logits, _ = next(sc.tracking_result[m])
            name = cam_name(c["perms"][m], c["prefix"], c["prefix1"])
            if name not in written:
                continue
            folder = os.path.join(out_dir, name, f"{frame_idx:d}")
            os.makedirs(folder, exist_ok=True)
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                cv2.imwrite(os.path.join(folder, f"{int(obj_id):d}.png"),
                            mask.squeeze().astype(np.uint8) * 255)
        print(f"  frame {frame_idx} written", flush=True)

    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
