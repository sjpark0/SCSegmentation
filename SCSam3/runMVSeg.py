#!/usr/bin/env python3
"""Run one SCSam3 demo algorithm over an MVSeg dataset and write per-object masks.

The output layout is the one Data/MVSeg/eval_jf.py expects, so J&F can be
computed straight afterwards:

    <dataset>/<out>/<cam>/<frame>/<object id>.png      binary, 0 or 255

What the run does, matching the SAM 2 baseline in
SCSam2/demo/sam2_demoVideoNew_maskSingleInputMVSeg.py so the numbers stay
comparable:

  1. open one temporal session per camera over that camera's JPEG folder,
  2. build one cross-view session from frame `start_frame` of every camera,
  3. pick the reference camera as the annotated camera whose ground truth at
     `start_frame` holds the most objects,
  4. feed that camera's ground-truth masks in as the only prompt,
  5. propagate across views, then track forward in time in each view.

Only step 5 needs a GPU.  Nothing here writes outside the dataset directory.

    python runMVSeg.py Blocks --algo OneStageNew
    python runMVSeg.py Blocks --algo OneStage --out SegMaskSam3OneStage

Both demos carry their own copy of the model code, so the algorithm is selected
by putting its directory first on sys.path.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
# let the model infer the cross-view start from the prompted view
SPATIAL_START_IMPLICIT = os.environ.get("SPATIAL_START_IMPLICIT", "1") == "1"
ALGOS = {"OneStage": "demoSCSam3OneStage",
         "OneStageNew": "demoSCSam3OneStageNew",
         "MVOpt": "demoSCSam3MVOpt"}
DEFAULT_OUT = {"OneStage": "SegMaskSam3OneStage",
               "OneStageNew": "SegMaskSam3OneStageNew",
               "MVOpt": "SegMaskSam3MVOpt"}
# MVOpt is the development target. As of 2026-09-04 demoSCSam3OneStageNew holds
# a byte-identical frozen snapshot of it (see that folder's FROZEN.md); the
# unpatched original that produced the published J&F numbers is at git tag
# baseline-onestagenew. Both share the cross-view memory design, so both need
# every view tracked.
NEEDS_ALL_VIEWS = ("OneStageNew", "MVOpt")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset name as it appears in MVSeg.json")
    ap.add_argument("--algo", choices=sorted(ALGOS), default="OneStageNew")
    ap.add_argument("--out", default=None,
                    help="output folder name under the dataset (default: per algo)")
    ap.add_argument("--config", default=os.path.join(HERE, "demo", "MVSeg.json"))
    ap.add_argument("--data-root", default=os.path.join(HERE, "..", "Data", "MVSeg"))
    ap.add_argument("--device", default=None, help="cuda (default), cpu or mps")
    ap.add_argument("--track-cams", choices=["written", "all"], default=None,
                    help="open a temporal session for only the scored cameras "
                         "(written) or for every camera (all). Default: written "
                         "for OneStage, all for OneStageNew, which needs every "
                         "view's state for its cross-view memory.")
    ap.add_argument("--overwrite", action="store_true",
                    help="rerun even if the output folder already holds masks")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the plan, load no model")
    return ap.parse_args()


def load_config(path, name):
    with open(path) as fh:
        cfg = json.load(fh)
    if name not in cfg:
        sys.exit(f"{name!r} is not in {path}. Available: {', '.join(sorted(cfg))}")
    d = cfg[name]
    perms = d.get("perms")
    if perms is None:
        perms = list(range(d["start_cam"], d["num_cam"] + d["start_cam"]))
    return {
        "folder": d["folder"],
        "start_frame": d["start_frame"],
        "num_frame": d["num_frame"],
        "cam_list": d["cam_list"],
        "perms": perms,
        "prefix": d["prefix"],
        "prefix1": d["prefix1"],
    }


def cam_name(c, prefix, prefix1):
    return f"{prefix}{c:0{prefix1}d}"


def pick_reference(ds_dir, c, cv2, np):
    """The annotated camera whose ground truth at start_frame holds most objects.

    Same rule as the SAM 2 baseline: more objects in the reference frame means
    more of them get a prompt at all.
    """
    best_cam, best_n = None, -1
    for cam in c["cam_list"]:
        p = os.path.join(ds_dir, "Mask", cam_name(cam, c["prefix"], c["prefix1"]),
                         f"{c['start_frame']:06d}.png")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"missing ground truth: {p}")
        n = int(np.max(img))
        if n > best_n:
            best_cam, best_n = cam, n
    return best_cam, best_n


def build_runner(algo):
    """Import the selected demo's model code and extend it for MVSeg."""
    algo_dir = os.path.join(HERE, ALGOS[algo])

    # SCSam3/sam3 is the upstream checkout: a directory with no __init__.py.
    # While this file's own directory is on sys.path it shadows the installed
    # sam3 package as a namespace package, whose __file__ is None, and the
    # model builder dies looking up the tokenizer inside it.  The demos do not
    # hit this because they run from inside their own folder.
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != HERE]
    sys.path.insert(0, algo_dir)
    os.chdir(algo_dir)          # the demo modules resolve some paths relatively

    import torch
    from SCSam3Video import SCSam3Video
    from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU

    class MVSegVideo(SCSam3Video):
        """SCSam3Video with the pieces MVSeg needs.

        The base class only knows how to load a folder of MP4s and always
        prompts on frame 0; MVSeg is folders of JPEGs and starts at an
        arbitrary frame.  LoadVideo_Folder_MVSeg and AddMaskSingle in the base
        class still target the old SAM 2 predictor API and would raise, so the
        two methods below replace them rather than calling into them.
        """

        @property
        def spatial(self):
            # OneStageNew keeps a separate predictor for the cross-view pass
            return getattr(self, "predictor_spatial", self.predictor)

        def LoadCameraFolders(self, video_root, cam_names, start_frame, track_idx):
            """Frame readers for every camera, temporal sessions for `track_idx`.

            Every camera contributes its `start_frame` to the cross-view pass,
            but only the cameras we actually score need a tracking session, and
            a session per camera is what runs the GPU out of memory on the
            45-camera scenes.  The readers are lazy, so the cameras we skip cost
            one decoded frame each.
            """
            self.track_views = list(track_idx)
            track_set = set(self.track_views)
            for i, name in enumerate(cam_names):
                folder = os.path.join(video_root, name)
                if not os.path.isdir(folder):
                    raise FileNotFoundError(folder)
                cpu_image, height, width = load_video_frames(video_path=folder)
                image = AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)
                self.images.append(image)
                self.cpu_images.append(cpu_image)
                if i not in track_set:
                    continue
                response = self.predictor.handle_request(
                    request=dict(type="start_session", images=image,
                                 orig_height=height, orig_width=width))
                session_id = response["session_id"]
                self.session_ids.append(session_id)
                self.predictor.handle_request(
                    request=dict(type="reset_session", session_id=session_id))

            self.numImage = len(cam_names)
            self.video_height, self.video_width = height, width

            # the cross-view "video" is frame `start_frame` of every camera
            frames = [self.images[i][start_frame] for i in range(self.numImage)]
            response = self.spatial.handle_request(
                request=dict(type="start_session", images=frames,
                             orig_height=height, orig_width=width))
            self.session_id_statial = response["session_id"]
            self.spatial.handle_request(
                request=dict(type="reset_session",
                             session_id=self.session_id_statial))

        def AddReferenceMask(self, view_index, mask, obj_id):
            """One binary ground-truth mask as the prompt for one object."""
            response = self.spatial.handle_request(
                request=dict(type="add_prompt",
                             session_id=self.session_id_statial,
                             frame_index=view_index,
                             mask=torch.tensor(mask, dtype=torch.float32),
                             obj_id=obj_id))
            self.obj_ids = response["outputs"]["out_obj_ids"].tolist()

        def PropagateAcrossViews(self, view_index):
            """Both directions, so views on either side of the reference are filled.

            The base class passes `start_frame_idx`, which the predictor does
            not read - it looks for `start_frame_index` - so the start view is
            spelled out here.
            """
            request = dict(type="propagate_in_video",
                           session_id=self.session_id_statial,
                           propagation_direction="both")
            if not SPATIAL_START_IMPLICIT:
                request["start_frame_index"] = view_index
            responses = self.spatial.handle_stream_request(request=request)
            for response in responses:
                out = response["outputs"]
                self.masks_spatial[response["frame_index"]] = {
                    obj_id: (out["out_binary_masks"][i] > 0.0)
                    for i, obj_id in enumerate(out["out_obj_ids"].tolist())
                }

        def TrackForward(self, start_frame, num_frame):
            """Seed each view with its propagated mask, then track forward only.

            Forward only, and capped at num_frame: MVSeg annotates
            start_frame .. start_frame + num_frame - 1, and letting it run
            backward would track hundreds of frames nobody scores.
            """
            zero = torch.zeros((self.video_height, self.video_width),
                               dtype=torch.float32)
            for j, view in enumerate(self.track_views):
                for obj_id in self.obj_ids:
                    mask = self.masks_spatial[view].get(obj_id)
                    self.predictor.handle_request(
                        request=dict(type="add_prompt",
                                     session_id=self.session_ids[j],
                                     frame_index=start_frame,
                                     mask=(torch.tensor(mask, dtype=torch.float32)
                                           if mask is not None else zero),
                                     obj_id=obj_id))

            self.tracking_result = [None] * len(self.track_views)
            for m in range(len(self.track_views)):
                request = dict(type="propagate_in_video",
                               propagation_direction="forward",
                               start_frame_index=start_frame,
                               max_frame_num_to_track=num_frame)
                # OneStageNew's tracker reads every session to share memory
                # across views; OneStage tracks one session at a time.
                # Test the flag, not the attribute: the cross-view model is
                # retired before this point, so its presence says nothing.
                # The attribute check is the fallback for packages that do not
                # set the flag. Since 2026-09-04 both OneStageNew and MVOpt set
                # it; OneStage sets neither and takes the single-session branch,
                # because its __init__ never builds predictor_spatial at all.
                if getattr(self, "uses_spatial_predictor",
                           getattr(self, "predictor_spatial", None) is not None):
                    request["session_ids"] = self.session_ids
                    request["spatial_idx"] = m
                else:
                    request["session_id"] = self.session_ids[m]
                self.tracking_result[m] = self.predictor.handle_stream_request(
                    request=request)

    return MVSegVideo, torch


def main():
    args = parse_args()
    out_name = args.out or DEFAULT_OUT[args.algo]
    c = load_config(args.config, args.dataset)
    data_root = os.path.abspath(args.data_root)
    ds_dir = os.path.join(data_root, c["folder"])
    video_root = os.path.join(ds_dir, "Video")
    cam_names = [cam_name(x, c["prefix"], c["prefix1"]) for x in c["perms"]]
    written_cams = [cam_name(x, c["prefix"], c["prefix1"]) for x in c["cam_list"]]

    missing = [n for n in cam_names if not os.path.isdir(os.path.join(video_root, n))]
    if missing:
        sys.exit(f"missing camera folders under {video_root}: {missing}")

    import cv2
    import numpy as np

    ref_cam, n_obj = pick_reference(ds_dir, c, cv2, np)
    ref_index = c["perms"].index(ref_cam)

    print(f"dataset        {args.dataset}  ({ds_dir})")
    print(f"algorithm      {args.algo}  ({ALGOS[args.algo]})")
    print(f"output         {os.path.join(ds_dir, out_name)}")
    print(f"cameras        {len(cam_names)} loaded, {len(written_cams)} written "
          f"{written_cams}")
    print(f"frames         {c['start_frame']} .. "
          f"{c['start_frame'] + c['num_frame'] - 1}  ({c['num_frame']})")
    print(f"reference      {cam_name(ref_cam, c['prefix'], c['prefix1'])} "
          f"(view index {ref_index}), {n_obj} objects prompted", flush=True)

    out_dir = os.path.join(ds_dir, out_name)
    if not args.overwrite and os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(f"{out_dir} already exists and is not empty; pass --overwrite")
    if args.dry_run:
        print("dry run, stopping before the model is built")
        return

    MVSegVideo, torch = build_runner(args.algo)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device         {device}", flush=True)

    track_mode = args.track_cams or ("all" if args.algo in NEEDS_ALL_VIEWS else "written")
    track_idx = (list(range(len(cam_names))) if track_mode == "all"
                 else [i for i, n in enumerate(cam_names) if n in written_cams])

    sc = MVSegVideo(device)
    sc.LoadCameraFolders(video_root, cam_names, c["start_frame"], track_idx)
    print(f"sessions ready ({sc.numImage} cameras loaded, "
          f"{len(track_idx)} tracked [{track_mode}], "
          f"{sc.video_width}x{sc.video_height})", flush=True)

    ref_gt = cv2.imread(
        os.path.join(ds_dir, "Mask", cam_name(ref_cam, c["prefix"], c["prefix1"]),
                     f"{c['start_frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
    for obj_id in range(1, n_obj + 1):
        sc.AddReferenceMask(ref_index, (ref_gt == obj_id).astype("float32"), obj_id)
    print(f"prompted       {len(sc.obj_ids)} objects on view {ref_index}", flush=True)

    sc.PropagateAcrossViews(ref_index)
    print(f"cross-view     done, {len(sc.masks_spatial)} views have masks", flush=True)

    # The cross-view model is dead weight from here on: ~3.2 GiB of parameters,
    # the N-view pseudo-video, its feature cache and its per-view tracker
    # memories.  Retiring it does not touch `masks_spatial`, which TrackForward
    # still reads.  The autocast cache pins a bf16 copy of every weight it ever
    # cast (the tracker enters torch.autocast permanently and never exits, so
    # ATen never clears it) and keys those entries on a weak ref to the now
    # dead fp32 parameters, so without this clear ~1.6 GiB stays stranded.
    # Clearing is value-neutral: entries are a deterministic `param.to(bf16)`
    # of weights that never change during inference, so they are recomputed
    # bit-for-bit.  Once here, not per frame: inside a frame the whole live
    # weight set is cached anyway, so per-frame clearing lowers no peak.
    if hasattr(sc, "RetireSpatialPredictor"):
        sc.RetireSpatialPredictor()
        torch.clear_autocast_cache()
        print("spatial model  retired", flush=True)

    sc.TrackForward(c["start_frame"], c["num_frame"])

    for _ in range(c["num_frame"]):
        for j, view in enumerate(sc.track_views):
            response = next(sc.tracking_result[j])
            if cam_names[view] not in written_cams:
                continue
            frame_idx = response["frame_index"]
            out = response["outputs"]
            folder = os.path.join(out_dir, cam_names[view], f"{frame_idx:d}")
            os.makedirs(folder, exist_ok=True)
            for i, obj_id in enumerate(out["out_obj_ids"].tolist()):
                mask = (out["out_binary_masks"][i] > 0.0)
                mask = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
                cv2.imwrite(os.path.join(folder, f"{obj_id:d}.png"),
                            mask.squeeze().astype(np.uint8) * 255)
        print(f"  frame {response['frame_index']} written", flush=True)

    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
