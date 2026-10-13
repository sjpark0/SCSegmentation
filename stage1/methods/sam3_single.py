"""Single-image, single-object SAM 3 segmentation from point/box prompts (shared by pi3_sam3 and dino_sam3).

Uses the tracker of the SCSam3 package as an image model: a fresh one-frame session per call, one
`add_tracker_new_points` with the prompts, no memory.  Frame 0 of an untracked session is an
"initial conditioning frame", so this behaves like SAM.  Labels follow the SAM 2 convention:
1 = positive point, 0 = negative point, 2 = box top-left corner, 3 = box bottom-right corner.
Runs inside `scsam3`.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

from methods import common


FILL_HOLE_AREA = 0   # the raw builder defaults to 16; the mainline and sam3_prop run with 0


def load_model(fill_hole_area=FILL_HOLE_AREA):
    sys.path.insert(0, common.SAM3_PKG)
    from build_scsam3 import build_scsam3_video_model
    model = build_scsam3_video_model()
    # Besides parity with the baseline, the hole-filling Triton connected-components kernel fails
    # ("invalid argument") on frames wider than ~2048 px, which killed six scenes in the first run.
    model.fill_hole_area = fill_hole_area
    # SCSam3TrackerPredictor.__init__ enters a process-wide torch.autocast(bf16) context that is never
    # exited (SCSam3TrackerPredictor.py:51-52).  autocast caches the bf16 casts of leaf parameters
    # until the OUTERMOST autocast context exits, so in this process the cache is never cleared: every
    # forward reuses the casts made at the FIRST forward.  For inference that only matters when weights
    # are replaced after a forward (checkpoint loading); for training it is fatal (the forward never
    # sees the updated weights and gradients stop at the cached casts).  Disable the cache.
    try:
        import torch
        torch.set_autocast_cache_enabled(False)
    except ImportError:          # host-side unit tests fake the builder and have no torch
        pass
    return model


def load_frame(path):
    """One image -> (frame tensor the tracker expects, h, w), via the package's own loader."""
    sys.path.insert(0, common.SAM3_PKG)
    from io_utils import AsyncVideoFrameCPUToGPU, load_video_frames
    tmp = tempfile.mkdtemp(prefix="mvseg2d_frame_")
    try:
        os.symlink(os.path.realpath(path), os.path.join(tmp, "000000.jpg"))
        cpu_image, h, w = load_video_frames(video_path=tmp)
        frame = AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)[0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return frame, h, w


def segment(model, frame, h, w, points_xy, labels):
    """Mask (bool [h, w]) of the single object described by the prompts; all-False if none returned."""
    import torch
    pts = torch.tensor([[(x + 0.5) / w, (y + 0.5) / h] for x, y in points_xy], dtype=torch.float32)
    lab = torch.tensor(list(labels), dtype=torch.int32)
    state = model.init_state([frame], h, w)
    try:
        _, out = model.add_tracker_new_points(inference_state=state, frame_idx=0, obj_id=1,
                                              points=pts, labels=lab)
        bm = out["out_binary_masks"]
        bm = bm.cpu().numpy() if hasattr(bm, "cpu") else np.asarray(bm)
        if len(bm) == 0:
            return np.zeros((h, w), bool)
        m = np.asarray(bm[0]).squeeze().astype(bool)
        assert m.shape == (h, w), (m.shape, (h, w))
        return m
    finally:
        del state
