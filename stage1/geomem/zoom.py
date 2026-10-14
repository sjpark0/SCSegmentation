"""S1-E10 (docs/stage1-E10-prereg.md section 2): training-free 2x crop-zoom re-decode after round 2.

For every (target camera, object) whose round-2 mask is non-empty, the object is decoded once more on a fixed
(H/2, W/2) image-aspect window centred on the mask's bounding box (skipped when the box does not fit), with the
SAME round-2 memory set cut to that object (B = 1), and the crop result replaces the round-2 mask ONLY inside the
one-cell boundary band (r = ceil(max(H, W) / 288) px) of the round-2 mask.  Nothing is ever deleted, re-interpreted
or grown beyond r: outside the band the round-2 decision stands, and the pass is a no-op for an empty prediction.

Torch-free helpers (window_for, band_mask, paste, verdict, cell_positions) are host-testable; `refine_camera`
needs the scsam3 image.  Constants below are the registered ones; nothing here is swept.
"""
import math
import time
from collections import Counter

import cv2
import numpy as np

ZOOM = 2            # fixed; the window is (H // ZOOM, W // ZOOM)
MARGIN = 0.10       # the bbox * (1 + 2 * MARGIN) must fit the window on both axes, else the pair is skipped
TAU_IOU = 0.5       # IoU(crop canvas, round-2 mask) below this -> keep round 2 (small-object sanity guard)
GRID = 72
LOW_RES = 288       # the tracker's low_res_mask_size (1008 // 14 * 4): one cell = max(H, W) / 288 original px
VERDICTS = ("skip_big", "kept_absent", "kept_iou", "kept_empty", "accepted")


def band_px(hw):
    """Boundary-band half-width in original pixels: one 288-grid cell of the longer side (7 at 1920 ... 10 at 2704)."""
    return int(math.ceil(max(hw) / LOW_RES))


def bbox_of(mask):
    """Exclusive (y0, y1, x0, x1) of a bool mask, or None when empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def window_for(bbox, hw, zoom=ZOOM, margin=MARGIN):
    """The fixed (H // zoom, W // zoom) window centred on the bbox and pushed inside the image, as an exclusive
    (y0, y1, x0, x1); None when bbox * (1 + 2 * margin) does not fit the window (the pair is skipped)."""
    H, W = hw
    wh, ww = H // zoom, W // zoom
    y0b, y1b, x0b, x1b = bbox
    if (y1b - y0b) * (1 + 2 * margin) > wh or (x1b - x0b) * (1 + 2 * margin) > ww:
        return None
    y0 = int(min(max(round((y0b + y1b) / 2 - wh / 2), 0), H - wh))
    x0 = int(min(max(round((x0b + x1b) / 2 - ww / 2), 0), W - ww))
    return y0, y0 + wh, x0, x0 + ww


def band_mask(mask, r):
    """dilate(mask, r) minus erode(mask, r) with a (2r+1)^2 square kernel -> bool."""
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    m = mask.astype(np.uint8)
    return (cv2.dilate(m, k) > 0) & ~(cv2.erode(m, k) > 0)


def paste(full, canvas, r):
    """The registered paste rule: the canvas inside band_r(full), the round-2 mask everywhere else."""
    return np.where(band_mask(full, r), canvas, full)


def mask_iou(a, b):
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b)) / union if union else 0.0


def verdict(presence_c, crop_any, iou, final_any, tau=TAU_IOU):
    """First rule that fires (prereg section 2 item 6)."""
    if presence_c <= 0 or not crop_any:
        return "kept_absent"
    if iou < tau:
        return "kept_iou"
    if not final_any:
        return "kept_empty"
    return "accepted"


def cell_positions(window, hw, grid=GRID):
    """S1-E10 C2: the crop's grid x grid token centres in FULL-VIEW cell units (token k = i * grid + j ->
    (t_x[k], t_y[k])); the full-image window gives the integer lattice (t_x = j, t_y = i) exactly."""
    y0, y1, x0, x1 = window
    H, W = hw
    wh, ww = y1 - y0, x1 - x0
    j = np.arange(grid, dtype=np.float64)
    i = np.arange(grid, dtype=np.float64)
    t_x = (x0 * grid + (j + 0.5) * ww) / W - 0.5
    t_y = (y0 * grid + (i + 0.5) * wh) / H - 0.5
    return np.tile(t_x, grid).astype(np.float32), np.repeat(t_y, grid).astype(np.float32)


def rides_edge(canvas, window, hw):
    """True when the crop mask touches a window edge that is not an image edge (diagnostic only)."""
    y0, y1, x0, x1 = window
    H, W = hw
    c = canvas[y0:y1, x0:x1]
    return bool((y0 > 0 and c[0].any()) or (y1 < H and c[-1].any()) or (x0 > 0 and c[:, 0].any()) or (x1 < W and c[:, -1].any()))


def refine_camera(wrapper, model, mem, image_path, cache, cam, ref_cam, hw, pred, pres_full, iou_full, objects,
                  zoom=ZOOM, zoom_rope="grid", margin=MARGIN, tau=TAU_IOU):
    """One target camera after round 2: pred [B,H,W] bool (round-2 masks) -> (pred refined in place, records
    {obj: record}, Counter).  mem = the camera's round-2 MemorySet (reference + gated neighbours); wrapper / model
    as in decode_scene (called inside its no_grad + autocast context)."""
    import torch
    import torch.nn.functional as F
    from geomem import attn_patch, pointmap, sam3_io
    from geomem.embed import grid_tensors
    assert zoom_rope in ("grid", "remap"), zoom_rope
    t0 = time.perf_counter()
    H, W = hw
    r = band_px(hw)
    bgr = cv2.imread(image_path)
    assert bgr is not None and bgr.shape[:2] == (H, W), (image_path, None if bgr is None else bgr.shape, hw)
    src = pointmap.ref_frame_points(cache, cam, ref_cam)
    records, n = {}, Counter()
    others_count = None            # per-pixel count of round-2 masks over all objects (diagnostic steal_px; review 2026-10-14)
    for i, o in enumerate(objects):
        if not pred[i].any():
            continue
        n["candidates"] += 1
        bbox = bbox_of(pred[i])
        win = window_for(bbox, hw, zoom, margin)
        rec = dict(bbox=list(bbox), window=None if win is None else list(win), presence_full=float(pres_full[i]),
                   iou_head_full=float(iou_full[i]), px_full=int(pred[i].sum()))
        if win is None:
            n["skipped_large"] += 1
            rec.update(verdict="skip_big", px_final=rec["px_full"], added=0, removed=0)
            records[str(int(o))] = rec
            continue
        n["run"] += 1
        y0, y1, x0, x1 = win
        img_c = sam3_io.load_image_array(np.ascontiguousarray(bgr[y0:y1, x0:x1]))
        bb_c = sam3_io.backbone_features(model, img_c)
        p_c, v_c = grid_tensors(pointmap.window_grid(src, win, hw), "cuda")
        rope_q = None
        if zoom_rope == "remap":
            t_x, t_y = cell_positions(win, hw)
            rope_q = attn_patch.axial_cis_at(model.tracker, t_x, t_y)
        idx = torch.tensor([i], device=p_c.device)
        out_c = wrapper.decode_view(bb_c, 1, mem.subset(idx), p_c, v_c, rope_q=rope_q)
        presence_c, iou_c = float(out_c["presence"][0]), float(out_c["ious"][0])
        m_c = (F.interpolate(out_c["logits_1008"].float(), size=(y1 - y0, x1 - x0), mode="bilinear",
                             align_corners=False)[0, 0] > 0).cpu().numpy()
        del img_c, bb_c, out_c
        canvas = pred[i].copy()
        canvas[y0:y1, x0:x1] = m_c
        iou = mask_iou(canvas, pred[i])
        final = paste(pred[i], canvas, r)
        v = verdict(presence_c, bool(m_c.any()), iou, bool(final.any()), tau)
        rec.update(verdict=v, presence_crop=presence_c, iou_head_crop=iou_c, iou_crop_vs_full=iou,
                   rides_edge=rides_edge(canvas, win, hw))
        if v == "accepted":
            if others_count is None:
                others_count = pred.sum(0, dtype=np.int16)       # the round-2 masks of every object, before any paste
                round2_masks = pred.copy()
            others = (others_count - round2_masks[i].astype(np.int16)) > 0     # O(HW) per object instead of O(B HW)
            added, removed = final & ~pred[i], pred[i] & ~final
            rec.update(px_final=int(final.sum()), added=int(added.sum()), removed=int(removed.sum()),
                       steal_px=int((added & others).sum()))
            if added.any() or removed.any():
                n["changed"] += 1
            n["px_added"] += int(added.sum())
            n["px_removed"] += int(removed.sum())
            pred[i] = final
        else:
            rec.update(px_final=rec["px_full"], added=0, removed=0)
        n[v] += 1
        records[str(int(o))] = rec
    n["seconds"] = time.perf_counter() - t0
    return pred, records, n
