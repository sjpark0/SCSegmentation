"""J (region IoU) and F (DAVIS boundary F-measure) for one (ground truth, prediction) pair.

`db_eval_iou`, `seg2bmap`, `disk`, `db_eval_boundary` and `crop_box` are the DAVIS toolkit
formulation as used by ../SCSegmentation/Data/MVSeg/eval_jf.py, copied so both benchmarks score
with the same arithmetic.  The boundary tolerance is 0.8 % of the image diagonal (DAVIS default).
"""
import math

import cv2
import numpy as np

BOUND_TH = 0.008


def db_eval_iou(segmentation, annotation):
    segmentation = segmentation.astype(bool)
    annotation = annotation.astype(bool)
    inters = np.count_nonzero(segmentation & annotation)
    union = np.count_nonzero(segmentation | annotation)
    return 1.0 if union == 0 else inters / union


def seg2bmap(seg):
    """Boundary map of a binary mask - the DAVIS toolkit's formulation."""
    seg = seg.astype(bool)
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = (seg ^ e) | (seg ^ s) | (seg ^ se)
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def disk(radius):
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ((x * x + y * y) <= r * r).astype(np.uint8)


def bound_pix(h, w):
    return int(math.ceil(BOUND_TH * math.hypot(h, w)))


def db_eval_boundary(foreground_mask, gt_mask, bound_px, se):
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)

    n_fg = np.count_nonzero(fg_boundary)
    n_gt = np.count_nonzero(gt_boundary)
    if n_fg == 0 and n_gt == 0:
        return 1.0
    if n_fg == 0 or n_gt == 0:
        return 0.0

    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), se)
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), se)

    gt_match = np.count_nonzero(gt_boundary & fg_dil.astype(bool))
    fg_match = np.count_nonzero(fg_boundary & gt_dil.astype(bool))

    precision = fg_match / n_fg
    recall = gt_match / n_gt
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def crop_box(a, b, margin, shape):
    """Union bounding box of two masks, grown by `margin` and clipped.

    Everything outside it is background in both masks, so J and F computed on the crop match the
    full-frame values as long as the margin is at least the dilation radius.
    """
    rows = np.flatnonzero(a.any(1) | b.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(a.any(0) | b.any(0))
    y0 = max(int(rows[0]) - margin, 0)
    y1 = min(int(rows[-1]) + margin + 1, shape[0])
    x0 = max(int(cols[0]) - margin, 0)
    x1 = min(int(cols[-1]) + margin + 1, shape[1])
    return y0, y1, x0, x1


def jf_pair(gt, pred, bp, se):
    """(J, F) of one pair, computed on the union crop grown by bp + 2 (identical to full frame)."""
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    box = crop_box(gt, pred, bp + 2, gt.shape)
    if box is None:
        return 1.0, 1.0
    y0, y1, x0, x1 = box
    g, p = gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]
    return db_eval_iou(p, g), db_eval_boundary(p, g, bp, se)
