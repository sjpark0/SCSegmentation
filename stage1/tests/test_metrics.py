import numpy as np

from mvseg2d.metrics import bound_pix, db_eval_boundary, db_eval_iou, disk, jf_pair


def _sq(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_bound_pix_1080p():
    assert bound_pix(1080, 1920) == 18


def test_identical_masks_are_perfect():
    g = _sq(100, 100, 20, 60, 20, 60)
    bp = bound_pix(100, 100)
    assert jf_pair(g, g.copy(), bp, disk(bp)) == (1.0, 1.0)


def test_disjoint_masks_are_zero():
    g = _sq(100, 100, 0, 10, 0, 10)
    p = _sq(100, 100, 50, 60, 50, 60)
    bp = bound_pix(100, 100)
    assert jf_pair(g, p, bp, disk(bp)) == (0.0, 0.0)


def test_half_overlap_iou():
    g = _sq(50, 50, 0, 10, 0, 20)
    p = _sq(50, 50, 0, 10, 10, 30)
    assert abs(db_eval_iou(p, g) - 1 / 3) < 1e-9


def test_one_pixel_shift_within_tolerance_has_F_one():
    g = _sq(200, 200, 50, 100, 50, 100)
    p = _sq(200, 200, 51, 101, 50, 100)
    bp = bound_pix(200, 200)
    J, F = jf_pair(g, p, bp, disk(bp))
    assert F == 1.0 and 0.9 < J < 1.0


def test_crop_matches_full_frame():
    g = _sq(300, 400, 100, 150, 100, 180)
    p = _sq(300, 400, 110, 160, 90, 170)
    bp = bound_pix(300, 400)
    se = disk(bp)
    assert jf_pair(g, p, bp, se) == (db_eval_iou(p, g), db_eval_boundary(p, g, bp, se))
