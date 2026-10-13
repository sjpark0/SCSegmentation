import numpy as np

from methods.pi3_sam3.run import prompts_from_mask


def test_empty_mask_gives_none():
    assert prompts_from_mask(np.zeros((10, 10), bool)) is None


def test_rectangle_box_and_interior_point():
    m = np.zeros((50, 60), bool)
    m[10:30, 5:35] = True                                   # rows 10..29, cols 5..34
    pts, labels = prompts_from_mask(m)
    assert labels == [2, 3, 1]
    assert pts[0] == (5, 10) and pts[1] == (34, 29)
    x, y = pts[2]
    assert m[int(y), int(x)]


def test_l_shape_positive_point_lies_on_mask():
    m = np.zeros((60, 60), bool)
    m[10:50, 10:20] = True
    m[40:50, 10:50] = True                                   # L: the bbox centre (30, 30) is NOT on the mask
    pts, labels = prompts_from_mask(m)
    x, y = pts[2]
    assert m[int(y), int(x)] and not m[30, 30]


def test_refine_propagates_source_error(fake_data):
    """A source scene whose RUN.json carries an error must fail here too, not become empty predictions."""
    import argparse
    import pytest
    from methods import common
    from methods.pi3_sam3.run import refine
    sc = common.load_scene(fake_data, "S1")
    args = argparse.Namespace(data=fake_data, out="x", source="broken")
    with pytest.raises(RuntimeError, match="boom"):
        refine(sc, args)
