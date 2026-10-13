import cv2
import numpy as np
import pytest

from mvseg2d.io import (BadMaskValues, labelmap_objects, labelmap_to_mask, load_labelmap,
                        read_pred_mask, size_bin, write_mask)


def test_size_bin_edges():
    assert size_bin(1999) == "<2000"
    assert size_bin(2000) == "2000-10000"
    assert size_bin(10000) == ">=10000"


def test_labelmap_roundtrip(tmp_path):
    lab = np.zeros((4, 5), np.uint8)
    lab[1:3, 1:3] = 7
    lab[0, 4] = 2
    cv2.imwrite(str(tmp_path / "m.png"), lab)
    got = load_labelmap(str(tmp_path / "m.png"))
    assert labelmap_objects(got) == [2, 7]
    assert labelmap_to_mask(got, 7).sum() == 4


def test_write_then_read_pred_mask(tmp_path):
    m = np.zeros((3, 3), bool)
    m[1, 1] = True
    write_mask(str(tmp_path / "a" / "1.png"), m)
    back = read_pred_mask(str(tmp_path / "a" / "1.png"))
    assert back.dtype == bool and back.sum() == 1


def test_read_pred_rejects_non_binary(tmp_path):
    cv2.imwrite(str(tmp_path / "x.png"), np.array([[0, 1]], np.uint8))
    with pytest.raises(BadMaskValues):
        read_pred_mask(str(tmp_path / "x.png"))


def test_read_pred_accepts_rgb_channel0(tmp_path):
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[0, 0] = 255
    cv2.imwrite(str(tmp_path / "x.png"), rgb)
    assert read_pred_mask(str(tmp_path / "x.png")).sum() == 1


def test_read_pred_rejects_coloured_rgb(tmp_path):
    """A red PIL-style mask must not be scored as all-background: channels must agree."""
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[0, 0] = (0, 0, 255)                  # red in cv2's BGR order -> channel 0 is 0
    cv2.imwrite(str(tmp_path / "x.png"), bgr)
    with pytest.raises(BadMaskValues):
        read_pred_mask(str(tmp_path / "x.png"))


def test_read_pred_accepts_grey_with_alpha(tmp_path):
    bgra = np.zeros((2, 2, 4), np.uint8)
    bgra[..., 3] = 255
    bgra[1, 1, :3] = 255
    cv2.imwrite(str(tmp_path / "x.png"), bgra)
    assert read_pred_mask(str(tmp_path / "x.png")).sum() == 1


def test_read_pred_rejects_palette_png(tmp_path):
    from PIL import Image
    im = Image.new("P", (3, 3))
    im.putpalette([0, 0, 0, 0, 255, 0] + [0] * 762)   # index 1 = green
    im.putpixel((1, 1), 1)
    im.save(str(tmp_path / "p.png"))
    with pytest.raises(BadMaskValues):
        read_pred_mask(str(tmp_path / "p.png"))


def test_camera_order_is_numeric():
    from mvseg2d.io import camera_order
    assert camera_order(["v0", "v1", "v10", "v2"]) == ["v0", "v1", "v2", "v10"]
    assert camera_order(["S1_CAM_1", "S1_CAM_10", "S1_CAM_2"]) == ["S1_CAM_1", "S1_CAM_2", "S1_CAM_10"]
    assert camera_order(["cam00", "cam01", "cam16"]) == ["cam00", "cam01", "cam16"]
