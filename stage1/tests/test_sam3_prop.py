import os

from methods import common
from methods.sam3_prop.run import build_pseudo_video, view_order


def test_view_order_ref_outward():
    assert view_order("ref_outward", 5, 2) == [2, 1, 3, 0, 4]
    assert view_order("ref_outward", 4, 0) == [0, 1, 2, 3]
    assert view_order("ref_outward", 4, 3) == [3, 2, 1, 0]


def test_view_order_index():
    assert view_order("index", 3, 0) == [0, 1, 2]


def test_build_pseudo_video_symlinks_in_sequence(fake_data, tmp_path):
    sc = common.load_scene(fake_data, "S1")
    seq = view_order("ref_outward", len(sc.cams), sc.cams.index(sc.ref_cam))
    folder = str(tmp_path / "pv")
    build_pseudo_video(sc, sc.cams, seq, folder)
    names = sorted(os.listdir(folder))
    assert names == [f"{i:06d}.jpg" for i in range(5)]
    assert os.path.realpath(os.path.join(folder, "000000.jpg")) == os.path.realpath(sc.image_paths["v2"])
    assert os.path.realpath(os.path.join(folder, "000001.jpg")) == os.path.realpath(sc.image_paths["v1"])
    # rebuilding into the same folder replaces stale links
    build_pseudo_video(sc, sc.cams, list(range(5)), folder)
    assert os.path.realpath(os.path.join(folder, "000000.jpg")) == os.path.realpath(sc.image_paths["v0"])


def test_build_pseudo_video_follows_ring_order(fake_data, tmp_path):
    """meta.json ring_order (CROOM): the pseudo-video walks the rig order, not the index order."""
    sc = common.load_scene(fake_data, "S1")
    assert sc.ring_order is None
    ring = [sc.cams[i] for i in (0, 3, 1, 4, 2)]
    seq = view_order("index", 5, 0)
    folder = str(tmp_path / "pv_ring")
    build_pseudo_video(sc, ring, seq, folder)
    assert os.path.realpath(os.path.join(folder, "000001.jpg")) == os.path.realpath(sc.image_paths[sc.cams[3]])
    assert os.path.realpath(os.path.join(folder, "000004.jpg")) == os.path.realpath(sc.image_paths[sc.cams[2]])
