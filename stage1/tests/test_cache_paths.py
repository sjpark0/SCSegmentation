import json
import os

from geomem.cache import feat_cache_path, missing_feats, pi3_frame_cache


def test_feat_cache_path():
    assert feat_cache_path("Data", "Blocks", 7, "cam0") == os.path.join("Data", "Blocks", "Cache", "sam3_feats", "000007", "cam0.pt")


def test_pi3_frame_cache_path():
    assert pi3_frame_cache("Data", "Frog", 180) == os.path.join("Data", "Frog", "Cache", "pi3_frames", "000180")


def test_missing_feats_lists_uncached_annotated_views(fake_data):
    meta_p = os.path.join(fake_data, "S1", "meta.json")
    meta = json.load(open(meta_p)); meta["train_frames"] = [0, 1]; json.dump(meta, open(meta_p, "w"))
    p = feat_cache_path(fake_data, "S1", 0, "v0")
    os.makedirs(os.path.dirname(p)); open(p, "wb").close()
    missing = missing_feats(fake_data, "S1")
    assert (0, "v0") not in missing
    assert set(missing) == {(0, "v2"), (0, "v4"), (1, "v0"), (1, "v2"), (1, "v4")}


def test_pos_enc_path():
    from geomem.cache import pos_enc_path
    assert pos_enc_path("Data", "Blocks") == os.path.join("Data", "Blocks", "Cache", "sam3_feats", "pos_enc.pt")


def test_missing_feats_all_views_and_stride(fake_data):
    meta_p = os.path.join(fake_data, "S1", "meta.json")
    meta = json.load(open(meta_p)); meta["train_frames"] = [0, 1, 2]; json.dump(meta, open(meta_p, "w"))
    missing = missing_feats(fake_data, "S1", all_views=True, frame_stride=2)
    assert set(missing) == {(t, c) for t in (0, 2) for c in ("v0", "v1", "v2", "v3", "v4")}
