import json
import os

import cv2
import numpy as np
import pytest

from geomem.data import Sample, choose_objects, enumerate_samples, objects_for, unannotated_views


def _add_frames(data, scene, frames, cams, mask_cams, hw=(40, 60)):
    for t in frames:
        d = os.path.join(data, scene, "frames", f"{t:06d}")
        os.makedirs(os.path.join(d, "images")); os.makedirs(os.path.join(d, "masks"))
        for c in cams:
            cv2.imwrite(os.path.join(d, "images", f"{c}.jpg"), np.zeros((*hw, 3), np.uint8))
        for c in mask_cams:
            lab = np.zeros(hw, np.uint8); lab[5:15, 5:25] = 1; lab[20:30, 30:50] = 2 if c != "v4" else 0
            cv2.imwrite(os.path.join(d, "masks", f"{c}.png"), lab)
    p = os.path.join(data, scene, "meta.json"); m = json.load(open(p)); m["train_frames"] = list(frames); json.dump(m, open(p, "w"))


@pytest.fixture
def frames_data(fake_data):
    _add_frames(fake_data, "S1", [0, 1, 2], ["v0", "v1", "v2", "v3", "v4"], ["v0", "v2", "v4"])
    return fake_data


def test_enumerate_samples_count(frames_data):
    s = enumerate_samples(frames_data, ["S1"])
    assert len(s) == 18                                   # 3 frames x 3 refs x 2 targets
    assert s[0] == Sample("S1", 0, "v0", "v2")
    assert all(x.ref_cam != x.tgt_cam for x in s) and all(x.ref_cam in ("v0", "v2", "v4") for x in s)


def test_objects_for_reads_reference_labels(frames_data):
    assert objects_for(Sample("S1", 1, "v0", "v2"), frames_data) == [1, 2]
    assert objects_for(Sample("S1", 1, "v4", "v2"), frames_data) == [1]


def test_sample_object_cap():
    rng = np.random.default_rng(0)
    ids = list(range(1, 21))
    got = choose_objects(ids, 16, rng)
    assert len(got) == 16 and len(set(got)) == 16 and set(got) <= set(ids)
    assert choose_objects([3, 1, 2], 16, rng) == [3, 1, 2]


def test_unannotated_views_excludes_annotated(frames_data):
    rng = np.random.default_rng(0)
    views = unannotated_views(frames_data, "S1", 2, 4, rng)
    assert 2 <= len(views) <= 2 and set(views) == {"v1", "v3"}     # S1 has exactly two unannotated cameras


def test_enumerate_samples_frame_stride(frames_data):
    s = enumerate_samples(frames_data, ["S1"], frame_stride=2)
    assert {x.t for x in s} == {0, 2} and len(s) == 12
    assert len(enumerate_samples(frames_data, ["S1"], frame_stride=1)) == 18


def test_third_camera(frames_data):
    from geomem.data import third_camera
    assert third_camera(frames_data, Sample("S1", 0, "v0", "v2")) == "v4"
    assert third_camera(frames_data, Sample("S1", 0, "v0", "v4")) == "v2"


def test_enumerate_pairs_count_and_targets(frames_data):
    from geomem.data import enumerate_pairs, pair_targets
    pairs = enumerate_pairs(frames_data, ["S1"], frame_stride=1)
    assert len(pairs) == 3 * 3
    assert pair_targets(frames_data, pairs[0]) == sorted(c for c in ["v0", "v2", "v4"] if c != pairs[0].ref_cam)
    assert len(enumerate_pairs(frames_data, ["S1"], frame_stride=2)) == 2 * 3


def test_hub_views_small_scene_uses_all():
    from geomem.data import hub_views
    v = hub_views(["v0", "v1", "v2", "v3", "v4"], "v2", ["v0", "v4"], k_random=5, rng=np.random.default_rng(0))
    assert v[0] == "v2" and v[1:3] == ["v0", "v4"] and sorted(v) == ["v0", "v1", "v2", "v3", "v4"]


def test_hub_views_large_scene_picks_k_random():
    from geomem.data import hub_views
    cams = [f"c{i}" for i in range(20)]
    v = hub_views(cams, "c0", ["c1", "c2"], k_random=5, rng=np.random.default_rng(0))
    assert len(v) == 8 and len(set(v)) == 8 and v[:3] == ["c0", "c1", "c2"]


def test_load_sample_pair_missing_hub_cache_names_file(frames_data):
    """Only the RANDOM hub view v1 lacks its cache (reference and targets are cached): the error must name v1's file."""
    from geomem.cache import feat_cache_path
    from geomem.data import PairSample, load_sample_pair
    for cam in ("v0", "v2", "v4"):
        p = feat_cache_path(frames_data, "S1", 0, cam)
        os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").close()
    s = PairSample("S1", 0, "v0")
    with pytest.raises(FileNotFoundError) as e:
        load_sample_pair(frames_data, s, [1], ["v0", "v2", "v4", "v1"], device="cpu")
    assert feat_cache_path(frames_data, "S1", 0, "v1") in str(e.value)


# ---------------------------------------------------------------- GeoSet-2: predicted-memory training samples
def test_random_memory_camera_excludes_ref_and_target(frames_data):
    from geomem.data import random_memory_camera
    rng = np.random.default_rng(0)
    seen = {random_memory_camera(frames_data, Sample("S1", 0, "v0", "v2"), rng) for _ in range(200)}
    assert seen == {"v1", "v3", "v4"}


def test_load_sample_pred_memory_missing_cache(frames_data):
    pytest.importorskip("torch")
    from geomem.data import load_sample
    with pytest.raises(FileNotFoundError) as e:
        load_sample(frames_data, Sample("S1", 0, "v0", "v2"), [1], device="cpu", extra_mem_cam="v1", extra_mem_gt=False)
    assert ".pt" in str(e.value) and "v1" in str(e.value)


def test_memory_rng_is_per_step_and_leaves_the_main_stream_alone():
    from geomem.data import memory_rng
    a, b = memory_rng(0, 7), memory_rng(0, 7)
    assert a.integers(1000) == b.integers(1000)                         # deterministic per (seed, step)
    assert memory_rng(0, 8).integers(1000) != a.integers(1000) or memory_rng(0, 9).integers(1000) != b.integers(1000)
    main = np.random.default_rng(0)
    before = main.random()
    memory_rng(0, 7).integers(1000)
    assert np.random.default_rng(0).random() == before                  # the shared stream is not consumed


def test_load_sample_aligned_mem_missing_cache(frames_data):
    pytest.importorskip("torch")
    from geomem.data import load_sample
    with pytest.raises(FileNotFoundError) as e:
        load_sample(frames_data, Sample("S1", 0, "v0", "v2"), [1], device="cpu", extra_mem_cam="v1", extra_mem_gt=False,
                    aligned_mem=True)
    assert "sam3_feats72" in str(e.value)
