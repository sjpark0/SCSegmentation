"""SCSam3/long_run.py: the long-run helpers are output-neutral (host; numpy, cv2)."""
import os
import random
import threading

import pytest

from conftest import REPO, _load

np = pytest.importorskip("numpy")
lr = _load("long_run", os.path.join(REPO, "SCSam3", "long_run.py"))


def frame_filter(nc, frame_idx, num_frames, max_obj_ptrs=16, thr=0.01):
    """sam3_tracker_base.frame_filter (stride 1, forward), vendored for the property test."""
    if frame_idx == 0:
        return []
    max_num = min(num_frames, max_obj_ptrs)
    valid = []
    for i in range(frame_idx - 1, 0, -1):
        if i not in nc or "eff_iou_score" not in nc[i]:
            continue
        if nc[i]["eff_iou_score"] > thr:
            valid.insert(0, i)
        if len(valid) >= max_num - 1:
            break
    if frame_idx - 1 not in valid:
        valid.append(frame_idx - 1)
    return valid


class Tracker:
    max_obj_ptrs_in_encoder, mf_threshold, use_memory_selection = 16, 0.01, True


@pytest.mark.parametrize("seed", range(6))
def test_prune_never_changes_what_memory_selection_returns(seed):
    rnd = random.Random(seed)
    T0, N, num_frames = 900, 3000, 9938
    full, pruned = {}, {}
    ts = {"output_dict": {"non_cond_frame_outputs": pruned}, "output_dict_per_obj": {0: {"non_cond_frame_outputs": {}}}}
    pr = lr.MemPrune(Tracker(), num_frames)
    p_low = rnd.choice([0.0, 0.3, 0.7, 0.95])                     # long stretches of invalid frames (occlusion)
    for t in range(T0 + 1, T0 + N):
        assert frame_filter(full, t, num_frames) == frame_filter(pruned, t, num_frames), t
        e = {"eff_iou_score": 0.0 if rnd.random() < p_low else rnd.random()} if rnd.random() > 0.02 else {}
        full[t] = dict(e)
        pruned[t] = dict(e)
        ts["output_dict_per_obj"][0]["non_cond_frame_outputs"][t] = pruned[t]
        pr.prune_state(ts, t)
        assert len(pruned) <= pr.K + 2
        assert set(ts["output_dict_per_obj"][0]["non_cond_frame_outputs"]) == set(pruned)
    assert pr.pruned > 0 and pr.checked > 0


def test_guard_trips_on_vanished_or_rescored_entries():
    ts = {"output_dict": {"non_cond_frame_outputs": {}}, "output_dict_per_obj": {}}
    nc = ts["output_dict"]["non_cond_frame_outputs"]
    pr = lr.MemPrune(Tracker(), 100)
    for t in range(1, 30):
        nc[t] = {"eff_iou_score": 0.5}
        pr.prune_state(ts, t)
    kept = sorted(f for f in nc if f < 28)
    nc[kept[0]]["eff_iou_score"] = 0.6                              # rescored
    with pytest.raises(lr.PruneGuardError):
        pr.prune_state(ts, 30)
    nc[kept[0]]["eff_iou_score"] = 0.5
    del nc[kept[1]]                                                  # vanished
    with pytest.raises(lr.PruneGuardError):
        pr.prune_state(ts, 30)


def test_memory_selection_off_is_refused():
    class Off(Tracker):
        use_memory_selection = False
    with pytest.raises(lr.PruneGuardError):
        lr.MemPrune(Off(), 100)


def test_lazy_crops_equal_the_list_crops(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.RandomState(0)
    for f in range(5, 12):
        cv2.imwrite(str(tmp_path / f"{f:06d}.jpg"), (rng.rand(60, 80, 3) * 255).astype(np.uint8))
    ct = _load("crop_track", os.path.join(REPO, "SCSam3", "crop_track.py"))
    frames = ct.read_frames(cv2, str(tmp_path), 5, 7)
    lst = [np.ascontiguousarray(f[10:42, 20:52]) for f in frames]
    lazy = lr.LazyCrops(cv2, np, str(tmp_path), 5, 7, 20, 10, 32, (60, 80))
    assert len(lazy) == 7
    for k in (0, 3, 3, 6, 1, 0):
        assert np.array_equal(lazy[k], lst[k]) and lazy[k].flags["C_CONTIGUOUS"]
    with pytest.raises(IndexError):
        lazy[7]
    with pytest.raises(ValueError):
        lr.LazyCrops(cv2, np, str(tmp_path), 5, 7, 20, 10, 32, (61, 80))[0]


def test_async_writer_writes_everything_and_reraises(tmp_path):
    cv2 = pytest.importorskip("cv2")
    w = lr.AsyncWriter(cv2, 4, depth=8)
    arrs = {}
    for i in range(50):
        a = (np.random.RandomState(i).rand(16, 16) > 0.5).astype(np.uint8) * 255
        arrs[i] = a
        w.write(str(tmp_path / f"{i}.png"), a)
    w.drain()
    for i, a in arrs.items():
        assert np.array_equal(cv2.imread(str(tmp_path / f"{i}.png"), cv2.IMREAD_GRAYSCALE), a)
    w.write(str(tmp_path / "nodir" / "x.png"), arrs[0])           # imwrite returns False -> IOError
    with pytest.raises(Exception):
        w.drain()
    w.pool.shutdown(wait=True)


def test_io_cache2_is_a_transparent_two_entry_cache():
    calls = []

    class Loader:
        def __init__(self):
            self.frame_idx = -1

        def __getitem__(self, i):
            calls.append(i)
            self.frame_idx = i
            return ("frame", i)

    class Mod:
        AsyncVideoFrameLoader = Loader
        AsyncVideoFrameCPUToGPU = type("G", (Loader,), {})
    lr.install_io_cache2(Mod)
    lr.install_io_cache2(Mod)                                        # idempotent
    x = Mod.AsyncVideoFrameLoader()
    assert [x[i] for i in (1, 2, 1, 2, 3, 1)] == [("frame", i) for i in (1, 2, 1, 2, 3, 1)]
    assert calls == [1, 2, 3, 1]
