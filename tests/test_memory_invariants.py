"""T7: the seven memory-optimization invariants (docs/memory-optimization.md S1-S7) on CPU.

P13 companion; container only (torch + sam3).  S6 is a source-level check because
building SCSam3Video needs a GPU.
"""
import importlib
import os
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sam3")

from conftest import PKG  # noqa: E402

SENTINEL = "_THIS_FRAME_HAS_OUTPUTS_"


def src(name):
    return open(os.path.join(PKG, name), encoding="utf-8").read()


def test_s1_sentinel_processing_order():
    import SCSam3VideoInferenceNewMem as V
    n, start = 100, 5
    state = {"num_frames": n, "previous_stages_out": [None] * n}
    state["previous_stages_out"][start] = SENTINEL          # truthy for `is not None` readers
    fn = V.SCSam3VideoInferenceNewMem._get_processing_order
    order, end = fn(None, state, None, 21, False)            # start derived from the sentinel
    assert list(order) == list(range(start, start + 22)) and end == start + 21
    order2, _ = fn(None, state, start, 21, False)
    assert list(order2) == list(order)
    rev, _ = fn(None, state, 30, 10, True)
    assert list(rev) == list(range(29, 19, -1))
    with pytest.raises(RuntimeError):
        fn(None, {"num_frames": n, "previous_stages_out": [None] * n}, None, 21, False)
    assert src("SCSam3VideoInferenceNewMem.py").count(f'["previous_stages_out"][frame_idx] = "{SENTINEL}"') >= 2


def test_s2_no_video_level_mask_inputs_write():
    text = src("SCSam3VideoInferenceNewMem.py")
    double_index_write = re.compile(r'inference_state\["mask_inputs_per_obj"\]\[[^\]]+\]\[[^\]]+\]\s*=')
    assert double_index_write.search(text) is None
    # the per-object registration (single index, empty dict) is still there
    assert 'inference_state["mask_inputs_per_obj"][obj_id] = {}' in text


def test_s3_meta_placeholder_contract():
    shape = (1, 1, 48, 64)
    t = torch.empty(shape, dtype=torch.bool, device="meta")
    assert t.is_meta and t.device.type == "meta"
    assert tuple(t.shape) == shape and t.numel() == 48 * 64 and t.dtype == torch.bool
    with pytest.raises(NotImplementedError):                 # no real bytes behind it
        t.tolist()
    with pytest.raises(NotImplementedError):
        t.cpu().numpy()
    text = src("SCSam3TrackerPredictorNewMem.py")
    assert re.search(r'mask_inputs_per_frame\[frame_idx\] = torch\.empty\(\s*mask_inputs_video_res\.shape,'
                     r'\s*dtype=torch\.bool,\s*device="meta"', text)


def test_s4_bool_fast_path_equals_float_path(cpu_tensors):
    from harness import bare_tracker
    tr = bare_tracker()
    g = torch.Generator().manual_seed(0)
    for n in (1, 2, 3, 5):
        for _ in range(3):
            masks = torch.rand((n, 1, 16, 20), generator=g) > 0.5
            scores = torch.rand((n, 1), generator=g) * 2 - 1
            fast = tr._apply_object_wise_non_overlapping_constraints(masks, scores, background_value=0)
            assert fast.dtype == torch.bool and fast.shape == masks.shape
            slow = tr._apply_object_wise_non_overlapping_constraints(masks.float(), scores, background_value=-10.0)
            assert slow.dtype == torch.float32
            assert torch.equal(fast, slow > 0)
            assert torch.equal(fast > 0, slow > 0)


def test_s5_async_frame_float32():
    import cv2
    import io_utils
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(3, 3, 3), dtype=np.uint8)
    loader = io_utils.AsyncVideoFrameCPUToGPU([img], image_size=8, offload_video_to_cpu=True)
    out = loader[0]
    assert out.dtype == torch.float32 and tuple(out.shape) == (3, 8, 8)
    ref = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (8, 8)) / 255.0    # float64, as io_utils
    ref = torch.from_numpy(ref).permute(2, 0, 1)
    ref = ((ref - 0.5) / 0.5).to(torch.float32)                              # cast once, at the end
    assert torch.equal(out, ref)
    assert loader[0] is out                                                   # cached frame


def test_s6_retire_spatial_predictor_source():
    text = src("SCSam3Video.py")
    assert "def RetireSpatialPredictor(self)" in text
    assert "self.uses_spatial_predictor = True" in text
    assert "self.predictor_spatial = None" in text


def test_s7_trim_flag_parsing_and_trim(monkeypatch):
    import SCSam3VideoInferenceNewMem as V
    key = "SCSAM3_TRIM_CACHED_OUTPUTS"
    try:
        for val in ("", " ", "0"):
            monkeypatch.setenv(key, val)
            importlib.reload(V)
            assert V.TRIM_CACHED_OUTPUTS is False, val
        for val in ("1", "false", "off", "00"):             # the documented trap: only ""/blank/"0" are off
            monkeypatch.setenv(key, val)
            importlib.reload(V)
            assert V.TRIM_CACHED_OUTPUTS is True, val
        fn = V.SCSam3VideoInferenceNewMem._trim_cached_frame_outputs
        state = {"action_history": [{"type": "add"}], "cached_frame_outputs": {f: {} for f in range(10)}}
        fn(None, state, 4)
        assert sorted(state["cached_frame_outputs"]) == [4, 5, 6, 7, 8, 9]
        state = {"action_history": [{"type": "add"}], "cached_frame_outputs": {f: {} for f in range(10)}}
        fn(None, state, 4, reverse=True)
        assert sorted(state["cached_frame_outputs"]) == [0, 1, 2, 3, 4]
        state = {"action_history": [{"type": "propagation_full"}],
                 "cached_frame_outputs": {f: {} for f in range(10)}}
        fn(None, state, 4)
        assert sorted(state["cached_frame_outputs"]) == list(range(10))
        monkeypatch.setenv(key, "0")
        importlib.reload(V)
        state = {"action_history": [], "cached_frame_outputs": {f: {} for f in range(10)}}
        V.SCSam3VideoInferenceNewMem._trim_cached_frame_outputs(None, state, 9)
        assert len(state["cached_frame_outputs"]) == 10                     # flag off: no-op
    finally:
        monkeypatch.delenv(key, raising=False)
        importlib.reload(V)
        assert V.TRIM_CACHED_OUTPUTS is False
