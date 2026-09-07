"""T9: XW mode E, the two-pass path (SPEC_P4 sections 2.2 vi, 2.3, 2.7).

Host part: the torch-free `run_two_pass` driver of SCSam3/runMVSeg.py with fakes, and
source-level checks that the default generator paths carry none of the new tokens.
Container part: `recompute_frame_all_views` (Jacobi compute-then-commit) on a bare
inference object with a stub tracker, and the tracker's `recompute_frame` /
`commit_frame` refusals and stores.
"""
import ast
import os
from collections import defaultdict

import pytest

from conftest import PKG, load_runmvseg


# ------------------------------------------------------------------ host: driver
def test_run_two_pass_driver(tmp_path):
    np = pytest.importorskip("numpy")
    mod = load_runmvseg()
    start, nf = 30, 4
    cams, written = ["c0", "c1", "c2", "c3"], ["c0", "c2"]
    log = []

    def pass1_mask(j):                      # all-True 2x2: the provisional mask
        return np.ones((1, 2, 2), dtype=np.float32) * (10 + j)

    def pass2_masks(j):                     # one True pixel per object: the written mask
        m = np.zeros((2, 2, 2), dtype=np.float32)
        m[0, 0, 0] = 100 + j
        m[1, 1, 1] = 100 + j
        return m

    def gen(j):
        for f in range(start, start + nf + 1):          # 22/21 contract: one more than pulled
            log.append(("next", j, f))
            yield {"frame_index": f,
                   "outputs": {"out_obj_ids": np.array([1]), "out_binary_masks": pass1_mask(j)}}

    class SC:
        track_views = [0, 1, 2, 3]

        def __init__(s):
            s.tracking_result = [gen(j) for j in range(4)]

        def RecomputeFrame(s, frame_idx, output_for=None):
            log.append(("recompute", frame_idx, tuple(output_for)))
            return [None if j not in output_for else
                    {"out_obj_ids": np.array([1, 2]), "out_binary_masks": pass2_masks(j)}
                    for j in range(4)]

    writes = []

    class CV2:
        @staticmethod
        def imwrite(path, arr):
            writes.append((path, arr.copy()))

    out = str(tmp_path / "SegMaskSam3XW1E")
    mod.run_two_pass(SC(), start, nf, cams, written, out, CV2, np)

    # next() order is t-major lockstep; one recompute per tracked frame, right after the
    # frame's last next() and before any next() of the following frame; none for the seed
    expected = []
    for f in range(start, start + nf):
        expected += [("next", j, f) for j in range(4)]
        if f != start:
            expected.append(("recompute", f, (0, 2)))
    assert log == expected
    # files: scored cameras only, <out>/<cam>/<frame>/<obj>.png; the seed frame from pass 1
    # (one object, all-True), every later frame from pass 2 (two objects, one pixel each)
    seen = {}
    for path, arr in writes:
        cam, frame, fn = os.path.relpath(path, out).split(os.sep)
        assert cam in written and fn.endswith(".png")
        assert arr.dtype == np.uint8 and arr.shape == (2, 2) and set(arr.flatten()) <= {0, 255}
        seen[(cam, int(frame), int(fn[:-4]))] = int(arr.sum() // 255)
    frames = range(start, start + nf)
    assert set(seen) == {(c, f, o) for c in written for f in frames for o in ((1,) if f == start else (1, 2))}
    for (cam, f, o), n_true in seen.items():
        assert n_true == (4 if f == start else 1), (cam, f, o)


# ------------------------------------------------------------------ host: source level
def _method_source(path, cls_name, fn_name):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    return ast.get_source_segment(src, fn), [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]


def test_default_generator_paths_untouched():
    tracker = os.path.join(PKG, "SCSam3TrackerPredictorNewMem.py")
    seg, names = _method_source(tracker, "SCSam3TrackerPredictorNewMem", "propagate_in_video")
    assert "xview_pass" not in seg and "recompute" not in seg
    assert "recompute_frame" in names and "commit_frame" in names
    inference = os.path.join(PKG, "SCSam3VideoInferenceNewMem.py")
    seg, names = _method_source(inference, "SCSam3VideoInferenceWithInstanceInteractivityNewMem",
                                "propagate_in_video")
    assert "recompute" not in seg
    assert "recompute_frame_all_views" in names          # lives on the subclass the builder returns
    seg, names = _method_source(inference, "SCSam3VideoInferenceNewMem", "propagate_in_video")
    assert "recompute" not in seg and "recompute_frame_all_views" not in names
    # the runner's legacy write loop is what it was; only the E branch calls the driver
    runner = open(os.path.join(os.path.dirname(PKG), "runMVSeg.py"), encoding="utf-8").read()
    assert runner.count("run_two_pass(") == 2            # definition + the E branch of main()


# ------------------------------------------------------------------ container
def _bare_tracker_module():
    pytest.importorskip("torch")
    pytest.importorskip("sam3")
    import harness
    return harness


def test_recompute_all_views_is_jacobi(cpu_tensors):
    torch = cpu_tensors
    harness = _bare_tracker_module()
    import SCSam3VideoInferenceNewMem as V
    real = harness.bare_tracker(xw=1, xh=True, xm="E")

    class Stub(V.SCSam3VideoInferenceWithInstanceInteractivityNewMem):
        @property
        def device(self):                                # the base property is read-only
            return torch.device("cpu")

    model = Stub.__new__(Stub)
    if isinstance(model, torch.nn.Module):
        torch.nn.Module.__init__(model)
    model.rank, model.world_size, model.fill_hole_area = 0, 1, 0
    calls = []

    class StubTracker:
        low_res_mask_size = 8
        cross_view_mode = "E"
        use_memory_selection = True

        def recompute_frame(s, iss, spatial_idx, frame_idx, reverse):
            st = iss[spatial_idx]
            snapshot = tuple(id(x["output_dict"]["non_cond_frame_outputs"][frame_idx])
                             for x in iss if x is not None)
            calls.append(("recompute", spatial_idx, st["obj_ids"][0], snapshot))
            out = {"maskmem_features": torch.zeros(1, harness.MEM, 8, 8),
                   "maskmem_pos_enc": [torch.zeros(1, harness.MEM, 8, 8)],
                   "pred_masks": torch.full((1, 1, 8, 8), 5.0),
                   "obj_ptr": torch.zeros(1, harness.C),
                   "object_score_logits": torch.tensor([[1.0]]),
                   "iou_score": torch.tensor([[0.9]]), "eff_iou_score": torch.tensor([[0.9]])}
            return (out, list(st["obj_ids"]), torch.full((1, 1, 8, 8), 5.0 + spatial_idx),
                    torch.tensor([[1.0 + spatial_idx + st["obj_ids"][0] / 10]]))

        def commit_frame(s, state, frame_idx, current_out, reverse):
            calls.append(("commit", state["_v"], state["obj_ids"][0]))
            real.commit_frame(state, frame_idx, current_out, reverse)

        def _apply_object_wise_non_overlapping_constraints(s, *a, **k):
            return real._apply_object_wise_non_overlapping_constraints(*a, **k)

    model.tracker = StubTracker()
    t, n_sessions, objs = 3, 3, [1, 2]
    states = []
    for v in range(n_sessions):
        trk = [{"_v": v, "obj_ids": [o],
                "output_dict": {"cond_frame_outputs": {0: {"seed": (v, o)}},
                                "non_cond_frame_outputs": {t: {"pass1": (v, o)}}},
                "output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
                "frames_already_tracked": {},
                "consolidated_frame_inds": {"cond_frame_outputs": {0}, "non_cond_frame_outputs": set()}}
               for o in objs]
        md = {"obj_ids_per_gpu": [__import__("numpy").array(objs)],
              "obj_id_to_score": {1: 0.5, 2: 0.6},
              "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
              "rank0_metadata": {"suppressed_obj_ids": defaultdict(set)}}
        states.append({"action_history": [{"type": "add", "obj_ids": [1], "frame_idx": 0},
                                          {"type": "propagation_partial", "obj_ids": list(objs), "frame_idx": 0}],
                       "tracker_inference_states": trk, "tracker_metadata": md,
                       "cached_frame_outputs": {t: {o: torch.zeros(1, 16, 16, dtype=torch.bool) for o in objs}},
                       "orig_height": 16, "orig_width": 16, "feature_cache": {t: "keep"}})
    pass1_ids = {id(s["output_dict"]["non_cond_frame_outputs"][t])
                 for st in states for s in st["tracker_inference_states"]}
    history_before = [list(s["action_history"]) for s in states]

    outs = model.recompute_frame_all_views(states, t, reverse=False, output_for=[0, 2])

    rec = [c for c in calls if c[0] == "recompute"]
    com = [c for c in calls if c[0] == "commit"]
    assert len(rec) == len(com) == n_sessions * len(objs)
    assert max(calls.index(c) for c in rec) < min(calls.index(c) for c in com)   # compute all, then commit
    assert [c[1:3] for c in rec] == [(v, o) for v in range(n_sessions) for o in objs]
    assert [c[1:3] for c in com] == [(v, o) for v in range(n_sessions) for o in objs]  # session order
    assert all(set(c[3]) <= pass1_ids and len(c[3]) == n_sessions for c in rec)       # Jacobi inputs
    # returned outputs: generator-shaped for output_for, None elsewhere
    assert outs[1] is None and outs[0] is not None and outs[2] is not None
    assert set(outs[0]) == {"out_obj_ids", "out_probs", "out_boxes_xywh", "out_binary_masks", "frame_stats"}
    assert outs[0]["out_obj_ids"].tolist() == [1, 2]
    assert outs[0]["out_binary_masks"].shape == (2, 16, 16) and outs[0]["out_binary_masks"].dtype == bool
    # state: history / feature cache untouched, caches hold CPU tensors, scores updated,
    # non_cond[t] of every object replaced by the pass-2 output (per-object slices too)
    assert [s["action_history"] for s in states] == history_before
    assert all(s["feature_cache"] == {t: "keep"} for s in states)
    for v, s in enumerate(states):
        cache = s["cached_frame_outputs"][t]
        assert set(cache) == set(objs)
        assert all(isinstance(m, torch.Tensor) and m.device.type == "cpu" and m.dtype == torch.bool
                   for m in cache.values())
        assert all(bool(m.any()) for m in cache.values())              # the recomputed masks, not zeros
        assert s["tracker_metadata"]["obj_id_to_tracker_score_frame_wise"][t] == {
            o: pytest.approx(1.0 + v + o / 10) for o in objs}
        for trk in s["tracker_inference_states"]:
            nc = trk["output_dict"]["non_cond_frame_outputs"][t]
            assert "pass1" not in nc and "maskmem_features" in nc
            assert trk["output_dict_per_obj"][0]["non_cond_frame_outputs"][t]["obj_ptr"].shape == (1, harness.C)
            assert trk["frames_already_tracked"] == {t: {"reverse": False}}
    # a session whose last action is not a suspended partial propagation is refused
    states[0]["action_history"].append({"type": "propagation_fetch", "obj_ids": objs, "frame_idx": 0})
    with pytest.raises(AssertionError):
        model.recompute_frame_all_views(states, t)


def test_tracker_recompute_refuses_prompt_frames(cpu_tensors):
    torch = cpu_tensors
    harness = _bare_tracker_module()
    tr = harness.bare_tracker(xw=1, xh=True, xm="E")
    st = {"obj_ids": [1],
          "output_dict": {"cond_frame_outputs": {0: {}}, "non_cond_frame_outputs": {}},
          "consolidated_frame_inds": {"cond_frame_outputs": {0}, "non_cond_frame_outputs": set()}}
    with pytest.raises(ValueError):
        tr.recompute_frame([st], 0, 0, False)             # the seed carries a prompt
    with pytest.raises(RuntimeError):
        tr.recompute_frame([st], 0, 3, False)             # no pass-1 output for frame 3
    with pytest.raises(AssertionError):
        harness.bare_tracker(xw=1, xh=True, xm="A").recompute_frame([st], 0, 3, False)
    # commit_frame: exactly :916 and :919-922 of propagate_in_video
    state = {"output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
             "output_dict_per_obj": {i: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
                                     for i in range(2)},
             "frames_already_tracked": {}}
    cur = {"maskmem_features": torch.arange(2.0).view(2, 1, 1, 1).expand(2, harness.MEM, 8, 8).clone(),
           "maskmem_pos_enc": [torch.zeros(2, harness.MEM, 8, 8)],
           "pred_masks": torch.zeros(2, 1, 8, 8), "obj_ptr": torch.arange(2.0).view(2, 1).expand(2, harness.C).clone(),
           "object_score_logits": torch.tensor([[1.0], [2.0]]), "iou_score": torch.tensor([[0.5], [0.6]])}
    tr.commit_frame(state, 7, cur, False)
    assert state["output_dict"]["non_cond_frame_outputs"] == {7: cur}
    assert state["output_dict"]["non_cond_frame_outputs"][7] is cur
    assert state["output_dict"]["cond_frame_outputs"] == {}
    for i in range(2):
        sl = state["output_dict_per_obj"][i]["non_cond_frame_outputs"][7]
        assert torch.equal(sl["obj_ptr"], cur["obj_ptr"][i:i + 1]) and sl["obj_ptr"].data_ptr() == cur["obj_ptr"][i:i + 1].data_ptr()
        assert torch.equal(sl["maskmem_features"], cur["maskmem_features"][i:i + 1])
        assert sl["iou_score"].item() == cur["iou_score"][i].item()
        assert state["output_dict_per_obj"][i]["cond_frame_outputs"] == {}
    assert state["frames_already_tracked"] == {7: {"reverse": False}}


def test_tracker_recompute_frame_pass2_call_and_deferred_store(cpu_tensors, monkeypatch):
    """The REAL recompute_frame is the pass-2 entry: it must hand
    _run_single_frame_inference_multiple the LIVE output_dict of every session (None where
    the session has no state) with xview_pass=2 and no prompt, on the default memory-encoder
    path, and store NOTHING until commit_frame."""
    import copy
    torch = cpu_tensors
    harness = _bare_tracker_module()
    tr = harness.bare_tracker(xw=1, xh=True, xm="E")
    t, objs = 5, [1, 2]

    def state(v):                                     # tensor-free: deepcopy/== are exact
        per_obj = {i: {"cond_frame_outputs": {0: {"seed": (v, i)}},
                       "non_cond_frame_outputs": {f: {"pass1": (v, f, i)} for f in range(1, t + 1)}}
                   for i in range(len(objs))}
        return {"obj_ids": list(objs), "obj_idx_to_id": dict(enumerate(objs)),
                "output_dict": {"cond_frame_outputs": {0: {"seed": v}},
                                "non_cond_frame_outputs": {f: {"pass1": (v, f)} for f in range(1, t + 1)}},
                "output_dict_per_obj": per_obj,
                "frames_already_tracked": {f: {"reverse": False} for f in range(1, t + 1)},
                "consolidated_frame_inds": {"cond_frame_outputs": {0}, "non_cond_frame_outputs": set()}}

    v, states = 1, [state(0), state(1), None]
    calls = []

    def fake(**kw):                                   # keyword-only: a positional call is a TypeError
        calls.append(kw)
        n = len(kw["inference_states"][kw["spatial_idx"]]["obj_ids"])
        out = {"maskmem_features": torch.full((n, harness.MEM, 8, 8), 2.0),
               "maskmem_pos_enc": [torch.zeros(n, harness.MEM, 8, 8)],
               "pred_masks": torch.full((n, 1, 8, 8), 3.0),
               "obj_ptr": torch.arange(float(n)).view(n, 1).expand(n, harness.C).clone(),
               "object_score_logits": torch.tensor([[7.0]] * n),
               "iou_score": torch.tensor([[0.8]] * n), "eff_iou_score": torch.tensor([[0.8]] * n)}
        return out, torch.full((n, 1, 8, 8), 3.0)
    monkeypatch.setattr(tr, "_run_single_frame_inference_multiple", fake)

    snap = copy.deepcopy([s for s in states if s is not None])
    pass1 = states[v]["output_dict"]["non_cond_frame_outputs"][t]
    cur, obj_ids, pred, scores = tr.recompute_frame(states, v, t, False)

    assert len(calls) == 1
    kw = calls[0]
    assert set(kw) == {"inference_states", "output_dicts", "spatial_idx", "frame_idx", "batch_size",
                       "is_init_cond_frame", "point_inputs", "mask_inputs", "reverse",
                       "run_mem_encoder", "xview_pass"}
    assert kw["xview_pass"] == 2
    assert kw["output_dicts"] == [s["output_dict"] if s else None for s in states]
    assert all(od is s["output_dict"] for od, s in zip(kw["output_dicts"], states) if s is not None)
    assert kw["inference_states"] is states
    assert (kw["spatial_idx"], kw["frame_idx"], kw["batch_size"]) == (v, t, len(objs))
    assert kw["is_init_cond_frame"] is False and kw["point_inputs"] is None and kw["mask_inputs"] is None
    assert kw["run_mem_encoder"] is True and kw["reverse"] is False
    # return contract: (current_out, obj_ids, low_res_masks, obj_scores)
    assert cur is not None and "object_score_logits" in cur
    assert obj_ids == objs and obj_ids is states[v]["obj_ids"]
    assert torch.equal(pred, torch.full((2, 1, 8, 8), 3.0)) and scores is cur["object_score_logits"]
    # nothing stored: every session is exactly what it was, the pass-1 entry of frame t included
    assert [s for s in states if s is not None] == snap
    assert states[v]["output_dict"]["non_cond_frame_outputs"][t] is pass1
    # commit_frame is the store, and it touches this session only
    tr.commit_frame(states[v], t, cur, False)
    assert states[v]["output_dict"]["non_cond_frame_outputs"][t] is cur
    for i in range(len(objs)):
        sl = states[v]["output_dict_per_obj"][i]["non_cond_frame_outputs"][t]
        assert torch.equal(sl["obj_ptr"], cur["obj_ptr"][i:i + 1]) and "pass1" not in sl
    assert states[v]["frames_already_tracked"][t] == {"reverse": False}
    assert states[0] == snap[0]
    # reverse / run_mem_encoder pass through unchanged; the pass is still 2
    tr.recompute_frame(states, 0, t, True, run_mem_encoder=False)
    assert (calls[-1]["reverse"], calls[-1]["run_mem_encoder"], calls[-1]["xview_pass"]) == (True, False, 2)
