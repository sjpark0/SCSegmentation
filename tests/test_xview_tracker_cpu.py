"""T2a-e: the REAL _prepare_memory_conditioned_features_multiple on CPU (bare instance).

Golden values are SPEC.md section 1 (N=10, t=5, lockstep), captured from the unmodified
code with the same harness.  Container only (torch + sam3).
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sam3")

from conftest import PKG  # noqa: E402
from harness import (GOLDEN_HW, SMALL_HW, bare_tracker, output_dicts_lockstep, run,  # noqa: E402
                     lockstep, same_inputs)

N, T5 = 10, 5


def own_tokens(v):
    """cond@0 (row 6) + non_cond frames 1..4 (t_pos 3..6 -> rows 3,2,1,0)."""
    return [1000 * v + f for f in range(0, 5)], [6, 3, 2, 1, 0]


def neighbour_tokens(v, W):
    """Under lockstep at t=5 a neighbour p = v+s_pos holds frame 5 iff 0 <= p (< v);
    with the legacy wrap, p < 0 maps to N+p >= v, which holds nothing -> None -> skipped."""
    src, rows = [], []
    for s in range(-W, 0):
        p = v + s
        if p >= 0:
            src.append(1000 * p + 5)
            rows.append(abs(s) - 1)
    return src, rows


# ----------------------------------------------------------------------- T2a
def test_real_method_legacy_golden(cpu_tensors):
    tr = bare_tracker(xw=4, xh=False)
    assert (tr.cross_view_window, tr.cross_view_hygiene) == (4, False)
    for v in range(N):
        r = run(tr, output_dicts_lockstep(N, v, T5), v, T5)
        osrc, orows = own_tokens(v)
        nsrc, nrows = neighbour_tokens(v, 4)
        assert r["n_mem"] == 5 + min(v, 4), (v, r)
        assert r["mem_src"] == osrc + nsrc, (v, r)
        assert r["tpos_rows"] == orows + nrows, (v, r)
        assert r["n_ptr_tokens"] == 16                       # 4 pointers x (256 // 64)
        first_ptr = 9000 if v <= 3 else 1000 * v              # C2: views 0..3 carry view 9's cond
        assert r["ptr_src"] == [first_ptr, 1000 * v + 4, 1000 * v + 3, 1000 * v + 2], (v, r)


# ----------------------------------------------------------------------- T2b
def test_real_method_hygiene(cpu_tensors):
    tr = bare_tracker(xw=4, xh=True)
    for v in range(N):
        r = run(tr, output_dicts_lockstep(N, v, T5), v, T5)
        assert r["n_mem"] == 5 + min(v, 4)
        assert r["ptr_src"][0] == 1000 * v                    # own cond, also for v <= 3
        assert r["mem_src"] == own_tokens(v)[0] + neighbour_tokens(v, 4)[0]
    tr0 = bare_tracker(xw=0, xh=True)
    for v in (0, 6):
        r = run(tr0, output_dicts_lockstep(N, v, T5), v, T5)
        assert r["n_mem"] == 5 and r["mem_src"] == own_tokens(v)[0]
        assert len(r["ptr_src"]) == 4 and r["n_ptr_tokens"] == 16
        assert r["ptr_src"][0] == 1000 * v
    tr6 = bare_tracker(xw=6, xh=True)
    for v in range(N):
        r = run(tr6, output_dicts_lockstep(N, v, T5), v, T5)
        k = min(v, 6)
        assert r["n_mem"] == 5 + k
        assert r["tpos_rows"] == [6, 3, 2, 1, 0] + [5, 4, 3, 2, 1, 0][6 - k:]
        assert r["mem_src"] == own_tokens(v)[0] + neighbour_tokens(v, 6)[0]
    r = run(bare_tracker(xw=4, xh=True), output_dicts_lockstep(3, 0, T5), 0, T5)   # no IndexError
    assert r["n_mem"] == 5


# ----------------------------------------------------------------------- T2c
def test_real_method_indexerror_n3_legacy(cpu_tensors):
    tr = bare_tracker(xw=4, xh=False)
    with pytest.raises(IndexError):
        run(tr, output_dicts_lockstep(3, 0, T5), 0, T5)


# ----------------------------------------------------------------------- T2d
def test_first_tracked_frame_seq_len(cpu_tensors):
    """t = start+1: no own non_cond memory; seq_len (C8) is defined by the cond entry."""
    tr = bare_tracker(xw=4, xh=False)
    r = run(tr, output_dicts_lockstep(N, 6, 1), 6, 1)
    assert r["n_mem"] == 5
    assert r["mem_src"] == [6000, 2001, 3001, 4001, 5001]
    assert r["tpos_rows"] == [6, 3, 2, 1, 0]
    assert r["n_ptr_tokens"] == 4 and r["ptr_src"] == [6000]
    r0 = run(bare_tracker(xw=0, xh=True), output_dicts_lockstep(N, 6, 1), 6, 1)
    assert r0["n_mem"] == 1 and r0["mem_src"] == [6000]


# ----------------------------------------------------------------------- T2e
def test_lockstep_21_frames_closure_equals_all(cpu_tensors):
    NN, K, T = 10, 5, 20
    h4 = bare_tracker(xw=4, xh=True)
    full = lockstep(h4, NN, T, hw=SMALL_HW)
    trunc = lockstep(h4, K, T, hw=SMALL_HW)
    bad = [(v, t) for v in range(K) for t in range(1, T + 1)
           if not same_inputs(full[(v, t)], trunc[(v, t)])]
    assert bad == []                                          # closure == all under hygiene
    # legacy: view 0 reads view N'-1's cond pointer, so the session count matters (C2)
    lp = bare_tracker(xw=None, xh=None)
    fl = lockstep(lp, NN, T, hw=SMALL_HW)
    tl = lockstep(lp, K, T, hw=SMALL_HW)
    assert any(not same_inputs(fl[(0, t)], tl[(0, t)]) for t in range(1, T + 1))
    # (views >= 4 wrap nothing themselves, but they read the memories views 0..3 stored,
    #  so the C2 difference propagates upward: legacy closure != all for every view)
    assert any(not same_inputs(fl[(4, t)], tl[(4, t)]) for t in range(1, T + 1))
    # W=0 never reads another session at all
    h0 = bare_tracker(xw=0, xh=True)
    a = lockstep(h0, NN, T, hw=SMALL_HW)
    assert all(same_inputs(full[(0, t)], a[(0, t)]) for t in range(1, T + 1))  # nb=0 identity (F4)

    # 21/22 contract: the inference module's processing order has num_frame + 1 entries
    # while the runner pulls num_frame results (SCSam3VideoInferenceNewMem.py:226-252)
    import SCSam3VideoInferenceNewMem as V
    start, num_frame, total = 30, 21, 300
    state = {"num_frames": total, "previous_stages_out": [None] * total}
    state["previous_stages_out"][start] = "_THIS_FRAME_HAS_OUTPUTS_"
    order, end = V.SCSam3VideoInferenceNewMem._get_processing_order(None, state, start, num_frame, False)
    assert list(order) == list(range(start, start + num_frame + 1)) and len(order) == num_frame + 1
    assert end == start + num_frame
    assert os.path.isfile(os.path.join(PKG, "SCSam3VideoInferenceNewMem.py"))
