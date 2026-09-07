"""T2a-j: the REAL _prepare_memory_conditioned_features_multiple on CPU (bare instance).

Golden values are SPEC.md section 1 (N=10, t=5, lockstep), captured from the unmodified
code with the same harness; T2f-j add the SPEC_P4 mode goldens, the closure cones per
mode and the Jacobi properties of mode E.  Container only (torch + sam3).
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sam3")

from conftest import PKG  # noqa: E402
from harness import (GOLDEN_HW, SMALL_HW, bare_tracker, mem_out, output_dicts_lockstep, run,  # noqa: E402
                     call, lockstep, same_inputs)

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


# ------------------------------------------------------------- P4: T2f-T2j
MODE_GATHER = (("A", "lower_t", None), ("D", "lower_tm1", None), ("B", "both_tm1", None),
               ("C", "mixed", None), ("E1", "both_tm1", 1), ("E2", "all_t", 2))
TWO_SIDED = ("both_tm1", "mixed", "all_t")
# neighbour part of mem_src (src = 1000*view+frame): SPEC_P4 section 4 goldens
GOLD = {(1, "A", 5, 0): [], (1, "A", 5, 1): [5], (1, "A", 5, 5): [4005], (1, "A", 5, 9): [8005],
        (1, "D", 5, 0): [], (1, "D", 5, 1): [4], (1, "D", 5, 5): [4004], (1, "D", 5, 9): [8004],
        (1, "B", 5, 0): [1004], (1, "B", 5, 1): [4, 2004], (1, "B", 5, 5): [4004, 6004], (1, "B", 5, 9): [8004],
        (1, "C", 5, 0): [1004], (1, "C", 5, 1): [5, 2004], (1, "C", 5, 5): [4005, 6004], (1, "C", 5, 9): [8005],
        (1, "E2", 5, 0): [1005], (1, "E2", 5, 1): [5, 2005], (1, "E2", 5, 5): [4005, 6005], (1, "E2", 5, 9): [8005],
        (1, "B", 31, 1): [30, 2030], (1, "C", 31, 5): [4031, 6030], (1, "E2", 31, 5): [4031, 6031],
        (2, "A", 5, 5): [3005, 4005], (2, "B", 5, 0): [1004, 2004], (2, "B", 5, 1): [4, 2004, 3004],
        (2, "B", 5, 5): [3004, 4004, 6004, 7004], (2, "B", 5, 9): [7004, 8004],
        (2, "C", 5, 5): [3005, 4005, 6004, 7004], (2, "E2", 5, 5): [3005, 4005, 6005, 7005]}
ROWS_GOLD = {(2, "A", 5, 5): [1, 0], (2, "B", 5, 0): [0, 1], (2, "B", 5, 1): [0, 0, 1],
             (2, "B", 5, 5): [1, 0, 0, 1], (2, "B", 5, 9): [1, 0]}


def expected_neighbours(mode, v, t, start, W, holders):
    """SPEC_P4 section 0 per entry, as (src, tpos rows); a None entry is skipped by the
    token loop, so it does not appear."""
    cells = []
    for s in list(range(-W, 0)) + (list(range(1, W + 1)) if mode in TWO_SIDED else []):
        m = v + s
        if not 0 <= m < N:
            continue
        f = t if (mode in ("lower_t", "all_t") or (mode == "mixed" and s < 0)) else t - 1
        if f == t and m not in holders:
            continue
        cells.append((s, 1000 * m + f))
    return [x for _, x in cells], [abs(s) - 1 for s, _ in cells]


# ----------------------------------------------------------------------- T2f
def test_real_method_mode_goldens(cpu_tensors):
    seen = {}
    for W in (1, 2):
        for letter, mode, xp in MODE_GATHER:
            tr = bare_tracker(xw=W, xh=True, xm=letter[0])
            assert (tr.cross_view_window, tr.cross_view_hygiene, tr.cross_view_mode) == (W, True, letter[0])
            for t, start in ((5, 0), (31, 30)):
                for v in (0, 1, 5, 9):
                    all_hold = mode == "all_t"
                    ods = output_dicts_lockstep(N, v, t, start, SMALL_HW, all_hold_t=all_hold)
                    r = run(tr, ods, v, t, SMALL_HW, start + 22, xview_pass=xp)
                    own = [1000 * v + f for f in range(start, t)]         # cond, then frames < t
                    assert r["mem_src"][:len(own)] == own, (W, letter, t, v)
                    src, rows = r["mem_src"][len(own):], r["tpos_rows"][len(own):]
                    holders = set(range(N)) if all_hold else set(range(v))
                    assert (src, rows) == expected_neighbours(mode, v, t, start, W, holders), (W, letter, t, v)
                    seen[(W, letter, t, v)] = (src, rows)
                    if t == 5:                                             # pointers unchanged in every mode
                        assert r["n_ptr_tokens"] == 16
                        assert r["ptr_src"] == [1000 * v, 1000 * v + 4, 1000 * v + 3, 1000 * v + 2]
                    else:
                        assert r["n_ptr_tokens"] == 4 and r["ptr_src"] == [1000 * v + 30]
    for k, want in GOLD.items():
        assert seen[k][0] == want, k
    for k, want in ROWS_GOLD.items():
        assert seen[k][1] == want, k
    for k in [k for k in seen if k[1] == "E1"]:                            # E pass 1 is B
        assert seen[k] == seen[(k[0], "B", k[2], k[3])], k


# ----------------------------------------------------------------------- T2h
def test_mode_A_identical_to_none(cpu_tensors):
    for W in (0, 1, 4, 6):
        a, b = bare_tracker(xw=W, xh=True, xm=None), bare_tracker(xw=W, xh=True, xm="A")
        assert (a.cross_view_mode, b.cross_view_mode) == ("A", "A")
        for v in range(N):
            for t, start in ((5, 0), (1, 0), (31, 30)):
                ods = output_dicts_lockstep(N, v, t, start, SMALL_HW)
                ka = call(a, ods, v, t, SMALL_HW, start + 22)
                for kb in (call(b, ods, v, t, SMALL_HW, start + 22),
                           call(a, ods, v, t, SMALL_HW, start + 22, xview_pass=1),
                           call(b, ods, v, t, SMALL_HW, start + 22, xview_pass=None)):
                    assert torch.equal(ka["prompt"], kb["prompt"]) and torch.equal(ka["prompt_pos"], kb["prompt_pos"])
                    assert ka["num_obj_ptr_tokens"] == kb["num_obj_ptr_tokens"]
        with pytest.raises(ValueError):                                    # pass 2 is E only
            call(b, output_dicts_lockstep(N, 5, T5, 0, SMALL_HW, all_hold_t=True), 5, T5, SMALL_HW, xview_pass=2)
    # the legacy (4, False) bare tracker still reproduces T2a
    tr = bare_tracker(xw=None, xh=None)
    assert (tr.cross_view_window, tr.cross_view_hygiene, tr.cross_view_mode) == (4, False, "A")
    for v in range(N):
        r = run(tr, output_dicts_lockstep(N, v, T5), v, T5)
        osrc, orows = own_tokens(v)
        nsrc, nrows = neighbour_tokens(v, 4)
        assert r["mem_src"] == osrc + nsrc and r["tpos_rows"] == orows + nrows
        first_ptr = 9000 if v <= 3 else 1000 * v
        assert r["ptr_src"] == [first_ptr, 1000 * v + 4, 1000 * v + 3, 1000 * v + 2]
    for xm in "BCDE":                                                      # letters need hygiene
        with pytest.raises(ValueError):
            bare_tracker(xw=4, xh=False, xm=xm)


# ----------------------------------------------------------------------- T2g
_FULL = {}


def cone_mismatches(xm, W, T, K, two_pass=False, NN=12, scored_max=2):
    """(v, t) of the scored views whose encoder inputs differ between an NN-session and a
    K-session lockstep drive."""
    tr = bare_tracker(xw=W, xh=True, xm=xm)
    key = (xm, W, T, two_pass, NN)
    if key not in _FULL:
        _FULL[key] = lockstep(tr, NN, T, two_pass=two_pass)
    full, trunc = _FULL[key], lockstep(tr, K, T, two_pass=two_pass)
    return [(v, t) for v in range(scored_max + 1) for t in range(1, T + 1)
            if not same_inputs(full[(v, t)], trunc[(v, t)])]


def test_closure_cone_per_mode(cpu_tensors):
    """SPEC_P4 section 3 lemma with K = max(scored)+1 = 3, T = 4 tracked frames, W = 1:
    A/D exact at K; B/C first exact at K + W*T = 7; E first exact at K + 2*W*T = 11."""
    assert cone_mismatches("A", 1, 4, 3) == []
    assert cone_mismatches("D", 1, 4, 3) == []
    for xm in ("B", "C"):
        assert cone_mismatches(xm, 1, 4, 6) != [], xm
        assert cone_mismatches(xm, 1, 4, 7) == [], xm
    assert cone_mismatches("E", 1, 4, 10, two_pass=True) != []
    assert cone_mismatches("E", 1, 4, 11, two_pass=True) == []
    assert cone_mismatches("B", 2, 3, 8) != []                              # K + W*T = 3 + 2*3
    assert cone_mismatches("B", 2, 3, 9) == []


# ----------------------------------------------------------------------- T2i
def test_two_pass_jacobi_order_free(cpu_tensors):
    tr = bare_tracker(xw=1, xh=True, xm="E")
    fwd = lockstep(tr, 8, 4, two_pass=True)
    rev = lockstep(tr, 8, 4, two_pass=True, pass2_reverse=True)
    assert fwd.keys() == rev.keys() and all(same_inputs(fwd[k], rev[k]) for k in fwd)
    gs = lockstep(tr, 8, 4, two_pass=True, gauss_seidel=True)
    gs_rev = lockstep(tr, 8, 4, two_pass=True, gauss_seidel=True, pass2_reverse=True)
    assert any(not same_inputs(fwd[k], gs[k]) for k in fwd)               # the test is sensitive
    assert any(not same_inputs(fwd[k], gs_rev[k]) for k in fwd)
    assert any(not same_inputs(gs[k], gs_rev[k]) for k in fwd)
    # the first frame's pass-2 inputs are order-free even under Gauss-Seidel for view 0
    # in forward order (it reads pass-1 t of view 1, not yet committed): sanity of the drive
    assert same_inputs(fwd[(0, 1)], gs[(0, 1)])


# ----------------------------------------------------------------------- T2j
def test_pass2_never_reads_own_pass1(cpu_tensors):
    for W in (1, 2):
        tr = bare_tracker(xw=W, xh=True, xm="E")
        for v in (0, 3, 5, 9):
            ods = output_dicts_lockstep(N, v, T5, 0, SMALL_HW, all_hold_t=True)
            r = run(tr, ods, v, T5, SMALL_HW, xview_pass=2)
            assert r["mem_src"][:5] == [1000 * v + f for f in range(0, 5)]     # cond + frames < t
            nb = [1000 * (v + s) + 5 for s in list(range(-W, 0)) + list(range(1, W + 1)) if 0 <= v + s < N]
            assert r["mem_src"][5:] == nb                                       # exactly v+-k @ t
            assert 1000 * v + 5 not in r["mem_src"]
            assert r["ptr_src"] == [1000 * v, 1000 * v + 4, 1000 * v + 3, 1000 * v + 2]
            assert r["tpos_rows"][5:] == [abs(s) - 1 for s in list(range(-W, 0)) + list(range(1, W + 1)) if 0 <= v + s < N]
        # pass 1 of E (= B) with every session holding t: t-1 on both sides, never t
        r = run(tr, output_dicts_lockstep(N, 5, T5, 0, SMALL_HW, all_hold_t=True), 5, T5, SMALL_HW, xview_pass=1)
        assert r["mem_src"][5:] == [1000 * (5 + s) + 4 for s in list(range(-W, 0)) + list(range(1, W + 1))]


# ----------------------------------------------------------------------- T2k
def test_real_method_mode_B_reverse_lag(cpu_tensors):
    """Reverse tracking flips the neighbour lag of the t-1 modes to t+1 (xview_gather
    `lag`), which only happens because the tracker hands its own `track_in_reverse` to
    the gather.  Every session holds cond[0], frames 1..4 and frame 6; nobody holds 5."""
    v, t = 5, T5

    def ods():
        return [{"cond_frame_outputs": {0: mem_out(0, m, SMALL_HW)},
                 "non_cond_frame_outputs": {f: mem_out(f, m, SMALL_HW) for f in (1, 2, 3, 4, 6)}}
                for m in range(N)]

    tr = bare_tracker(xw=1, xh=True, xm="B")
    r = run(tr, ods(), v, t, SMALL_HW, rev=True)
    assert r["mem_src"] == [5000, 5006, 4006, 6006]      # own: cond + the frame after t; nb: v+-1 @ t+1
    assert r["tpos_rows"] == [6, 0, 0, 0]
    f = run(tr, ods(), v, t, SMALL_HW, rev=False)         # no reverse lag: v+-1 @ t-1
    assert f["mem_src"] == [5000, 5001, 5002, 5003, 5004, 4004, 6004]
    assert f["tpos_rows"] == [6, 3, 2, 1, 0, 0, 0]
    # W=2: mirror order (-2, -1, +1, +2), all at t+1
    r2 = run(bare_tracker(xw=2, xh=True, xm="B"), ods(), v, t, SMALL_HW, rev=True)
    assert r2["mem_src"] == [5000, 5006, 3006, 4006, 6006, 7006] and r2["tpos_rows"][2:] == [1, 0, 0, 1]
    # the lag is per mode: E pass 1 is B; D reads the lower side at t+1; C keeps the lower
    # side at t (nobody holds 5 -> skipped) and lags the upper side to t+1
    assert run(bare_tracker(xw=1, xh=True, xm="E"), ods(), v, t, SMALL_HW, rev=True, xview_pass=1)["mem_src"] == r["mem_src"]
    assert run(bare_tracker(xw=1, xh=True, xm="D"), ods(), v, t, SMALL_HW, rev=True)["mem_src"] == [5000, 5006, 4006]
    assert run(bare_tracker(xw=1, xh=True, xm="C"), ods(), v, t, SMALL_HW, rev=True)["mem_src"] == [5000, 5006, 6006]
