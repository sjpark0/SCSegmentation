"""T8: the Phase 3 / P4 gather variants of SCSam3/demoSCSam3MVOpt/xview_gather.py.

Torch-free, host.  Fake dicts as in T1: the seed of session m is "cond[m]", a tracked
frame f is "mem[m]@f" (SPEC_P4 section 4 writes the same goldens as 1000*view+frame).
`select_fn` raises: no mode, and hygiene in general, may reach the legacy fallback.
"""
import random

import pytest

from xview_gather import (GATHER_MODES, MODES, UNCHANGED, gather_cross_view_memories,
                          gather_mode_for, resolve_cross_view_mode)

MAXC = 4
LOWER_MODES = ("lower_t", "lower_tm1")
TWO_SIDED = ("both_tm1", "mixed", "all_t")


def no_fallback(*a, **k):
    raise AssertionError("the legacy cond-frame fallback ran")


def gather(ods, v, t, W, mode, hygiene=True, rev=False):
    return gather_cross_view_memories(ods, v, t, W, hygiene, MAXC, no_fallback, mode=mode,
                                      track_in_reverse=rev)


def mk_frames(N, start, upto, hold_t_for):
    """Every session m: cond[start] and non_cond[start+1..upto]; the sessions in
    hold_t_for also hold non_cond[upto+1] (t, the frame being computed)."""
    t = upto + 1
    ods = []
    for m in range(N):
        nc = {f: f"mem[{m}]@{f}" for f in range(start + 1, upto + 1)}
        if m in hold_t_for:
            nc[t] = f"mem[{m}]@{t}"
        ods.append({"cond_frame_outputs": {start: f"cond[{m}]"}, "non_cond_frame_outputs": nc})
    return ods


def lockstep_frames(N, v, t, start=0, all_hold_t=False):
    """Runner lockstep: sessions < v already hold t (all_hold_t: pass-2 inputs of mode E)."""
    return mk_frames(N, start, t - 1, range(N) if all_hold_t else range(v))


def reference(mode, N, v, t, start, W, holders):
    """SPEC_P4 section 0 spelled out per entry: lower side v-W..v-1 (farthest first),
    then for two-sided modes v+1..v+W (nearest first); frame t for lower_t / all_t and
    for the lower side of mixed, t-1 otherwise; a t-1 == start read is the cond entry;
    a frame-t read of a session that has not computed t is None."""
    def stored(m, f):
        if f == start:
            return f"cond[{m}]"
        if f == t:
            return f"mem[{m}]@{t}" if m in holders else None
        return f"mem[{m}]@{f}"
    exp = []
    for s in range(-W, 0):
        if v + s >= 0:
            exp.append((s, stored(v + s, t if mode in ("lower_t", "mixed", "all_t") else t - 1)))
    if mode in TWO_SIDED:
        for s in range(1, W + 1):
            if v + s < N:
                exp.append((s, stored(v + s, t if mode == "all_t" else t - 1)))
    return exp


# SPEC_P4 section 4 goldens (N=10, lockstep holdings; all_t with every session holding t)
GOLD_W1_T5 = {
    "lower_t":   {0: [], 1: [(-1, "mem[0]@5")], 5: [(-1, "mem[4]@5")], 9: [(-1, "mem[8]@5")]},
    "lower_tm1": {0: [], 1: [(-1, "mem[0]@4")], 5: [(-1, "mem[4]@4")], 9: [(-1, "mem[8]@4")]},
    "both_tm1":  {0: [(1, "mem[1]@4")], 1: [(-1, "mem[0]@4"), (1, "mem[2]@4")],
                  5: [(-1, "mem[4]@4"), (1, "mem[6]@4")], 9: [(-1, "mem[8]@4")]},
    "mixed":     {0: [(1, "mem[1]@4")], 1: [(-1, "mem[0]@5"), (1, "mem[2]@4")],
                  5: [(-1, "mem[4]@5"), (1, "mem[6]@4")], 9: [(-1, "mem[8]@5")]},
    "all_t":     {0: [(1, "mem[1]@5")], 1: [(-1, "mem[0]@5"), (1, "mem[2]@5")],
                  5: [(-1, "mem[4]@5"), (1, "mem[6]@5")], 9: [(-1, "mem[8]@5")]},
}
GOLD_W1_T31 = {("both_tm1", 1): [(-1, "cond[0]"), (1, "cond[2]")],
               ("mixed", 5): [(-1, "mem[4]@31"), (1, "cond[6]")],
               ("all_t", 5): [(-1, "mem[4]@31"), (1, "mem[6]@31")]}
GOLD_W2_T5 = {("lower_t", 5): [(-2, "mem[3]@5"), (-1, "mem[4]@5")],
              ("both_tm1", 0): [(1, "mem[1]@4"), (2, "mem[2]@4")],
              ("both_tm1", 1): [(-1, "mem[0]@4"), (1, "mem[2]@4"), (2, "mem[3]@4")],
              ("both_tm1", 5): [(-2, "mem[3]@4"), (-1, "mem[4]@4"), (1, "mem[6]@4"), (2, "mem[7]@4")],
              ("both_tm1", 9): [(-2, "mem[7]@4"), (-1, "mem[8]@4")],
              ("mixed", 5): [(-2, "mem[3]@5"), (-1, "mem[4]@5"), (1, "mem[6]@4"), (2, "mem[7]@4")],
              ("all_t", 5): [(-2, "mem[3]@5"), (-1, "mem[4]@5"), (1, "mem[6]@5"), (2, "mem[7]@5")]}
ROWS_W2_T5 = {("lower_t", 5): [1, 0], ("both_tm1", 0): [0, 1], ("both_tm1", 1): [0, 0, 1],
              ("both_tm1", 5): [1, 0, 0, 1], ("both_tm1", 9): [1, 0]}


# ------------------------------------------------------------------------ T8a
@pytest.mark.parametrize("W", (1, 2))
@pytest.mark.parametrize("t,start", ((5, 0), (31, 30)))
def test_tables(W, t, start):
    N = 10
    for mode in GATHER_MODES:
        all_hold = mode == "all_t"
        for v in (0, 1, 5, 9):
            ods = lockstep_frames(N, v, t, start, all_hold_t=all_hold)
            got, rebound = gather(ods, v, t, W, mode)
            assert rebound is UNCHANGED
            holders = set(range(N)) if all_hold else set(range(v))
            assert got == reference(mode, N, v, t, start, W, holders), (mode, v)
            if W == 1 and t == 5:
                assert got == GOLD_W1_T5[mode][v], (mode, v)
            if W == 1 and t == 31 and (mode, v) in GOLD_W1_T31:
                assert got == GOLD_W1_T31[(mode, v)], (mode, v)
            if W == 2 and t == 5 and (mode, v) in GOLD_W2_T5:
                assert got == GOLD_W2_T5[(mode, v)], (mode, v)
                if (mode, v) in ROWS_W2_T5:                 # tpos row = abs(s_pos) - 1
                    assert [abs(s) - 1 for s, _ in got] == ROWS_W2_T5[(mode, v)]
    # all_t under plain lockstep holdings: the upper side has not computed t yet -> None
    got, _ = gather(lockstep_frames(N, 5, t, start), 5, t, W, "all_t")
    assert got == [(s, f"mem[{5 + s}]@{t}") for s in range(-W, 0)] + [(s, None) for s in range(1, W + 1)]


# ------------------------------------------------------------------------ T8b
def test_first_tracked_frame_reads_seed():
    N = 10
    ods = mk_frames(N, 0, 0, range(1))                   # t = 1: only session 0 holds t
    assert gather(ods, 1, 1, 1, "lower_tm1")[0] == [(-1, "cond[0]")]
    assert gather(ods, 1, 1, 2, "lower_tm1")[0] == [(-1, "cond[0]")]
    assert gather(ods, 1, 1, 2, "both_tm1")[0] == [(-1, "cond[0]"), (1, "cond[2]"), (2, "cond[3]")]
    assert gather(ods, 1, 1, 2, "mixed")[0] == [(-1, "mem[0]@1"), (1, "cond[2]"), (2, "cond[3]")]
    assert gather(ods, 1, 1, 2, "lower_t")[0] == [(-1, "mem[0]@1")]
    assert gather(ods, 1, 1, 2, "all_t")[0] == [(-1, "mem[0]@1"), (1, None), (2, None)]
    nobody = mk_frames(N, 0, 0, ())
    assert gather(nobody, 1, 1, 2, "all_t")[0] == [(-1, None), (1, None), (2, None)]
    assert gather(nobody, 1, 1, 2, "lower_t")[0] == [(-1, None)]
    ods = mk_frames(N, 30, 30, range(5))                 # dataset start 30
    assert gather(ods, 5, 31, 1, "both_tm1")[0] == [(-1, "cond[4]"), (1, "cond[6]")]
    assert gather(ods, 5, 31, 1, "lower_tm1")[0] == [(-1, "cond[4]")]
    assert gather(ods, 5, 31, 1, "mixed")[0] == [(-1, "mem[4]@31"), (1, "cond[6]")]


# ------------------------------------------------------------------------ T8c
def test_boundaries():
    N, t = 10, 5
    for mode in GATHER_MODES:
        all_hold = mode == "all_t"
        for W in (1, 2, 4, 6):
            s, _ = gather(lockstep_frames(N, 0, t, all_hold_t=all_hold), 0, t, W, mode)
            assert all(sp > 0 for sp, _ in s)                                 # v = 0: no lower entry
            assert len(s) == (min(W, N - 1) if mode in TWO_SIDED else 0)
            s, _ = gather(lockstep_frames(N, N - 1, t, all_hold_t=all_hold), N - 1, t, W, mode)
            assert all(sp < 0 for sp, _ in s) and len(s) == min(W, N - 1)    # v = N-1: no upper entry
            for n in (1, 2, 3):                                               # N < W+1: no IndexError
                for v in range(n):
                    s, _ = gather(lockstep_frames(n, v, t, all_hold_t=all_hold), v, t, W, mode)
                    assert all(0 <= v + sp < n for sp, _ in s)
    ods = lockstep_frames(N, 5, t, all_hold_t=True)                          # None sessions
    ods[4] = None
    ods[6] = None
    for mode in TWO_SIDED:
        s, _ = gather(ods, 5, t, 2, mode)
        assert [sp for sp, _ in s] == [-2, -1, 1, 2]
        assert s[1] == (-1, None) and s[2] == (1, None)
        assert s[0][1] is not None and s[3][1] is not None
    for mode in LOWER_MODES:
        s, _ = gather(ods, 5, t, 2, mode)
        assert [sp for sp, _ in s] == [-2, -1] and s[1] == (-1, None) and s[0][1] is not None


# ------------------------------------------------------------------------ T8d
def test_order_lower_then_upper():
    N, t = 20, 5
    ods = lockstep_frames(N, 10, t, all_hold_t=True)
    for W in range(0, 7):
        for mode in TWO_SIDED:
            s, _ = gather(ods, 10, t, W, mode)
            assert [sp for sp, _ in s] == list(range(-W, 0)) + list(range(1, W + 1))
        for mode in LOWER_MODES:
            s, _ = gather(ods, 10, t, W, mode)
            assert [sp for sp, _ in s] == list(range(-W, 0))


# ------------------------------------------------------------------------ T8e
def test_mode_errors():
    ods = lockstep_frames(10, 5, 5)
    for mode in GATHER_MODES:
        if mode != "lower_t":
            with pytest.raises(ValueError):
                gather(ods, 5, 5, 1, mode, hygiene=False)
    for bad in ("upper_t", "A", "", None):
        with pytest.raises(ValueError):
            gather(ods, 5, 5, 1, bad)
    assert GATHER_MODES == ("lower_t", "lower_tm1", "both_tm1", "mixed", "all_t")


# ------------------------------------------------------------------------ T8f
def test_resolve_cross_view_mode():
    assert resolve_cross_view_mode(None, False) == "A"
    assert resolve_cross_view_mode(None, True) == "A"
    assert resolve_cross_view_mode("A", False) == "A"
    assert resolve_cross_view_mode("b", True) == "B"
    assert resolve_cross_view_mode("E", True) == "E"
    for bad in (("B", False), ("D", False), ("X", True), ("", True)):
        with pytest.raises(ValueError):
            resolve_cross_view_mode(*bad)
    assert MODES == ("A", "B", "C", "D", "E")


# ------------------------------------------------------------------------ T8g
def test_gather_mode_for():
    want = {"A": "lower_t", "B": "both_tm1", "C": "mixed", "D": "lower_tm1", "E": "both_tm1"}
    for m, g in want.items():
        assert gather_mode_for(m) == g and gather_mode_for(m, None) == g and gather_mode_for(m, 1) == g
    assert gather_mode_for("E", 2) == "all_t"
    for m in "ABCD":
        with pytest.raises(ValueError):
            gather_mode_for(m, 2)
    with pytest.raises(ValueError):
        gather_mode_for("E", 3)
    with pytest.raises(ValueError):
        gather_mode_for("A", 0)


# ------------------------------------------------------------------------ T8h
def random_frames(rng, N, v, t, start):
    ods = []
    for m in range(N):
        if m != v and rng.random() < 0.15:
            ods.append(None)
            continue
        nc = {}
        for f in range(start + 1, t):
            if rng.random() < 0.6:
                nc[f] = f"mem[{m}]@{f}"
        if rng.random() < 0.7:
            nc[t] = f"mem[{m}]@{t}"
        ods.append({"cond_frame_outputs": {start: f"cond[{m}]"}, "non_cond_frame_outputs": nc})
    return ods


def test_truncation_invariance_per_mode():
    """Closure lemma at gather level: lower modes read sessions < v, two-sided modes
    sessions v-W..v+W; truncating a two-sided call at v+1 removes exactly the upper side."""
    rng = random.Random(2)
    for trial in range(3000):
        N = rng.randint(1, 14)
        start = rng.choice((0, 30))
        t = start + rng.randint(1, 20)
        v = rng.randrange(N)
        W = rng.randint(0, 6)
        ods = random_frames(rng, N, v, t, start)
        for mode in GATHER_MODES:
            full = gather(ods, v, t, W, mode)
            assert full[1] is UNCHANGED
            cut = v + 1 if mode in LOWER_MODES else v + W + 1
            assert full == gather(ods[:cut], v, t, W, mode), (trial, mode)
            if mode in LOWER_MODES:
                blank = [od if m <= v else None for m, od in enumerate(ods)]
                assert full == gather(blank, v, t, W, mode), (trial, mode)
            else:
                lower_only = gather(ods[:v + 1], v, t, W, mode)
                assert lower_only[0] == [p for p in full[0] if p[0] < 0], (trial, mode)


# ------------------------------------------------------------------------ T8i
def test_reverse_reads_t_plus_1():
    N, t = 10, 5
    ods = mk_frames(N, 0, 5, range(N))                   # everyone holds 1..6
    for m in range(N):
        ods[m]["non_cond_frame_outputs"][6] = f"mem[{m}]@6"
    assert gather(ods, 5, t, 1, "lower_tm1", rev=True)[0] == [(-1, "mem[4]@6")]
    assert gather(ods, 5, t, 1, "both_tm1", rev=True)[0] == [(-1, "mem[4]@6"), (1, "mem[6]@6")]
    assert gather(ods, 5, t, 1, "mixed", rev=True)[0] == [(-1, "mem[4]@5"), (1, "mem[6]@6")]
    assert gather(ods, 5, t, 1, "lower_t", rev=True)[0] == [(-1, "mem[4]@5")]
    assert gather(ods, 5, t, 1, "all_t", rev=True)[0] == [(-1, "mem[4]@5"), (1, "mem[6]@5")]
    assert gather(ods, 5, t, 1, "both_tm1")[0] == [(-1, "mem[4]@4"), (1, "mem[6]@4")]   # forward
    # reverse at the last frame of a reverse walk falls through to the cond entry
    ods = mk_frames(N, 7, 6, ())                         # cond[7] only (no tracked frame)
    assert gather(ods, 5, 6, 1, "both_tm1", rev=True)[0] == [(-1, "cond[4]"), (1, "cond[6]")]
