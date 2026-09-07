"""T1a-g and T4: the torch-free gather (SCSam3/demoSCSam3MVOpt/xview_gather.py).

Runs on the host.  The oracle is tests/legacy_ref.py (verbatim pre-change loop); the
real sam3 select_closest_cond_frames is used when importable (container), else the
transliteration.
"""
import itertools
import random

import pytest

import legacy_ref
from xview_gather import (UNCHANGED, LEGACY_WINDOW, LEGACY_HYGIENE,
                          gather_cross_view_memories, resolve_cross_view)

try:
    from sam3.model.sam3_tracker_utils import select_closest_cond_frames as SELECT
    REAL_SELECT = True
except Exception:                       # host: no sam3 install
    SELECT = legacy_ref.select_closest_cond_frames
    REAL_SELECT = False

MAXC = 4                                 # max_cond_frames_in_attn of the built model


def gather(ods, v, t, window, hygiene):
    return gather_cross_view_memories(ods, v, t, window, hygiene, MAXC, SELECT)


def apply(ods, v, t, self_sel, window=LEGACY_WINDOW, hygiene=LEGACY_HYGIENE):
    """What the tracker does with the return value (SPEC 2.2 iii)."""
    s, rebound = gather(ods, v, t, window, hygiene)
    return s, (self_sel if rebound is UNCHANGED else rebound)


def mk(N, frame_idx, have_t, start=0):
    """N sessions; each has cond {start} and non_cond {frame_idx} iff have_t[m]."""
    return [{"cond_frame_outputs": {start: f"cond[{m}]"},
             "non_cond_frame_outputs": ({frame_idx: f"mem[{m}]@{frame_idx}"} if have_t[m] else {})}
            for m in range(N)]


def lockstep_have(N, v):
    return [m < v for m in range(N)]


def random_ods(rng, N, v, t):
    ods = []
    for m in range(N):
        if rng.random() < 0.15 and m != v:
            ods.append(None)
            continue
        conds = {c: object() for c in rng.sample(range(0, 31), rng.randint(1, 6))}
        nc = {}
        if rng.random() < 0.7:
            nc[t] = object()
        if rng.random() < 0.3:
            nc[t - 1] = object()
        ods.append({"cond_frame_outputs": conds, "non_cond_frame_outputs": nc})
    if ods[v] is None:
        ods[v] = {"cond_frame_outputs": {0: object()}, "non_cond_frame_outputs": {}}
    return ods


# ----------------------------------------------------------------------- T1a
def test_legacy_equals_transliteration():
    rng = random.Random(0)
    n_ok = n_exc = 0
    for trial in range(20000):
        N = rng.randint(1, 9)
        t = rng.randint(1, 30)
        v = rng.randrange(N)
        ods = random_ods(rng, N, v, t)
        self_sel = ods[v]["cond_frame_outputs"]
        try:
            exp = legacy_ref.legacy_gather(ods, v, t, self_sel, MAXC, select_fn=SELECT)
        except IndexError:
            exp = IndexError
        try:
            got = apply(ods, v, t, self_sel)
        except IndexError:
            got = IndexError
        if exp is IndexError or got is IndexError:
            assert exp is got, (trial, exp, got)          # IndexError parity
            n_exc += 1
            continue
        assert exp[0] == got[0], (trial, exp[0], got[0])  # same tuples, same order, same objects
        # the only consumer (:1348) iterates .items(): same keys and the same value objects
        assert exp[1].keys() == got[1].keys() and all(exp[1][k] is got[1][k] for k in exp[1]), trial
        assert (exp[1] is self_sel) == (got[1] is self_sel), trial   # rebound iff legacy rebound
        owners = [od["cond_frame_outputs"] for od in ods if od is not None]
        if any(exp[1] is o for o in owners):     # legacy handed back a session's own cond dict
            assert exp[1] is got[1], trial       # -> the refactor hands back the very same object
        n_ok += 1
    assert n_ok + n_exc == 20000 and n_ok > 0 and n_exc > 0


# ----------------------------------------------------------------------- T1b
def test_legacy_lockstep_n10():
    N, t = 10, 5
    for v in range(N):
        ods = mk(N, t, lockstep_have(N, v))
        s, rebound = gather(ods, v, t, 4, False)
        fed = [p for p in s if p[1] is not None]
        assert len(fed) == min(v, 4)
        if v <= 3:
            assert rebound is ods[9]["cond_frame_outputs"]     # C2: view N-1's cond pointer
        else:
            assert rebound is UNCHANGED


# ----------------------------------------------------------------------- T1c
def test_legacy_indexerror_small_n():
    t = 5
    for N in (1, 2, 3):
        ods = mk(N, t, lockstep_have(N, 0))
        with pytest.raises(IndexError):
            gather(ods, 0, t, 4, False)
    ods = mk(4, t, lockstep_have(4, 3))
    s, rebound = gather(ods, 3, t, 4, False)
    assert rebound is ods[3]["cond_frame_outputs"]             # wraps to itself: self-alias
    ods = mk(4, t, lockstep_have(4, 0))
    s, rebound = gather(ods, 0, t, 4, False)
    assert rebound is ods[3]["cond_frame_outputs"]


# ----------------------------------------------------------------------- T1d
def test_hygiene_no_wrap_no_rebind():
    t = 5
    for N in (1, 3, 10):
        for v in range(N):
            ods = mk(N, t, lockstep_have(N, v))
            for W in (0, 1, 2, 4, 6):
                s, rebound = gather(ods, v, t, W, True)
                assert rebound is UNCHANGED
                assert all(v + sp >= 0 for sp, _ in s)
                assert len([p for p in s if p[1] is not None]) == min(v, W)
                if W == 0:
                    assert s == []


# ----------------------------------------------------------------------- T1e
def test_hygiene_truncation_invariance():
    """Closure lemma at gather level: under hygiene view v reads only sessions < v."""
    rng = random.Random(1)
    for trial in range(3000):
        N = rng.randint(1, 12)
        t = rng.randint(1, 20)
        v = rng.randrange(N)
        ods = random_ods(rng, N, v, t)
        for W in (0, 1, 2, 4, 6):
            full = gather(ods, v, t, W, True)
            trunc = gather(ods[:v + 1], v, t, W, True)
            blank = gather([od if m <= v else None for m, od in enumerate(ods)], v, t, W, True)
            assert full == trunc == blank, (trial, W)
            assert full[1] is UNCHANGED
    # legacy is NOT invariant for v < 4 (why closure needs hygiene), invariant for v >= 4
    N, t = 10, 5
    for v in range(N):
        ods = mk(N, t, lockstep_have(N, v))
        full = gather(ods, v, t, 4, False)
        try:
            trunc = gather(ods[:v + 1], v, t, 4, False)
        except IndexError:
            assert v < 3
            continue
        if v < 4:
            assert trunc[1] is not full[1] or trunc[0] != full[0]
        else:
            assert trunc == full and trunc[1] is full[1]


# ----------------------------------------------------------------------- T1f
def test_dead_fallback_bruteforce():
    """`unselected.get(frame_idx)` is None for every configuration: the fallback at
    :1271-1273 never fed a memory, its only effect was the rebinding."""
    hits = trials = 0
    for ncond in range(1, 9):
        for frame_idx in range(0, 12):
            for conds in itertools.combinations(range(12), ncond):
                cond = {c: f"c{c}" for c in conds}
                for maxc in (4, -1, 2, 3):
                    for kf in (False, True):
                        sel, unsel = SELECT(frame_idx, cond, maxc, kf)
                        trials += 1
                        if unsel.get(frame_idx) is not None:
                            hits += 1
    assert trials == 364416 and hits == 0


# ----------------------------------------------------------------------- T1g
def test_resolve_cross_view():
    assert resolve_cross_view(None, None, 7) == (4, False)
    assert resolve_cross_view(0, True, 7) == (0, True)
    assert resolve_cross_view(6, True, 7) == (6, True)
    assert resolve_cross_view(4, False, 7) == (4, False)
    assert resolve_cross_view(4, True, 7) == (4, True)
    assert resolve_cross_view(4, None, 7) == (4, False)
    assert resolve_cross_view(None, True, 7) == (4, True)
    for bad in ((7, True), (-1, True), (2, False), (0, False), (6, None)):
        with pytest.raises(ValueError):
            resolve_cross_view(bad[0], bad[1], 7)


# ------------------------------------------------------------------------ T4
def test_tpos_rows():
    """Neighbour s_pos=-k adds maskmem_tpos_enc[k-1] (:1319); rows 0..5 are the temporal
    rows, row 6 the cond row.  W <= 6 never touches row 6; W = 7 would."""
    num_maskmem = 7
    for W in range(0, 7):
        rows = [abs(sp) - 1 for sp in range(-W, 0)]
        assert all(0 <= r <= num_maskmem - 2 for r in rows)
    assert (num_maskmem - 1) in [abs(sp) - 1 for sp in range(-7, 0)]
    with pytest.raises(ValueError):
        resolve_cross_view(7, True, num_maskmem)
