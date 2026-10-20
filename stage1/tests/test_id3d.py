"""S1-E15 (docs/stage1-E15-prereg.md §3): Fix A math and the relay geometry of geomem/id3d.py (host, numpy)."""
import json
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e15_synth as SY  # noqa: E402
from geomem import id3d  # noqa: E402
from geomem.id3d import ABSTAIN, EMPTY, FLAG, PASS  # noqa: E402


@pytest.fixture(scope="module")
def ring(tmp_path_factory):
    d = SY.write_scene(str(tmp_path_factory.mktemp("e15ring")))
    lab = d["labels"][d["ref"]]
    d["masks"] = np.stack([lab == 1, lab == 2])
    return d


def _judge(ring, family, **kw):
    return id3d.Judge(ring["cache"], ring["ref"], ring["masks"], [1, 2], family, **kw)


def _cstar():
    return dict(SY.P1_CSTAR)                    # pinned (not id3d.CSTAR): unchanged by the Step-1 commit


# ------------------------------------------------------------------ lifting
def test_mask_to_work_is_area_coverage():
    m = np.zeros((8, 8), bool)
    m[0:2, 0:2] = True                  # one full work cell
    m[2, 2] = m[2, 3] = True            # half a cell: coverage 0.5 counts
    m[4, 4] = True                      # a quarter: does not
    assert np.array_equal(np.flatnonzero(id3d.mask_to_work(m, (4, 4))), [0, 5])


def test_lift_erodes_then_falls_back_to_the_unEroded_mask():
    P = np.arange(10 * 10 * 3, dtype=np.float64).reshape(10, 10, 3)
    valid = np.ones((10, 10), bool)
    big = np.zeros((10, 10), bool)
    big[1:9, 1:9] = True                                  # erosion leaves 6x6 = 36 >= 20
    assert len(id3d.lift(big, P, valid)) == 36
    mid = np.zeros((10, 10), bool)
    mid[2:7, 2:7] = True                                  # erosion leaves 3x3 = 9 < 20 -> the 25 un-eroded points
    assert len(id3d.lift(mid, P, valid)) == 25
    small = np.zeros((10, 10), bool)
    small[0:3, 0:3] = True                                # still < N_MIN after the fall-back: the caller ABSTAINs
    assert len(id3d.lift(small, P, valid)) == 9 < id3d.N_MIN
    part = valid.copy()
    part[:, :5] = False                                   # invalid Pi3 points never lift: eroded 6 x 3 = 18 < 20 ->
    assert len(id3d.lift(big, P, part)) == 8 * 4          # the un-eroded mask's valid 8 x 4


def test_border_fraction():
    m = np.zeros((100, 200), bool)
    assert id3d.border_fraction(m) == 0.0
    m[0:10, 50:60] = True                                 # band = max(2, int(0.5 % of 100)) = 2 rows: 20 of 100 px
    assert id3d.border_fraction(m) == pytest.approx(0.2)


# ------------------------------------------------------------------ the families on the synthetic ring
@pytest.mark.parametrize("family", ["c0", "cstar", "vote", "vote_inf"])
def test_correct_masks_pass_and_swaps_flag(ring, family, monkeypatch):
    monkeypatch.setattr(id3d, "CSTAR_SELECTED", True)
    J = _judge(ring, family, consts=_cstar() if family == "cstar" else None)
    n_pass = n_flag = 0
    for c in ring["cams"][1:]:
        L = ring["labels"][c]
        own, swap = np.stack([L == 1, L == 2]), np.stack([L == 2, L == 1])
        s_own, _ = J.statuses(c, own, [3.0, 3.0])
        s_swap, _ = J.statuses(c, swap, [3.0, 3.0])
        for i in range(2):
            if not own[i].any():
                assert s_own[i] == EMPTY
                continue
            assert s_own[i] == PASS, (family, c, i)
            n_pass += 1
            if swap[i].any():
                assert s_swap[i] == FLAG, (family, c, i)          # another object's mask lifted -> the 3D swap is caught
                n_flag += 1
    assert n_pass >= 18 and n_flag >= 16


def test_spill_is_not_a_vote_pass(ring):
    """A-v: a mask that covers its object plus a large background area does not PASS (ov3d < 0.5)."""
    J = _judge(ring, "vote")
    c = "cam02"
    L = ring["labels"][c]
    spill = (L == 1) | (np.arange(L.shape[1])[None, :] < L.shape[1] // 2) & (L == 0)
    st, info = J.statuses(c, np.stack([spill, L == 2]), [3.0, 3.0])
    assert st[0] != PASS and info[0]["ov3d"] < id3d.THETA_V
    assert st[1] == PASS


def test_empty_and_presence(ring):
    J = _judge(ring, "vote")
    L = ring["labels"]["cam01"]
    st, _ = J.statuses("cam01", np.stack([L == 1, np.zeros_like(L, bool)]), [-0.1, 4.0])
    assert st == [EMPTY, EMPTY]                           # presence <= 0 / no pixel


def test_cstar_abstain_rules(ring):
    J = _judge(ring, "c0")                                # only the reference models are used below
    M = J.models
    L = ring["labels"]["cam01"]
    from geomem.pointmap import ref_frame_points
    P, valid, _ = ref_frame_points(ring["cache"], "cam01", ring["ref"])
    Q = id3d.lift(id3d.mask_to_work(L == 1, P.shape[:2]), P, valid)
    k = _cstar()
    assert id3d.status_cstar(Q, 1, M, **k)[0] == PASS
    assert id3d.status_cstar(Q[:5], 1, M, **k)[1]["why"] == "small"
    assert id3d.status_cstar(Q, 3, M, **k)[1]["why"] == "model"                       # no reference model for object 3
    assert id3d.status_cstar(Q, 1, M, **dict(k, rho_env=0.001))[1]["why"] == "wide"
    cut = {o: dict(m, border=0.5) if o == 1 else m for o, m in M.items()}
    assert id3d.status_cstar(Q, 1, cut, **k)[1]["why"] == "cut"
    assert id3d.status_cstar(Q, 1, cut, **dict(k, b_max=math.inf))[0] == PASS
    assert id3d.status_cstar(Q, 1, M, **dict(k, gamma=0.0, sigma=0.0))[1]["why"] == "spread"
    assert id3d.status_cstar(Q, 1, M, rig_ok=False, **k)[1]["why"] == "rig"
    # a wide competitor is not a competitor: object 2's swap of object 1 is no longer FLAGged, only ABSTAINed
    Q2 = id3d.lift(id3d.mask_to_work(L == 2, P.shape[:2]), P, valid)
    assert id3d.status_cstar(Q2, 1, M, **k)[0] == FLAG
    wide2 = {o: dict(m, rho=0.9) if o == 2 else m for o, m in M.items()}
    assert id3d.status_cstar(Q2, 1, wide2, **k)[0] == ABSTAIN


def test_c0_abstains_without_two_models(ring):
    J = _judge(ring, "c0")
    one = {1: J.models[1], 2: dict(J.models[2], has=False)}
    Q = np.zeros((30, 3))
    assert id3d.status_c0(Q, 1, one)[0] == ABSTAIN
    assert id3d.status_c0(Q, 2, J.models)[0] in (PASS, FLAG)


def test_vote_plurality_ties_and_unassigned():
    R1 = np.array([[0.0, 0, 0]] * 30)
    R2 = np.array([[1.0, 0, 0]] * 30)
    models = {1: dict(R=R1, has=True), 2: dict(R=R2, has=True)}
    mid = np.array([[0.5, 0, 0]] * 30)                    # equidistant: the nearest label is the LOWER id
    st, info = id3d.status_vote(mid, 2, models, 0.6)
    assert info["plur"] == 1 and info["ov3d"] == 0.0 and st == FLAG
    far = np.array([[5.0, 0, 0]] * 30)                    # beyond r_max: unassigned -> plur 0 -> ABSTAIN
    st, info = id3d.status_vote(far, 1, models, 0.2)
    assert (st, info["plur"], info["ov3d"]) == (ABSTAIN, 0, 0.0)
    st, info = id3d.status_vote(far, 1, models, math.inf)  # r_max inf: everything is labelled
    assert st == FLAG and info["plur"] == 2


@pytest.mark.parametrize("r_max", [0.2, math.inf])
def test_vote_labels_equal_brute_force(r_max):
    rng = np.random.default_rng(0)
    models = {j: dict(R=rng.normal(0.3 * j, 0.1, (1500, 3)), has=True) for j in (1, 2, 5)}
    ref = id3d.vote_reference(models)
    Q = rng.normal(0.6, 0.3, (5000, 3))                      # > N_Q: subsampled; > _CHUNK: several blocks
    Qs = id3d.subsample(Q, id3d.N_Q)
    P, lab = ref
    assert len(P) == 3 * id3d.N_REF and list(np.unique(lab)) == [1, 2, 5]
    d = np.sqrt(((Qs[:, None, :] - P[None]) ** 2).sum(-1))
    j = d.argmin(1)
    expect = np.where(d[np.arange(len(Qs)), j] <= r_max, lab[j], 0)
    assert np.array_equal(id3d.vote_labels(Q, ref, r_max), expect)


def test_vote_subsample_is_deterministic_stride():
    X = np.arange(5000 * 3, dtype=np.float64).reshape(5000, 3)
    s = id3d.subsample(X, 1024)
    assert len(s) == 1024 and s[0, 0] == 0 and s[-1, 0] == X[-1, 0] and np.array_equal(s, id3d.subsample(X, 1024))
    assert id3d.subsample(X[:10], 1024) is not None and len(id3d.subsample(X[:10], 1024)) == 10


def test_hybrid_rules():
    R1 = np.array([[0.0, 0, 0]] * 30)
    R2 = np.array([[1.0, 0, 0]] * 30)
    models = {1: dict(R=R1, has=True), 2: dict(R=R2, has=True)}
    own, other, nowhere = (np.array([[x, 0, 0]] * 30) for x in (0.0, 1.0, 5.0))
    assert id3d.status_hybrid(other, 1, models, 100, 90, 0.9)[0] == FLAG          # A-v FLAG wins over a good overlap
    assert id3d.status_hybrid(nowhere, 1, models, 100, 90, 0.9)[0] == PASS        # visible: the reprojection decides
    assert id3d.status_hybrid(nowhere, 1, models, 100, 90, 0.2)[0] == ABSTAIN
    assert id3d.status_hybrid(nowhere, 1, models, 100, 90, float("nan"))[0] == ABSTAIN
    assert id3d.status_hybrid(own, 1, models, 100, 10, 0.0)[0] == PASS            # hidden from c_ini: A-v's state
    assert id3d.status_hybrid(nowhere, 1, models, 100, 10, 0.9)[0] == ABSTAIN


# ------------------------------------------------------------------ trust and the §3.4 choice
def test_trust_rule_and_reproj_equivalence():
    from methods.geomem.run import object_conf_mode
    states = [PASS, PASS, ABSTAIN, ABSTAIN, ABSTAIN, FLAG, EMPTY, PASS]
    presence = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    ious = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9]
    overlap = [0.0, 0.9, 0.9, float("nan"), float("nan"), 0.9, 0.9, 0.9]
    t = id3d.trust(states, presence, ious, overlap)
    assert t.tolist() == [True, False, True, True, False, False, False, False]
    rep = object_conf_mode("reproj", presence, ious, overlap)
    assert np.array_equal(id3d.reproj_rule(presence, ious, overlap), rep)          # ABSTAIN = the adopted reproj rule


def test_select_final_table():
    for x in (PASS, ABSTAIN):
        for s1 in (PASS, FLAG, ABSTAIN, EMPTY):
            assert id3d.select_final(x, s1, "chain") == "X"
    for x in (FLAG, EMPTY):
        assert id3d.select_final(x, PASS, "empty") == "S1"
        for s1 in (FLAG, ABSTAIN, EMPTY):
            for fb in ("keep", "chain", "empty"):
                assert id3d.select_final(x, s1, fb) == fb


def test_final_trust_never_trusts_a_fallback():
    from methods.geomem.run import e15_final_trust
    choices = ["X", "S1", "keep", "chain", "empty", "X"]
    tx = [True, False, True, True, True, False]
    ts1 = [False, True, False, True, True, True]
    assert e15_final_trust(choices, tx, ts1).tolist() == [True, True, True, False, False, False]


# ------------------------------------------------------------------ anchor, angles, relay order
def test_object_anchor_and_scene_fallback(ring):
    z = np.load(os.path.join(ring["cache"], f"{ring['ref']}.npz"))
    union = id3d.mask_to_work(ring["masks"].any(0), z["depth"].shape)
    X, kind = id3d.object_anchor(z["depth"], z["conf"], z["K"], z["c2w"], union)
    assert kind == "obj" and abs(X[0]) < 0.05 and -0.5 < X[2] < 0       # front of the two spheres, between them
    Xs, kind_s = id3d.object_anchor(z["depth"], z["conf"], z["K"], z["c2w"], np.zeros_like(union))
    assert kind_s == "scene" and Xs[2] > 1.0                               # the background behind the objects
    tiny = np.zeros_like(union)
    tiny[20, 28] = True                                                    # < N_MIN valid points in the union -> scene
    assert id3d.object_anchor(z["depth"], z["conf"], z["K"], z["c2w"], tiny)[1] == "scene"


def _circle(angles_deg, radius=3.0):
    return {f"c{i:02d}": np.array([radius * math.sin(math.radians(a)), 0.0, -radius * math.cos(math.radians(a))])
            for i, a in enumerate(angles_deg)}


def test_angles_and_split():
    centres = _circle([0, 30, 60, 90, 300, 330])
    X = np.zeros(3)
    a = id3d.view_angles(centres, "c00", X)
    assert {k: round(v, 6) for k, v in a.items()} == {"c01": 30, "c02": 60, "c03": 90, "c04": 60, "c05": 30}
    cams = list(centres)
    assert id3d.split_views(a, cams, 45.0) == (["c01", "c05"], ["c02", "c03", "c04"])
    assert id3d.split_views(a, cams, math.inf) == (["c01", "c02", "c03", "c04", "c05"], [])
    assert round(id3d.hop_angle(centres, X, "c02", "c04"), 6) == 120


def test_relay_order_on_a_ring_with_a_gap():
    # ring with a 60 deg gap between 150 and 210 and one camera off the ring (elevated, > theta from everything)
    angles_ring = [0, 30, 60, 90, 120, 150, 210, 240, 270, 300, 330]
    centres = _circle(angles_ring)
    centres["c11"] = np.array([0.0, -40.0, 0.5])
    X = np.zeros(3)
    cams = list(centres)
    a = id3d.view_angles(centres, "c00", X)

    def hop(m, v):
        return id3d.hop_angle(centres, X, m, v)
    S, F = id3d.split_views(a, cams, 45.0)
    assert S == ["c01", "c10"] and "c11" in F
    seq, unreachable = id3d.relay_order(S, F, a, hop, 45.0, cams)
    # both sides advance by a_v; ties (60 vs 60, ...) go to camera order; the 150-210 gap is never hopped
    assert seq == ["c02", "c09", "c03", "c08", "c04", "c07", "c05", "c06"]
    assert unreachable == ["c11"]
    assert id3d.reachable(S, F, hop, 45.0) == set(seq)
    seq_a, unr_a = id3d.relay_order(S, F, a, hop, 45.0, cams, order="angle")       # ABnh: pure a_v, no reachability
    assert unr_a == [] and seq_a == sorted(F, key=lambda c: (a[c], cams.index(c)))
    # a hop threshold that bridges the gap still relays nearest-angle first
    seq60, unr60 = id3d.relay_order(*id3d.split_views(a, cams, 65.0), a, hop, 65.0, cams)
    assert seq60[0] == "c03" and unr60 == ["c11"]


def test_relay_order_tie_goes_to_camera_order():
    a = {"b": 50.0, "a": 50.0}
    seq, _ = id3d.relay_order(["s"], ["b", "a"], dict(a, s=10.0), lambda m, v: 10.0, 45.0, ["s", "b", "a"])
    assert seq == ["b", "a"]


def test_relay_candidates_and_neighbours():
    centres = {c: np.array([float(i), 0.0, 0.0]) for i, c in enumerate("vabcdefgh")}
    cams = list(centres)
    pool = list("hgfedcba")                                       # pool order never matters: centre distance does
    assert id3d.relay_candidates("v", [], centres, cams) == []    # empty pool
    cands = id3d.relay_candidates("v", pool, centres, cams)
    assert cands == list("abcdef")                                # K_SEARCH = 6 nearest
    trusted = {m: m in "bdfg" for m in pool}
    assert id3d.relay_neighbours(cands, trusted) == list("bdf")   # first K_RELAY = 3 trusted among the six
    only7 = {m: m == "g" for m in pool}
    assert id3d.relay_neighbours(cands, only7) == []              # the 7th is never searched: orphan
    hop = {("a", "v"): 10.0, ("b", "v"): 80.0}.get
    cands_h = id3d.relay_candidates("v", ["a", "b"], centres, cams, hop=lambda m, v: hop((m, v)), theta=45.0)
    assert cands_h == ["a"]                                       # beyond theta: not a candidate
    centres_tie = {"v": np.zeros(3), "x": np.array([1.0, 0, 0]), "y": np.array([-1.0, 0, 0])}
    assert id3d.relay_candidates("v", ["y", "x"], centres_tie, ["v", "x", "y"]) == ["x", "y"]    # tie -> camera order


def test_orphan_objects():
    trust = {"a": np.array([True, False, False]), "b": np.array([False, False, True])}
    assert id3d.orphan_objects(["a", "b"], trust) == [1]
    assert id3d.orphan_objects([], trust) is None


# ------------------------------------------------------------------ rig free-space rate
def _plane(depth, c2w=np.eye(4)):
    K = np.array([[10.0, 0, 9.5], [0, 10.0, 9.5], [0, 0, 1]])
    return dict(depth=np.full((20, 20), depth, np.float32), conf=np.ones((20, 20), np.float32), K=K, c2w=c2w)


def test_free_space_rate():
    assert id3d.free_space(_plane(2.0), _plane(2.0)) == 0.0          # the same surface: no violation
    near, far = _plane(2.0), _plane(4.0)
    assert id3d.free_space(near, far) == 1.0                          # a's surface floats in b's free space
    assert id3d.free_space(far, near) == 0.0                          # behind b's surface: occluded, allowed
    side = np.eye(4)
    side[0, 3] = 100.0                                                # b far to the side: nothing projects inside
    assert id3d.free_space(near, _plane(4.0, side)) == 0.0


def test_rig_phi(ring):
    med, per = id3d.rig_phi(ring["cache"], ring["ref"], ring["cams"])
    assert set(per) == set(ring["cams"][1:]) and 0.0 <= med <= 1.0


# ------------------------------------------------------------------ GT identity (oracle) and the host re-judgement
def test_covers_and_gt_states():
    gt = np.zeros((40, 40), np.uint8)
    gt[0:20, 0:20] = 1
    gt[20:40, 20:40] = 2
    m1 = gt == 1
    m_spill = (gt == 2) | (np.arange(40)[None] < 4)
    assert id3d.covers(m1, gt) == 1 and id3d.covers(m_spill, gt) == 2 and id3d.covers(np.zeros_like(m1), gt) == 0
    bg = np.zeros((40, 40), bool)
    bg[0:4, 30:40] = True
    assert id3d.covers(bg, gt) == 0                                   # overlaps no object
    assert id3d.gt_states(np.stack([m1, m1]), [1.0, 1.0], [1, 2], gt) == [PASS, FLAG]
    assert id3d.gt_states(np.stack([m1, bg]), [-1.0, 1.0], [1, 2], gt) == [EMPTY, FLAG]
    up = cv2.resize(m1.astype(np.uint8), (80, 80), interpolation=cv2.INTER_NEAREST) > 0
    assert id3d.covers(up, gt) == 1                                   # a prediction at another resolution is resized


def test_rejudge_reads_written_masks(ring, tmp_path):
    J = _judge(ring, "vote")
    conf, expect = {}, {}
    for c in ring["cams"][1:4]:
        L = ring["labels"][c]
        pred = np.stack([L == 2, L == 2])                   # object 1 swapped, object 2 right
        for i, o in enumerate((1, 2)):
            if pred[i].any():
                os.makedirs(tmp_path / c, exist_ok=True)
                cv2.imwrite(str(tmp_path / c / f"{o}.png"), pred[i].astype(np.uint8) * 255)
        conf[c] = {str(o): dict(presence=2.0, iou=0.9, overlap=None, n_valid=0) for o in (1, 2)}
        st, _ = J.statuses(c, pred, [2.0, 2.0])
        expect[c] = {"1": st[0], "2": st[1]}
    conf = json.loads(json.dumps(conf))
    got = id3d.rejudge(ring["cache"], ring["ref"], ring["masks"], [1, 2], "vote", str(tmp_path), conf,
                       ring["cams"][1:4], (SY.H, SY.W))
    assert got == expect and got["cam01"]["1"] == FLAG and got["cam01"]["2"] == PASS


# ------------------------------------------------------------------ review 0 #9: boundaries; the decide helpers the Step-1 grid uses
def test_split_and_hops_are_inclusive_at_theta():
    """§3.1 / §3.5: a_v == theta is a star view; a hop == theta is admissible (reachability, order and candidates)."""
    a = {"s": 30.0, "b": 45.0, "f": 45.0000001}
    assert id3d.split_views(a, ["s", "b", "f"], 45.0) == (["s", "b"], ["f"])
    hops = {("s", "f"): 45.0, ("b", "f"): 50.0}

    def hop(m, v):
        return hops.get((m, v), 90.0)
    assert id3d.reachable(["s", "b"], ["f"], hop, 45.0) == {"f"}
    assert id3d.relay_order(["s", "b"], ["f"], a, hop, 45.0, ["s", "b", "f"]) == (["f"], [])
    assert id3d.relay_order(["s", "b"], ["f"], a, hop, 44.999, ["s", "b", "f"]) == ([], ["f"])
    centres = {"s": np.zeros(3), "b": np.ones(3), "f": np.full(3, 2.0)}
    assert id3d.relay_candidates("f", ["s", "b"], centres, ["s", "b", "f"], hop=hop, theta=45.0) == ["s"]


def _random_models(rng, ids=(1, 2, 3, 4), wide=(4,)):
    out = {}
    for j in ids:
        R = rng.normal(0.4 * j, 0.15 if j not in wide else 0.8, (400, 3))
        c = np.median(R, 0)
        out[j] = dict(R=R, c=c, rho=float(np.percentile(np.linalg.norm(R - c, axis=1), 90)), n=400, has=True,
                      border=float(rng.choice([0.0, 0.02, 0.2])))
    out[5] = dict(R=np.zeros((3, 3)), c=None, rho=None, n=3, has=False, border=0.0)
    return out


def test_decide_helpers_equal_the_status_functions():
    """centroid_features + c0_decide / cstar_decide and vote_nearest + vote_decide (the Step-1 grid's path) give exactly
    the decode's status_* results over random queries and every A-c* grid point."""
    import itertools
    rng = np.random.default_rng(1)
    models = _random_models(rng)
    ref = id3d.vote_reference(models)
    grid = [dict(zip(id3d.CSTAR_GRID, v)) for v in itertools.product(*id3d.CSTAR_GRID.values())]
    for trial in range(25):
        n = int(rng.choice([5, 25, 300, 5000]))
        Q = rng.normal(rng.uniform(0, 2), rng.uniform(0.05, 0.6), (n, 3))
        o = int(rng.choice([1, 2, 3, 4, 5, 9]))
        feat = id3d.centroid_features(Q, models)
        assert id3d.c0_decide(feat, o, models) == id3d.status_c0(Q, o, models)
        for k in grid[trial::25]:
            for rig_ok in (True, False):
                assert id3d.cstar_decide(feat, o, models, rig_ok=rig_ok, **k) == \
                    id3d.status_cstar(Q, o, models, rig_ok=rig_ok, **k)
        near = id3d.vote_nearest(Q, ref)
        for r_max in (0.2, math.inf):
            lab = id3d.vote_threshold(near, r_max)
            assert np.array_equal(lab, id3d.vote_labels(Q, ref, r_max))
            sv = id3d.vote_decide(lab, len(Q), o, models)
            assert sv == id3d.status_vote(Q, o, models, r_max, ref)
            if r_max == 0.2:
                for nv, ov in ((0, float("nan")), (n, 0.9), (n, 0.1), (n // 3, 0.9)):
                    assert id3d.hybrid_decide(*sv, n, nv, ov) == id3d.status_hybrid(Q, o, models, n, nv, ov, ref=ref)


def test_vote_nearest_without_reference_points():
    near, dist = id3d.vote_nearest(np.ones((30, 3)), (np.zeros((0, 3)), np.zeros(0, np.int64)))
    assert not near.any() and np.isinf(dist).all()
    assert not id3d.vote_threshold((near, dist), math.inf).any()


def test_covers_with_labels_above_255():
    gt = np.zeros((40, 40), np.int32)
    gt[0:20, 0:20] = 300
    gt[20:40, 20:40] = 2
    assert id3d.covers(gt == 300, gt) == 300 and id3d.covers(gt == 2, gt) == 2
