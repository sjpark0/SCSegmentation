"""T10a-l: the Phase 3 / P12 conditioning knobs G (gate), P (pointer), S (tpos row shift).

Host part (torch-free): the pure resolver/gate/counter helpers of xview_gather and the
runner's JSON summary.  Container part (torch + sam3): the REAL
_prepare_memory_conditioned_features_multiple driven through tests/harness.py, with the
knob goldens of p12/golden.py (E1-E6) and the no-flag identity digests of
p12/golden_hash_HEAD.json.

Reading the goldens: a memory/pointer source is 1000*view + frame (harness.mem_out), a
"row" is a maskmem_tpos_enc row index (harness sets row i to the constant i) and a
"t_diff" is an entry of the rel_pos_list handed to _get_tpos_enc.
"""
import json

import pytest

from conftest import load_runmvseg
from xview_gather import (admits, gate_cross_view, new_xview_stats,  # noqa: E402
                          record_xview, resolve_cross_view_knobs)

OWN_N = 5                   # own memory tokens in the t=5 fixtures: cond@0 + frames 1..4
PTR_TOK = 4                 # hidden_dim // mem_dim: tokens per object pointer


def _harness():
    """The container-only import; skips the test on a host without torch/sam3."""
    pytest.importorskip("torch")
    pytest.importorskip("sam3")
    import harness
    return harness


# =============================================================== T10a (host)
def test_resolve_knobs():
    """The constructor resolver: defaults, the S bound W + s <= num_maskmem - 1, and the
    hygiene requirement (p12/golden.py E4)."""
    R = resolve_cross_view_knobs
    for W, h in ((4, False), (0, True), (1, True), (6, True)):
        assert R(None, None, None, W, h, 7) == (False, False, 0), (W, h)
    for W, s in ((1, 0), (1, 2), (1, 4), (1, 5), (2, 4), (4, 2), (6, 0)):
        assert R(None, None, s, W, True, 7) == (False, False, s), (W, s)
    for W, s in ((1, 6), (3, 4), (6, 1), (4, 3)):
        with pytest.raises(ValueError):
            R(None, None, s, W, True, 7)
    with pytest.raises(ValueError):
        R(None, None, -1, 1, True, 7)
    for kw in (dict(cross_view_gate=True), dict(cross_view_ptr=True),
               dict(cross_view_tpos_shift=2)):
        with pytest.raises(ValueError):                      # knobs are XW lineage only
            R(kw.get("cross_view_gate"), kw.get("cross_view_ptr"),
              kw.get("cross_view_tpos_shift"), 4, False, 7)
    assert R(1, 0, 3, 1, True, 7) == (True, False, 3)        # coerced, not passed through
    assert R(False, False, 0, 4, False, 7) == (False, False, 0)   # explicit "no knob"
    assert R(None, None, 2, 2, True, 5) == (False, False, 2)      # bound reads num_maskmem
    with pytest.raises(ValueError):
        R(None, None, 2, 3, True, 5)                             # 3 + 2 > 5 - 1


# =============================================================== T10b (host)
def test_admits_and_gate():
    """`admits` is frame_filter's score test with the no-key case inverted; the gate
    counts what it sees and (only under G) drops what fails."""
    assert admits({"eff_iou_score": 0.011}, 0.01) is True
    assert admits({"eff_iou_score": 0.01}, 0.01) is False        # strict >
    assert admits({"eff_iou_score": 0.0}, 0.01) is False
    assert admits({}, 0.01) is True                              # no key -> admitted
    assert admits({"eff_iou_score": 0.4}, 0.5) is False          # the threshold is a parameter
    a = {"eff_iou_score": 0.0}
    b = {"eff_iou_score": 1.0}
    inp = [(-2, a), (-1, None), (1, b)]
    kept, seen, fail = gate_cross_view(inp, 0.01, apply=True)
    assert (kept, seen, fail) == ([(1, b)], 2, 1)
    same, seen, fail = gate_cross_view(inp, 0.01, apply=False)
    assert same is inp and (seen, fail) == (2, 1)                # unchanged list object
    assert gate_cross_view([], 0.01, apply=True) == ([], 0, 0)
    assert gate_cross_view([(-1, None)], 0.01, apply=False)[1:] == (0, 0)
    st = new_xview_stats()
    assert st == {"calls": 0, "nb_seen": 0, "nb_fail": 0, "per_view": {}}
    record_xview(st, 3, 5, 1, 0)
    record_xview(st, 3, 5, 1, 1)
    assert st["per_view"][3][5] == (2, 1)
    assert (st["calls"], st["nb_seen"], st["nb_fail"]) == (2, 2, 1)


# =============================================================== T10c (host)
def test_stats_summary():
    """runMVSeg.xview_stats_summary: JSON-clean, None when no knob ran."""
    mod = load_runmvseg()
    assert mod.xview_stats_summary(None, True) is None
    assert mod.xview_stats_summary(new_xview_stats(), True) is None      # calls == 0
    st = new_xview_stats()
    record_xview(st, 3, 5, 1, 0)
    record_xview(st, 3, 5, 1, 1)
    s = mod.xview_stats_summary(st, True)
    assert s["gate"] is True and s["calls"] == 2
    assert (s["nb_seen"], s["nb_fail"], s["nb_dropped"]) == (2, 1, 1)
    assert (s["cells"], s["cells_fail"]) == (1, 1)
    assert s["per_view"] == {"3": {"seen": 2, "fail": 1, "frames_fail": [[5, 1]]}}
    assert list(s["per_view"]) == ["3"] and all(isinstance(k, str) for k in s["per_view"])
    assert json.loads(json.dumps(s)) == s
    assert mod.xview_stats_summary(st, False)["nb_dropped"] == 0         # P-only: counted, not dropped
    assert mod.xview_stats_summary(st, False)["nb_fail"] == 1


# ------------------------------------------------------- container fixtures
def ods_with_eff(H, N, v, t, start=0, eff_at_t=None, drop_key_at_t=(), no_eff_anywhere=False):
    """Lockstep dicts (harness.output_dicts_lockstep) with the neighbours' frame-t entry
    doctored: eff_at_t {view: eff}, drop_key_at_t = views whose entry has no
    "eff_iou_score" at all (memory selection off, or a consolidated seed cond entry)."""
    import torch
    ods = H.output_dicts_lockstep(N, v, t, start, H.SMALL_HW)
    for m, od in enumerate(ods):
        if no_eff_anywhere:
            for d in (od["cond_frame_outputs"], od["non_cond_frame_outputs"]):
                for o in d.values():
                    o.pop("eff_iou_score", None)
            continue
        out = od["non_cond_frame_outputs"].get(t)
        if out is None:
            continue
        if eff_at_t and m in eff_at_t:
            out["eff_iou_score"] = torch.tensor(float(eff_at_t[m]))
            out["object_score_logits"] = torch.tensor([[-3.0 if eff_at_t[m] <= 0.01 else 10.0]])
        if m in drop_key_at_t:
            out.pop("eff_iou_score", None)
    return ods


# =============================================================== T10d (container)
# sha256 over the harness lockstep records (prompt, prompt_pos, num_obj_ptr_tokens) in
# (t, v) order, N=10 sessions, T=20 frames, SMALL_HW; captured at HEAD 9bd4607 by
# p12/golden_hash.py -> p12/golden_hash_HEAD.json.  Tied to this image's torch
# (2.10.0+cu128) and to the harness RNG seeds: regenerate from the UNPATCHED method if
# either changes, never from a knobbed run.
HEAD_DIGESTS = {
    "legacy": ("00842562c1f211c581091ad019b468ab3a3e15778621a491c35c0c8233576c60", 118776),
    "XW0": ("514b8bf5810163175394fb60c0bbf45a1cd69151a94955f32a5171f4ddaabd96", 80376),
    "XW1": ("9e82c6dcff4466d84aa577fe00f1251ecdbbd9efe1d0cf6dd484a538a809c888", 91896),
    "XW1A": ("9e82c6dcff4466d84aa577fe00f1251ecdbbd9efe1d0cf6dd484a538a809c888", 91896),
    "XW2": ("b5d06cbf8398468849a58fa1f53517b918d0dded10982936be7a406f746c45a9", 102136),
    "XW4": ("78d56f94cde3f8d929c392d1ef0c6da03fb60f142d2cb4d80137db7d9e62c7ab", 118776),
    "XW6": ("37c3babaf338998dfb5896fa4bdb22e41019c7e2ab0a4ad963189508b95f3ef4", 130296),
    "XW1B": ("e42f753b353d30c743bcd9b64b9624c29c2c3c4f51342239c06968e6dc24d8e3", 103416),
    "XW1C": ("6239b4faf08533f072fa07f56d99dfdd1774bc693e432cd0a187c834f28425d6", 103416),
    "XW1D": ("1b97c09c2025f974edf389e23a9e36a4a22c7702fd49bfd7f7d099769d3af97d", 91896),
    "XW1E": ("44c1697a299f7a3ad1aabd8e7c254bbc129902649254faf72d153eeb5bc35dc0", 103416),
}
NO_KNOB_CONFIGS = [("legacy", dict(xw=None, xh=None, xm=None), False),
                   ("XW0", dict(xw=0, xh=True, xm=None), False),
                   ("XW1", dict(xw=1, xh=True, xm=None), False),
                   ("XW1A", dict(xw=1, xh=True, xm="A"), False),
                   ("XW2", dict(xw=2, xh=True, xm=None), False),
                   ("XW4", dict(xw=4, xh=True, xm=None), False),
                   ("XW6", dict(xw=6, xh=True, xm=None), False),
                   ("XW1B", dict(xw=1, xh=True, xm="B"), False),
                   ("XW1C", dict(xw=1, xh=True, xm="C"), False),
                   ("XW1D", dict(xw=1, xh=True, xm="D"), False),
                   ("XW1E", dict(xw=1, xh=True, xm="E"), True)]


def lockstep_digest(rec):
    import hashlib
    h = hashlib.sha256()
    n_tok = 0
    for key in sorted(rec, key=lambda k: (k[1], k[0])):          # (t, v) order
        p, pp, n = rec[key]
        h.update(p.detach().numpy().tobytes())
        h.update(pp.detach().numpy().tobytes())
        h.update(str(n).encode())
        n_tok += p.shape[0]
    return h.hexdigest(), n_tok


def test_no_knob_identity_digests(cpu_tensors):
    """Every no-flag configuration is byte-identical to HEAD, with the knobs absent and
    with them passed explicitly as (False, False, 0); nothing is counted either."""
    H = _harness()
    for name, cfg, two in NO_KNOB_CONFIGS:
        want, want_tokens = HEAD_DIGESTS[name]
        for knobs in ({}, dict(xg=False, xp=False, xs=0)):
            tr = H.bare_tracker(**cfg, **knobs)
            assert (tr.cross_view_gate, tr.cross_view_ptr, tr.cross_view_tpos_shift) == (False, False, 0)
            rec = H.lockstep(tr, 10, 20, hw=H.SMALL_HW, two_pass=two)
            got, tokens = lockstep_digest(rec)
            assert len(rec) == 200, (name, knobs)
            assert (got, tokens) == (want, want_tokens), (name, knobs)
            assert tr.xview_stats["calls"] == 0, (name, knobs)


# =============================================================== T10e (container)
EFF_FAIL = {1: 0.0, 4: 0.005}          # frame-5 entries that fail eff_iou_score > 0.01
DROP_KEY = (3,)                        # frame-5 entry without the key -> admitted
KNOB_VARIANTS = [("XW1", {}), ("XW1G", dict(xg=True)), ("XW1P", dict(xp=True)),
                 ("XW1GP", dict(xg=True, xp=True)), ("XW1S2", dict(xs=2)),
                 ("XW1S4", dict(xs=4)), ("XW1GPS2", dict(xg=True, xp=True, xs=2))]


def test_knob_goldens_t5(cpu_tensors):
    """p12/golden.py E1: N=6, t=5, W=1.  The memory and pointer channels are independent
    (G alone never touches pointers, P alone never touches memory), G covers both, and S
    shifts the row and the pointer position by s."""
    H = _harness()
    N, T = 6, 5
    table = {}
    for name, knobs in KNOB_VARIANTS:
        tr = H.bare_tracker(xw=1, xh=True, **knobs)
        for v in range(N):
            ods = ods_with_eff(H, N, v, T, eff_at_t=EFF_FAIL, drop_key_at_t=DROP_KEY)
            r = H.run(tr, ods, v, T, H.SMALL_HW)
            table[(name, v)] = r
            table[(name, v, "stats")] = tr.xview_stats["per_view"].get(v, {}).get(T)
            assert r["max_abs_pos"] == 16, (name, v)
    for v in range(1, N):
        nb = v - 1
        base = table[("XW1", v)]
        assert base["mem_src"][OWN_N:] == [1000 * nb + T] and base["tpos_rows"][OWN_N:] == [0]
        assert base["ptr_src"] == [1000 * v, 1000 * v + 4, 1000 * v + 3, 1000 * v + 2]
        assert base["ptr_pos"] == [5, 1, 2, 3] and base["n_ptr_tokens"] == 16
        assert table[("XW1", v, "stats")] is None            # no knob -> no bookkeeping
        fails = nb in EFF_FAIL
        g = table[("XW1G", v)]
        if fails:
            assert (g["n_mem"], g["mem_src"]) == (OWN_N, base["mem_src"][:OWN_N]), v
        else:
            assert g["mem_src"] == base["mem_src"] and g["tpos_rows"] == base["tpos_rows"], v
        assert g["ptr_src"] == base["ptr_src"] and g["ptr_pos"] == base["ptr_pos"], v
        assert table[("XW1G", v, "stats")] == (1, 1 if fails else 0), v
        p = table[("XW1P", v)]
        assert p["mem_src"] == base["mem_src"] and p["tpos_rows"] == base["tpos_rows"], v
        assert p["ptr_src"] == base["ptr_src"] + [1000 * nb + T], v
        assert p["ptr_pos"] == [5, 1, 2, 3, 1], v
        assert p["n_ptr_tokens"] == base["n_ptr_tokens"] + PTR_TOK == 20, v
        assert table[("XW1P", v, "stats")] == (1, 1 if fails else 0), v
        gp = table[("XW1GP", v)]
        if fails:
            assert gp["n_mem"] == OWN_N and gp["ptr_src"] == base["ptr_src"], v
            assert gp["n_ptr_tokens"] == 16, v
        else:
            assert gp["mem_src"] == base["mem_src"] and gp["ptr_src"] == p["ptr_src"], v
        for s in (2, 4):
            r = table[(f"XW1S{s}", v)]
            assert r["mem_src"] == base["mem_src"], (v, s)
            assert r["tpos_rows"] == base["tpos_rows"][:OWN_N] + [s], (v, s)
            assert r["ptr_src"] == base["ptr_src"] and r["ptr_pos"] == base["ptr_pos"], (v, s)
        r = table[("XW1GPS2", v)]
        if not fails:
            assert r["tpos_rows"][OWN_N:] == [2] and r["ptr_pos"] == [5, 1, 2, 3, 3], v
    for name, _ in KNOB_VARIANTS:                     # v = 0 has no neighbour at all
        r0 = table[(name, 0)]
        assert (r0["n_mem"], r0["n_ptr_tokens"]) == (OWN_N, 16), name
        assert r0["mem_src"] == table[("XW1", 0)]["mem_src"], name
        assert r0["ptr_src"] == table[("XW1", 0)]["ptr_src"], name
        assert table[(name, 0, "stats")] in (None, (0, 0)), name


# =============================================================== T10f (container)
def test_first_tracked_frame(cpu_tensors):
    """p12/golden.py E2: at t = start+1 the own pointer list is the cond pointer alone; in
    mode A the lower neighbours already hold non_cond[t] (lockstep), so P appends theirs."""
    H = _harness()
    N = 6
    base = H.run(H.bare_tracker(xw=1, xh=True), ods_with_eff(H, N, 3, 1), 3, 1, H.SMALL_HW, 22)
    assert base["mem_src"] == [3000, 2001] and base["tpos_rows"] == [6, 0]
    assert base["ptr_src"] == [3000] and base["ptr_pos"] == [1] and base["n_ptr_tokens"] == 4
    p = H.run(H.bare_tracker(xw=1, xh=True, xp=True), ods_with_eff(H, N, 3, 1), 3, 1, H.SMALL_HW, 22)
    assert p["ptr_src"] == [3000, 2001] and p["ptr_pos"] == [1, 1] and p["n_ptr_tokens"] == 8
    gp = H.run(H.bare_tracker(xw=1, xh=True, xg=True, xp=True),
               ods_with_eff(H, N, 3, 1, eff_at_t={2: 0.0}), 3, 1, H.SMALL_HW, 22)
    assert gp["mem_src"] == [3000] and gp["ptr_src"] == [3000] and gp["n_ptr_tokens"] == 4
    s2 = H.run(H.bare_tracker(xw=1, xh=True, xg=True, xp=True, xs=2),
               ods_with_eff(H, N, 3, 1), 3, 1, H.SMALL_HW, 22)
    assert s2["tpos_rows"] == [6, 2] and s2["ptr_pos"] == [1, 3]
    s4 = H.run(H.bare_tracker(xw=1, xh=True, xs=4), ods_with_eff(H, N, 3, 1), 3, 1, H.SMALL_HW, 22)
    assert s4["tpos_rows"] == [6, 4] and s4["ptr_src"] == base["ptr_src"]
    for knobs in ({}, dict(xp=True), dict(xg=True, xp=True), dict(xg=True, xp=True, xs=2), dict(xs=4)):
        r0 = H.run(H.bare_tracker(xw=1, xh=True, **knobs), ods_with_eff(H, N, 0, 1), 0, 1, H.SMALL_HW, 22)
        assert (r0["mem_src"], r0["tpos_rows"], r0["n_ptr_tokens"]) == ([0], [6], 4), knobs
    late = H.run(H.bare_tracker(xw=1, xh=True, xp=True),
                 H.output_dicts_lockstep(N, 3, 31, 30, H.SMALL_HW), 3, 31, H.SMALL_HW, 52)
    assert late["mem_src"] == [3030, 2031] and late["ptr_src"] == [3030, 2031]
    assert late["ptr_pos"] == [1, 1]


# =============================================================== T10g (container)
def test_pointer_cap_extra(cpu_tensors):
    """p12/golden.py E3 + p12/cap_probe.py: the own pointer list keeps its cap and the
    neighbour pointer is EXTRA.  The own list is 1 cond + <= 15 non-cond = <= 16 pointers
    with memory selection ON and OFF alike (frame_filter appends must_include after its
    >= max_num-1 break), so XW1P/XW1GP reach <= 17 pointers = 68 tokens on late frames --
    the nominal max_obj_ptrs_in_encoder (16) is exceeded by one, by design."""
    H = _harness()
    import torch
    N, v, t = 6, 3, 20
    seq = H.SMALL_HW * H.SMALL_HW
    for use_sel, n_own in ((True, 15), (False, 16)):
        base = None
        for name, knobs in (("XW1", {}), ("XW1P", dict(xp=True)), ("XW1GP", dict(xg=True, xp=True))):
            tr = H.bare_tracker(xw=1, xh=True, use_sel=use_sel, **knobs)
            ods = ods_with_eff(H, N, v, t, no_eff_anywhere=not use_sel)
            r = H.run(tr, ods, v, t, H.SMALL_HW, 22)
            if name == "XW1":
                base = r
                assert len(r["ptr_src"]) == n_own and r["n_ptr_tokens"] == PTR_TOK * n_own
                assert r["ptr_pos"] == [20] + list(range(1, n_own)), (use_sel, r["ptr_pos"])
                continue
            assert len(r["ptr_src"]) == n_own + 1, (use_sel, name)
            assert r["n_ptr_tokens"] == PTR_TOK * (n_own + 1), (use_sel, name)
            assert r["ptr_src"][:n_own] == base["ptr_src"] and r["ptr_pos"][:n_own] == base["ptr_pos"]
            assert r["ptr_src"][n_own] == 1000 * (v - 1) + t and r["ptr_pos"][n_own] == 1
            # own memories (7) + neighbour (1) + own pointer tokens are byte-identical
            keep = 8 * seq + PTR_TOK * n_own          # 7 own memories + 1 neighbour + own ptrs
            assert torch.equal(r["prompt"][:keep], base["prompt"][:keep])
            assert torch.equal(r["ppos"][:keep], base["ppos"][:keep])
    # the bound case (selection ON): frame t-1 fails, 15 earlier frames pass ->
    # len(valid_indices) == 16 -> 16 own pointers, 17 with P
    NN, vv, tt, num_frames = 3, 2, 20, 40
    ods = []
    for m in range(NN):
        nc = {f: H.mem_out(f, m, H.SMALL_HW, eff=(0.0 if f == tt - 1 else 1.0)) for f in range(1, tt)}
        if m < vv:
            nc[tt] = H.mem_out(tt, m, H.SMALL_HW, eff=1.0)
        ods.append({"cond_frame_outputs": {0: H.mem_out(0, m, H.SMALL_HW)},
                    "non_cond_frame_outputs": nc})
    tr = H.bare_tracker(xw=1, xh=True)
    assert len(tr.frame_filter(ods[vv], False, tt, num_frames, 1)) == 16
    b = H.run(tr, ods, vv, tt, H.SMALL_HW, num_frames)
    assert len(b["ptr_src"]) == 16 and b["n_ptr_tokens"] == 64
    pr = H.run(H.bare_tracker(xw=1, xh=True, xp=True), ods, vv, tt, H.SMALL_HW, num_frames)
    assert len(pr["ptr_src"]) == 17 and pr["n_ptr_tokens"] == 68
    assert pr["ptr_src"][:16] == b["ptr_src"] and pr["ptr_pos"][:16] == b["ptr_pos"]
    assert pr["ptr_src"][16] == 1000 * (vv - 1) + tt and pr["ptr_pos"][16] == 1


# =============================================================== T10h (container)
def test_selection_off_and_no_key(cpu_tensors):
    """p12/golden.py E5: an entry without "eff_iou_score" is admitted -- frame_filter
    skips such an entry, G deliberately keeps it (memory selection off, or a seed cond
    entry reached under a t-1 mode)."""
    H = _harness()
    N, v, T = 6, 3, 5
    for name, knobs in (("XW1", {}), ("XW1G", dict(xg=True)), ("XW1GP", dict(xg=True, xp=True))):
        tr = H.bare_tracker(xw=1, xh=True, use_sel=False, **knobs)
        r = H.run(tr, ods_with_eff(H, N, v, T, no_eff_anywhere=True), v, T, H.SMALL_HW)
        assert r["mem_src"][OWN_N:] == [2005], name
        if knobs:
            assert tr.xview_stats["per_view"][v][T] == (1, 0), name
        if name == "XW1GP":
            assert r["ptr_src"][-1] == 2005 and r["ptr_pos"][-1] == 1
    tr = H.bare_tracker(xw=1, xh=True, xg=True, xp=True)          # selection ON, key missing
    r = H.run(tr, ods_with_eff(H, N, v, T, drop_key_at_t=(2,)), v, T, H.SMALL_HW)
    assert r["mem_src"][OWN_N:] == [2005] and r["ptr_src"][-1] == 2005
    assert tr.xview_stats["per_view"][v][T] == (1, 0)


# =============================================================== T10i (container)
def test_reverse_positions(cpu_tensors):
    """p12/golden.py E6: the neighbour pointer position is k + s, positive in both
    tracking directions (the own non-cond convention)."""
    H = _harness()
    N, v, T = 6, 3, 5

    def ods():
        out = [{"cond_frame_outputs": {0: H.mem_out(0, m, H.SMALL_HW)},
                "non_cond_frame_outputs": {f: H.mem_out(f, m, H.SMALL_HW) for f in (1, 2, 3, 4, 6)}}
               for m in range(N)]
        for m in range(v):
            out[m]["non_cond_frame_outputs"][T] = H.mem_out(T, m, H.SMALL_HW)
        return out

    p = H.run(H.bare_tracker(xw=1, xh=True, xp=True), ods(), v, T, H.SMALL_HW, rev=True)
    assert p["mem_src"] == [3000, 3006, 2005] and p["tpos_rows"] == [6, 0, 0]
    assert p["ptr_src"] == [2005] and p["ptr_pos"] == [1]
    s2 = H.run(H.bare_tracker(xw=1, xh=True, xg=True, xp=True, xs=2), ods(), v, T, H.SMALL_HW, rev=True)
    assert s2["tpos_rows"] == [6, 0, 2] and s2["ptr_pos"] == [3]


# =============================================================== T10j (container)
def test_sensitivity_and_index0(cpu_tensors):
    """Each knob changes the drive (the digests are sensitive), the index-0 session is
    identical to XW1's under every knob (sentinel Z1 at harness level), and the counters
    see one call per (session, frame) and one neighbour per session with an index >= 1."""
    H = _harness()
    NN, T = 10, 20
    ref_tr = H.bare_tracker(xw=1, xh=True)
    ref = H.lockstep(ref_tr, NN, T, hw=H.SMALL_HW)
    assert ref_tr.xview_stats["calls"] == 0
    digests = {"XW1": lockstep_digest(ref)[0]}
    for name, knobs in (("G", dict(xg=True)), ("P", dict(xp=True)), ("S2", dict(xs=2)),
                        ("GP", dict(xg=True, xp=True))):
        tr = H.bare_tracker(xw=1, xh=True, **knobs)
        rec = H.lockstep(tr, NN, T, hw=H.SMALL_HW)
        digests[name] = lockstep_digest(rec)[0]
        for t in range(1, T + 1):
            assert H.same_inputs(rec[(0, t)], ref[(0, t)]), (name, t)     # index 0: no neighbour
        st = tr.xview_stats
        assert st["calls"] == NN * T, (name, st["calls"])
        assert st["nb_seen"] == (NN - 1) * T, (name, st["nb_seen"])
        assert st["nb_fail"] > 0, name
        # view 0 is recorded (one cell per frame) but never sees a neighbour
        assert all(c == (0, 0) for c in st["per_view"][0].values()), name
        assert len(st["per_view"][0]) == T, name
    assert len(set(digests.values())) == len(digests), digests


# =============================================================== T10k (container)
def test_cone_with_knobs(cpu_tensors):
    """The dependency cone of mode A is unchanged by the knobs (they read only the entries
    the gather returned), plus a mode C + GP + S2 smoke: the knobs are defined for every
    mode (`abs(s_pos)`), only mode A is run."""
    H = _harness()
    T, K, NN, scored_max = 4, 3, 12, 2
    for knobs in (dict(xg=True), dict(xp=True), dict(xg=True, xp=True, xs=2)):
        tr = H.bare_tracker(xw=1, xh=True, **knobs)
        full = H.lockstep(tr, NN, T, hw=H.SMALL_HW)
        tr2 = H.bare_tracker(xw=1, xh=True, **knobs)
        trunc = H.lockstep(tr2, K, T, hw=H.SMALL_HW)
        assert [(v, t) for v in range(scored_max + 1) for t in range(1, T + 1)
                if not H.same_inputs(full[(v, t)], trunc[(v, t)])] == [], knobs
    r = H.run(H.bare_tracker(xw=1, xh=True, xm="C", xg=True, xp=True, xs=2),
              H.output_dicts_lockstep(10, 5, 5, 0, H.SMALL_HW), 5, 5, H.SMALL_HW)
    assert r["mem_src"][OWN_N:] == [4005, 6004] and r["tpos_rows"][OWN_N:] == [2, 2]
    assert r["ptr_src"] == [5000, 5004, 5003, 5002, 4005, 6004]
    assert r["ptr_pos"] == [5, 1, 2, 3, 3, 3]


# =============================================================== T10l (container)
def test_gate_reads_mf_threshold(cpu_tensors):
    """The gate uses the tracker's mf_threshold, not a hard-coded 0.01 (adversarial
    review A5): at mf_threshold = 0.5 a neighbour scoring 0.2 is dropped and one scoring
    0.6 is kept, while at 0.01 both are kept."""
    H = _harness()
    N, v, T = 6, 3, 5
    for mf, dropped in ((0.5, True), (0.01, False)):
        tr = H.bare_tracker(xw=1, xh=True, xg=True, xp=True, mf=mf)
        assert tr.mf_threshold == mf
        r = H.run(tr, ods_with_eff(H, N, v, T, eff_at_t={2: 0.2}), v, T, H.SMALL_HW)
        assert (r["mem_src"][OWN_N:] == []) is dropped, mf
        assert (r["ptr_src"] == [3000, 3004, 3003, 3002]) is dropped, mf
        assert tr.xview_stats["per_view"][v][T] == (1, 1 if dropped else 0), mf
    tr = H.bare_tracker(xw=1, xh=True, xg=True, mf=0.5)
    r = H.run(tr, ods_with_eff(H, N, v, T, eff_at_t={2: 0.6}), v, T, H.SMALL_HW)
    assert r["mem_src"][OWN_N:] == [2005]
    assert tr.xview_stats["per_view"][v][T] == (1, 0)
