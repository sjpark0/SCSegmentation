"""GPU (scsam3): S1-E11 in-view exclusivity -- the registered rule on synthetic logits, the flag-off path byte-identical
to the adopted output (Z4 at Blocks scale), and the X1 / X2 / D invariants on a real Blocks decode (Z1, Z2, Z5x)."""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods import common  # noqa: E402
from methods.geomem import run as R  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data")
RUNS = os.path.join(REPO, "geomem", "runs")
E8 = os.path.join(DATA, "Blocks", "Result", "geoset_pred_ft_rep_e8")
needs_data = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks")) or not os.path.isdir(E8)
    or not os.path.isdir(os.path.join(RUNS, "geoset_pred_ft")),
    reason="needs GPU, Data/Blocks, geoset_pred_ft checkpoints and the _e8 result folder")


# ------------------------------------------------------------------ the rule on synthetic logits (CPU torch)
def _logits(shapes, h=40, w=60, base=5.0):
    """shapes: list of (y0, y1, x0, x1, value) rectangles -> [B,h,w] logits, NO_OBJ elsewhere."""
    L = torch.full((len(shapes), h, w), -1024.0)
    for b, (y0, y1, x0, x1, v) in enumerate(shapes):
        L[b, y0:y1, x0:x1] = v
    return L


def test_exclusive_masks_non_nested_loser_keeps_only_winning_pixels():
    L = _logits([(0, 20, 0, 40, 6.0), (0, 20, 20, 70, 4.0)], w=80)      # overlap x in [20,40): object 0 wins; object 1 loses 40 %
    keep, c, recs = R.exclusive_masks(L)
    assert keep[0].sum() == 20 * 40 and keep[1].sum() == 20 * 30
    assert not (keep[0] & keep[1]).any()
    assert c == dict(contested_px=400, removed_entries=400, masks_changed=1, masks_exempt=0, masks_emptied=0, masks_present=2)
    assert recs[1]["lost"] == 400 and recs[1]["exempt"] is False and recs[0]["removed"] == 0


def test_exclusive_masks_nested_loser_is_exempt_and_container_keeps_everything():
    L = _logits([(0, 40, 0, 60, 6.0), (10, 20, 10, 20, 4.0)])            # object 1 fully inside object 0 and losing everywhere
    keep, c, recs = R.exclusive_masks(L)
    assert torch.equal(keep, L > 0), "the exempt mask and its container must be unchanged"
    assert c["masks_exempt"] == 1 and c["masks_changed"] == 0 and c["removed_entries"] == 0 and c["contested_px"] == 100
    keep_d, c_d, _ = R.exclusive_masks(L, tau_keep=None)                 # D: no cap -> the nested loser is emptied
    assert keep_d[1].sum() == 0 and c_d["masks_emptied"] == 1 and c_d["masks_exempt"] == 0 and c_d["removed_entries"] == 100


def test_exclusive_masks_cap_threshold_exact_and_ties_to_lowest_index():
    L = _logits([(0, 10, 0, 20, 6.0), (0, 10, 10, 30, 4.0)])            # object 1: 200 px, loses 100 = exactly 50 % -> exempt
    keep, c, recs = R.exclusive_masks(L)
    assert recs[1]["exempt"] is True and torch.equal(keep, L > 0)
    L2 = _logits([(0, 10, 0, 20, 5.0), (0, 10, 10, 40, 5.0)])           # exact tie -> lowest index wins; object 1 loses 100/300
    keep2, _, recs2 = R.exclusive_masks(L2)
    assert keep2[0].sum() == 200 and keep2[1].sum() == 200 and recs2[1]["exempt"] is False
    absent = _logits([(0, 10, 0, 20, 5.0)]); absent = torch.cat([absent, torch.full_like(absent, -1024.0)])
    keep3, c3, _ = R.exclusive_masks(absent)
    assert keep3[1].sum() == 0 and c3["masks_present"] == 1 and c3["contested_px"] == 0


def test_exclusive_logits_1008_rule():
    L = _logits([(0, 20, 0, 40, 6.0), (0, 20, 20, 70, 4.0), (0, 20, 30, 50, 8.0)], h=20, w=80)[:, None]
    out, c = R.exclusive_logits_1008(L, [True, True, False])            # object 2 not eligible: untouched and never wins
    assert torch.equal(out[2], L[2]) and torch.equal(out[0], L[0])
    assert (out[1, 0, :, 20:40] == -10.0).all() and (out[1, 0, :, 40:70] == 4.0).all()   # object 1 loses 400 of 1000 px (40 %)
    assert c == dict(contested_cells=400, clamped_cells=400, masks_exempt=0, eligible=2)
    same, c0 = R.exclusive_logits_1008(L, [True, False, False])
    assert same is L and c0["clamped_cells"] == 0
    nested = _logits([(0, 20, 0, 60, 6.0), (5, 10, 5, 10, 4.0)], h=20, w=80)[:, None]
    out2, c2 = R.exclusive_logits_1008(nested, [True, True])
    assert torch.equal(out2, nested) and c2["masks_exempt"] == 1


# ------------------------------------------------------------------ S1-E14 S: the shifted threshold (CPU torch)
def test_shift_zero_is_the_adopted_rule_bit_for_bit():
    g = torch.Generator().manual_seed(0)
    L = torch.randn(5, 30, 40, generator=g) * 3
    L[3] = -1024.0                                                          # an absent object
    a, ca, ra = R.exclusive_masks(L)
    b, cb, rb = R.exclusive_masks(L, R.XCL_TAU_KEEP, 0.0)
    assert torch.equal(a, b) and ca == cb and ra == rb


def test_shift_claims_only_where_the_object_is_the_argmax():
    L = torch.full((2, 4, 16), -1024.0)
    L[0, :, 0:4] = 1.0                                                      # A holds x 0..3
    L[0, :, 4:6] = -0.3                                                     # A, unclaimed, above -0.5: gained
    L[0, :, 6:8] = -0.6                                                     # A below -0.5: not gained
    L[1, :, 2:4] = 0.8                                                      # B inside A's pixels, B < A: never taken from A
    L[1, :, 8:14] = 1.0                                                     # B's own pixels
    L[1, :, 14:16] = -0.3                                                   # B alone above -0.5: gained
    keep0, _, r0 = R.exclusive_masks(L)
    keep, c, recs = R.exclusive_masks(L, R.XCL_TAU_KEEP, R.XCL_SHIFT)
    assert keep0[0, 0].tolist() == [True] * 4 + [False] * 12 and keep0[1, 0].tolist() == [False] * 8 + [True] * 6 + [False] * 2
    assert keep[0, 0].tolist() == [True] * 6 + [False] * 10
    assert keep[1, 0].tolist() == [False] * 8 + [True] * 8
    assert r0[1]["exempt"] is False and recs[1] == dict(px=40, lost=8, exempt=False, kept=32, removed=8)
    absent = torch.cat([L[:1], torch.full_like(L[:1], -1024.0)])
    k2, c2, _ = R.exclusive_masks(absent, R.XCL_TAU_KEEP, R.XCL_SHIFT)
    assert k2[1].sum() == 0 and c2["masks_present"] == 1


def test_shift_superset_except_exempt_to_nonexempt_flips():
    # B: 100 px of its own + 40 contested px it loses to A -> not exempt at 0 (40 / 140) nor under the shift
    L = torch.full((2, 10, 60), -1024.0)
    L[0, :, 0:24] = 2.0                                                     # A
    L[1, :, 20:34] = 1.0                                                    # B: x 20..23 contested (A wins), x 24..33 own
    L[1, :, 34:44] = -0.3                                                   # B gains 100 px under the shift
    L[0, :, 44:60] = -0.3; L[1, :, 44:60] = -0.2                            # contested only under the shift: B is the argmax
    k0, _, r0 = R.exclusive_masks(L)
    ks, cs, rs = R.exclusive_masks(L, R.XCL_TAU_KEEP, R.XCL_SHIFT)
    assert r0[1]["exempt"] is False and rs[1]["exempt"] is False and rs[0]["exempt"] is False
    assert (k0 & ~ks).sum() == 0, "a pixel was removed from an object without an exemption flip"
    assert ks[1].sum() == k0[1].sum() + 100 + 160 and torch.equal(ks[0], k0[0])
    # off -> on flip: B loses 40 of 100 at 0 (not exempt); under the shift it adds 100 px it loses to A -> 140 / 200, exempt
    L2 = torch.full((2, 10, 60), -1024.0)
    L2[0, :, 0:24] = 2.0; L2[0, :, 30:40] = 2.0
    L2[1, :, 20:30] = 1.0; L2[1, :, 30:40] = -0.3
    k0, _, r0 = R.exclusive_masks(L2)
    ks, _, rs = R.exclusive_masks(L2, R.XCL_TAU_KEEP, R.XCL_SHIFT)
    assert r0[1]["exempt"] is False and rs[1]["exempt"] is True and (k0 & ~ks).sum() == 0
    # on -> off flip: B exempt at 0 (loses exactly 50 %), gains 100 own px under the shift -> 50 / 200, not exempt:
    # B drops the 50 contested px it kept while exempt -- the only legitimate removal (I2 excludes these objects)
    L3 = torch.full((2, 10, 60), -1024.0)
    L3[0, :, 0:10] = 2.0
    L3[1, :, 5:15] = 1.0
    L3[1, :, 15:25] = -0.3
    k0, _, r0 = R.exclusive_masks(L3)
    ks, _, rs = R.exclusive_masks(L3, R.XCL_TAU_KEEP, R.XCL_SHIFT)
    assert r0[1]["exempt"] is True and rs[1]["exempt"] is False
    assert (k0[1] & ~ks[1]).sum() == 50 and (k0[0] & ~ks[0]).sum() == 0 and ks[1].sum() == 150


def test_shift_never_empties_a_mask_and_keeps_the_true_argmax():
    L = torch.full((2, 8, 8), -1024.0)
    L[0] = 3.0
    L[1, 2:6, 2:6] = 1.0; L[1, 0:2, :] = -0.2                               # B loses all of its pos to A -> exempt, kept whole
    ks, cs, rs = R.exclusive_masks(L, R.XCL_TAU_KEEP, R.XCL_SHIFT)
    assert rs[1]["exempt"] is True and ks[1].sum() == 16 + 16 and cs["masks_emptied"] == 0
    # two logits one float32 ulp apart at pixel 0: the comparison form keeps the true argmax (B); adding 0.5 to the
    # logits could round them to one value and hand the pixel to the lower index (A)
    a = torch.tensor(0x3E99999B, dtype=torch.int32).view(torch.float32)   # 0.30000004 (review: a pair that DOES tie at +0.5)
    b = torch.nextafter(a, torch.tensor(1.0))
    assert bool(a < b) and bool((a + 0.5) == (b + 0.5)), "the case must be a float32 tie under the addition form"
    T = torch.full((2, 1, 4), -1024.0)
    T[0, 0, 0] = a; T[1, 0, 0] = b; T[0, 0, 1:] = 5.0                       # A: 4 pos px, loses pixel 0 (25 %, not exempt)
    ks, _, rs = R.exclusive_masks(T, R.XCL_TAU_KEEP, R.XCL_SHIFT)
    assert bool(ks[1, 0, 0]) and not bool(ks[0, 0, 0]) and rs[0]["lost"] == 1 and rs[0]["exempt"] is False
    ka, _, _ = R.exclusive_masks(T + 0.5)                                   # the forbidden addition form: the tie goes to A
    assert bool(ka[0, 0, 0]), "the addition form no longer ties here -- the test would not guard the comparison form"


# ------------------------------------------------------------------ real decodes on Blocks
@pytest.fixture(scope="module")
def model():
    from methods import sam3_single
    if R.decode_scene.model is None:
        R.decode_scene.model = sam3_single.load_model()
        R.decode_scene.init_state = R.snapshot_init(R.decode_scene.model)
    yield R.decode_scene.model
    R.decode_scene.model = None
    R.decode_scene.init_state = None
    torch.cuda.empty_cache()


def _args(out, **kw):
    base = dict(data=DATA, out=out, r1_out=out + "_r1", arm="geoset_pred_ft", gate="reproj", untrained=False, runs=RUNS,
                tau=0.7, k=3, round2="pred")
    base.update(kw)
    return argparse.Namespace(**base)


def _pngs(d, skip=()):
    out = {}
    for cam in sorted(os.listdir(d)):
        if cam in skip or not os.path.isdir(os.path.join(d, cam)):
            continue
        for f in sorted(os.listdir(os.path.join(d, cam))):
            if f.endswith(".png"):
                out[(cam, f)] = open(os.path.join(d, cam, f), "rb").read()
    return out


def _masks(d, skip=()):
    return {k: cv2.imread(os.path.join(d, *k), 0) > 127 for k in _pngs(d, skip)}


def _rm(*names):
    for n in names:
        shutil.rmtree(os.path.join(DATA, "Blocks", "Result", n), ignore_errors=True)


@needs_data
def test_xcl_none_is_byte_identical_to_the_adopted_output(model):
    scene = common.load_scene(DATA, "Blocks")
    _rm("dbg_x0", "dbg_x0_r1")
    cfg = R.decode_scene(scene, _args("dbg_x0", xcl="none"))
    assert "xcl" not in cfg
    a, b = _pngs(os.path.join(DATA, "Blocks", "Result", "dbg_x0"), skip=(scene.ref_cam,)), _pngs(E8, skip=(scene.ref_cam,))
    assert a and set(a) == set(b) and all(a[k] == b[k] for k in a)


@needs_data
def test_outs_invariants_on_blocks(model):
    """S1-E14 S on a real Blocks decode against `out` (the adopted rule): round 1 untouched, arm >= out for every object
    except an exempt -> non-exempt flip, nothing emptied, key set >= out's, kept == PNG pixels, config records the shift."""
    scene = common.load_scene(DATA, "Blocks")
    _rm("dbg_xs_out", "dbg_xs_out_r1", "dbg_xs_outs", "dbg_xs_outs_r1")
    c_out = R.decode_scene(scene, _args("dbg_xs_out", xcl="out"))
    cfg = R.decode_scene(scene, _args("dbg_xs_outs", xcl="outs"))
    assert cfg["xcl"] == "outs" and cfg["xcl_shift"] == 0.5 and cfg["xcl_tau_keep"] == 0.5 and "xcl_shift" not in c_out
    d, d0 = (os.path.join(DATA, "Blocks", "Result", n) for n in ("dbg_xs_outs", "dbg_xs_out"))
    assert _pngs(d + "_r1") == _pngs(d0 + "_r1")
    assert open(os.path.join(d, "round1_conf.json"), "rb").read() == open(os.path.join(d0, "round1_conf.json"), "rb").read()
    recs, recs0 = (json.load(open(os.path.join(x, "xcl_conf.json"))) for x in (d, d0))
    arm, base = _masks(d, skip=(scene.ref_cam,)), _masks(d0, skip=(scene.ref_cam,))
    assert set(arm) >= set(base) and cfg["n_xcl_masks_emptied"] == 0
    added = removed_flip = removed_other = flips_on = flips_off = 0
    for cam in recs:
        for o, r in recs[cam].items():
            k = (cam, f"{o}.png")
            a = arm.get(k)
            assert r["kept"] == (0 if a is None else int(a.sum())), (k, r)
            b = base.get(k)
            if b is None:
                continue
            a = a if a is not None else np.zeros_like(b)
            rem = int((b & ~a).sum())
            e0, e1 = recs0[cam][o]["exempt"], r["exempt"]
            flips_on += int(e1 and not e0); flips_off += int(e0 and not e1)
            if e0 and not e1:
                removed_flip += rem
            else:
                removed_other += rem
            added += int((a & ~b).sum())
    assert removed_other == 0, "a pixel was removed from an object without an exempt -> non-exempt flip"
    assert added > 0, "the shift added nothing (dead branch?)"
    assert sum(r["exempt"] for v in recs.values() for r in v.values()) == cfg["n_xcl_masks_exempt"]
    print(f"\n[outs] added {added} px, removed (on->off flips) {removed_flip}, flips on {flips_on} off {flips_off}, "
          f"exempt {cfg['n_xcl_masks_exempt']} (out {c_out['n_xcl_masks_exempt']})")
    _rm("dbg_xs_out", "dbg_xs_out_r1", "dbg_xs_outs", "dbg_xs_outs_r1")


@needs_data
@pytest.mark.parametrize("mode", ["out", "out_nocap", "outmem"])
def test_xcl_invariants_on_blocks(model, mode):
    scene = common.load_scene(DATA, "Blocks")
    out = f"dbg_x_{mode}"
    _rm(out, out + "_r1")
    cfg = R.decode_scene(scene, _args(out, xcl=mode))
    d = os.path.join(DATA, "Blocks", "Result", out)
    assert cfg["xcl"] == mode and cfg["xcl_tau_keep"] == (None if mode == "out_nocap" else 0.5)
    # Z2: round 1 untouched
    assert _pngs(d + "_r1", skip=(scene.ref_cam,)) == _pngs(os.path.join(DATA, "Blocks", "Result", "geoset_pred_ft_r1_e8"), skip=(scene.ref_cam,))
    assert open(os.path.join(d, "round1_conf.json"), "rb").read() == open(os.path.join(E8, "round1_conf.json"), "rb").read()
    recs = json.load(open(os.path.join(d, "xcl_conf.json")))
    assert sum(len(v) for v in recs.values()) == len([c for c in scene.cams if c != scene.ref_cam]) * len(scene.objects)
    arm, base = _masks(d, skip=(scene.ref_cam,)), _masks(E8, skip=(scene.ref_cam,))
    if mode in ("out", "out_nocap"):
        # Z5x: keep is a subset of C0, only C0-contested pixels change, at most one non-exempt survivor per contested pixel
        assert set(arm) <= set(base)
        n_removed = n_changed = 0
        for cam in sorted({k[0] for k in base}):
            objs = sorted(k[1] for k in base if k[0] == cam)
            B0 = np.stack([base[(cam, f)] for f in objs])
            A = np.stack([arm[(cam, f)] if (cam, f) in arm else np.zeros_like(base[(cam, f)]) for f in objs])
            contested = B0.sum(0) >= 2
            assert not (A & ~B0).any(), "a pixel was added"
            diff = A != B0
            assert not diff[:, ~contested].any(), "a non-contested pixel changed"
            exempt = np.array([recs[cam][f[:-4]]["exempt"] for f in objs])
            survivors = A[~exempt].sum(0) if (~exempt).any() else np.zeros_like(contested, dtype=int)
            assert survivors.max() <= 1, "two non-exempt masks still overlap"
            n_removed += int((B0 & ~A).sum()); n_changed += int(diff.reshape(len(objs), -1).any(1).sum())
        assert n_removed == cfg["n_xcl_removed_entries"] and n_changed == cfg["n_xcl_masks_changed"]
        # all five registered counters against the PNGs / records (Z1)
        n_contested = sum(int((np.stack([base[(cam, f)] for f in sorted(k[1] for k in base if k[0] == cam)]).sum(0) >= 2).sum()) for cam in {k[0] for k in base})
        n_exempt = sum(1 for cam in recs for o, r in recs[cam].items() if r["exempt"])
        n_emptied = sum(1 for cam in recs for o, r in recs[cam].items() if r["px"] > 0 and r["kept"] == 0)
        assert n_contested == cfg["n_xcl_contested_px"] and n_exempt == cfg["n_xcl_masks_exempt"] and n_emptied == cfg["n_xcl_masks_emptied"]
        if mode == "out":
            assert set(arm) == set(base) and cfg["n_xcl_masks_emptied"] == 0
        else:
            assert cfg["n_xcl_masks_emptied"] == len(set(base) - set(arm)) and cfg["n_xcl_masks_exempt"] == 0
        print(f"\n[{mode}] contested px {cfg['n_xcl_contested_px']} removed {cfg['n_xcl_removed_entries']} masks changed {cfg['n_xcl_masks_changed']} "
              f"exempt {cfg['n_xcl_masks_exempt']} emptied {cfg['n_xcl_masks_emptied']}")
    else:
        # X2: the output rule holds (as for X1) and the memory rule is live: counters non-zero, output differs from X1 somewhere
        assert cfg["n_xmem_contested_cells"] > 0 and cfg["n_xmem_eligible"] > 0
        assert set(arm) == set(base) and cfg["n_xcl_masks_emptied"] == 0
        for cam in sorted({k[0] for k in arm}):
            objs = sorted(k[1] for k in arm if k[0] == cam)
            A = np.stack([arm[(cam, f)] for f in objs])
            exempt = np.array([recs[cam][f[:-4]]["exempt"] for f in objs])
            if (~exempt).any():
                assert (A[~exempt].sum(0) > 1).sum() == 0, "two non-exempt masks overlap in the X2 output"
        x1 = os.path.join(DATA, "Blocks", "Result", "dbg_x_out")
        if os.path.isdir(x1):
            p1, p2 = _pngs(x1, skip=(scene.ref_cam,)), _pngs(d, skip=(scene.ref_cam,))
            assert set(p1) == set(p2) and any(p1[k] != p2[k] for k in p1), "the memory rule left the output unchanged (dead branch?)"
        print(f"\n[outmem] mem contested cells {cfg['n_xmem_contested_cells']} clamped {cfg['n_xmem_clamped_cells']} "
              f"exempt {cfg['n_xmem_masks_exempt']} eligible {cfg['n_xmem_eligible']}")
