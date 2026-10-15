#!/usr/bin/env python3
"""S1-E14 judge (docs/stage1-E14-prereg.md sections 3-7).  Host, stage1/.venv (numpy, cv2):
    PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python MVSeed/e14_judge.py --stage A|B|1|2 [--audited]

  --stage A   DEV-A, before DEV-B: Z-emul (arm PNGs == host emulation of the round-2 logit dump, pos = L > -0.5 with the
              argmax on the unshifted L; dev_score rows == cand/bil2_tau0.5.json; dev_compare reproduces the DEV-A
              contrast), invariants I1-I4 + Z1 against the DEV-A base, mechanism and quality reference values.
              Exit 2 on any failure.
  --stage B   DEV-B: pair-set check (exit 3), invariants I1-I4 + Z1 (exit 2 = a bug: fix and redo the decode), the
              stop-and-audit check (all-pair dJ&F > +0.006: exit 4, only that number is printed, until --audited),
              then B1-B3 and the report values.  Exit 0 whenever judged (pass or fail is in the JSON).
  --stage 1   keyframe (664 pairs): G1/G2 against C0 = geoset_pred_ft_rep_x_e11, the audit threshold (+0.008: exit 4
              until --audited), Z1/Z2 and the report values.
  --stage 2   stage 2: the E8 section-5 rule against the headline SegMaskSam3XW1CGPS4MSdgeoset_rep_xFsCs10kG2Rs.
All stages merge into MVSeed/runs/e14/e14_judge.json.
"""
import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(REPO, "stage1")
DATA = os.path.join(S1, "Data")
DEV = os.path.join(DATA, "_dev")
OUT_DIR = os.path.join(REPO, "MVSeed", "runs", "e14")
OUT = os.path.join(OUT_DIR, "e14_judge.json")
BASE_PRESET, ARM_PRESET = "geoset_rep_x", "geoset_rep_xs"
TAG = ARM_PRESET
SHIFT, TAU_KEEP = 0.5, 0.5
DUMP = os.path.join(S1, "geomem", "runs", "_e14_logitdump", "geoset_rep_x_A")
DUMP_CODE = os.path.join(REPO, "MVSeed", "refloss", "e14", "logitdump", "code")
CAND = os.path.join(REPO, "MVSeed", "refloss", "e14", "threshold-upsample", "cand", "bil2_tau0.5.json")
AUDIT_B, B3_FLOOR = 0.006, -0.010           # section 4
AUDIT_KF, SAFETY, DEFENCE = 0.008, -0.01, -1.0
BAND_ORACLE_A = 0.0401                      # DEV-A good-seed +-2 px band oracle gain (0.9767 - 0.9366), the P6 scale
C0, C0_R1 = "geoset_pred_ft_rep_x_e11", "geoset_pred_ft_r1_x_e11"
ARM, ARM_R1 = "geoset_pred_ft_rep_xs_e14", "geoset_pred_ft_r1_xs_e14"
HEADLINE = "SegMaskSam3XW1CGPS4MSdgeoset_rep_xFsCs10kG2Rs"
METHOD2 = f"SegMaskSam3XW1CGPS4MSd{TAG}FsCs10kG2Rs"
SCENES = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "CBABasketball", "Carpark", "CoffeeMartini",
          "Dog", "Fencing", "FlameSteak", "Frog", "MATF", "MartialArts", "Painter", "PoznanStreet", "Welder")
SIZE_BINS = (("<2k", 0, 2_000), ("2k-10k", 2_000, 10_000), ("10k-100k", 10_000, 100_000), (">=100k", 100_000, float("inf")))
NPROC = max(1, min(14, (os.cpu_count() or 2) - 2))


def _load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _finite(*xs):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _save(stage_key, res):
    out = _load_json(OUT) if os.path.isfile(OUT) else {}
    out["prereg"] = "docs/stage1-E14-prereg.md"
    out[stage_key] = res
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))


# ----------------------------------------------------------------------------------------------- pure rules (unit-tested)
def pair_key(r):
    return (r["scene"], int(r["t"]), r["cam"], int(r["obj"]))


def seed_kind(r):
    """Failure type of a seed (MVSeed/refloss/seedcat.py rules; as e11_judge._kind)."""
    if r["seed_px"] == 0 or (r["recall"] or 0) < 0.05:
        return "miss" if r["seed_px"] == 0 or r["gt_px"] < 50 else "wrong-object"
    if (r["precision"] or 0) < 0.3:
        return "wrong-object"
    if r["frac_other"] >= 0.2:
        return "spill-onto-other" + ("(twin)" if r["twin"] else "")
    if r["recall"] < 0.85 and r["precision"] >= 0.85:
        return "under-cover" + ("(beyond c_ini)" if r["vis_ratio"] > 1.2 else "")
    return "boundary/mixed"


def check_pairs(base, arm):
    """The DEV pair sets must be identical (section 4 'abort')."""
    kb, ka = {pair_key(r) for r in base}, {pair_key(r) for r in arm}
    return dict(ok=kb == ka and len(kb) == len(base) and len(ka) == len(arm), n_base=len(base), n_arm=len(arm),
                only_base=len(kb - ka), only_arm=len(ka - kb))


def audit_needed(delta, limit):
    """Stop and audit when the effect is implausibly large (or not a number)."""
    return not _finite(delta) or delta > limit


def dev_b_rule(jf, good):
    """B1-B3 from the all-pair J&F contrast {delta, ci, per_scene} and the good-seed contrast {delta, ci}; NaN fails."""
    worst = min(jf["per_scene"].items(), key=lambda x: x[1]) if jf.get("per_scene") else (None, float("nan"))
    b1 = _finite(jf["delta"], jf["ci"][0]) and jf["delta"] > 0 and jf["ci"][0] > 0
    b2 = _finite(good["delta"]) and good["delta"] >= 0
    b3 = bool(jf.get("per_scene")) and all(_finite(v) for v in jf["per_scene"].values()) and worst[1] >= B3_FLOOR
    return dict(B1=b1, B2=b2, B3=b3, worst_scene=worst[0], worst_delta=worst[1], **{"pass": b1 and b2 and b3})


def keyframe_gate(jf):
    worst = min(jf["per_scene"].items(), key=lambda x: x[1]) if jf.get("per_scene") else (None, float("nan"))
    g1 = _finite(jf["delta"], jf["ci"][0]) and jf["delta"] > 0 and jf["ci"][0] > 0
    g2 = bool(jf.get("per_scene")) and all(_finite(v) for v in jf["per_scene"].values()) and worst[1] >= SAFETY
    return dict(G1=g1, G2=g2, worst_scene=worst[0], worst_delta=worst[1], **{"pass": g1 and g2})


def object_class(rec_base, rec_arm):
    """Which invariant applies to one object-view: 'same' (I2: arm >= base), 'on' (non-exempt -> exempt: arm >= base
    holds too), 'off' (exempt -> non-exempt: the only legitimate removal)."""
    eb, ea = bool(rec_base["exempt"]), bool(rec_arm["exempt"])
    return "same" if eb == ea else ("on" if ea else "off")


def emul_keep(L, shift=SHIFT, tau=TAU_KEEP):
    """The registered S rule in numpy, independent of run.py: pos = L > -shift, argmax on the unshifted L (ties -> lowest
    index), exempt = lost >= tau * |pos|; -> (keep [B,h,w], exempt [B])."""
    import numpy as np
    B = L.shape[0]
    pos = L > np.float32(-shift)
    win = L.argmax(0)[None] == np.arange(B)[:, None, None]
    npos = pos.reshape(B, -1).sum(1)
    lost = (pos & ~win).reshape(B, -1).sum(1)
    exempt = (npos > 0) & (lost.astype(np.float32) >= np.float32(tau) * npos.astype(np.float32))
    return pos & (win | exempt[:, None, None]), exempt


# ----------------------------------------------------------------------------------------------- DEV frame checks
def _read(path, hw):
    import numpy as np
    from mvseg2d.io import read_pred_mask
    if not os.path.isfile(path):
        return np.zeros(hw, bool)
    m = read_pred_mask(path)
    assert m.shape == tuple(hw), (path, m.shape, hw)
    return m


def _tree_bytes(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for cam in sorted(os.listdir(d)):
        p = os.path.join(d, cam)
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".png"):
                    with open(os.path.join(p, f), "rb") as fh:
                        out[(cam, f)] = fh.read()
    return out


def _bytes(p):
    with open(p, "rb") as f:
        return f.read()


def frame_check(job):
    """I1-I4, Z1 and the mechanism counters of one DEV frame (base root vs arm root)."""
    sys.path.insert(0, S1)
    import numpy as np
    from mvseg2d.io import load_labelmap
    scene, t, base_root, arm_root = job
    fb, fa = os.path.join(base_root, scene, f"{t:06d}"), os.path.join(arm_root, scene, f"{t:06d}")
    c, bad = Counter(), []
    for d in (fb, fa):
        if not os.path.isfile(os.path.join(d, "DONE")):
            return dict(counts=c, bad=[("missing", d)], seconds=(0.0, 0.0))
    cb, ca = _load_json(os.path.join(fb, "config.json")), _load_json(os.path.join(fa, "config.json"))
    if not (ca.get("xcl") == "outs" and ca.get("xcl_shift") == SHIFT and ca.get("xcl_tau_keep") == TAU_KEEP):
        bad.append(("I4", scene, t, ca.get("xcl"), ca.get("xcl_shift")))
    if not (cb.get("xcl") == "out" and "xcl_shift" not in cb):
        bad.append(("I4-base", scene, t, cb.get("xcl")))
    r1b, r1a = _tree_bytes(os.path.join(fb, "_r1")), _tree_bytes(os.path.join(fa, "_r1"))
    if set(r1b) != set(r1a) or any(r1b[k] != r1a[k] for k in r1b):
        bad.append(("I1-png", scene, t, len(r1b), len(r1a), sum(r1b[k] != r1a.get(k) for k in r1b)))
    if _bytes(os.path.join(fb, "round1_conf.json")) != _bytes(os.path.join(fa, "round1_conf.json")):
        bad.append(("I1-conf", scene, t))
    c["r1_png"] += len(r1a)
    if ca["objects"] != cb["objects"] or ca["eval_cams"] != cb["eval_cams"]:
        bad.append(("objects/cams differ", scene, t))
        return dict(counts=c, bad=bad, seconds=(cb.get("seconds", 0.0), ca.get("seconds", 0.0)))
    xb, xa = _load_json(os.path.join(fb, "xcl_conf.json")), _load_json(os.path.join(fa, "xcl_conf.json"))
    objs = [int(o) for o in ca["objects"]]
    for cam in ca["eval_cams"]:
        gt = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
        hw = gt.shape
        Bm = np.stack([_read(os.path.join(fb, cam, f"{o}.png"), hw) for o in objs]) if objs else np.zeros((0, *hw), bool)
        Am = np.stack([_read(os.path.join(fa, cam, f"{o}.png"), hw) for o in objs]) if objs else np.zeros((0, *hw), bool)
        for i, o in enumerate(objs):
            rb, ra = xb[cam][str(o)], xa[cam][str(o)]
            c["object_views"] += 1
            nb, na = int(Bm[i].sum()), int(Am[i].sum())
            if ra["kept"] != na or rb["kept"] != nb:
                bad.append(("Z1-kept", scene, t, cam, o, ra["kept"], na, rb["kept"], nb))
            if ra["px"] > 0 and ra["kept"] == 0:
                bad.append(("I3-emptied", scene, t, cam, o))
            if nb > 0 and na == 0:
                bad.append(("I3-keyset", scene, t, cam, o))
            if nb == 0 and na > 0:
                c["newly_nonempty"] += 1
            rem = int((Bm[i] & ~Am[i]).sum())
            k = object_class(rb, ra)
            c[f"class_{k}"] += 1
            if k == "same" and rem:
                bad.append(("I2-removed", scene, t, cam, o, rem))
            c[f"removed_{k}"] += rem
            c["exempt_base"] += int(bool(rb["exempt"])); c["exempt_arm"] += int(bool(ra["exempt"]))
        if not objs:
            continue
        added = Am & ~Bm
        union_b = Bm.any(0)
        c["added_px"] += int(added.sum())
        c["added_unclaimed"] += int((added & ~union_b[None]).sum())
        c["multi_base"] += int((Bm.sum(0) >= 2).sum()); c["multi_arm"] += int((Am.sum(0) >= 2).sum())
        ids = np.array(objs)[:, None, None]
        own = added & (gt[None] == ids)
        other = added & np.isin(gt, objs)[None] & ~own
        c["added_own"] += int(own.sum()); c["added_other"] += int(other.sum())
        c["added_undecoded"] += int((added & ~np.isin(gt, objs)[None]).sum())
        newly = Am.any(0) & ~union_b
        c["newly_claimed_px"] += int(newly.sum())
        c["newly_claimed_correct"] += int((newly & (Am & (gt[None] == ids)).any(0)).sum())
        c["unclaimed_base_px"] += int((~union_b).sum())
    return dict(counts=c, bad=bad, seconds=(float(cb.get("seconds", 0.0)), float(ca.get("seconds", 0.0))))


def dev_invariants(split):
    from multiprocessing import Pool
    base_root, arm_root = os.path.join(DEV, f"{BASE_PRESET}_{split}"), os.path.join(DEV, f"{ARM_PRESET}_{split}")
    idx = _load_json(os.path.join(arm_root, "index.json"))
    idx_b = _load_json(os.path.join(base_root, "index.json"))
    jobs = [(s, int(t), base_root, arm_root) for s in sorted(idx["scenes"]) for t in idx["scenes"][s]["frames"]]
    res = dict(split=split, frames=len(jobs), index_equal=idx["scenes"] == idx_b["scenes"] and idx["split"] == split == idx_b["split"])
    with Pool(NPROC) as pool:
        parts = pool.map(frame_check, jobs, chunksize=1)
    c, bad = Counter(), []
    sec_b = sec_a = 0.0
    for p in parts:
        c.update(p["counts"]); bad.extend(p["bad"])
        sec_b += p["seconds"][0]; sec_a += p["seconds"][1]
    scenes_done = {s for s in idx["scenes"] if idx["scenes"][s]["frames"]}
    res.update(counts=dict(c), bad=bad[:200], n_bad=len(bad), seconds_base=sec_b, seconds_arm=sec_a,
               time_ratio=sec_a / sec_b if sec_b else None, scenes=len(scenes_done))
    res["mechanism"] = dict(
        added_px=c["added_px"], added_unclaimed_share=c["added_unclaimed"] / c["added_px"] if c["added_px"] else None,
        removed_same=c["removed_same"], removed_on=c["removed_on"], removed_off=c["removed_off"],
        flips_on=c["class_on"], flips_off=c["class_off"], newly_nonempty=c["newly_nonempty"],
        added_own_share=c["added_own"] / c["added_px"] if c["added_px"] else None,
        added_other_share=c["added_other"] / c["added_px"] if c["added_px"] else None,
        added_undecoded_share=c["added_undecoded"] / c["added_px"] if c["added_px"] else None,
        newly_claimed_owner_accuracy=c["newly_claimed_correct"] / c["newly_claimed_px"] if c["newly_claimed_px"] else None,
        newly_claimed_share_of_unclaimed=c["newly_claimed_px"] / c["unclaimed_base_px"] if c["unclaimed_base_px"] else None,
        multi_base=c["multi_base"], multi_arm=c["multi_arm"], multi_ratio=c["multi_arm"] / c["multi_base"] if c["multi_base"] else None)
    res["ok"] = res["n_bad"] == 0 and res["index_equal"] and res["frames"] > 0
    return res


def empty_gt_stats(base, arm):
    def st(d):
        e = d.get("empty_gt", [])
        return dict(n=len(e), nonempty=sum(r["seed_px"] > 0 for r in e), px=sum(int(r["seed_px"]) for r in e))
    return dict(base=st(base), arm=st(arm))


def quality(base_rows, arm_rows):
    """Good-seed (base J&F >= 0.8) boundary proxies, pair means: 1 - recall, 1 - precision; size bins by GT px."""
    ka = {pair_key(r): r for r in arm_rows}
    good = [r for r in base_rows if r["JF"] >= 0.8]

    def m(rs, f):
        xs = [f(r) for r in rs]
        return statistics.mean(xs) if xs else None
    q = dict(n_good=len(good),
             base_1_recall=m(good, lambda r: 1 - r["recall"]), arm_1_recall=m(good, lambda r: 1 - ka[pair_key(r)]["recall"]),
             base_1_precision=m(good, lambda r: 1 - (r["precision"] or 0)),
             arm_1_precision=m(good, lambda r: 1 - (ka[pair_key(r)]["precision"] or 0)))
    q["size_bins_good"] = {}
    for name, lo, hi in SIZE_BINS:
        sel = [r for r in good if lo <= r["gt_px"] < hi]
        q["size_bins_good"][name] = dict(n=len(sel), dJF=m(sel, lambda r: ka[pair_key(r)]["JF"] - r["JF"]))
    return q


def dev_contrast(base_rows, arm_rows):
    sys.path.insert(0, S1)
    from mvseg2d.dev_compare import compare
    from mvseg2d.paired import paired_scene_bootstrap
    cmp = compare(base_rows, arm_rows)
    ka = {pair_key(r): r for r in arm_rows}
    out = dict(compare=cmp)
    for name, sel in (("good", [r for r in base_rows if r["JF"] >= 0.8]), ("fail", [r for r in base_rows if r["JF"] < 0.8])):
        b = paired_scene_bootstrap(sel, [ka[pair_key(r)] for r in sel], key="JF", n=10_000, seed=0, view_key="view")
        out[name] = dict(n=len(sel), delta=b["delta"], ci=list(b["ci"]), per_scene=b["per_scene"],
                         pair_mean=statistics.mean(ka[pair_key(r)]["JF"] - r["JF"] for r in sel) if sel else None)
    d = [ka[pair_key(r)]["JF"] - r["JF"] for r in base_rows]
    out["down_gt_0.01"], out["up_gt_0.01"] = sum(x < -0.01 for x in d), sum(x > 0.01 for x in d)
    out["pair_mean"] = statistics.mean(d) if d else None
    return out


def _load_rows(split, preset):
    sys.path.insert(0, S1)
    from mvseg2d.dev_compare import load
    p = os.path.join(DEV, f"{preset}_{split}", "dev_pairs.json")
    return load(p), _load_json(p)


# ----------------------------------------------------------------------------------------------- stage A (DEV-A)
def _emul_view(job):
    sys.path.insert(0, DUMP_CODE)
    sys.path.insert(0, S1)
    import numpy as np
    import e14dump as E
    scene, t, cam, arm_root = job
    d = E.load(DUMP, scene, t, cam)
    objs, hw = [int(o) for o in d["objects"]], tuple(int(v) for v in d["hw"])
    L = E.logits_full_np(d["low_res"], hw)
    keep, exempt = emul_keep(L)
    recs = _load_json(os.path.join(arm_root, scene, f"{t:06d}", "xcl_conf.json"))[cam]
    out = dict(object_views=len(objs), identical=0, diff_px=0, exempt_mismatch=0, diffs=[])
    for i, o in enumerate(objs):
        a = _read(os.path.join(arm_root, scene, f"{t:06d}", cam, f"{o}.png"), hw)
        n = int((a != keep[i]).sum())
        out["identical"] += int(n == 0); out["diff_px"] += n
        out["exempt_mismatch"] += int(bool(recs[str(o)]["exempt"]) != bool(exempt[i]))
        if n:
            out["diffs"].append((scene, t, cam, o, n))
    return out


def stage_a():
    from multiprocessing import Pool
    sys.path.insert(0, DUMP_CODE)
    sys.path.insert(0, S1)
    import e14dump as E
    from mvseg2d.dev_compare import compare, load
    arm_root = os.path.join(DEV, f"{ARM_PRESET}_A")
    views = E.views(DUMP)
    with Pool(NPROC) as pool:
        parts = pool.map(_emul_view, [(s, t, c, arm_root) for s, t, c in views], chunksize=1)
    emul = dict(views=len(views), object_views=sum(p["object_views"] for p in parts), identical=sum(p["identical"] for p in parts),
                diff_px=sum(p["diff_px"] for p in parts), exempt_mismatch=sum(p["exempt_mismatch"] for p in parts),
                diffs=[d for p in parts for d in p["diffs"]][:50])
    emul["ok"] = emul["views"] == 170 and emul["object_views"] == 3952 and emul["identical"] == emul["object_views"] \
        and emul["diff_px"] == 0 and emul["exempt_mismatch"] == 0
    base, base_raw = _load_rows("A", BASE_PRESET)
    arm, arm_raw = _load_rows("A", ARM_PRESET)
    cand = load(CAND)
    kc, ka = {pair_key(r): r for r in cand}, {pair_key(r): r for r in arm}
    rows_eq = dict(n_arm=len(arm), n_cand=len(cand), same_keys=set(kc) == set(ka),
                   max_abs_dJ=max((abs(ka[k]["J"] - kc[k]["J"]) for k in ka if k in kc), default=None),
                   max_abs_dF=max((abs(ka[k]["F"] - kc[k]["F"]) for k in ka if k in kc), default=None))
    rows_eq["ok"] = rows_eq["same_keys"] and rows_eq["n_arm"] == rows_eq["n_cand"] == 3300 and rows_eq["max_abs_dJ"] <= 1e-12 \
        and rows_eq["max_abs_dF"] <= 1e-12
    c_arm, c_cand = compare(base, arm), compare(base, cand)
    pa, pc = c_arm["paired"]["JF"], c_cand["paired"]["JF"]
    contrast = dict(arm=dict(delta=pa["delta"], ci=pa["ci"]), cand=dict(delta=pc["delta"], ci=pc["ci"]),
                    ok=abs(pa["delta"] - pc["delta"]) <= 1e-12 and all(abs(a - b) <= 1e-12 for a, b in zip(pa["ci"], pc["ci"])))
    pairs = check_pairs(base, arm)
    inv = dev_invariants("A")
    con = dev_contrast(base, arm)
    res = dict(Z_emul=emul, rows_equal_cand=rows_eq, contrast_reproduced=contrast, pairs=pairs, invariants=inv,
               contrast=con, empty_gt=empty_gt_stats(base_raw, arm_raw), quality=quality(base, arm))
    res["ok"] = emul["ok"] and rows_eq["ok"] and contrast["ok"] and pairs["ok"] and inv["ok"]
    _save("stageA", res)
    p = con["compare"]["paired"]
    print(f"Z-emul: {emul['identical']}/{emul['object_views']} object-views identical, {emul['diff_px']} px differ, exempt mismatches "
          f"{emul['exempt_mismatch']} -> {'OK' if emul['ok'] else 'FAIL'}")
    print(f"rows == cand/bil2_tau0.5.json: {rows_eq}")
    print(f"DEV-A contrast: arm {pa['delta']:+.4f} [{pa['ci'][0]:+.4f}, {pa['ci'][1]:+.4f}]  cand {pc['delta']:+.4f} "
          f"[{pc['ci'][0]:+.4f}, {pc['ci'][1]:+.4f}] -> {'OK' if contrast['ok'] else 'FAIL'}")
    print(f"   dJ {p['J']['delta']:+.4f}  dF {p['F']['delta']:+.4f}  good {con['good']['delta']:+.5f} [{con['good']['ci'][0]:+.5f}, "
          f"{con['good']['ci'][1]:+.5f}]  fail {con['fail']['delta']:+.4f}  down/up {con['down_gt_0.01']}/{con['up_gt_0.01']}")
    print(f"invariants (I1-I4, Z1): {inv['frames']} frames, bad {inv['n_bad']} {inv['bad'][:5]} -> {'OK' if inv['ok'] else 'FAIL'}")
    print(f"mechanism {json.dumps(inv['mechanism'])}\nempty-GT {res['empty_gt']}\nquality {json.dumps(res['quality'])}"
          f"\ntime arm/base {inv['time_ratio']}")
    print(f"stage A: {'OK' if res['ok'] else 'FAIL'}")
    if not res["ok"]:
        sys.exit(2)


# ----------------------------------------------------------------------------------------------- stage B (DEV-B)
def stage_b(audited=False):
    base, base_raw = _load_rows("B", BASE_PRESET)
    arm, arm_raw = _load_rows("B", ARM_PRESET)
    pairs = check_pairs(base, arm)
    if not pairs["ok"]:
        _save("stageB", dict(pairs=pairs, note="pair sets differ: abort (section 4)"))
        print(f"ABORT: pair sets differ {pairs}")
        sys.exit(3)
    inv = dev_invariants("B")
    if not inv["ok"]:
        _save("stageB", dict(pairs=pairs, invariants=inv, note="invariant violated: a bug, fix and redo the decode (section 4)"))
        print(f"ABORT: invariants violated ({inv['n_bad']}): {inv['bad'][:10]}")
        sys.exit(2)
    con = dev_contrast(base, arm)
    jf = con["compare"]["paired"]["JF"]
    audit = audit_needed(jf["delta"], AUDIT_B)
    rule = dev_b_rule(jf, con["good"])
    res = dict(pairs=pairs, invariants=inv, contrast=con, rule=rule, audit_required=audit, audited=bool(audited),
               empty_gt=empty_gt_stats(base_raw, arm_raw), quality=quality(base, arm))
    p = con["compare"]["paired"]
    sp = con["compare"]["splits"]
    mech, q = inv["mechanism"], res["quality"]
    pr = dict(
        P1=dict(delta=jf["delta"], in_80=0.0015 <= jf["delta"] <= 0.0040, good_in=0.0013 <= con["good"]["delta"] <= 0.0025,
                dJ_gt_dF=p["J"]["delta"] > p["F"]["delta"], down_share=con["down_gt_0.01"] / len(base),
                down_share_in=0.015 <= con["down_gt_0.01"] / len(base) <= 0.035,
                best_scene=max(jf["per_scene"].items(), key=lambda x: x[1])[0], worst_scene=rule["worst_scene"]),
        P2=dict(unclaimed_share_ok=(mech["added_unclaimed_share"] or 0) >= 0.98, removed_same_ok=mech["removed_same"] == 0,
                flips_ok=mech["flips_on"] + mech["flips_off"] <= 10, newly_nonempty_ok=mech["newly_nonempty"] <= 3,
                empty_gt_count_ok=abs(res["empty_gt"]["arm"]["nonempty"] - res["empty_gt"]["base"]["nonempty"]) <= 2,
                empty_gt_px_ok=res["empty_gt"]["arm"]["px"] <= 1.05 * res["empty_gt"]["base"]["px"],
                owner_accuracy_ok=mech["newly_claimed_owner_accuracy"] is not None and 0.60 <= mech["newly_claimed_owner_accuracy"] <= 0.70,
                multi_ratio_ok=mech["multi_ratio"] is not None and 1.2 <= mech["multi_ratio"] <= 1.7),
        P6=dict(band_oracle_closed=con["good"]["delta"] / BAND_ORACLE_A, ok_band=con["good"]["delta"] / BAND_ORACLE_A <= 0.06,
                spill=sp.get("kind:spill-onto-other", {}).get("dJF"), twin=sp.get("kind:spill-onto-other(twin)", {}).get("dJF"),
                under=sp.get("kind:under-cover", {}).get("dJF"), under_beyond=sp.get("kind:under-cover(beyond c_ini)", {}).get("dJF")),
        cost=dict(time_ratio=inv["time_ratio"], ok=inv["time_ratio"] is not None and inv["time_ratio"] <= 1.05))
    res["predictions"] = pr
    _save("stageB", res)
    if audit and not audited:
        print(f"STOP AND AUDIT (section 4): all-pair dJ&F {jf['delta']:+.4f} > +{AUDIT_B} -- audit for a bug before reading "
              f"anything else; rerun with --audited once the audit finds none")
        sys.exit(4)
    print(f"DEV-B {len(base)} pairs: dJ&F {jf['delta']:+.4f} [{jf['ci'][0]:+.4f}, {jf['ci'][1]:+.4f}]  dJ {p['J']['delta']:+.4f} "
          f"[{p['J']['ci'][0]:+.4f}, {p['J']['ci'][1]:+.4f}]  dF {p['F']['delta']:+.4f} [{p['F']['ci'][0]:+.4f}, {p['F']['ci'][1]:+.4f}]")
    print(f"   good {con['good']['n']} {con['good']['delta']:+.5f} [{con['good']['ci'][0]:+.5f}, {con['good']['ci'][1]:+.5f}] (pair mean "
          f"{con['good']['pair_mean']:+.5f})  fail {con['fail']['n']} {con['fail']['delta']:+.4f} [{con['fail']['ci'][0]:+.4f}, "
          f"{con['fail']['ci'][1]:+.4f}]  down/up {con['down_gt_0.01']}/{con['up_gt_0.01']}")
    print("   per scene " + "  ".join(f"{s} {v:+.4f}" for s, v in sorted(jf["per_scene"].items(), key=lambda x: x[1])))
    print(f"   splits {json.dumps(sp)}\n   size bins (good) {json.dumps(q['size_bins_good'])}")
    print(f"   good 1-recall {q['base_1_recall']:.4f} -> {q['arm_1_recall']:.4f}, 1-precision {q['base_1_precision']:.4f} -> {q['arm_1_precision']:.4f}")
    print(f"   mechanism {json.dumps(mech)}\n   empty-GT {res['empty_gt']}\n   invariants OK ({inv['frames']} frames, {inv['counts'].get('object_views')} object-views)"
          f"  time arm/base {inv['time_ratio']}")
    print(f"   predictions {json.dumps(pr)}")
    print(f"DEV-B rule: B1 {rule['B1']}  B2 {rule['B2']}  B3 {rule['B3']} (worst {rule['worst_scene']} {rule['worst_delta']:+.4f}) -> "
          + ("PASS: go once to the keyframe gate" if rule["pass"] else "FAIL: S1-E14 closed as 'no effect' (DEV-B spent)"))


# ----------------------------------------------------------------------------------------------- stage 1 (keyframe)
def _result(scene, folder):
    return os.path.join(DATA, scene, "Result", folder)


def kf_checks():
    """Z2 (round 1 == C0's, 17/17) and Z1 / containment on the keyframe folders."""
    import numpy as np
    sys.path.insert(0, S1)
    from mvseg2d.io import read_pred_mask
    z2 = dict(scenes_ok=0, conf_same=0, png_diff=0, key_mismatch=0, missing=[])
    z1 = dict(scenes=0, config_ok=0, kept_mismatch=0, emptied=0, keyset_lost=0, counter_ok=0, removed_same=0, removed_on=0,
              removed_off=0, flips_on=0, flips_off=0, added_px=0, object_views=0, bad=[])
    for sc in SCENES:
        a, b = _tree_bytes(_result(sc, ARM_R1)), _tree_bytes(_result(sc, C0_R1))
        if not a:
            z2["missing"].append(sc)
        else:
            same = sum(a[k] == b.get(k) for k in a)
            z2["png_diff"] += len(a) - same
            z2["key_mismatch"] += int(set(a) != set(b))
            z2["scenes_ok"] += int(set(a) == set(b) and same == len(a))
            ca, cb = os.path.join(_result(sc, ARM), "round1_conf.json"), os.path.join(_result(sc, C0), "round1_conf.json")
            z2["conf_same"] += int(os.path.isfile(ca) and os.path.isfile(cb) and _bytes(ca) == _bytes(cb))
        d, d0 = _result(sc, ARM), _result(sc, C0)
        if not os.path.isfile(os.path.join(d, "xcl_conf.json")):
            z1["bad"].append(("missing", sc)); continue
        z1["scenes"] += 1
        cfg = _load_json(os.path.join(d, "RUN.json")).get("config", {})
        z1["config_ok"] += int(cfg.get("xcl") == "outs" and cfg.get("xcl_shift") == SHIFT and cfg.get("xcl_tau_keep") == TAU_KEEP)
        ra, rb = _load_json(os.path.join(d, "xcl_conf.json")), _load_json(os.path.join(d0, "xcl_conf.json"))
        n_ex = sum(bool(r["exempt"]) for v in ra.values() for r in v.values())
        n_em = sum(1 for v in ra.values() for r in v.values() if r["px"] > 0 and r["kept"] == 0)
        z1["counter_ok"] += int(cfg.get("n_xcl_masks_exempt") == n_ex and cfg.get("n_xcl_masks_emptied") == n_em == 0)
        z1["emptied"] += n_em
        for cam, recs in ra.items():
            for o, r in recs.items():
                pa, pb = os.path.join(d, cam, f"{o}.png"), os.path.join(d0, cam, f"{o}.png")
                ma = read_pred_mask(pa) if os.path.isfile(pa) else None
                mb = read_pred_mask(pb) if os.path.isfile(pb) else None
                na, nb = (0 if ma is None else int(ma.sum())), (0 if mb is None else int(mb.sum()))
                z1["object_views"] += 1
                if r["kept"] != na:
                    z1["kept_mismatch"] += 1
                if nb > 0 and na == 0:
                    z1["keyset_lost"] += 1
                if mb is None:
                    z1["added_px"] += na; continue
                ma = ma if ma is not None else np.zeros_like(mb)
                rem = int((mb & ~ma).sum())
                k = object_class(rb[cam][o], r)
                z1[f"removed_{k}"] += rem
                if k != "same":
                    z1[f"flips_{k}"] += 1
                z1["added_px"] += int((ma & ~mb).sum())
    z2["ok"] = z2["scenes_ok"] == 17 and z2["conf_same"] == 17 and z2["png_diff"] == 0 and not z2["missing"]
    z1["ok"] = z1["scenes"] == 17 and z1["config_ok"] == 17 and z1["counter_ok"] == 17 and z1["kept_mismatch"] == 0 \
        and z1["emptied"] == 0 and z1["keyset_lost"] == 0 and z1["removed_same"] == 0 and z1["removed_on"] == 0
    return z2, z1


def stage1(audited=False):
    sys.path.insert(0, S1)
    from mvseg2d.paired import paired_scene_bootstrap, scene_means
    from mvseg2d.report import flatten
    out_all = _load_json(OUT) if os.path.isfile(OUT) else {}
    if not out_all.get("stageB", {}).get("rule", {}).get("pass"):
        raise SystemExit("the DEV-B rule did not pass (or was not judged): no keyframe run (section 4)")

    def rows_of(name):
        rs = flatten(_load_json(os.path.join(OUT_DIR, f"kf_{name}.json")))
        for r in rs:
            r["JF"] = 0.5 * (r["J"] + r["F"])
        return rs

    def hallu(name):
        return sum(s["hallucinations"] for s in _load_json(os.path.join(OUT_DIR, f"kf_{name}.json"))["scenes"].values() if s["status"] == "ok")

    key = lambda r: (r["scene"], r["cam"], r["obj"])
    r0, ra = rows_of(C0), rows_of(ARM)
    k0, ka = {key(r): r for r in r0}, {key(r): r for r in ra}
    if set(k0) != set(ka) or len(k0) != 664:
        raise SystemExit(f"keyframe pair sets differ or are incomplete ({len(k0)} vs {len(ka)}) -- fix the decode first")
    st = {k: paired_scene_bootstrap(r0, ra, key=k) for k in ("J", "F", "JF")}
    gate = keyframe_gate(st["JF"])
    audit = audit_needed(st["JF"]["delta"], AUDIT_KF)
    d = {k: ka[k]["JF"] - k0[k]["JF"] for k in k0}
    good = {k for k, r in k0.items() if r["JF"] >= 0.8}

    def split(sel):
        sel = list(sel)
        return dict(n=len(sel), dJF=statistics.mean(d[k] for k in sel) if sel else None,
                    up=sum(d[k] > 0.01 for k in sel), down=sum(d[k] < -0.01 for k in sel))
    splits = dict(good=split(good), fail=split(k for k in k0 if k not in good),
                  **{f"good_{n}": split(k for k in good if lo <= k0[k]["gt_px"] < hi) for n, lo, hi in SIZE_BINS})
    good_rows = [k0[k] for k in sorted(good)]
    gb = paired_scene_bootstrap(good_rows, [ka[key(r)] for r in good_rows], key="JF")
    kinds = None
    sf0, sfa = (os.path.join(OUT_DIR, f"sf_{n}.json") for n in (C0, ARM))
    if os.path.isfile(sf0) and os.path.isfile(sfa):
        kind0 = {(r["scene"], r["cam"], int(r["obj"])): r for r in _load_json(sf0) if (r["scene"], r["cam"], int(r["obj"])) in k0}
        kinda = {(r["scene"], r["cam"], int(r["obj"])): r for r in _load_json(sfa) if (r["scene"], r["cam"], int(r["obj"])) in k0}
        fail0 = {k: seed_kind(r) for k, r in kind0.items() if r["JF"] < 0.8}
        kinds = dict(C0=dict(Counter(fail0.values())),
                     arm=dict(Counter(seed_kind(r) for r in kinda.values() if r["JF"] < 0.8)),
                     dJF_by_C0_kind={kd: statistics.mean(d[k] for k, v in fail0.items() if v == kd) for kd in set(fail0.values())})
    z2, z1 = kf_checks()
    summ = lambda rs: {m: statistics.mean(scene_means(rs, m).values()) for m in ("J", "F", "JF")}
    res = dict(C0=dict(name=C0, **summ(r0), fails=sum(r["J"] < 0.5 for r in r0), hallucinations=hallu(C0)),
               arm=dict(name=ARM, tag=TAG, **summ(ra), fails=sum(r["J"] < 0.5 for r in ra), hallucinations=hallu(ARM)),
               paired={m: dict(delta=v["delta"], ci=list(v["ci"]), per_scene=v["per_scene"]) for m, v in st.items()},
               good_hier=dict(delta=gb["delta"], ci=list(gb["ci"])), gate=gate, audit_required=audit, audited=bool(audited),
               splits=splits, failure_kinds=kinds, Z1=z1, Z2=z2)
    dj = st["JF"]["delta"]
    res["predictions"] = dict(
        P3=dict(delta=dj, in_80=0.0010 <= dj <= 0.0045, dJ_gt_dF=st["J"]["delta"] > st["F"]["delta"],
                fails=res["arm"]["fails"], fails_ok=res["arm"]["fails"] in (13, 14), halluc=res["arm"]["hallucinations"],
                halluc_ok=res["arm"]["hallucinations"] == res["C0"]["hallucinations"], round1_identical=z2["ok"],
                good_bins_positive=all((splits[f"good_{n}"]["dJF"] or 0) > 0 for n, _, _ in SIZE_BINS if splits[f"good_{n}"]["n"]),
                dev_b_delta=out_all["stageB"]["contrast"]["compare"]["paired"]["JF"]["delta"]),
        P6=dict(band_oracle_closed=gb["delta"] / BAND_ORACLE_A, ok=gb["delta"] / BAND_ORACLE_A <= 0.06))
    passing = [TAG] if gate["pass"] and not audit and z1["ok"] and z2["ok"] else []
    if audit and audited and gate["pass"] and z1["ok"] and z2["ok"]:
        passing = [TAG]
    res["passing_tags"] = passing
    _save("stage1", res)
    if audit and not audited:
        print(f"STOP AND AUDIT (section 5.1): keyframe dJ&F {dj:+.4f} > +{AUDIT_KF} -- check for a leak or bug before stage 2; "
              f"rerun with --audited once the audit finds none")
        sys.exit(4)
    p = res["paired"]
    print(f"keyframe 664 pairs: C0 J&F {res['C0']['JF']:.4f} -> arm {res['arm']['JF']:.4f}  dJ&F {dj:+.4f} "
          f"[{p['JF']['ci'][0]:+.4f}, {p['JF']['ci'][1]:+.4f}]  dJ {p['J']['delta']:+.4f}  dF {p['F']['delta']:+.4f}  worst "
          f"{gate['worst_scene']} {gate['worst_delta']:+.4f}  fails {res['C0']['fails']}->{res['arm']['fails']}  halluc "
          f"{res['C0']['hallucinations']}->{res['arm']['hallucinations']}")
    print(f"   good {gb['delta']:+.4f} [{gb['ci'][0]:+.4f}, {gb['ci'][1]:+.4f}]  splits {json.dumps(splits)}")
    print("   per scene " + "  ".join(f"{s} {v:+.4f}" for s, v in sorted(p["JF"]["per_scene"].items(), key=lambda x: x[1])))
    print(f"   kinds {kinds}\n   Z1 {z1}\n   Z2 {z2}\n   predictions {json.dumps(res['predictions'])}")
    print(f"keyframe gate: G1 {gate['G1']}  G2 {gate['G2']}  (Z1 {z1['ok']}, Z2 {z2['ok']}) -> passing {passing}")
    if not (z1["ok"] and z2["ok"]):
        sys.exit(2)


# ----------------------------------------------------------------------------------------------- stage 2
def stage2():
    sys.path.insert(0, os.path.join(REPO, "MVSeed"))
    import numpy as np
    import e7_judge as E7
    from e9_judge import z3_digest
    out = _load_json(OUT) if os.path.isfile(OUT) else {}
    s1 = out.get("stage1")
    if s1 is None or "passing_tags" not in s1:
        raise SystemExit("keyframe judgement missing (run --stage 1 first)")
    if not s1["passing_tags"]:
        _save("stage2", dict(note="the keyframe gate did not pass: no stage 2 (section 5.1)", adopted=None))
        print("stage-2: the keyframe gate did not pass; nothing to judge"); return
    d = E7.load(["jf_e14.json"])
    Rm, Rb = list(E7.rows(d, METHOD2)), list(E7.rows(d, HEADLINE))
    if not Rm or not Rb:
        raise SystemExit(f"no rows for {METHOD2} or {HEADLINE} (not scored yet?)")
    sm, sb = E7.scene_scores(Rm), E7.scene_scores(Rb)
    cm, cb = E7.scene_scores(list(E7.rows(d, METHOD2, "complete"))), E7.scene_scores(list(E7.rows(d, HEADLINE, "complete")))
    st, stc = E7.paired(sb, sm), E7.paired(cb, cm)
    worst = min(st["per_scene"].values())
    rule = dict(mean_gt_0=st["mean"] > 0, ci_lower_gt_0=st["ci"][0] > 0, defence=worst >= DEFENCE, worst=worst, complete_ge_0=stc["mean"] >= 0)
    rule["pass"] = all(rule[k] for k in ("mean_gt_0", "ci_lower_gt_0", "defence", "complete_ge_0"))
    z3 = z3_digest("geoset_pred_ft", METHOD2, result=ARM)
    z3b = dict(same=0, different=0, missing=0)
    from PIL import Image
    for r in Rm:
        if r["ref"]:
            continue
        c = E7.CFG[r["scene"]]; f, s0 = c["folder"], c["start_frame"]
        a = os.path.join(_result(r["scene"], ARM), r["cam"], f"{r['obj']}.png")
        b = os.path.join(E7.DATA, f, METHOD2, r["cam"], str(s0), f"{r['obj']}.png")
        if not os.path.exists(a) and not os.path.exists(b):
            continue
        if not os.path.exists(a) or not os.path.exists(b):
            z3b["missing"] += 1; continue
        ma = np.asarray(Image.open(a).convert("L")) > 127; mb = np.asarray(Image.open(b).convert("L")) > 127
        z3b["same" if ma.shape == mb.shape and bool((ma == mb).all()) else "different"] += 1
    ref0 = [E7.jf(r, 0) for r in Rm if r["ref"] and r["gt"][0]]
    kf = s1["paired"]["JF"]["delta"]
    ratio = st["mean"] / (66.7 * kf) if kf else None
    run = dict(method=METHOD2, tag=TAG, base=HEADLINE, basic=statistics.mean(sm.values()), base_basic=statistics.mean(sb.values()),
               paired=st, complete=statistics.mean(cm.values()), base_complete=statistics.mean(cb.values()), paired_complete=stc,
               rule=rule, Z3_mask_digest=z3, Z3b_frame0_pixels=z3b, ref_frame0_is_gt=all(x is not None and x > 0.999 for x in ref0))
    P = dict(P4=dict(ratio=ratio, basic_delta=st["mean"], ok=ratio is not None and 0.3 <= ratio <= 2.0 and stc["mean"] >= 0 and worst > DEFENCE),
             Z3=isinstance(z3, dict) and z3["bad"] == [] and z3["ok"] == 17,
             Z3b=z3b["different"] == 0 and z3b["missing"] == 0)
    _save("stage2", dict(run=run, adopted=TAG if rule["pass"] else None, predictions=P))
    print(f"{METHOD2}\n   basic {run['basic']:.3f} vs {run['base_basic']:.3f}  Δ {st['mean']:+.3f}  CI [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]  "
          f"{st['wins']}↑/{st['losses']}↓  worst {st['worst'][1]} {st['worst'][0]:+.2f}\n   complete {run['complete']:.3f} vs "
          f"{run['base_complete']:.3f}  Δ {stc['mean']:+.3f}  rule {rule}\n   Z3 {z3}  Z3b {z3b}  ref frame 0 = GT {run['ref_frame0_is_gt']}")
    print(f"\nstage-2: {'ADOPTED ' + TAG if rule['pass'] else 'not adopted'}   predictions {P}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("A", "B", "1", "2"), required=True)
    ap.add_argument("--audited", action="store_true", help="the stop-and-audit check found no bug: print and judge as registered")
    a = ap.parse_args()
    {"A": stage_a, "B": lambda: stage_b(a.audited), "1": lambda: stage1(a.audited), "2": stage2}[a.stage]()
