#!/usr/bin/env python3
"""S2-R5 judge (docs/stage2-R5-prereg.md sections 3-4): the E8 headline and control with the stage-1
fine-tuned memory attention (H-FT, C-FT) against their E8 bases on the MUVOD protocol.  Checks the
watch values Z1-Z4 first (no verdict when one fails), then the registered rule on H-FT (adopt /
follow-up open / close), the per-group contribution split (reference camera, F0, N0, other: sums to
the paired basic delta), frame tables, predictions P1-P4.  Host python (json, hashlib).

    python3 eval/eval_jf.py --methods <the two folders> --out jf_s2r5.json   # in the container
    python3 MVSeed/s2r5_judge.py      ->  MVSeed/runs/s2r5_judge.json
"""
import hashlib, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_judge as E7                     # load / rows / jf / scene_scores / paired: the E7 aggregation, unchanged
from e7_contrib import pair_scores

REPO, DATA, CFG = E7.REPO, E7.DATA, E7.CFG
SCENES = ["AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "CBABasketball", "Carpark",
          "CoffeeMartini", "Dog", "Fencing", "FlameSteak", "Frog", "MATF", "MartialArts", "Painter", "PoznanStreet", "Welder"]
RUNS = [("H", "SegMaskSam3XW1CGPS4MSdgeoset_repFsFtgpCs10kG2", "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2"),
        ("C", "SegMaskSam3XW0MSdgeoset_repFsFtgp", "SegMaskSam3XW0MSdgeoset_repFs")]
REGISTERED_SHA256 = {0: "03f5d407b8bed936617395d1730db266c15a16f27f7fe7cdbd8cae5e153ea498",
                     1: "8e043c4d48327ec2bbf69414bb06ad3e70e84094ccb9f96085e27323542a2518",
                     2: "5b9ef4885d9e1f09caa88b5622c7b4442b57a11352447a5a26d8e7a8a0b7fc2f",
                     3: "9b5e2b3010c4440ca8dda6621abd4231a457cdad3b6c1504f88b3167b663649d",
                     4: "05402ec462fb0d90a0801c1b440e420639bdedf72832c2bfc6186b2db60b77af"}
REGISTERED_FOLDS_SHA256 = "ab12356d9aaeef823c5100e90b6692ae6b486ca8de665ca461484210dfb0c0d7"
REGISTERED_FOLD = {"AlexaMeadeExhibit": 0, "Blocks": 0, "Fencing": 0, "Painter": 0,                       # prereg section 1
                   "AlexaMeadeFacePaint": 1, "Breakfast": 1, "FlameSteak": 1, "PoznanStreet": 1,
                   "CBABasketball": 2, "Dog": 2, "Frog": 2, "Carpark": 3, "MATF": 3, "Welder": 3,
                   "Barn": 4, "CoffeeMartini": 4, "MartialArts": 4}
ARM = "geoset_pred_ft"
ADOPT_MIN, ADOPT_WORST, FOLLOW_REF = 0.3, -0.3, -0.05      # prereg section 3
FRAMES = (0, 1, 5, 10, 20)
GROUPS = ("ref", "F0", "N0", "other_nonref")


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def manifest(scene, method):
    p = os.path.join(DATA, CFG[scene]["folder"], method, "MANIFEST.json")
    return json.load(open(p)).get("provenance", {}) if os.path.exists(p) else None


def z1_frame0(scene, m, base):
    """Every frame-0 PNG of every written camera: same file set, same bytes (Fs -> the seed)."""
    f, s0 = CFG[scene]["folder"], CFG[scene]["start_frame"]
    same = diff = only_base = only_run = 0
    bdir, mdir = os.path.join(DATA, f, base), os.path.join(DATA, f, m)
    for cam in sorted(set(os.listdir(bdir)) | set(os.listdir(mdir))):
        a, b = os.path.join(bdir, cam, str(s0)), os.path.join(mdir, cam, str(s0))
        if not (os.path.isdir(a) or os.path.isdir(b)): continue
        fa = set(os.listdir(a)) if os.path.isdir(a) else set(); fb = set(os.listdir(b)) if os.path.isdir(b) else set()
        only_base += len(fa - fb); only_run += len(fb - fa)
        for n in fa & fb:
            if sha(os.path.join(a, n)) == sha(os.path.join(b, n)): same += 1
            else: diff += 1
    return dict(same=same, different=diff, only_base=only_base, only_run=only_run)


def z2_crop(scene, m, base):
    """H-FT's crop pass ran on the restored original weights: same targets, verdicts, iou_f1, and the
    kept pairs' frames 1..20 PNGs byte-identical to the E8 base."""
    pm, pb = manifest(scene, m), manifest(scene, base)
    if pm is None or pb is None: return dict(ok=False, why="manifest absent")
    cm, cb = pm.get("crop_small"), pb.get("crop_small")
    key = lambda t: (t["cam"], t["obj"], t["kept"], t["why"], t["iou_f1_seed"], t["crop"], t["side"])
    tm, tb = sorted(map(key, cm["targets"])), sorted(map(key, cb["targets"]))
    f, s0, n = CFG[scene]["folder"], CFG[scene]["start_frame"], CFG[scene]["num_frame"]
    same = diff = missing = 0
    for t in cb["targets"]:
        if not t["kept"]: continue
        for k in range(1, n):
            a = os.path.join(DATA, f, base, t["cam"], str(s0 + k), f"{t['obj']}.png")
            b = os.path.join(DATA, f, m, t["cam"], str(s0 + k), f"{t['obj']}.png")
            if not (os.path.exists(a) and os.path.exists(b)): missing += 1
            elif sha(a) == sha(b): same += 1
            else: diff += 1
    ok = tm == tb and cm["n_kept"] == cb["n_kept"] and diff == 0 and missing == 0
    return dict(ok=ok, targets=cm["n_targets"], kept=cm["n_kept"], base_targets=cb["n_targets"], base_kept=cb["n_kept"],
                same_target_records=tm == tb, kept_png_same=same, kept_png_different=diff, kept_png_missing=missing)


def z3_ft(scene, m, headline):
    """The manifest's ft_encoder block against the registration (prereg section 1): arm, fold table,
    checkpoint and fold-table hashes, no leak, parameter digests and the functional probe (the
    encoder's outputs through the live autocast cache changed after the load, and came back after
    the restore)."""
    p = manifest(scene, m)
    ft = (p or {}).get("ft_encoder")
    if not ft: return dict(ok=False, why="no ft_encoder block")
    checks = dict(arm=ft["arm"] == ARM, fold_registered=ft["fold"] == REGISTERED_FOLD[scene],
                  sha_registered=ft["checkpoint_sha256"] == REGISTERED_SHA256.get(ft["fold"]),
                  folds_sha_registered=ft["folds_sha256"] == REGISTERED_FOLDS_SHA256,
                  no_leak=scene not in ft["train_scenes"],
                  loaded_is_checkpoint=ft["digest_loaded"] == ft["digest_checkpoint"],
                  loaded_differs=ft["digest_loaded"] != ft["digest_original"],
                  probe_changed=ft.get("probe_loaded") is not None and ft.get("probe_loaded") != ft.get("probe_original"))
    if headline:
        checks["restored"] = (bool(ft["restored_before_crop"]) and ft["digest_restored"] == ft["digest_original"]
                              and ft.get("probe_restored") == ft.get("probe_original"))
    return dict(ok=all(checks.values()), **checks, fold=ft["fold"])


def z4_changed(scene, m, base, headline):
    """Frames 1..20 PNGs outside the crop-kept pairs that differ from the base: a swap that never
    reached the forwards would leave every one of them identical (Δ = 0 and a spurious verdict)."""
    f, s0, n = CFG[scene]["folder"], CFG[scene]["start_frame"], CFG[scene]["num_frame"]
    kept = set()
    if headline:
        pb = manifest(scene, base) or {}
        kept = {(t["cam"], int(t["obj"])) for t in (pb.get("crop_small") or {}).get("targets", []) if t["kept"]}
    same = diff = one_side = 0
    bdir, mdir = os.path.join(DATA, f, base), os.path.join(DATA, f, m)
    for cam in sorted(set(os.listdir(bdir)) | set(os.listdir(mdir))):
        for k in range(1, n):
            a, b = os.path.join(bdir, cam, str(s0 + k)), os.path.join(mdir, cam, str(s0 + k))
            if not (os.path.isdir(a) or os.path.isdir(b)): continue
            fa = set(os.listdir(a)) if os.path.isdir(a) else set(); fb = set(os.listdir(b)) if os.path.isdir(b) else set()
            for name in fa | fb:
                if (cam, int(name.split(".")[0])) in kept: continue
                if name not in fa or name not in fb: one_side += 1
                elif sha(os.path.join(a, name)) == sha(os.path.join(b, name)): same += 1
                else: diff += 1
    return dict(same=same, different=diff, one_side=one_side)


def frame_table(R):
    out = {}
    for grp, sel in (("ref", lambda r: r["ref"]), ("nonref", lambda r: not r["ref"])):
        out[grp] = {t: (statistics.mean(v) if (v := [x for r in R if sel(r) and len(r["J"]) > t and (x := E7.jf(r, t)) is not None]) else None)
                    for t in FRAMES}
    return out


def main():
    d = E7.load(["jf_s2r5.json", "jf_e8.json"])
    F0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed", "runs", "F0.json")))["pairs"]}
    N0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed", "runs", "N0.json")))["pairs"]}
    out = {"prereg": "docs/stage2-R5-prereg.md", "runs": {}}
    for rid, m, base in RUNS:
        Rm, Rb = list(E7.rows(d, m)), list(E7.rows(d, base))
        if not Rm: print(f"{rid}: no rows for {m} (not scored yet?)"); continue
        sm, sb = E7.scene_scores(Rm), E7.scene_scores(Rb)
        cm, cb = E7.scene_scores(list(E7.rows(d, m, "complete"))), E7.scene_scores(list(E7.rows(d, base, "complete")))
        st, stc = E7.paired(sb, sm), E7.paired(cb, cm)
        res = dict(method=m, base=base, n_scenes=len(sm), basic=statistics.mean(sm.values()), base_basic=statistics.mean(sb.values()),
                   paired=st, complete=statistics.mean(cm.values()), base_complete=statistics.mean(cb.values()), paired_complete=stc)
        # contribution split (as e8_contrib.py): benchmark weight x (run - base) pair score, summed per group
        ref = {E7.pair_key(r) for r in Rm if r["ref"]}
        pm, pb = pair_scores(Rm), pair_scores(Rb)
        assert set(pm) == set(pb), (rid, len(set(pm) ^ set(pb)))
        contrib, counts = {}, {}
        for k in pm:
            g = "ref" if k in ref else "F0" if k in F0 else "N0" if k in N0 else "other_nonref"
            contrib[g] = contrib.get(g, 0.0) + pm[k][1] * (pm[k][0] - pb[k][0]); counts[g] = counts.get(g, 0) + 1
        res["contrib"] = {g: contrib.get(g, 0.0) for g in GROUPS}; res["contrib_n"] = counts      # raw: the rule reads it
        res["contrib_sum_minus_delta"] = round(sum(contrib.values()) - st["mean"], 6)
        # frames 1-20 pair means, F0 / N0 recovered / broken (as e8_judge.py)
        p1 = lambda R: {E7.pair_key(r): statistics.mean(v) for r in R if (v := [x for x in (E7.jf(r, t) for t in range(1, len(r["J"]))) if x is not None])}
        qm, qb = p1(Rm), p1(Rb); split = {}
        for name, pop in (("ref", ref), ("F0", F0), ("N0", N0)):
            ks = [k for k in pop if k in qm and k in qb]; dd = [qm[k] - qb[k] for k in ks]
            split[name] = dict(n=len(ks), mean_dJF=statistics.mean(dd) if dd else None,
                               recovered=sum(1 for k in ks if qb[k] < 0.5 <= qm[k]), broken=sum(1 for k in ks if qm[k] < 0.5 <= qb[k]),
                               up=sum(1 for x in dd if x > 0.05), down=sum(1 for x in dd if x < -0.05))
        res["pairs_1_20"] = split
        res["frames"] = dict(run=frame_table(Rm), base=frame_table(Rb))
        # watch values
        res["Z1_frame0"] = {sc: z1_frame0(sc, m, base) for sc in SCENES if sc in sm}
        res["Z3_ft"] = {sc: z3_ft(sc, m, rid == "H") for sc in SCENES if sc in sm}
        res["Z4_changed"] = {sc: z4_changed(sc, m, base, rid == "H") for sc in SCENES if sc in sm}
        z1 = all(v["different"] == 0 and v["only_base"] == 0 and v["only_run"] == 0 for v in res["Z1_frame0"].values())
        z3 = all(v["ok"] for v in res["Z3_ft"].values())
        z4 = sum(v["different"] + v["one_side"] for v in res["Z4_changed"].values()) > 0
        res["Z_pass"] = dict(Z1=z1, Z3=z3, Z4=z4)
        if rid == "H":
            res["Z2_crop"] = {sc: z2_crop(sc, m, base) for sc in SCENES if sc in sm}
            res["Z_pass"]["Z2"] = all(v["ok"] for v in res["Z2_crop"].values())
        out["runs"][rid] = res
        print(f"{rid} {m}\n   basic {res['basic']:.3f} vs base {res['base_basic']:.3f}  Δ {st['mean']:+.3f}  CI [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]"
              f"  {st['wins']}↑/{st['losses']}↓  worst {st['worst'][1]} {st['worst'][0]:+.2f}  best {st['best'][1]} {st['best'][0]:+.2f}")
        print(f"   complete {res['complete']:.3f} vs base {res['base_complete']:.3f}  Δ {stc['mean']:+.3f}  CI [{stc['ci'][0]:+.3f}, {stc['ci'][1]:+.3f}]")
        print(f"   contrib {{{', '.join(f'{g}: {v:+.4f}' for g, v in res['contrib'].items())}}}  (sum − Δ {res['contrib_sum_minus_delta']:+.6f})")
        print(f"   Z4 changed PNGs (frames 1-20, crop-kept excluded): {sum(v['different'] for v in res['Z4_changed'].values())} differ, "
              f"{sum(v['same'] for v in res['Z4_changed'].values())} same, {sum(v['one_side'] for v in res['Z4_changed'].values())} one side")
        for k, v in split.items(): print(f"   {k}: n={v['n']} ΔJ&F(1-20) {v['mean_dJF']:+.4f}  recovered {v['recovered']} broken {v['broken']}  up {v['up']} down {v['down']}")
        for g in ("ref", "nonref"):
            print(f"   frames {g}: run " + " ".join(f"{t}:{res['frames']['run'][g][t]:.4f}" for t in FRAMES)
                  + " | base " + " ".join(f"{t}:{res['frames']['base'][g][t]:.4f}" for t in FRAMES))
        print(f"   Z {res['Z_pass']}")

    R = out["runs"]
    verdict = None
    if "H" in R:
        h = R["H"]; st = h["paired"]
        complete = "C" in R and all(R[k]["n_scenes"] == len(SCENES) for k in ("H", "C"))
        zok = complete and all(h["Z_pass"].values()) and all(R["C"]["Z_pass"].values())
        c_ref = R["C"]["contrib"]["ref"] if "C" in R else None
        rule = dict(delta=st["mean"], ci_lower=st["ci"][0], worst=st["worst"][0], complete_delta=h["paired_complete"]["mean"],
                    ref_contrib=h["contrib"]["ref"], ref_contrib_control=c_ref)
        rule["adopt"] = st["mean"] >= ADOPT_MIN and st["ci"][0] > 0 and st["worst"][0] >= ADOPT_WORST and h["paired_complete"]["mean"] >= 0
        rule["follow_up"] = ((not rule["adopt"]) and st["mean"] >= 0 and h["contrib"]["ref"] >= FOLLOW_REF
                             and c_ref is not None and c_ref >= FOLLOW_REF)
        verdict = ("no verdict: both runs over all 17 scenes are required" if not complete else
                   "no verdict: a watch value failed" if not zok else
                   "adopt" if rule["adopt"] else "follow-up open" if rule["follow_up"] else "close")
        out["rule"] = rule; out["verdict"] = verdict
        pr = dict(P1=-1.0 <= st["mean"] < 0.3, P2=-0.4 <= h["contrib"]["ref"] <= 0.1)
        if "C" in R:
            c = R["C"]["paired"]["mean"]
            pr["P3"] = (c >= 0) == (st["mean"] >= 0) and abs(c - st["mean"]) <= 0.4
            pr["P4"] = zok
            out["H_minus_C"] = st["mean"] - c
        out["predictions"] = pr
        print(f"\nrule {rule}\nverdict: {verdict}\npredictions {pr}")
    json.dump(out, open(os.path.join(REPO, "MVSeed", "runs", "s2r5_judge.json"), "w"), indent=1)
    print("wrote MVSeed/runs/s2r5_judge.json")


if __name__ == "__main__":
    main()
