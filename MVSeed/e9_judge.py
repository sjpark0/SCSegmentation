#!/usr/bin/env python3
"""S1-E9 judge (docs/stage1-E9-prereg.md sections 3-7).

  --stage 1   keyframe scores (stage1/mvseg2d.evaluate JSONs in MVSeed/runs/e9/kf_<name>.json): D1, D2 against D0,
              scene-cluster bootstrap on J, F, J&F (mvseg2d.paired, 10 000, seed 0), gate G1/G2, predictions P1-P4,
              good/failing-seed split, watch value Z1 (trainable parameters).  Run with stage1/.venv (numpy):
                  stage1/.venv/bin/python MVSeed/e9_judge.py --stage 1
  --stage 2   MUVOD basic/complete of the passing arms' in-process headline runs (Data/MVSeg/jf_e9.json, scored by
              eval/eval_jf.py) against the current headline, the E8 adoption rule, selection, P5-P6, Z3.  Plain python3.
Both write into MVSeed/runs/e9/e9_judge.json (stage 2 merges into stage 1's file).
"""
import argparse, json, os, statistics, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "MVSeed", "runs", "e9")
OUT = os.path.join(OUT_DIR, "e9_judge.json")
D0 = "geoset_pred_ft_rep_e8"
ARMS = [("D1", "geoset_dec_ft", "geoset_dec_rep"), ("D2", "geoset_decb_ft", "geoset_decb_rep")]
BASE_PARAMS = 5_922_321                     # geoset_pred_ft trainable parameters
DECODER_PARAMS = 4_190_437                  # sam_mask_decoder without conv_s0 / conv_s1 (Z1 = BASE + this = 10 112 758)
HEADLINE = "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs"
DEFENCE, TIE = -1.0, 0.05
SAFETY = -0.01                              # G2: no scene below D0 - 0.01 (keyframe J&F)


def _load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- stage 1
def stage1():
    sys.path.insert(0, os.path.join(REPO, "stage1"))
    from mvseg2d.paired import paired_scene_bootstrap, scene_means
    from mvseg2d.report import flatten

    def rows_of(name):
        rs = flatten(_load_json(os.path.join(OUT_DIR, f"kf_{name}.json")))
        for r in rs:
            r["JF"] = 0.5 * (r["J"] + r["F"])
        return rs

    def hallu(name):
        return sum(s["hallucinations"] for s in _load_json(os.path.join(OUT_DIR, f"kf_{name}.json"))["scenes"].values() if s["status"] == "ok")

    def summary(rs):
        m = {k: scene_means(rs, k) for k in ("J", "F", "JF")}
        return {k: statistics.mean(v.values()) for k, v in m.items()}

    key = lambda r: (r["scene"], r["cam"], r["obj"])
    r0 = rows_of(D0); k0 = {key(r): r for r in r0}
    good = {k for k, r in k0.items() if r["JF"] >= 0.8}
    res = {"D0": dict(name=D0, n=len(r0), **summary(r0), hallucinations=hallu(D0), fails=sum(r["J"] < 0.5 for r in r0))}
    passing = []
    for rid, arm, tag in ARMS:
        name = f"{arm}_rep_e9"
        p = os.path.join(OUT_DIR, f"kf_{name}.json")
        if not os.path.isfile(p):
            print(f"{rid}: {p} missing (not decoded / scored yet)"); continue
        rs = rows_of(name); kk = {key(r): r for r in rs}
        if set(kk) != set(k0):
            st_ = {sc: v["status"] for sc, v in _load_json(p)["scenes"].items() if v["status"] != "ok"}
            raise SystemExit(f"{rid}: pair set differs from D0 ({len(kk)} vs {len(k0)}); scenes not ok: {st_} -- fix the decode first")
        st = {k: paired_scene_bootstrap(r0, rs, key=k) for k in ("J", "F", "JF")}
        worst = min(st["JF"]["per_scene"].items(), key=lambda x: x[1])
        gate = dict(G1_mean_gt_0=st["JF"]["delta"] > 0, G1_ci_lower_gt_0=st["JF"]["ci"][0] > 0,
                    G2_no_scene_below=worst[1] >= SAFETY, worst_scene=worst[0], worst_delta=worst[1])
        gate["pass"] = gate["G1_mean_gt_0"] and gate["G1_ci_lower_gt_0"] and gate["G2_no_scene_below"]
        d_good = [kk[k]["JF"] - k0[k]["JF"] for k in good]
        d_fail = [kk[k]["JF"] - k0[k]["JF"] for k in k0 if k not in good]
        by_size = {}
        for b in ("<2000", "2000-10000", ">=10000"):
            ks = [k for k, r in k0.items() if r["size_bin"] == b]
            by_size[b] = dict(n=len(ks), dJF=statistics.mean(kk[k]["JF"] - k0[k]["JF"] for k in ks) if ks else None)
        cfg_params = None
        cfg = os.path.join(REPO, "stage1", "geomem", "runs", arm, "fold0", "config.json")
        if os.path.isfile(cfg):
            c = _load_json(cfg); cfg_params = dict(trainable_params=c.get("trainable_params"), conv_s_in_names=any("conv_s" in n for n in c.get("trainable_names", [])),
                                                   seconds=c.get("seconds"), steps=c.get("steps"))
        r = dict(name=name, tag=tag, **summary(rs), hallucinations=hallu(name), fails=sum(x["J"] < 0.5 for x in rs),
                 paired={k: dict(delta=v["delta"], ci=list(v["ci"]), per_scene=v["per_scene"]) for k, v in st.items()},
                 gate=gate, good_seeds=dict(n=len(good), dJF=statistics.mean(d_good), up=sum(x > 0.01 for x in d_good), down=sum(x < -0.01 for x in d_good)),
                 fail_seeds=dict(n=len(d_fail), dJF=statistics.mean(d_fail), up=sum(x > 0.01 for x in d_fail), down=sum(x < -0.01 for x in d_fail)),
                 by_size=by_size, fold0_config=cfg_params)
        res[rid] = r
        if gate["pass"]:
            passing.append(tag)
        print(f"{rid} {name}: J {r['J']:.4f} F {r['F']:.4f} J&F {r['JF']:.4f} (D0 {res['D0']['JF']:.4f})  "
              f"ΔJ&F {st['JF']['delta']:+.4f} CI [{st['JF']['ci'][0]:+.4f}, {st['JF']['ci'][1]:+.4f}]  ΔJ {st['J']['delta']:+.4f} [{st['J']['ci'][0]:+.4f}, {st['J']['ci'][1]:+.4f}]"
              f"  ΔF {st['F']['delta']:+.4f}  worst {worst[0]} {worst[1]:+.4f}  fails {res['D0']['fails']}→{r['fails']}  halluc {res['D0']['hallucinations']}→{r['hallucinations']}")
        print(f"   good seeds (≥0.8, n={len(good)}) ΔJ&F {r['good_seeds']['dJF']:+.4f} up {r['good_seeds']['up']} down {r['good_seeds']['down']}"
              f"  | failing (n={len(d_fail)}) ΔJ&F {r['fail_seeds']['dJF']:+.4f}  | by size {{{', '.join(f'{b}: {v['dJF']:+.4f}' for b, v in by_size.items())}}}"
              f"  | fold0 {cfg_params}   gate {'PASS' if gate['pass'] else 'FAIL'} {gate}")
    # predictions P1-P4 (section 4)
    pr = {}
    if "D1" in res:
        pr["P1"] = 0.005 <= res["D1"]["paired"]["JF"]["delta"] <= 0.025
    if "D1" in res and "D2" in res:
        pr["P2"] = res["D2"]["paired"]["J"]["delta"] >= res["D1"]["paired"]["J"]["delta"] and \
            all(abs(res[r]["paired"]["F"]["delta"]) < abs(res[r]["paired"]["J"]["delta"]) for r in ("D1", "D2"))
    for r in ("D1", "D2"):
        if r in res:
            pr[f"P3_{r}"] = abs(res[r]["fail_seeds"]["dJF"]) < 0.02 and res[r]["good_seeds"]["dJF"] >= res[r]["fail_seeds"]["dJF"]
            pr[f"P4_{r}"] = res[r]["fails"] - res["D0"]["fails"] <= 3
            z1 = res[r]["fold0_config"]        # exact count: the names list in config.json is truncated to 8, so only the count is informative
            pr[f"Z1_{r}"] = z1 is not None and z1["trainable_params"] == BASE_PARAMS + DECODER_PARAMS
    out = _load_json(OUT) if os.path.isfile(OUT) else {}
    out.update({"prereg": "docs/stage1-E9-prereg.md", "stage1": dict(arms=res, passing_tags=passing, predictions=pr)})
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"stage-1 gate: passing {passing}   predictions {pr}")


def z3_digest(arm, method, result=None):
    """Watch value Z3 (prereg section 7): per scene, MANIFEST.json provenance.seeds_from.mask_digest of the in-process
    run == stage1_supply.mask_digest over the seeds rebuilt from the stage-1 CLI folder stage1/Data/<scene>/Result/<arm>_rep_e9
    (to_seeds: every Video camera x basic object, missing PNG = empty; the reference camera's GT is in that folder too)."""
    import importlib.util
    import numpy as np
    from PIL import Image
    spec = importlib.util.spec_from_file_location("stage1_supply", os.path.join(REPO, "SCSam3", "stage1_supply.py"))
    SF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SF)
    cfg = _load_json(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json"))
    census = _load_json(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json"))
    out = dict(ok=0, bad=[], absent=[])
    for sc in sorted(census):
        c = cfg[sc]; f = c["folder"]
        man = os.path.join(REPO, "Data", "MVSeg", f, method, "MANIFEST.json")
        if not os.path.exists(man):
            out["absent"].append(sc); continue
        want = _load_json(man).get("provenance", {}).get("seeds_from", {}).get("mask_digest")
        res_dir = os.path.join(REPO, "stage1", "Data", sc, "Result", result or f"{arm}_rep_e9")
        video_cams = sorted(x for x in os.listdir(os.path.join(REPO, "Data", "MVSeg", f, "Video")) if os.path.isdir(os.path.join(REPO, "Data", "MVSeg", f, "Video", x)))
        basic = census[sc]["seed_ids_c_ini"]; c_ini = census[sc]["c_ini"]
        shape = np.asarray(Image.open(os.path.join(REPO, "Data", "MVSeg", f, "Mask", c_ini, "%06d.png" % c["start_frame"]))).shape[:2]
        final = {}
        for cam in os.listdir(res_dir):
            d = os.path.join(res_dir, cam)
            if not os.path.isdir(d): continue
            for fn in os.listdir(d):
                if fn.endswith(".png"):
                    final.setdefault(cam, {})[int(fn[:-4])] = np.asarray(Image.open(os.path.join(d, fn)).convert("L")) > 127
        seeds, _ = SF.to_seeds(final, video_cams, basic, tuple(shape), c_ini, np)
        got = SF.mask_digest(seeds, np)
        if got == want: out["ok"] += 1
        else: out["bad"].append(sc)
    return out


# ---------------------------------------------------------------- stage 2
def stage2():
    sys.path.insert(0, os.path.join(REPO, "MVSeed"))
    import e7_judge as E7
    from e8_judge import ci_at
    out = _load_json(OUT) if os.path.isfile(OUT) else {}
    s1 = out.get("stage1")
    if s1 is None or "passing_tags" not in s1:
        raise SystemExit("stage-1 judgement missing (run --stage 1 first)")
    tags = s1["passing_tags"]
    if not tags:
        out["stage2"] = dict(runs={}, passed=[], adopted=None, adopted_tag=None, predictions={}, note="no arm passed the stage-1 gate")
        json.dump(out, open(OUT, "w"), indent=1)
        print("stage-2: no arm passed the stage-1 gate; nothing to judge (prereg section 5-1)"); return
    d = E7.load(["jf_e9.json"])
    res = {}
    for rid, arm, tag in ARMS:
        if tag not in tags:
            continue
        m = f"SegMaskSam3XW1CGPS4MSd{tag}FsCs10kG2Rs"
        Rm = list(E7.rows(d, m)); Rb = list(E7.rows(d, HEADLINE))
        if not Rm or not Rb:
            print(f"{rid}: no rows for {m} or {HEADLINE} (not scored yet?)"); continue
        sm, sb = E7.scene_scores(Rm), E7.scene_scores(Rb)
        cm, cb = E7.scene_scores(list(E7.rows(d, m, "complete"))), E7.scene_scores(list(E7.rows(d, HEADLINE, "complete")))
        st, stc = E7.paired(sb, sm), E7.paired(cb, cm)
        worst = min(st["per_scene"].values())
        rule = dict(mean_gt_0=st["mean"] > 0, ci_lower_gt_0=st["ci"][0] > 0, defence=worst >= DEFENCE, worst=worst, complete_ge_0=stc["mean"] >= 0)
        rule["pass"] = all(rule[k] for k in ("mean_gt_0", "ci_lower_gt_0", "defence", "complete_ge_0"))
        # Z3 (registered): the in-process MANIFEST's seeds_from.mask_digest == the digest of the stage-1 CLI output,
        # rebuilt through the same to_seeds rules (every Video camera x basic object, missing = empty)
        z3 = z3_digest(arm, m)
        # Z3b: stage-2 frame-0 PNGs of the scored non-reference pairs == the stage-1 CLI PNGs (pixels), basic objects
        z3b = dict(same=0, different=0, missing=0)
        try:
            from PIL import Image
            import numpy as np
            for r in Rm:
                if r["ref"]: continue
                c = E7.CFG[r["scene"]]; f, s0 = c["folder"], c["start_frame"]
                a = os.path.join(REPO, "stage1", "Data", r["scene"], "Result", f"{arm}_rep_e9", r["cam"], f"{r['obj']}.png")
                b = os.path.join(E7.DATA, f, m, r["cam"], str(s0), f"{r['obj']}.png")
                if not os.path.exists(a) and not os.path.exists(b): continue           # empty seed: no PNG on either side
                if not os.path.exists(a) or not os.path.exists(b): z3b["missing"] += 1; continue
                ma = np.asarray(Image.open(a).convert("L")) > 127; mb = np.asarray(Image.open(b).convert("L")) > 127
                z3b["same" if ma.shape == mb.shape and bool((ma == mb).all()) else "different"] += 1
        except ImportError as e:
            raise SystemExit(f"stage 2 needs numpy + PIL (run under stage1/.venv): {e}")
        # P6: the reference camera's frame 0 is the given GT in both runs (J&F 1 on every scored reference pair)
        ref0 = [E7.jf(r, 0) for r in Rm if r["ref"] and r["gt"][0]]
        p6 = all(x is not None and x > 0.999 for x in ref0)
        res[rid] = dict(method=m, tag=tag, base=HEADLINE, n_scenes=len(sm), basic=statistics.mean(sm.values()), base_basic=statistics.mean(sb.values()),
                        paired=st, ci_bonferroni3=ci_at(sb, sm, 1 - 0.05 / 3), complete=statistics.mean(cm.values()), base_complete=statistics.mean(cb.values()),
                        paired_complete=stc, rule=rule, Z3_mask_digest=z3, Z3b_frame0_pixels=z3b, P6_ref_frame0_is_gt=p6)
        print(f"{rid} {m}\n   basic {res[rid]['basic']:.3f} vs {res[rid]['base_basic']:.3f}  Δ {st['mean']:+.3f}  CI [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]"
              f"  (98.33% [{res[rid]['ci_bonferroni3'][0]:+.3f}, {res[rid]['ci_bonferroni3'][1]:+.3f}])  {st['wins']}↑/{st['losses']}↓  worst {st['worst'][1]} {st['worst'][0]:+.2f}"
              f"\n   complete {res[rid]['complete']:.3f} vs {res[rid]['base_complete']:.3f}  Δ {stc['mean']:+.3f}  rule {rule}\n   Z3 {z3}  Z3b {z3b}  P6 {p6}")
    passed = [r for r in ("D1", "D2") if r in res and res[r]["rule"]["pass"]]
    adopted = None
    if passed:
        best = max(res[r]["basic"] for r in passed)
        adopted = next(r for r in ("D1", "D2") if r in passed and res[r]["basic"] >= best - TIE)      # tie -> D1
    pr = {}
    for r in res:
        kf = s1.get("arms", {}).get(r, {}).get("paired", {}).get("JF", {}).get("delta")
        if kf:
            ratio = res[r]["paired"]["mean"] / (66.7 * kf)
            pr[f"P5_{r}"] = dict(ratio=ratio, ok=0.4 <= ratio <= 1.0)
        pr[f"P6_{r}"] = res[r]["P6_ref_frame0_is_gt"]
        pr[f"Z3_{r}"] = isinstance(res[r]["Z3_mask_digest"], dict) and res[r]["Z3_mask_digest"]["bad"] == [] and res[r]["Z3_mask_digest"]["ok"] == 17
    out["stage2"] = dict(runs=res, passed=passed, adopted=adopted, adopted_tag=res[adopted]["tag"] if adopted else None, predictions=pr)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nstage-2: passed {passed} -> adopted {adopted} ({out['stage2']['adopted_tag']})   predictions {pr}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    a = ap.parse_args()
    stage1() if a.stage == 1 else stage2()
