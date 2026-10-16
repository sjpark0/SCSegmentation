#!/usr/bin/env python3
"""S2-R7 judge (docs/stage2-R7-prereg.md section 8; E8 section-5 rule, e7_judge helpers as in e14_judge --stage 2).
Arm M = SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2RmRs vs base H = ...Cs10kG2Rs (scored in jf_e14.json, not rerun).
Needs MVSeed/runs/s2r7_check.json ok (Z1-Z6).  Host python:
    PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python MVSeed/s2r7_judge.py      -> MVSeed/runs/s2r7_judge.json
"""
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "MVSeed", "runs", "s2r7_judge.json")
CHECK = os.path.join(REPO, "MVSeed", "runs", "s2r7_check.json")
H = "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs"
M = "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2RmRs"
DEFENCE = -1.0


def main():
    sys.path.insert(0, os.path.join(REPO, "MVSeed"))
    import e7_judge as E7
    chk = json.load(open(CHECK)) if os.path.isfile(CHECK) else {}
    if not chk.get("ok"):
        raise SystemExit("watch values Z1-Z6 not OK (run MVSeed/s2r7_check.py first): no verdict (prereg section 7)")
    sys.path.insert(0, os.path.join(REPO, "eval"))
    import manifest as Mf
    stale = [sc for sc, dig in chk["content_digest_M"].items()
             if Mf.content_digest(os.path.join(REPO, "Data", "MVSeg", E7.CFG[sc]["folder"], M))[0] != dig]
    if stale:
        raise SystemExit(f"arm folders changed after the watch-value check: {stale} (rerun s2r7_check.py and the scoring)")
    d = E7.load(["jf_s2r7.json", "jf_e14.json"])
    Rm, Rh = list(E7.rows(d, M)), list(E7.rows(d, H))
    if not Rm or not Rh:
        raise SystemExit(f"no scored rows for {M if not Rm else H}: score first (no verdict)")
    sm, sh = E7.scene_scores(Rm), E7.scene_scores(Rh)
    cm, ch = E7.scene_scores(list(E7.rows(d, M, "complete"))), E7.scene_scores(list(E7.rows(d, H, "complete")))
    st, stc = E7.paired(sh, sm), E7.paired(ch, cm)
    n17 = len(sm) == len(sh) == len(cm) == len(ch) == 17 and st["n"] == stc["n"] == 17
    if not n17:
        with open(OUT, "w") as f:
            json.dump(dict(prereg="docs/stage2-R7-prereg.md", note="fewer than 17 scenes scored: not judged",
                           n=dict(arm=len(sm), base=len(sh), arm_complete=len(cm), base_complete=len(ch))), f, indent=1)
        raise SystemExit(f"not all 17 scenes scored (arm {len(sm)}, base {len(sh)}): no verdict (prereg section 8)")
    worst = st["worst"][0]
    rule = dict(all_17_scenes=n17, watch_values=True, mean_gt_0=st["mean"] > 0, ci_lower_gt_0=st["ci"][0] > 0,
                defence=worst >= DEFENCE, worst=worst, complete_ge_0=stc["mean"] >= 0)
    rule["pass"] = all(rule[k] for k in ("all_17_scenes", "watch_values", "mean_gt_0", "ci_lower_gt_0", "defence", "complete_ge_0"))
    verdict = "ADOPTED" if rule["pass"] else ("not adopted: harms" if st["ci"][1] < 0 else
                                              "not adopted: feeding the S2-R6 decisions into memory gives no measurable gain")
    # descriptives: c_ini vs other cameras, other cameras by session distance from c_ini
    def cam_scores(R):
        per = {}
        for r in R:
            v = [x for x in (E7.jf(r, t) for t in range(len(r["J"]))) if x is not None]
            if v:
                per.setdefault((r["scene"], r["cam"], r["ref"]), []).append(statistics.mean(v))
        return {k: 100 * statistics.mean(v) for k, v in per.items()}
    ca, cb = cam_scores(Rm), cam_scores(Rh)
    common = sorted(set(ca) & set(cb))
    cini = [ca[k] - cb[k] for k in common if k[2]]
    other = [ca[k] - cb[k] for k in common if not k[2]]
    # session distance d of each scored camera from c_ini (arm MANIFEST track_idx / reference_view_index, perms fallback)
    dist = {}
    for sc in E7.CEN:
        c = E7.CFG[sc]
        prov = json.load(open(os.path.join(REPO, "Data", "MVSeg", c["folder"], M, "MANIFEST.json")))["provenance"]
        pm = c.get("perms") or list(range(c["start_cam"], c["start_cam"] + c["num_cam"]))
        tidx, r = prov["track_idx"], prov["track_idx"].index(prov["reference_view_index"])
        for i, p in enumerate(pm):
            if i in tidx:
                dist[(sc, f"{c['prefix']}{p:0{c['prefix1']}d}")] = tidx.index(i) - r
    by_d = {}
    for k in common:
        dd = abs(dist.get((k[0], k[1]), 999))
        by_d.setdefault(str(dd), []).append(ca[k] - cb[k])
    desc = dict(cini_mean_delta=statistics.mean(cini) if cini else None, other_mean_delta=statistics.mean(other) if other else None,
                cini_basic_contribution=sum(cini) / 3 / 17 if cini else None, other_basic_contribution=sum(other) / 3 / 17 if other else None,
                by_abs_session_distance={dd: dict(n=len(v), basic_contribution=sum(v) / 3 / 17) for dd, v in sorted(by_d.items(), key=lambda x: int(x[0]))},
                per_camera={f"{k[0]}/{k[1]}{' (c_ini)' if k[2] else ''}": round(ca[k] - cb[k], 4) for k in common})
    ci_, ot_ = desc["cini_basic_contribution"], desc["other_basic_contribution"]
    P = dict(P5=dict(basic=st["mean"], in_80=-0.03 <= st["mean"] <= 0.04, complete_minus_basic=stc["mean"] - st["mean"],
                     cdiff_ok=abs(stc["mean"] - st["mean"]) <= 0.03, worst=worst, worst_ok=worst >= -0.2,
                     cini=ci_, cini_ok=ci_ is not None and -0.02 <= ci_ <= 0.03, other=ot_, other_ok=ot_ is not None and -0.01 <= ot_ <= 0.01),
             P1=dict(injected=chk["total_injected"], ok=3150 <= chk["total_injected"] <= 3183),     # erratum: bound 3 183
             P2=dict(snaps=chk["total_snaps_arm"], ok=4560 <= chk["total_snaps_arm"] <= 4600))
    res = dict(prereg="docs/stage2-R7-prereg.md", method=M, base=H, basic=statistics.mean(sm.values()), base_basic=statistics.mean(sh.values()),
               paired=st, complete=statistics.mean(cm.values()), base_complete=statistics.mean(ch.values()), paired_complete=stc,
               rule=rule, verdict=verdict, descriptives=desc, predictions=P)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{M}\n   basic {res['base_basic']:.3f} -> {res['basic']:.3f}  Δ {st['mean']:+.4f}  CI [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}]  "
          f"{st['wins']}↑/{st['losses']}↓  worst {st['worst'][1]} {worst:+.3f}\n   complete {res['base_complete']:.3f} -> {res['complete']:.3f}  "
          f"Δ {stc['mean']:+.4f}  CI [{stc['ci'][0]:+.4f}, {stc['ci'][1]:+.4f}]\n   per scene {st['per_scene']}\n   c_ini contribution "
          f"{desc['cini_basic_contribution']}, other cameras {desc['other_basic_contribution']}\n   rule {rule}\n   predictions {P}")
    print(f"\nVERDICT (registered): {verdict}")


if __name__ == "__main__":
    main()
