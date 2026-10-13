#!/usr/bin/env python3
"""S2-R6 judge (docs/stage2-R6-prereg.md sections 3-4): the reference-camera static snap applied to
the E8 headline (H-Rs, the adoption candidate) and control (C-Rs, the arm not used to pick the rule),
against their E8 bases.  Watch values Z1-Z4 first (no verdict when one fails), then the registered
rule, the contribution split, predictions P1-P4.  Host python (json, hashlib).

    python3 MVSeed/s2r6_judge.py      ->  MVSeed/runs/s2r6_judge.json
"""
import hashlib, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_judge as E7                     # load / rows / jf / scene_scores / paired: the E7 aggregation, unchanged
from e7_contrib import pair_scores

REPO, DATA, CFG = E7.REPO, E7.DATA, E7.CFG
SCENES = ["AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "CBABasketball", "Carpark",
          "CoffeeMartini", "Dog", "Fencing", "FlameSteak", "Frog", "MATF", "MartialArts", "Painter", "PoznanStreet", "Welder"]
RUNS = [("H", "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs", "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2"),
        ("C", "SegMaskSam3XW0MSdgeoset_repFsRs", "SegMaskSam3XW0MSdgeoset_repFs")]
RULE = dict(tau=0.8, delta=4.0, dilate=31)
SIM_REF_H, SIM_TOL = 0.3645, 0.002                          # prereg section 3, Z4
ADOPT_MIN, ADOPT_WORST = 0.3, -0.3
OBJ = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def tree(d):
    """{relative path: sha256} of every regular file under d, MANIFEST.json at the top excluded."""
    out = {}
    for root, _, files in os.walk(d):
        for n in files:
            p = os.path.join(root, n); rel = os.path.relpath(p, d)
            if rel != "MANIFEST.json": out[rel] = sha(p)
    return out


def committed_snap_sha(rev):
    import subprocess
    r = subprocess.run(["git", "-C", REPO, "show", f"{rev}:MVSeed/s2r6_snap.py"], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def watch(scene, m, base):
    """Z1 (untouched files identical, same file set), Z2 (changed PNGs = the manifest's snapped list,
    each equal to its frame-0 PNG), Z3 (manifest rule / source / snap script = the committed file at the
    manifest's git_head; the dirty count is recorded only -- the tree carries an unrelated pre-existing edit)."""
    f, s0, ref = CFG[scene]["folder"], str(CFG[scene]["start_frame"]), OBJ[scene]["c_ini"]
    bdir, mdir = os.path.join(DATA, f, base), os.path.join(DATA, f, m)
    tb, tm = tree(bdir), tree(mdir)
    z1 = dict(same=0, different=0, only_one_side=len(set(tb) ^ set(tm)))
    changed = set(); not_f0 = 0
    for rel in set(tb) & set(tm):
        cam, frame, n = (rel.split(os.sep) + ["", "", ""])[:3]
        if cam == ref and frame != s0:
            if tb[rel] != tm[rel]:
                changed.add(rel); not_f0 += tm[rel] != tb.get(os.path.join(cam, s0, n))
        elif tb[rel] == tm[rel]: z1["same"] += 1
        else: z1["different"] += 1
    z1["ok"] = z1["different"] == 0 and z1["only_one_side"] == 0
    man = json.load(open(os.path.join(mdir, "MANIFEST.json"))) if os.path.exists(os.path.join(mdir, "MANIFEST.json")) else {}
    st = man.get("stats", {})
    listed = {x.replace("/", os.sep) for x in st.get("snapped_files", [])}
    noop = listed - changed                                          # snapped, but the source was already the frame-0 PNG
    z2 = dict(changed=len(changed), snapped=st.get("snapped"), changed_not_frame0=not_f0, changed_not_listed=len(changed - listed),
              noop_snaps=len(noop), noop_not_frame0=sum(1 for r in noop if tb.get(r) != tb.get(os.path.join(r.split(os.sep)[0], s0, r.split(os.sep)[2]))))
    z2["ok"] = (not_f0 == 0 and z2["changed_not_listed"] == 0 and z2["noop_not_frame0"] == 0
                and st.get("snapped") == len(listed) == len(st.get("snapped_files", [])))
    z3 = dict(rule=man.get("rule") == RULE, source=man.get("source_method") == base, ref=st.get("ref") == ref,
              script=man.get("git_head") is not None and man.get("script_sha256") == committed_snap_sha(man.get("git_head")))
    z3["ok"] = all(z3.values())
    return z1, z2, z3


def main():
    d = E7.load(["jf_s2r6.json", "jf_e8.json"])
    F0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed", "runs", "F0.json")))["pairs"]}
    N0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed", "runs", "N0.json")))["pairs"]}
    out = {"prereg": "docs/stage2-R6-prereg.md", "runs": {}}
    for rid, m, base in RUNS:
        Rm, Rb = list(E7.rows(d, m)), list(E7.rows(d, base))
        if not Rm: print(f"{rid}: no rows for {m} (not scored yet?)"); continue
        sm, sb = E7.scene_scores(Rm), E7.scene_scores(Rb)
        cm, cb = E7.scene_scores(list(E7.rows(d, m, "complete"))), E7.scene_scores(list(E7.rows(d, base, "complete")))
        st, stc = E7.paired(sb, sm), E7.paired(cb, cm)
        res = dict(method=m, base=base, n_scenes=len(sm), basic=statistics.mean(sm.values()), base_basic=statistics.mean(sb.values()),
                   paired=st, complete=statistics.mean(cm.values()), base_complete=statistics.mean(cb.values()), paired_complete=stc)
        ref = {E7.pair_key(r) for r in Rm if r["ref"]}
        pm, pb = pair_scores(Rm), pair_scores(Rb)
        assert set(pm) == set(pb), (rid, len(set(pm) ^ set(pb)))
        contrib = {}
        for k in pm:
            g = "ref" if k in ref else "F0" if k in F0 else "N0" if k in N0 else "other_nonref"
            contrib[g] = contrib.get(g, 0.0) + pm[k][1] * (pm[k][0] - pb[k][0])
        res["contrib"] = contrib
        res["contrib_sum_minus_delta"] = round(sum(contrib.values()) - st["mean"], 6)
        W = {sc: watch(sc, m, base) for sc in SCENES if sc in sm}
        res["Z1"] = {sc: w[0] for sc, w in W.items()}; res["Z2"] = {sc: w[1] for sc, w in W.items()}; res["Z3"] = {sc: w[2] for sc, w in W.items()}
        nonref = sum(v for g, v in contrib.items() if g != "ref")
        res["Z_pass"] = dict(Z1=all(w[0]["ok"] for w in W.values()), Z2=all(w[1]["ok"] for w in W.values()),
                             Z3=all(w[2]["ok"] for w in W.values()),
                             Z4=(abs(contrib.get("ref", 0) - SIM_REF_H) <= SIM_TOL if rid == "H" else True) and nonref == 0)
        res["snapped"] = sum(w[1]["snapped"] or 0 for w in W.values()); res["changed"] = sum(w[1]["changed"] for w in W.values())
        out["runs"][rid] = res
        print(f"{rid} {m}\n   basic {res['basic']:.3f} vs base {res['base_basic']:.3f}  Δ {st['mean']:+.4f}  CI [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]"
              f"  {st['wins']}↑/{st['losses']}↓  worst {st['worst'][1]} {st['worst'][0]:+.3f}  best {st['best'][1]} {st['best'][0]:+.3f}")
        print(f"   complete {res['complete']:.3f} vs base {res['base_complete']:.3f}  Δ {stc['mean']:+.4f}  CI [{stc['ci'][0]:+.3f}, {stc['ci'][1]:+.3f}]")
        print(f"   contrib {{{', '.join(f'{g}: {v:+.4f}' for g, v in contrib.items())}}}  (sum − Δ {res['contrib_sum_minus_delta']:+.6f})")
        print(f"   snapped {res['snapped']}  changed PNGs {res['changed']}  Z {res['Z_pass']}")

    R = out["runs"]
    if "H" in R:
        h = R["H"]; st = h["paired"]
        complete = "C" in R and all(R[k]["n_scenes"] == len(SCENES) for k in ("H", "C"))
        zok = complete and all(h["Z_pass"].values()) and all(R["C"]["Z_pass"].values())
        c = R["C"]["paired"] if "C" in R else None
        rule = dict(delta=st["mean"], ci_lower=st["ci"][0], worst=st["worst"][0], complete_delta=h["paired_complete"]["mean"],
                    control_delta=c["mean"] if c else None, control_ci_lower=c["ci"][0] if c else None)
        rule["adopt"] = (st["mean"] >= ADOPT_MIN and st["ci"][0] > 0 and st["worst"][0] >= ADOPT_WORST and h["paired_complete"]["mean"] >= 0
                         and c is not None and c["mean"] > 0 and c["ci"][0] > 0)
        verdict = ("no verdict: both runs over all 17 scenes are required" if not complete else
                   "no verdict: a watch value failed" if not zok else "adopt" if rule["adopt"] else "reject")
        pr = dict(P1=0.35 <= st["mean"] <= 0.38)
        if c:
            pr["P2"] = 0.25 <= c["mean"] <= 0.60
            pr["P3"] = all(0.7 <= R[k]["paired_complete"]["mean"] / R[k]["paired"]["mean"] <= 1.1 for k in ("H", "C") if R[k]["paired"]["mean"])
            pr["P4"] = zok
        out.update(rule=rule, verdict=verdict, predictions=pr)
        print(f"\nrule {rule}\nverdict: {verdict}\npredictions {pr}")
    json.dump(out, open(os.path.join(REPO, "MVSeed", "runs", "s2r6_judge.json"), "w"), indent=1)
    print("wrote MVSeed/runs/s2r6_judge.json")


if __name__ == "__main__":
    main()
