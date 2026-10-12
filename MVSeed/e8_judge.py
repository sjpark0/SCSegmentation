#!/usr/bin/env python3
"""S1-E8 judge (docs/stage1-E8-prereg.md sections 4-7): scores the six GeoSet-seed runs against
their G0 bases on the MUVOD protocol, applies the adoption rule to the three headline runs, checks
predictions P1-P6 and watch values Z1-Z2, adds the comparison table (MUVOD XMem, submitted headline,
E7 ground-truth seeds) and writes MVSeed/runs/e8_judge.json.  Host python (json, hashlib).

    python3 eval/eval_jf.py --methods <the six folders> --out jf_e8.json   # in the container
    python3 MVSeed/e8_judge.py
"""
import hashlib, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_judge as E7                     # load / rows / jf / scene_scores / paired: the E7 aggregation, unchanged

REPO, DATA, CFG, SF = E7.REPO, E7.DATA, E7.CFG, E7.SF
HEAD_BASE, CTRL_BASE = "SegMaskSam3XW1CGPS4MFsCs10kG2", "SegMaskSam3XW0MFs"
ARMS = [("A", "geoset_rep"), ("B", "geoset_abs"), ("C", "geoalign_rep")]
RUNS = [(f"{a}2", f"SegMaskSam3XW1CGPS4MSd{t}FsCs10kG2", HEAD_BASE, f"MVSeed_{t}") for a, t in ARMS] + \
       [(f"{a}1", f"SegMaskSam3XW0MSd{t}Fs", CTRL_BASE, f"MVSeed_{t}") for a, t in ARMS]
TIE_ORDER = ["A2", "C2", "B2"]            # prereg section 5: 2D-repo recommendation order
DEFENCE = -1.0                            # no scene below this basic delta
TIE = 0.05
E7_GT = {"control": "SegMaskSam3XW0MSdgtFs", "headline": "SegMaskSam3XW1CGPS4MSdgtFsCs10kG2"}


def ci_at(a, b, level):
    """Scene-cluster bootstrap CI of mean(b - a) at `level`, same resampling stream as E7.paired."""
    import random
    scenes = sorted(set(a) & set(b)); d = [b[s] - a[s] for s in scenes]
    rng = random.Random(E7.SEED); n = len(d)
    boots = sorted(statistics.mean(d[rng.randrange(n)] for _ in range(n)) for _ in range(E7.N_BOOT))
    lo = (1 - level) / 2
    return [boots[int(lo * E7.N_BOOT)], boots[int((1 - lo) * E7.N_BOOT) - 1]]


def xmem():
    t = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_tables.json")))
    return {k: {r["label"]: r["MUVOD XMem"] for r in t[f"muvod-{k}"]["rows"] if r.get("MUVOD XMem") is not None} for k in ("basic", "complete")}


def main():
    d = E7.load(["jf_e8.json", "jf_e7.json", "jf_r4.json"])
    F0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed", "runs", "F0.json")))["pairs"]}
    N0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed", "runs", "N0.json")))["pairs"]}
    out = {"prereg": "docs/stage1-E8-prereg.md", "runs": {}}
    cache = {}
    def scores(m, objects):
        k = (m, objects)
        if k not in cache:
            cache[k] = E7.scene_scores(list(E7.rows(d, m, objects)))
        return cache[k]
    for rid, m, base, seeds in RUNS:
        Rm = list(E7.rows(d, m)); Rb = list(E7.rows(d, base))
        if not Rm: print(f"{rid}: no rows for {m} (not scored yet?)"); continue
        sm, sb = scores(m, "basic"), scores(base, "basic")
        cm, cb = scores(m, "complete"), scores(base, "complete")
        st, stc = E7.paired(sb, sm), E7.paired(cb, cm)
        res = dict(method=m, base=base, seeds=seeds, n_scenes=len(sm), basic=statistics.mean(sm.values()), base_basic=statistics.mean(sb.values()),
                   paired=st, ci_bonferroni3=ci_at(sb, sm, 1 - 0.05 / 3),
                   complete=statistics.mean(cm.values()), base_complete=statistics.mean(cb.values()), paired_complete=stc)
        pm = {E7.pair_key(r): statistics.mean(v) for r in Rm if (v := [x for x in (E7.jf(r, t) for t in range(1, len(r["J"]))) if x is not None])}
        pb = {E7.pair_key(r): statistics.mean(v) for r in Rb if (v := [x for x in (E7.jf(r, t) for t in range(1, len(r["J"]))) if x is not None])}
        split = {}
        for name, pop in (("F0", F0), ("N0", N0)):
            ks = [k for k in pop if k in pm and k in pb]; dd = [pm[k] - pb[k] for k in ks]
            split[name] = dict(n=len(ks), mean_dJF=statistics.mean(dd) if dd else None,
                               recovered=sum(1 for k in ks if pb[k] < 0.5 <= pm[k]), broken=sum(1 for k in ks if pm[k] < 0.5 <= pb[k]),
                               up=sum(1 for x in dd if x > 0.05), down=sum(1 for x in dd if x < -0.05))
        res["pairs_1_20"] = split
        # frame-0 J of non-reference scored pairs with GT present (the seed quality the stage-2 run actually started from)
        f0 = [r["J"][0] for r in Rm if not r["ref"] and r["gt"][0] and r["J"][0] is not None]
        res["frame0_J_nonref"] = dict(n=len(f0), mean=statistics.mean(f0) if f0 else None)
        # Z1 frame-0 PNG == seed PNG bytes
        same = diff = missing = 0
        for r in Rm:
            c = CFG[r["scene"]]; f, s0 = c["folder"], c["start_frame"]
            a = os.path.join(DATA, f, seeds, r["cam"], str(s0), f"{r['obj']}.png"); b = os.path.join(DATA, f, m, r["cam"], str(s0), f"{r['obj']}.png")
            if not os.path.exists(a): continue
            if not os.path.exists(b): missing += 1; continue
            if hashlib.sha256(open(a, "rb").read()).digest() == hashlib.sha256(open(b, "rb").read()).digest(): same += 1
            else: diff += 1
        res["Z1_frame0_bytes"] = dict(same=same, different=diff, missing=missing)
        # Z2 manifest provenance digest == seed folder digest
        z2 = {"ok": 0, "bad": [], "absent": 0}
        for sc in sorted(sm):
            f = CFG[sc]["folder"]; man = os.path.join(DATA, f, m, "MANIFEST.json")
            if not os.path.exists(man): z2["absent"] += 1; continue
            prov = json.load(open(man)).get("provenance", {}).get("seeds_from", {})
            dig, _ = SF.png_digest(os.path.join(DATA, f, seeds))
            if prov.get("png_digest") == dig: z2["ok"] += 1
            else: z2["bad"].append(sc)
        res["Z2_provenance"] = z2
        if rid.endswith("2"):
            worst = min(st["per_scene"].values())
            res["rule"] = dict(ci_lower_gt_0=st["ci"][0] > 0, mean_gt_0=st["mean"] > 0, defence=worst >= DEFENCE, worst=worst,
                               complete_ge_0=stc["mean"] >= 0)
            res["rule"]["pass"] = all(res["rule"][k] for k in ("ci_lower_gt_0", "mean_gt_0", "defence", "complete_ge_0"))
        out["runs"][rid] = res
        print(f"{rid} {m}\n   basic {res['basic']:.3f} vs base {res['base_basic']:.3f}  Δ {st['mean']:+.3f}  CI [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]"
              f"  (98.33% [{res['ci_bonferroni3'][0]:+.3f}, {res['ci_bonferroni3'][1]:+.3f}])  {st['wins']}↑/{st['losses']}↓"
              f"  worst {st['worst'][1]} {st['worst'][0]:+.2f}  best {st['best'][1]} {st['best'][0]:+.2f}")
        print(f"   complete {res['complete']:.3f} vs base {res['base_complete']:.3f}  Δ {stc['mean']:+.3f}  CI [{stc['ci'][0]:+.3f}, {stc['ci'][1]:+.3f}]")
        for k, v in split.items(): print(f"   {k}: n={v['n']} ΔJ&F(1-20) {v['mean_dJF']:+.4f}  recovered {v['recovered']} broken {v['broken']}  up {v['up']} down {v['down']}")
        print(f"   frame-0 J (non-ref) {res['frame0_J_nonref']}   Z1 {res['Z1_frame0_bytes']}   Z2 {z2}")
        if "rule" in res: print(f"   rule {res['rule']}")

    # adoption (section 5)
    passed = [r for r in TIE_ORDER if r in out["runs"] and out["runs"][r]["rule"]["pass"]]
    adopted = None
    if passed:
        best = max(out["runs"][r]["basic"] for r in passed)
        adopted = next(r for r in TIE_ORDER if r in passed and out["runs"][r]["basic"] >= best - TIE)
    out["adoption"] = dict(passed=passed, adopted=adopted, adopted_method=out["runs"][adopted]["method"] if adopted else None,
                           headline_runs_scored=[r for r in TIE_ORDER if r in out["runs"]])
    print(f"\nadoption: passed {passed} -> adopted {adopted}")

    # predictions (section 4)
    R = out["runs"]; g = lambda r: R[r]["paired"]["mean"] if r in R else None
    pr = {}
    if g("A1") is not None: pr["P1"] = 0.3 <= g("A1") <= 1.5
    if g("A2") is not None: pr["P2"] = 0.2 <= g("A2") <= 1.5
    if all(g(r) is not None for r in ("A2", "B2", "C2")):
        v = [g("A2"), g("B2"), g("C2")]; pr["P3"] = max(v) - min(v) <= 0.3
        pr["P4"] = R["B2"]["paired"]["per_scene"].get("Welder", 0) < R["A2"]["paired"]["per_scene"].get("Welder", 0)
    if "A1" in R: pr["P5"] = R["A1"]["pairs_1_20"]["F0"]["recovered"] > R["A1"]["pairs_1_20"]["F0"]["broken"] and R["A1"]["pairs_1_20"]["N0"]["broken"] <= 3
    if R: pr["P6"] = all(v["Z1_frame0_bytes"]["different"] == 0 and v["Z1_frame0_bytes"]["missing"] == 0 for v in R.values())
    out["predictions"] = pr
    print(f"predictions {pr}")

    # comparison table (section 6)
    xm = xmem(); table = []
    def row(label, m=None, fixed=None):
        if m:
            b, c = scores(m, "basic"), scores(m, "complete")
            if not b: return
            table.append(dict(label=label, method=m, basic=statistics.mean(b.values()), complete=statistics.mean(c.values()), n=len(b)))
        else:
            table.append(dict(label=label, method=None, basic=fixed[0], complete=fixed[1], n=17))
    row("MUVOD XMem (paper Table III/IV)", fixed=(statistics.mean(xm["basic"].values()), statistics.mean(xm["complete"].values())))
    row("Control, G0 seeds", CTRL_BASE)
    row("Submitted headline, G0 seeds", HEAD_BASE)
    for rid, m, _, _ in RUNS: row(f"{rid} ({m})", m)
    row("E7 control + GT seeds (upper bound)", E7_GT["control"])
    row("E7 headline + GT seeds (upper bound)", E7_GT["headline"])
    out["table"] = table
    print("\n| config | basic | complete |\n|---|---|---|")
    for t in table: print(f"| {t['label']} | {t['basic']:.3f} | {t['complete']:.3f} |")
    # per-scene basic, every scored config + XMem
    cols = [("XMem", None), ("G0 ctrl", CTRL_BASE), ("G0 head", HEAD_BASE)] + [(rid, m) for rid, m, _, _ in RUNS if rid in R]
    per = {}
    for sc in sorted(xm["basic"]):
        per[sc] = {"XMem": xm["basic"][sc]}
        for name, m in cols[1:]: per[sc][name] = scores(m, "basic").get(sc)
    out["per_scene_basic"] = per
    json.dump(out, open(os.path.join(REPO, "MVSeed", "runs", "e8_judge.json"), "w"), indent=1)
    print("wrote MVSeed/runs/e8_judge.json")


if __name__ == "__main__":
    main()
