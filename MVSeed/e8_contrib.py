#!/usr/bin/env python3
"""S1-E8 descriptive decomposition (docs/stage1-E8-prereg.md section 6; not a registered rule).
The e7_contrib.py split applied to the six E8 runs: each run's benchmark change
ΔB = Σ_i w_i (S_i^new − S_i^old) over the reference camera, the F0 / N0 nonreference populations
and the remaining nonreference rows, with the MUVOD basic aggregation weight w_i, so the groups sum
exactly to the paired scene-mean Δ of e8_judge.py.

    python3 MVSeed/e8_contrib.py      ->  MVSeed/runs/e8_contrib.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e7_judge import load, rows, pair_key, REPO
from e7_contrib import pair_scores
from e8_judge import RUNS

GROUPS = ("ref", "F0", "N0", "other_nonref")


def main():
    d = load(["jf_e8.json", "jf_e7.json", "jf_r4.json"])
    F0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed", "runs", "F0.json")))["pairs"]}
    N0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed", "runs", "N0.json")))["pairs"]}
    out = {"note": __doc__.strip().splitlines()[0], "runs": {}}
    for rid, m, base, seeds in RUNS:
        Rm, Rb = list(rows(d, m)), list(rows(d, base))
        if not Rm: print(f"{rid}: no rows"); continue
        ref = {pair_key(r) for r in Rm if r["ref"]}
        pm, pb = pair_scores(Rm), pair_scores(Rb)
        assert set(pm) == set(pb), (rid, len(set(pm) ^ set(pb)))
        wdiff = max(abs(pm[k][1] - pb[k][1]) for k in pm)
        pops = {"ref": ref, "F0": F0, "N0": N0}
        contrib, counts = {}, {}
        for k in pm:
            g = next((name for name, pop in pops.items() if k in pop), "other_nonref")
            contrib[g] = contrib.get(g, 0.0) + pm[k][1] * (pm[k][0] - pb[k][0]); counts[g] = counts.get(g, 0) + 1
        total = sum(contrib.values())
        out["runs"][rid] = dict(method=m, base=base, seeds=seeds, total=total, weight_mismatch=wdiff,
                                contrib={g: round(v, 4) for g, v in contrib.items()}, n=counts)
        print(f"{rid} {m}: ΔB {total:+.4f}  " + "  ".join(f"{g} {contrib[g]:+.3f} (n={counts[g]})" for g in GROUPS if g in contrib)
              + f"  max|Δw| {wdiff:.2e}")
    json.dump(out, open(os.path.join(REPO, "MVSeed", "runs", "e8_contrib.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
