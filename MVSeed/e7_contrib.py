#!/usr/bin/env python3
"""S1-E7 descriptive decomposition (not a registered rule; docs/stage1-E7.md §6).  Splits each
run's benchmark change  ΔB = Σ_i w_i (S_i^new − S_i^old)  over all scored (camera, object) pairs
into the reference camera, the F0 / N0 nonreference populations and the remaining nonreference
rows (GT-empty at the seed frame or unmatched).  S_i = the pair's mean J&F over all 21 frames,
w_i = 1 / (n_scenes · n_cams(scene) · n_obj(scene, cam)) = the MUVOD basic aggregation weight,
so the groups sum exactly to the paired scene-mean Δ of e7_judge.py.

    python3 MVSeed/e7_contrib.py      ->  MVSeed/runs/e7_contrib.json
"""
import json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e7_judge import load, rows, jf, pair_key, RUNS, REPO


def pair_scores(R):
    """{(scene, cam, obj): (mean J&F over frames 0..T, benchmark weight in points)}"""
    per = {}
    for r in R:
        v = [x for x in (jf(r, t) for t in range(len(r["J"]))) if x is not None]
        if v: per.setdefault(r["scene"], {}).setdefault(r["cam"], {})[r["obj"]] = statistics.mean(v)
    n_sc = len(per); out = {}
    for sc, cams in per.items():
        for cam, objs in cams.items():
            for o, v in objs.items():
                out[(sc, cam, o)] = (v, 100.0 / (n_sc * len(cams) * len(objs)))
    return out


def main():
    d = load(["jf_e7.json", "jf_r4.json"])
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
        groups = {"ref": ref, "F0": F0, "N0": N0}
        contrib, counts = {}, {}
        for k in pm:
            g = next((name for name, pop in groups.items() if k in pop), "other_nonref")
            contrib[g] = contrib.get(g, 0.0) + pm[k][1] * (pm[k][0] - pb[k][0]); counts[g] = counts.get(g, 0) + 1
        total = sum(contrib.values())
        res = dict(method=m, base=base, total=total, weight_mismatch=wdiff,
                   contrib={g: round(v, 4) for g, v in contrib.items()}, n={g: counts[g] for g in counts},
                   share_of_total={g: round(v / total, 4) for g, v in contrib.items()} if abs(total) > 1e-9 else None)
        out["runs"][rid] = res
        print(f"{rid} {m}: ΔB {total:+.3f}  " + "  ".join(f"{g} {contrib[g]:+.3f} (n={counts[g]})" for g in ("ref", "F0", "N0", "other_nonref") if g in contrib) + f"  max|Δw| {wdiff:.2e}")
    json.dump(out, open(os.path.join(REPO, "MVSeed", "runs", "e7_contrib.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
