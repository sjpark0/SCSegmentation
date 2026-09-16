#!/usr/bin/env python3
"""S1-E7 judge (docs/stage1-E7-prereg.md sections 4-6): scores the four seed-supply runs
against their bases on the MUVOD basic protocol, checks predictions P1-P6 and the watch
values Z1-Z2, and writes MVSeed/runs/e7_judge.json.  Host python (json, hashlib, PIL).

    python3 eval/eval_jf.py --methods SegMaskSam3XW0MSdgtFs SegMaskSam3XW1CGPS4MSdgtFsCs10kG2 \
        SegMaskSam3XW0MSde6_voteFs SegMaskSam3XW0MSde6_oracleFs --out jf_e7.json   # in the container
    python3 MVSeed/e7_judge.py
"""
import hashlib, json, os, random, statistics, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data", "MVSeg")
sys.path.insert(0, os.path.join(REPO, "SCSam3", "demoSCSam3MVOpt"))
import seeds_from as SF
CEN = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json"), encoding="utf-8"))
RUNS = [  # (id, method folder, base folder, seed folder)
    ("R1", "SegMaskSam3XW0MSdgtFs", "SegMaskSam3XW0MFs", "MVSeed_gt"),
    ("R2", "SegMaskSam3XW1CGPS4MSdgtFsCs10kG2", "SegMaskSam3XW1CGPS4MFsCs10kG2", "MVSeed_gt"),
    ("R3", "SegMaskSam3XW0MSde6_voteFs", "SegMaskSam3XW0MFs", "MVSeed_e6_vote"),
    ("R4", "SegMaskSam3XW0MSde6_oracleFs", "SegMaskSam3XW0MFs", "MVSeed_e6_oracle"),
]
N_BOOT, SEED = 10000, 0


def load(names):
    d = []
    for n in names:
        d += json.load(open(os.path.join(DATA, n)))
    return d


def rows(d, method):
    for r in d:
        if r["method"] != method or not r.get("result"): continue
        sc = r["dataset"]; basic = set(CEN[sc]["seed_ids_c_ini"]); ref = CEN[sc]["c_ini"]; pf = r["result"]["per_frame"]
        for i, o in enumerate(r["objects"]):
            if o not in basic: continue
            yield dict(scene=sc, cam=r["camera"], obj=o, ref=r["camera"] == ref, J=pf["J"][i], F=pf["F"][i], gt=pf["gt_area"][i])


def jf(r, t):
    return None if r["J"][t] is None or r["F"][t] is None else 0.5 * (r["J"][t] + r["F"][t])


def scene_scores(R):
    """{scene: basic J&F^3 (%)}: mean over objects -> mean over cameras -> per scene."""
    per = {}
    for r in R:
        v = [x for x in (jf(r, t) for t in range(len(r["J"]))) if x is not None]
        if v: per.setdefault(r["scene"], {}).setdefault(r["cam"], []).append(statistics.mean(v))
    return {s: 100 * statistics.mean([statistics.mean(v) for v in cams.values()]) for s, cams in per.items()}


def paired(a, b):
    """scene-cluster bootstrap of mean(b - a), Random(SEED), N_BOOT resamples."""
    scenes = sorted(set(a) & set(b)); d = [b[s] - a[s] for s in scenes]
    rng = random.Random(SEED); n = len(d); boots = []
    for _ in range(N_BOOT):
        boots.append(statistics.mean(d[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    return dict(n=n, mean=statistics.mean(d), ci=[boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]],
                wins=sum(x > 0 for x in d), losses=sum(x < 0 for x in d),
                worst=min(zip(d, scenes)), best=max(zip(d, scenes)), per_scene={s: round(x, 3) for s, x in zip(scenes, d)})


def pair_key(r): return (r["scene"], r["cam"], r["obj"])


def main():
    d = load(["jf_e7.json", "jf_r4.json"])
    F0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed", "runs", "F0.json")))["pairs"]}
    N0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed", "runs", "N0.json")))["pairs"]}
    out = {"prereg": "docs/stage1-E7-prereg.md", "runs": {}}
    base_cache = {}
    for rid, m, base, seeds in RUNS:
        Rm = list(rows(d, m)); Rb = base_cache.setdefault(base, list(rows(d, base)))
        if not Rm: print(f"{rid}: no rows for {m} (not scored yet?)"); continue
        sm, sb = scene_scores(Rm), scene_scores(Rb)
        st = paired(sb, sm)
        res = dict(method=m, base=base, seeds=seeds, n_scenes=len(sm), basic=statistics.mean(sm.values()), base_basic=statistics.mean(sb.values()), paired=st)
        # per-pair mean J&F over frames 1..20, F0 / N0 split (non-reference rows only)
        pm = {pair_key(r): statistics.mean(v) for r in Rm if (v := [x for x in (jf(r, t) for t in range(1, len(r["J"]))) if x is not None])}
        pb = {pair_key(r): statistics.mean(v) for r in Rb if (v := [x for x in (jf(r, t) for t in range(1, len(r["J"]))) if x is not None])}
        split = {}
        for name, pop in (("F0", F0), ("N0", N0)):
            ks = [k for k in pop if k in pm and k in pb]
            dd = [pm[k] - pb[k] for k in ks]
            split[name] = dict(n=len(ks), mean_dJF=statistics.mean(dd) if dd else None,
                               recovered=sum(1 for k in ks if pb[k] < 0.5 <= pm[k]), broken=sum(1 for k in ks if pm[k] < 0.5 <= pb[k]),
                               up=sum(1 for x in dd if x > 0.05), down=sum(1 for x in dd if x < -0.05))
        res["pairs_1_20"] = split
        # P3: frame-0 J == 1.0 for GT-present pairs when seeds are ground truth
        if seeds == "MVSeed_gt":
            viol = [pair_key(r) for r in Rm if r["gt"][0] and r["J"][0] is not None and abs(r["J"][0] - 1.0) > 1e-9]
            res["P3_frame0_J_is_1"] = dict(n_checked=sum(1 for r in Rm if r["gt"][0] and r["J"][0] is not None), violations=len(viol), examples=viol[:5])
        # Z1: frame-0 PNGs byte-identical to the seed folder (scored cameras, objects in the seed folder)
        same = diff = missing = 0
        for r in Rm:
            c = CFG[r["scene"]]; f = c["folder"]; s0 = c["start_frame"]
            a = os.path.join(DATA, f, seeds, r["cam"], str(s0), f"{r['obj']}.png"); b = os.path.join(DATA, f, m, r["cam"], str(s0), f"{r['obj']}.png")
            if not os.path.exists(a): continue
            if not os.path.exists(b): missing += 1; continue
            same += hashlib.sha256(open(a, "rb").read()).digest() == hashlib.sha256(open(b, "rb").read()).digest(); diff += hashlib.sha256(open(a, "rb").read()).digest() != hashlib.sha256(open(b, "rb").read()).digest()
        res["Z1_frame0_bytes"] = dict(same=same, different=diff, missing=missing)
        # Z2: manifest provenance digest == seed folder digest, per scene
        z2 = {"ok": 0, "bad": [], "absent": 0}
        for sc in sorted(sm):
            f = CFG[sc]["folder"]; man = os.path.join(DATA, f, m, "MANIFEST.json")
            if not os.path.exists(man): z2["absent"] += 1; continue
            prov = json.load(open(man)).get("provenance", {}).get("seeds_from", {})
            dig, _ = SF.png_digest(os.path.join(DATA, f, seeds))
            if prov.get("png_digest") == dig: z2["ok"] += 1
            else: z2["bad"].append(sc)
        res["Z2_provenance"] = z2
        out["runs"][rid] = res
        print(f"{rid} {m}\n   basic {res['basic']:.3f} vs base {res['base_basic']:.3f}  Δ {st['mean']:+.3f}  CI [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]  {st['wins']}↑/{st['losses']}↓  worst {st['worst'][1]} {st['worst'][0]:+.2f}  best {st['best'][1]} {st['best'][0]:+.2f}")
        for k, v in split.items(): print(f"   {k}: n={v['n']} ΔJ&F(1-20) {v['mean_dJF']:+.4f}  recovered {v['recovered']} broken {v['broken']}  up {v['up']} down {v['down']}")
        if "P3_frame0_J_is_1" in res: print(f"   P3 frame-0 J==1: {res['P3_frame0_J_is_1']}")
        print(f"   Z1 {res['Z1_frame0_bytes']}   Z2 {z2}")
    json.dump(out, open(os.path.join(REPO, "MVSeed", "runs", "e7_judge.json"), "w"), indent=1)
    print("wrote MVSeed/runs/e7_judge.json")


if __name__ == "__main__":
    main()
