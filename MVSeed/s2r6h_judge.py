#!/usr/bin/env python3
"""S2-R6H judge (docs/stage2-R6H-prereg.md sections 3-7).  Host python (cv2, numpy):

  --check   watch values Z1-Z3 on the snapped folders (GT-free except Z3, which reads only the frame-0 GT = the prompt)
  --judge   E8-constant verdict REPLICATES / HARMS / DOES NOT REPLICATE on Data/MVSeg/jf_s2r6h.json (+ Z4, descriptives)
Both merge into MVSeed/runs/s2r6h_judge.json.
"""
import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data", "MVSeg")
OUT = os.path.join(REPO, "MVSeed", "runs", "s2r6h_judge.json")
PRIMARY, DESCRIPTIVE = "SegMaskSam3XW1CGPS4MSdgtFsCs10kG2", "SegMaskSam3XW0MSdgtFs"
SUFFIX = "Rh"
PRECOUNT = dict(AlexaMeadeExhibit=1050, AlexaMeadeFacePaint=229, Barn=689, Blocks=229, Breakfast=145, CBABasketball=145,
                Carpark=272, CoffeeMartini=1352, Dog=181, Fencing=54, FlameSteak=1105, Frog=0, MATF=392, MartialArts=165,
                Painter=346, PoznanStreet=171, Welder=275)          # prereg Z1 (primary); total 6 800
PRECOUNT_TOTAL = {PRIMARY: 6800, DESCRIPTIVE: 6489}
HELDOUT_FRAME0 = 662                                                # prereg Z3
DEFENCE = -1.0
DESIGN_PER_VIEW = 1.094                                             # 3 x 0.3645 (E8 pair, c_ini)
DESIGN_PER_SNAP = 2.45


def _j(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save(key, val):
    out = _j(OUT) if os.path.isfile(OUT) else {}
    out["prereg"] = "docs/stage2-R6H-prereg.md"
    out[key] = val
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)


def _sha(p):
    import hashlib
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _pngs(d):
    out = []
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.endswith(".png"):
                out.append(os.path.relpath(os.path.join(root, fn), d))
    return sorted(out)


def check_method(method):
    import cv2
    import numpy as np
    sys.path.insert(0, os.path.join(REPO, "MVSeed"))
    import e7_judge as E7
    res = dict(method=method, scenes={}, bad=[])
    total = 0
    for sc in sorted(E7.CEN):
        f, start = E7.CFG[sc]["folder"], E7.CFG[sc]["start_frame"]
        src, dst = os.path.join(DATA, f, method), os.path.join(DATA, f, method + SUFFIX)
        man = _j(os.path.join(dst, "MANIFEST.json"))
        snapped = {f"{cam}/{x}" if not x.startswith(cam + "/") else x
                   for cam, st in man["per_cam"].items() for x in st["snapped_files"]}
        n = len(snapped)
        total += n
        r = dict(snapped=n)
        # Z1
        if method == PRIMARY and n != PRECOUNT[sc]:
            res["bad"].append(("Z1", sc, n, PRECOUNT[sc]))
        # Z2: changed files == snapped list, each == its frame-0 PNG; the rest hard links; file sets equal; src MANIFEST unchanged
        a, b = _pngs(src), _pngs(dst)
        if a != b:
            res["bad"].append(("Z2-fileset", sc, len(a), len(b)))
        changed = set()
        for rel in b:
            if not os.path.samefile(os.path.join(src, rel), os.path.join(dst, rel)):
                changed.add(rel)
        if changed != snapped:
            res["bad"].append(("Z2-changed!=snapped", sc, len(changed), n, sorted(changed ^ snapped)[:5]))
        for rel in snapped:
            cam, t, o = rel.split("/")
            if _sha(os.path.join(dst, rel)) != _sha(os.path.join(dst, cam, str(start), o)):
                res["bad"].append(("Z2-not-frame0", sc, rel))
        smf = os.path.join(src, "MANIFEST.json")
        if man["source"]["manifest_sha256"] != (_sha(smf) if os.path.isfile(smf) else None):
            res["bad"].append(("Z2-src-manifest", sc))
        if set(man["per_cam"]) & {E7.CEN[sc]["c_ini"]}:
            res["bad"].append(("Z2-c_ini-touched", sc))
        # Z3: held-out frame-0 PNGs == GT == obj (reads only the frame-0 GT, the prompt)
        z3 = 0
        for cam in man["heldout"]:
            gt = cv2.imread(os.path.join(DATA, f, "Mask", cam, "%06d.png" % start), cv2.IMREAD_GRAYSCALE)
            d0 = os.path.join(dst, cam, str(start))
            for fn in sorted(os.listdir(d0)):
                m = cv2.imread(os.path.join(d0, fn), cv2.IMREAD_GRAYSCALE) > 127
                if not np.array_equal(m, gt == int(fn[:-4])):
                    res["bad"].append(("Z3", sc, cam, fn))
                z3 += 1
        r["frame0_pngs"] = z3
        res["scenes"][sc] = r
    res["total_snapped"] = total
    res["frame0_pngs"] = sum(r["frame0_pngs"] for r in res["scenes"].values())
    if total != PRECOUNT_TOTAL[method]:
        res["bad"].append(("Z1-total", total, PRECOUNT_TOTAL[method]))
    if method == PRIMARY and res["frame0_pngs"] != HELDOUT_FRAME0:
        res["bad"].append(("Z3-count", res["frame0_pngs"], HELDOUT_FRAME0))
    res["ok"] = not res["bad"]
    return res


def check():
    out = {m: check_method(m) for m in (PRIMARY, DESCRIPTIVE)}
    out["ok"] = all(v["ok"] for v in out.values())
    _save("check", out)
    for m in (PRIMARY, DESCRIPTIVE):
        v = out[m]
        print(f"{m}: snapped {v['total_snapped']} (pre-count {PRECOUNT_TOTAL[m]}), held-out frame-0 PNGs {v['frame0_pngs']}, "
              f"bad {len(v['bad'])} {v['bad'][:5]}")
    print(f"check: {'OK' if out['ok'] else 'FAIL'}")
    if not out["ok"]:
        sys.exit(2)


def judge():
    import cv2
    sys.path.insert(0, os.path.join(REPO, "MVSeed"))
    import e7_judge as E7
    out_all = _j(OUT) if os.path.isfile(OUT) else {}
    if not out_all.get("check", {}).get("ok"):
        raise SystemExit("watch values Z1-Z3 not OK (run --check first): no verdict (prereg section 3)")
    d = E7.load(["jf_s2r6h.json"])
    res = {}
    for m in (PRIMARY, DESCRIPTIVE):
        arm = m + SUFFIX
        Rb, Ra = list(E7.rows(d, m)), list(E7.rows(d, arm))
        sb, sa = E7.scene_scores(Rb), E7.scene_scores(Ra)
        cb, ca = E7.scene_scores(list(E7.rows(d, m, "complete"))), E7.scene_scores(list(E7.rows(d, arm, "complete")))
        n17 = len(sb) == len(sa) == len(cb) == len(ca) == 17
        st, stc = E7.paired(sb, sa), E7.paired(cb, ca)
        # Z4: c_ini rows identical, Frog delta exactly 0
        kb = {(r["scene"], r["cam"], r["obj"]): r for r in Rb}
        z4_cini = sum(1 for r in Ra if r["ref"] and (r["J"] != kb[(r["scene"], r["cam"], r["obj"])]["J"]
                                                    or r["F"] != kb[(r["scene"], r["cam"], r["obj"])]["F"]))
        z4 = dict(cini_rows_differ=z4_cini, frog_delta=(sa.get("Frog", 0) - sb.get("Frog", 0)))
        z4["ok"] = z4_cini == 0 and z4["frog_delta"] == 0
        worst = st["worst"][0]
        verdict = None
        if m == PRIMARY:
            if not (n17 and z4["ok"]):
                verdict = "NOT JUDGED"
            elif st["mean"] > 0 and st["ci"][0] > 0 and worst >= DEFENCE and stc["mean"] >= 0:
                verdict = "REPLICATES"
            elif st["ci"][1] < 0:
                verdict = "HARMS"
            else:
                verdict = "DOES NOT REPLICATE"
        # per-snap and GT-identity descriptives
        man_snaps = {}
        for sc in E7.CEN:
            man = _j(os.path.join(DATA, E7.CFG[sc]["folder"], arm, "MANIFEST.json"))
            for cam, s in man["per_cam"].items():
                for x in s["snapped_files"]:
                    c, t, o = x.split("/")
                    man_snaps.setdefault((sc, c, int(o[:-4])), set()).add(int(t))
        ka = {(r["scene"], r["cam"], r["obj"]): r for r in Ra}
        per_snap, ident_d, nonident_d = [], [], []
        gt_cache = {}
        for key, ts in man_snaps.items():
            if key not in ka or key not in kb:
                continue
            sc, cam, o = key
            start = E7.CFG[sc]["start_frame"]
            for t in sorted(ts):
                i = t - start
                ja, jb = E7.jf(ka[key], i), E7.jf(kb[key], i)
                if ja is None or jb is None:
                    continue
                dlt = 100 * (ja - jb)
                per_snap.append(dlt)
                gk0, gkt = (sc, cam, start), (sc, cam, t)
                for gk in (gk0, gkt):
                    if gk not in gt_cache:
                        gt_cache[gk] = cv2.imread(os.path.join(DATA, E7.CFG[sc]["folder"], "Mask", cam, "%06d.png" % gk[2]),
                                                  cv2.IMREAD_GRAYSCALE)
                same = bool(((gt_cache[gk0] == o) == (gt_cache[gkt] == o)).all())
                (ident_d if same else nonident_d).append(dlt)
            if len(gt_cache) > 400:
                gt_cache.clear()
        tot = sum(per_snap)
        res[m] = dict(arm=arm, n_scenes=(len(sb), len(sa), len(cb), len(ca)), basic=statistics.mean(sa.values()),
                      base_basic=statistics.mean(sb.values()), paired=st, complete=statistics.mean(ca.values()),
                      base_complete=statistics.mean(cb.values()), paired_complete=stc, worst_unrounded=worst, Z4=z4,
                      verdict=verdict, per_view_delta=1.5 * st["mean"], per_view_ci=[1.5 * x for x in st["ci"]],
                      size_reading=("smaller than on the design camera" if 1.5 * st["ci"][1] < DESIGN_PER_VIEW else "at design size"),
                      per_snap=dict(n=len(per_snap), mean=statistics.mean(per_snap) if per_snap else None),
                      gt_identical=dict(n=len(ident_d), sum=sum(ident_d), share=(sum(ident_d) / tot) if tot else None,
                                        mean=statistics.mean(ident_d) if ident_d else None),
                      gt_nonidentical=dict(n=len(nonident_d), sum=sum(nonident_d),
                                           mean=statistics.mean(nonident_d) if nonident_d else None))
    p = res[PRIMARY]
    st = p["paired"]
    up = sum(1 for s, v in st["per_scene"].items() if s != "Frog" and v > 0)
    res["predictions"] = dict(
        P1=dict(basic=st["mean"], in_80=0.25 <= st["mean"] <= 0.80, per_view=p["per_view_delta"]),
        P2=dict(replicates=p["verdict"] == "REPLICATES", up_of_16=up, up_ok=up >= 13, worst=p["worst_unrounded"], worst_ok=p["worst_unrounded"] >= -0.3),
        P3=dict(ratio=(p["paired_complete"]["mean"] / st["mean"]) if st["mean"] else None,
                ok=bool(st["mean"]) and 0.7 <= p["paired_complete"]["mean"] / st["mean"] <= 1.1),
        P4=dict(per_snap=p["per_snap"]["mean"], ok=p["per_snap"]["mean"] is not None and 1.0 <= p["per_snap"]["mean"] <= 2.6),
        P5=dict(identical_share=p["gt_identical"]["share"], ok=p["gt_identical"]["share"] is not None and 0.5 <= p["gt_identical"]["share"] <= 0.8))
    _save("judge", res)
    for m in (PRIMARY, DESCRIPTIVE):
        r = res[m]; s = r["paired"]; c = r["paired_complete"]
        print(f"{r['arm']} vs {m}{'' if m == PRIMARY else '  (descriptive)'}\n   basic {r['base_basic']:.3f} -> {r['basic']:.3f}  "
              f"Δ {s['mean']:+.3f} CI [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  {s['wins']}↑/{s['losses']}↓  worst {s['worst'][1]} {s['worst'][0]:+.3f}"
              f"\n   complete {r['base_complete']:.3f} -> {r['complete']:.3f}  Δ {c['mean']:+.3f} CI [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
              f"\n   per view Δ {r['per_view_delta']:+.3f} [{r['per_view_ci'][0]:+.3f}, {r['per_view_ci'][1]:+.3f}] ({r['size_reading']}; design +{DESIGN_PER_VIEW})"
              f"\n   per snap {r['per_snap']}  GT-identical {r['gt_identical']}  non-identical {r['gt_nonidentical']}\n   Z4 {r['Z4']}  per scene {s['per_scene']}")
    print(f"\npredictions {json.dumps(res['predictions'])}\nVERDICT (registered, primary arm): {p['verdict']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    check() if a.check else judge()
