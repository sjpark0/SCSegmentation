"""Host report of null_compose.json: for each null E, the add-only composition C0 + (E & INV & U, continuation-only)
against C0 — scene-cluster bootstrap (mvseg2d.paired, 10 000, seed 0) on J, F, J&F; splits good605 / fail59 / under22;
pairs down/up > 0.01; worst scene; added pixels (TP / FP by class); and the RAW E (uncomposed) for contrast.
    PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python -I null_report.py null_compose.json"""
import json
import os
import statistics as st
import sys

REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from mvseg2d.paired import paired_scene_bootstrap, scene_means  # noqa: E402

s = open(sys.argv[1]).read()
s = s[s.index("{"):]
d = json.loads(s)
rows = d["rows"]
nulls = d["meta"]["nulls"]
print("meta:", d["meta"])


def rows_for(name, raw=False):
    out = []
    for r in rows:
        x = r[name]
        J, F = (x["raw_J"], x["raw_F"]) if raw else (x["J"], x["F"])
        out.append({"scene": r["scene"], "cam": r["cam"], "obj": r["obj"], "J": J, "F": F, "JF": 0.5 * (J + F)})
    return out


def groups(r):
    g = ["all664"]
    if r["JF_e8"] >= 0.8:
        g.append("good605")
    else:
        g.append("fail59")
        if r["kind"].startswith("under"):
            g.append("under22")
    return g


r0 = rows_for("C0")
k0 = {(r["scene"], r["cam"], r["obj"]): r for r in r0}
m0 = scene_means(r0, "JF")
print(f"C0 re-scored: scene-mean J&F {st.mean(m0.values()):.4f} (E11 judge: 0.9148)")
for name in nulls:
    for raw in (False, True):
        rs = rows_for(name, raw)
        kk = {(r["scene"], r["cam"], r["obj"]): r for r in rs}
        b = {k: paired_scene_bootstrap(r0, rs, key=k) for k in ("J", "F", "JF")}
        worst = min(b["JF"]["per_scene"].items(), key=lambda x: x[1])
        dd = {k: kk[k]["JF"] - k0[k]["JF"] for k in k0}
        tag = "RAW  " if raw else "COMP "
        print(f"\n== {tag} {name}: dJ&F {b['JF']['delta']:+.4f} [{b['JF']['ci'][0]:+.4f}, {b['JF']['ci'][1]:+.4f}]  "
              f"dJ {b['J']['delta']:+.4f} [{b['J']['ci'][0]:+.4f}, {b['J']['ci'][1]:+.4f}]  dF {b['F']['delta']:+.4f} [{b['F']['ci'][0]:+.4f}, {b['F']['ci'][1]:+.4f}]  "
              f"worst {worst[0]} {worst[1]:+.4f}  down>0.01 {sum(v < -0.01 for v in dd.values())}  up>0.01 {sum(v > 0.01 for v in dd.values())}  "
              f"G1 {'pass' if b['JF']['ci'][0] > 0 else 'fail'}  G2 {'pass' if worst[1] >= -0.01 else 'fail'}")
        by = {}
        for r in rows:
            k = (r["scene"], r["cam"], r["obj"])
            for g in groups(r):
                by.setdefault(g, []).append(dd[k])
        print("   splits dJ&F pair-mean: " + "  ".join(f"{g} {st.mean(v):+.4f} (n {len(v)})" for g, v in by.items()))
        if not raw:
            tot = {}
            for r in rows:
                x = r[name]
                for g in groups(r):
                    t = tot.setdefault(g, {})
                    for key in ("add_raw", "add", "add_tp", "add_fp", "add_fp_outside", "add_fp_occluded", "add_fp_nodepth",
                                "add_tp_outside", "add_tp_occluded", "add_tp_nodepth", "ext_extra_inv", "ext_extra_vis", "c0_px"):
                        t[key] = t.get(key, 0) + (x[key] if key != "c0_px" else r["c0_px"])
            for g, t in tot.items():
                prec = t["add_tp"] / max(t["add"], 1)
                print(f"   {g:8s} E-beyond-C0: inv {t['ext_extra_inv']:,} vis {t['ext_extra_vis']:,} | add raw {t['add_raw']:,} -> kept {t['add']:,} "
                      f"(= {t['add'] / max(t['c0_px'], 1) * 100:.2f} % of C0 px) | TP {t['add_tp']:,} FP {t['add_fp']:,} precision {prec:.3f} | "
                      f"FP out/occ/nod {t['add_fp_outside']:,}/{t['add_fp_occluded']:,}/{t['add_fp_nodepth']:,} | TP out/occ/nod {t['add_tp_outside']:,}/{t['add_tp_occluded']:,}/{t['add_tp_nodepth']:,}")
            # per-scene table for the composed arm
            print("   per-scene dJ&F: " + "  ".join(f"{sc} {v:+.4f}" for sc, v in sorted(b["JF"]["per_scene"].items(), key=lambda x: x[1])))
            # under-cover pairs individually
            print("   under22 per pair (dJ&F, add TP/FP):")
            for r in rows:
                if "under22" in groups(r):
                    x = r[name]
                    print(f"      {r['scene']:20s} {r['cam']:12s} {r['obj']:3d} C0 {0.5 * (r['C0']['J'] + r['C0']['F']):.3f} "
                          f"d {dd[(r['scene'], r['cam'], r['obj'])]:+.4f}  add {x['add']:,} TP {x['add_tp']:,} FP {x['add_fp']:,}  raw-E JF {0.5 * (x['raw_J'] + x['raw_F']):.3f}")
