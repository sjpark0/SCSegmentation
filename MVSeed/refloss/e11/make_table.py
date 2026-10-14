"""Merge coupling.json / coupling2.json / coupling3.json into the designers' table of the 59 failing pairs
(fail59.tsv) with the final decode-only vs new-information verdict, and print the verdict totals.

    stage1/.venv/bin/python make_table.py <dir>
"""
import collections
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "."
C = json.load(open(os.path.join(D, "coupling.json")))
C2 = {(r["scene"], r["cam"], r["obj"]): r for r in json.load(open(os.path.join(D, "coupling2.json")))}
C3 = json.load(open(os.path.join(D, "coupling3.json")))
U3 = {(r["scene"], r["cam"], r["obj"]): r for r in C3["under"]}
pct = lambda x: "" if x is None else f"{100 * x:.0f}"


def verdict(p, a, u):
    g, k = p["group"], p["kind"]
    if g == "under":
        mem = a["missed_in_ref_or_admitted"] or 0.0
        anyc = u["union_r1_cov"] if u else 0.0
        if mem >= 0.5:
            return "decode-only plausible", f"missed part already in the round-2 memory set ({pct(mem)}% under projected c_ini GT or an admitted neighbour's round-1 mask)"
        if anyc >= 0.5:
            return "decode-only plausible", f"missed part in some camera's round-1 mask ({pct(anyc)}% union over all cams; nearest-3 admitted only {pct(a['missed_in_admitted_nb'])}%): neighbour re-selection"
        mr = p["missed_reproj"]
        return "needs new information", (f"missed part not in c_ini ({pct(mr['outfov'])}% out of its FOV, {pct(mr['occl'])}% occluded, {pct(mr['lowconf'])}% no Pi3 geometry), "
                                         f"not in any camera's round-1 mask (union {pct(anyc)}%, nearest-3 {pct(a['missed_in_admitted_nb'])}%): only the target image can supply it")
    if g == "spill" or k == "wrong-object":
        same = a["extra_on_other_seed_same_view"] or 0.0
        refo = a["extra_on_ref_other_proj"] or 0.0
        if same >= 0.5:
            return "decode-only plausible", f"{pct(same)}% of the wrong px are also claimed by another object's seed in the same view (in-view exclusivity; objects are decoded independently)"
        if refo >= 0.5:
            return "decode-only plausible", f"{pct(refo)}% of the wrong px land on another object's reference GT when projected (reference-geometry exclusivity)"
        return "needs new information", (f"wrong px are not claimed by another seed ({pct(same)}%) nor on another object's projected reference GT ({pct(refo)}%); "
                                         f"the admitted neighbours' round-1 masks agree with the spill on {pct(a['extra_in_admitted_nb'])}%")
    if k == "miss":
        if p["gt_px"] < 50:
            return "low value", f"GT {p['gt_px']} px"
        return "needs new information", f"GT {p['gt_px']} px at vis_ratio {p['vis_ratio']}; no camera's round-1 mask covers it ({pct(a['missed_in_any_nb'])}% nearest-3)"
    return "boundary", "boundary/mixed: decode-only in principle (E9 decoder fine-tuning, E10 2x crop re-decode and every cheap post-process failed to move it)"


cols = ["scene", "cam", "obj", "label", "otype", "kind", "group", "twin", "cost", "cost_frame0", "cost_later", "max_cost", "JF", "r1_JF",
        "gt_px", "seed_px", "ref_gt_px", "vis_ratio", "recall", "precision", "frac_other", "other_obj", "border", "angle", "in_F0",
        "n_neigh_admitted", "missed_in_ref_proj%", "missed_in_admitted_nb%", "missed_in_any_cam_r1%", "extra_on_other_seed_same_view%",
        "extra_on_ref_other_proj%", "extra_in_admitted_nb%", "verdict", "why"]
out = []
tot = collections.defaultdict(lambda: [0, 0.0])
for p in C["pairs"]:
    k = (p["scene"], p["cam"], p["obj"])
    a = C2.get(k, {})
    u = U3.get(k)
    v, why = verdict(p, a, u) if a else ("boundary", "boundary/mixed: decode-only in principle (E9 decoder fine-tuning, E10 2x crop re-decode and every cheap post-process failed to move it)")
    dec = C["decomposition"]["|".join(map(str, k))]
    row = dict(scene=p["scene"], cam=p["cam"], obj=p["obj"], label=p["label"], otype=p["otype"], kind=p["kind"], group=p["group"], twin=p["twin"],
               cost=f"{p['cost']:+.4f}", cost_frame0=f"{p['cost_frame0']:+.4f}", cost_later=f"{p['cost_later']:+.4f}", max_cost=f"{dec['w'] * dec['n_frames']:.4f}",
               JF=f"{p['JF']:.3f}", r1_JF=f"{p['r1_JF']:.3f}", gt_px=p["gt_px"], seed_px=p["seed_px"], ref_gt_px=p["ref_gt_px"], vis_ratio=p["vis_ratio"],
               recall=p["recall"], precision=p["precision"], frac_other=p["frac_other"], other_obj=p["other_obj"], border=p["border"], angle=p["angle"],
               in_F0=p["in_F0"], n_neigh_admitted=p["n_neigh_admitted"])
    row["missed_in_ref_proj%"] = pct(a.get("missed_in_ref_proj")) if a else ""
    row["missed_in_admitted_nb%"] = pct(a.get("missed_in_admitted_nb")) if a else ""
    row["missed_in_any_cam_r1%"] = pct(u["union_r1_cov"]) if u else ""
    row["extra_on_other_seed_same_view%"] = pct(a.get("extra_on_other_seed_same_view")) if a else ""
    row["extra_on_ref_other_proj%"] = pct(a.get("extra_on_ref_other_proj")) if a else ""
    row["extra_in_admitted_nb%"] = pct(a.get("extra_in_admitted_nb")) if a else ""
    row["verdict"], row["why"] = v, why
    out.append(row)
    tot[(p["group"], v)][0] += 1
    tot[(p["group"], v)][1] += p["cost"]
with open(os.path.join(D, "fail59.tsv"), "w") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in out:
        fh.write("\t".join(str(r[c]) for c in cols) + "\n")
print("verdict totals (group, verdict): n, cost")
for (g, v), (n, c) in sorted(tot.items(), key=lambda x: -x[1][1]):
    print(f"   {g:6s} {v:24s} n={n:2d} cost {c:+.4f}")
vt = collections.defaultdict(lambda: [0, 0.0])
for (g, v), (n, c) in tot.items():
    vt[v][0] += n
    vt[v][1] += c
print("verdict totals:", {v: (n, round(c, 4)) for v, (n, c) in sorted(vt.items(), key=lambda x: -x[1][1])})
print("compact list (cost desc): scene cam obj label kind cost gt_px vis twin verdict")
for r in out:
    print(f"   {r['scene'][:14]:14s} {r['cam']:12s} {r['obj']:3d} {str(r['label'])[:16]:16s} {r['kind'][:26]:26s} {r['cost']} {r['gt_px']:8d} {r['vis_ratio']:5.2f} {'T' if r['twin'] else '.'} {r['verdict']}")
print("wrote", os.path.join(D, "fail59.tsv"))
