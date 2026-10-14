#!/usr/bin/env python3
"""Summarise extent_vis.json (host: stage1/.venv/bin/python extent_vis_report.py extent_vis.json).

Tables: (1) per under-cover failing pair (22) — missed-pixel split, J&F oracles, neighbour knowledge;
(2) totals for the 22 under-cover failing pairs, the 59 failing pairs and all 664 pairs (pixel-weighted and
pair-mean); (3) the GT-free unions as candidate post-processes (mean ΔJ&F over 664 and up/down counts)."""
import json
import sys

import numpy as np

src = open(sys.argv[1]).read()
d = json.loads(src[src.index("{"):])
rows = d["rows"]
meta = d["meta"]
print(f"meta: {meta}\n")

VARS = ["seed", "seed_or_missvis", "seed_or_missnovis", "seed_or_gt", "seed_or_miss_onrefgt",
        "seed_or_miss_nb_all", "seed_or_miss_nb_k3", "free_seed_or_refproj", "free_seed_or_nb_k3", "free_seed_or_nb_all",
        "free_seed_or_refproj_or_nb_k3"]


def frac(a, b):
    return a / b if b else float("nan")


def best_nb(r):
    m = r["missed_px"]
    items = sorted(r["nb"].items(), key=lambda kv: -kv[1]["cover"])
    return [(nb, frac(v["cover"], m), nb in r["k3"]) for nb, v in items[:3]]


def group(name, sel):
    n = len(sel)
    if n == 0:
        return
    M = sum(r["missed_px"] for r in sel)
    keys = ["missed_vis", "missed_outside", "missed_occluded", "missed_nodepth", "missed_on_refgt",
            "missed_nb_any_vis", "missed_nb_union_all", "missed_nb_union_k3"]
    tot = {k: sum(r[k] for r in sel) for k in keys}
    pm = {k: float(np.nanmean([frac(r[k], r["missed_px"]) for r in sel if r["missed_px"] > 0])) for k in keys}
    print(f"== {name}: {n} pairs, missed px total {M:,} (GT px {sum(r['gt_px'] for r in sel):,}; "
          f"stage-2 cost {sum(r['cost'] for r in sel):.3f})")
    print("   fraction of MISSED pixels          pixel-weighted   mean of per-pair fractions")
    for k, lab in [("missed_vis", "(a) visible from reference"), ("missed_outside", "(b1) outside ref image / behind"),
                   ("missed_occluded", "(b2) occluded for the reference"), ("missed_nodepth", "(b3) no valid Pi3 depth"),
                   ("missed_on_refgt", "    (a) and lands on ref GT of same obj"),
                   ("missed_nb_any_vis", "visible from >= 1 other non-ref camera"),
                   ("missed_nb_union_all", "covered by some neighbour's reprojected seed (any of the other cams)"),
                   ("missed_nb_union_k3", "covered by a k=3-nearest neighbour's reprojected seed")]:
        print(f"   {lab:<66s} {frac(tot[k], M):7.3f}        {pm[k]:7.3f}")
    nv = tot["missed_outside"] + tot["missed_occluded"] + tot["missed_nodepth"]
    print(f"   (b) not visible total: pixel-weighted {frac(nv, M):.3f}, pair-mean "
          f"{float(np.nanmean([frac(r['missed_outside'] + r['missed_occluded'] + r['missed_nodepth'], r['missed_px']) for r in sel if r['missed_px'] > 0])):.3f}")
    print(f"   of the ref-visible missed pixels, share landing on the ref GT of the same object: pixel-weighted "
          f"{frac(tot['missed_on_refgt'], tot['missed_vis']):.3f}")
    if "hit_px" in sel[0]:
        H = sum(r["hit_px"] for r in sel)
        print(f"   CONTROL on the HIT region (GT & seed, {H:,} px): same reprojection, same neighbours -> pixel-weighted / pair-mean")
        for k, lab in [("hit_vis", "visible from reference"), ("hit_on_refgt", "visible and on ref GT of same obj"),
                       ("hit_nb_any_vis", "visible from >= 1 other non-ref camera"),
                       ("hit_nb_union_all", "covered by union of all neighbours' reprojected seeds"),
                       ("hit_nb_union_k3", "covered by union of k=3 nearest neighbours' reprojected seeds")]:
            pmh = float(np.nanmean([frac(r[k], r["hit_px"]) for r in sel if r["hit_px"] > 0]))
            print(f"      {lab:<63s} {frac(sum(r[k] for r in sel), H):7.3f}        {pmh:7.3f}")
    print("   mean J&F (over the group) :", end="")
    base = np.mean([r["jf"]["seed"]["JF"] for r in sel])
    for v in VARS:
        m = np.mean([r["jf"][v]["JF"] for r in sel])
        print(f"\n      {v:<32s} {m:.4f}  (Δ {m - base:+.4f})", end="")
    print("\n   mean J  (over the group) :", end="")
    baseJ = np.mean([r["jf"]["seed"]["J"] for r in sel])
    for v in VARS:
        m = np.mean([r["jf"][v]["J"] for r in sel])
        print(f"\n      {v:<32s} {m:.4f}  (Δ {m - baseJ:+.4f})", end="")
    print("\n")


uc = [r for r in rows if r["failing"] and r["kind"].startswith("under-cover")]
fail = [r for r in rows if r["failing"]]
good = [r for r in rows if not r["failing"]]
assert len(rows) == 664 and len(fail) == 59 and len(uc) == 22, (len(rows), len(fail), len(uc))

print("=== per under-cover failing pair (sorted by stage-2 cost) ===")
print("scene cam obj label kind | cost JF | missed px: vis / outside / occl / nodepth / on-refGT | "
      "J&F: +a +b +both +onrefGT | nb: any-vis union-all union-k3 best3 (cam:frac,*=k3) | oracle +nb_all +nb_k3 | free: refproj nb_k3")
for r in sorted(uc, key=lambda r: -r["cost"]):
    m = r["missed_px"]
    j = r["jf"]
    b3 = " ".join(f"{nb}:{f:.2f}{'*' if k else ''}" for nb, f, k in best_nb(r))
    print(f"{r['scene']} {r['cam']} {r['obj']} {r['label']} {r['kind']} | {r['cost']:.3f} {r['JF']:.3f} | "
          f"{m:,}: {frac(r['missed_vis'], m):.2f} / {frac(r['missed_outside'], m):.2f} / {frac(r['missed_occluded'], m):.2f} / "
          f"{frac(r['missed_nodepth'], m):.2f} / {frac(r['missed_on_refgt'], m):.2f} | "
          f"{j['seed_or_missvis']['JF']:.3f} {j['seed_or_missnovis']['JF']:.3f} {j['seed_or_gt']['JF']:.3f} {j['seed_or_miss_onrefgt']['JF']:.3f} | "
          f"{frac(r['missed_nb_any_vis'], m):.2f} {frac(r['missed_nb_union_all'], m):.2f} {frac(r['missed_nb_union_k3'], m):.2f} [{b3}] | "
          f"{j['seed_or_miss_nb_all']['JF']:.3f} {j['seed_or_miss_nb_k3']['JF']:.3f} | "
          f"{j['free_seed_or_refproj']['JF']:.3f} {j['free_seed_or_nb_k3']['JF']:.3f}")
print()
print("=== other failing pairs (37) — same columns ===")
for r in sorted([r for r in fail if r not in uc], key=lambda r: -r["cost"]):
    m = r["missed_px"]
    j = r["jf"]
    b3 = " ".join(f"{nb}:{f:.2f}{'*' if k else ''}" for nb, f, k in best_nb(r))
    print(f"{r['scene']} {r['cam']} {r['obj']} {r['label']} {r['kind']} | {r['cost']:.3f} {r['JF']:.3f} | "
          f"{m:,}: {frac(r['missed_vis'], m):.2f} / {frac(r['missed_outside'], m):.2f} / {frac(r['missed_occluded'], m):.2f} / "
          f"{frac(r['missed_nodepth'], m):.2f} / {frac(r['missed_on_refgt'], m):.2f} | "
          f"{j['seed_or_missvis']['JF']:.3f} {j['seed_or_missnovis']['JF']:.3f} {j['seed_or_gt']['JF']:.3f} {j['seed_or_miss_onrefgt']['JF']:.3f} | "
          f"{frac(r['missed_nb_any_vis'], m):.2f} {frac(r['missed_nb_union_all'], m):.2f} {frac(r['missed_nb_union_k3'], m):.2f} [{b3}] | "
          f"{j['seed_or_miss_nb_all']['JF']:.3f} {j['seed_or_miss_nb_k3']['JF']:.3f} | "
          f"{j['free_seed_or_refproj']['JF']:.3f} {j['free_seed_or_nb_k3']['JF']:.3f}")
print()

group("under-cover failing (22)", uc)
group("under-cover failing without Dog camera_0004 tree (21; that pair is 40% of the 22-pair missed mass, 91% no-depth)",
      [r for r in uc if not (r["scene"] == "Dog" and r["cam"] == "camera_0004" and r["obj"] == 3)])
group("under-cover beyond c_ini failing (9)", [r for r in uc if r["kind"] == "under-cover(beyond c_ini)"])
group("under-cover within c_ini failing (13)", [r for r in uc if r["kind"] == "under-cover"])
group("all failing (59)", fail)
group("good seeds (605)", good)
group("all 664", rows)

print("=== GT-free unions as post-processes, all 664 pairs: ΔJ&F vs seed, pairs up / down by > 0.01 ===")
for v in ("free_seed_or_refproj", "free_seed_or_nb_k3", "free_seed_or_nb_all", "free_seed_or_refproj_or_nb_k3"):
    dd = np.array([r["jf"][v]["JF"] - r["jf"]["seed"]["JF"] for r in rows])
    du = np.array([r["jf"][v]["JF"] - r["jf"]["seed"]["JF"] for r in uc])
    print(f"   {v:<32s} all664 mean Δ {dd.mean():+.4f}  up {int((dd > 0.01).sum())} / down {int((dd < -0.01).sum())}; "
          f"under-cover-22 mean Δ {du.mean():+.4f} up {int((du > 0.01).sum())} / down {int((du < -0.01).sum())}")
print()
print("=== neighbour 'knows' summary over the 22 under-cover failing pairs ===")
for th in (0.5, 0.8):
    n_all = sum(1 for r in uc if frac(r["missed_nb_union_all"], r["missed_px"]) >= th)
    n_k3 = sum(1 for r in uc if frac(r["missed_nb_union_k3"], r["missed_px"]) >= th)
    n_one = sum(1 for r in uc if best_nb(r)[0][1] >= th)
    n_vis = sum(1 for r in uc if frac(r["missed_nb_any_vis"], r["missed_px"]) >= th)
    print(f"   >= {th:.0%} of the missed pixels: visible from some other cam {n_vis}/22; covered by union of all neighbours' seeds {n_all}/22; "
          f"by union of k=3 nearest {n_k3}/22; by one single neighbour {n_one}/22")
print()
print("=== reference-GT reprojection sanity (project_reference_masks vs the target GT), all 664: "
      f"recall mean {np.mean([r['refproj_recall'] for r in rows]):.3f}, precision mean {np.mean([r['refproj_precision'] for r in rows]):.3f}")
