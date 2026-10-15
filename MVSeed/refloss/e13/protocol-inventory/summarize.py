#!/usr/bin/env python3
"""Protocol inventory, part 3: summarise dev_reach.json (DEV rows only) into reach tables.

    python summarize.py dev_reach.json availability.json > dev_reach_report.txt
"""
import collections
import json
import sys

import numpy as np

CLS = ("vis", "outside", "occluded", "nodepth")
R = (1, 2, 3)
# keyframe class mix of the 22 under-cover failing pairs' missed pixels, QUOTED from the existing
# MVSeed/refloss/e11/extent_vis_report.txt (pixel-weighted; not recomputed here)
KF = {"all22": (1325193, {"vis": 0.091, "outside": 0.366, "occluded": 0.160, "nodepth": 0.384}),
      "21_noDogTree": (798194, {"vis": 0.149, "outside": 0.591, "occluded": 0.227, "nodepth": 0.033})}
# the 22 under-cover (scene, cam, obj) of the E11 report -- used only to pick the SAME object/camera on DEV frames
UC22 = [("Dog", "camera_0004", 3), ("Dog", "camera_0004", 4), ("CBABasketball", "v24", 20), ("CBABasketball", "v06", 20),
        ("Welder", "camera_0003", 6), ("Breakfast", "v9", 8), ("PoznanStreet", "v8", 18), ("PoznanStreet", "v0", 24),
        ("MATF", "S1_CAM_1", 11), ("FlameSteak", "cam01", 32), ("Barn", "v10", 23), ("CoffeeMartini", "cam10", 12),
        ("MartialArts", "v13", 11), ("CoffeeMartini", "cam02", 29), ("MartialArts", "v13", 15), ("CoffeeMartini", "cam10", 16),
        ("Painter", "v15", 19), ("PoznanStreet", "v8", 13), ("Breakfast", "v9", 29), ("Fencing", "v9", 3),
        ("FlameSteak", "cam10", 5), ("MATF", "S1_CAM_1", 21)]

d = json.load(open(sys.argv[1]))
av = json.load(open(sys.argv[2]))["scenes"]
rows = d["rows"]
names = {s: collections.Counter(av[s]["label_names"]) for s in av}
meta = {s: json.load(open(f"/home/sjpark/Documents/SCSegmentation/stage1/Data/{s}/meta.json")) for s in av}


def dev_sub(r):
    """the dev-route agent's DEV subset: non-ref train frames [3::4] (frames ref+4, +8, ..., +20)."""
    m = meta[r["scene"]]
    dev = [t for t in m["train_frames"] if t != m["ref_frame"]]
    return r["t"] in dev[3::4]


def agg(rs):
    A = collections.defaultdict(lambda: np.zeros(4))
    for r in rs:
        for k in ("n", "B_vis", "B_ongt", "any_vis", "nonann_vis", "fwd_hit", "fwd_said"):
            A[k] += np.array(r[k], float)
        for k in ("occ_dyn", "occ_self", "occ_unl", "occ_tvis", "nod_tval", "so_dyn", "so_self", "so_stat", "tvis_dyn", "tvis_self", "tvis_stat"):
            A[k][0] += r[k]
        uniq = names[r["scene"]][r["label"]] == 1
        A["uniq"] += np.array(r["n"], float) * uniq
        A["twin_x"] += np.array(r["n"], float) * r["same_label_in_x"]
        A["twin_c"] += np.array(r["n"], float) * r["same_label_in_c"]
        A["stat_occ"][0] += r["n"][2] if r["type"] in ("static", "env") else 0
        A["absent_c"] += np.array(r["n"], float) * (not r["in_cini_t"])
    return A


def fmt_block(title, rs, out):
    A = agg(rs)
    n = A["n"]
    G = n.sum()
    Rn = n[1:].sum()
    out.append(f"== {title}: rows {len(rs)}, target GT px {G:,.0f}; beyond c_ini R = {Rn:,.0f} ({Rn / max(G, 1):.3f} of GT): "
               + " / ".join(f"{c} {n[i] / max(G, 1):.3f}" for i, c in enumerate(CLS)))
    out.append("   share of R reached, by class           outside   occluded   nodepth  |  R total  (vis class for reference)")

    def line(name, key, note=""):
        v = A[key]
        tot = v[1:].sum() / max(Rn, 1)
        per = [v[i] / max(n[i], 1) for i in range(4)]
        out.append(f"   {name:38s} {per[1]:8.3f} {per[2]:10.3f} {per[3]:9.3f}  |  {tot:6.3f}   ({per[0]:.3f}) {note}")
    line("3rd annotated cam B: visible", "B_vis")
    line("3rd annotated cam B: lands on B's GT o", "B_ongt")
    line(">=1 other rig cam visible (all)", "any_vis")
    line(">=1 non-annotated rig cam visible", "nonann_vis")
    line("c_ini GT forward-splat lands on o", "fwd_hit")
    fs = A["fwd_said"]
    out.append(f"   {'  forward-splat precision (o said -> o)':38s} " + "  ".join(f"{A['fwd_hit'][i] / max(fs[i], 1):8.3f}" for i in (1, 2, 3))
               + f"  |  {A['fwd_hit'][1:].sum() / max(fs[1:].sum(), 1):6.3f}   ({A['fwd_hit'][0] / max(fs[0], 1):.3f})")
    line("label unique in scene (name prompt)", "uniq")
    line("same-label twin present in X at t", "twin_x")
    line("same-label twin present in c_ini at t", "twin_c")
    occ = max(n[2], 1)
    out.append(f"   occluded px by occluder (c_ini label at the reprojected spot): dynamic {A['occ_dyn'][0] / occ:.3f}, "
               f"self {A['occ_self'][0] / occ:.3f}, unlabelled {A['occ_unl'][0] / occ:.3f}, "
               f"other static/env {1 - (A['occ_dyn'][0] + A['occ_self'][0] + A['occ_unl'][0]) / occ:.3f}")
    so = max(A["stat_occ"][0], 1)
    out.append(f"   occluded px of static/env objects visible from c_ini at >= 1 other DEV frame: {A['occ_tvis'][0] / so:.3f} "
               f"of those ({A['occ_tvis'][0] / occ:.3f} of all occluded; static/env are {A['stat_occ'][0] / occ:.3f} of occluded)")
    out.append(f"   ... split by occluder (static/env objects): dynamic occluder {A['tvis_dyn'][0] / max(A['so_dyn'][0], 1):.3f} of {A['so_dyn'][0]:,.0f} px; "
               f"self {A['tvis_self'][0] / max(A['so_self'][0], 1):.3f} of {A['so_self'][0]:,.0f}; other static/env occluder {A['tvis_stat'][0] / max(A['so_stat'][0], 1):.3f} of {A['so_stat'][0]:,.0f} "
               f"(self / static-occluder rates = cross-run Pi3 noise floor: neither occluder can move); "
               f"excess over the static-occluder floor on dynamic-occluder px = {(A['tvis_dyn'][0] / max(A['so_dyn'][0], 1) - A['tvis_stat'][0] / max(A['so_stat'][0], 1)) * A['so_dyn'][0] / occ:.3f} of all occluded")
    out.append(f"   nodepth px with valid Pi3 depth of the same X pixel at >= 1 other DEV frame: {A['nod_tval'][0] / max(n[3], 1):.3f}")
    out.append(f"   GT px of objects absent from c_ini's label map at t: {A['absent_c'].sum() / max(G, 1):.3f}")
    return A


out = []
out.append(f"DEV rows {len(rows)} (scene, t != ref_frame, target = non-c_ini annotated cam, basic-set object present in target); "
           f"scenes {len(set(r['scene'] for r in rows))}; frames per scene {sorted(set(len(set(r['t'] for r in rows if r['scene'] == s)) for s in av))}")
Aall = fmt_block("ALL DEV rows (20 non-ref frames x 2 targets x 17 scenes)", rows, out)
fmt_block("DEV subset non-ref[3::4] (frames +4,+8,+12,+16,+20; the dev-route DEV set)", [r for r in rows if dev_sub(r)], out)
fmt_block("DEV rows with R/GT >= 0.5 (mostly-invisible objects, the under-cover regime)",
          [r for r in rows if sum(r["n"][1:]) >= 0.5 * max(sum(r["n"]), 1)], out)
fmt_block("DEV rows of env-type objects", [r for r in rows if r["type"] == "env"], out)
fmt_block("DEV rows of the SAME 22 under-cover (scene, cam, obj) on non-keyframe frames", [r for r in rows if (r["scene"], r["cam"], r["obj"]) in UC22], out)
fmt_block("  of which Dog camera_0004 tree (obj 3)", [r for r in rows if (r["scene"], r["cam"], r["obj"]) == ("Dog", "camera_0004", 3)], out)
fmt_block("  the 21 others", [r for r in rows if (r["scene"], r["cam"], r["obj"]) in UC22 and (r["scene"], r["cam"], r["obj"]) != ("Dog", "camera_0004", 3)], out)

# DEV per-class reach rates applied to the keyframe under-cover class mix (quoted from the E11 report)
out.append("\n== DEV per-class reach rates (all DEV rows / same-22 DEV rows) applied to the QUOTED keyframe under-cover class mix")
same22 = agg([r for r in rows if (r["scene"], r["cam"], r["obj"]) in UC22])
for kname, (px, mix) in KF.items():
    for src, A in (("allDEV", Aall), ("same22DEV", same22)):
        n = A["n"]
        def proj(key):
            return sum(mix[c] * A[key][i] / max(n[i], 1) for i, c in enumerate(CLS) if i > 0)
        out.append(f"   {kname:12s} ({px:,} px) rates from {src:9s}: B_vis {proj('B_vis'):.3f}  B_ongt {proj('B_ongt'):.3f}  "
                   f"any_vis {proj('any_vis'):.3f}  nonann_vis {proj('nonann_vis'):.3f}  fwd_splat {proj('fwd_hit'):.3f}  unique-label {proj('uniq'):.3f}  "
                   f"(fractions of ALL missed px; beyond-c_ini share {1 - mix['vis']:.3f})")

# per scene
out.append("\n== per scene (all DEV rows): R/GT, class shares of R, B_ongt / any_vis / unique-label share of R, rig rigidity (max rot / tdir deg over DEV runs)")
for s in sorted(av):
    A = agg([r for r in rows if r["scene"] == s])
    n = A["n"]; Rn = n[1:].sum()
    pose = d["pose"][s]
    pr = max(p["max_rot_deg"] for p in pose.values()); pt = max(p["max_tdir_deg"] for p in pose.values())
    out.append(f"   {s:20s} R/GT {Rn / max(n.sum(), 1):.3f}  out {n[1] / max(Rn, 1):.2f} occ {n[2] / max(Rn, 1):.2f} nod {n[3] / max(Rn, 1):.2f} | "
               f"B_ongt {A['B_ongt'][1:].sum() / max(Rn, 1):.3f} any_vis {A['any_vis'][1:].sum() / max(Rn, 1):.3f} "
               f"uniq {A['uniq'][1:].sum() / max(Rn, 1):.3f} | rigid rot {pr:.2f} tdir {pt:.2f}")

# per label of the 22 under-cover objects: scene-level label multiplicity (metadata only)
out.append("\n== label multiplicity of the 22 under-cover objects (objects_labels.json of the scene; no GT read)")
labs = json.load(open("/home/sjpark/Documents/SCSegmentation/stage1/Data/Dog/objects_labels.json"))
for s, cam, o in UC22:
    lab = json.load(open(f"/home/sjpark/Documents/SCSegmentation/stage1/Data/{s}/objects_labels.json"))
    l = lab[str(o)]
    out.append(f"   {s:14s} {cam:12s} {o:3d} {l['label']:20s} {l['type']:8s} objects with this label in scene: {names[s][l['label']]}")
print("\n".join(out))
