"""S1-E11 prep, stage2-coupling: failing-seed cost by group x scene, per-pair stage-2 decomposition
(frame 0 vs frames 1-20), F0/N0 membership, reprojection anatomy of the 59 failing pairs and a
decode-only vs new-information verdict.  Host: stage1/.venv/bin/python (numpy, cv2).  Read-only.

    stage1/.venv/bin/python coupling.py <out_dir>
"""
import collections
import json
import math
import os
import sys

import cv2
import numpy as np

REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "MVSeed"))
sys.path.insert(0, os.path.join(REPO, "eval"))
from e7_judge import load, rows, jf          # noqa: E402
import eval_jf as E                           # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
SF = json.load(open(os.path.join(REPO, "MVSeed/runs/seedfail.json")))
R = SF["rows"]
H, G = SF["headline"], SF["gt_seed"]
CFG = json.load(open(os.path.join(REPO, "SCSam3/demo/MVSeg.json")))
OBJ = json.load(open(os.path.join(REPO, "docs/raw/muvod_object_sets.json")))
F0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed/runs/F0.json")))["pairs"]}
N0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open(os.path.join(REPO, "MVSeed/runs/N0.json")))["pairs"]}
S1 = os.path.join(REPO, "stage1/Data")
MV = os.path.join(REPO, "Data/MVSeg")
SEED, R1 = "geoset_pred_ft_rep_e8", "geoset_pred_ft_r1_e8"
CONF_MIN, OCC_TOL, THETA, TAU, K = 0.1, 0.01, 0.5, 0.7, 3
MIN_VALID = 64


def grp(kind):
    return "spill" if kind.startswith("spill") else "under" if kind.startswith("under") else "rest"


def key(r):
    return (r["scene"], r["cam"], r["obj"])


# ------------------------------------------------------------------ A. groups x scene
fail = [r for r in R if r["JF"] < 0.8]
good = [r for r in R if r["JF"] >= 0.8]
assert len(R) == 664 and len(fail) == 59, (len(R), len(fail))
print(f"A. rows {len(R)}  fail {len(fail)} cost {sum(r['cost'] for r in fail):.4f}  "
      f"good {len(good)} cost {sum(r['cost'] for r in good):.4f}  total {sum(r['cost'] for r in R):.4f}")
kinds = collections.defaultdict(list)
for r in fail:
    kinds[r["kind"]].append(r)
for k, v in sorted(kinds.items(), key=lambda x: -sum(r["cost"] for r in x[1])):
    print(f"   {k:30s} n={len(v):2d} cost {sum(r['cost'] for r in v):+.4f}")
groups = collections.defaultdict(list)
for r in fail:
    groups[grp(r["kind"])].append(r)
print("A. groups:", {g: (len(v), round(sum(r["cost"] for r in v), 4)) for g, v in groups.items()})
scenes = sorted({r["scene"] for r in R})
tab = {}
for sc in scenes:
    row = {}
    for g in ("spill", "under", "rest"):
        v = [r for r in groups[g] if r["scene"] == sc]
        row[g] = dict(n=len(v), cost=round(sum(r["cost"] for r in v), 4))
    fs = [r for r in fail if r["scene"] == sc]
    gs = [r for r in good if r["scene"] == sc]
    row["fail"] = dict(n=len(fs), cost=round(sum(r["cost"] for r in fs), 4))
    row["good"] = dict(n=len(gs), cost=round(sum(r["cost"] for r in gs), 4))
    row["all"] = dict(n=len(fs) + len(gs), cost=round(sum(r["cost"] for r in fs + gs), 4))
    tab[sc] = row
print("A. per scene (n / cost):")
print(f"   {'scene':20s} {'spill':>14s} {'under':>14s} {'rest':>14s} {'fail':>14s} {'good':>14s} {'all':>14s}")
for sc in sorted(scenes, key=lambda s: -tab[s]["fail"]["cost"]):
    t = tab[sc]
    print(f"   {sc:20s} " + " ".join(f"{t[g]['n']:3d} / {t[g]['cost']:+.3f}" for g in ("spill", "under", "rest", "fail", "good", "all")))

# ------------------------------------------------------------------ B. stage-2 per-frame decomposition
d = load(["jf_e8.json", "jf_e7.json"])


def table(m):
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows(d, m):
        per[r["scene"]][r["cam"]][r["obj"]] = r
    out = {}
    for sc, cams in per.items():
        for cam, objs in cams.items():
            for o, r in objs.items():
                out[(sc, cam, o)] = (r, 100 / (17 * len(cams) * len(objs)) / len(r["J"]))
    return out


T = {m: table(m) for m in (H, G)}


def loss(m, k, frames=None):
    r, w = T[m][k]
    return sum(w * (1 - jf(r, t)) for t in range(len(r["J"])) if (frames is None or t in frames) and jf(r, t) is not None)


dec = {}
max_diff = 0.0
for r in fail:
    k = key(r)
    rH, w = T[H][k]
    n_fr = len(rH["J"])
    later = set(range(1, n_fr))
    c0 = loss(H, k, {0}) - loss(G, k, {0})
    cl = loss(H, k, later) - loss(G, k, later)
    tot = loss(H, k) - loss(G, k)
    max_diff = max(max_diff, abs(tot - r["cost"]))
    jf0_H = jf(rH, 0)
    jfl_H = [jf(rH, t) for t in later if jf(rH, t) is not None]
    rG, _ = T[G][k]
    jfl_G = [jf(rG, t) for t in later if jf(rG, t) is not None]
    dec[k] = dict(w=w, n_frames=n_fr, cost0=c0, cost_later=cl, cost=tot, LH=loss(H, k), LG=loss(G, k),
                  jf0_H=jf0_H, jf0_G=jf(rG, 0), jf_later_H=float(np.mean(jfl_H)) if jfl_H else None,
                  jf_later_G=float(np.mean(jfl_G)) if jfl_G else None,
                  in_F0=k in F0, in_N0=k in N0)
print(f"B. cost recomputed from jf_e8/jf_e7 vs seedfail.json rows: max |diff| = {max_diff:.2e}")
print(f"B. frame-0 J&F of the headline == stage-1 seed J&F? max |diff| = "
      f"{max(abs(dec[key(r)]['jf0_H'] - r['JF']) for r in fail):.4f}")
for g in ("spill", "under", "rest"):
    v = groups[g]
    c0 = sum(dec[key(r)]["cost0"] for r in v)
    cl = sum(dec[key(r)]["cost_later"] for r in v)
    print(f"B. {g:6s} n={len(v):2d} cost {c0 + cl:+.4f} = frame0 {c0:+.4f} + frames1-20 {cl:+.4f}  "
          f"(per-frame later avg {cl / 20:+.4f}; headline JF: f0 {np.mean([dec[key(r)]['jf0_H'] for r in v]):.3f} "
          f"-> later {np.mean([dec[key(r)]['jf_later_H'] for r in v]):.3f}; GT-seed later {np.mean([dec[key(r)]['jf_later_G'] for r in v]):.3f})"
          f"  in F0 {sum(dec[key(r)]['in_F0'] for r in v)} / in N0 {sum(dec[key(r)]['in_N0'] for r in v)}")
c0 = sum(dec[key(r)]["cost0"] for r in fail)
cl = sum(dec[key(r)]["cost_later"] for r in fail)
print(f"B. all fail: cost {c0 + cl:+.4f} = frame0 {c0:+.4f} + later {cl:+.4f}; in F0 {sum(dec[key(r)]['in_F0'] for r in fail)}, "
      f"F0 pairs total {len(F0)}, F0 pairs with seed JF>=0.8 now: {sum(1 for r in good if key(r) in F0)}")
# tracking amplification: pairs whose later J&F is worse than frame 0 by > 0.02 / better
amp = collections.Counter()
for r in fail:
    x = dec[key(r)]
    amp["better" if x["jf_later_H"] > x["jf0_H"] + 0.02 else "worse" if x["jf_later_H"] < x["jf0_H"] - 0.02 else "same"] += 1
print("B. fail pairs, frames 1-20 mean J&F vs frame 0 (+-0.02):", dict(amp))

# ------------------------------------------------------------------ C. geometry / visibility per failing pair
_cache = {}


def scene_cfg(sc):
    c = CFG.get(sc) or next(v for v in CFG.values() if v.get("folder") == sc)
    return c["folder"], c["start_frame"]


def read_gt(folder, cam, st):
    g = cv2.imread(os.path.join(MV, folder, "Mask", cam, "%06d.png" % st), cv2.IMREAD_UNCHANGED)
    return g[..., 0] if g.ndim == 3 else g


def read_mask(sc, res, cam, o, hw):
    p = os.path.join(S1, sc, "Result", res, cam, f"{o}.png")
    s = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
    if s is None:
        return np.zeros(hw, bool)
    if s.shape != hw:
        s = cv2.resize(s, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return s > 127


def jf_pair(g, p):
    h, w = g.shape
    bp = int(math.ceil(E.BOUND_TH * np.linalg.norm((h, w))))
    se = E._disk(bp)
    box = E.crop_box(g, p, bp + 2, g.shape)
    if box is None:
        return 1.0
    y0, y1, x0, x1 = box
    return 0.5 * (E.db_eval_iou(g[y0:y1, x0:x1], p[y0:y1, x0:x1]) + E.db_eval_boundary(p[y0:y1, x0:x1], g[y0:y1, x0:x1], bp, se))


def reproj_labels(sc, cam, ref_cam, hw, G0):
    """Per target pixel: the reference GT label it reprojects onto (Pi3 points, occlusion-checked as
    geomem.warp_torch.reproject_coords; nearest sampling), or -1 where the reprojection is invalid
    (low Pi3 conf, no depth, outside the reference image, occluded)."""
    cache = os.path.join(S1, sc, "Cache", "pi3")
    h, w = hw
    zu = np.load(os.path.join(cache, f"{cam}.npz"))
    zr = np.load(os.path.join(cache, f"{ref_cam}.npz"))
    v, u = np.mgrid[0:zu["depth"].shape[0], 0:zu["depth"].shape[1]].astype(np.float64)
    z = zu["depth"].astype(np.float64)
    Ku = zu["K"]
    p = np.stack([(u - Ku[0, 2]) / Ku[0, 0] * z, (v - Ku[1, 2]) / Ku[1, 1] * z, z], -1)
    p[zu["depth"] <= 0] = 0
    Tm = np.linalg.inv(zr["c2w"].astype(np.float64)) @ zu["c2w"].astype(np.float64)
    pts = (p @ Tm[:3, :3].T + Tm[:3, 3]).astype(np.float32)
    valid_u = ((zu["conf"] > CONF_MIN) & (zu["depth"] > 0)).astype(np.uint8)
    pts_full = cv2.resize(pts, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_full = cv2.resize(valid_u, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    depth_ref = cv2.resize(zr["depth"], (w, h), interpolation=cv2.INTER_NEAREST)
    hw_work = zr["depth"].shape
    sx, sy = w / hw_work[1], h / hw_work[0]
    Kr = zr["K"]
    fx, fy, cx, cy = Kr[0, 0] * sx, Kr[1, 1] * sy, (Kr[0, 2] + 0.5) * sx - 0.5, (Kr[1, 2] + 0.5) * sy - 0.5
    zz = pts_full[..., 2].astype(np.float64)
    zs = np.where(np.abs(zz) < 1e-9, 1e-9, zz)
    uu = fx * pts_full[..., 0] / zs + cx
    vv = fy * pts_full[..., 1] / zs + cy
    valid = (zz > 0) & (uu >= -0.5) & (uu <= w - 0.5) & (vv >= -0.5) & (vv <= h - 0.5) & np.isfinite(uu) & np.isfinite(vv)
    col = np.clip(np.round(uu), 0, w - 1).astype(np.int64)
    row = np.clip(np.round(vv), 0, h - 1).astype(np.int64)
    dr = depth_ref[row, col]
    occl = ~((dr <= 0) | (zz <= dr * (1.0 + OCC_TOL)))
    cause = np.zeros(hw, np.int8)
    cause[valid & occl] = 3
    cause[~valid] = 2
    cause[~valid_full] = 1
    valid &= ~occl
    valid &= valid_full
    lab = np.where(valid, G0[row, col].astype(np.int64), -1)
    return lab, cause


def centres(sc, cams):
    cache = os.path.join(S1, sc, "Cache", "pi3")
    return {c: np.load(os.path.join(cache, f"{c}.npz"))["c2w"][:3, 3].astype(np.float64) for c in cams}


def conf_reproj(e):
    present = e["presence"] > 0
    if e["overlap"] is None:
        return bool(present and e["iou"] >= TAU)
    return bool(present and e["overlap"] >= THETA)


def frac(mask, sel):
    n = int(mask.sum())
    return (float((mask & sel).sum()) / n) if n else None


def anat_of(m, valid, own, other, bg, lowconf, outfov, occl):
    return dict(valid=frac(m, valid), own=frac(m, own), other=frac(m, other), bg=frac(m, bg),
                lowconf=frac(m, lowconf), outfov=frac(m, outfov), occl=frac(m, occl))


per_scene = collections.defaultdict(list)
for r in fail:
    per_scene[r["scene"]].append(r)
anat = {}
for sc, prs in sorted(per_scene.items()):
    folder, st = scene_cfg(sc)
    meta = json.load(open(os.path.join(S1, sc, "meta.json")))
    ref = OBJ[sc]["c_ini"]
    assert meta["ref_cam"] == ref and meta["ref_frame"] == st
    ann = list(OBJ[sc]["cameras"])
    GT = {c: read_gt(folder, c, st) for c in ann}
    G0 = GT[ref]
    hw = G0.shape
    r1conf = json.load(open(os.path.join(S1, sc, "Result", SEED, "round1_conf.json")))
    cen = centres(sc, meta["cams"])
    labs = {}
    for r in prs:
        cam, o = r["cam"], r["obj"]
        if cam not in labs:
            labs[cam] = reproj_labels(sc, cam, ref, hw, G0)
        lab, cause = labs[cam]
        lowconf, outfov, occl = cause == 1, cause == 2, cause == 3
        g = GT[cam] == o
        s = read_mask(sc, SEED, cam, o, hw)
        s1 = read_mask(sc, R1, cam, o, hw)
        valid = lab >= 0
        own = lab == o
        other = valid & (lab != o) & (lab != 0)
        bg = lab == 0
        missed = g & ~s
        extra = s & ~g
        cand = sorted((np.linalg.norm(cen[c] - cen[cam]), c) for c in cen if c != cam)
        cand = [c for _, c in cand[:K] if c != ref]
        neigh = []
        for c in cand:
            e = r1conf[c][str(o)]
            n_gt = int((GT[c] == o).sum()) if c in GT else None
            neigh.append(dict(cam=c, confident=conf_reproj(e), presence=round(e["presence"], 2), iou=round(e["iou"], 3),
                              overlap=None if e["overlap"] is None else round(e["overlap"], 3), n_valid=e["n_valid"],
                              annotated=c in GT, gt_px=n_gt,
                              r1_jf=(round(jf_pair(GT[c] == o, read_mask(sc, R1, c, o, hw)), 4) if c in GT and n_gt else None)))
        own_e = r1conf[cam][str(o)]
        # other-object pixels in the target seed: which objects, and where do those seed pixels reproject in the reference?
        oth = r["other_obj"]
        spill_px = (s & (GT[cam] == oth)) if oth else np.zeros(hw, bool)
        anat[key(r)] = dict(
            gt_px=int(g.sum()), seed_px=int(s.sum()), ref_gt_px=int((G0 == o).sum()),
            gt_px_other_cams={c: int((GT[c] == o).sum()) for c in ann if c not in (cam, ref)},
            gt_reproj=anat_of(g, valid, own, other, bg, lowconf, outfov, occl),
            missed_px=int(missed.sum()),
            missed_reproj=anat_of(missed, valid, own, other, bg, lowconf, outfov, occl),
            extra_px=int(extra.sum()),
            extra_reproj=anat_of(extra, valid, own, other, bg, lowconf, outfov, occl),
            spill_px=int(spill_px.sum()),
            spill_reproj=anat_of(spill_px, valid, own, other, bg, lowconf, outfov, occl), r1_n_valid_mine=int((s1 & valid).sum()),
            other_obj_in_ref=(int((G0 == oth).sum()) if oth else None),
            other_obj_is_basic=(oth in OBJ[sc]["seed_ids_c_ini"]) if oth else None,
            r1_jf=round(jf_pair(g, s1), 4), r1_px=int(s1.sum()), r1_vs_final_iou=(float((s1 & s).sum() / max(1, (s1 | s).sum()))),
            own_r1=dict(confident=conf_reproj(own_e), presence=round(own_e["presence"], 2), iou=round(own_e["iou"], 3),
                        overlap=None if own_e["overlap"] is None else round(own_e["overlap"], 3), n_valid=own_e["n_valid"]),
            neighbours=neigh, n_neigh_admitted=sum(n["confident"] for n in neigh),
            annotated_cams=ann, ref=ref)
    print(f"C. {sc}: {len(prs)} failing pairs, annotated cams {ann}, ref {ref}, stage-1 cams {len(meta['cams'])}")


# ------------------------------------------------------------------ D. verdict
def verdict(r, a):
    """decode-only plausible vs needs new information, by the reprojection anatomy.
    under-cover: the missed GT pixels are 'in c_ini' when they reproject onto the object's own reference GT
      (the memory already holds them) -> decode-only (memory/attention) plausible; when most are invalid or on
      reference background, c_ini does not see them -> only a neighbour's round-1 mask or the target image can supply
      them (new information unless an admitted neighbour already covers it).
    spill / wrong object: the wrong seed pixels reproject onto ANOTHER object's reference GT or reference background
      -> a geometric exclusivity rule can detect them -> decode-only plausible; when they land on the own reference GT
      or are invalid, geometry cannot tell -> needs new information.
    miss: tiny GT -> low value.  boundary/mixed: decode-only in principle, but E9/E10 show it is hard."""
    k = r["kind"]
    if k.startswith("under"):
        m = a["missed_reproj"]
        own = m["own"] or 0.0
        if own >= 0.5:
            return "decode-only plausible (missed part visible in c_ini: %.0f%% on own ref GT)" % (100 * own)
        return "needs new information (missed part not in c_ini: %.0f%% out of c_ini FOV, %.0f%% occluded, %.0f%% no Pi3 geometry, %.0f%% ref bg, %.0f%% own)" % (
            100 * (m["outfov"] or 0), 100 * (m["occl"] or 0), 100 * (m["lowconf"] or 0), 100 * (m["bg"] or 0), 100 * own)
    if k.startswith("spill") or k == "wrong-object":
        e = a["extra_reproj"]
        det = (e["other"] or 0.0) + (e["bg"] or 0.0)
        if det >= 0.5:
            return "decode-only plausible (wrong px geometrically separable: %.0f%% other ref GT, %.0f%% ref bg)" % (
                100 * (e["other"] or 0), 100 * (e["bg"] or 0))
        return "needs new information (wrong px not separable by geometry: %.0f%% own ref GT, %.0f%% out of c_ini FOV, %.0f%% occluded, %.0f%% no Pi3 geometry)" % (
            100 * (e["own"] or 0), 100 * (e["outfov"] or 0), 100 * (e["occl"] or 0), 100 * (e["lowconf"] or 0))
    if k == "miss":
        return "low value (gt %d px)" % r["gt_px"]
    return "boundary (decode-only in principle; E9/E10 found it hard)"


listing = []
for r in sorted(fail, key=lambda r: -r["cost"]):
    a = anat[key(r)]
    x = dec[key(r)]
    listing.append(dict(scene=r["scene"], cam=r["cam"], obj=r["obj"], label=r["label"], otype=r["otype"], kind=r["kind"],
                        group=grp(r["kind"]), cost=round(r["cost"], 4), cost_frame0=round(x["cost0"], 4),
                        cost_later=round(x["cost_later"], 4), JF=round(r["JF"], 4), r1_JF=a["r1_jf"],
                        gt_px=r["gt_px"], seed_px=r["seed_px"], vis_ratio=round(r["vis_ratio"], 3), twin=r["twin"],
                        recall=round(r["recall"], 3) if r["recall"] is not None else None,
                        precision=round(r["precision"], 3) if r["precision"] is not None else None,
                        frac_other=round(r["frac_other"], 3), other_obj=r["other_obj"], border=r["border"],
                        angle=round(r["angle"], 1) if r["angle"] is not None else None,
                        in_F0=x["in_F0"], jf_later_H=round(x["jf_later_H"], 3), jf_later_G=round(x["jf_later_G"], 3),
                        n_neigh_admitted=a["n_neigh_admitted"], neighbours=a["neighbours"], own_r1=a["own_r1"],
                        gt_reproj=a["gt_reproj"], missed_reproj=a["missed_reproj"], extra_reproj=a["extra_reproj"],
                        spill_reproj=a["spill_reproj"], other_obj_in_ref_px=a["other_obj_in_ref"],
                        gt_px_other_cams=a["gt_px_other_cams"], ref_gt_px=a["ref_gt_px"],
                        verdict=verdict(r, a)))
print("D. 59 failing pairs (cost desc):")
hdr = f"   {'scene':18s} {'cam':12s} {'obj':>3s} {'label':14s} {'kind':28s} {'cost':>7s} {'f0':>7s} {'later':>7s} {'JF':>5s} {'r1JF':>5s} {'gt_px':>7s} {'vis':>5s} tw nb verdict"
print(hdr)
for p in listing:
    print(f"   {p['scene'][:18]:18s} {p['cam']:12s} {p['obj']:3d} {str(p['label'])[:14]:14s} {p['kind']:28s} {p['cost']:+.4f} {p['cost_frame0']:+.4f} "
          f"{p['cost_later']:+.4f} {p['JF']:.3f} {p['r1_JF']:.3f} {p['gt_px']:7d} {p['vis_ratio']:5.2f} {'T' if p['twin'] else '.'}  {p['n_neigh_admitted']}  {p['verdict']}")
vs = collections.defaultdict(lambda: [0, 0.0])
for p in listing:
    v = p["verdict"].split(" (")[0]
    vs[(p["group"], v)][0] += 1
    vs[(p["group"], v)][1] += p["cost"]
print("C. validation: my valid-reprojected count of the round-1 mask vs round1_conf n_valid (official torch path):")
for p in listing:
    a = anat[(p["scene"], p["cam"], p["obj"])]
    print(f"   {p['scene'][:12]:12s} {p['cam']:12s} {p['obj']:3d}  mine {a['r1_n_valid_mine']:8d}  official {a['own_r1']['n_valid']:8d}  r1_px {a['r1_px']:8d}")
print("D. neighbours of the under/spill/wrong pairs (cam: admitted?, presence, iou, overlap | annotated gt_px, r1 J&F):")
for p in listing:
    if p["group"] == "rest" and p["kind"] != "wrong-object":
        continue
    nb = "; ".join(f"{n['cam']}:{'A' if n['confident'] else 'x'} p{n['presence']:+.1f} iou{n['iou']:.2f} ov{n['overlap'] if n['overlap'] is not None else 'nan'}"
                   + (f" |gt {n['gt_px']} r1JF {n['r1_jf']}" if n['annotated'] else "") for n in p["neighbours"])
    print(f"   {p['scene'][:12]:12s} {p['cam']:12s} {p['obj']:3d} {p['kind'][:14]:14s} ref_gt {p['ref_gt_px']:7d} gt {p['gt_px']:7d} other annotated cams gt {p['gt_px_other_cams']}  own r1: {'A' if p['own_r1']['confident'] else 'x'} p{p['own_r1']['presence']:+.1f} iou{p['own_r1']['iou']:.2f} ov{p['own_r1']['overlap']}  nb {nb}")
print("D. cost by otype within group:")
for g in ("spill", "under", "rest"):
    c = collections.defaultdict(lambda: [0, 0.0])
    for r in groups[g]:
        c[r["otype"]][0] += 1; c[r["otype"]][1] += r["cost"]
    print(f"   {g:6s} " + "  ".join(f"{k}: n={v[0]} cost {v[1]:+.4f}" for k, v in sorted(c.items(), key=lambda x: -x[1][1])))
print("D. verdict totals:")
for (g, v), (n, c) in sorted(vs.items(), key=lambda x: -x[1][1]):
    print(f"   {g:6s} {v:40s} n={n:2d} cost {c:+.4f}")
# round 1 -> round 2 on the failing pairs
r1_better = sum(1 for p in listing if p["r1_JF"] > p["JF"] + 0.02)
r1_worse = sum(1 for p in listing if p["r1_JF"] < p["JF"] - 0.02)
print(f"D. round-1 vs final on the 59: r1 better by >0.02 in {r1_better}, worse in {r1_worse}, "
      f"mean r1 JF {np.mean([p['r1_JF'] for p in listing]):.4f} vs final {np.mean([p['JF'] for p in listing]):.4f}; "
      f"pairs with 0 admitted neighbours {sum(p['n_neigh_admitted'] == 0 for p in listing)}")
json.dump(dict(doc="S1-E11 prep stage2-coupling", seeds=SEED, headline=H, gt_seed=G,
               groups={g: dict(n=len(v), cost=round(sum(r["cost"] for r in v), 4)) for g, v in groups.items()},
               per_scene=tab, decomposition={"|".join(map(str, k)): v for k, v in dec.items()},
               pairs=listing),
          open(os.path.join(OUT, "coupling.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "coupling.json"))
