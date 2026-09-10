#!/usr/bin/env python3
"""S1-E4a: does geometry put the reference mask in the right place?  (docs/stage1-E4a-prereg.md)

For every scene, the reference camera's ground-truth mask of each object is transferred to the two
other annotated cameras through COLMAP poses and dense MVS depth (MVSeed/geo/warp.py), and the
projected mask is scored against that camera's ground truth.  Same 796 rows as S1-E2 (664 gt-present
= F0 41 + N0 623, plus 132 gt-empty), so the two experiments are directly comparable.

Arms: `back` (backward warp), `fwd` (forward splat + closing), `union` = back | fwd (registered).
Rows of a scene whose reconstruction is unavailable -- COLMAP registered neither the reference nor
the target camera, or no depth map exists -- are kept in the population with an empty projection
(J = 0), never dropped: the gate denominator is fixed at the registered 16 pairs.

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD 3dgs \\
        python MVSeed/geo/e4a_project.py [--scenes A,B] [--save-masks]
    ... same container, --summary-only            # re-summarise the part files (needs numpy: container only)
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ci = _load("MVSeed/geo/colmap_io.py", "colmap_io")
warp = _load("MVSeed/geo/warp.py", "warp")
gate = _load("MVSeed/gate_det.py", "gate_det")

DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
CACHE = os.path.join(REPO, "MVSeed", "cache")
OUT = os.path.join(REPO, "MVSeed", "runs", "e4a_project.json")
PARTS = os.path.join(REPO, "MVSeed", "runs", "e4a_project_parts")
ARMS = ("back", "fwd", "union")
VERDICT_ARM = "union"
J_PASS = 0.5              # a pair passes when the projected mask reaches this IoU
GATE_MIN = 8              # F0 & pinhole & gt_px >= 2000: registered 16 pairs, >= 45 % -> >= 8
GATE_N = 16
HEMISPHERE = gate.HEMISPHERE


def scene_paths(scene):
    work = os.path.join(CACHE, scene, "pose")
    return dict(sparse=os.path.join(work, "sparse", "0", "txt"), dense=os.path.join(work, "dense"),
                dense_txt=os.path.join(work, "dense", "txt"),
                depth=os.path.join(work, "dense", "stereo", "depth_maps"))


def load_scene_geometry(scene, cams):
    """(orig cameras, undistorted cameras+poses, depth maps) or None when the scene is unavailable."""
    p = scene_paths(scene)
    if not (os.path.isdir(p["sparse"]) and os.path.isdir(p["dense_txt"])):
        return None
    ocams, oimgs = ci.read_model_txt(p["sparse"])
    ucams, uimgs = ci.read_model_txt(p["dense_txt"])
    out = {}
    for cam in cams:
        name = next((n for n in uimgs if os.path.splitext(n)[0] == cam), None)
        dm = os.path.join(p["depth"], f"{name}.geometric.bin") if name else None
        if name is None or not os.path.exists(dm):
            continue
        out[cam] = dict(orig=ocams[oimgs[name].camera_id], und=ucams[uimgs[name].camera_id],
                        img=uimgs[name], depth=ci.read_depth_map(dm))
    return out


def gt_mask(scene, cfg, cam):
    import cv2
    d = cfg[scene]
    p = os.path.join(DATA, d["folder"], "Mask", cam, f"{d['start_frame']:06d}.png")
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    assert m is not None, p
    return m


def camera_maps(geo, h, w):
    """Per camera: undistorted->original pixel map (to read masks) and original->undistorted (to write)."""
    uh, uw = geo["depth"].shape
    return dict(u2o=warp.undistorted_to_original_map(geo["und"], geo["orig"], uh, uw),
                o2u=warp.original_to_undistorted_map(geo["orig"], geo["und"], h, w))


def run_scene(scene, cfg, census, log, save_masks=None):
    import numpy as np
    ref_cam = census[scene]["c_ini"]
    cams = census[scene]["cameras"]
    targets = [c for c in cams if c != ref_cam]
    objects = [int(o) for o in census[scene]["seed_ids_c_ini"]]
    gts = {c: gt_mask(scene, cfg, c) for c in cams}
    h, w = gts[ref_cam].shape
    geo = load_scene_geometry(scene, cams)
    have = {} if geo is None else geo
    maps = {c: camera_maps(have[c], h, w) for c in have}
    rows = []
    ref_ok = ref_cam in have
    ref_masks = {}
    if ref_ok:
        u2o = maps[ref_cam]["u2o"]
        ref_lab, _ = warp.sample_nearest(gts[ref_cam], u2o[0], u2o[1])
        for obj in objects:
            ref_masks[obj] = ref_lab == obj
    for tgt in targets:
        pair_ok = ref_ok and tgt in have
        corr_b = corr_f = None
        if pair_ok:
            a, b = have[ref_cam], have[tgt]
            corr_b = warp.backward_correspondence(a["und"], a["img"], a["depth"], b["und"], b["img"], b["depth"])
            corr_f = warp.forward_correspondence(a["und"], a["img"], a["depth"], b["und"], b["img"], b["depth"])
        for obj in objects:
            g = gts[tgt] == obj
            rec = dict(scene=scene, cam=tgt, obj=obj, ref_cam=ref_cam, gt_px=int(g.sum()),
                       rig="hemisphere" if scene in HEMISPHERE else "pinhole",
                       available=bool(pair_ok), ref_registered=bool(ref_ok), tgt_registered=tgt in have,
                       ref_gt_px=int((gts[ref_cam] == obj).sum()))
            masks = {}
            if pair_ok:
                rm = ref_masks[obj]
                masks["back"] = warp.backward_from(corr_b, rm)
                masks["fwd"] = warp.forward_from(corr_f, rm)
                masks["union"] = masks["back"] | masks["fwd"]
                rec["ref_depth_frac"] = round(float(((have[ref_cam]["depth"] > 0) & rm).sum() / max(1, rm.sum())), 4)
                dt = have[tgt]["depth"]
                o2u = maps[tgt]["o2u"]
                dt_orig, _ = warp.sample_nearest(dt, o2u[0], o2u[1])
                rec["tgt_depth_frac"] = round(float((dt_orig[g] > 0).mean()), 4) if g.any() else None
            for arm in ARMS:
                if pair_ok:
                    m = warp.to_original(masks[arm], have[tgt]["orig"], have[tgt]["und"], h, w, cache=maps[tgt]["o2u"])
                else:
                    m = np.zeros((h, w), bool)
                rec[f"px_{arm}"] = int(m.sum())
                j = warp.iou(m, g) if g.any() else None
                rec[f"J_{arm}"] = round(j, 4) if j is not None else None
                rec[f"pass_{arm}"] = bool(j is not None and j > J_PASS)      # decided on the exact IoU
                if arm == VERDICT_ARM:
                    c = warp.centre_offset(m, g) if g.any() else None
                    rec["centre_offset"] = round(c, 3) if c is not None else None
                    if save_masks and g.any():
                        import cv2
                        d = os.path.join(save_masks, scene, tgt, str(obj))
                        os.makedirs(os.path.dirname(d), exist_ok=True)
                        cv2.imwrite(d + ".png", (m * 255).astype("uint8"))
            rows.append(rec)
        log(f"  {scene:16s} {ref_cam}->{tgt:12s} {'ok ' if pair_ok else 'N/A'} "
            + " ".join(f"{o}:{rows[-len(objects) + i]['J_union']}" for i, o in enumerate(objects[:6])) + " ...")
    return rows


# ------------------------------------------------------------------ summary (pure)
def _group(rows, arm=VERDICT_ARM):
    ok = [r for r in rows if r[f"J_{arm}"] is not None]
    hit = [r for r in ok if r[f"pass_{arm}"]]
    g = {"n": len(rows), "measured": len(ok), "pass": len(hit),
         "pass_rate": gate._rate(len(hit), len(rows)),
         "J_median": gate._median([r[f"J_{arm}"] for r in ok]),
         "J_mean": round(sum(r[f"J_{arm}"] for r in ok) / len(ok), 4) if ok else None,
         "available": sum(r["available"] for r in rows),
         "centre_offset_median": gate._median([r["centre_offset"] for r in ok if r["centre_offset"] is not None]),
         "tgt_depth_frac_median": gate._median([r["tgt_depth_frac"] for r in rows
                                                if r.get("tgt_depth_frac") is not None])}
    for a in ARMS:
        g[f"pass_{a}"] = sum(1 for r in rows if r[f"pass_{a}"])
        g[f"J_median_{a}"] = gate._median([r[f"J_{a}"] for r in rows if r[f"J_{a}"] is not None])
    return g


def summarize(rows):
    f0 = [r for r in rows if r["set"] == "F0"]
    gt = [r for r in rows if r["set"] in ("F0", "N0")]
    s = {"n_rows": len(rows),
         "f0_ge2k_pinhole": _group([r for r in f0 if r["gt_px"] >= 2000 and r["rig"] == "pinhole"]),
         "f0_ge2k": _group([r for r in f0 if r["gt_px"] >= 2000]),
         "f0_lt2k": _group([r for r in f0 if r["gt_px"] < 2000]),
         "f0_hemisphere": _group([r for r in f0 if r["rig"] == "hemisphere"]),
         "f0": _group(f0),
         "n0": _group([r for r in rows if r["set"] == "N0"]),
         "n0_by_size": {b: _group([r for r in rows if r["set"] == "N0" and r["size_bin"] == b])
                        for b in gate.SIZE_BINS},
         "by_scene": {sc: _group([r for r in gt if r["scene"] == sc])
                      for sc in sorted({r["scene"] for r in gt})}}
    empty = [r for r in rows if r["set"] == "gt_empty"]
    av = [r for r in empty if r["available"]]
    hal = sum(r[f"px_{VERDICT_ARM}"] > 0 for r in empty)     # unavailable rows project nothing -> no hallucination
    s["gt_empty"] = {"n": len(empty), "available": len(av), "halluc": hal,
                     "halluc_rate": gate._rate(hal, len(empty)),          # registered denominator: all 132 rows
                     "halluc_rate_available": gate._rate(hal, len(av))}
    counts = {"f0_ge2k_pinhole": s["f0_ge2k_pinhole"]["n"], "f0_ge2k": s["f0_ge2k"]["n"],
              "f0_lt2k": s["f0_lt2k"]["n"], "n0": s["n0"]["n"], "gt_empty": s["gt_empty"]["n"]}
    registered = {"f0_ge2k_pinhole": GATE_N, "f0_ge2k": 19, "f0_lt2k": 22, "n0": 623, "gt_empty": 132}
    s["denominators"] = {"registered": registered, "measured": counts, "match": counts == registered}
    g = s["f0_ge2k_pinhole"]
    passed = g["pass"] >= GATE_MIN and g["n"] == GATE_N
    s["gates"] = {"gate_projection": {"pass": bool(passed), "arm": VERDICT_ARM, "value": g["pass"],
                                      "n": g["n"], "registered_n": GATE_N, "min": GATE_MIN,
                                      "envelope_points": round(g["pass"] * 0.50 / GATE_N, 3)},
                  "verdict": ("proceed with the geometry arm" if passed else
                              "discard this arm: projection gate failed")}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--parts-dir", default=PARTS)
    ap.add_argument("--save-masks", default=None, help="also write the union masks under this folder (debug)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    census = json.load(open(CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(census)
    unknown = [s for s in scenes if s not in census]
    assert not unknown, f"unknown scenes: {unknown}"
    f0, n0 = gate.membership(json.load(open(gate.spike.F0))["pairs"], json.load(open(gate.spike.N0))["pairs"])
    os.makedirs(args.parts_dir, exist_ok=True)
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    if not args.summary_only:
        for s in scenes:
            part = os.path.join(args.parts_dir, f"{s}.json")
            keys = gate.expected_keys(s, cfg, census)
            if gate.load_part(part, keys) is not None and not args.overwrite:
                log(f"  {s}: complete part exists, skipping")
                continue
            t1 = time.time()
            res = run_scene(s, cfg, census, log, args.save_masks)
            order = {k: i for i, k in enumerate(keys)}
            res.sort(key=lambda r: order[(r["cam"], r["obj"])])
            json.dump(res, open(part + ".tmp", "w"), indent=1)
            os.replace(part + ".tmp", part)
            log(f"  {s}: {len(res)} rows in {time.time() - t1:.0f}s")
    rows, missing = [], []
    for s in sorted(census):
        got = gate.load_part(os.path.join(args.parts_dir, f"{s}.json"), gate.expected_keys(s, cfg, census))
        if got is None:
            missing.append(s)
        else:
            rows += [gate.tag_row(r, f0, n0) for r in got]
    summary = summarize(rows)
    summary["missing_scenes"] = missing
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    doc = {"schema": "MVSeed e4a_project", "schema_version": 1, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "git_head": head, "generated_by": "MVSeed/geo/e4a_project.py", "argv": sys.argv,
           "prereg": "docs/stage1-E4a-prereg.md",
           "rules": {"arms": list(ARMS), "verdict_arm": VERDICT_ARM, "j_pass": J_PASS,
                     "occ_tol": warp.OCC_TOL, "close_kernel": warp.CLOSE_KERNEL,
                     "gate": f"F0 & pinhole & gt_px>=2000: J_{VERDICT_ARM} > {J_PASS} for >= {GATE_MIN} of {GATE_N}",
                     "unavailable_rows": "kept with an empty projection (J = 0)"},
           "summary": summary, "rows": rows}
    json.dump(doc, open(args.out, "w"), indent=1)
    print_summary(summary)
    print(f"wrote {args.out}  ({len(rows)} rows, missing {missing}, {time.time() - t0:.0f}s)")


def print_summary(s):
    def line(name, g):
        print(f"  {name:20s} n={g['n']:3d} avail {g['available']:3d}  pass {g['pass']:3d} ({g['pass_rate']})  "
              f"back {g['pass_back']:3d} fwd {g['pass_fwd']:3d}  J med {g['J_median']}  "
              f"centre {g['centre_offset_median']}  tgt-depth {g['tgt_depth_frac_median']}")
    print(f"\n[S1-E4a projection] rows {s['n_rows']}  missing scenes {s.get('missing_scenes')}  "
          f"denominators match: {s['denominators']['match']} {s['denominators']['measured']}")
    for k in ("f0_ge2k_pinhole", "f0_ge2k", "f0_lt2k", "f0_hemisphere", "f0", "n0"):
        line(k, s[k])
    for k, g in s["n0_by_size"].items():
        line(f"n0 {k}", g)
    e = s["gt_empty"]
    print(f"  gt_empty             n={e['n']:3d} avail {e['available']:3d}  halluc {e['halluc']} "
          f"({e['halluc_rate']} of 132; {e['halluc_rate_available']} of the available)")
    for k, g in sorted(s["by_scene"].items()):
        line(f"scene {k}", g)
    for k, v in s["gates"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
