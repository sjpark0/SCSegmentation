#!/usr/bin/env python3
"""S1-E4c: does a 3DGS render put the reference mask in the right place?  (docs/stage1-E4c-prereg.md)

Identical to S1-E4a in every respect but one: the depth of each annotated camera comes from a
render of the scene's trained 3D Gaussian Splatting model (MVSeed/geo/gs_depth.py) instead of
COLMAP dense MVS.  Poses, the undistorted/original round trip, the transfer (MVSeed/geo/warp.py),
the arms, the gate and the 796-row population are e4a's, reused unchanged so the two arms are
directly comparable pair by pair.

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD 3dgs \\
        python MVSeed/geo/e4c_project.py [--scenes A,B]
    ... same container, --depth-check          # report-only: GS depth vs E4a dense MVS depth
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "MVSeed", "geo"))
import importlib.util


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e4a = _load("MVSeed/geo/e4a_project.py", "e4a_project")
ci, warp, gate = e4a.ci, e4a.warp, e4a.gate

CACHE = e4a.CACHE
OUT = os.path.join(REPO, "MVSeed", "runs", "e4c_project.json")
PARTS = os.path.join(REPO, "MVSeed", "runs", "e4c_project_parts")
ITERATION = 30000


DEPTH_DIR = "depth"          # the registered read-out; "depth_lin" is the report-only diagnostic


def scene_paths(scene):
    work = os.path.join(CACHE, scene)
    return dict(sparse=os.path.join(work, "pose", "sparse", "0", "txt"),
                und_txt=os.path.join(work, "undist", "sparse", "0", "txt"),
                depth=os.path.join(work, "gs", DEPTH_DIR),
                mvs=os.path.join(work, "pose", "dense", "stereo", "depth_maps"),
                mvs_txt=os.path.join(work, "pose", "dense", "txt"))


def load_scene_geometry(scene, cams):
    """(orig cameras, undistorted cameras+poses, rendered depth) or None when the scene is unavailable."""
    import numpy as np
    p = scene_paths(scene)
    if not (os.path.isdir(p["sparse"]) and os.path.isdir(p["und_txt"]) and os.path.isdir(p["depth"])):
        return None
    ocams, oimgs = ci.read_model_txt(p["sparse"])
    ucams, uimgs = ci.read_model_txt(p["und_txt"])
    out = {}
    for cam in cams:
        name = next((n for n in uimgs if os.path.splitext(n)[0] == cam), None)
        npz = os.path.join(p["depth"], f"{cam}.npz")
        if name is None or not os.path.exists(npz):
            continue
        und = ucams[uimgs[name].camera_id]
        depth = np.load(npz)["depth"]
        assert depth.shape == (und.height, und.width), (scene, cam, depth.shape, und.height, und.width)
        out[cam] = dict(orig=ocams[oimgs[name].camera_id], und=und, img=uimgs[name], depth=depth)
    return out


def depth_check(scene, census, log=print):
    """Report-only: median relative difference between the render and E4a's dense MVS depth."""
    import numpy as np
    p = scene_paths(scene)
    if not os.path.isdir(p["mvs_txt"]):
        return None
    mcams, mimgs = ci.read_model_txt(p["mvs_txt"])
    geo = load_scene_geometry(scene, census[scene]["cameras"])
    rows = []
    for cam, g in (geo or {}).items():
        name = next((n for n in mimgs if os.path.splitext(n)[0] == cam), None)
        dm = os.path.join(p["mvs"], f"{name}.geometric.bin") if name else None
        if dm is None or not os.path.exists(dm):
            continue
        mvs = ci.read_depth_map(dm)                       # dense workspace: max side 1600
        gs = g["depth"]
        # both are depth maps of the same camera at different scales -> compare on the MVS grid
        ys = (np.arange(mvs.shape[0]) * gs.shape[0] / mvs.shape[0]).astype(int)
        xs = (np.arange(mvs.shape[1]) * gs.shape[1] / mvs.shape[1]).astype(int)
        gsr = gs[ys[:, None], xs[None, :]]
        ok = (mvs > 0) & (gsr > 0)
        if ok.sum() < 100:
            continue
        signed = (gsr[ok] - mvs[ok]) / mvs[ok]        # < 0 = the render puts the surface nearer than MVS
        rel = np.abs(signed)
        rows.append(dict(scene=scene, cam=cam, n=int(ok.sum()), overlap=round(float(ok.mean()), 4),
                         rel_median=round(float(np.median(rel)), 4), rel_p90=round(float(np.percentile(rel, 90)), 4),
                         signed_median=round(float(np.median(signed)), 4),
                         nearer_frac=round(float((signed < 0).mean()), 4)))
        log(f"  {scene:16s} {cam:14s} overlap {rows[-1]['overlap']:.2f}  rel median {rows[-1]['rel_median']:.4f}"
            f"  signed {rows[-1]['signed_median']:+.4f}  nearer {rows[-1]['nearer_frac']:.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--parts-dir", default=PARTS)
    ap.add_argument("--save-masks", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--depth-check", action="store_true", help="report-only GS-vs-MVS depth comparison")
    ap.add_argument("--linear", action="store_true",
                    help="report-only: read gs/depth_lin (linear depth expectation) instead of gs/depth")
    args = ap.parse_args()
    cfg = json.load(open(e4a.CONFIG, encoding="utf-8"))
    census = json.load(open(e4a.CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(census)
    assert not [s for s in scenes if s not in census], "unknown scene"
    if args.linear:
        globals()["DEPTH_DIR"] = "depth_lin"                 # report-only diagnostic, never the verdict
    e4a.load_scene_geometry = load_scene_geometry            # the one and only difference from E4a
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    if args.depth_check:
        rows = [r for s in scenes for r in (depth_check(s, census, log) or [])]
        out = os.path.join(REPO, "MVSeed", "runs",
                           "e4c_depth_check_lin.json" if args.linear else "e4c_depth_check.json")
        med = gate._median([r["rel_median"] for r in rows])
        json.dump({"rows": rows, "rel_median_overall": med}, open(out, "w"), indent=1)
        print(f"depth check: {len(rows)} cameras, median relative difference {med}  -> {out}")
        return
    f0, n0 = gate.membership(json.load(open(gate.spike.F0))["pairs"], json.load(open(gate.spike.N0))["pairs"])
    os.makedirs(args.parts_dir, exist_ok=True)
    if not args.summary_only:
        for s in scenes:
            part = os.path.join(args.parts_dir, f"{s}.json")
            keys = gate.expected_keys(s, cfg, census)
            if gate.load_part(part, keys) is not None and not args.overwrite:
                log(f"  {s}: complete part exists, skipping")
                continue
            t1 = time.time()
            res = e4a.run_scene(s, cfg, census, log, args.save_masks)
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
    summary = e4a.summarize(rows)
    summary["missing_scenes"] = missing
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    doc = {"schema": "MVSeed e4c_project", "schema_version": 1,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head": head,
           "generated_by": "MVSeed/geo/e4c_project.py", "argv": sys.argv,
           "prereg": "docs/stage1-E4c-prereg.md",
           "rules": {"arms": list(e4a.ARMS), "verdict_arm": e4a.VERDICT_ARM, "j_pass": e4a.J_PASS,
                     "occ_tol": warp.OCC_TOL, "close_kernel": warp.CLOSE_KERNEL, "iteration": ITERATION,
                     "depth": "3DGS render, alpha-normalised inverse depth, alpha >= 0.5 (MVSeed/geo/gs_depth.py)",
                     "gate": f"F0 & pinhole & gt_px>=2000: J_{e4a.VERDICT_ARM} > {e4a.J_PASS} "
                             f"for >= {e4a.GATE_MIN} of {e4a.GATE_N}",
                     "unavailable_rows": "kept with an empty projection (J = 0)"},
           "summary": summary, "rows": rows}
    json.dump(doc, open(args.out, "w"), indent=1)
    e4a.print_summary(summary)
    print(f"wrote {args.out}  ({len(rows)} rows, missing {missing}, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
