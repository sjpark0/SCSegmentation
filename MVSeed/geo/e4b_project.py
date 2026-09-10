#!/usr/bin/env python3
"""S1-E4b: the same mask transfer as S1-E4a, but with pi^3 (Pi3X) poses and dense pointmaps.

    docker run --rm --gpus all --shm-size=16g --user $(id -u):$(id -g) -v /:/host -w /host$PWD \\
      -e HF_HOME=/opt/hf -e HF_HUB_OFFLINE=1 scgeo:pi3 python MVSeed/geo/e4b_project.py [--scenes A,B]

Stage 1 (GPU) runs Pi3X once per scene over ALL its cameras with the reference camera at index 0, and
caches the three annotated cameras' depth, confidence, intrinsics and pose under MVSeed/cache/<scene>/pi3/.
Stage 2 reuses MVSeed/geo/warp.py and e4a_project's row enumeration and summary unchanged, so the two
arms differ only in the geometry provider (docs/stage1-E4b-prereg.md section 2).
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
e4a = _load("MVSeed/geo/e4a_project.py", "e4a_project")
gate = e4a.gate

PIXEL_LIMIT = 255_000        # authors' default; both sides rounded to a multiple of 14
PATCH = 14
CONF_MIN = 0.1               # sigmoid(conf) above this counts as valid depth
MODEL_ID = "yyfz233/Pi3X"
OUT = os.path.join(REPO, "MVSeed", "runs", "e4b_project.json")
PARTS = os.path.join(REPO, "MVSeed", "runs", "e4b_project_parts")


def target_size(w, h, pixel_limit=PIXEL_LIMIT, patch=PATCH):
    """Aspect-preserving size under `pixel_limit`, both sides a multiple of `patch` (>= patch)."""
    s = (pixel_limit / float(w * h)) ** 0.5
    return (max(patch, int(round(w * s / patch)) * patch),
            max(patch, int(round(h * s / patch)) * patch))


def cameras_from_pi3(K, pose_c2w, w, h):
    """Pi3X intrinsics/pose of one view -> (colmap_io.Camera at the Pi3 grid, Image with world-to-camera).

    `recover_intrinsic_from_rays_d` fits the principal point in PIXEL-INDEX coordinates, while COLMAP
    (and MVSeed/geo/warp.py) put the centre of pixel (col, row) at (col + 0.5, row + 0.5); the +0.5 below
    converts between the two.  `camera_poses` is camera-to-world, so it is inverted here.
    """
    import numpy as np
    cam = ci.Camera(1, "PINHOLE", w, h, [float(K[0, 0]), float(K[1, 1]),
                                         float(K[0, 2]) + 0.5, float(K[1, 2]) + 0.5])
    R_c2w, t_c2w = np.asarray(pose_c2w)[:3, :3], np.asarray(pose_c2w)[:3, 3]
    R = R_c2w.T
    img = ci.Image(1, rotmat_to_qvec(R), (-R @ t_c2w).tolist(), 1, "pi3")
    return cam, img


def rotmat_to_qvec(R):
    """3x3 rotation -> COLMAP quaternion (qw, qx, qy, qz)."""
    import numpy as np
    R = np.asarray(R, dtype=np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = (t + 1.0) ** 0.5 * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return q


def depth_agreement(pi3_depth, cam_pi3, scene, camname):
    """Scale-aligned median relative difference between the Pi3 depth and COLMAP's MVS depth, or None.

    Diagnostic only (docs/stage1-E4b-prereg.md section 2): the two providers have unrelated global
    scales, so the ratio is normalised by its own median before the error is taken.
    """
    import numpy as np
    p = e4a.scene_paths(scene)
    dm = os.path.join(p["depth"], f"{camname}.jpg.geometric.bin")
    if not (os.path.isdir(p["dense_txt"]) and os.path.exists(dm)):
        return None
    ucams, uimgs = ci.read_model_txt(p["dense_txt"])
    name = f"{camname}.jpg"
    if name not in uimgs:
        return None
    mvs = ci.read_depth_map(dm)
    cam_mvs = ucams[uimgs[name].camera_id]
    h, w = pi3_depth.shape
    u, v = warp._grid(h, w)
    x = (u - cam_pi3.cx) / cam_pi3.fx
    y = (v - cam_pi3.cy) / cam_pi3.fy
    um, ok = warp.sample_nearest(mvs, cam_mvs.fx * x + cam_mvs.cx, cam_mvs.fy * y + cam_mvs.cy)
    good = ok & (um > 0) & (pi3_depth > 0)
    if good.sum() < 100:
        return None
    ratio = um[good] / pi3_depth[good]
    rel = np.abs(ratio / np.median(ratio) - 1.0)
    return dict(n=int(good.sum()), frac=round(float(good.mean()), 4),
                rel_err_median=round(float(np.median(rel)), 4),
                rel_err_p90=round(float(np.percentile(rel, 90)), 4))


def run_pi3(scene, cfg, census, log, pixel_limit=PIXEL_LIMIT):
    """Pi3X over every camera of a scene -> {cam: dict(und, img, depth, conf, orig)} for the annotated three."""
    import cv2
    import numpy as np
    import torch
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d
    d = cfg[scene]
    start, folder = d["start_frame"], d["folder"]
    vdir = os.path.join(e4a.DATA, folder, "Video")
    ref = census[scene]["c_ini"]
    cams = [ref] + sorted(c for c in os.listdir(vdir) if c != ref)
    imgs, hw0 = [], None
    for c in cams:
        im = cv2.cvtColor(cv2.imread(os.path.join(vdir, c, f"{start:06d}.jpg")), cv2.COLOR_BGR2RGB)
        hw0 = im.shape[:2]
        w, h = target_size(hw0[1], hw0[0], pixel_limit)
        imgs.append(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))
    t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()[None] / 255.0
    model = run_pi3.model
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        r = model(t.cuda())
    lp = r["local_points"][0].float()
    conf = torch.sigmoid(r["conf"][0].float())[..., 0]
    K = recover_intrinsic_from_rays_d(torch.nn.functional.normalize(lp, dim=-1),
                                      force_center_principal_point=True).cpu().numpy()
    poses = r["camera_poses"][0].float().cpu().numpy()
    h, w = lp.shape[1], lp.shape[2]
    out = {}
    for i, c in enumerate(cams):
        if c not in census[scene]["cameras"]:
            continue
        depth = lp[i, ..., 2].cpu().numpy().astype("float32")
        cf = conf[i].cpu().numpy()
        depth = np.where((cf > CONF_MIN) & (depth > 0), depth, 0.0).astype("float32")
        und, img = cameras_from_pi3(K[i], poses[i], w, h)
        sx, sy = hw0[1] / float(w), hw0[0] / float(h)
        orig = ci.Camera(2, "PINHOLE", hw0[1], hw0[0],
                         [und.fx * sx, und.fy * sy, und.cx * sx, und.cy * sy])
        out[c] = dict(und=und, orig=orig, img=img, depth=depth,
                      conf_frac=round(float((cf > CONF_MIN).mean()), 4),
                      agreement=depth_agreement(depth, und, scene, c))
    log(f"  {scene:16s} {len(cams)} cams -> {w}x{h}, conf frac "
        + " ".join(f"{c}:{out[c]['conf_frac']}" for c in out))
    del r, lp, conf, t
    torch.cuda.empty_cache()
    return out


def run_scene(scene, cfg, census, log, save_masks=None, pixel_limit=PIXEL_LIMIT):
    """Same measurement as e4a_project.run_scene, with the Pi3 geometry substituted."""
    import numpy as np
    ref_cam = census[scene]["c_ini"]
    cams = census[scene]["cameras"]
    targets = [c for c in cams if c != ref_cam]
    objects = [int(o) for o in census[scene]["seed_ids_c_ini"]]
    gts = {c: e4a.gt_mask(scene, cfg, c) for c in cams}
    h, w = gts[ref_cam].shape
    have = run_pi3(scene, cfg, census, log, pixel_limit)
    maps = {c: e4a.camera_maps(have[c], h, w) for c in have}
    rows = []
    ref_ok = ref_cam in have
    ref_masks = {}
    if ref_ok:
        u2o = maps[ref_cam]["u2o"]
        ref_lab, _ = warp.sample_nearest(gts[ref_cam], u2o[0], u2o[1])
        ref_masks = {o: ref_lab == o for o in objects}
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
                       rig="hemisphere" if scene in e4a.HEMISPHERE else "pinhole",
                       available=bool(pair_ok), ref_registered=bool(ref_ok), tgt_registered=tgt in have,
                       ref_gt_px=int((gts[ref_cam] == obj).sum()),
                       conf_frac=have[tgt]["conf_frac"] if pair_ok else None,
                       depth_agreement=have[tgt]["agreement"] if pair_ok else None)
            masks = {}
            if pair_ok:
                rm = ref_masks[obj]
                masks["back"] = warp.backward_from(corr_b, rm)
                masks["fwd"] = warp.forward_from(corr_f, rm)
                masks["union"] = masks["back"] | masks["fwd"]
                rec["ref_depth_frac"] = round(float(((have[ref_cam]["depth"] > 0) & rm).sum() / max(1, rm.sum())), 4)
                o2u = maps[tgt]["o2u"]
                dt_orig, _ = warp.sample_nearest(have[tgt]["depth"], o2u[0], o2u[1])
                rec["tgt_depth_frac"] = round(float((dt_orig[g] > 0).mean()), 4) if g.any() else None
            for arm in e4a.ARMS:
                m = (warp.to_original(masks[arm], have[tgt]["orig"], have[tgt]["und"], h, w, cache=maps[tgt]["o2u"])
                     if pair_ok else np.zeros((h, w), bool))
                rec[f"px_{arm}"] = int(m.sum())
                j = warp.iou(m, g) if g.any() else None
                rec[f"J_{arm}"] = round(j, 4) if j is not None else None
                rec[f"pass_{arm}"] = bool(j is not None and j > e4a.J_PASS)
                if arm == e4a.VERDICT_ARM:
                    c = warp.centre_offset(m, g) if g.any() else None
                    rec["centre_offset"] = round(c, 3) if c is not None else None
                    if save_masks and g.any():
                        import cv2
                        p = os.path.join(save_masks, scene, tgt, f"{obj}.png")
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        cv2.imwrite(p, (m * 255).astype("uint8"))
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--parts-dir", default=PARTS)
    ap.add_argument("--save-masks", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--pixel-limit", type=int, default=PIXEL_LIMIT,
                    help="registered value is 255000; a larger value is a post-hoc diagnostic, not the arm")
    args = ap.parse_args()
    cfg = json.load(open(e4a.CONFIG, encoding="utf-8"))
    census = json.load(open(e4a.CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(census)
    unknown = [s for s in scenes if s not in census]
    assert not unknown, f"unknown scenes: {unknown}"
    f0, n0 = gate.membership(json.load(open(gate.spike.F0))["pairs"], json.load(open(gate.spike.N0))["pairs"])
    os.makedirs(args.parts_dir, exist_ok=True)
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    if not args.summary_only:
        import torch
        from pi3.models.pi3x import Pi3X
        run_pi3.model = Pi3X.from_pretrained(MODEL_ID).eval().cuda()
        for s in scenes:
            part = os.path.join(args.parts_dir, f"{s}.json")
            keys = gate.expected_keys(s, cfg, census)
            if gate.load_part(part, keys) is not None and not args.overwrite:
                log(f"  {s}: complete part exists, skipping")
                continue
            t1 = time.time()
            res = run_scene(s, cfg, census, log, args.save_masks, args.pixel_limit)
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
    ag = [r["depth_agreement"]["rel_err_median"] for r in rows
          if r.get("depth_agreement") and r["depth_agreement"].get("rel_err_median") is not None]
    summary["depth_agreement_vs_mvs"] = {"n_rows": len(ag), "rel_err_median": gate._median(ag)}
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    doc = {"schema": "MVSeed e4b_project", "schema_version": 1,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head": head,
           "generated_by": "MVSeed/geo/e4b_project.py", "argv": sys.argv,
           "prereg": "docs/stage1-E4b-prereg.md",
           "rules": {"model": MODEL_ID, "pixel_limit": args.pixel_limit, "registered_pixel_limit": PIXEL_LIMIT, "patch": PATCH, "conf_min": CONF_MIN,
                     "arms": list(e4a.ARMS), "verdict_arm": e4a.VERDICT_ARM, "j_pass": e4a.J_PASS,
                     "occ_tol": warp.OCC_TOL, "close_kernel": warp.CLOSE_KERNEL,
                     "gate": f"F0 & pinhole & gt_px>=2000: J_union > {e4a.J_PASS} for >= {e4a.GATE_MIN} of {e4a.GATE_N}"},
           "summary": summary, "rows": rows}
    json.dump(doc, open(args.out, "w"), indent=1)
    e4a.print_summary(summary)
    print(f"  depth agreement vs MVS: {summary['depth_agreement_vs_mvs']}")
    print(f"wrote {args.out}  ({len(rows)} rows, missing {missing}, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
