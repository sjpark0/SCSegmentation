"""Rig calibration probe (input only): run Pi3X on the target scene's 32 images PLUS anchor images of the same fixed rig
(another instant of the same take, and/or T06 #900), then keep only the target cameras' geometry in
Data/<scene>/Cache/pi3_multi_<tag>/ (same format as Cache/pi3, MANIFEST lists the anchors).  The idea: the ring is the
same physical rig, so extra instants with more structure (people) can anchor the registration that fails on a sparse
instant (docs/croom-dev-T04.md 2).  Uses no ground truth and no predictions.

    python stage1/scripts/croom/pi3_multi.py --scene CROOM_T04_001000_r00 --tag t1819 \
        --anchor annot:T04_sidestep_out:1819 [--anchor scene:T06_flag_ganggang_F] [--anchor-step 2]
"""
import argparse
import json
import os
import sys

S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, S1)
sys.path.insert(0, os.path.join(S1, "methods"))
import common  # noqa: E402
from pi3_reproj.geometry import CONF_MIN, MODEL_ID, PIXEL_LIMIT, target_size  # noqa: E402


def anchor_paths(spec, data, step):
    kind, rest = spec.split(":", 1)
    if kind == "annot":
        take, frame = rest.split(":")
        d = os.path.join(data, "_annot", "CROOM", take, f"{int(frame):06d}", "images")
        ks = sorted((int(f[:-4]) for f in os.listdir(d) if f.endswith(".jpg")))
        return [(f"{spec}/cam_{k:02d}", os.path.join(d, f"{k}.jpg")) for k in ks[::step]]
    if kind == "scene":
        meta = common.load_meta(data, rest)
        return [(f"{spec}/{c}", os.path.join(data, rest, "images", f"{c}.jpg")) for c in list(meta["cams"])[::step]]
    raise ValueError(spec)


def main():
    import cv2
    import numpy as np
    import torch
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default=os.path.join(S1, "Data"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--anchor", action="append", required=True, help="annot:<take>:<frame> | scene:<scene>")
    ap.add_argument("--anchor-step", type=int, default=1, help="use every k-th anchor camera")
    ap.add_argument("--pixel-limit", type=int, default=PIXEL_LIMIT)
    a = ap.parse_args()

    sc = common.load_scene(a.data, a.scene)
    cams = [sc.ref_cam] + [c for c in sc.cams if c != sc.ref_cam]
    items = [(c, sc.image_paths[c]) for c in cams]
    for spec in a.anchor:
        items += anchor_paths(spec, a.data, a.anchor_step)
    w, h = target_size(sc.hw[1], sc.hw[0], a.pixel_limit)
    imgs = []
    for _, p in items:
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        assert im.shape[:2] == sc.hw, (p, im.shape)
        imgs.append(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))
    t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()[None] / 255.0
    print(f"{a.scene}: {len(cams)} target + {len(items) - len(cams)} anchor images at {w}x{h}", flush=True)

    model = Pi3X.from_pretrained(MODEL_ID).eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        r = model(t.cuda())
    peak = torch.cuda.max_memory_allocated() / 2**30
    lp = r["local_points"][0].float()
    conf = torch.sigmoid(r["conf"][0].float())[..., 0]
    K = recover_intrinsic_from_rays_d(torch.nn.functional.normalize(lp, dim=-1),
                                      force_center_principal_point=True).cpu().numpy()
    poses = r["camera_poses"][0].float().cpu().numpy()

    out = os.path.join(a.data, a.scene, "Cache", f"pi3_multi_{a.tag}")
    os.makedirs(out, exist_ok=True)
    conf_frac = {}
    for i, c in enumerate(cams):                       # target cameras only (input indices 0..31)
        depth = lp[i, ..., 2].cpu().numpy().astype("float32")
        cf = conf[i].cpu().numpy().astype("float32")
        depth = np.where((cf > CONF_MIN) & (depth > 0), depth, 0.0).astype("float32")
        np.savez(os.path.join(out, f"{c}.npz"), depth=depth, conf=cf, K=K[i].astype("float64"), c2w=poses[i].astype("float64"))
        conf_frac[c] = round(float((cf > CONF_MIN).mean()), 4)
    anchors_c2w = {name: poses[len(cams) + j].tolist() for j, (name, _) in enumerate(items[len(cams):])}
    manifest = {"model": MODEL_ID, "pixel_limit": a.pixel_limit, "work_hw": [int(h), int(w)], "orig_hw": list(sc.hw),
                "cams": cams, "ref_cam": sc.ref_cam, "conf_min": CONF_MIN, "conf_frac": conf_frac,
                "anchors": [n for n, _ in items[len(cams):]], "anchor_step": a.anchor_step,
                "anchors_c2w": anchors_c2w, "peak_gib": round(peak, 2)}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    print(f"wrote {out} ({len(cams)} cams, peak {peak:.1f} GiB)", flush=True)


if __name__ == "__main__":
    main()
