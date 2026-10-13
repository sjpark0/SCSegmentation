"""Stage A of pi3_reproj: run Pi3X once over all cameras of a scene and cache the geometry.

Writes Data/<scene>/Cache/pi3/<cam>.npz with
    depth  float32 [h, w]  z of the local pointmap at the work resolution, 0 where conf <= CONF_MIN
    conf   float32 [h, w]  sigmoid confidence
    K      float64 [3, 3]  intrinsics at the work resolution (pixel-index principal point)
    c2w    float64 [4, 4]  camera-to-world pose
and Cache/pi3/MANIFEST.json.  The reference camera is input index 0 (as in S1-E4b).  Runs in
`scgeo:pi3`; the model call, `target_size` and the intrinsics recovery follow
../SCSegmentation/MVSeed/geo/e4b_project.py.
"""
import json
import os

MODEL_ID = "yyfz233/Pi3X"
PATCH = 14
CONF_MIN = 0.1          # sigmoid(conf) above this counts as valid depth (E4b)
PIXEL_LIMIT = 255_000   # Pi3 authors' default working resolution


def target_size(w, h, pixel_limit=PIXEL_LIMIT, patch=PATCH):
    """Aspect-preserving (w, h) under `pixel_limit`, both multiples of `patch` (>= patch)."""
    s = (pixel_limit / float(w * h)) ** 0.5
    return (max(patch, int(round(w * s / patch)) * patch),
            max(patch, int(round(h * s / patch)) * patch))


def run_pi3(scene, out_dir, pixel_limit=PIXEL_LIMIT):
    import cv2
    import numpy as np
    import torch
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d

    cams = [scene.ref_cam] + [c for c in scene.cams if c != scene.ref_cam]
    imgs = []
    for c in cams:
        im = cv2.cvtColor(cv2.imread(scene.image_paths[c]), cv2.COLOR_BGR2RGB)
        assert im.shape[:2] == scene.hw, (c, im.shape, scene.hw)
        w, h = target_size(scene.hw[1], scene.hw[0], pixel_limit)
        imgs.append(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))
    t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()[None] / 255.0

    model = run_pi3.model
    if model is None:
        model = Pi3X.from_pretrained(MODEL_ID).eval().cuda()
        run_pi3.model = model
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        r = model(t.cuda())
    lp = r["local_points"][0].float()
    conf = torch.sigmoid(r["conf"][0].float())[..., 0]
    K = recover_intrinsic_from_rays_d(torch.nn.functional.normalize(lp, dim=-1),
                                      force_center_principal_point=True).cpu().numpy()
    poses = r["camera_poses"][0].float().cpu().numpy()
    h, w = lp.shape[1], lp.shape[2]

    os.makedirs(out_dir, exist_ok=True)
    conf_frac = {}
    for i, c in enumerate(cams):
        depth = lp[i, ..., 2].cpu().numpy().astype("float32")
        cf = conf[i].cpu().numpy().astype("float32")
        depth = np.where((cf > CONF_MIN) & (depth > 0), depth, 0.0).astype("float32")
        np.savez(os.path.join(out_dir, f"{c}.npz"), depth=depth, conf=cf,
                 K=K[i].astype("float64"), c2w=poses[i].astype("float64"))
        conf_frac[c] = round(float((cf > CONF_MIN).mean()), 4)
    manifest = {"model": MODEL_ID, "pixel_limit": pixel_limit, "work_hw": [int(h), int(w)],
                "orig_hw": list(scene.hw), "cams": cams, "ref_cam": scene.ref_cam,
                "conf_min": CONF_MIN, "conf_frac": conf_frac}
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    del r, lp, conf, t
    torch.cuda.empty_cache()
    return manifest


run_pi3.model = None
