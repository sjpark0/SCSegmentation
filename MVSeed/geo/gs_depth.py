#!/usr/bin/env python3
"""S1-E4c plumbing: render metric depth from a trained 3DGS at the annotated cameras.

The rasterizer returns an *unnormalised* inverse-depth expectation, `sum_i (1/d_i) * alpha_i * T_i`
(`diff-gaussian-rasterization/cuda_rasterizer/forward.cu`), so a single render with every Gaussian
colour forced to 1 gives both halves at once: the colour channel accumulates `sum_i 1 * alpha_i * T_i`
= the accumulated opacity, and the depth channel accumulates the inverse depth.  Metric depth is
their ratio.  A pixel whose accumulated opacity stays below ALPHA_MIN is written as 0 = "no depth",
the same convention dense MVS holes get in S1-E4a (docs/stage1-E4c-prereg.md section 2).

    docker run --rm --gpus all --user $(id -u):$(id -g) -v /:/host -w /host$HOME/Documents/gaussian-splatting \
        3dgs python /host$PWD/MVSeed/geo/gs_depth.py --scenes Barn,FlameSteak

Writes MVSeed/cache/<scene>/gs/depth/<image name>.npz (depth float32, alpha float16) and depth/index.json.
Runs in the `3dgs` container: needs torch, the CUDA rasteriser and the gaussian-splatting repo on sys.path.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GS_REPO = os.environ.get("GS_REPO", os.path.join(os.path.expanduser("~"), "Documents", "gaussian-splatting"))
CACHE = os.path.join(REPO, "MVSeed", "cache")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
ITERATION = 30000
ALPHA_MIN = 0.5       # a rendered pixel carries depth only once it is this opaque
SH_DEGREE = 3
ZNEAR, ZFAR = 0.01, 100.0


def camera_list(scene):
    """(name, width, height, fovx, fovy, R, T) for every image of the undistorted model."""
    import numpy as np
    sys.path.insert(0, GS_REPO)
    from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary
    from utils.graphics_utils import focal2fov
    sparse = os.path.join(CACHE, scene, "undist", "sparse", "0")
    extr = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    out = []
    for e in extr.values():
        c = intr[e.camera_id]
        assert c.model == "PINHOLE", c.model
        fx, fy = c.params[0], c.params[1]
        out.append(dict(name=e.name, width=c.width, height=c.height,
                        fovx=focal2fov(fx, c.width), fovy=focal2fov(fy, c.height),
                        R=np.transpose(qvec2rotmat(e.qvec)), T=np.array(e.tvec)))
    return out


def metric_depth(alpha, inv, alpha_min=ALPHA_MIN):
    """Metric depth from the two accumulators: alpha = sum a_i T_i, inv = sum (1/d_i) a_i T_i.

    Their ratio is the alpha-normalised depth expectation.  A pixel with too little accumulated
    opacity, or with no inverse depth at all, is written as 0 = "no depth" (numpy in, numpy out)."""
    import numpy as np
    ok = (inv > 0) & (alpha >= alpha_min)
    return np.where(ok, alpha / np.where(inv > 0, inv, 1.0), 0.0).astype("float32")


def render_depth(gaussians, cam, pipe, background):
    """Metric depth (float32, 0 where invalid) and accumulated opacity for one camera."""
    import torch
    from gaussian_renderer import render
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2
    w2v = torch.tensor(getWorld2View2(cam["R"], cam["T"])).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=ZNEAR, zfar=ZFAR, fovX=cam["fovx"], fovY=cam["fovy"]).transpose(0, 1).cuda()
    view = MiniCam(cam["width"], cam["height"], cam["fovy"], cam["fovx"], ZNEAR, ZFAR,
                   w2v, (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0))
    ones = torch.ones_like(gaussians.get_xyz)
    with torch.no_grad():
        out = render(view, gaussians, pipe, background, override_color=ones)
    alpha = out["render"][0].cpu().numpy().astype("float64")
    inv = out["depth"][0].cpu().numpy().astype("float64")
    return metric_depth(alpha, inv), alpha.astype("float16")


def render_depth_linear(gaussians, cam, pipe, background, alpha):
    """Diagnostic read-out: the *linear* depth expectation, sum_i d_i alpha_i T_i, normalised by alpha.

    The registered read-out is the inverse-depth expectation the rasteriser accumulates natively; it
    weights near contributions by 1/d and so is pulled towards floaters.  Feeding each Gaussian's own
    view-space z in as its colour makes the same rasteriser accumulate the linear expectation instead,
    with no change to the CUDA code.  Report-only (docs/stage1-E4c.md), never the verdict arm.
    """
    import numpy as np
    import torch
    from gaussian_renderer import render
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2
    w2v_np = getWorld2View2(cam["R"], cam["T"])
    w2v = torch.tensor(w2v_np).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=ZNEAR, zfar=ZFAR, fovX=cam["fovx"], fovY=cam["fovy"]).transpose(0, 1).cuda()
    view = MiniCam(cam["width"], cam["height"], cam["fovy"], cam["fovx"], ZNEAR, ZFAR,
                   w2v, (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0))
    xyz = gaussians.get_xyz
    row = torch.tensor(w2v_np[2, :3], dtype=xyz.dtype, device=xyz.device)
    z = (xyz @ row + float(w2v_np[2, 3])).clamp(min=0.0)
    zmax = float(z.max())                       # the renderer clamps its output to [0, 1], so carry
    colour = (z / zmax).unsqueeze(1).repeat(1, 3)   # z/zmax through it and scale back afterwards
    with torch.no_grad():
        acc = render(view, gaussians, pipe, background, override_color=colour)["render"][0]
    acc = acc.cpu().numpy().astype("float64") * zmax
    ok = alpha >= ALPHA_MIN
    return np.where(ok, acc / np.where(alpha > 0, alpha, 1.0), 0.0).astype("float32")


def render_rgb_psnr(gaussians, cam, pipe, background, image_path):
    """Plumbing check: PSNR between the render at this camera and the image the model was trained on.

    A low value means the camera was rebuilt wrongly (pose, intrinsics or convention); a high value
    means the camera is right and anything odd in the depth belongs to the model, not to this script.
    """
    import numpy as np
    import torch
    from PIL import Image
    from gaussian_renderer import render
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2
    w2v = torch.tensor(getWorld2View2(cam["R"], cam["T"])).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=ZNEAR, zfar=ZFAR, fovX=cam["fovx"], fovY=cam["fovy"]).transpose(0, 1).cuda()
    view = MiniCam(cam["width"], cam["height"], cam["fovy"], cam["fovx"], ZNEAR, ZFAR,
                   w2v, (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0))
    with torch.no_grad():
        out = render(view, gaussians, pipe, background)["render"].clamp(0, 1)
    gt = np.asarray(Image.open(image_path).convert("RGB"), dtype="float32") / 255.0
    pred = out.permute(1, 2, 0).cpu().numpy()
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    mse = float(((pred - gt) ** 2).mean())
    return 10.0 * float(np.log10(1.0 / max(mse, 1e-12)))


def run_scene(scene, cams_wanted, log=print, psnr_check=False, linear=False):
    import numpy as np
    import torch
    sys.path.insert(0, GS_REPO)
    from argparse import Namespace
    from scene.gaussian_model import GaussianModel
    ply = os.path.join(CACHE, scene, "gs", "point_cloud", f"iteration_{ITERATION}", "point_cloud.ply")
    if not os.path.exists(ply):
        log(f"{scene:16s} no trained model, skipping")
        return None
    gaussians = GaussianModel(SH_DEGREE)
    gaussians.load_ply(ply)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    outdir = os.path.join(CACHE, scene, "gs", "depth")
    os.makedirs(outdir, exist_ok=True)
    index = {}
    for cam in camera_list(scene):
        stem = os.path.splitext(cam["name"])[0]
        if cams_wanted is not None and stem not in cams_wanted:
            continue
        depth, alpha = render_depth(gaussians, cam, pipe, background)
        if linear:
            lin = render_depth_linear(gaussians, cam, pipe, background, alpha.astype("float64"))
            os.makedirs(outdir + "_lin", exist_ok=True)
            np.savez_compressed(os.path.join(outdir + "_lin", f"{stem}.npz"), depth=lin)
        psnr = None
        if psnr_check:
            psnr = round(render_rgb_psnr(gaussians, cam, pipe, background,
                                         os.path.join(CACHE, scene, "undist", "images", cam["name"])), 2)
        np.savez_compressed(os.path.join(outdir, f"{stem}.npz"), depth=depth, alpha=alpha)
        valid = float((depth > 0).mean())
        med = float(np.median(depth[depth > 0])) if valid else 0.0
        index[stem] = dict(name=cam["name"], width=cam["width"], height=cam["height"],
                           valid_frac=round(valid, 4), depth_median=round(med, 4), psnr_train=psnr)
        log(f"{scene:16s} {stem:14s} {cam['width']}x{cam['height']}  valid {valid:.3f}  "
            f"median depth {med:.2f}" + (f"  PSNR {psnr}" if psnr is not None else ""))
    index["_meta"] = dict(iteration=ITERATION, alpha_min=ALPHA_MIN, n_gaussians=int(gaussians.get_xyz.shape[0]))
    json.dump(index, open(os.path.join(outdir, "index.json"), "w"), indent=1)
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default=None, help="comma separated; default = every scene with a trained model")
    ap.add_argument("--all-cameras", action="store_true", help="render every image, not just the annotated three")
    ap.add_argument("--psnr", action="store_true", help="also report the train-view PSNR (camera plumbing check)")
    ap.add_argument("--linear", action="store_true", help="also write the linear-depth diagnostic to gs/depth_lin")
    args = ap.parse_args()
    census = json.load(open(CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(
        s for s in census if os.path.exists(os.path.join(CACHE, s, "gs", "summary.json")))
    for s in scenes:
        run_scene(s, None if args.all_cameras else set(census[s]["cameras"]),
                  psnr_check=args.psnr, linear=args.linear)


if __name__ == "__main__":
    main()
