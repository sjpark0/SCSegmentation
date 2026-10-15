"""S1-E14 round-2 logit dump: reader + the adopted re-threshold (scratch, git-ignored; torch only for masks_torch).

Dump layout (made by dump_logits.py; preset geoset_rep_x on DEV-A, restricted cameras, decode path unchanged):
    <root>/index.json, dump_meta.json, verify_*.json
    <root>/<scene>/<t:06d>/<cam>/<obj>.png                  the adopted final masks (dev_decode layout; absent = empty)
    <root>/<scene>/<t:06d>/_logits/<cam>.npz                 per EVAL camera, round 2, B objects in `objects` order:
        objects  int32 [B]        object ids (= config.json "objects", the c_ini basic-set objects present at t)
        hw       int32 [2]        image (h, w)
        low_res  float16 [B,288,288]  the decoder's low_res logits as decode_scene used them (NO_OBJ_SCORE -1024
                                  where presence <= 0; the stability-selected token) -- out["low_res"][:, 0]
        ious     float32 [B]      predicted IoU of the selected token (out["ious"])
        presence float32 [B]      object score logit (out["presence"]; present iff > 0)
        token    int8 [B]         selected decoder token: 0 = single-mask token (stable), 1..3 = multimask fallback
        stability float32 [B]     stability score of token 0 (IoU of logits > +0.05 vs > -0.05 at 288; >= 0.98 = stable)
    <root>/<scene>/<t:06d>/_logits/<cam>_mm.npz              the raw decoder outputs of all 4 mask tokens:
        mm_low_res float16 [B,4,288,288]  token 0 = single-mask, 1..3 = multimask candidates, BEFORE the NO_OBJ replacement
        mm_ious    float32 [B,4]          their predicted IoUs
    low_res[b] == where(presence[b] > 0, mm_low_res[b, token[b]], -1024)  and  ious[b] == mm_ious[b, token[b]]  (checked).

Adopted re-threshold (run.decode_scene, --xcl out):  low_res (float32) -> bilinear 1008x1008 (align_corners False; the
tracker's own interpolate) -> bilinear to (h, w) (sam3_io.logits_to_original) -> run.exclusive_masks(L, 0.5) -> keep.
masks_torch() is that exact path (byte-identical to the adopted PNGs on CUDA: verify_torch.json); masks_numpy() is the
host emulation of the CUDA kernel in float32 with its FMA contraction (bit-identical logits on the probe views,
probe_emul.json; mask agreement over the whole dump: verify_numpy.json).
"""
import glob
import json
import os

import numpy as np

NO_OBJ_SCORE = -1024.0
IMAGE_SIZE = 1008
XCL_TAU_KEEP = 0.5


def logit_path(root, scene, t, cam, mm=False):
    return os.path.join(root, scene, f"{int(t):06d}", "_logits", f"{cam}{'_mm' if mm else ''}.npz")


def load(root, scene, t, cam, mm=False):
    """-> dict of the numpy arrays of <cam>.npz (and <cam>_mm.npz with mm=True)."""
    with np.load(logit_path(root, scene, t, cam)) as z:
        d = {k: z[k] for k in z.files}
    if mm:
        with np.load(logit_path(root, scene, t, cam, mm=True)) as z:
            d.update({k: z[k] for k in z.files})
    return d


def views(root):
    """[(scene, t, cam)] of every dumped eval camera, sorted."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "_logits", "*.npz"))):
        if p.endswith("_mm.npz"):
            continue
        parts = p.split(os.sep)
        out.append((parts[-4], int(parts[-3]), os.path.basename(p)[:-4]))
    return out


def read_png_masks(mask_root, scene, t, cam, objects, hw):
    """The dev_decode PNGs of one view as bool [B,h,w] (absent PNG = empty)."""
    import cv2
    out = np.zeros((len(objects), *hw), bool)
    for i, o in enumerate(objects):
        p = os.path.join(mask_root, scene, f"{int(t):06d}", cam, f"{int(o)}.png")
        if os.path.isfile(p):
            m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            assert m is not None and m.shape == tuple(hw), (p, None if m is None else m.shape, hw)
            out[i] = m > 0
    return out


# ----------------------------------------------------------------------------------------------- torch (adopted) path
def logits_full_torch(low_res, hw, device="cuda"):
    """low_res [B,288,288] (any float dtype, numpy or torch) -> L [B,h,w] float32 torch, the adopted path."""
    import torch
    import torch.nn.functional as F
    x = torch.as_tensor(np.asarray(low_res, np.float32) if isinstance(low_res, np.ndarray) else low_res)
    x = x.to(device=device, dtype=torch.float32)[:, None]
    hi = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)   # tracker._forward_sam_heads
    return F.interpolate(hi.float(), size=tuple(int(v) for v in hw), mode="bilinear", align_corners=False)[:, 0]  # logits_to_original


def masks_torch(low_res, hw, xcl=True, device="cuda"):
    """-> bool [B,h,w] numpy: threshold 0, then (xcl) run.exclusive_masks with the nested-mask cap 0.5."""
    L = logits_full_torch(low_res, hw, device)
    if not xcl:
        return (L > 0).cpu().numpy()
    from methods.geomem.run import exclusive_masks
    keep, _, _ = exclusive_masks(L, XCL_TAU_KEEP)
    return keep.cpu().numpy()


# ----------------------------------------------------------------------------------------------- numpy host emulation
def _fma(a, b, c):
    """float32 fused multiply-add (a*b exact in float64, one rounding to float32) -- nvcc contracts a*b+c to FMA."""
    return (np.asarray(a, np.float64) * np.asarray(b, np.float64) + np.asarray(c, np.float64)).astype(np.float32)


def _axis(in_size, out_size, fma=True):
    """PyTorch CUDA bilinear (align_corners False) source taps along one axis, float32:
    scale = float(in)/out; src = max(fma(scale, dst + 0.5, -0.5), 0); i0 = int(src); i1 = i0 + (i0 < in-1); l1 = src - i0."""
    f32 = np.float32
    scale = f32(in_size) / f32(out_size)
    dst = np.arange(out_size, dtype=f32)
    src = _fma(scale, dst + f32(0.5), f32(-0.5)) if fma else (scale * (dst + f32(0.5)) - f32(0.5)).astype(f32)
    src = np.where(src < 0, f32(0), src).astype(f32)
    i0 = src.astype(np.int64)
    i1 = i0 + (i0 < in_size - 1)
    l1 = (src - i0.astype(f32)).astype(f32)
    l0 = (f32(1) - l1).astype(f32)
    return i0, i1, l0, l1


def resize_bilinear_np(x, out_hw, fma=True):
    """x [B,h,w] float32 -> [B,H,W] float32 as the CUDA kernel: fma(h0l, fma(w0l, a, w1l*b), h1l*fma(w0l, c, w1l*d)).
    fma=True is bit-identical to torch 2.10 CUDA F.interpolate (probe_emul.json); fma=False = plain float32 order."""
    x = np.asarray(x, np.float32)
    H, W = out_hw
    hi0, hi1, h0l, h1l = _axis(x.shape[1], H, fma)
    wi0, wi1, w0l, w1l = _axis(x.shape[2], W, fma)
    out = np.empty((x.shape[0], H, W), np.float32)
    for b in range(x.shape[0]):                                                     # per object (float64 temporaries)
        if fma:
            rows = _fma(w0l[None], x[b][:, wi0], w1l[None] * x[b][:, wi1])          # [h,W]
            out[b] = _fma(h0l[:, None], rows[hi0], h1l[:, None] * rows[hi1])
        else:
            rows = w0l[None] * x[b][:, wi0] + w1l[None] * x[b][:, wi1]
            out[b] = h0l[:, None] * rows[hi0] + h1l[:, None] * rows[hi1]
    return out


def logits_full_np(low_res, hw, fma=True):
    """low_res [B,288,288] -> L [B,h,w] float32 on the host (the adopted two bilinear stages)."""
    return resize_bilinear_np(resize_bilinear_np(np.asarray(low_res, np.float32), (IMAGE_SIZE, IMAGE_SIZE), fma), tuple(int(v) for v in hw), fma)


def exclusive_masks_np(L, tau_keep=XCL_TAU_KEEP):
    """run.exclusive_masks in numpy (argmax ties -> lowest index, as torch)."""
    B = L.shape[0]
    pos = L > 0
    w = L.argmax(0)
    win = w[None] == np.arange(B)[:, None, None]
    npos = pos.reshape(B, -1).sum(1)
    lost = (pos & ~win).reshape(B, -1).sum(1)
    exempt = np.zeros(B, bool) if tau_keep is None else (npos > 0) & (lost.astype(np.float32) >= np.float32(tau_keep) * npos.astype(np.float32))
    return pos & (win | exempt[:, None, None])


def masks_numpy(low_res, hw, xcl=True, fma=True):
    """Host version of masks_torch (bit-identical logits with fma=True; see verify_numpy.json)."""
    L = logits_full_np(low_res, hw, fma)
    return exclusive_masks_np(L) if xcl else L > 0


def dev_pairs(pairs_json):
    """{(scene, t, cam, obj)} of the scored DEV pairs (dev_score rows)."""
    rows = json.load(open(pairs_json))["rows"]
    return {(r["scene"], int(r["t"]), r["cam"], int(r["obj"])) for r in rows}
