#!/usr/bin/env python3
"""Which float32 op order reproduces the CUDA bilinear interpolate bit for bit?  (scsam3, GPU; a few dumped views)
Each stage is tested on its own input (stage 2 gets the GPU stage-1 output), bitwise and by sign (> 0)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, S1)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e14dump as E  # noqa: E402

f32 = np.float32


def fma(a, b, c):
    return (np.asarray(a, np.float64) * np.asarray(b, np.float64) + np.asarray(c, np.float64)).astype(f32)


def axis(n_in, n_out, src_fma):
    scale = f32(n_in) / f32(n_out)
    dst = np.arange(n_out, dtype=f32)
    src = fma(scale, dst + f32(0.5), f32(-0.5)) if src_fma else (scale * (dst + f32(0.5)) - f32(0.5)).astype(f32)
    src = np.where(src < 0, f32(0), src).astype(f32)
    i0 = src.astype(np.int64)
    i1 = i0 + (i0 < n_in - 1)
    l1 = (src - i0.astype(f32)).astype(f32)
    return i0, i1, (f32(1) - l1).astype(f32), l1


def lerp(l0, a, l1, b, mode):
    if mode == "plain":
        return (l0 * a + l1 * b).astype(f32)
    if mode == "fma0":                      # fma(l0, a, l1*b)
        return fma(l0, a, (l1 * b).astype(f32))
    if mode == "fma1":                      # fma(l1, b, l0*a)
        return fma(l1, b, (l0 * a).astype(f32))
    raise ValueError(mode)


def resize(x, out_hw, src_fma, inner, outer):
    H, W = out_hw
    hi0, hi1, h0, h1 = axis(x.shape[1], H, src_fma)
    wi0, wi1, w0, w1 = axis(x.shape[2], W, src_fma)
    out = np.empty((x.shape[0], H, W), f32)
    for b in range(x.shape[0]):
        top = lerp(w0[None], x[b][:, wi0], w1[None], x[b][:, wi1], inner)        # [h, W] rows of the input
        out[b] = lerp(h0[:, None], top[hi0], h1[:, None], top[hi1], outer)
    return out


def main():
    root = os.path.join(S1, "geomem", "runs", "_e14_logitdump", "geoset_rep_x_A")
    vs = E.views(root)[:int(sys.argv[1]) if len(sys.argv) > 1 else 4]
    variants = [(s, i, o) for s in (False, True) for i in ("plain", "fma0", "fma1") for o in ("plain", "fma0", "fma1")]
    res = {}
    for scene, t, cam in vs:
        z = E.load(root, scene, t, cam)
        x = z["low_res"][:4].astype(f32)                       # first 4 objects of the view (cost)
        hw = tuple(int(v) for v in z["hw"])
        xt = torch.from_numpy(x).cuda()[:, None]
        g1 = F.interpolate(xt, size=(1008, 1008), mode="bilinear", align_corners=False)
        g2 = F.interpolate(g1, size=hw, mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        g1 = g1[:, 0].cpu().numpy()
        for v in variants:
            k = "src_fma=%d inner=%s outer=%s" % v
            e1 = resize(x, (1008, 1008), *v)
            e2 = resize(g1, hw, *v)
            r = res.setdefault(k, dict(s1_bits=0, s1_sign=0, s2_bits=0, s2_sign=0, n1=0, n2=0))
            r["s1_bits"] += int((e1.view(np.uint32) != g1.view(np.uint32)).sum())
            r["s1_sign"] += int(((e1 > 0) != (g1 > 0)).sum())
            r["s2_bits"] += int((e2.view(np.uint32) != g2.view(np.uint32)).sum())
            r["s2_sign"] += int(((e2 > 0) != (g2 > 0)).sum())
            r["n1"] += e1.size
            r["n2"] += e2.size
        print(scene, t, cam, flush=True)
    for k, r in sorted(res.items(), key=lambda kv: kv[1]["s1_bits"] + kv[1]["s2_bits"]):
        print(k, r)
    with open(os.path.join(root, "probe_emul.json"), "w") as f:
        json.dump(dict(views=[list(v) for v in vs], torch=torch.__version__, results=res), f, indent=1)


if __name__ == "__main__":
    main()
