#!/usr/bin/env python3
"""Baseline: run SAM 3 on every view independently, with no cross-view link.

Each of the 32 first frames gets its own SAM 3 image session and the same text
prompt.  Nothing carries object identity from one view to the next, so the id a
person receives is just their rank in that view's detection list - which is what
makes the colours disagree across views.

Run inside the scsam3 container:

  docker run --rm -v /:/host -w /host/home/sjpark/Documents/SCSegmentation/figures \\
      scsam3 python run_independent_baseline.py --device cpu

Outputs <out>/{view}.png overlays plus <out>/detections.json.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# same camera ordering the demo uses
PERMS = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23,
         24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13, 14, 15]

# matplotlib tab10, so the overlays match the ones misc.show_mask_cv produces
TAB10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
         (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
         (188, 189, 34), (23, 190, 207)]


def first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {path}")
    return frame                                   # BGR, HxWx3 uint8


def overlay(image_bgr, masks, alpha=0.6):
    """Colour every mask by its index in this view's detection list."""
    out = image_bgr.astype(np.float32)
    for i, m in enumerate(masks):
        color = np.array(TAB10[i % len(TAB10)][::-1], dtype=np.float32)  # -> BGR
        sel = m.astype(bool)
        out[sel] = color * alpha + out[sel] * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


# SAM 3 assumes a GPU in a few places that have nothing to do with the maths:
# constant buffers allocated with a hardcoded device="cuda" (the
# positional-encoding cache, the decoder's coordinate grid) and a pin_memory()
# staging copy in the geometry encoder.  On CPU those are redirected / made
# no-ops; the model itself is untouched.
_FACTORIES = ("zeros", "ones", "arange", "empty", "full", "tensor", "linspace",
              "eye", "rand", "randn")


def enable_cpu_mode(device):
    def redirect(fn):
        def wrapper(*a, **kw):
            if str(kw.get("device", "")).startswith("cuda"):
                kw["device"] = device
            return fn(*a, **kw)
        return wrapper

    for n in _FACTORIES:
        setattr(torch, n, redirect(getattr(torch, n)))
    torch.Tensor.pin_memory = lambda self, *a, **kw: self


def build_model(device):
    if device != "cuda":
        enable_cpu_mode(device)
    return build_sam3_image_model(device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="/host/home/sjpark/Documents/SCSegmentation/Data/VideoSample_1")
    ap.add_argument("--out", default="/host/home/sjpark/Documents/SCSegmentation/figures/baseline_independent")
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--views", type=int, default=len(PERMS))
    ap.add_argument("--width", type=int, default=1920, help="width of the saved overlay")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    names = sorted(p for p in os.listdir(args.videos)
                   if os.path.splitext(p)[-1].lower() in (".mp4", ".mov"))
    print(f"{len(names)} videos in {args.videos}", flush=True)

    t0 = time.time()
    model = build_model(args.device)
    processor = Sam3Processor(model, device=args.device,
                              confidence_threshold=args.threshold)
    print(f"model ready in {time.time() - t0:.0f}s on {args.device}", flush=True)

    report = {}
    for m in range(args.views):
        t = time.time()
        frame = first_frame(os.path.join(args.videos, names[PERMS[m]]))
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        state = processor.set_image(pil)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=args.prompt, state=state)

        masks = state["masks"].squeeze(1).cpu().numpy()
        scores = state["scores"].float().cpu().numpy().tolist()

        vis = overlay(frame, masks)
        h = round(vis.shape[0] * args.width / vis.shape[1])
        cv2.imwrite(os.path.join(args.out, f"{m}.png"),
                    cv2.resize(vis, (args.width, h), interpolation=cv2.INTER_AREA))

        report[m] = {"camera": PERMS[m], "file": names[PERMS[m]],
                     "n": int(masks.shape[0]),
                     "scores": [round(s, 4) for s in scores],
                     "areas": [int(mm.sum()) for mm in masks]}
        print(f"view {m:2d} (cam {PERMS[m]:02d}): {masks.shape[0]} objects "
              f"in {time.time() - t:.1f}s", flush=True)

    with open(os.path.join(args.out, "detections.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
