#!/usr/bin/env python3
"""S1-E13 candidate E1 (text-concept), GPU part: SAM 3 concept (text) detections with each seeded object's CLASS NAME.

PROTOCOL CHANGE (separate track): the label names come from the annotation metadata (objects_labels.json), which is not
part of MUVOD's input.  DEV ONLY (geomem/runs/_e13_dev/index.json frames; asserts t != ref_frame).  Design: REGISTRATION.txt.
Per dev view (scene, t, cam): Sam3Processor(model.detector).set_image(target) once; per distinct prompt string of the
seeded objects (split and joined name forms) set_text_prompt -> detections with prob > 0.3 (full-res masks, sigmoid > 0.5).
No GT of the target is read here.
Writes <out>/<scene>/<t:06d>/<cam>/{dets.npz, meta.json, DONE}
    dets.npz: packed [N, ceil(H*W/8)] uint8 (np.packbits of the row-major mask), prompt [N] int, prob [N] f4, box [N,4] f4
    flock <lock> stage1/scripts/docker_run.sh scsam3 python geomem/runs/_e13_text-concept/detect.py [--scenes A,B]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, S1)
import cv2  # noqa: E402

from methods import sam3_single  # noqa: E402
from mvseg2d.io import load_labelmap, load_labels, load_meta  # noqa: E402

DATA = os.path.join(S1, "Data")
IDX = os.path.join(S1, "geomem", "runs", "_e13_dev", "index.json")
STORE_TAU = 0.3


def _clean(s):
    s = s.replace("_", " ").strip().lower()
    if s.startswith("other "):
        s = s[len("other "):]
    return s


def prompt_strings(label):
    """REGISTRATION rule: split form (primary) and joined form."""
    lab = label.lower()
    split = [_clean(p) for p in lab.split("_or_")]
    joined = [_clean(lab)]
    return split, joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--out", default=os.path.join(HERE, "dets"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    idx = json.load(open(IDX))
    scenes = sorted(idx["scenes"]) if a.scenes == "all" else a.scenes.split(",")

    import torch
    from PIL import Image
    t0 = time.time()
    model = sam3_single.load_model()
    from sam3.model.sam3_image_processor import Sam3Processor
    det = model.detector
    proc = Sam3Processor(det, resolution=1008, device="cuda", confidence_threshold=STORE_TAU)
    print(f"model {time.time() - t0:.1f}s", flush=True)

    with torch.inference_mode():
        for scene in scenes:
            Sc = idx["scenes"][scene]
            ref, basic = Sc["ref_cam"], [int(o) for o in Sc["basic"]]
            assert int(load_meta(DATA, scene)["ref_frame"]) == Sc["ref_frame"]
            labels = load_labels(DATA, scene)
            for t in Sc["frames"]:
                assert t != Sc["ref_frame"], (scene, t)
                fdir = os.path.join(DATA, scene, "frames", f"{t:06d}")
                ref_lab = load_labelmap(os.path.join(fdir, "masks", f"{ref}.png"))
                objs = [o for o in basic if (ref_lab == o).any()]
                obj_prompts = {}
                for o in objs:
                    sp, jo = prompt_strings(labels[o]["label"])
                    obj_prompts[o] = dict(label=labels[o]["label"], type=labels[o].get("type"), split=sp, joined=jo)
                strings = sorted({s for v in obj_prompts.values() for s in v["split"] + v["joined"]})
                for cam in Sc["eval_cams"]:
                    vdir = os.path.join(a.out, scene, f"{t:06d}", cam)
                    if os.path.isfile(os.path.join(vdir, "DONE")) and not a.overwrite:
                        print(f"{scene} {t} {cam} skip", flush=True)
                        continue
                    os.makedirs(vdir, exist_ok=True)
                    ts = time.time()
                    img = cv2.cvtColor(cv2.imread(os.path.join(fdir, "images", f"{cam}.jpg")), cv2.COLOR_BGR2RGB)
                    H, W = img.shape[:2]
                    st = proc.set_image(Image.fromarray(img))
                    packed, pidx, probs, boxes, maxprob, ndet = [], [], [], [], {}, {}
                    for i, s in enumerate(strings):
                        proc.reset_all_prompts(st)
                        st = proc.set_text_prompt(s, st)
                        m = st["masks"][:, 0]
                        n = int(m.shape[0])
                        ndet[s] = n
                        maxprob[s] = float(st["scores"].max().item()) if n else None
                        if n:
                            mc = m.cpu().numpy().reshape(n, -1)
                            packed.append(np.packbits(mc, axis=1))
                            pidx += [i] * n
                            probs.append(st["scores"].float().cpu().numpy())
                            boxes.append(st["boxes"].float().cpu().numpy())
                    proc.reset_all_prompts(st)
                    nb = (H * W + 7) // 8
                    np.savez_compressed(
                        os.path.join(vdir, "dets.npz"),
                        packed=np.concatenate(packed, 0) if packed else np.zeros((0, nb), np.uint8),
                        prompt=np.array(pidx, np.int32),
                        prob=np.concatenate(probs).astype(np.float32) if probs else np.zeros(0, np.float32),
                        box=np.concatenate(boxes, 0).astype(np.float32) if boxes else np.zeros((0, 4), np.float32))
                    meta = dict(scene=scene, t=t, cam=cam, ref_cam=ref, hw=[H, W], objs=objs,
                                obj_prompts={str(o): v for o, v in obj_prompts.items()}, strings=strings,
                                n_det=ndet, max_prob=maxprob, store_tau=STORE_TAU, seconds=time.time() - ts)
                    with open(os.path.join(vdir, "meta.json"), "w") as f:
                        json.dump(meta, f)
                    open(os.path.join(vdir, "DONE"), "w").close()
                    del st
                    torch.cuda.empty_cache()
                    print(f"{scene:20s} t={t:4d} {cam:12s} objs {len(objs):2d} strings {len(strings):2d} "
                          f"dets {len(pidx):4d} {meta['seconds']:5.1f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB",
                          flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
