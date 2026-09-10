#!/usr/bin/env python3
"""S2-R1' guard: revert crop-tracked pairs whose frame-1 mask does not overlap the seed.

Rule (docs/stage2-R1b-prereg.md section 2, fixed before the confirmatory scoring):
for every target (camera, object) of a crop-tracked derived folder, compute
IoU(crop-tracked mask at frame start+1, seed mask at frame start); if it is below
--tau (0.05) the pair is judged to have jumped to another object inside the window
on the first frame, and its frames start+1 .. start+20 are copied back from the base
folder.  No ground truth is read.  Everything else is copied unchanged, so the
guarded folder differs from the crop folder only in the reverted pairs.

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \
        python SCSam3/crop_guard.py --suffix Cr2k [--tau 0.05] [--scenes ...]

Writes Data/MVSeg/<scene>/<base><suffix>G/ and a "guard" block in its MANIFEST.json.
Runs in the container (cv2, numpy); the host has neither.
"""
import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")


def iou(a, b):
    """IoU of two bool arrays; two empty masks count as 1.0 (nothing to disagree on)."""
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def guard_scene(scene, base, suffix, tau, cfg, cv2, np, zoom_min=0.0, border_max=None, out_suffix="G"):
    c = cfg[scene]
    folder, start, num = c["folder"], c["start_frame"], c["num_frame"]
    src = os.path.join(DATA, folder, base + suffix)
    dst = os.path.join(DATA, folder, base + suffix + out_suffix)
    basedir = os.path.join(DATA, folder, base)
    man = json.load(open(os.path.join(src, "MANIFEST.json")))
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    reverted, kept = [], []
    seeds_folder = man["derived"].get("seeds", "MVSeed_e0_index")
    c_ini = json.load(open(CENSUS))[scene]["c_ini"]
    for t in man["derived"]["targets"]:
        cam, obj = t["cam"], t["obj"]
        # the seed is the prompt crop_track used (registered rule: "frame-0 seed"), not the
        # base folder's frame-0 PNG -- the base tracker drops some objects at frame 0 and
        # then no PNG exists there although the prompt was non-empty
        if cam == c_ini:
            gt = cv2.imread(os.path.join(DATA, folder, "Mask", cam, f"{start:06d}.png"), cv2.IMREAD_GRAYSCALE)
            seed = (gt == obj)
        else:
            sp = cv2.imread(os.path.join(DATA, folder, seeds_folder, cam, str(start), f"{obj}.png"), cv2.IMREAD_GRAYSCALE)
            seed = (sp > 127) if sp is not None else None
        f1 = cv2.imread(os.path.join(src, cam, str(start + 1), f"{obj}.png"), cv2.IMREAD_GRAYSCALE)
        f1 = (f1 > 127) if f1 is not None else (np.zeros_like(seed) if seed is not None else None)
        score = iou(f1, seed) if seed is not None else 1.0
        rec = dict(cam=cam, obj=obj, iou_f1_seed=score, zoom=t.get("zoom"), border=t.get("frames_touching_border"))
        # v2 (docs/stage2-R1c-prereg.md): no resolution gain, or the object rides the window edge
        no_gain = zoom_min and (t.get("zoom") or 0.0) < zoom_min
        rides_edge = border_max is not None and (t.get("frames_touching_border") or 0) >= border_max
        rec["why"] = "iou" if score < tau else "zoom" if no_gain else "border" if rides_edge else None
        if score < tau or no_gain or rides_edge:
            for fr in range(start + 1, start + num):
                d = os.path.join(dst, cam, str(fr), f"{obj}.png")
                b = os.path.join(basedir, cam, str(fr), f"{obj}.png")
                if os.path.exists(b):
                    shutil.copyfile(b, d)
                elif os.path.exists(d):
                    os.remove(d)      # the base tracker had dropped the object: scored as empty
            reverted.append(rec)
        else:
            kept.append(rec)
    man["guard"] = dict(rule="revert frames start+1..start+num-1 to base when IoU(crop frame start+1, seed) < tau"
                             " or zoom < zoom_min or window-edge frames >= border_max",
                        tau=tau, zoom_min=zoom_min, border_max=border_max, reverted=reverted, kept=len(kept),
                        n_targets=len(reverted) + len(kept),
                        prereg="docs/stage2-R1b-prereg.md" if not (zoom_min or border_max) else "docs/stage2-R1c-prereg.md")
    json.dump(man, open(os.path.join(dst, "MANIFEST.json"), "w"), indent=1)
    return reverted, len(kept)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--base", default="SegMaskSam3XW0MFs")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--zoom-min", type=float, default=0.0, help="v2: revert when 1008/side < this (0 = off)")
    ap.add_argument("--border-max", type=int, default=None, help="v2: revert when window-edge frames >= this")
    ap.add_argument("--out-suffix", default="G", help="folder suffix appended after --suffix (G = v1, G2 = v2)")
    args = ap.parse_args(argv)
    import cv2
    import numpy as np
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    scenes = args.scenes or sorted(json.load(open(CENSUS)))
    total_rev = total_kept = 0
    for s in scenes:
        rev, kept = guard_scene(s, args.base, args.suffix, args.tau, cfg, cv2, np,
                                zoom_min=args.zoom_min, border_max=args.border_max, out_suffix=args.out_suffix)
        total_rev += len(rev); total_kept += kept
        tag = ", ".join(f"{r['cam']}/{r['obj']} ({r['why']} {r['iou_f1_seed']:.2f}/z{(r['zoom'] or 0):.1f}/b{r['border']})" for r in rev)
        print(f"{s:22} targets {len(rev) + kept:3}  reverted {len(rev):2}  {tag}", flush=True)
    print(f"{args.base}{args.suffix}{args.out_suffix}: {total_rev} reverted / {total_rev + total_kept} targets (tau {args.tau})")


if __name__ == "__main__":
    main()
