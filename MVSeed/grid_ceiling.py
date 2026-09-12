#!/usr/bin/env python3
"""S1-E0''': how many MUVOD points does the 288 mask grid make unreachable?  (docs/stage1-E0c-prereg.md)

Every mask the tracker writes is expressed on a low_res_mask_size x low_res_mask_size grid
(sam3_tracker_base.py:130, 1008 // 14 * 4 = 288) before it is upsampled to video resolution.
Passing the GROUND TRUTH through that same grid therefore gives the best mask any method can
write at that grid -- a ceiling, measured with no model.  S1-E0' established the round trip on
the 398 reference pairs at frame 0; this runs it over the whole scored population and all 21
frames, and converts it into the MUVOD J&F^3 scale so it can be set against the 8.985-point gap
of docs/error-budget.md.

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \\
        python MVSeed/grid_ceiling.py [--grids 288,576,1008] [--scenes A,B]

CPU only.  J and F come from eval/eval_jf.py unchanged (imported, never edited).
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval/eval_jf.py", "eval_jf")          # db_eval_iou / db_eval_boundary, untouched
rt = _load("MVSeed/ceiling_roundtrip.py", "ceiling_roundtrip")
DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
OUT = os.path.join(REPO, "MVSeed", "runs", "grid_ceiling.json")
GRIDS = (288, 576, 1008)
HEADLINE = 91.015                 # docs/stage2-R2.md
GAP = 100.0 - HEADLINE            # 8.985, docs/error-budget.md


def scene_rows(scene, cfg, census, grids, log):
    import cv2
    import numpy as np
    d = cfg[scene]
    start, num, folder = d["start_frame"], d["num_frame"], d["folder"]
    ref = census[scene]["c_ini"]
    cams = census[scene]["cameras"]
    objects = sorted(int(o) for o in census[scene]["seed_ids_c_ini"])
    rows = []
    for cam in cams:
        acc = {g: {o: {"J": [], "F": []} for o in objects} for g in grids}
        gt_px = {o: [] for o in objects}
        for fr in range(start, start + num):
            gt = cv2.imread(os.path.join(DATA, folder, "Mask", cam, f"{fr:06d}.png"),
                            cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            h, w = gt.shape
            bound_pix = int(math.ceil(ev.BOUND_TH * np.linalg.norm((h, w))))
            se = ev._disk(bound_pix)
            truth = [gt == o for o in objects]
            for g in grids:
                ceil = rt.roundtrip(truth, low_res=g)      # resample + the overlap rule, as the chain
                for o, t, c in zip(objects, truth, ceil):
                    if not t.any():
                        continue
                    acc[g][o]["J"].append(float(ev.db_eval_iou(t, c)))
                    acc[g][o]["F"].append(float(ev.db_eval_boundary(c, t, bound_pix, se)))
            for o, t in zip(objects, truth):
                if t.any():
                    gt_px[o].append(int(t.sum()))
        for o in objects:
            if not gt_px[o]:
                continue
            rec = dict(scene=scene, cam=cam, obj=o, is_ref=(cam == ref),
                       rig=rt.rig_of(scene), n_frames=len(gt_px[o]),
                       gt_px_median=int(sorted(gt_px[o])[len(gt_px[o]) // 2]))
            rec["size_bin"] = rt.size_bin(rec["gt_px_median"])
            for g in grids:
                J = sum(acc[g][o]["J"]) / len(acc[g][o]["J"])
                F = sum(acc[g][o]["F"]) / len(acc[g][o]["F"])
                rec[f"J_{g}"] = round(J, 6)
                rec[f"F_{g}"] = round(F, 6)
                rec[f"JF_{g}"] = round((J + F) / 2, 6)
            rows.append(rec)
    log(f"  {scene:16s} {len(rows)} (cam, object) rows")
    return rows


def muvod(rows, grids, key="JF"):
    """Per-camera mean over objects -> mean over the annotated cameras -> mean over scenes."""
    out = {}
    for g in grids:
        per = {}
        for r in rows:
            per.setdefault(r["scene"], {}).setdefault(r["cam"], []).append(r[f"{key}_{g}"])
        sc = {s: sum(sum(v) / len(v) for v in cams.values()) / len(cams) for s, cams in per.items()}
        out[g] = round(sum(sc.values()) / len(sc) * 100, 3) if sc else None
    return out


def summarize(rows, grids):
    s = {"n_rows": len(rows), "n_frames": sum(r["n_frames"] for r in rows)}
    for key in ("JF", "J", "F"):
        s[f"ceiling_{key}"] = muvod(rows, grids, key)
    s["unreachable_points"] = {g: round(100.0 - s["ceiling_JF"][g], 3) for g in grids}
    s["share_of_gap"] = {g: round(s["unreachable_points"][g] / GAP, 4) for g in grids}
    for name, sel in (("ref", lambda r: r["is_ref"]), ("nonref", lambda r: not r["is_ref"]),
                      ("pinhole", lambda r: r["rig"] == "pinhole"),
                      ("hemisphere", lambda r: r["rig"] == "hemisphere"),
                      *[(b, (lambda b: (lambda r: r["size_bin"] == b))(b)) for b in rt.SIZE_BINS]):
        sub = [r for r in rows if sel(r)]
        s[f"ceiling_JF_{name}"] = muvod(sub, grids) if sub else None
        s[f"n_{name}"] = len(sub)
    g0 = grids[0]
    share = s["share_of_gap"][g0]
    s["reading"] = ("유효 해상도가 남은 오차의 지배적 단일 원인" if share >= 0.40 else
                    "격자는 부차적" if share < 0.20 else "상당하지만 지배적이지 않음")
    s["reading_rule"] = "prereg section 4: >= 0.40 dominant, < 0.20 secondary, else substantial"
    s["headline"] = HEADLINE
    s["gap"] = round(GAP, 3)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", default=",".join(str(g) for g in GRIDS))
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    grids = tuple(int(g) for g in args.grids.split(","))
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    census = json.load(open(CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(census)
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    rows = []
    for s in scenes:
        rows += scene_rows(s, cfg, census, grids, log)
    summary = summarize(rows, grids)
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    json.dump({"schema": "MVSeed grid_ceiling", "schema_version": 1, "git_head": head,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "grids": list(grids),
               "prereg": "docs/stage1-E0c-prereg.md", "argv": sys.argv,
               "summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"wrote {args.out}  ({len(rows)} rows, {summary['n_frames']} frame evals, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
