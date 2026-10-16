#!/usr/bin/env python3
"""Readings of a full-length CROOM T06 run (docs/plenoptic-F.md section 4).  Descriptive only: nothing here is used to
choose or tune anything on T06.

    score_full.py --jf <eval_jf output> --method <name> --fwd <merged F method dir> [--bwd <B method dir>] --out <json>

frame900   the only ground truth: J and F at source frame 900 (the prompt frame) per (camera, person), read from the
           frozen scorer's output (eval/eval_jf.py over the merged tree; it aligns predictions by frame number).
           Q1 mean / median J over the non-reference (camera, person) pairs with the person in the GT, Q2 by ring
           distance from cam_06, Q3 pairs below J 0.5 and the worst eight, Q5 MUVOD-style scene J&F (objects ->
           cameras -> scene, all 32 cameras incl. cam_06) -- the S3-P readings of plenoptic/score.py at frame 900.
prompt     the backward run starts from the same frame-900 images and prompt: its index-0 PNGs against the forward
           run's frame-900 PNGs, byte for byte (merge_back keeps the forward copy).
census     from MANIFEST view_areas (the tracker's own per-view areas, every tracked view, before write-time
           substitutions), per (view, person) with a seed area: frames with area 0, the longest such stretch, and
           frames below 0.1 x the seed area, separately forward (901..9937) and backward (899..0).
run        what the run recorded: git revision, stage-1 checkpoint, crop-small targets, S2-R6 replacements, pruning.
"""
import argparse
import hashlib
import json
import os
import statistics

REF, NCAM, NOBJ, FRAME = 6, 32, 6, 900


def ring_distance(k):
    d = abs(k - REF)
    return min(d, NCAM - d)


def frame900(entries, method):
    rows = {}
    for e in entries:
        if e["method"] != method or not e.get("result"):
            continue
        pf, k = e["result"]["per_frame"], int(e["camera"].split("_")[1])
        fi = e["meta"]["frames"].index(FRAME)
        for i, o in enumerate(e["objects"]):
            rows[(k, o)] = (pf["J"][i][fi], pf["F"][i][fi], pf["gt_area"][i][fi], pf["pred_missing"][i][fi])
    present = {ko: v for ko, v in rows.items() if ko[0] != REF and v[2]}
    J = [v[0] for v in present.values()]
    byd = {}
    for (k, o), v in present.items():
        byd.setdefault(ring_distance(k), []).append(v[0])
    percam = {}
    for (k, o), (j, f, g, _) in rows.items():
        percam.setdefault(k, []).append(0.5 * (j + f))
    return dict(
        n_cameras=len({k for k, _ in rows}), n_present=len(present),
        n_gt_empty=sum(1 for (k, o), v in rows.items() if k != REF and not v[2]),
        n_pred_missing=sum(1 for v in rows.values() if v[3]),
        Q1_mean_J=statistics.mean(J), Q1_median_J=statistics.median(J),
        Q2_by_distance={d: dict(n=len(v), mean_J=statistics.mean(v), min_J=min(v)) for d, v in sorted(byd.items())},
        Q3_fail_lt05=sum(j < 0.5 for j in J), Q3_worst=sorted((round(v[0], 3), f"cam_{k:02d}", o)
                                                             for (k, o), v in present.items())[:8],
        Q5_scene_JF=100 * statistics.mean(statistics.mean(v) for v in percam.values()),
        ref_J={o: rows[(REF, o)][0] for o in range(1, NOBJ + 1) if (REF, o) in rows},
        per_pair={f"cam_{k:02d}/{o}": dict(J=v[0], F=v[1], gt_area=v[2]) for (k, o), v in sorted(rows.items())})


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def prompt_identity(fwd, bwd):
    same, diff, only = 0, [], []
    for cam in sorted(c for c in os.listdir(fwd) if c.startswith("cam_")):
        a, b = os.path.join(fwd, cam, str(FRAME)), os.path.join(bwd, cam, "0")
        fa = set(os.listdir(a)) if os.path.isdir(a) else set()
        fb = set(os.listdir(b)) if os.path.isdir(b) else set()
        only += [f"{cam}/{x}" for x in sorted(fa ^ fb)]
        for x in sorted(fa & fb):
            if _sha(os.path.join(a, x)) == _sha(os.path.join(b, x)):
                same += 1
            else:
                diff.append(f"{cam}/{x}")
    return dict(identical=same, different=len(diff), different_list=diff[:40], in_one_only=only[:40])


def census(va, frames):
    out, per_obj = {}, {}
    for v, seed in va["seed"].items():
        tr = va["tracked"].get(v, {})
        for o, a0 in seed.items():
            if not a0:
                continue
            zero = low = run = longest = 0
            for t in frames:
                a = tr.get(str(t), {}).get(o, 0)
                zero += a == 0
                low += a < 0.1 * a0
                run = run + 1 if a == 0 else 0
                longest = max(longest, run)
            out[f"v{v}/{o}"] = dict(zero=zero, longest_zero=longest, below_0p1_seed=low)
            per_obj.setdefault(o, []).append(zero / len(frames))
    return dict(n_frames=len(frames), n_pairs=len(out),
                mean_zero_fraction=statistics.mean(x for v in per_obj.values() for x in v) if per_obj else None,
                per_person_mean_zero_fraction={o: statistics.mean(v) for o, v in sorted(per_obj.items(), key=lambda x: int(x[0]))},
                pairs_ever_zero=sum(1 for r in out.values() if r["zero"]), per_pair=out)


def run_record(prov):
    keep = ("git_rev", "git_dirty", "docker_image_id", "start_frame", "num_frame", "reference", "n_sessions",
            "stage1", "long_run", "frame0_seed_stats")
    r = {k: prov.get(k) for k in keep}
    cs = prov.get("crop_small") or {}
    r["crop_small"] = {k: v for k, v in cs.items() if k not in ("rule", "prompt")} if cs else None
    rs = prov.get("ref_snap")
    r["ref_snap"] = {k: rs[k] for k in ("frames", "objects_frames", "snapped", "no_prompt", "missing") if k in rs} if rs else None
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jf", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--fwd", required=True)
    ap.add_argument("--bwd")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = dict(method=a.method, frame900=frame900(json.load(open(a.jf)), a.method))
    mf = json.load(open(os.path.join(a.fwd, "MANIFEST.json")))["provenance"]
    last = mf["start_frame"] + mf["num_frame"] - 1
    res["run_forward"] = run_record(mf)
    res["census_forward"] = census(mf["view_areas"], range(FRAME + 1, last + 1))
    if a.bwd:
        res["prompt_identity"] = prompt_identity(a.fwd, a.bwd)
        mb = json.load(open(os.path.join(a.bwd, "MANIFEST.json")))["provenance"]
        res["run_backward"] = run_record(mb)
        res["census_backward"] = census(mb["view_areas"], range(1, mb["num_frame"]))
    json.dump(res, open(a.out, "w"), indent=1)
    f9 = res["frame900"]
    print(f"{a.method}: frame 900 Q1 mean J {f9['Q1_mean_J']:.4f} (median {f9['Q1_median_J']:.4f}, n {f9['n_present']}), "
          f"J<0.5 {f9['Q3_fail_lt05']}, scene J&F {f9['Q5_scene_JF']:.2f}; forward zero-area share "
          f"{res['census_forward']['mean_zero_fraction']}"
          + (f"; prompt identical {res['prompt_identity']['identical']}/"
             f"{res['prompt_identity']['identical'] + res['prompt_identity']['different']}" if a.bwd else ""))


if __name__ == "__main__":
    main()
