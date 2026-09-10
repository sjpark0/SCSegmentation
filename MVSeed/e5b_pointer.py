#!/usr/bin/env python3
"""S1-E5b: does SAM 3's own object pointer tell a bad seed from a good one?  (docs/stage1-E5b-prereg.md)

Measurement only.  For every (view, object) of the same 796 rows, the cosine between
  * the object pointer SAM 3 makes from the GROUND-TRUTH mask on the reference camera, and
  * the object pointer it makes from the RAW SEED mask on that view
is computed and fed through the same readout as S1-E5a (hits on the 38 misplaced pairs at the false
positive count of the P5 area rule).  The package is not modified: `_use_mask_as_output` is shadowed
on the tracker instance to capture the pointer it already returns.

    docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \\
      --user $(id -u):$(id -g) -v /:/host -w /host$PWD \\
      -e HF_HOME=/host$PWD/SCSam3/hf_cache -e HF_HUB_OFFLINE=1 scsam3 python MVSeed/e5b_pointer.py
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "SCSam3", "demoSCSam3MVOpt")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load("MVSeed/gate_det.py", "gate_det")
e5a = _load("MVSeed/geo/e5a_signals.py", "e5a_signals")
DATA = e5a.DATA
OUT = os.path.join(REPO, "MVSeed", "runs", "e5b_pointer.json")
PARTS = os.path.join(REPO, "MVSeed", "runs", "e5b_pointer_parts")
SEEDS = "MVSeed_e0_index"


class PointerTap:
    """Instance-level shadow of `tracker._use_mask_as_output` that records the object pointer.

    The pointer is the sixth element of that function's return tuple (sam3_tracker_base.py 426-434)
    and depends only on this frame's features and the mask, so measurements do not interact.
    """

    def __init__(self, tracker):
        self.tracker = tracker
        self.last = None
        self._orig = tracker._use_mask_as_output
        tracker._use_mask_as_output = self._wrapped

    def _wrapped(self, backbone_features, high_res_features, mask_inputs):
        out = self._orig(backbone_features, high_res_features, mask_inputs)
        self.last = out[5].detach().float().reshape(-1).clone()
        return out

    def remove(self):
        del self.tracker._use_mask_as_output


def cosine(a, b):
    import torch
    if a is None or b is None:
        return None
    d = (a.norm() * b.norm()).clamp(min=1e-12)
    return round(float((a * b).sum() / d), 4)


def run_scene(scene, cfg, census, log):
    import cv2
    import numpy as np
    import torch
    from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU
    d = cfg[scene]
    start, folder = d["start_frame"], d["folder"]
    ref = census[scene]["c_ini"]
    cams = [ref] + [c for c in census[scene]["cameras"] if c != ref]
    objects = [int(o) for o in census[scene]["seed_ids_c_ini"]]
    ds_dir = os.path.join(DATA, folder)
    frames, gts, hw = [], {}, None
    for cam in cams:
        cpu_image, h, w = load_video_frames(video_path=os.path.join(ds_dir, "Video", cam))
        frames.append(AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)[start])
        g = cv2.imread(os.path.join(ds_dir, "Mask", cam, f"{start:06d}.png"), cv2.IMREAD_GRAYSCALE)
        assert g is not None and g.shape == (h, w)
        gts[cam] = g
        hw = (h, w)
    model = run_scene.model
    tap = run_scene.tap

    def pointers(frame, masks):
        """Object pointers for one camera: a single-frame session, one fresh object id per mask.

        A session per camera keeps the object count at the mainline's (one per labelled object).  One
        session holding all three cameras reached about a hundred objects and the tracker's Triton
        connected-components kernel then failed with "invalid argument" on the large hemisphere frames.
        """
        state = model.init_state([frame], hw[0], hw[1])
        out, next_id = {}, 1000
        for key, m in masks.items():
            if m.sum() == 0:
                out[key] = None
                continue
            tap.last = None
            next_id += 1
            model.add_tracker_new_mask(inference_state=state, frame_idx=0, obj_id=next_id,
                                       mask=torch.tensor(m, dtype=torch.float32))
            out[key] = tap.last
        del state
        torch.cuda.empty_cache()
        return out

    ref_ptr = pointers(frames[0], {o: gts[ref] == o for o in objects})
    rows = []
    for i, cam in enumerate(cams[1:], start=1):
        seeds = {}
        for obj in objects:
            sp = os.path.join(ds_dir, SEEDS, cam, str(start), f"{obj}.png")
            m = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
            seeds[obj] = (m > 127) if m is not None else np.zeros(hw, bool)
        ptr = pointers(frames[i], seeds)
        for obj in objects:
            rows.append(dict(scene=scene, cam=cam, obj=obj, ref_cam=ref,
                             gt_px=int((gts[cam] == obj).sum()), seed_px=int(seeds[obj].sum()),
                             ref_gt_px=int((gts[ref] == obj).sum()),
                             rig="hemisphere" if scene in gate.HEMISPHERE else "pinhole",
                             signals=dict(S4_cos=cosine(ref_ptr[obj], ptr[obj]))))
    del frames, gts
    torch.cuda.empty_cache()
    got = [r["signals"]["S4_cos"] for r in rows if r["signals"]["S4_cos"] is not None]
    log(f"  {scene:16s} {len(rows)} rows, cos median "
        f"{sorted(got)[len(got) // 2] if got else None}")
    return rows


def summarize(rows):
    """Same readout as S1-E5a, on the single signal S4_cos."""
    for r in rows:
        r["signals"].setdefault("S4_cos", None)
    pos = [r for r in rows if r["set"] == "F0" and r.get("f0_kind") == "misplaced"]
    neg = [r for r in rows if r["set"] == "N0"]
    empty = [r for r in rows if r["set"] == "F0" and r.get("f0_kind") != "misplaced"]
    gt_empty = [r for r in rows if r["set"] == "gt_empty" and r["seed_px"] > 0]
    dist = lambda rs: {"n": len(rs), "median": gate._median([r["signals"]["S4_cos"] for r in rs
                                                            if r["signals"]["S4_cos"] is not None]),
                       "undefined": sum(1 for r in rs if r["signals"]["S4_cos"] is None)}
    s = {"n_rows": len(rows), "n_misplaced": len(pos), "n_normal": len(neg),
         "distribution": {"misplaced": dist(pos), "normal": dist(neg), "empty_seed": dist(empty),
                          "gt_empty_nonempty_seed": dist(gt_empty)},
         "normal_percentiles": {}}
    vals = sorted(r["signals"]["S4_cos"] for r in neg if r["signals"]["S4_cos"] is not None)
    for q in (1, 5, 10, 25, 50):
        s["normal_percentiles"][f"p{q}"] = vals[max(0, len(vals) * q // 100 - 1)] if vals else None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--parts-dir", default=PARTS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(e5a.CONFIG, encoding="utf-8"))
    census = json.load(open(e5a.CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(census)
    assert not [s for s in scenes if s not in census]
    f0, n0 = gate.membership(json.load(open(gate.spike.F0))["pairs"], json.load(open(gate.spike.N0))["pairs"])
    os.makedirs(args.parts_dir, exist_ok=True)
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    if not args.summary_only:
        sys.path.insert(0, PKG)
        import torch
        from build_scsam3 import build_scsam3_video_model
        run_scene.model = build_scsam3_video_model()
        run_scene.model.eval()
        # the mainline turns hole filling off (SCSam3Video.py 18, 35) while the raw builder defaults to
        # 16; leaving it on runs a Triton connected-components kernel that fails on the 2560x1920 frames
        run_scene.model.fill_hole_area = 0
        run_scene.tap = PointerTap(run_scene.model.tracker)
        with torch.inference_mode():
            for s in scenes:
                part = os.path.join(args.parts_dir, f"{s}.json")
                keys = gate.expected_keys(s, cfg, census)
                if gate.load_part(part, keys) is not None and not args.overwrite:
                    log(f"  {s}: complete part exists, skipping")
                    continue
                t1 = time.time()
                res = run_scene(s, cfg, census, log)
                order = {k: i for i, k in enumerate(keys)}
                res.sort(key=lambda r: order[(r["cam"], r["obj"])])
                json.dump(res, open(part + ".tmp", "w"), indent=1)
                os.replace(part + ".tmp", part)
                log(f"  {s}: {len(res)} rows in {time.time() - t1:.0f}s")
        run_scene.tap.remove()
    rows, missing = [], []
    for s in sorted(census):
        got = gate.load_part(os.path.join(args.parts_dir, f"{s}.json"), gate.expected_keys(s, cfg, census))
        if got is None:
            missing.append(s)
        else:
            rows += [gate.tag_row(r, f0, n0) for r in got]
    summary = summarize(rows)
    summary["missing_scenes"] = missing
    # the joint readout against S1-E5a's signals, when that run exists
    p = os.path.join(REPO, "MVSeed", "runs", "e5a_signals.json")
    if os.path.exists(p) and not missing:
        a = {(r["scene"], r["cam"], r["obj"]): r for r in json.load(open(p))["rows"]}
        merged = []
        for r in rows:
            q = dict(a[(r["scene"], r["cam"], r["obj"])])
            q["signals"] = dict(q["signals"], S4_cos=r["signals"]["S4_cos"])
            merged.append(q)
        pos = [r for r in merged if r["set"] == "F0" and r.get("f0_kind") == "misplaced"]
        neg = [r for r in merged if r["set"] == "N0"]
        base = sum(e5a.flagged(r, "P5area") for r in neg)
        summary["matched_fp"] = {"target_fp": base,
                                 "S4_cos": e5a.matched_threshold(pos, neg, "S4_cos", base, low=True),   # a LOW cosine is suspicious
                                 "P5area": {"hits": sum(e5a.flagged(r, "P5area") for r in pos), "fp": base},
                                 "S1_rev": e5a.matched_threshold(pos, neg, "S1_rev", base)}
        for name in ("S4_cos",):
            for other in ("S1_rev", "P5area"):
                th = summary["matched_fp"]["S4_cos"]["threshold"]
                f = lambda r: (r["signals"]["S4_cos"] is None or (th is not None and r["signals"]["S4_cos"] < th)) \
                    and e5a.flagged(r, other)
                summary.setdefault("combinations", {})[f"{name} AND {other}"] = {
                    "hits": sum(1 for r in pos if f(r)), "fp": sum(1 for r in neg if f(r))}
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    doc = {"schema": "MVSeed e5b_pointer", "schema_version": 1,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head": head,
           "generated_by": "MVSeed/e5b_pointer.py", "argv": sys.argv, "prereg": "docs/stage1-E5b-prereg.md",
           "rules": {"seeds": SEEDS, "pointer": "sam3_tracker_base._use_mask_as_output return index 5",
                     "undefined_when": "the seed mask is empty -> flagged"},
           "summary": summary, "rows": rows}
    json.dump(doc, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}  ({len(rows)} rows, missing {missing}, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
