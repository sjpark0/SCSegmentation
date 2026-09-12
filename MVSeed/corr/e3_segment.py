#!/usr/bin/env python3
"""S1-E3 pass 2: segment each DINOv2 candidate point with SAM 3 and read the registered gate.

Pass 1 (MVSeed/corr/dino_prompt.py) wrote K positive points per (camera, object).  Here each point
becomes one single-object, single-frame SAM 3 session -- candidates are never put in one session,
so overlap suppression cannot contaminate the comparison (docs/stage1-E3-prereg.md section 2.6).
The gate arithmetic and its constants are imported from MVSeed/gate_det.py, the module that judged
S1-E2, so the two arms are judged by the same numbers.

    docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \\
      --user $(id -u):$(id -g) -v /:/host -w /host$PWD \\
      -e HF_HOME=/host$PWD/SCSam3/hf_cache -e HF_HUB_OFFLINE=1 scsam3 python MVSeed/corr/e3_segment.py
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(REPO, "SCSam3", "demoSCSam3MVOpt")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load("MVSeed/gate_det.py", "gate_det")
DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
PROMPTS_IN = os.path.join(REPO, "MVSeed", "runs", "e3_prompts.json")
OUT = os.path.join(REPO, "MVSeed", "runs", "e3_gate.json")
PARTS = os.path.join(REPO, "MVSeed", "runs", "e3_gate_parts")
TAU_SIM = 0.5               # registered: a candidate exists only at cosine >= this and a non-empty mask
IOU_EXIST = gate.IOU_EXIST  # 0.5, same object as S1-E2


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def run_scene(scene, cfg, census, prompts, log):
    import cv2
    import numpy as np
    import torch
    from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU
    d = cfg[scene]
    start, folder = d["start_frame"], d["folder"]
    ref = census[scene]["c_ini"]
    targets = [c for c in census[scene]["cameras"] if c != ref]
    objects = [int(o) for o in census[scene]["seed_ids_c_ini"]]
    ds = os.path.join(DATA, folder)
    model = run_scene.model
    rows = []
    for cam in targets:
        cpu_image, h, w = load_video_frames(video_path=os.path.join(ds, "Video", cam))
        frame = AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)[start]
        gt = cv2.imread(os.path.join(ds, "Mask", cam, f"{start:06d}.png"), cv2.IMREAD_GRAYSCALE)
        assert gt is not None and gt.shape == (h, w)
        for obj in objects:
            g = gt == obj
            cands = []
            for c in prompts[scene]["cams"][cam].get(str(obj), []):
                x, y = c["point"]
                if c["cos"] < TAU_SIM:
                    cands.append(dict(point=[x, y], cos=c["cos"], px=0, iou=0.0, kept=False))
                    continue
                state = model.init_state([frame], h, w)
                pt = torch.tensor([[(x + 0.5) / w, (y + 0.5) / h]], dtype=torch.float32)
                _, out = model.add_tracker_new_points(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=pt, labels=torch.tensor([1], dtype=torch.int32))
                bm = out["out_binary_masks"]
                bm = bm.cpu().numpy() if hasattr(bm, "cpu") else np.asarray(bm)
                m = bm[0].astype(bool) if len(bm) else np.zeros((h, w), bool)
                del state
                cands.append(dict(point=[x, y], cos=c["cos"], px=int(m.sum()),
                                  iou=round(iou(m, g), 4), kept=bool(m.sum() > 0)))
            kept = [c for c in cands if c["kept"]]
            hits = [c for c in kept if c["iou"] >= IOU_EXIST]
            best = max(kept, key=lambda c: c["iou"]) if kept else None
            rows.append(dict(
                scene=scene, cam=cam, obj=obj, ref_cam=ref, gt_px=int(g.sum()),
                rig="hemisphere" if scene in gate.HEMISPHERE else "pinhole",
                candidates=cands, n_cand=len(kept), exist=bool(hits),
                best_iou=round(best["iou"], 4) if best else 0.0,
                best_rank=(kept.index(best) + 1) if best else None,
                top1_hit=bool(kept and kept[0]["iou"] >= IOU_EXIST),
                top1_cos=kept[0]["cos"] if kept else None))
        torch.cuda.empty_cache()
    ex = sum(r["exist"] for r in rows)
    log(f"  {scene:16s} {len(rows)} rows, exist {ex}")
    return rows


# ------------------------------------------------------------------ summary (pure)
def _group(rows):
    ex = [r for r in rows if r["exist"]]
    return {"n": len(rows), "exist": len(ex), "exist_rate": gate._rate(len(ex), len(rows)),
            "rank_median": gate._median([r["best_rank"] for r in ex]),
            "rank_list": sorted(r["best_rank"] for r in ex),
            "top1_hit": sum(r["top1_hit"] for r in rows),
            "n_cand_median": gate._median([r["n_cand"] for r in rows]),
            "best_iou_median": gate._median([r["best_iou"] for r in rows]),
            "exist_by_rig": {rig: [sum(1 for r in rows if r["rig"] == rig and r["exist"]),
                                   sum(1 for r in rows if r["rig"] == rig)]
                             for rig in ("pinhole", "hemisphere")}}


def summarize(rows):
    """Same three gates and the same constants as S1-E2 (MVSeed/gate_det.py)."""
    f0 = [r for r in rows if r["set"] == "F0"]
    s = {"n_rows": len(rows),
         "f0_ge2k": _group([r for r in f0 if r["gt_px"] >= 2000]),
         "f0_ge2k_pinhole": _group([r for r in f0 if r["gt_px"] >= 2000 and r["rig"] == "pinhole"]),
         "f0_lt2k": _group([r for r in f0 if r["gt_px"] < 2000]),
         "f0_by_kind": {k: _group([r for r in f0 if r.get("f0_kind") == k])
                        for k in sorted({r.get("f0_kind") for r in f0})},
         "n0": _group([r for r in rows if r["set"] == "N0"]),
         "n0_by_size": {b: _group([r for r in rows if r["set"] == "N0" and r["size_bin"] == b])
                        for b in gate.SIZE_BINS},
         "by_scene": {sc: _group([r for r in rows if r["scene"] == sc and r["set"] in ("F0", "N0")])
                      for sc in sorted({r["scene"] for r in rows})}}
    empty = [r for r in rows if r["set"] == "gt_empty"]
    hal = [r for r in empty if r["n_cand"] >= 1]
    s["gt_empty"] = {"n": len(empty), "halluc": len(hal),
                     "halluc_rate": gate._rate(len(hal), len(empty)),
                     "halluc_top1_cos_median": gate._median([r["top1_cos"] for r in hal if r["top1_cos"] is not None])}
    counts = {"f0_ge2k": s["f0_ge2k"]["n"], "f0_ge2k_pinhole": s["f0_ge2k_pinhole"]["n"],
              "f0_lt2k": s["f0_lt2k"]["n"], "n0": s["n0"]["n"], "gt_empty": s["gt_empty"]["n"]}
    s["denominators"] = {"registered": gate.REGISTERED, "measured": counts,
                         "match": counts == gate.REGISTERED}
    n1, n3 = s["f0_ge2k"]["n"], s["gt_empty"]["n"]
    g1 = n1 > 0 and 100 * s["f0_ge2k"]["exist"] >= round(gate.GATE1_MIN_RATE * 100) * n1
    g1p = s["f0_ge2k_pinhole"]["exist"] >= gate.GATE1P_MIN
    g2 = s["f0_ge2k"]["rank_median"] is not None and s["f0_ge2k"]["rank_median"] <= gate.GATE2_MAX_RANK_MEDIAN
    g3 = n3 > 0 and round(1 / gate.GATE3_MAX_RATE) * s["gt_empty"]["halluc"] <= n3
    s["gates"] = {
        "gate1_exist_ge2k": {"pass": g1, "value": s["f0_ge2k"]["exist"], "n": n1,
                             "min_rate": gate.GATE1_MIN_RATE},
        "gate1p_pinhole": {"pass": g1p, "value": s["f0_ge2k_pinhole"]["exist"],
                           "n": s["f0_ge2k_pinhole"]["n"], "min": gate.GATE1P_MIN},
        "gate2_rank": {"pass": g2, "value": s["f0_ge2k"]["rank_median"],
                       "max": gate.GATE2_MAX_RANK_MEDIAN},
        "gate3_halluc": {"pass": g3, "value": s["gt_empty"]["halluc_rate"], "max": gate.GATE3_MAX_RATE},
        "verdict": ("proceed with G-corr" if (g1 and g2 and g3) else
                    "discard G-corr: " + ", ".join(n for n, ok in
                                                   (("gate 1 existence", g1), ("gate 2 rank", g2),
                                                    ("gate 3 hallucination", g3)) if not ok) + " failed")}
    return s


def tau_curve(rows, taus=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)):
    """Report only: how the three readouts move with the similarity threshold."""
    out = {}
    for t in taus:
        f0 = [r for r in rows if r["set"] == "F0" and r["gt_px"] >= 2000]
        ex = sum(any(c["kept"] and c["cos"] >= t and c["iou"] >= IOU_EXIST for c in r["candidates"]) for r in f0)
        emp = [r for r in rows if r["set"] == "gt_empty"]
        hal = sum(any(c["kept"] and c["cos"] >= t for c in r["candidates"]) for r in emp)
        out[f"{t:g}"] = dict(f0_ge2k_exist=ex, n=len(f0), halluc=hal, halluc_rate=gate._rate(hal, len(emp)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--prompts", default=PROMPTS_IN)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--parts-dir", default=PARTS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    census = json.load(open(CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(census)
    assert not [s for s in scenes if s not in census]
    f0, n0 = gate.membership(json.load(open(gate.spike.F0))["pairs"], json.load(open(gate.spike.N0))["pairs"])
    os.makedirs(args.parts_dir, exist_ok=True)
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    if not args.summary_only:
        prompts = json.load(open(args.prompts))["scenes"]
        sys.path.insert(0, PKG)
        import torch
        from build_scsam3 import build_scsam3_video_model
        run_scene.model = build_scsam3_video_model()
        run_scene.model.eval()
        run_scene.model.fill_hole_area = 0      # as S1-E5b: the raw builder defaults to 16, mainline uses 0
        with torch.inference_mode():
            for s in scenes:
                part = os.path.join(args.parts_dir, f"{s}.json")
                keys = gate.expected_keys(s, cfg, census)
                if gate.load_part(part, keys) is not None and not args.overwrite:
                    log(f"  {s}: complete part exists, skipping")
                    continue
                t1 = time.time()
                res = run_scene(s, cfg, census, prompts, log)
                order = {k: i for i, k in enumerate(keys)}
                res.sort(key=lambda r: order[(r["cam"], r["obj"])])
                json.dump(res, open(part + ".tmp", "w"), indent=1)
                os.replace(part + ".tmp", part)
                log(f"  {s}: {len(res)} rows in {time.time() - t1:.0f}s")
    rows, missing = [], []
    for s in sorted(census):
        got = gate.load_part(os.path.join(args.parts_dir, f"{s}.json"), gate.expected_keys(s, cfg, census))
        if got is None:
            missing.append(s)
        else:
            rows += [gate.tag_row(r, f0, n0) for r in got]
    summary = summarize(rows)
    summary["missing_scenes"] = missing
    summary["tau_curve"] = tau_curve(rows)
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    doc = {"schema": "MVSeed e3_gate", "schema_version": 1,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head": head,
           "generated_by": "MVSeed/corr/e3_segment.py", "argv": sys.argv,
           "prereg": "docs/stage1-E3-prereg.md",
           "rules": {"tau_sim": TAU_SIM, "iou_exist": IOU_EXIST,
                     "gate1_min_rate": gate.GATE1_MIN_RATE, "gate1p_min": gate.GATE1P_MIN,
                     "gate2_max_rank_median": gate.GATE2_MAX_RANK_MEDIAN,
                     "gate3_max_rate": gate.GATE3_MAX_RATE,
                     "session": "one single-object single-frame SAM 3 session per candidate"},
           "summary": summary, "rows": rows}
    json.dump(doc, open(args.out, "w"), indent=1)
    print_summary(summary)
    print(f"wrote {args.out}  ({len(rows)} rows, missing {missing}, {time.time() - t0:.0f}s)")


def print_summary(s):
    def line(name, g):
        print(f"  {name:22s} n={g['n']:3d}  exist {g['exist']:3d} ({g['exist_rate']})  "
              f"rank med {g['rank_median']}  top1 {g['top1_hit']:3d}  cand med {g['n_cand_median']}  "
              f"bestIoU med {g['best_iou_median']}  rig {g['exist_by_rig']}")
    print(f"\n[S1-E3 G-corr] rows {s['n_rows']}  missing {s.get('missing_scenes')}  "
          f"denominators match: {s['denominators']['match']} {s['denominators']['measured']}")
    for k in ("f0_ge2k", "f0_ge2k_pinhole", "f0_lt2k", "n0"):
        line(k, s[k])
    for k, g in s["n0_by_size"].items():
        line(f"n0 {k}", g)
    e = s["gt_empty"]
    print(f"  gt_empty               n={e['n']:3d}  halluc {e['halluc']} ({e['halluc_rate']})  "
          f"top1 cos median {e['halluc_top1_cos_median']}")
    for k, v in s["gates"].items():
        print(f"  {k}: {v}")
    print(f"  tau curve: {s.get('tau_curve')}")


if __name__ == "__main__":
    main()
