#!/usr/bin/env python3
"""S1-E2: G-det gate on the frozen baseline (docs/stage1-E2-prereg.md).

For every scene: the c_ini frame and the two other annotated cameras at start_frame.  For every
object of the basic set present in c_ini (census `seed_ids_c_ini`) the box and the interior-point
exemplar prompts are built on c_ini (spike_exemplar.exemplar_prompt, S1-E2-0'' rules), their
exemplar tokens are captured (ExemplarInjector) and injected into each target view; the
candidates of the two runs are pooled as a plain union (prob descending, no cross-run
suppression) and scored offline against the ground truth.  796 rows = 664 gt-present
(F0 41 + N0 623) + 132 gt-empty.  Measurement only: no seed folder is written.

    docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \\
      --user $(id -u):$(id -g) -v /:/host -w /host$PWD \\
      -e HF_HOME=/host$PWD/SCSam3/hf_cache -e HF_HUB_OFFLINE=1 scsam3 \\
      python MVSeed/gate_det.py [--scenes A,B]          # part files per scene, resumable
    python MVSeed/gate_det.py --summary-only            # re-summarise the part files (no torch)
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("spike_exemplar", os.path.join(REPO, "MVSeed", "spike_exemplar.py"))
spike = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spike)

CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
OUT = os.path.join(REPO, "MVSeed", "runs", "e2_gate.json")
PARTS = os.path.join(REPO, "MVSeed", "runs", "e2_gate_parts")
PROMPTS = ("box", "pts")
IOU_EXIST = 0.5
GATE1_MIN_RATE = 0.55       # F0 & gt_px >= 2000: existence rate
GATE1P_MIN = 8              # ... pinhole subset: existence count (registered 8/18 -> 8/16)
GATE2_MAX_RANK_MEDIAN = 3   # rank of the best-IoU candidate among the existing pairs
GATE3_MAX_RATE = 0.10       # gt-empty rows with >= 1 candidate
REGISTERED = {"f0_ge2k": 19, "f0_ge2k_pinhole": 16, "f0_lt2k": 22, "n0": 623, "gt_empty": 132}
DEDUPE_IOU = 0.5            # report only: union size after cross-run duplicate removal
HEMISPHERE = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder")   # == MVSeed/score_seeds.HEMISPHERE (asserted in the GPU branch)
SIZE_BINS = ("<2000", "2000-10000", ">=10000")                                # == MVSeed/score_seeds.SIZE_BINS
CANDIDATE_RULE = (0.5, 0.1)                                                   # registered: score_threshold_detection, det_nms_thresh


# ----------------------------------------------------------------------------- pure parts
def kept_candidates(out, model):
    """The package's candidate rule (prob > score_threshold after NMS) -> [(prob, low-res logits)]."""
    from sam3.perflib.nms import nms_masks
    probs = out["pred_logits"][0, :, 0].sigmoid()
    masks = out["pred_masks"][0]
    thr, nms_thr = model.score_threshold_detection, model.det_nms_thresh
    keep = nms_masks(probs, masks, thr, nms_thr) if nms_thr > 0.0 else probs > thr
    keep = keep & (probs > thr)
    idx = keep.nonzero()[:, 0]
    return [(float(probs[i]), masks[i].detach().clone()) for i in idx]


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def score_union(runs, gt_bool, hw):
    """Pool the candidates of several runs (plain union, prob descending) and score them offline.

    runs: {name: [(prob, low-res logits), ...]}, gt_bool: [H, W] bool (all False = gt-empty row).
    IoU with an empty ground truth is 0 by definition.  Returns a plain dict."""
    import torch
    import torch.nn.functional as F
    pooled = sorted(((p, m, name) for name, cs in runs.items() for p, m in cs), key=lambda t: -t[0])
    full = []
    for p, m, name in pooled:
        full.append(F.interpolate(m[None, None].float(), size=hw, mode="bilinear", align_corners=False)[0, 0] > 0)
    ious = [_iou(f, gt_bool) if gt_bool.any() else 0.0 for f in full]
    rec = {"n_cand": {name: len(cs) for name, cs in runs.items()}, "n_union": len(pooled)}
    low = [m > 0 for _, m, _ in pooled]
    # union size after cross-run duplicate removal (greedy, prob order, low-res mask IoU > DEDUPE_IOU)
    kept = []
    for i in range(len(pooled)):
        if all(_iou(low[i], low[j]) <= DEDUPE_IOU for j in kept):
            kept.append(i)
    rec["n_union_dedup"] = len(kept)
    per_run, exist_run = {}, {}
    for name in runs:
        vals = [iou for iou, (p, m, n) in zip(ious, pooled) if n == name]
        per_run[name] = round(max(vals), 4) if vals else 0.0
        exist_run[name] = bool(vals) and max(vals) >= IOU_EXIST          # decided on the unrounded IoU
    rec["best_iou_by_run"] = per_run
    rec["exist_by_run"] = exist_run
    if pooled:
        b = max(range(len(pooled)), key=lambda i: ious[i])
        # the best-IoU candidate may have a near-duplicate copy from the other run ranked higher;
        # the registered rank is that of the highest-ranked copy (docs/stage1-E2-prereg.md section 2)
        first = min(j for j in range(b + 1) if j == b or _iou(low[j], low[b]) > DEDUPE_IOU)
        rec.update({"exist": ious[b] >= IOU_EXIST, "best_iou": round(ious[b], 4), "best_rank": first + 1,
                    "best_rank_raw": b + 1, "best_prob": round(pooled[b][0], 4), "best_src": pooled[b][2],
                    "top1_iou": round(ious[0], 4), "top1_hit": ious[0] >= IOU_EXIST, "top1_prob": round(pooled[0][0], 4),
                    "top1_src": pooled[0][2], "probs": [round(p, 3) for p, _, _ in pooled[:10]]})
    else:
        rec.update({"exist": False, "best_iou": 0.0, "best_rank": None, "best_rank_raw": None, "best_prob": None,
                    "best_src": None, "top1_iou": 0.0, "top1_hit": False, "top1_prob": None, "top1_src": None, "probs": []})
    return rec


def membership(f0_pairs, n0_pairs):
    f0 = {(p["scene"], p["cam"], p["obj"]): p for p in f0_pairs}
    n0 = {(p["scene"], p["cam"], p["obj"]) for p in n0_pairs}
    return f0, n0


def tag_row(row, f0, n0):
    key = (row["scene"], row["cam"], row["obj"])
    if row["gt_px"] == 0:
        row["set"] = "gt_empty"
        assert key not in f0 and key not in n0, key
    elif key in f0:
        row["set"] = "F0"
        row["f0_kind"] = f0[key]["kind"]
        row["raw_J"] = f0[key]["raw_J"]
    else:
        assert key in n0, ("gt-present row outside F0 and N0", key)
        row["set"] = "N0"
    row["size_bin"] = None if row["gt_px"] == 0 else SIZE_BINS[0] if row["gt_px"] < 2000 else SIZE_BINS[1] if row["gt_px"] < 10000 else SIZE_BINS[2]
    return row


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def _rate(h, n):
    return round(h / n, 4) if n else None


def _group(rows):
    ex = [r for r in rows if r["exist"]]
    g = {"n": len(rows), "exist": len(ex), "exist_rate": _rate(len(ex), len(rows)),
         "rank_median": _median([r["best_rank"] for r in ex]),
         "rank_list": sorted(r["best_rank"] for r in ex),
         "top1_hit": sum(r["top1_hit"] for r in rows),
         "rank_raw_median": _median([r["best_rank_raw"] for r in ex]),
         "n_union_median": _median([r["n_union"] for r in rows]),
         "n_union_dedup_median": _median([r["n_union_dedup"] for r in rows]),
         "best_iou_median": _median([r["best_iou"] for r in rows])}
    for name in PROMPTS:
        g[f"exist_{name}"] = sum(r["exist_by_run"].get(name, False) for r in rows)
    g["exist_by_rig"] = {rig: [sum(1 for r in rows if r["rig"] == rig and r["exist"]),
                               sum(1 for r in rows if r["rig"] == rig)] for rig in ("pinhole", "hemisphere")}
    return g


def summarize_gate(rows):
    """Populations, the three gates and the verdict (pure)."""
    f0 = [r for r in rows if r["set"] == "F0"]
    s = {"n_rows": len(rows),
         "f0_ge2k": _group([r for r in f0 if r["gt_px"] >= 2000]),
         "f0_ge2k_pinhole": _group([r for r in f0 if r["gt_px"] >= 2000 and r["rig"] == "pinhole"]),
         "f0_lt2k": _group([r for r in f0 if r["gt_px"] < 2000]),
         "f0_by_kind": {k: _group([r for r in f0 if r.get("f0_kind") == k]) for k in sorted({r.get("f0_kind") for r in f0})},
         "n0": _group([r for r in rows if r["set"] == "N0"]),
         "n0_by_size": {b: _group([r for r in rows if r["set"] == "N0" and r["size_bin"] == b]) for b in SIZE_BINS}}
    empty = [r for r in rows if r["set"] == "gt_empty"]
    hal = [r for r in empty if r["n_union"] >= 1]
    s["gt_empty"] = {"n": len(empty), "halluc": len(hal), "halluc_rate": _rate(len(hal), len(empty)),
                     "halluc_by_run": {name: sum(r["n_cand"].get(name, 0) >= 1 for r in empty) for name in PROMPTS},
                     "halluc_top1_prob_median": _median([r["top1_prob"] for r in hal])}
    counts = {"f0_ge2k": s["f0_ge2k"]["n"], "f0_ge2k_pinhole": s["f0_ge2k_pinhole"]["n"], "f0_lt2k": s["f0_lt2k"]["n"],
              "n0": s["n0"]["n"], "gt_empty": s["gt_empty"]["n"]}
    s["denominators"] = {"registered": REGISTERED, "measured": counts, "match": counts == REGISTERED}
    # integer comparisons (no rounding): 11/19 >= 0.55 <=> 100*11 >= 55*19; 13/132 <= 0.10 <=> 10*13 <= 132
    n1, n3 = s["f0_ge2k"]["n"], s["gt_empty"]["n"]
    g1 = n1 > 0 and 100 * s["f0_ge2k"]["exist"] >= round(GATE1_MIN_RATE * 100) * n1
    g1p = s["f0_ge2k_pinhole"]["exist"] >= GATE1P_MIN
    g2 = s["f0_ge2k"]["rank_median"] is not None and s["f0_ge2k"]["rank_median"] <= GATE2_MAX_RANK_MEDIAN
    g3 = n3 > 0 and round(1 / GATE3_MAX_RATE) * s["gt_empty"]["halluc"] <= n3
    if g1 and g2:
        verdict = "proceed" + ("" if g1p else " (hemisphere-limited: pinhole sub-gate failed)") + \
                  ("" if g3 else "; hallucination gate failed -> G-det never fills an empty G0")
    else:
        verdict = "discard G-det: " + ("gate 1 (existence) failed" if not g1 else "gate 2 (rank) failed")
    s["gates"] = {"gate1_exist_ge2k": {"pass": g1, "value": s["f0_ge2k"]["exist"], "n": s["f0_ge2k"]["n"], "min_rate": GATE1_MIN_RATE},
                  "gate1p_pinhole": {"pass": g1p, "value": s["f0_ge2k_pinhole"]["exist"], "n": s["f0_ge2k_pinhole"]["n"], "min": GATE1P_MIN},
                  "gate2_rank": {"pass": g2, "value": s["f0_ge2k"]["rank_median"], "max": GATE2_MAX_RANK_MEDIAN},
                  "gate3_halluc": {"pass": g3, "value": s["gt_empty"]["halluc_rate"], "max": GATE3_MAX_RATE},
                  "verdict": verdict}
    return s


def expected_keys(scene, cfg, census):
    """[(cam, obj)] rows of a scene in run order: the two non-c_ini annotated cameras x seed_ids_c_ini."""
    d = cfg[scene]
    ref_cam = spike.cam_name(d["c_ini"], d)
    cams = [c for c in census[scene]["cameras"] if c != ref_cam]
    return [(cam, int(o)) for o in census[scene]["seed_ids_c_ini"] for cam in cams]


def load_part(path, keys):
    """A scene's part file, or None when it is missing, unreadable or not the expected enumeration."""
    if not os.path.exists(path):
        return None
    try:
        got = json.load(open(path))
    except (OSError, ValueError):
        return None
    if [(r["cam"], r["obj"]) for r in got] != keys:
        return None
    return got


# ----------------------------------------------------------------------------- GPU part
def run_scene(model, inj, scene, cfg, census, data_root, log):
    import cv2
    import torch
    from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU
    d = cfg[scene]
    start = d["start_frame"]
    ds_dir = os.path.join(data_root, d["folder"])
    ref_cam = spike.cam_name(d["c_ini"], d)
    cams = [ref_cam] + [c for c in census[scene]["cameras"] if c != ref_cam]
    assert len(cams) == 3 and census[scene]["c_ini"] == ref_cam, (scene, cams, census[scene]["c_ini"])
    objects = [int(o) for o in census[scene]["seed_ids_c_ini"]]
    device = model.device
    detector = model.detector
    frames, gts, hw = [], {}, None
    for cam in cams:
        cpu_image, height, width = load_video_frames(video_path=os.path.join(ds_dir, "Video", cam))
        image = AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)
        frames.append(image[start])
        gt = cv2.imread(os.path.join(ds_dir, "Mask", cam, f"{start:06d}.png"), cv2.IMREAD_GRAYSCALE)
        assert gt is not None and gt.shape == (height, width), (cam, None if gt is None else gt.shape, height, width)
        gts[cam] = torch.from_numpy(gt).to(device)
        hw = (height, width)
    state = model.init_state(frames, hw[0], hw[1])
    ib = state["input_batch"]
    for t in range(len(frames)):
        ib.find_inputs[t].text_ids[...] = model.TEXT_ID_FOR_VISUAL
    text_outputs = detector.backbone.forward_text(ib.find_text_batch, device=device)
    fresh = lambda: {"img_batch_all_stages": ib.img_batch, **text_outputs}
    cam_idx = {c: i for i, c in enumerate(cams)}
    empty = state["constants"]["empty_geometric_prompt"]
    bo_cache = {}
    inj.mode = None
    for cam in cams[1:]:
        out = detector.forward_grounding(fresh(), ib.find_inputs[cam_idx[cam]], None, empty)
        bo_cache[cam] = out["prev_encoder_out"]["backbone_out"]
        del out
    bo_ref = None
    rows = []
    for obj in objects:
        gt_ref = gts[ref_cam] == obj
        assert int(gt_ref.sum()) > 0, (scene, obj, "basic object missing in c_ini")
        tokens, prompt_meta = {}, {}
        for name in PROMPTS:
            prompt, meta = spike.exemplar_prompt(name, gts[ref_cam], obj, device)
            inj.begin_capture(name, gt_ref)
            out = detector.forward_grounding(bo_ref if bo_ref is not None else fresh(), ib.find_inputs[0], None, prompt)
            bo_ref = out["prev_encoder_out"]["backbone_out"]
            inj.store()
            inj.end_capture()
            tokens[name] = (inj.embed, inj.mask)
            prompt_meta[name] = {"n_points": len(meta["points"]), "box": meta["box"]}
            del out
        for cam in cams[1:]:
            gt_tgt = gts[cam] == obj
            runs = {}
            for name in PROMPTS:
                inj.embed, inj.mask = tokens[name]
                inj.mode = "slot"
                out2 = detector.forward_grounding(bo_cache[cam], ib.find_inputs[cam_idx[cam]], None, empty)
                runs[name] = kept_candidates(out2, model)
                inj.mode = None
                del out2
            rec = score_union(runs, gt_tgt, hw)
            rec.update({"scene": scene, "cam": cam, "obj": obj, "gt_px": int(gt_tgt.sum()), "ref_cam": ref_cam,
                        "ref_gt_px": int(gt_ref.sum()), "rig": "hemisphere" if scene in HEMISPHERE else "pinhole",
                        "prompts": prompt_meta})
            rows.append(rec)
            log(f"  {scene:20s} {cam:12s} obj {obj:3d} gt {rec['gt_px']:8d}  union {rec['n_union']:2d} "
                f"best {rec['best_iou']:.2f} r{rec['best_rank']} ({rec['best_src']})  box {rec['best_iou_by_run']['box']:.2f} "
                f"pts {rec['best_iou_by_run']['pts']:.2f}  top1 {rec['top1_iou']:.2f}")
    del bo_cache, bo_ref, state, frames, gts
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="", help="comma-separated subset")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--parts-dir", default=PARTS)
    ap.add_argument("--data-root", default=spike.DATA)
    ap.add_argument("--config", default=spike.CONFIG)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="summarise existing part files, no GPU")
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    census = json.load(open(CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(census)
    unknown = [x for x in scenes if x not in census]
    assert not unknown, f"unknown scenes (not in the census): {unknown}"
    f0, n0 = membership(json.load(open(spike.F0))["pairs"], json.load(open(spike.N0))["pairs"])
    os.makedirs(args.parts_dir, exist_ok=True)
    rows, t0 = [], time.time()
    log = lambda s: print(s, flush=True)
    thresholds = None
    if not args.summary_only:
        sys.path.insert(0, os.path.join(REPO, "MVSeed"))
        from score_seeds import HEMISPHERE as _H, SIZE_BINS as _B
        assert tuple(_H) == HEMISPHERE and tuple(_B) == SIZE_BINS, (_H, _B)
        sys.path.insert(0, spike.PKG)
        import torch
        from build_scsam3 import build_scsam3_video_model
        model = build_scsam3_video_model()
        model.eval()
        thresholds = (float(model.score_threshold_detection), float(model.det_nms_thresh))
        assert thresholds == CANDIDATE_RULE, ("candidate rule differs from the registered one", thresholds)
        inj = spike.ExemplarInjector(model.detector)
        with torch.inference_mode():
            for s in scenes:
                part = os.path.join(args.parts_dir, f"{s}.json")
                if load_part(part, expected_keys(s, cfg, census)) is not None and not args.overwrite:
                    log(f"  {s}: complete part exists, skipping")
                    continue
                t1 = time.time()
                res = run_scene(model, inj, s, cfg, census, args.data_root, log)
                json.dump(res, open(part + ".tmp", "w"), indent=1)
                os.replace(part + ".tmp", part)                       # atomic: no truncated part on interruption
                log(f"  {s}: {len(res)} rows in {time.time() - t1:.0f}s")
        inj.remove()
    missing = []
    for s in sorted(census):
        got = load_part(os.path.join(args.parts_dir, f"{s}.json"), expected_keys(s, cfg, census))
        if got is None:
            missing.append(s)
        else:
            rows += [tag_row(r, f0, n0) for r in got]
    summary = summarize_gate(rows)
    summary["missing_scenes"] = missing
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    doc = {"schema": "MVSeed e2_gate", "schema_version": 1, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "git_head": head, "generated_by": "MVSeed/gate_det.py", "argv": sys.argv, "prereg": "docs/stage1-E2-prereg.md",
           "rules": {"prompts": list(PROMPTS), "union": "plain union of the two runs' kept candidates, prob descending",
                     "candidate_rule": "package: prob > score_threshold_detection after nms_masks(det_nms_thresh)",
                     "score_threshold_detection": CANDIDATE_RULE[0], "det_nms_thresh": CANDIDATE_RULE[1],
                     "model_thresholds_read_back": thresholds,
                     "best_rank": "rank of the highest-ranked cross-run copy (low-res IoU > dedupe) of the best-IoU candidate",
                     "iou_exist": IOU_EXIST, "gate1_min_rate": GATE1_MIN_RATE, "gate1p_min": GATE1P_MIN,
                     "gate2_max_rank_median": GATE2_MAX_RANK_MEDIAN, "gate3_max_rate": GATE3_MAX_RATE,
                     "dedupe_iou_report_only": DEDUPE_IOU, "text": "TEXT_ID_FOR_VISUAL ('visual')"},
           "summary": summary, "rows": rows}
    json.dump(doc, open(args.out, "w"), indent=1)
    print_summary(summary)
    print(f"wrote {args.out}  ({len(rows)} rows, missing scenes {missing}, {time.time() - t0:.0f}s)")


def print_summary(s):
    def line(name, g):
        print(f"  {name:18s} n={g['n']:3d}  exist {g['exist']:3d} ({g['exist_rate']})  box {g['exist_box']:3d} pts {g['exist_pts']:3d}  "
              f"rank med {g['rank_median']} (raw {g['rank_raw_median']})  top1 {g['top1_hit']:3d}  union med {g['n_union_median']} (dedup {g['n_union_dedup_median']})  "
              f"rig {g['exist_by_rig']}")
    print(f"\n[S1-E2 gate] rows {s['n_rows']}  denominators match registered: {s['denominators']['match']} {s['denominators']['measured']}")
    for k in ("f0_ge2k", "f0_ge2k_pinhole", "f0_lt2k", "n0"):
        line(k, s[k])
    for k, g in s["f0_by_kind"].items():
        line(f"f0 kind={k}", g)
    for k, g in s["n0_by_size"].items():
        line(f"n0 {k}", g)
    e = s["gt_empty"]
    print(f"  gt_empty           n={e['n']:3d}  halluc {e['halluc']:3d} ({e['halluc_rate']})  by run {e['halluc_by_run']}  top1 prob med {e['halluc_top1_prob_median']}")
    for k, g in s["gates"].items():
        print(f"  {k}: {g}")


if __name__ == "__main__":
    main()
