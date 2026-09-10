#!/usr/bin/env python3
"""S1-E2-0: does the frozen SAM 3 detector accept a cross-image exemplar?  (docs/stage1-E2-0-prereg.md)

The frozen package only uses a box exemplar inside the image it was drawn on: `add_prompt`
turns the first box into that frame's geometric prompt (SCSam3VideoInference.py 177-218,
898-902), `Sam3Image.forward_grounding` never fills the `visual_prompt_embed` argument of
`_encode_prompt` (sam3_image.py 446-448 vs 166-209), and `inference_state["visual_prompt_embed"]`
is written (None) but never read.  This spike fills that slot from a script -- no package
file is modified -- and measures whether the detector finds the same object in another view.

What is wrapped (all on the model *instance*, inside `ExemplarInjector`):
  * `detector._encode_prompt` is shadowed by a bound wrapper that passes the stored c_ini
    features as `visual_prompt_embed` / `visual_prompt_mask` (mode "slot").  The original
    concatenates them after the text and geometry tokens (sam3_image.py 204-205), and that
    prompt sequence feeds the fusion encoder, the decoder's text cross-attention, the
    dot-product scorer (mean-pooled) and the segmentation head's prompt cross-attention.
  * a forward hook on `detector.geometry_encoder` (a) captures its output (geo_feats,
    geo_masks) when c_ini is prompted with the ground-truth box -- that output *is* the
    exemplar feature `_encode_prompt` builds (ROI-align box token + CLS, three cross-attention
    layers over the c_ini feature grid) -- and (b) in mode "replace" returns the stored c_ini
    output in place of the target view's own (empty) geometry tokens.
  * detections are read from `detector.forward_grounding` directly, one frame per call, i.e.
    the call `run_backbone_and_detection` makes with `allow_new_detections=True`
    (sam3_video_base.py 345-374) minus the tracker; the package's own post-processing is
    reproduced exactly: `nms_masks(prob > score_threshold_detection, iou > det_nms_thresh)`
    and the bilinear upsample + `> 0` of `build_outputs` (sam3_video_base.py 977-984).

Text side: as `add_prompt(text_str=None)` does, the text id is TEXT_ID_FOR_VISUAL, i.e. the
text tokens are those of the word "visual" (find_text_batch[1]); no noun phrase anywhere.

Arms measured per (scene, object):
  same     c_ini box -> c_ini detections                              (criterion 1)
  slot     c_ini geo_feats -> other view, visual_prompt_embed slot    (criterion 2, registered)
  replace  c_ini geo_feats -> other view, replacing its geo tokens    (exploratory, reported only)
  base     other view, no exemplar at all ("visual" text only)        (control: is the exemplar
                                                                        doing anything?)
A "candidate" is a detection the package would keep: prob > 0.5 after NMS(0.1).  Per arm we
record the best candidate IoU with the ground truth, the rank (by prob) of that candidate,
the candidate count, and -- diagnostics -- the best IoU over all 200 queries and its rank.

Pair selection (registered): N0.json rows (raw seed J >= 0.5, so no F0 pair), gt_px >= 2000,
the other camera is one of the scene's annotated cameras; per scene the 4 rows with the
smallest view_distance (ties: camera name, object id); the 5 scenes ranked by that smallest
view_distance (ties: scene name).  A supplementary pinhole-only set (next 5 pinhole scenes by
the same rule, registered scenes excluded) is run and reported but does not enter the verdict.

    docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \\
      --user $(id -u):$(id -g) -v /:/host -w /host$PWD \\
      -e HF_HOME=/host$PWD/SCSam3/hf_cache -e HF_HUB_OFFLINE=1 scsam3 \\
      python MVSeed/spike_exemplar.py --set registered          # then --set all
    python MVSeed/spike_exemplar.py --dry-run                   # selection only, no torch
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "SCSam3", "demoSCSam3MVOpt")
DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
N0 = os.path.join(REPO, "MVSeed", "runs", "N0.json")
F0 = os.path.join(REPO, "MVSeed", "runs", "F0.json")
OUT = os.path.join(REPO, "MVSeed", "runs", "e2_0_spike.json")
MIN_GT_PX = 2000
PER_SCENE = 4
N_SCENES = 5
IOU_SAME = 0.8      # criterion 1: same-image candidate IoU > 0.8
IOU_CROSS = 0.5     # criterion 2: cross-view candidate IoU >= 0.5
KILL_SAME = 0.8     # criterion 1 rate below this -> discard
KILL_CROSS = 0.7    # criterion 2 rate below this -> discard
ARMS = ("same", "slot", "replace", "base")


def cam_name(c, d):
    return f"{d['prefix']}{c:0{d['prefix1']}d}"


def select_pairs(n0_pairs, f0_pairs, cfg, per_scene=PER_SCENE, n_scenes=N_SCENES,
                 min_gt_px=MIN_GT_PX):
    """Registered and supplementary pair sets.  Pure; see the module docstring for the rule."""
    f0 = {(p["scene"], p["cam"], p["obj"]) for p in f0_pairs}
    by_scene = defaultdict(list)
    for p in n0_pairs:
        key = (p["scene"], p["cam"], p["obj"])
        if key in f0 or p["gt_px"] < min_gt_px or p["scene"] not in cfg:
            continue
        d = cfg[p["scene"]]
        if "c_ini" not in d:
            continue
        annotated = {cam_name(c, d) for c in d["cam_list"]} - {cam_name(d["c_ini"], d)}
        if p["cam"] not in annotated:
            continue
        by_scene[p["scene"]].append(p)
    ranked = []
    for scene, rows in by_scene.items():
        rows.sort(key=lambda p: (p["view_distance"], p["cam"], p["obj"]))
        if len(rows) < per_scene:
            continue
        top = rows[:per_scene]
        ranked.append((top[0]["view_distance"], scene, top))
    ranked.sort(key=lambda r: (r[0], r[1]))
    registered = ranked[:n_scenes]
    reg_names = {r[1] for r in registered}
    pinhole = [r for r in ranked if r[1] not in reg_names and r[2][0]["rig"] == "pinhole"]
    supplementary = pinhole[:n_scenes]

    def rows(sel, tag):
        out = []
        for _, scene, top in sel:
            for p in top:
                out.append({"set": tag, "scene": scene, "cam": p["cam"], "obj": int(p["obj"]),
                            "gt_px": int(p["gt_px"]), "view_distance": int(p["view_distance"]),
                            "rig": p["rig"], "raw_J": p["raw_J"]})
        return out
    return rows(registered, "registered") + rows(supplementary, "pinhole_supp")


def summarize(records, thresholds):
    """Rates for the verdict and the tables.  Pure (plain dicts in, plain dicts out)."""
    out = {}
    for tag in ("registered", "pinhole_supp"):
        rs = [r for r in records if r["set"] == tag]
        if not rs:
            continue
        s = {"n_pairs": len(rs)}
        same = [r["arms"]["same"] for r in rs]
        s["c1_same_image"] = {
            "n": len(same),
            "hits": sum(a["cand_best_iou"] > IOU_SAME for a in same),
            "rate": _rate(sum(a["cand_best_iou"] > IOU_SAME for a in same), len(same)),
            "any_query_hits": sum(a["all_best_iou"] > IOU_SAME for a in same),
            "n_cand_median": _median([a["n_cand"] for a in same]),
        }
        for arm in ("slot", "replace", "base"):
            arms = [r["arms"][arm] for r in rs]
            hits = [a for a in arms if a["cand_best_iou"] >= IOU_CROSS]
            s[f"c2_{arm}"] = {
                "n": len(arms),
                "hits": len(hits),
                "rate": _rate(len(hits), len(arms)),
                "rank_of_hit_median": _median([a["cand_best_rank"] for a in hits]),
                "rank_of_hit_list": sorted(a["cand_best_rank"] for a in hits),
                "n_cand_median": _median([a["n_cand"] for a in arms]),
                "n_cand_list": [a["n_cand"] for a in arms],
                "any_query_hits": sum(a["all_best_iou"] >= IOU_CROSS for a in arms),
                "top1_is_hit": sum(a["cand_top1_iou"] >= IOU_CROSS for a in arms),
                "cand_best_iou_median": _median([a["cand_best_iou"] for a in arms]),
            }
            if arm != "base":
                s[f"c2_{arm}"]["max_abs_dlogit_vs_base_min"] = min(
                    a["max_abs_dlogit_vs_base"] for a in arms)
        out[tag] = s
    reg = out.get("registered")
    if reg:
        c1, c2 = reg["c1_same_image"]["rate"], reg["c2_slot"]["rate"]
        if c1 < KILL_SAME:
            verdict = "discard: criterion 1 (same-image exemplar) below 0.8"
        elif c2 < KILL_CROSS:
            verdict = "discard: criterion 2 (cross-view slot injection) below 0.7"
        else:
            verdict = "pass: proceed to S1-E2"
        out["verdict"] = {"criterion_1_rate": c1, "criterion_2_rate": c2, "kill_rules": {
            "c1_min": KILL_SAME, "c2_min": KILL_CROSS}, "verdict": verdict}
    out["thresholds"] = thresholds
    return out


def _rate(h, n):
    return round(h / n, 4) if n else None


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


# ----------------------------------------------------------------------------- GPU part


class ExemplarInjector:
    """Instance-level wrapper: no package file or class is modified.

    mode None      pass-through; the geometry-encoder hook records its output in `captured`
    mode "slot"    `_encode_prompt` receives the stored exemplar as visual_prompt_embed/mask
    mode "replace" the geometry encoder's output is replaced by the stored exemplar
    """

    def __init__(self, detector):
        self.detector = detector
        self.mode = None
        self.embed = None
        self.mask = None
        self.captured = None
        self._orig_encode_prompt = detector._encode_prompt          # bound Sam3Image method
        detector._encode_prompt = self._encode_prompt                # shadows it on the instance
        self._hook = detector.geometry_encoder.register_forward_hook(self._geo_hook)

    def _encode_prompt(self, backbone_out, find_input, geometric_prompt,
                       visual_prompt_embed=None, visual_prompt_mask=None,
                       encode_text=True, prev_mask_pred=None):
        if self.mode == "slot":
            assert visual_prompt_embed is None and self.embed is not None
            visual_prompt_embed, visual_prompt_mask = self.embed, self.mask
        return self._orig_encode_prompt(
            backbone_out, find_input, geometric_prompt,
            visual_prompt_embed=visual_prompt_embed, visual_prompt_mask=visual_prompt_mask,
            encode_text=encode_text, prev_mask_pred=prev_mask_pred)

    def _geo_hook(self, module, args, output):
        if self.mode == "replace":
            assert self.embed is not None
            return (self.embed, self.mask)
        self.captured = output          # (geo_feats [seq, bs, C], geo_masks [bs, seq])
        return None

    def store(self):
        feats, masks = self.captured
        self.embed, self.mask = feats.detach().clone(), masks.detach().clone()

    def remove(self):
        self._hook.remove()
        del self.detector._encode_prompt


def box_prompt_from_mask(gt, device):
    """The exemplar Prompt exactly as `_get_visual_prompt` builds it from a UI box."""
    import torch
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.model.geometry_encoders import Prompt
    ys, xs = gt.nonzero(as_tuple=True)
    H, W = gt.shape
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    xywh = torch.tensor([[x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]], dtype=torch.float32)
    cxcywh = box_xywh_to_cxcywh(xywh)
    prompt = Prompt(
        box_embeddings=cxcywh[None, 0:1, :].to(device),             # (seq=1, bs=1, 4)
        box_mask=None,
        box_labels=torch.ones(1, 1, dtype=torch.long, device=device),
        point_embeddings=None, point_mask=None, point_labels=None)
    return prompt, [int(x0), int(y0), int(x1), int(y1)]


def score_detections(out, gt_bool, model, chunk=25):
    """The package's candidate rule + IoU against the ground truth (offline measurement)."""
    import torch
    import torch.nn.functional as F
    from sam3.perflib.nms import nms_masks
    probs = out["pred_logits"][0, :, 0].sigmoid()                    # (Q,)
    masks = out["pred_masks"][0]                                     # (Q, h, w) logits
    thr, nms_thr = model.score_threshold_detection, model.det_nms_thresh
    if nms_thr > 0.0:                                                # run_nms in run_backbone_and_detection
        keep = nms_masks(probs, masks, thr, nms_thr)
    else:
        keep = probs > thr
    keep = keep & (probs > thr)
    H, W = gt_bool.shape
    Q = probs.numel()
    ious = torch.zeros(Q, device=probs.device)
    gt_sum = gt_bool.sum()
    for i in range(0, Q, chunk):
        m = F.interpolate(masks[i:i + chunk, None].float(), size=(H, W),
                          mode="bilinear", align_corners=False)[:, 0] > 0
        inter = (m & gt_bool).flatten(1).sum(1).float()
        union = m.flatten(1).sum(1).float() + gt_sum - inter
        ious[i:i + chunk] = inter / union.clamp(min=1)
        del m
    order = probs.argsort(descending=True)
    rank_all = torch.empty_like(order)
    rank_all[order] = torch.arange(1, Q + 1, device=order.device)
    best_all = int(ious.argmax())
    rec = {"n_cand": int(keep.sum()),
           "all_best_iou": round(float(ious[best_all]), 4),
           "all_best_rank": int(rank_all[best_all]),
           "all_best_prob": round(float(probs[best_all]), 4)}
    if rec["n_cand"] > 0:
        cidx = keep.nonzero()[:, 0]
        cidx = cidx[probs[cidx].argsort(descending=True)]           # candidates by prob, desc
        ci = ious[cidx]
        b = int(ci.argmax())
        rec.update({"cand_best_iou": round(float(ci[b]), 4),
                    "cand_best_rank": b + 1,
                    "cand_best_prob": round(float(probs[cidx[b]]), 4),
                    "cand_top1_iou": round(float(ci[0]), 4),
                    "cand_top1_prob": round(float(probs[cidx[0]]), 4),
                    "cand_probs": [round(float(p), 3) for p in probs[cidx][:10]]})
    else:
        rec.update({"cand_best_iou": 0.0, "cand_best_rank": None, "cand_best_prob": None,
                    "cand_top1_iou": 0.0, "cand_top1_prob": None, "cand_probs": []})
    return rec, probs


def run_scene(model, inj, scene, rows, cfg, data_root, log):
    import cv2
    import numpy as np
    import torch
    from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU
    d = cfg[scene]
    start = d["start_frame"]
    ds_dir = os.path.join(data_root, d["folder"])
    ref_cam = cam_name(d["c_ini"], d)
    cams = [ref_cam] + sorted({r["cam"] for r in rows})
    device = model.device
    detector = model.detector

    frames, gts, hw = [], {}, None
    for cam in cams:
        cpu_image, height, width = load_video_frames(video_path=os.path.join(ds_dir, "Video", cam))
        image = AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)
        frames.append(image[start])
        gt = cv2.imread(os.path.join(ds_dir, "Mask", cam, f"{start:06d}.png"), cv2.IMREAD_GRAYSCALE)
        assert gt is not None and gt.shape == (height, width), (cam, gt is None or gt.shape, height, width)
        gts[cam] = torch.from_numpy(gt).to(device)
        hw = (height, width)
    # the package's own input batch (BatchedDatapoint) for a pseudo-video [c_ini, other cams]
    state = model.init_state(frames, hw[0], hw[1])
    ib = state["input_batch"]
    for t in range(len(frames)):                                     # add_prompt(text_str=None) does this
        ib.find_inputs[t].text_ids[...] = model.TEXT_ID_FOR_VISUAL
    text_outputs = detector.backbone.forward_text(ib.find_text_batch, device=device)
    fresh = lambda: {"img_batch_all_stages": ib.img_batch, **text_outputs}
    cam_idx = {c: i for i, c in enumerate(cams)}

    # baseline detections on every other view: no exemplar at all
    base_out, base_probs, bo_cache = {}, {}, {}
    inj.mode = None
    for cam in cams[1:]:
        empty = state["constants"]["empty_geometric_prompt"]
        out = detector.forward_grounding(fresh(), ib.find_inputs[cam_idx[cam]], None, empty)
        bo_cache[cam] = out["prev_encoder_out"]["backbone_out"]     # image features, reused below
        base_out[cam] = out
    bo_ref = None
    results = []
    for r in rows:
        obj, cam = r["obj"], r["cam"]
        gt_ref = gts[ref_cam] == obj
        gt_tgt = gts[cam] == obj
        assert int(gt_tgt.sum()) == r["gt_px"], (scene, cam, obj, int(gt_tgt.sum()), r["gt_px"])
        prompt, box = box_prompt_from_mask(gt_ref, device)
        # (1) c_ini box -> c_ini detections; the geometry-encoder hook captures the exemplar
        inj.mode = None
        out = detector.forward_grounding(bo_ref if bo_ref is not None else fresh(),
                                         ib.find_inputs[0], None, prompt)
        bo_ref = out["prev_encoder_out"]["backbone_out"]
        inj.store()
        same, _ = score_detections(out, gt_ref, model)
        shapes = {"geo_feats": list(inj.embed.shape), "geo_masks": list(inj.mask.shape),
                  "txt_tokens": int(text_outputs["language_features"].shape[0])}
        # (2) exemplar -> other view
        arms = {"same": same}
        empty = state["constants"]["empty_geometric_prompt"]
        base, base_p = score_detections(base_out[cam], gt_tgt, model)
        arms["base"] = base
        for mode in ("slot", "replace"):
            inj.mode = mode
            out2 = detector.forward_grounding(bo_cache[cam], ib.find_inputs[cam_idx[cam]], None, empty)
            rec, p2 = score_detections(out2, gt_tgt, model)
            rec["max_abs_dlogit_vs_base"] = round(float(
                (out2["pred_logits"][0, :, 0] - base_out[cam]["pred_logits"][0, :, 0]).abs().max()), 4)
            arms[mode] = rec
            del out2
        inj.mode = None
        rec = dict(r)
        rec.update({"ref_cam": ref_cam, "ref_gt_px": int(gt_ref.sum()), "exemplar_box_xyxy": box,
                    "orig_hw": list(hw), "prompt_shapes": shapes, "arms": arms})
        results.append(rec)
        log(f"  {scene:20s} obj {obj:3d} {ref_cam}->{cam}  same {same['cand_best_iou']:.3f}/{same['n_cand']:2d}  "
            f"slot {arms['slot']['cand_best_iou']:.3f} r{arms['slot']['cand_best_rank']}/{arms['slot']['n_cand']:2d}  "
            f"repl {arms['replace']['cand_best_iou']:.3f} r{arms['replace']['cand_best_rank']}/{arms['replace']['n_cand']:2d}  "
            f"base {base['cand_best_iou']:.3f}/{base['n_cand']:2d}  dlogit {arms['slot']['max_abs_dlogit_vs_base']:.2f}")
    del base_out, bo_cache, bo_ref, state, frames, gts
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=("registered", "pinhole_supp", "all"), default="all")
    ap.add_argument("--scenes", default="", help="comma-separated subset (debug)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--parts-dir", default=os.path.join(REPO, "MVSeed", "runs", "e2_0_spike_parts"))
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--overwrite", action="store_true", help="recompute scenes that have a part file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    n0, f0 = json.load(open(N0)), json.load(open(F0))
    rows = select_pairs(n0["pairs"], f0["pairs"], cfg)
    if args.set != "all":
        rows = [r for r in rows if r["set"] == args.set]
    if args.scenes:
        rows = [r for r in rows if r["scene"] in args.scenes.split(",")]
    scenes = []
    for r in rows:
        if r["scene"] not in scenes:
            scenes.append(r["scene"])
    print(f"pairs {len(rows)} in {len(scenes)} scenes:")
    for s in scenes:
        rs = [r for r in rows if r["scene"] == s]
        print(f"  {rs[0]['set']:13s} {s:20s} vd {rs[0]['view_distance']:2d} {rs[0]['rig']:10s} "
              + ", ".join(f"{r['cam']}:{r['obj']}({r['gt_px']})" for r in rs))
    if args.dry_run:
        return

    sys.path.insert(0, PKG)
    import torch
    from build_scsam3 import build_scsam3_video_model
    os.makedirs(args.parts_dir, exist_ok=True)
    t0 = time.time()
    model = build_scsam3_video_model()
    model.eval()
    print(f"model loaded in {time.time() - t0:.0f}s; score_threshold_detection={model.score_threshold_detection} "
          f"det_nms_thresh={model.det_nms_thresh}", flush=True)
    inj = ExemplarInjector(model.detector)
    log = lambda s: print(s, flush=True)
    records = []
    with torch.inference_mode():
        for s in scenes:
            part = os.path.join(args.parts_dir, f"{s}.json")
            rs = [r for r in rows if r["scene"] == s]
            if os.path.exists(part) and not args.overwrite:
                got = json.load(open(part))
                if [(g["cam"], g["obj"]) for g in got] == [(r["cam"], r["obj"]) for r in rs]:
                    log(f"  {s}: part exists, skipping")
                    records += got
                    continue
            t1 = time.time()
            res = run_scene(model, inj, s, rs, cfg, args.data_root, log)
            json.dump(res, open(part, "w"), indent=1)
            log(f"  {s}: {len(res)} pairs in {time.time() - t1:.0f}s")
            records += res
    inj.remove()

    thresholds = {"score_threshold_detection": model.score_threshold_detection,
                  "det_nms_thresh": model.det_nms_thresh, "iou_same": IOU_SAME, "iou_cross": IOU_CROSS,
                  "min_gt_px": MIN_GT_PX, "text_id": "TEXT_ID_FOR_VISUAL ('visual')",
                  "autocast": "none (detector path in the mainline runs without autocast)"}
    summary = summarize(records, thresholds)
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    doc = {"schema": "MVSeed e2_0_spike", "schema_version": 1,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head": head,
           "generated_by": "MVSeed/spike_exemplar.py", "argv": sys.argv,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "selection_rule": "N0 (no F0), gt_px>=2000, other annotated cam; per scene 4 smallest view_distance "
                             "(tie cam,obj); 5 scenes by smallest view_distance (tie scene name); "
                             "pinhole_supp = next 5 pinhole scenes",
           "arms": {"same": "c_ini GT box as geometric prompt -> c_ini detections",
                    "slot": "c_ini geo_feats injected as visual_prompt_embed on the other view (registered)",
                    "replace": "c_ini geo_feats replace the other view's geometry-encoder output (exploratory)",
                    "base": "other view with no exemplar (control)"},
           "summary": summary, "pairs": records}
    json.dump(doc, open(args.out, "w"), indent=1)
    print_summary(summary)
    print(f"wrote {args.out}  ({len(records)} pairs, {time.time() - t0:.0f}s)")


def print_summary(summary):
    for tag in ("registered", "pinhole_supp"):
        s = summary.get(tag)
        if not s:
            continue
        c1 = s["c1_same_image"]
        print(f"\n[{tag}] n={s['n_pairs']}")
        print(f"  (1) same-image  IoU>{IOU_SAME}: {c1['hits']}/{c1['n']} = {c1['rate']}  "
              f"(any of 200 queries: {c1['any_query_hits']}; candidates median {c1['n_cand_median']})")
        for arm in ("slot", "replace", "base"):
            c = s[f"c2_{arm}"]
            extra = f"  dlogit_min {c['max_abs_dlogit_vs_base_min']}" if arm != "base" else ""
            print(f"  (2) {arm:8s} IoU>={IOU_CROSS}: {c['hits']}/{c['n']} = {c['rate']}  rank med {c['rank_of_hit_median']} "
                  f"{c['rank_of_hit_list']}  n_cand med {c['n_cand_median']}  top1-hit {c['top1_is_hit']}  "
                  f"any-query {c['any_query_hits']}{extra}")
    v = summary.get("verdict")
    if v:
        print(f"\nverdict: (1)={v['criterion_1_rate']}  (2 slot)={v['criterion_2_rate']}  -> {v['verdict']}")


if __name__ == "__main__":
    main()
