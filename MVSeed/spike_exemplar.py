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

S1-E2-0' (docs/stage1-E2-0b-prereg.md, --variant e2_0b): the exemplar token is built from the
ground-truth MASK instead of its box.  Only the appearance part of the box token changes -- the
7x7 ROI-align over the feature grid (`_encode_boxes`, geometry_encoders.py 641-665) -- via a
second instance-level shadow, `geometry_encoder._encode_boxes`:
  mask     ROI-align(feats*m) / ROI-align(m): per-bin mean of the features inside the mask;
           bins with coverage < FILL_MIN take the object's coverage-weighted mean feature
           (registered arm; m == 1 everywhere reproduces the box token exactly -- self-test)
  mask0    ROI-align(feats*m) without normalisation ("zero outside the mask", exploratory)
  maskimg  the whole feature grid zeroed outside the mask before the encoder, so pooling AND
           the three cross-attention layers see object cells only (exploratory)
m is the c_ini ground-truth mask area-averaged down to the feature grid (frames are plainly
resized to 1008x1008, so a full-frame resize matches).  Box coordinates, CLS and the
cross-attention context are unchanged.  Per pair every exemplar variant gets its own
same-image (criterion 1) and slot-injection (criterion 2) measurement; the verdict of
variant e2_0b is taken on `mask` alone.

S1-E2-0'' (docs/stage1-E2-0c-prereg.md, --variant e2_0c): mask information through the prompt paths the
shipped weights were trained with, no encoder change -- `exemplar_prompt` builds, per variant, a geometric
prompt of positive/negative boxes and positive points (Prompt(box_embeddings, box_labels, point_embeddings,
point_labels); labels 1 = positive, 0 = negative, exactly the package's `add_prompt` format):
  pts      positive box + K_POINTS positive points inside the mask (farthest-point sampling on the interior)
  neg      positive box + negative boxes = other c_ini objects whose tight box overlaps the target box
  ptsonly / negall / ptsneg   exploratory (points without the box; all other objects as negatives; both)
The verdict of variant e2_0c is taken per registered arm (pts, neg) with the E2-0 kill rules plus an
adoption rule against the box arm (summarize()).

    docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \\
      --user $(id -u):$(id -g) -v /:/host -w /host$PWD \\
      -e HF_HOME=/host$PWD/SCSam3/hf_cache -e HF_HUB_OFFLINE=1 scsam3 \\
      python MVSeed/spike_exemplar.py --set registered          # then --set all
    python MVSeed/spike_exemplar.py --variant e2_0b --set all   # S1-E2-0' (mask exemplar)
    python MVSeed/spike_exemplar.py --variant e2_0c --set all   # S1-E2-0'' (points / negatives)
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
FILL_MIN = 0.05     # `mask` arm: ROI bins with mask coverage below this take the object-mean feature
VARIANTS = {
    # exemplar variants measured (each -> same_<v> and slot_<v>), extra arms, verdict arm, output
    "e2_0": dict(exemplars=("box",), extra=("replace",), verdict="box",
                 out="e2_0_spike.json", parts="e2_0_spike_parts", prereg="docs/stage1-E2-0-prereg.md"),
    "e2_0b": dict(exemplars=("box", "mask", "mask0", "maskimg"), extra=(), verdict="mask",
                  out="e2_0b_spike.json", parts="e2_0b_spike_parts", prereg="docs/stage1-E2-0b-prereg.md"),
    # S1-E2-0'' (docs/stage1-E2-0c-prereg.md): mask information through the shipped prompt paths --
    # positive points inside the mask, negative box exemplars on the other c_ini objects
    "e2_0c": dict(exemplars=("box", "pts", "neg", "ptsonly", "negnc", "negall", "ptsneg"), extra=(), verdict=("pts", "neg"),
                  out="e2_0c_spike.json", parts="e2_0c_spike_parts", prereg="docs/stage1-E2-0c-prereg.md"),
}
K_POINTS = 8        # `pts`: positive points inside the mask (farthest-point sampling on the interior)
INTERIOR_PX = 3     # `pts`: interior = distance to the mask boundary >= this many original-resolution pixels
MAX_NEG = 8         # `neg`/`negall`: at most this many negative boxes
NEG_MIN_OVERLAP = 0.01   # `neg`: another object's tight box counts when its intersection with the target box >= 1% of the target box
MASK_EXEMPLARS = ("mask", "mask0", "maskimg")
PROMPT_EXEMPLARS = ("box", "pts", "neg", "ptsonly", "negnc", "negall", "ptsneg")


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


def arm_keys(variant):
    """Exemplar variant -> (same-image arm key, slot-injection arm key).  E2-0 keys stay unchanged."""
    return ("same", "slot") if variant == "box" else (f"same_{variant}", f"slot_{variant}")


def _c1(arms):
    return {"n": len(arms),
            "hits": sum(a["cand_best_iou"] > IOU_SAME for a in arms),
            "rate": _rate(sum(a["cand_best_iou"] > IOU_SAME for a in arms), len(arms)),
            "any_query_hits": sum(a["all_best_iou"] > IOU_SAME for a in arms),
            "n_cand_median": _median([a["n_cand"] for a in arms]),
            "cand_best_iou_median": _median([a["cand_best_iou"] for a in arms])}


def _c2(arms, with_dlogit):
    hits = [a for a in arms if a["cand_best_iou"] >= IOU_CROSS]
    c = {"n": len(arms), "hits": len(hits), "rate": _rate(len(hits), len(arms)),
         "rank_of_hit_median": _median([a["cand_best_rank"] for a in hits]),
         "rank_of_hit_list": sorted(a["cand_best_rank"] for a in hits),
         "n_cand_median": _median([a["n_cand"] for a in arms]),
         "n_cand_list": [a["n_cand"] for a in arms],
         "any_query_hits": sum(a["all_best_iou"] >= IOU_CROSS for a in arms),
         "top1_is_hit": sum(a["cand_top1_iou"] >= IOU_CROSS for a in arms),
         "cand_best_iou_median": _median([a["cand_best_iou"] for a in arms])}
    if with_dlogit:
        c["max_abs_dlogit_vs_base_min"] = min(a["max_abs_dlogit_vs_base"] for a in arms)
    return c


def _paired(rs, key_a, key_b, fill_key="fill_ratio", tol=0.05):
    """Pair-level comparison of arm key_a against arm key_b (cand_best_iou), split by box fill ratio."""
    def block(rows):
        d = [r["arms"][key_a]["cand_best_iou"] - r["arms"][key_b]["cand_best_iou"] for r in rows]
        return {"n": len(rows), "d_median": _median(d),
                "n_up": sum(x > tol for x in d), "n_down": sum(x < -tol for x in d),
                f"hits_{key_b}": sum(r["arms"][key_b]["cand_best_iou"] >= IOU_CROSS for r in rows),
                f"hits_{key_a}": sum(r["arms"][key_a]["cand_best_iou"] >= IOU_CROSS for r in rows),
                "d_list": [round(x, 4) for x in d]}
    out = {"all": block(rs)}
    lo = [r for r in rs if r.get(fill_key) is not None and r[fill_key] < 0.5]
    hi = [r for r in rs if r.get(fill_key) is not None and r[fill_key] >= 0.5]
    out["fill_lt_0.5"] = block(lo)
    out["fill_ge_0.5"] = block(hi)
    return out


def compare_with_reference(records, reference, arms=("same", "slot", "base")):
    """Determinism check: the recomputed box arms against E2-0's stored pairs (same scene/cam/obj)."""
    ref = {(r["scene"], r["cam"], r["obj"]): r for r in reference}
    n = n_eq = 0
    max_diff = 0.0
    for r in records:
        q = ref.get((r["scene"], r["cam"], r["obj"]))
        if q is None:
            continue
        n += 1
        diff = max(abs(r["arms"][a]["cand_best_iou"] - q["arms"][a]["cand_best_iou"]) for a in arms)
        same_n = all(r["arms"][a]["n_cand"] == q["arms"][a]["n_cand"] for a in arms)
        max_diff = max(max_diff, diff)
        n_eq += int(diff == 0.0 and same_n)
    return {"n_compared": n, "n_equal": n_eq, "max_abs_diff_cand_best_iou": round(max_diff, 4), "arms": list(arms)}


def summarize(records, thresholds, variant="e2_0", reference=None):
    """Rates for the verdict and the tables.  Pure (plain dicts in, plain dicts out)."""
    spec = VARIANTS[variant]
    out = {}
    for tag in ("registered", "pinhole_supp"):
        rs = [r for r in records if r["set"] == tag]
        if not rs:
            continue
        s = {"n_pairs": len(rs)}
        for v in spec["exemplars"]:
            sk, lk = arm_keys(v)
            s["c1_same_image" if v == "box" else f"c1_{sk}"] = _c1([r["arms"][sk] for r in rs])
        slot_arms = [arm_keys(v)[1] for v in spec["exemplars"]] + list(spec["extra"]) + ["base"]
        for arm in slot_arms:
            s[f"c2_{arm}"] = _c2([r["arms"][arm] for r in rs], with_dlogit=(arm != "base"))
        for v in spec["exemplars"]:
            if v != "box":
                s[f"paired_slot_{v}_vs_box"] = _paired(rs, arm_keys(v)[1], "slot")
                s[f"paired_same_{v}_vs_box"] = _paired(rs, arm_keys(v)[0], "same")
        fills = [r["fill_ratio"] for r in rs if r.get("fill_ratio") is not None]
        if fills:
            s["fill_ratio_median"] = _median(fills)
        out[tag] = s
    reg = out.get("registered")
    if reg:
        multi = isinstance(spec["verdict"], tuple)          # e2_0c: two registered arms + adoption rule
        arms = spec["verdict"] if multi else (spec["verdict"],)
        verdicts = {}
        for vk in arms:
            sk, lk = arm_keys(vk)
            c1k = "c1_same_image" if vk == "box" else f"c1_{sk}"
            c1, c2 = reg[c1k]["rate"], reg[f"c2_{lk}"]["rate"]
            if c1 < KILL_SAME:
                verdict = f"discard: criterion 1 (same-image exemplar, arm {vk}) below {KILL_SAME}"
            elif c2 < KILL_CROSS:
                verdict = f"discard: criterion 2 (cross-view slot injection, arm {vk}) below {KILL_CROSS}"
            else:
                verdict = f"pass: proceed to S1-E2 with the {vk} exemplar"
            v = {"arm": vk, "criterion_1_rate": c1, "criterion_2_rate": c2,
                 "kill_rules": {"c1_min": KILL_SAME, "c2_min": KILL_CROSS}, "verdict": verdict}
            if multi:
                # adoption over the box exemplar (docs/stage1-E2-0c-prereg.md section 3): no loss on (1),
                # (2) >= 0.7, paired (2) hits >= box hits and paired median difference >= 0
                pr = reg[f"paired_slot_{vk}_vs_box"]["all"]
                adopt = (reg[c1k]["hits"] >= reg["c1_same_image"]["hits"] and c2 >= KILL_CROSS
                         and pr[f"hits_slot_{vk}"] >= pr["hits_slot"] and pr["d_median"] >= 0)
                v["adopt_over_box"] = bool(adopt)
                v["adopt_rule"] = ("c1 hits >= box c1 hits and c2 >= 0.7 and paired c2 hits >= box and "
                                   "paired median d(cand_best_iou) >= 0")
                v["paired_hits"], v["paired_d_median"] = pr[f"hits_slot_{vk}"], pr["d_median"]
            verdicts[vk] = v
        if multi:
            # between-arm selection (section 3): among adoptable arms, more paired (2) hits, then larger
            # median difference; ties keep the registration order (pts first).  None -> box retained.
            adoptable = [vk for vk in arms if verdicts[vk]["adopt_over_box"]]
            adopted = max(adoptable, key=lambda vk: (verdicts[vk]["paired_hits"], verdicts[vk]["paired_d_median"],
                                                     -arms.index(vk))) if adoptable else None
            out["verdicts"] = verdicts
            out["adopted"] = adopted
            out["adoption_rule"] = "adoptable arms ranked by paired c2 hits, then paired median d; tie -> first registered (pts); none -> box retained"
            out["verdict"] = dict(verdicts[adopted] if adopted else verdicts[arms[0]],
                                  adopted=adopted, box_retained=adopted is None)
        else:
            out["verdict"] = verdicts[arms[0]]
    if reference is not None:
        out["box_vs_e2_0_reference"] = compare_with_reference(records, reference)
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

    Exemplar variants (S1-E2-0', docs/stage1-E2-0b-prereg.md) act only while `capturing`:
    "box"      untouched package path
    "mask"     `_encode_boxes` pools the features inside the c_ini mask only (masked_roi_pool, normalised)
    "mask0"    `_encode_boxes` pools feats * m without normalisation
    "maskimg"  a forward pre-hook zeroes the last-level feature grid outside the mask
    """

    def __init__(self, detector):
        self.detector = detector
        self.mode = None
        self.embed = None
        self.mask = None
        self.captured = None
        self.exemplar = "box"
        self.capturing = False
        self.gt = None                  # c_ini ground-truth mask, bool [H0, W0], for the mask variants
        self._grid_cache = {}
        self.selftest = None            # max |mask(m=1) - box| token difference, set on the first mask capture
        self._orig_encode_prompt = detector._encode_prompt          # bound Sam3Image method
        detector._encode_prompt = self._encode_prompt                # shadows it on the instance
        geo = detector.geometry_encoder
        self._orig_encode_boxes = geo._encode_boxes                  # bound SequenceGeometryEncoder method
        geo._encode_boxes = self._encode_boxes                       # shadows it on the instance
        self._hook = geo.register_forward_hook(self._geo_hook)
        self._pre = geo.register_forward_pre_hook(self._geo_pre_hook, with_kwargs=True)

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

    def grid_mask(self, H, W, device, dtype):
        key = (H, W)
        if key not in self._grid_cache:
            self._grid_cache[key] = feat_mask_from_gt(self.gt, H, W)
        return self._grid_cache[key].to(device=device, dtype=dtype)

    def _geo_pre_hook(self, module, args, kwargs):
        """maskimg: the last-level feature grid [H*W, B, C] is zeroed outside the mask (new tensors only)."""
        if not (self.capturing and self.exemplar == "maskimg"):
            return None
        assert not args, "the package calls the geometry encoder with keyword arguments"
        feats = list(kwargs["img_feats"])
        H, W = kwargs["img_sizes"][-1]
        f = feats[-1]
        assert f.shape[0] == H * W and f.shape[1] == 1
        m = self.grid_mask(H, W, f.device, f.dtype).reshape(H * W, 1, 1)
        feats[-1] = f * m
        kwargs = dict(kwargs)
        kwargs["img_feats"] = feats
        return (args, kwargs)

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
        if not (self.capturing and self.exemplar in ("mask", "mask0")):
            return self._orig_encode_boxes(boxes, boxes_mask, boxes_labels, img_feats)
        import torch
        enc = self.detector.geometry_encoder
        H, W = img_feats.shape[-2:]
        m = self.grid_mask(H, W, img_feats.device, img_feats.dtype)
        if self.selftest is None and self.exemplar == "mask":
            ref, _ = self._orig_encode_boxes(boxes, boxes_mask, boxes_labels, img_feats)
            got, _ = masked_encode_boxes(enc, boxes, boxes_mask, boxes_labels, img_feats,
                                         torch.ones_like(m), normalize=True, fill_min=FILL_MIN)
            self.selftest = float((got - ref).abs().max())
            assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), f"mask(m=1) != box token: {self.selftest}"
        return masked_encode_boxes(enc, boxes, boxes_mask, boxes_labels, img_feats, m,
                                   normalize=(self.exemplar == "mask"), fill_min=FILL_MIN)

    def begin_capture(self, exemplar, gt=None):
        assert exemplar in MASK_EXEMPLARS + PROMPT_EXEMPLARS, exemplar
        self.exemplar, self.capturing, self.mode = exemplar, True, None
        if gt is not None and gt is not self.gt:
            self.gt, self._grid_cache = gt, {}

    def end_capture(self):
        self.capturing = False
        self.exemplar = "box"

    def store(self):
        feats, masks = self.captured
        self.embed, self.mask = feats.detach().clone(), masks.detach().clone()

    def remove(self):
        self._hook.remove()
        self._pre.remove()
        del self.detector._encode_prompt
        del self.detector.geometry_encoder._encode_boxes


def feat_mask_from_gt(gt_bool, H, W):
    """Coverage of the ground-truth mask on the H x W feature grid (area average), float [1, 1, H, W].

    The package resizes every frame to image_size x image_size with a plain cv2.resize (no
    padding, no aspect preservation), so an area resize of the full-frame mask is consistent.
    """
    import torch
    import torch.nn.functional as F
    g = gt_bool.float()[None, None]
    return F.interpolate(g, size=(int(H), int(W)), mode="area")


def masked_roi_pool(img_feats, m, rois, roi_size, normalize, fill_min):
    """ROI-align of the features inside the mask.

    img_feats [B, C, H, W], m [B, 1, H, W] coverage in [0, 1], rois: list of B tensors [n_b, 4] (xyxy, grid px).
    normalize=True : per-bin mean over the mask, ROI-align(feats*m) / ROI-align(m); bins whose coverage is
                     below fill_min take the object's coverage-weighted mean feature (so the 7x7 grid the
                     projection conv sees is object appearance everywhere, at the box token's magnitude)
    normalize=False: ROI-align(feats*m) as is (the naive "zero outside the mask")
    Same roi_align defaults as the package (spatial_scale 1, sampling_ratio -1, aligned False).
    m == 1 everywhere reproduces torchvision.ops.roi_align(img_feats, rois, roi_size) exactly.
    """
    import torch
    import torchvision
    num = torchvision.ops.roi_align(img_feats * m, rois, roi_size)            # [sum n_b, C, r, r]
    if not normalize:
        return num
    den = torchvision.ops.roi_align(m, rois, roi_size)                        # [sum n_b, 1, r, r]
    pooled = num / den.clamp(min=1e-6)
    fill = (img_feats * m).sum((2, 3)) / m.sum((2, 3)).clamp(min=1e-6)        # [B, C]
    counts = [int(r.shape[0]) for r in rois]
    fill = torch.cat([fill[b:b + 1].expand(n, -1) for b, n in enumerate(counts)], 0)[:, :, None, None]
    return torch.where(den < fill_min, fill.expand_as(pooled), pooled)


def masked_encode_boxes(enc, boxes, boxes_mask, boxes_labels, img_feats, m, normalize, fill_min):
    """`SequenceGeometryEncoder._encode_boxes` (geometry_encoders.py 632-681) with the ROI pooling
    replaced by masked_roi_pool; direct projection, box position encoding and label embedding unchanged."""
    import torch
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    assert enc.boxes_direct_project is not None and enc.boxes_pool_project is not None \
        and enc.boxes_pos_enc_project is not None, "the shipped encoder has all three box paths"
    n_boxes, bs = boxes.shape[:2]
    boxes_embed = enc.boxes_direct_project(boxes)
    H, W = img_feats.shape[-2:]
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device).view(1, 1, 4)
    rois = (boxes_xyxy * scale).float().transpose(0, 1).unbind(0)
    assert tuple(m.shape) == (img_feats.shape[0], 1, H, W), (tuple(m.shape), tuple(img_feats.shape))
    sampled = masked_roi_pool(img_feats, m.to(img_feats.dtype), rois, enc.roi_size, normalize, fill_min)
    assert list(sampled.shape) == [bs * n_boxes, enc.d_model, enc.roi_size, enc.roi_size]
    proj = enc.boxes_pool_project(sampled).view(bs, n_boxes, enc.d_model).transpose(0, 1)
    boxes_embed = boxes_embed + proj
    cx, cy, w, h = boxes.unbind(-1)
    pe = enc.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
    pe = pe.view(boxes.shape[0], boxes.shape[1], pe.shape[-1])
    boxes_embed = boxes_embed + enc.boxes_pos_enc_project(pe)
    type_embed = enc.label_embed(boxes_labels.long())
    return type_embed + boxes_embed, boxes_mask


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


def tight_box_xyxy(mask):
    ys, xs = mask.nonzero(as_tuple=True)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def interior_points(gt, k=K_POINTS, interior_px=INTERIOR_PX):
    """K positive points inside the mask: farthest-point sampling over the interior (exact Euclidean
    distance to the mask boundary >= interior_px on the original-resolution mask; the frame border counts
    as boundary, so no point lands on the outermost rows/columns), seeded at the deepest pixel.
    Falls back to the whole mask when the interior has fewer than k pixels.  Returns (k', 2) xy pixels."""
    import cv2
    import numpy as np
    import torch
    m = gt.detach().cpu().numpy().astype(np.uint8)
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    cand = np.argwhere(dt >= interior_px) if (dt >= interior_px).sum() >= k else np.argwhere(m > 0)
    if len(cand) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if len(cand) > 20000:                       # deterministic stride subsample (row-major order)
        cand = cand[:: int(np.ceil(len(cand) / 20000))]
    pts = torch.from_numpy(cand.astype(np.float32))              # (n, 2) as (y, x)
    depth = torch.from_numpy(dt[cand[:, 0], cand[:, 1]])
    chosen = [int(depth.argmax())]
    dmin = ((pts - pts[chosen[0]]) ** 2).sum(1)
    while len(chosen) < min(k, len(pts)):
        nxt = int(dmin.argmax())
        chosen.append(nxt)
        dmin = torch.minimum(dmin, ((pts - pts[nxt]) ** 2).sum(1))
    yx = cand[chosen]
    return yx[:, ::-1].copy()                                     # (k', 2) as (x, y)


def negative_boxes(gt_all, obj, box, max_neg=MAX_NEG, min_overlap=NEG_MIN_OVERLAP, mode="overlap"):
    """Other labelled objects of the same c_ini frame as negative exemplar boxes (xyxy pixels).
    mode "overlap": tight boxes intersecting the target box by >= min_overlap of the target box area,
                    largest intersection first, at most max_neg.
    mode "partial": as "overlap" but boxes that contain the whole target box (floor, wall, table) are skipped.
    mode "all"    : every other object, nearest box centre first, at most max_neg."""
    import torch
    x0, y0, x1, y1 = box
    barea = float((x1 - x0) * (y1 - y0))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    cands = []
    for v in torch.unique(gt_all).tolist():
        v = int(v)
        if v in (0, obj):
            continue
        ob = tight_box_xyxy(gt_all == v)
        ix = max(0, min(x1, ob[2]) - max(x0, ob[0]))
        iy = max(0, min(y1, ob[3]) - max(y0, ob[1]))
        inter = float(ix * iy)
        dist = ((ob[0] + ob[2]) / 2.0 - cx) ** 2 + ((ob[1] + ob[3]) / 2.0 - cy) ** 2
        contains = ob[0] <= x0 and ob[1] <= y0 and ob[2] >= x1 and ob[3] >= y1
        if mode in ("overlap", "partial"):
            if inter >= min_overlap * barea and not (mode == "partial" and contains):
                cands.append((-inter, v, ob))
        else:
            cands.append((dist, v, ob))
    cands.sort(key=lambda t: (t[0], t[1]))
    return [(v, list(ob)) for _, v, ob in cands[:max_neg]]


def exemplar_prompt(variant, gt_all, obj, device):
    """The geometric prompt of one exemplar variant for target `obj` in the c_ini frame.

    box     : positive tight box                                   (E2-0)
    pts     : positive box + K positive interior points
    ptsonly : K positive interior points, no box                   (exploratory)
    neg     : positive box + negative boxes of the other objects overlapping it
    negnc   : as neg, but negatives whose box contains the target box are skipped     (exploratory)
    negall  : positive box + negative boxes of all other objects (nearest first, <= MAX_NEG) (exploratory)
    ptsneg  : positive box + interior points + overlapping negatives (exploratory)
    Returns (Prompt, meta) with meta = {"box": xyxy, "points": [[x, y], ...], "negatives": [[id, xyxy], ...]}.
    """
    import torch
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.model.geometry_encoders import Prompt
    gt = gt_all == obj
    H, W = gt.shape
    box = tight_box_xyxy(gt)
    boxes, labels = [], []
    if variant != "ptsonly":
        boxes.append(box); labels.append(1)
    negs = []
    if variant in ("neg", "ptsneg"):
        negs = negative_boxes(gt_all, obj, box, mode="overlap")
    elif variant == "negnc":
        negs = negative_boxes(gt_all, obj, box, mode="partial")
    elif variant == "negall":
        negs = negative_boxes(gt_all, obj, box, mode="all")
    for _, nb in negs:
        boxes.append(nb); labels.append(0)
    pts = interior_points(gt) if variant in ("pts", "ptsonly", "ptsneg") else None
    kw = {}
    if boxes:
        xywh = torch.tensor([[b[0] / W, b[1] / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H] for b in boxes], dtype=torch.float32)
        kw["box_embeddings"] = box_xywh_to_cxcywh(xywh)[:, None, :].to(device)            # (n, bs=1, 4)
        kw["box_mask"] = torch.zeros(1, len(boxes), dtype=torch.bool, device=device)         # nothing padded
        kw["box_labels"] = torch.tensor(labels, dtype=torch.long, device=device)[:, None]    # (n, bs)
    if pts is not None and len(pts):
        # pixel centres in [0, 1]: grid_sample(align_corners=False) maps (i + 0.5) / W to the centre of cell i
        xy = torch.tensor([[(float(x) + 0.5) / W, (float(y) + 0.5) / H] for x, y in pts], dtype=torch.float32)
        kw["point_embeddings"] = xy[:, None, :].to(device)                                    # (k, bs, 2)
        kw["point_mask"] = torch.zeros(1, len(pts), dtype=torch.bool, device=device)
        kw["point_labels"] = torch.ones(len(pts), 1, dtype=torch.long, device=device)
    assert kw, variant
    meta = {"box": list(box), "points": [[int(x), int(y)] for x, y in (pts if pts is not None else [])],
            "negatives": [[v, ob] for v, ob in negs]}
    return Prompt(**kw), meta


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


def run_scene(model, inj, scene, rows, cfg, data_root, log, variant="e2_0"):
    spec = VARIANTS[variant]
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
        fill = int(gt_ref.sum()) / float((box[2] - box[0]) * (box[3] - box[1]))
        empty = state["constants"]["empty_geometric_prompt"]
        base, base_p = score_detections(base_out[cam], gt_tgt, model)
        arms, shapes, prompts = {"base": base}, None, {}
        for v in spec["exemplars"]:
            sk, lk = arm_keys(v)
            if v in PROMPT_EXEMPLARS and v != "box":
                vprompt, meta = exemplar_prompt(v, gts[ref_cam], obj, device)
                assert meta["box"] == box or v == "ptsonly"
                prompts[v] = {"n_points": len(meta["points"]), "points": meta["points"],
                              "negatives": meta["negatives"], "n_neg": len(meta["negatives"])}
            else:
                vprompt = prompt
            # (1) c_ini exemplar -> c_ini detections; the geometry-encoder hook captures the exemplar token
            inj.begin_capture(v, gt_ref)
            out = detector.forward_grounding(bo_ref if bo_ref is not None else fresh(),
                                             ib.find_inputs[0], None, vprompt)
            bo_ref = out["prev_encoder_out"]["backbone_out"]
            inj.store()
            inj.end_capture()
            arms[sk], _ = score_detections(out, gt_ref, model)
            if shapes is None:
                shapes = {"geo_feats": list(inj.embed.shape), "geo_masks": list(inj.mask.shape),
                          "txt_tokens": int(text_outputs["language_features"].shape[0])}
            if v in prompts:
                prompts[v]["geo_feats"] = list(inj.embed.shape)
            del out
            # (2) exemplar -> other view
            for mode in ("slot",) + (tuple(spec["extra"]) if v == "box" else ()):
                inj.mode = mode
                out2 = detector.forward_grounding(bo_cache[cam], ib.find_inputs[cam_idx[cam]], None, empty)
                rec, p2 = score_detections(out2, gt_tgt, model)
                rec["max_abs_dlogit_vs_base"] = round(float(
                    (out2["pred_logits"][0, :, 0] - base_out[cam]["pred_logits"][0, :, 0]).abs().max()), 4)
                arms[lk if mode == "slot" else mode] = rec
                del out2
            inj.mode = None
        rec = dict(r)
        rec.update({"ref_cam": ref_cam, "ref_gt_px": int(gt_ref.sum()), "exemplar_box_xyxy": box,
                    "fill_ratio": round(fill, 4), "orig_hw": list(hw), "prompt_shapes": shapes,
                    "selftest_mask_eq_box_maxdiff": inj.selftest, "prompts": prompts, "arms": arms})
        results.append(rec)
        parts = []
        for v in spec["exemplars"]:
            sk, lk = arm_keys(v)
            parts.append(f"{v} {arms[sk]['cand_best_iou']:.2f}/{arms[lk]['cand_best_iou']:.2f}"
                         f"r{arms[lk]['cand_best_rank']}")
        log(f"  {scene:20s} obj {obj:3d} {ref_cam}->{cam} fill {fill:.2f}  " + "  ".join(parts)
            + (f"  repl {arms['replace']['cand_best_iou']:.2f}" if "replace" in arms else "")
            + f"  base {base['cand_best_iou']:.2f}/{base['n_cand']}")
    del base_out, bo_cache, bo_ref, state, frames, gts
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=("registered", "pinhole_supp", "all"), default="all")
    ap.add_argument("--scenes", default="", help="comma-separated subset (debug)")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="e2_0",
                    help="e2_0 = box exemplar (S1-E2-0); e2_0b = box + mask exemplars (S1-E2-0')")
    ap.add_argument("--out", default=None, help="default MVSeed/runs/<variant out>")
    ap.add_argument("--parts-dir", default=None)
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--overwrite", action="store_true", help="recompute scenes that have a part file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    spec = VARIANTS[args.variant]
    args.out = args.out or os.path.join(REPO, "MVSeed", "runs", spec["out"])
    args.parts_dir = args.parts_dir or os.path.join(REPO, "MVSeed", "runs", spec["parts"])

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
            res = run_scene(model, inj, s, rs, cfg, args.data_root, log, variant=args.variant)
            json.dump(res, open(part, "w"), indent=1)
            log(f"  {s}: {len(res)} pairs in {time.time() - t1:.0f}s")
            records += res
    inj.remove()
    if args.variant != "e2_0" and inj.selftest is not None:
        print(f"self-test mask(m=1) vs box token: max |diff| = {inj.selftest:.2e}", flush=True)

    thresholds = {"score_threshold_detection": model.score_threshold_detection,
                  "det_nms_thresh": model.det_nms_thresh, "iou_same": IOU_SAME, "iou_cross": IOU_CROSS,
                  "min_gt_px": MIN_GT_PX, "text_id": "TEXT_ID_FOR_VISUAL ('visual')",
                  "autocast": "none (detector path in the mainline runs without autocast)",
                  "fill_min": FILL_MIN}
    reference = None
    if args.variant != "e2_0" and os.path.exists(OUT):
        reference = json.load(open(OUT))["pairs"]
    summary = summarize(records, thresholds, variant=args.variant, reference=reference)
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
           "variant": args.variant, "prereg": spec["prereg"],
           "arms": {"same": "c_ini GT box as geometric prompt -> c_ini detections",
                    "slot": "c_ini geo_feats injected as visual_prompt_embed on the other view (registered)",
                    "replace": "c_ini geo_feats replace the other view's geometry-encoder output (exploratory)",
                    "base": "other view with no exemplar (control)",
                    "same_mask/slot_mask": "exemplar token pooled inside the c_ini mask (normalised, object-mean fill)",
                    "same_mask0/slot_mask0": "feats*m pooled without normalisation",
                    "same_maskimg/slot_maskimg": "feature grid zeroed outside the mask before the encoder",
                    "same_pts/slot_pts": f"positive box + {K_POINTS} positive interior points (farthest-point, interior >= {INTERIOR_PX} px)",
                    "same_ptsonly/slot_ptsonly": "interior points only, no box (exploratory)",
                    "same_neg/slot_neg": f"positive box + negative tight boxes of other c_ini objects overlapping it (>= {NEG_MIN_OVERLAP:g} of box area, <= {MAX_NEG})",
                    "same_negnc/slot_negnc": "as neg, skipping negatives whose box contains the target box (exploratory)",
                    "same_negall/slot_negall": f"positive box + negative boxes of all other objects (nearest first, <= {MAX_NEG}) (exploratory)",
                    "same_ptsneg/slot_ptsneg": "box + interior points + overlapping negatives (exploratory)"},
           "selftest_mask_eq_box_maxdiff": inj.selftest,
           "summary": summary, "pairs": records}
    json.dump(doc, open(args.out, "w"), indent=1)
    print_summary(summary, args.variant)
    print(f"wrote {args.out}  ({len(records)} pairs, {time.time() - t0:.0f}s)")


def print_summary(summary, variant="e2_0"):
    spec = VARIANTS[variant]
    for tag in ("registered", "pinhole_supp"):
        s = summary.get(tag)
        if not s:
            continue
        print(f"\n[{tag}] n={s['n_pairs']}  fill_ratio median {s.get('fill_ratio_median')}")
        for v in spec["exemplars"]:
            sk, _ = arm_keys(v)
            c1 = s["c1_same_image" if v == "box" else f"c1_{sk}"]
            print(f"  (1) same-image {v:8s} IoU>{IOU_SAME}: {c1['hits']}/{c1['n']} = {c1['rate']}  "
                  f"(any of 200 queries: {c1['any_query_hits']}; candidates median {c1['n_cand_median']}; "
                  f"best IoU median {c1['cand_best_iou_median']})")
        slot_arms = [arm_keys(v)[1] for v in spec["exemplars"]] + list(spec["extra"]) + ["base"]
        for arm in slot_arms:
            c = s[f"c2_{arm}"]
            extra = f"  dlogit_min {c['max_abs_dlogit_vs_base_min']}" if arm != "base" else ""
            print(f"  (2) {arm:12s} IoU>={IOU_CROSS}: {c['hits']}/{c['n']} = {c['rate']}  rank med {c['rank_of_hit_median']} "
                  f"{c['rank_of_hit_list']}  n_cand med {c['n_cand_median']}  top1-hit {c['top1_is_hit']}  "
                  f"any-query {c['any_query_hits']}  best IoU med {c['cand_best_iou_median']}{extra}")
        for v in spec["exemplars"]:
            if v == "box":
                continue
            for which in ("slot", "same"):
                pr = s[f"paired_{which}_{v}_vs_box"]
                a, b = pr["fill_lt_0.5"], pr["fill_ge_0.5"]
                print(f"      {which}_{v} - box: d med {pr['all']['d_median']:+.3f} up {pr['all']['n_up']} down {pr['all']['n_down']}"
                      f" | fill<0.5 n{a['n']} d med {a['d_median']} hits box {a[f'hits_{which}']}->{a[f'hits_{arm_keys(v)[0] if which == 'same' else arm_keys(v)[1]}']}"
                      f" | fill>=0.5 n{b['n']} d med {b['d_median']} hits box {b[f'hits_{which}']}->{b[f'hits_{arm_keys(v)[0] if which == 'same' else arm_keys(v)[1]}']}")
    ref = summary.get("box_vs_e2_0_reference")
    if ref:
        print(f"\nbox arms vs E2-0 stored pairs: {ref['n_equal']}/{ref['n_compared']} identical, "
              f"max |diff| cand_best_iou {ref['max_abs_diff_cand_best_iou']}")
    for v in (summary.get("verdicts") or {"_": summary.get("verdict")}).values():
        if v:
            extra = (f"  adopt_over_box={v['adopt_over_box']} (paired hits {v['paired_hits']}, d med {v['paired_d_median']:+.3f})"
                     if "adopt_over_box" in v else "")
            print(f"\nverdict (arm {v['arm']}): (1)={v['criterion_1_rate']}  (2)={v['criterion_2_rate']}  -> {v['verdict']}{extra}")
    if "adopted" in summary:
        print(f"adopted exemplar: {summary['adopted'] or 'none (box retained)'}")


if __name__ == "__main__":
    main()
