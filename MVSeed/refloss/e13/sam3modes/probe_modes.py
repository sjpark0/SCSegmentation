#!/usr/bin/env python3
"""S1-E13 sam3-modes probe: single-image (no memory) prompting modes of the frozen SAM 3 package.

DEV ONLY: one non-keyframe frame (default Dog t=10); refuses the scene's ref_frame.  Everything is
measured against that dev frame's GT label maps.  One model load (methods/sam3_single.load_model ->
build_scsam3_video_model, the cached facebook/sam3 checkpoint) serves every mode:

  A  tracker as image model (methods/sam3_single.segment): point / box; add_tracker_new_mask (dense mask)
  B  SAM-1-task interactive predictor (sam1_task_predictor.SAM3InteractiveImagePredictor around
     model.tracker, features from the detector backbone's sam2 neck exactly as Sam3Image.predict_inst):
     point / box / dense mask_input / combinations
  C  detector (Sam3Processor around model.detector): text concept prompts; same-image exemplar box;
     cross-image exemplar (box token sequence encoded on the reference image, transplanted into the
     target image's prompt via the same concatenation Sam3Image._encode_prompt does)
  D  everything mode: not shipped by SAM 3 -> SAM-1-style point-grid emulation through the interactive
     head (32x32 grid, multimask, pred-IoU/stability/NMS filters) and a generic text prompt.
  E  dense mask prompts from methods/geomem/run.project_reference_masks (reference GT of the SAME dev
     frame reprojected with that frame's Pi3 cache, Cache/pi3_frames/<t>).

Run (inside scsam3, cwd stage1):
  python geomem/runs/_e13_sam3modes/probe_modes.py --scene Dog --t 10 --tgt camera_0004
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1 = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, STAGE1)

import cv2  # noqa: E402

from geomem.cache import pi3_frame_cache  # noqa: E402
from methods import sam3_single  # noqa: E402
from methods.geomem.run import project_reference_masks, reproj_inputs  # noqa: E402
from mvseg2d.io import load_labelmap, load_labels, load_meta  # noqa: E402


# ----------------------------------------------------------------------------------------------- utils
def interior_point(m):
    """(x, y) pixel of the mask's deepest interior point (max of the distance transform)."""
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(x), float(y)


def interior_points(m, k=3):
    """k well-separated interior points: repeated distance-transform argmax with a suppression disk."""
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    pts = []
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(d)), d.shape)
        if d[y, x] <= 0:
            break
        pts.append((float(x), float(y)))
        cv2.circle(d, (int(x), int(y)), int(max(2 * d[y, x], 40)), 0, -1)
    return pts


def prec_rec(p, g):
    i = np.logical_and(p, g).sum()
    return [r4(i / max(1, p.sum())), r4(i / max(1, g.sum()))]


def bbox_xyxy(m):
    ys, xs = np.nonzero(m)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def iou_np(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def r4(x):
    return None if x is None else (round(float(x), 4) if np.isfinite(x) else None)


class Timer:
    def __init__(self, torch):
        self.torch = torch

    def __call__(self, fn, reps=3):
        """(result of last call, median wall ms over `reps` calls after one warm-up call)."""
        t = self.torch
        t.cuda.synchronize()
        out = fn()
        t.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            t.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return out, round(1000 * float(np.median(ts)), 1)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(STAGE1, "Data"))
    ap.add_argument("--scene", default="Dog")
    ap.add_argument("--t", type=int, default=10)
    ap.add_argument("--tgt", default="camera_0004")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--grid", type=int, default=32)
    a = ap.parse_args()

    meta = load_meta(a.data, a.scene)
    assert a.t != int(meta["ref_frame"]), "DEV/TEST rule: the scene's ref_frame (keyframe) is the test set"
    assert a.t in [int(x) for x in meta["train_frames"]], (a.t, meta["train_frames"])
    ref_cam = meta["ref_cam"]
    assert a.tgt in meta["mask_cams"] and a.tgt != ref_cam
    fdir = os.path.join(a.data, a.scene, "frames", f"{a.t:06d}")
    labels = load_labels(a.data, a.scene)

    import torch
    import torch.nn.functional as F
    import torchvision
    from PIL import Image

    timer = Timer(torch)
    R = {"scene": a.scene, "t": a.t, "ref_frame": int(meta["ref_frame"]), "ref_cam": ref_cam, "tgt": a.tgt}

    t0 = time.perf_counter()
    model = sam3_single.load_model()
    torch.cuda.synchronize()
    R["model_load_s"] = round(time.perf_counter() - t0, 1)
    from sam3.model.geometry_encoders import Prompt
    from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model import box_ops
    from sam3.model.data_misc import interpolate

    det, trk = model.detector, model.tracker
    R["classes"] = {"model": type(model).__name__, "detector": type(det).__name__, "tracker": type(trk).__name__,
                    "detector.inst_interactive_predictor": type(det.inst_interactive_predictor).__name__,
                    "geometry_encoder.mask_encoder": type(det.geometry_encoder.mask_encoder).__name__,
                    "autocast_enabled": bool(torch.is_autocast_enabled()),
                    "sam_prompt_encoder.mask_input_size": list(trk.sam_prompt_encoder.mask_input_size),
                    "tracker.input_mask_size": int(trk.input_mask_size)}

    def rgb(cam):
        return cv2.cvtColor(cv2.imread(os.path.join(fdir, "images", f"{cam}.jpg")), cv2.COLOR_BGR2RGB)

    img_t, img_r = rgb(a.tgt), rgb(ref_cam)
    gt_t = load_labelmap(os.path.join(fdir, "masks", f"{a.tgt}.png"))
    gt_r = load_labelmap(os.path.join(fdir, "masks", f"{ref_cam}.png"))
    H, W = gt_t.shape
    assert img_t.shape[:2] == (H, W)
    ids_t = [int(i) for i in np.unique(gt_t) if i]
    ids_r = [int(i) for i in np.unique(gt_r) if i]
    R["objects_tgt"] = {i: {"label": labels.get(i, {}).get("label"), "area": int((gt_t == i).sum())} for i in ids_t}
    R["objects_ref"] = ids_r
    gt_tc = torch.from_numpy(gt_t).cuda()

    def iou_t(pred_bool, oid):
        g = gt_tc == oid
        u = (pred_bool | g).sum().item()
        return (pred_bool & g).sum().item() / u if u else float("nan")

    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        # ============================================================ A: tracker as image model (sam3_single)
        A = {}
        frame, h, w = sam3_single.load_frame(os.path.join(fdir, "images", f"{a.tgt}.jpg"))
        assert (h, w) == (H, W)
        for oid in ids_t:
            g = gt_t == oid
            px, py = interior_point(g)
            x0, y0, x1, y1 = bbox_xyxy(g)
            mp = sam3_single.segment(model, frame, h, w, [(px, py)], [1])
            mb = sam3_single.segment(model, frame, h, w, [(x0, y0), (x1 - 1, y1 - 1)], [2, 3])
            A[oid] = {"point_iou": r4(iou_np(mp, g)), "box_iou": r4(iou_np(mb, g)),
                      "point_area": int(mp.sum()), "box_area": int(mb.sum())}
        oid0 = ids_t[0]
        g0 = gt_t == oid0
        _, ms_seg = timer(lambda: sam3_single.segment(model, frame, h, w, [interior_point(g0)], [1]))
        A["_cost_ms_per_call(one object, includes backbone+detector pass)"] = ms_seg
        R["A_tracker_sam3_single"] = A

        # ============================================================ B: shared backbone + interactive head
        proc = Sam3Processor(det, resolution=1008, device="cuda", confidence_threshold=0.5)
        pil_t, pil_r = Image.fromarray(img_t), Image.fromarray(img_r)
        st_t, ms_bb = timer(lambda: proc.set_image(pil_t))
        st_r = proc.set_image(pil_r)
        R["cost_backbone_ms(Sam3Processor.set_image, ViT 1008 + both necks)"] = ms_bb

        ip = SAM3InteractiveImagePredictor(trk)

        def ip_features(state, hw):
            sam2 = state["backbone_out"]["sam2_backbone_out"]
            fpn = list(sam2["backbone_fpn"])
            fpn[0] = trk.sam_mask_decoder.conv_s0(fpn[0])
            fpn[1] = trk.sam_mask_decoder.conv_s1(fpn[1])
            bo = dict(sam2)
            bo["backbone_fpn"] = fpn
            _, vf, _, _ = trk._prepare_backbone_features(bo)
            vf[-1] = vf[-1] + trk.no_mem_embed
            feats = [f.permute(1, 2, 0).reshape(1, -1, *s) for f, s in zip(vf[::-1], ip._bb_feat_sizes[::-1])][::-1]
            return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}, [hw]

        feats_t, hw_t = ip_features(st_t, (H, W))

        def ip_set(feats, hw):
            ip._features, ip._orig_hw, ip._is_image_set, ip._is_batch = feats, hw, True, False

        ip_set(feats_t, hw_t)

        def ip_predict(**kw):
            m, s, low = ip.predict(**kw)
            k = int(np.argmax(s))
            return m[k] > 0 if m.dtype != bool else m[k], float(s[k]), low[k]

        B = {}
        for oid in ids_t:
            g = gt_t == oid
            px, py = interior_point(g)
            box = np.array(bbox_xyxy(g), np.float32)
            mp, sp, _ = ip_predict(point_coords=np.array([[px, py]], np.float32), point_labels=np.array([1]),
                                   multimask_output=True)
            mb, sb, _ = ip_predict(box=box, multimask_output=False)
            B[oid] = {"point_best_of_3_iou": r4(iou_np(mp, g)), "point_pred_iou": r4(sp),
                      "box_iou": r4(iou_np(mb, g)), "box_pred_iou": r4(sb),
                      "agree_A_point": r4(iou_np(mp, sam3_single.segment(model, frame, h, w, [(px, py)], [1]))),
                      }
        _, ms_ip = timer(lambda: ip.predict(point_coords=np.array([[W / 2, H / 2]], np.float32),
                                            point_labels=np.array([1]), multimask_output=True))
        B["_cost_ms_per_prompt(head only, features cached)"] = ms_ip
        R["B_interactive_predictor_oracle_tgt_gt"] = B

        # ============================================================ E: dense mask prompts from projection
        cache = pi3_frame_cache(a.data, a.scene, a.t)
        common_ids = [i for i in ids_t if i in ids_r]
        ref_masks = torch.from_numpy(np.stack([gt_r == i for i in common_ids])).cuda()
        pm, vis = project_reference_masks(ref_masks, *reproj_inputs(cache, a.tgt, ref_cam, (H, W), "cuda"))
        pm = pm.cpu().numpy()
        vis_np = vis.cpu().numpy()
        E = {"cache": os.path.relpath(cache, STAGE1), "vis_frac_of_image": r4(vis_np.mean())}
        S = trk.input_mask_size

        def mask_inputs(m, how):
            mt = torch.from_numpy(m.astype(np.float32))[None, None].cuda()
            if how == "logit10":       # binary -> +-10 logits at the prompt encoder's 288x288 (as _use_mask_as_output)
                lo = F.interpolate(mt, size=tuple(trk.sam_prompt_encoder.mask_input_size), mode="bilinear",
                                   align_corners=False, antialias=True)
                return (lo * 20.0 - 10.0)[0].float().cpu().numpy()
            # "downsample": the tracker's own learned conv (1152 -> 288), used when a mask enters as input
            hi = F.interpolate(mt, size=(S, S), mode="bilinear", align_corners=False, antialias=True)
            return trk.mask_downsample(hi)[0].float().cpu().numpy()

        for j, oid in enumerate(common_ids):
            g = gt_t == oid
            p = pm[j]
            rec = {"proj_area": int(p.sum()), "gt_area": int(g.sum()), "proj_iou": r4(iou_np(p, g)),
                   "proj_recall": r4(np.logical_and(p, g).sum() / max(1, g.sum())),
                   "gt_frac_visible_from_ref(vis)": r4(np.logical_and(vis_np, g).sum() / max(1, g.sum()))}
            if p.sum() >= 20:
                px, py = interior_point(p)
                box = np.array(bbox_xyxy(p), np.float32)
                for how in ("logit10", "downsample"):
                    mi = mask_inputs(p, how)
                    m1, s1, _ = ip_predict(mask_input=mi, multimask_output=False)
                    m3, s3, _ = ip_predict(mask_input=mi, multimask_output=True)
                    mbx, sbx, _ = ip_predict(mask_input=mi, box=box, multimask_output=False)
                    mpt, spt, _ = ip_predict(mask_input=mi, point_coords=np.array([[px, py]], np.float32),
                                             point_labels=np.array([1]), multimask_output=False)
                    rec[f"mask[{how}]_iou"] = r4(iou_np(m1, g))
                    rec[f"mask[{how}]_pred_iou"] = r4(s1)
                    rec[f"mask[{how}]_multi_best_iou"] = r4(iou_np(m3, g))
                    rec[f"mask[{how}]+box_iou"] = r4(iou_np(mbx, g))
                    rec[f"mask[{how}]+point_iou"] = r4(iou_np(mpt, g))
                mb, _, _ = ip_predict(box=box, multimask_output=False)
                mpo, _, _ = ip_predict(point_coords=np.array([[px, py]], np.float32), point_labels=np.array([1]),
                                       multimask_output=True)
                rec["proj_box_only_iou"] = r4(iou_np(mb, g))
                rec["proj_prec_rec"] = prec_rec(p, g)
                rec["proj_box_only_prec_rec"] = prec_rec(mb, g)
                mi = mask_inputs(p, "downsample")
                mbx, _, _ = ip_predict(mask_input=mi, box=box, multimask_output=False)
                rec["mask[downsample]+box_prec_rec"] = prec_rec(mbx, g)
                pts3 = interior_points(p, 3)
                mp3, _, _ = ip_predict(box=box, point_coords=np.array(pts3, np.float32),
                                       point_labels=np.ones(len(pts3), np.int64), multimask_output=False)
                rec["proj_box+3pts_iou"] = r4(iou_np(mp3, g))
                mmp3, _, _ = ip_predict(mask_input=mi, box=box, point_coords=np.array(pts3, np.float32),
                                        point_labels=np.ones(len(pts3), np.int64), multimask_output=False)
                rec["mask[downsample]+box+3pts_iou"] = r4(iou_np(mmp3, g))
                rec["mask[downsample]+box+3pts_prec_rec"] = prec_rec(mmp3, g)
                rec["proj_point_only_iou"] = r4(iou_np(mpo, g))
                # tracker dense-mask path (frame 0 of a 1-frame session): pass-through or refinement?
                try:
                    stv = model.init_state([frame], h, w)
                    _, out = model.add_tracker_new_mask(inference_state=stv, frame_idx=0, obj_id=1,
                                                        mask=torch.from_numpy(p))
                    bm = out["out_binary_masks"]
                    bm = bm.cpu().numpy() if hasattr(bm, "cpu") else np.asarray(bm)
                    mo = np.asarray(bm[0]).squeeze().astype(bool)
                    rec["tracker_add_new_mask_iou_vs_input"] = r4(iou_np(mo, p))
                    rec["tracker_add_new_mask_iou"] = r4(iou_np(mo, g))
                    del stv
                except Exception as e:  # report, do not hide
                    rec["tracker_add_new_mask_error"] = repr(e)[:300]
            E[oid] = rec
        if common_ids:
            mi = mask_inputs(pm[0], "downsample")
            _, ms_mask = timer(lambda: ip.predict(mask_input=mi, multimask_output=False))
            E["_cost_ms_mask_prompt(head only)"] = ms_mask
        R["E_mask_prompt_from_projection"] = E

        # ============================================================ C: detector (concept prompts)
        def post(out, hw, thr=0.5):
            probs = out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)
            probs = probs.squeeze(-1)[0]
            keep = probs > thr
            masks = interpolate(out["pred_masks"][0][keep].unsqueeze(1).float(), hw, mode="bilinear",
                                align_corners=False).sigmoid()[:, 0] > 0.5
            boxes = box_ops.box_cxcywh_to_xyxy(out["pred_boxes"][0][keep].float())
            return masks, probs[keep].float(), boxes, probs

        def summarise(masks, scores, oid):
            n = int(masks.shape[0])
            if n == 0:
                return {"n_det": 0}
            ious = [iou_t(masks[k], oid) for k in range(n)]
            k = int(np.argmax(ious))
            return {"n_det": n, "scores_top5": [r4(s) for s in sorted(scores.tolist(), reverse=True)[:5]],
                    "best_inst_iou": r4(ious[k]), "best_inst_score": r4(scores[k].item()),
                    "top_score_inst_iou": r4(ious[int(torch.argmax(scores))]),
                    "union_iou": r4(iou_t(masks.any(0), oid))}

        def grounding(state, prompt, prompt_mask):
            """Sam3Image.forward_grounding after _encode_prompt, with a caller-built prompt sequence."""
            find = proc.find_stage
            bo, enc, _ = det._run_encoder(state["backbone_out"], find, prompt, prompt_mask)
            out = {"encoder_hidden_states": enc["encoder_hidden_states"]}
            out, hs = det._run_decoder(memory=out["encoder_hidden_states"], pos_embed=enc["pos_embed"],
                                       src_mask=enc["padding_mask"], out=out, prompt=prompt,
                                       prompt_mask=prompt_mask, encoder_out=enc)
            det._run_segmentation_heads(out=out, backbone_out=bo, img_ids=find.img_ids,
                                        vis_feat_sizes=enc["vis_feat_sizes"],
                                        encoder_hidden_states=out["encoder_hidden_states"], prompt=prompt,
                                        prompt_mask=prompt_mask, hs=hs)
            return out

        def encode_box_tokens(state, box_cxcywh):
            """Geometry-encoder tokens of one positive box on `state`'s image (box pooled from that image)."""
            find = proc.find_stage
            _, img_feats, img_pos, sizes = det._get_img_feats(state["backbone_out"], find.img_ids)
            pr = Prompt(box_embeddings=torch.tensor(box_cxcywh, device="cuda", dtype=torch.float32).view(1, 1, 4),
                        box_mask=torch.zeros(1, 1, dtype=torch.bool, device="cuda"),
                        box_labels=torch.ones(1, 1, dtype=torch.long, device="cuda"))
            return det.geometry_encoder(geo_prompt=pr, img_feats=img_feats, img_sizes=sizes, img_pos_embeds=img_pos)

        def text_tokens(state, text):
            t_out = det.backbone.forward_text([text], device="cuda")
            state["backbone_out"].update(t_out)
            return t_out["language_features"][:, [0]], t_out["language_mask"][[0]]

        def cxcywh_norm(xyxy, hw):
            x0, y0, x1, y1 = xyxy
            hh, ww = hw
            return [(x0 + x1) / 2 / ww, (y0 + y1) / 2 / hh, (x1 - x0) / ww, (y1 - y0) / hh]

        C = {"text": {}, "exemplar_same_image_oracle": {}, "exemplar_cross_image_from_ref": {}}
        for oid in ids_t:
            lab = labels.get(oid, {}).get("label")
            if not lab:
                continue
            txt = lab.replace("_", " ")
            proc.reset_all_prompts(st_t)
            st = proc.set_text_prompt(txt, st_t)
            C["text"][oid] = {"prompt": txt, **summarise(st["masks"][:, 0], st["scores"], oid)}
        proc.reset_all_prompts(st_t)
        _, ms_txt = timer(lambda: proc.set_text_prompt("dog", st_t))
        C["_cost_ms_per_text_prompt(text enc + grounding, backbone cached)"] = ms_txt
        proc.reset_all_prompts(st_t)
        st = proc.set_text_prompt("visual", st_t)
        C["text_visual_placeholder_no_box_n_det"] = int(st["masks"].shape[0])
        for generic in ("object", "thing"):
            proc.reset_all_prompts(st_t)
            st = proc.set_text_prompt(generic, st_t)
            C[f"text_generic[{generic}]"] = {"n_det": int(st["masks"].shape[0]),
                                             "best_iou_per_obj": {o: r4(max([iou_t(st["masks"][k, 0], o) for k in range(st["masks"].shape[0])] or [0.0])) for o in ids_t}}

        # same-image exemplar via the shipped API (box from the target's own GT: an oracle, shows the mode)
        for oid in ids_t:
            g = gt_t == oid
            proc.reset_all_prompts(st_t)
            st = proc.add_geometric_prompt(cxcywh_norm(bbox_xyxy(g), (H, W)), True, st_t)
            C["exemplar_same_image_oracle"][oid] = summarise(st["masks"][:, 0], st["scores"], oid)

        # sanity: the transplant path reproduces add_geometric_prompt when the exemplar image IS the target
        g = gt_t == ids_t[0]
        bx = cxcywh_norm(bbox_xyxy(g), (H, W))
        proc.reset_all_prompts(st_t)
        st = proc.add_geometric_prompt(bx, True, st_t)
        ref_probs = proc.model.forward_grounding(backbone_out=st_t["backbone_out"], find_input=proc.find_stage,
                                                 geometric_prompt=st_t["geometric_prompt"], find_target=None)
        txt_f, txt_m = text_tokens(st_t, "visual")
        geo, geo_m = encode_box_tokens(st_t, bx)
        out = grounding(st_t, torch.cat([txt_f, geo], 0), torch.cat([txt_m, geo_m], 1))
        d = (out["pred_logits"].float() - ref_probs["pred_logits"].float()).abs().max().item()
        C["_sanity_transplant_same_image_max_abs_logit_diff"] = r4(d)

        # cross-image exemplar: box from the REFERENCE camera's GT (same dev frame), encoded on the
        # reference image, its token sequence placed in the target image's prompt after the "visual" text
        for oid in ids_t:
            if oid not in ids_r:
                continue
            gr = gt_r == oid
            bx_r = cxcywh_norm(bbox_xyxy(gr), gt_r.shape)
            geo, geo_m = encode_box_tokens(st_r, bx_r)
            txt_f, txt_m = text_tokens(st_t, "visual")
            out = grounding(st_t, torch.cat([txt_f, geo], 0), torch.cat([txt_m, geo_m], 1))
            masks, scores, _, allp = post(out, (H, W))
            rec = summarise(masks, scores, oid)
            rec["max_prob_any_query"] = r4(allp.max().item())
            # label-text + cross exemplar together
            txt_f2, txt_m2 = text_tokens(st_t, labels[oid]["label"].replace("_", " "))
            out2 = grounding(st_t, torch.cat([txt_f2, geo], 0), torch.cat([txt_m2, geo_m], 1))
            m2, s2, _, _ = post(out2, (H, W))
            rec["with_label_text"] = summarise(m2, s2, oid)
            C["exemplar_cross_image_from_ref"][oid] = rec
        geo, geo_m = encode_box_tokens(st_r, cxcywh_norm(bbox_xyxy(gt_r == ids_r[0]), gt_r.shape))
        txt_f, txt_m = text_tokens(st_t, "visual")
        _, ms_x = timer(lambda: grounding(st_t, torch.cat([txt_f, geo], 0), torch.cat([txt_m, geo_m], 1)))
        C["_cost_ms_per_exemplar_grounding(backbones cached)"] = ms_x
        R["C_detector"] = C

        # ============================================================ D: everything mode (emulated)
        def amg(feats, hw, n_side, chunk=256, iou_th=0.8, stab_th=0.95, nms_th=0.7):
            xs = (np.arange(n_side) + 0.5) / n_side
            gx, gy = np.meshgrid(xs, xs)
            pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], -1) * trk.image_size, dtype=torch.float32,
                               device="cuda")[:, None, :]
            lab = torch.ones(pts.shape[0], 1, dtype=torch.int32, device="cuda")
            lows, ious, objs = [], [], []
            pe = trk.sam_prompt_encoder.get_dense_pe()
            for i in range(0, pts.shape[0], chunk):
                sp, de = trk.sam_prompt_encoder(points=(pts[i:i + chunk], lab[i:i + chunk]), boxes=None, masks=None)
                lo, io, _, ob = trk.sam_mask_decoder(image_embeddings=feats["image_embed"], image_pe=pe,
                                                     sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                                                     multimask_output=True, repeat_image=True,
                                                     high_res_features=feats["high_res_feats"])
                lows.append(lo.float().flatten(0, 1))
                ious.append(io.float().flatten())
                objs.append(ob.float().repeat_interleave(lo.shape[1], 0).flatten())
            lo, io, ob = torch.cat(lows), torch.cat(ious), torch.cat(objs)
            n_cand = int(lo.shape[0])
            inter = (lo > 1.0).flatten(1).sum(1).float()
            union = (lo > -1.0).flatten(1).sum(1).float()
            stab = inter / union.clamp(min=1)
            keep = (io >= iou_th) & (stab >= stab_th) & (ob > 0) & (inter > 0)
            lo, io = lo[keep], io[keep]
            m = lo > 0
            rows, cols = m.any(2), m.any(1)
            ar = torch.arange(m.shape[-1], device="cuda")
            x0 = torch.where(cols, ar, 10 ** 6).min(1).values
            x1 = torch.where(cols, ar, -1).max(1).values
            y0 = torch.where(rows, ar, 10 ** 6).min(1).values
            y1 = torch.where(rows, ar, -1).max(1).values
            boxes = torch.stack([x0, y0, x1, y1], 1).float()
            k = torchvision.ops.nms(boxes, io, nms_th)
            return lo[k], io[k], n_cand, int(keep.sum())

        D = {}
        for n_side in (16, a.grid):
            (lo, io, n_cand, n_filt), ms_amg = timer(lambda: amg(feats_t, hw_t, n_side), reps=2)
            best = {o: 0.0 for o in ids_t}
            for i in range(0, lo.shape[0], 16):
                up = F.interpolate(lo[i:i + 16, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0] > 0
                for o in ids_t:
                    for k in range(up.shape[0]):
                        best[o] = max(best[o], iou_t(up[k], o))
            D[f"grid{n_side}"] = {"n_points": n_side * n_side, "n_candidates(3/pt)": n_cand,
                                  "n_after_iou0.8_stab0.95_obj": n_filt, "n_after_nms0.7": int(lo.shape[0]),
                                  "cost_ms(head only, features cached)": ms_amg,
                                  "best_iou_per_obj": {o: r4(v) for o, v in best.items()}}
        R["D_everything_emulated"] = D
        R["peak_gpu_mem_GB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)

        # ============================================================ visual check (1/8 res)
        try:
            rows = []
            sc = 8
            small = cv2.resize(img_t, (W // sc, H // sc))
            for j, oid in enumerate(common_ids):
                g = gt_t == oid
                tiles = [g, pm[j]]
                if pm[j].sum() >= 20:
                    mi = mask_inputs(pm[j], "downsample")
                    tiles.append(ip_predict(mask_input=mi, box=np.array(bbox_xyxy(pm[j]), np.float32),
                                            multimask_output=False)[0])
                else:
                    tiles.append(np.zeros_like(g))
                proc.reset_all_prompts(st_t)
                st = proc.set_text_prompt(labels[oid]["label"].replace("_", " "), st_t)
                tiles.append(st["masks"][:, 0].any(0).cpu().numpy() if st["masks"].shape[0] else np.zeros_like(g))
                row = []
                for tm in tiles:
                    ov = small.copy()
                    ms = cv2.resize(tm.astype(np.uint8), (W // sc, H // sc), interpolation=cv2.INTER_NEAREST) > 0
                    ov[ms] = (0.45 * ov[ms] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
                    row.append(ov)
                rows.append(np.concatenate(row, 1))
            cv2.imwrite(os.path.join(a.out, f"probe_modes_viz_{a.scene}_t{a.t:02d}_{a.tgt}.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
            R["viz"] = f"probe_modes_viz_{a.scene}_t{a.t:02d}_{a.tgt}.jpg (rows: objects in both views; cols: GT | projected ref GT | mask+box prompt | label-text union)"
        except Exception as e:
            R["viz_error"] = repr(e)[:300]

    path = os.path.join(a.out, f"probe_modes_{a.scene}_t{a.t:02d}_{a.tgt}.json")
    with open(path, "w") as f:
        json.dump(R, f, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    print("wrote", path)


if __name__ == "__main__":
    main()
