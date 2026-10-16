"""Mechanism probe for an online S2-R6 (reference-camera memory snap).  NOT a GT-scored run.

Runs SCSam3/runMVSeg.py main() unchanged, with the c_ini session's generator wrapped so that,
after every frame t > start that c_ini yields (and BEFORE the next session is advanced), we
  1. evaluate the S2-R6 rule online on the tracker's own mask_t (ref_snap.decide, same constants),
  2. inspect the stored memory entry non_cond_frame_outputs[t] of each c_ini object tracker state,
  3. re-encode the stored decoded mask (sanity: must reproduce the stored maskmem_features),
  4. encode mask_0 (frame-0 PNG == prompt) on frame t's features in two ways and an mask-as-output pointer,
  5. PROBE_MODE=inject_mem / inject_memptr: overwrite the stored entry in place for snapped objects.
Usage (inside the container, cwd SCSam3/):
  PROBE_MODE=observe PROBE_JSON=... PROBE_OUTDIR=... python probe_mem.py <runMVSeg args>
"""
import importlib.util
import json
import os
import sys
import time

HERE_SC = os.getcwd()                                   # .../SCSam3
spec = importlib.util.spec_from_file_location("runMVSeg", os.path.join(HERE_SC, "runMVSeg.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)

MODE = os.environ.get("PROBE_MODE", "observe")
assert MODE in ("observe", "inject_mem", "inject_memptr", "inject_cond"), MODE
OUT_JSON = os.environ["PROBE_JSON"]
OUT_DIR = os.environ["PROBE_OUTDIR"]                    # the run's output folder (frame-0 PNGs)

# camera names / video root from the same config the run uses
argv = sys.argv[1:]
dataset = argv[0]
def _arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default
cfg = rm.load_config(_arg("--config", os.path.join(HERE_SC, "demo", "MVSeg.json")), dataset)
data_root = os.path.abspath(_arg("--data-root", os.path.join(HERE_SC, "..", "Data", "MVSeg")))
cam_names = [rm.cam_name(x, cfg["prefix"], cfg["prefix1"]) for x in cfg["perms"]]
START = cfg["start_frame"]
video_root = os.path.join(data_root, cfg["folder"], "Video")

REC = {"mode": MODE, "dataset": dataset, "frames": [], "first_entry": None, "timing": {},
       "ref_view": None, "ref_session_j": None, "neighbour_sessions": None}

_orig_build = rm.build_runner


def build_runner(algo):
    MVSegVideo, torch = _orig_build(algo)
    import cv2
    import numpy as np
    import torch.nn.functional as F
    rs = rm.ref_snap_module()

    class Probe:
        def __init__(self, sc, j, view, gen):
            self.sc, self.j, self.view, self.gen = sc, j, view, gen
            self.cam = cam_names[view]
            self.m0 = None
            self.i0 = None
            self.prev_feat = {}

        def __iter__(self):
            return self

        def img(self, t, shape):
            im = cv2.imread(os.path.join(video_root, self.cam, "%06d.jpg" % int(t)), cv2.IMREAD_GRAYSCALE)
            if im.shape != shape:
                im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            return im.astype(np.int16)

        def __next__(self):
            resp = next(self.gen)
            t = resp["frame_index"]
            if t == START:
                return resp
            t0 = time.time()
            sc = self.sc
            if self.m0 is None:                      # frame-0 PNGs, written by main in round 0
                d = os.path.join(OUT_DIR, self.cam, str(START))
                self.m0 = {int(o.split(".")[0]): cv2.imread(os.path.join(d, o), cv2.IMREAD_GRAYSCALE) > 127
                           for o in os.listdir(d)}
                shape = next(iter(self.m0.values())).shape
                self.i0 = self.img(START, shape)
                # frame-0 PNG == masks_spatial (seed) for c_ini?
                ms = sc.masks_spatial.get(self.view, {})
                REC["frame0_png_equals_masks_spatial"] = {
                    str(o): bool(np.array_equal(self.m0[o], np.squeeze(np.asarray(ms[o])).astype(bool)))
                    for o in self.m0 if o in ms}
            shape = next(iter(self.m0.values())).shape
            it = self.img(t, shape)
            out = resp["outputs"]
            ids = out["out_obj_ids"].tolist()
            masks = {o: np.asarray(out["out_binary_masks"][i]).astype(bool) for i, o in enumerate(ids)}
            t1 = time.time()
            dec = {}
            for o in sorted(self.m0):
                mt = masks.get(o, np.zeros(shape, bool))
                s, iou, diff = rs.decide(self.m0[o], mt, self.i0, it, np, cv2)
                dec[o] = dict(snap=bool(s), iou=None if iou is None else round(float(iou), 5),
                              diff=None if diff is None else round(float(diff), 4), present=o in masks)
            t2 = time.time()
            # memory entries of the c_ini tracker states
            pred = sc.predictor
            model = pred.model
            tracker = model.tracker
            state = pred._ALL_INFERENCE_STATES[sc.session_ids[self.j]]["state"]
            fr = {"t": int(t), "objs": {}, "t_decide_s": round(t2 - t1, 4), "t_read_s": round(t1 - t0, 4)}
            enc_times = []
            with torch.inference_mode():
                for o in sorted(self.m0):
                    tss = model._get_tracker_inference_states_by_obj_ids(state, [o])
                    if not tss:
                        fr["objs"][str(o)] = dict(dec[o], tracker_state=False)
                        continue
                    ts = tss[0]
                    od = ts["output_dict"]
                    e = od["non_cond_frame_outputs"].get(t)
                    rec = dict(dec[o])
                    if e is None:
                        rec["entry"] = None
                        fr["objs"][str(o)] = rec
                        continue
                    if REC["first_entry"] is None:
                        REC["first_entry"] = {
                            k: (dict(shape=list(v.shape), dtype=str(v.dtype), device=str(v.device))
                                if hasattr(v, "shape") else
                                [dict(shape=list(x.shape), dtype=str(x.dtype), device=str(x.device)) for x in v]
                                if isinstance(v, list) else repr(v))
                            for k, v in e.items()}
                        REC["first_entry"]["_cond_frames"] = sorted(int(k) for k in od["cond_frame_outputs"])
                        REC["first_entry"]["_non_cond_frames"] = sorted(int(k) for k in od["non_cond_frame_outputs"])
                        REC["first_entry"]["_storage_device"] = str(ts["storage_device"])
                        REC["first_entry"]["_cached_feature_keys"] = sorted(
                            str(k) for k in ts["cached_features"].keys())
                        REC["first_entry"]["_per_obj_slice_shares_storage"] = bool(
                            ts["output_dict_per_obj"][0]["non_cond_frame_outputs"][t]["maskmem_features"].data_ptr()
                            == e["maskmem_features"].data_ptr())
                        REC["first_entry"]["_tracker"] = dict(
                            use_memory_selection=bool(tracker.use_memory_selection),
                            mf_threshold=float(tracker.mf_threshold),
                            num_maskmem=int(tracker.num_maskmem),
                            max_cond_frames_in_attn=int(tracker.max_cond_frames_in_attn),
                            max_obj_ptrs_in_encoder=int(tracker.max_obj_ptrs_in_encoder),
                            keep_first_cond_frame=bool(tracker.keep_first_cond_frame),
                            clear_non_cond_mem_around_input=bool(tracker.clear_non_cond_mem_around_input),
                            add_all_frames_to_correct_as_cond=bool(tracker.add_all_frames_to_correct_as_cond),
                            non_overlap_masks_for_mem_enc=bool(tracker.non_overlap_masks_for_mem_enc),
                            sigmoid_scale_for_mem_enc=float(tracker.sigmoid_scale_for_mem_enc),
                            sigmoid_bias_for_mem_enc=float(tracker.sigmoid_bias_for_mem_enc),
                            image_size=int(tracker.image_size), low_res=int(tracker.low_res_mask_size),
                            input_mask_size=int(tracker.input_mask_size),
                            trim_past_non_cond_mem_for_eval=bool(tracker.trim_past_non_cond_mem_for_eval),
                            offload_output_to_cpu_for_eval=bool(tracker.offload_output_to_cpu_for_eval),
                            memory_temporal_stride_for_eval=int(tracker.memory_temporal_stride_for_eval),
                            fill_hole_area_model=int(model.fill_hole_area))
                    import hashlib
                    rec["pm_digest"] = hashlib.sha1(e["pred_masks"].float().cpu().numpy().tobytes()).hexdigest()[:16]
                    rec["eff_iou_score"] = float(e["eff_iou_score"].float().mean()) if "eff_iou_score" in e else None
                    rec["obj_score"] = float(e["object_score_logits"].float().mean())
                    stored = e["maskmem_features"]
                    te = time.time()
                    image, _, feats, _, sizes = tracker._get_image_feature(ts, t, 1)
                    dev = feats[-1].device
                    # (a) sanity: re-encode the stored decoded mask (low-res logits -> 1008, sigmoid path)
                    hr = F.interpolate(e["pred_masks"].to(dev).float(), size=(tracker.image_size, tracker.image_size),
                                       mode="bilinear", align_corners=False)
                    mf_a, _ = tracker._encode_new_memory(image=image, current_vision_feats=feats, feat_sizes=sizes,
                                                         pred_masks_high_res=hr,
                                                         object_score_logits=e["object_score_logits"].to(dev),
                                                         is_mask_from_pts=False)
                    mf_a = mf_a.to(torch.bfloat16).to(stored.device)
                    torch.cuda.synchronize()
                    enc_times.append(time.time() - te)
                    rec["reencode_equal"] = bool(torch.equal(mf_a, stored))
                    rec["reencode_maxabs"] = float((mf_a.float() - stored.float()).abs().max())
                    sn = float(stored.float().norm())
                    # natural frame-to-frame scale of the stored memory
                    if o in self.prev_feat:
                        rec["rel_vs_prev_frame"] = float((stored.float() - self.prev_feat[o]).norm() / sn)
                    self.prev_feat[o] = stored.float().clone()
                    if ("p", o) in self.prev_feat:
                        rec["cos_ptr_vs_prev"] = float(F.cosine_similarity(
                            e["obj_ptr"].float().flatten(), self.prev_feat[("p", o)].flatten(), dim=0))
                    self.prev_feat[("p", o)] = e["obj_ptr"].float().clone()
                    # (b) mask_0 as +-10 logits at 1008 (antialiased resize, >0.5), binarised path
                    m0t = torch.from_numpy(self.m0[o].astype(np.float32))[None, None].to(dev)
                    m1008 = F.interpolate(m0t, size=(tracker.image_size, tracker.image_size), mode="bilinear",
                                          align_corners=False, antialias=True) > 0.5
                    hr_b = m1008.float() * 20.0 - 10.0
                    pos = torch.full_like(e["object_score_logits"].to(dev), 10.0)
                    mf_b, _ = tracker._encode_new_memory(image=image, current_vision_feats=feats, feat_sizes=sizes,
                                                         pred_masks_high_res=hr_b, object_score_logits=pos,
                                                         is_mask_from_pts=True)
                    mf_b = mf_b.to(torch.bfloat16)
                    # (c) mask_0 through the cond-frame route: video res -> 288 (antialias) -> 1008 -> >0
                    m288 = F.interpolate(m0t, size=(tracker.low_res_mask_size, tracker.low_res_mask_size),
                                         mode="bilinear", align_corners=False, antialias=True) * 20.0 - 10.0
                    hr_c = F.interpolate(m288, size=(tracker.image_size, tracker.image_size), mode="bilinear",
                                         align_corners=False)
                    mf_c, _ = tracker._encode_new_memory(image=image, current_vision_feats=feats, feat_sizes=sizes,
                                                         pred_masks_high_res=hr_c, object_score_logits=pos,
                                                         is_mask_from_pts=True)
                    mf_c = mf_c.to(torch.bfloat16)
                    rec["rel_m0bin_vs_stored"] = float((mf_b.float().cpu() - stored.float().cpu()).norm() / sn)
                    rec["rel_m0rt_vs_stored"] = float((mf_c.float().cpu() - stored.float().cpu()).norm() / sn)
                    rec["rel_m0bin_vs_m0rt"] = float((mf_b.float() - mf_c.float()).norm() / mf_c.float().norm())
                    # cond memory of frame 0 (the GT prompt, frame-0 image)
                    c0 = od["cond_frame_outputs"].get(START)
                    if c0 is not None and c0.get("maskmem_features") is not None:
                        cf = c0["maskmem_features"].float().cpu()
                        rec["rel_cond0_vs_stored"] = float((cf - stored.float().cpu()).norm() / sn)
                        rec["rel_cond0_vs_m0rt"] = float((cf - mf_c.float().cpu()).norm() / cf.norm())
                    # (d) pointer from mask_0 as mask-as-output (cond-frame pointer recipe)
                    pix = feats[-1].permute(1, 2, 0).reshape(-1, tracker.hidden_dim, *sizes[-1])
                    hrf = [x.permute(1, 2, 0).reshape(x.size(1), x.size(2), *s) for x, s in zip(feats[:-1], sizes[:-1])]
                    m_in = F.interpolate(m0t, size=(tracker.input_mask_size, tracker.input_mask_size), mode="bilinear",
                                         align_corners=False, antialias=True)
                    so = tracker._use_mask_as_output(pix, hrf, m_in)
                    ptr_m0 = so[5]
                    sp = e["obj_ptr"].float().to(dev)
                    rec["cos_ptr_m0_vs_stored"] = float(F.cosine_similarity(ptr_m0.float().flatten(), sp.flatten(), dim=0))
                    if c0 is not None:
                        rec["cos_ptr_cond0_vs_stored"] = float(F.cosine_similarity(
                            c0["obj_ptr"].float().to(dev).flatten(), sp.flatten(), dim=0))
                    # (e) injection
                    if MODE == "inject_cond" and dec[o]["snap"]:
                        # SAM-native correction: frame t becomes a conditioning frame of this object
                        tracker.add_new_mask(ts, t, o, torch.from_numpy(self.m0[o].astype(np.float32)))
                        tracker.propagate_in_video_preflight(ts, run_mem_encoder=True)
                        rec["injected"] = True
                        rec["n_cond_after"] = len(od["cond_frame_outputs"])
                        rec["nc_t_left"] = t in od["non_cond_frame_outputs"]
                    elif MODE != "observe" and dec[o]["snap"]:
                        stored.copy_(mf_c.to(stored.device))      # in place: per-object slices share it
                        if MODE == "inject_memptr":
                            e["obj_ptr"].copy_(ptr_m0.to(e["obj_ptr"].dtype))
                        rec["injected"] = True
                    fr["objs"][str(o)] = rec
            fr["t_mem_s"] = round(time.time() - t2, 4)
            fr["enc_ms_mean"] = round(1000 * sum(enc_times) / max(1, len(enc_times)), 3)
            REC["frames"].append(fr)
            return resp

    class ProbeVideo(MVSegVideo):
        def AddReferenceMask(self, view_index, mask, obj_id):
            REC["ref_view"] = int(view_index)
            return super().AddReferenceMask(view_index, mask, obj_id)

        def TrackForward(self, start_frame, num_frame):
            super().TrackForward(start_frame, num_frame)
            for j, view in enumerate(self.track_views):
                if view == REC["ref_view"]:
                    REC["ref_session_j"] = j
                    REC["neighbour_sessions"] = [k for k in (j - 1, j + 1) if 0 <= k < len(self.track_views)]
                    self.tracking_result[j] = Probe(self, j, view, self.tracking_result[j])

    return ProbeVideo, torch


rm.build_runner = build_runner
sys.argv = [os.path.join(HERE_SC, "runMVSeg.py")] + argv
t_all = time.time()
try:
    rm.main()
finally:
    REC["timing"]["total_s"] = round(time.time() - t_all, 1)
    with open(OUT_JSON, "w") as fh:
        json.dump(REC, fh, indent=1)
    print(f"probe -> {OUT_JSON}", flush=True)
