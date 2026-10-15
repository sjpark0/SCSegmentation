#!/usr/bin/env python3
"""S1-E13 candidate C (sam3-prompt-extend), GPU part: SAM 3 single-image extent proposals on the DEV set.

DEV ONLY (geomem/runs/_e13_dev/index.json frames; asserts t != ref_frame).  Design: REGISTRATION.txt next to this file.
Per dev view (scene, t, cam):  P_o, vis = project_reference_masks(c_ini GT at t restricted to the basic set) with the
frame's Pi3 cache; target features once (Sam3Processor.set_image -> sam2 neck -> tracker conv_s0/s1 + no_mem_embed, as
Sam3Image.predict_inst); per object with |P_o| >= MIN_PROJ_PX the 5 registered prompt variants through
SAM3InteractiveImagePredictor(model.tracker).
Writes <out>/<scene>/<t:06d>/<cam>/{cls.png, proj/<o>.png, <variant>/<o>.png, meta.json, DONE}
    cls.png: 0 vis / 1 outside / 2 occluded / 3 nodepth (the classes of dev_score.py / extent_vis.py)
    flock <lock> stage1/scripts/docker_run.sh scsam3 python geomem/runs/_e13_sam3-prompt-extend/propose.py [--scenes A,B]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, S1)
import cv2  # noqa: E402

from geomem.cache import pi3_frame_cache  # noqa: E402
from methods import sam3_single  # noqa: E402
from methods.geomem.run import project_reference_masks, reproj_inputs  # noqa: E402
from mvseg2d.io import load_labelmap, load_meta  # noqa: E402

DATA = os.path.join(S1, "Data")
IDX = os.path.join(S1, "geomem", "runs", "_e13_dev", "index.json")
MIN_PROJ_PX = 20
VARIANTS = ("pts", "pts_whole", "pts_neg", "box", "mask_box")
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def interior_points(m, k=3):
    """probe_modes.interior_points: k interior points by repeated distance-transform argmax with a suppression disk."""
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    pts = []
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(d)), d.shape)
        if d[y, x] <= 0:
            break
        pts.append((float(x), float(y)))
        cv2.circle(d, (int(x), int(y)), int(max(2 * d[y, x], 40)), 0, -1)
    return pts


def bbox_xyxy(m):
    ys, xs = np.nonzero(m)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def wr(path, m):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert cv2.imwrite(path, np.asarray(m, np.uint8) * 255 if m.dtype == bool else m, PNG), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--out", default=os.path.join(HERE, "props"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    idx = json.load(open(IDX))
    scenes = sorted(idx["scenes"]) if a.scenes == "all" else a.scenes.split(",")

    import torch
    import torch.nn.functional as F
    from PIL import Image
    t0 = time.time()
    model = sam3_single.load_model()
    from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
    from sam3.model.sam3_image_processor import Sam3Processor
    det, trk = model.detector, model.tracker
    proc = Sam3Processor(det, resolution=1008, device="cuda", confidence_threshold=0.5)
    ip = SAM3InteractiveImagePredictor(trk)
    print(f"model {time.time() - t0:.1f}s", flush=True)

    def ip_features(state, hw):                      # probe_modes.ip_features (= Sam3Image.predict_inst features)
        sam2 = state["backbone_out"]["sam2_backbone_out"]
        fpn = list(sam2["backbone_fpn"])
        fpn[0] = trk.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = trk.sam_mask_decoder.conv_s1(fpn[1])
        bo = dict(sam2)
        bo["backbone_fpn"] = fpn
        _, vf, _, _ = trk._prepare_backbone_features(bo)
        vf[-1] = vf[-1] + trk.no_mem_embed
        feats = [f.permute(1, 2, 0).reshape(1, -1, *s) for f, s in zip(vf[::-1], ip._bb_feat_sizes[::-1])][::-1]
        ip._features, ip._orig_hw, ip._is_image_set, ip._is_batch = \
            {"image_embed": feats[-1], "high_res_feats": feats[:-1]}, [hw], True, False

    S = trk.input_mask_size

    def mask_input(m):                               # probe_modes.mask_inputs(..., "downsample")
        mt = torch.from_numpy(m.astype(np.float32))[None, None].cuda()
        hi = F.interpolate(mt, size=(S, S), mode="bilinear", align_corners=False, antialias=True)
        return trk.mask_downsample(hi)[0].float().cpu().numpy()

    def run(**kw):
        m, s, _ = ip.predict(**kw)
        return m > 0, s

    with torch.inference_mode():
        for scene in scenes:
            Sc = idx["scenes"][scene]
            ref, basic = Sc["ref_cam"], [int(o) for o in Sc["basic"]]
            assert int(load_meta(DATA, scene)["ref_frame"]) == Sc["ref_frame"]
            for t in Sc["frames"]:
                assert t != Sc["ref_frame"], (scene, t)
                fdir = os.path.join(DATA, scene, "frames", f"{t:06d}")
                ref_lab = load_labelmap(os.path.join(fdir, "masks", f"{ref}.png"))
                objs = [o for o in basic if (ref_lab == o).any()]
                cache = pi3_frame_cache(DATA, scene, t)
                assert json.load(open(os.path.join(cache, "MANIFEST.json")))["ref_cam"] == ref
                for cam in Sc["eval_cams"]:
                    vdir = os.path.join(a.out, scene, f"{t:06d}", cam)
                    if os.path.isfile(os.path.join(vdir, "DONE")) and not a.overwrite:
                        print(f"{scene} {t} {cam} skip", flush=True)
                        continue
                    ts = time.time()
                    img = cv2.cvtColor(cv2.imread(os.path.join(fdir, "images", f"{cam}.jpg")), cv2.COLOR_BGR2RGB)
                    H, W = img.shape[:2]
                    assert ref_lab.shape == (H, W)
                    ref_masks = torch.from_numpy(np.stack([ref_lab == o for o in objs])).cuda()
                    pts, valid_full, depth_ref, K_ref = reproj_inputs(cache, cam, ref, (H, W), "cuda")
                    pm, vis = project_reference_masks(ref_masks, pts, valid_full, depth_ref, K_ref)
                    from geomem.warp_torch import reproject_coords
                    _, inside = reproject_coords(pts, K_ref, (H, W), depth_ref=None)
                    inside = inside & valid_full
                    cls = torch.full((H, W), 255, dtype=torch.uint8, device="cuda")
                    cls[vis] = 0
                    cls[valid_full & ~inside] = 1
                    cls[inside & ~vis] = 2
                    cls[~valid_full] = 3
                    cls = cls.cpu().numpy()
                    assert (cls != 255).all()
                    pm = pm.cpu().numpy()
                    del ref_masks, pts, valid_full, depth_ref, K_ref, vis, inside
                    wr(os.path.join(vdir, "cls.png"), cls)
                    st = proc.set_image(Image.fromarray(img))
                    ip_features(st, (H, W))
                    proj_px = {o: int(pm[j].sum()) for j, o in enumerate(objs)}
                    deep = {o: interior_points(pm[j], 1)[0] for j, o in enumerate(objs) if proj_px[o] >= MIN_PROJ_PX}
                    meta = dict(scene=scene, t=t, cam=cam, ref_cam=ref, hw=[H, W], objs=objs, proj_px=proj_px,
                                min_proj_px=MIN_PROJ_PX, prompted=[], props={})
                    for j, o in enumerate(objs):
                        p = pm[j]
                        if p.any():
                            wr(os.path.join(vdir, "proj", f"{o}.png"), p)
                        if proj_px[o] < MIN_PROJ_PX:
                            continue
                        meta["prompted"].append(o)
                        pos = interior_points(p, 3)
                        neg = [deep[q] for q in deep if q != o]
                        box = np.array(bbox_xyxy(p), np.float32)
                        rec = dict(n_pos=len(pos), n_neg=len(neg), box=box.tolist())
                        m3, s3 = run(point_coords=np.array(pos, np.float32), point_labels=np.ones(len(pos), np.int64),
                                     multimask_output=True)
                        areas = m3.reshape(3, -1).sum(1)
                        k_iou, k_big = int(np.argmax(s3)), int(np.argmax(areas))
                        out = {"pts": (m3[k_iou], float(s3[k_iou])), "pts_whole": (m3[k_big], float(s3[k_big]))}
                        rec["pts3"] = dict(pred_iou=[float(x) for x in s3], area=[int(x) for x in areas],
                                           k_iou=k_iou, k_big=k_big)
                        cc = np.array(pos + neg, np.float32)
                        ll = np.array([1] * len(pos) + [0] * len(neg), np.int64)
                        mn, sn = run(point_coords=cc, point_labels=ll, multimask_output=False)
                        out["pts_neg"] = (mn[0], float(sn[0]))
                        mb, sb = run(box=box, multimask_output=False)
                        out["box"] = (mb[0], float(sb[0]))
                        mmb, smb = run(mask_input=mask_input(p), box=box, multimask_output=False)
                        out["mask_box"] = (mmb[0], float(smb[0]))
                        for v in VARIANTS:
                            m, s = out[v]
                            rec[v] = dict(pred_iou=s, area=int(m.sum()))
                            if m.any():
                                wr(os.path.join(vdir, v, f"{o}.png"), m)
                        meta["props"][o] = rec
                    meta["seconds"] = time.time() - ts
                    with open(os.path.join(vdir, "meta.json"), "w") as f:
                        json.dump(meta, f)
                    open(os.path.join(vdir, "DONE"), "w").close()
                    del st
                    torch.cuda.empty_cache()
                    print(f"{scene:20s} t={t:4d} {cam:12s} objs {len(objs):2d} prompted {len(meta['prompted']):2d} "
                          f"{meta['seconds']:5.1f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
