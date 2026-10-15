#!/usr/bin/env python3
"""S1-E14 multimask (scratch, git-ignored): the reference GT of c_ini at frame t reprojected into every DEV-A eval view.

GT-free at test time: the input is c_ini's GT (the method's own prompt) and the frame's Pi3 cache; nothing of the eval
view's GT is read.  Uses run.project_reference_masks / run.reproj_inputs unchanged (the GeoProj / reproj-gate code):
    P [B,h,w] = c_ini GT of object b sampled at the reprojection of each eval-view pixel (occlusion-checked), & vis
    vis [h,w] = valid reprojection (Pi3 conf, in front, inside, not occluded by the reference surface)
Objects = the dump's config.json "objects" order (asserted equal to frame_scene(...).objects).
Check: the round-1 overlap that decode_scene recorded (round1_conf.json, run.reproj_overlap of the round-1 mask) is
recomputed from the _r1 PNG and this P / vis -> must agree (same code, same inputs).
Output: <out>/<scene>/<t:06d>/<cam>.npz  objects int32 [B], hw int32 [2], P packbits uint8 [B, ceil(h*w/8)],
        vis packbits uint8 [ceil(h*w/8)], nP int64 [B] (= |P|, P already inside vis), nvis int64.
    cd stage1 && flock <lock> scripts/docker_run.sh scsam3 python geomem/runs/_e14_multimask/code/proj_ref.py
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, S1)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from geomem import devset  # noqa: E402
from geomem.cache import pi3_frame_cache  # noqa: E402
from methods.geomem import dev_decode as D  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from mvseg2d.io import read_pred_mask  # noqa: E402

DUMP = os.path.join(S1, "geomem", "runs", "_e14_logitdump", "geoset_rep_x_A")
OUT = os.path.join(S1, "geomem", "runs", "_e14_multimask", "proj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--scenes", default="all")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.dump, "index.json")))
    scenes = sorted(idx["scenes"]) if a.scenes == "all" else a.scenes.split(",")
    log = dict(views=0, r1_checked=0, r1_maxdiff=0.0, r1_nan_mismatch=0, r1_nvalid_mismatch=0, seconds=0.0)
    t0 = time.time()
    for s in scenes:
        info = idx["scenes"][s]
        for t in info["frames"]:
            assert t != info["ref_frame"]
            fdir = os.path.join(a.dump, s, f"{t:06d}")
            cfg = json.load(open(os.path.join(fdir, "config.json")))
            r1c = json.load(open(os.path.join(fdir, "round1_conf.json")))
            sc = D.frame_scene(s, t, info["basic"])
            assert [int(o) for o in sc.objects] == [int(o) for o in cfg["objects"]], (s, t)
            cache = pi3_frame_cache(D.DATA, s, t)
            masks = torch.from_numpy(np.stack([sc.ref_labels == o for o in sc.objects])).cuda()
            for cam in info["eval_cams"]:
                with torch.no_grad():
                    inp = R.reproj_inputs(cache, cam, sc.ref_cam, sc.hw, "cuda")
                    P, vis = R.project_reference_masks(masks, *inp)
                    # check vs the recorded round-1 overlap of this eval cam
                    r1 = torch.zeros_like(P)
                    for i, o in enumerate(sc.objects):
                        p = os.path.join(fdir, "_r1", cam, f"{int(o)}.png")
                        if os.path.isfile(p):
                            r1[i] = torch.from_numpy(read_pred_mask(p)).cuda()
                    ov, nv = R.reproj_overlap(r1, *inp, masks)
                ov, nv = ov.cpu().numpy(), nv.cpu().numpy()
                for i, o in enumerate(sc.objects):
                    rec = r1c[cam][str(int(o))]
                    log["r1_checked"] += 1
                    if int(nv[i]) != rec["n_valid"]:
                        log["r1_nvalid_mismatch"] += 1
                    if (rec["overlap"] is None) != bool(np.isnan(ov[i])):
                        log["r1_nan_mismatch"] += 1
                    elif rec["overlap"] is not None:
                        log["r1_maxdiff"] = max(log["r1_maxdiff"], abs(float(ov[i]) - rec["overlap"]))
                Pn, vn = P.cpu().numpy(), vis.cpu().numpy()
                d = os.path.join(a.out, s, f"{t:06d}")
                os.makedirs(d, exist_ok=True)
                np.savez_compressed(os.path.join(d, f"{cam}.npz"), objects=np.asarray(sc.objects, np.int32),
                                    hw=np.asarray(sc.hw, np.int32), P=np.packbits(Pn.reshape(len(sc.objects), -1), axis=1),
                                    vis=np.packbits(vn.reshape(-1)), nP=Pn.reshape(len(sc.objects), -1).sum(1).astype(np.int64),
                                    nvis=np.int64(vn.sum()))
                log["views"] += 1
            print(f"{s:20s} t={t:4d} done ({time.time() - t0:6.1f}s) {log}", flush=True)
            del masks
    log["seconds"] = time.time() - t0
    json.dump(log, open(os.path.join(a.out, "proj_meta.json"), "w"), indent=1)
    print("DONE", log, flush=True)


if __name__ == "__main__":
    main()
