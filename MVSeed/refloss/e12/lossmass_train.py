#!/usr/bin/env python3
"""S1-E12 design check, in-sample version: the adopted checkpoints' focal loss on their OWN training samples
(fold k checkpoint on 3 of fold k's training scenes, first training frame, all 6 ordered pairs, the trainer's
load_sample / choose_objects(seed 0)), split by reference visibility (train-frame Pi3 cache) x GT / non-GT.
    stage1/scripts/docker_run.sh scsam3 python -c "$(cat lossmass_train.py)" > lossmass_train.json
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = "/host/home/sjpark/Documents/SCSegmentation"
S1 = os.path.join(REPO, "stage1")
sys.path.insert(0, S1)
from geomem.cache import pi3_frame_cache, train_frames  # noqa: E402
from geomem.data import MAX_OBJECTS, Sample, choose_objects, load_sample, objects_for  # noqa: E402
from geomem.folds import load_folds, train_scenes  # noqa: E402
from geomem.sam3_io import logits_to_original  # noqa: E402
from geomem.train import build_wrapper  # noqa: E402
from methods import sam3_single  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from mvseg2d.io import load_meta  # noqa: E402

DATA = os.path.join(S1, "Data")
RUNS = os.path.join(S1, "geomem", "runs")
ARM = "geoset_pred_ft"
ALPHA, GAMMA = 0.25, 2.0
SCENES_PER_FOLD = 3


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def focal_map(logits, t):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p_t = p * t + (1 - p) * (1 - t)
    return (ALPHA * t + (1 - ALPHA) * (1 - t)) * ce * (1 - p_t) ** GAMMA


def main():
    t0 = time.time()
    folds = load_folds()
    model = sam3_single.load_model()
    init_state = R.snapshot_init(model)
    wrapper = build_wrapper(model, ARM)
    wrapper.eval()
    rng = np.random.default_rng(0)
    rows = []
    for k in range(5):
        ckpt = os.path.join(RUNS, ARM, f"fold{k}", "ckpt.pt")
        ck = R.load_checkpoint_into(wrapper, wrapper.geo_embed, ckpt)
        try:
            for sc in train_scenes(folds, k)[::max(1, len(train_scenes(folds, k)) // SCENES_PER_FOLD)][:SCENES_PER_FOLD]:
                assert folds[sc] != k
                meta = load_meta(DATA, sc)
                cams = list(meta["mask_cams"])
                t = int(train_frames(DATA, sc)[0])
                for r in cams:
                    for v in cams:
                        if r == v:
                            continue
                        s = Sample(sc, t, r, v)
                        objs = choose_objects(objects_for(s, DATA), MAX_OBJECTS, rng)
                        b = load_sample(DATA, s, objs, "cuda")
                        B = b["ref_masks"].shape[0]
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            ref = wrapper.encode_reference(b["ref_image"], b["ref_backbone"], b["ref_masks"], b["p_ref"], b["valid_ref"])
                            out = wrapper.decode_view(b["tgt_backbone"], B, ref, b["p_tgt"], b["valid_tgt"])
                        logits = logits_to_original(out["logits_1008"], b["hw"]).float()[:, 0]
                        gt = b["tgt_masks"]
                        cache = pi3_frame_cache(DATA, sc, t)
                        inp = R.reproj_inputs(cache, v, r, b["hw"], "cuda")
                        _, vis = R.project_reference_masks(b["ref_masks"][:1], *inp)
                        del inp
                        inv = ~vis
                        fm = focal_map(logits, gt.float())
                        pos = logits > 0
                        for i in range(B):
                            g, f, pp = gt[i], fm[i], pos[i]
                            row = {"fold": k, "scene": sc, "t": t, "ref": r, "cam": v, "obj": int(objs[i]), "img_px": int(g.numel())}
                            for cname, cm in (("vis", vis), ("inv", inv)):
                                row[f"gt_{cname}"] = int((g & cm).sum()); row[f"ngt_{cname}"] = int((~g & cm).sum())
                                row[f"focal_gt_{cname}"] = float(f[g & cm].sum()); row[f"focal_ngt_{cname}"] = float(f[~g & cm].sum())
                                row[f"fn_{cname}"] = int((g & ~pp & cm).sum()); row[f"fp_{cname}"] = int((~g & pp & cm).sum())
                                row[f"tp_{cname}"] = int((g & pp & cm).sum())
                            rows.append(row)
                        del b, ref, out, logits, gt, fm, pos, vis, inv
                        torch.cuda.empty_cache()
                log(f"fold {k} {sc}: {len(rows)} rows, {time.time() - t0:.0f} s")
        finally:
            R.restore_init(model, init_state, ck)
    wrapper.close()
    print(json.dumps({"meta": {"arm": ARM, "seconds": time.time() - t0}, "rows": rows}))


main()
