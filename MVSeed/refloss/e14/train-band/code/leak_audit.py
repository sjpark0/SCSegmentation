#!/usr/bin/env python3
"""S1-E14 train-band: is DEV-A a leak-free yardstick for a TRAINED arm?  (host, no images read)

For each arm's 5 fold checkpoints (config.json scenes / held_out / frame_stride) and each of the 17 scenes:
  own fold      the checkpoint that decodes the scene (folds.json) must not list it in `scenes` (run.assert_no_leak)
  frames        which of the scene's TEST (ref) / DEV-A / DEV-B frames are training samples of the OTHER folds' models
                (geomem.data.enumerate_pairs: train_frames[::frame_stride], all ordered (r, v) annotated-camera pairs)
  direction     whether the DEV view direction (c_ini -> eval cam) is among the training (r, v) pairs
    stage1/.venv/bin/python leak_audit.py --out leak_audit.json
"""
import argparse
import json
import os
import sys

S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
from geomem import devset  # noqa: E402
from geomem.cache import train_frames  # noqa: E402
from mvseg2d.io import load_meta  # noqa: E402

DATA = os.path.join(S1, "Data")
RUNS = os.path.join(S1, "geomem", "runs")
ARMS = ("geoset_pred_ft", "geoset_cut0_ft", "geoset_dec_ft", "geoset_decb_ft")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
folds = json.load(open(os.path.join(S1, "geomem", "folds.json")))
scenes = sorted(folds)
res = dict(arms={}, scenes={})
for arm in ARMS:
    cfg = {k: json.load(open(os.path.join(RUNS, arm, f"fold{k}", "config.json"))) for k in range(5)}
    own_ok = all(s not in cfg[folds[s]]["scenes"] and s in cfg[folds[s]]["held_out"] for s in scenes)
    other_cover = all(all(s in cfg[k]["scenes"] for k in range(5) if k != folds[s]) for s in scenes)
    res["arms"][arm] = dict(own_fold_never_trained_on_scene=own_ok, scene_in_all_other_folds_training=other_cover,
                            frame_stride=sorted({cfg[k]["frame_stride"] for k in range(5)}),
                            samples=[cfg[k]["samples"] for k in range(5)], steps=[cfg[k]["steps"] for k in range(5)],
                            seconds=[round(cfg[k]["seconds"]) for k in range(5)], git=sorted({cfg[k]["git"] for k in range(5)}))
stride = 2
for s in scenes:
    m = load_meta(DATA, s)
    tr = train_frames(DATA, s)[::stride]
    A, B = devset.dev_frames(DATA, s, "A"), devset.dev_frames(DATA, s, "B")
    cams = list(m["mask_cams"])
    dirs = {(r, v) for r in cams for v in cams if r != v}
    res["scenes"][s] = dict(fold=folds[s], ref=int(m["ref_frame"]), n_train_frames=len(train_frames(DATA, s)),
                            train_stride2=tr, test_in_train=int(m["ref_frame"]) in tr,
                            devA=A, devA_in_train=sum(t in tr for t in A), devB_in_train=sum(t in tr for t in B), n_devB=len(B),
                            dev_dirs_in_train=all((m["ref_cam"], c) in dirs for c in devset.eval_cams(DATA, s)))
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res["arms"], indent=0))
print("scenes: TEST frame in other folds' training", sum(v["test_in_train"] for v in res["scenes"].values()), "/17;",
      "DEV-A frames in training", sum(v["devA_in_train"] for v in res["scenes"].values()), "/", sum(len(v["devA"]) for v in res["scenes"].values()),
      "; DEV-B", sum(v["devB_in_train"] for v in res["scenes"].values()), "/", sum(v["n_devB"] for v in res["scenes"].values()),
      "; DEV directions among training pairs", sum(v["dev_dirs_in_train"] for v in res["scenes"].values()), "/17")
