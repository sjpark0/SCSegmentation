#!/usr/bin/env python3
"""S1-E12 post-hoc Zt value (review 2026-10-15): the 'not cut' share of M+ — eligible objects (>= 20 cells on the
72x72 reference grid) that drew a cut but lost no object cell — which the training log cannot give (it keeps n_cut,
not the number of eligible objects).  Replays each fold's sample order (SampleIterator(seed 0) over
enumerate_samples(fold scenes, stride 2)), checks the logged objects against the step's reference label map, counts
the eligible ones and compares with the logged n_cut.  Host (stage1/.venv):
    PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python MVSeed/refloss/e12/notcut_share.py
"""
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
S1 = os.path.join(REPO, "stage1")
sys.path.insert(0, S1)
from geomem.data import SampleIterator, _labelmap, enumerate_samples, objects_for  # noqa: E402
from geomem.folds import load_folds, train_scenes  # noqa: E402

DATA = os.path.join(S1, "Data")
RUNS = os.path.join(S1, "geomem", "runs", "geoset_cutall_ft")


def cells(lab, o):
    h, w = lab.shape
    l = lab[(np.arange(1008) * h // 1008)][:, (np.arange(1008) * w // 1008)]
    return int(((l == o).reshape(72, 14, 72, 14).mean(axis=(1, 3)) >= 0.5).sum())


def main():
    folds = load_folds()
    out, cache = {}, {}
    tot_elig = tot_cut = 0
    for k in range(5):
        log = list(csv.DictReader(open(os.path.join(RUNS, f"fold{k}", "log.csv"))))
        cfg = json.load(open(os.path.join(RUNS, f"fold{k}", "config.json")))
        samples = enumerate_samples(DATA, train_scenes(folds, k), cfg["frame_stride"])
        it = SampleIterator(samples, seed=cfg["seed"])
        seq = []
        while len(seq) < len(log):
            seq += it.epoch()
        elig = cut = mismatch = 0
        for row, s in zip(log, seq):
            objs = [int(o) for o in row["objs"].split()]
            key = (s.scene, s.t, s.ref_cam)
            if key not in cache:
                lab = _labelmap(DATA, s, s.ref_cam)
                cache[key] = (lab, set(objects_for(s, DATA)))
            lab, ids = cache[key]
            if not set(objs) <= ids:
                mismatch += 1
                continue
            e = sum(cells(lab, o) >= 20 for o in objs)
            elig += e
            cut += int(row["n_cut"])
        out[f"fold{k}"] = dict(steps=len(log), eligible=elig, cut=cut, not_cut_share=1 - cut / elig if elig else None, replay_mismatch=mismatch)
        tot_elig += elig; tot_cut += cut
        print(f"fold{k}: steps {len(log)}  eligible objects {elig}  cut {cut}  not cut {1 - cut / elig:.4f}  replay mismatches {mismatch}", flush=True)
    out["all"] = dict(eligible=tot_elig, cut=tot_cut, not_cut_share=1 - tot_cut / tot_elig)
    print(f"all: not cut {out['all']['not_cut_share']:.4f} (registered watch value <= 0.02)")
    json.dump(out, open(os.path.join(REPO, "MVSeed", "runs", "e12", "notcut_share.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
