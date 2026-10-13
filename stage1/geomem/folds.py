"""Scene-level, rig-stratified 5-fold split of the MVSeg scenes (spec 4.2).

Hemisphere scenes go to folds 0..3 one each (sorted order); pinhole scenes, sorted, are dealt
round-robin starting at fold 4 so the fold sizes are 4,4,3,3,3.  Deterministic, no seed.

    python -m geomem.folds --write        # regenerate geomem/folds.json from Data/
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mvseg2d.io import HEMISPHERE  # noqa: E402

N_FOLDS = 5
FOLDS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folds.json")


def make_folds(scenes, n_folds=N_FOLDS):
    hemi = sorted(s for s in scenes if s in HEMISPHERE)
    pin = sorted(s for s in scenes if s not in HEMISPHERE)
    folds = {}
    for i, s in enumerate(hemi):
        folds[s] = i % n_folds
    k = n_folds - 1
    for s in pin:
        folds[s] = k
        k = (k + 1) % n_folds
    return folds


def load_folds(path=FOLDS_JSON):
    with open(path, encoding="utf-8") as f:
        return {k: int(v) for k, v in json.load(f).items()}


def train_scenes(folds, k):
    return sorted(s for s, f in folds.items() if f != k)


def eval_scenes(folds, k):
    return sorted(s for s, f in folds.items() if f == k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    scenes = sorted(d for d in os.listdir(a.data) if os.path.isfile(os.path.join(a.data, d, "meta.json")))
    folds = make_folds(scenes)
    if a.write:
        with open(FOLDS_JSON, "w", encoding="utf-8") as f:
            json.dump(folds, f, indent=1, sort_keys=True)
    for k in range(N_FOLDS):
        print(k, eval_scenes(folds, k))


if __name__ == "__main__":
    main()
