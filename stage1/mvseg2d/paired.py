"""Paired, scene-level comparison of two methods with a scene-cluster bootstrap (spec 5 / plan Task 10).

The unit of resampling is the scene (the clusters are scenes; pairs within a scene are correlated).
`delta` = mean over common scenes of (scene mean of b − scene mean of a), where a scene mean is the
pair → view → scene hierarchy used by the headline.  `ci` is the percentile 95 % interval of
`delta` over `n` resamples of the scenes with replacement.
"""
from collections import defaultdict

import numpy as np


def scene_means(rows, key):
    """{scene: mean over views of (mean over pairs of row[key])}."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["scene"]][r["cam"]].append(float(r[key]))
    return {s: float(np.mean([np.mean(v) for v in views.values()])) for s, views in by.items()}


def paired_scene_bootstrap(rows_a, rows_b, key="J", n=10_000, seed=0):
    ma, mb = scene_means(rows_a, key), scene_means(rows_b, key)
    scenes = sorted(set(ma) & set(mb))
    d = np.array([mb[s] - ma[s] for s in scenes], dtype=np.float64)
    if d.size == 0:
        return {"delta": float("nan"), "ci": (float("nan"), float("nan")), "n_scenes": 0, "per_scene": {}}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci": (float(lo), float(hi)), "n_scenes": int(d.size),
            "per_scene": {s: float(x) for s, x in zip(scenes, d)}}
