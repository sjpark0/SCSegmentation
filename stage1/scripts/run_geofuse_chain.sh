#!/usr/bin/env bash
# GeoFuse (docs/superpowers/plans/2026-10-09-geofuse.md Task 7): level-72 cache, five folds of geofuse_ft, then the
# two-round scoring with the reprojection gate (writes geofuse_ft_r1 and geofuse_ft_rep) and the evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/docker_run.sh scsam3 python -m geomem.cache_feats --scene all --level72 --frame-stride 2
for k in 0 1 2 3 4; do
  scripts/run_geomem_train.sh geofuse_ft "$k"
done
scripts/run_geomem.sh all --arm geofuse_ft --gate reproj --out geofuse_ft_rep --r1-out geofuse_ft_r1
for name in geofuse_ft_r1 geofuse_ft_rep; do
  .venv/bin/python -m mvseg2d.evaluate --data Data --method "$name"
done
echo "GEOFUSE CHAIN DONE"
