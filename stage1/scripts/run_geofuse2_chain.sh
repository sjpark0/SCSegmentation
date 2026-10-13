#!/usr/bin/env bash
# GeoFuse discriminating rerun: geofuse2_ft (two-sided alignment validity, own view excluded), five folds, then the
# two-round scoring with the reprojection gate (geofuse2_ft_r1, geofuse2_ft_rep) and the evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/docker_run.sh scsam3 python -m geomem.cache_feats --scene all --level72 --frame-stride 2
for k in 0 1 2 3 4; do
  scripts/run_geomem_train.sh geofuse2_ft "$k"
done
scripts/run_geomem.sh all --arm geofuse2_ft --gate reproj --out geofuse2_ft_rep --r1-out geofuse2_ft_r1
for name in geofuse2_ft_r1 geofuse2_ft_rep; do
  .venv/bin/python -m mvseg2d.evaluate --data Data --method "$name"
done
echo "GEOFUSE2 CHAIN DONE"
