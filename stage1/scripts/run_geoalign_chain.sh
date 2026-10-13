#!/usr/bin/env bash
# GeoAlign (docs/superpowers/plans/2026-10-10-geoalign.md Task 5): level-72 cache (skips existing), five folds of
# geoalign_ft, two-round scoring with the reprojection gate (geoalign_ft_r1, geoalign_ft_rep) and the evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/docker_run.sh scsam3 python -m geomem.cache_feats --scene all --level72 --frame-stride 2
for k in 0 1 2 3 4; do
  scripts/run_geomem_train.sh geoalign_ft "$k"
done
scripts/run_geomem.sh all --arm geoalign_ft --gate reproj --out geoalign_ft_rep --r1-out geoalign_ft_r1
for name in geoalign_ft_r1 geoalign_ft_rep; do
  .venv/bin/python -m mvseg2d.evaluate --data Data --method "$name"
done
echo "GEOALIGN CHAIN DONE"
