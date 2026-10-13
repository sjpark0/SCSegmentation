#!/usr/bin/env bash
# GeoSet-2 (docs/superpowers/plans/2026-10-09-geoset2.md Task 6): train both arms on all five folds, then score
# the seven result names, then evaluate each.  Sequential on purpose: never two GPU jobs at once.
set -euo pipefail
cd "$(dirname "$0")/.."
for arm in geoset_only_ft geoset_pred_ft; do
  for k in 0 1 2 3 4; do
    scripts/run_geomem_train.sh "$arm" "$k"
  done
done
scripts/run_geomem.sh all --arm geoset_ft --gate obj --out geoset_ft_obj --r1-out none
for arm in geoset_only_ft geoset_pred_ft; do
  scripts/run_geomem.sh all --arm "$arm"
  scripts/run_geomem.sh all --arm "$arm" --gate obj --out "${arm}_obj" --r1-out none
done
for name in geoset_ft_obj geoset_only_ft_r1 geoset_only_ft geoset_only_ft_obj geoset_pred_ft_r1 geoset_pred_ft geoset_pred_ft_obj; do
  .venv/bin/python -m mvseg2d.evaluate --data Data --method "$name"
done
echo "GEOSET2 CHAIN DONE"
