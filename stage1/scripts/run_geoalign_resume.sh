#!/usr/bin/env bash
# Resume the GeoAlign chain after the host GPU driver mismatch of 2026-10-10 06:13 KST (folds 0-3 are done).
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/run_geomem_train.sh geoalign_ft 4
scripts/run_geomem.sh all --arm geoalign_ft --gate reproj --out geoalign_ft_rep --r1-out geoalign_ft_r1
for name in geoalign_ft_r1 geoalign_ft_rep; do
  .venv/bin/python -m mvseg2d.evaluate --data Data --method "$name"
done
echo "GEOALIGN CHAIN DONE"
