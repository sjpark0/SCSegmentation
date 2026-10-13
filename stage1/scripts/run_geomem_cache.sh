#!/usr/bin/env bash
# scripts/run_geomem_cache.sh <scene|all>   — Pi3 for every training frame (scgeo:pi3) + SAM 3 backbone features (scsam3)
set -euo pipefail
SCENE=$1
D="$(dirname "$0")"
"$D/docker_run.sh" scgeo:pi3 python -m geomem.pi3_frames --scene "$SCENE"
exec "$D/docker_run.sh" scsam3 python -m geomem.cache_feats --scene "$SCENE"
