#!/usr/bin/env bash
# scripts/run_dino_sam3.sh <scene|all> [--out NAME] [--overwrite]
# Stage A (DINOv2 candidates) in scgeo:dino, stage B (SAM 3) in scsam3.
set -euo pipefail
SCENE=$1; shift
OUT=dino_sam3
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
D="$(dirname "$0")"
"$D/docker_run.sh" scgeo:dino python methods/dino_sam3/prompts.py --scene "$SCENE"
exec "$D/docker_run.sh" scsam3 python methods/dino_sam3/run.py --scene "$SCENE" --out "$OUT" "${ARGS[@]}"
