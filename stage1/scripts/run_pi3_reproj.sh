#!/usr/bin/env bash
# scripts/run_pi3_reproj.sh <scene|all> [--pixel-limit N] [--out NAME] [--skip-geometry] [--overwrite]
set -euo pipefail
SCENE=$1; shift
OUT=pi3_reproj
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec "$(dirname "$0")/docker_run.sh" scgeo:pi3 python methods/pi3_reproj/run.py --scene "$SCENE" --out "$OUT" "${ARGS[@]}"
