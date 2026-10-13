#!/usr/bin/env bash
# scripts/run_pi3_sam3.sh <scene|all> [--source NAME] [--out NAME] [--overwrite]
set -euo pipefail
SCENE=$1; shift
OUT=pi3_sam3
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec "$(dirname "$0")/docker_run.sh" scsam3 python methods/pi3_sam3/run.py --scene "$SCENE" --out "$OUT" "${ARGS[@]}"
