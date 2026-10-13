#!/usr/bin/env bash
# scripts/run_sam3_prop.sh <scene|all> [--order index|ref_outward] [--out NAME] [--overwrite]
set -euo pipefail
SCENE=$1; shift
OUT=sam3_prop
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec "$(dirname "$0")/docker_run.sh" scsam3 python methods/sam3_prop/run.py --scene "$SCENE" --out "$OUT" "${ARGS[@]}"
