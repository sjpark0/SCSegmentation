#!/usr/bin/env bash
# scripts/run_geomem.sh <scene|all> [--arm X] [--untrained] [--out NAME] [--overwrite]
set -euo pipefail
SCENE=$1; shift
OUT=""
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [ -z "$OUT" ]; then
  OUT=geomem
  for a in "${ARGS[@]}"; do [ "$a" = "--untrained" ] && OUT=star; done
  for i in "${!ARGS[@]}"; do [ "${ARGS[$i]}" = "--arm" ] && OUT="${ARGS[$((i+1))]}"; done
  for i in "${!ARGS[@]}"; do [ "${ARGS[$i]}" = "--gate" ] && [ "${ARGS[$((i+1))]}" = "obj" ] && OUT="${OUT}_obj"; done
fi
exec "$(dirname "$0")/docker_run.sh" scsam3 python methods/geomem/run.py --scene "$SCENE" --out "$OUT" "${ARGS[@]}"
