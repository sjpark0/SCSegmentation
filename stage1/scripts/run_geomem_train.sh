#!/usr/bin/env bash
# scripts/run_geomem_train.sh <arm> <fold> [extra geomem.train args]
set -euo pipefail
ARM=$1; FOLD=$2; shift 2
exec "$(dirname "$0")/docker_run.sh" scsam3 python -m geomem.train --arm "$ARM" --fold "$FOLD" "$@"
