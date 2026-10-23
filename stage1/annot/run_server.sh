#!/usr/bin/env bash
# CROOM annotation server in the scsam3 image with the port published (docker_run.sh does not publish ports).
#   stage1/annot/run_server.sh [port]            -> http://localhost:<port>  (default 7861)
set -euo pipefail
PORT=${1:-7861}
REPO=$(cd "$(dirname "$0")/.." && pwd)
MAIN=$(cd "$REPO/.." && pwd)
exec docker run --rm --gpus all --shm-size=16g --user "$(id -u):$(id -g)" -p "127.0.0.1:$PORT:$PORT" \
  -v "$MAIN:/host$MAIN" -w "/host$REPO" -e HF_HOME="/host$MAIN/SCSam3/hf_cache" -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH="/host$REPO" -e HOME=/tmp -e PYTHONUNBUFFERED=1 --name croom_annot \
  scsam3 python -m annot.server --root "/host$REPO/Data/_annot/CROOM" --port "$PORT"
