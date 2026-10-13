#!/usr/bin/env bash
# Run a command inside one of the method images with the repo mounted at /host$PWD.
#   scripts/docker_run.sh <image> <cmd...>
# HF_HOME: scsam3 reads the SAM 3 snapshot from the main repo's SCSam3/hf_cache (the image's own copy is
# root-only); scgeo:* images bake their weights at /opt/hf.  REPO is <main>/stage1, MAIN its parent.
set -euo pipefail
IMAGE=$1; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
MAIN=$(cd "$REPO/.." && pwd)
# PYTHONPATH: the repo first; scgeo:pi3 also needs the image's own /opt/Pi3 (its Dockerfile sets
# PYTHONPATH=/opt/Pi3, which a -e PYTHONPATH would otherwise replace).
PP="/host$REPO"
case "$IMAGE" in
  scsam3*)   HF="/host$MAIN/SCSam3/hf_cache" ;;
  scgeo:pi3) HF="/opt/hf"; PP="$PP:/opt/Pi3" ;;
  *)         HF="/opt/hf" ;;
esac
# Mounts: the main repo (this stage-1 tree read-write, SCSam3 = SAM 3 package + weights) at the same
# /host$... path so relative symlinks and recorded paths stay valid.  Nothing else of the host is
# visible to the container.
exec docker run --rm --gpus all --shm-size=32g --user "$(id -u):$(id -g)" \
  -v "$MAIN:/host$MAIN" -w "/host$REPO" \
  -e HF_HOME="$HF" -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONPATH="$PP" \
  -e HOME=/tmp \
  "$IMAGE" "$@"
