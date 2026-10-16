#!/usr/bin/env bash
# usage: run_probe.sh <mode> <scene> <outname>
set -uo pipefail
MODE=$1; DS=$2; OUTN=$3
ROOT=/home/sjpark/Documents/SCSegmentation
S=/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/s2r6x/mem
FOLDER=$DS
IMAGE_ID="$(docker images --no-trunc -q scsam3:latest 2>/dev/null | head -n1)"
start=$SECONDS
timeout 1500 docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
  -e SCSAM3_TRIM_CACHED_OUTPUTS=1 -e SCSAM3_IMAGE_ID="${IMAGE_ID}" -e PYTHONDONTWRITEBYTECODE=1 \
  -e PROBE_MODE="$MODE" -e PROBE_JSON="/host$S/probe_${DS}_${MODE}.json" \
  -e PROBE_OUTDIR="/host$S/data/$FOLDER/$OUTN" \
  -v /:/host -w "/host${ROOT}/SCSam3" scsam3 \
  python "/host$S/probe_mem.py" "$DS" --algo MVOpt --data-root "/host$S/data" --out "$OUTN" \
    --xview-window 1 --xview-mode C --xview-gate --xview-ptr --xview-tpos-shift 4 --ref-cam muvod \
    --stage1 geoset_rep_xs --frame0-seed --crop-small 10000 --crop-gate v2 --ref-snap --overwrite \
  > "$S/log_${DS}_${MODE}.txt" 2>&1
st=$?
echo "exit $st after $(( SECONDS - start )) s"
docker run --rm -v /:/host scsam3 chown -R "$(id -u):$(id -g)" "/host$S/data" "/host$S" >/dev/null 2>&1
exit $st
