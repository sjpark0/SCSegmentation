#!/usr/bin/env bash
set -uo pipefail
S=/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/s2r6x
start=$SECONDS
docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g   -e SCSAM3_TRIM_CACHED_OUTPUTS=1 -e PYTHONDONTWRITEBYTECODE=1   -v /:/host -w /host/home/sjpark/Documents/SCSegmentation/SCSam3 scsam3   python runMVSeg.py "$1" --algo MVOpt --xview-window 0 --ref-cam muvod --frame0-seed --ref-snap --save-stages ${EXTRA:-} --config /host$S/svdata/sv_config.json --data-root /host$S/svdata --overwrite
st=$?
docker run --rm -v /:/host scsam3 chown -R 1000:1000 /host$S/svdata >/dev/null 2>&1
echo "exit $st  $((SECONDS-start)) s"
