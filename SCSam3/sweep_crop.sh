#!/usr/bin/env bash
# S2-R1: crop tracking over the 17 MUVOD scenes, one foreground container per scene.
#
#   SCSam3/sweep_crop.sh <max-area> <suffix> [scene ...]
#   SCSam3/sweep_crop.sh 2000 Cr2k
#
# Same shape as MVSeed/sweep_seeds.sh: one container per scene so the harness's
# low-memory watchdog never sees a long background docker task, and each call stays
# under 600 s.  Base folder and seed folder are fixed to the registered ones
# (docs/stage2-R1-prereg.md §2).  Logs in SCSam3/logs/crop-<suffix>/<scene>.log.
set -euo pipefail
MAXAREA="${1:?max seed area in px, e.g. 2000}"
SUFFIX="${2:?derived-folder suffix, e.g. Cr2k}"
shift 2
BASE="${BASE:-SegMaskSam3XW0MFs}"
SEEDS="${SEEDS:-MVSeed_e0_index}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
if [ "$#" -gt 0 ]; then
  SCENES=("$@")
else
  mapfile -t SCENES < <(python3 -c "import json; print('\n'.join(sorted(json.load(open('docs/raw/muvod_object_sets.json')))))")
fi
mkdir -p "SCSam3/logs/crop-$SUFFIX"
for s in "${SCENES[@]}"; do
  log="SCSam3/logs/crop-$SUFFIX/$s.log"
  echo "== $s -> $BASE$SUFFIX  $(date +%H:%M:%S)"
  docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
    --user "$(id -u):$(id -g)" -v /:/host -w "/host$REPO/SCSam3" \
    -e HF_HOME="/host$REPO/SCSam3/hf_cache" -e HF_HUB_OFFLINE=1 scsam3 \
    python crop_track.py "$s" --base "$BASE" --seeds "$SEEDS" --max-area "$MAXAREA" \
      --suffix "$SUFFIX" --overwrite 2>&1 | tee "$log" | tail -2
done
echo "done: $BASE$SUFFIX"
