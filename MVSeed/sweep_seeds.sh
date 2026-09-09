#!/usr/bin/env bash
# Run MVSeed/run_seed.py over the 17 MUVOD scenes, one foreground container per scene.
#
#   MVSeed/sweep_seeds.sh <order> <out-folder> [scene ...]
#   MVSeed/sweep_seeds.sh index MVSeed_e0_index
#
# One container per scene (not one long background job) because the harness's
# low-memory watchdog kills background docker tasks at container transitions
# (docs/operations.md trap 11); a foreground call per scene stays under its radar
# and under the 600 s call limit.  Scenes come from docs/raw/muvod_object_sets.json
# unless given on the command line.  Logs land in MVSeed/logs/<out>/<scene>.log.
set -euo pipefail
ORDER="${1:?order: index|reverse|ref_outward}"
OUT="${2:?output folder name, must start with MVSeed_}"
shift 2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
if [ "$#" -gt 0 ]; then
  SCENES=("$@")
else
  mapfile -t SCENES < <(python3 -c "import json; print('\n'.join(sorted(json.load(open('docs/raw/muvod_object_sets.json')))))")
fi
mkdir -p "MVSeed/logs/$OUT"
for s in "${SCENES[@]}"; do
  log="MVSeed/logs/$OUT/$s.log"
  if [ -f "Data/MVSeg/$s/$OUT/SEED_MANIFEST.json" ]; then
    echo "skip $s (manifest exists)"; continue
  fi
  echo "== $s -> $OUT ($ORDER)  $(date +%H:%M:%S)"
  docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
    --user "$(id -u):$(id -g)" -v /:/host -w "/host$REPO" \
    -e HF_HOME="/host$REPO/SCSam3/hf_cache" -e HF_HUB_OFFLINE=1 scsam3 \
    python MVSeed/run_seed.py "$s" --order "$ORDER" --out "$OUT" 2>&1 | tee "$log" | tail -3
done
echo "done: $OUT"
