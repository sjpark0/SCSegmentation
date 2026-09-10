#!/usr/bin/env bash
# S1-E4a plumbing: COLMAP dense MVS depth for the three annotated cameras of a scene.
#
#   MVSeed/geo/dense_depth.sh <scene> [<scene> ...]
#
# Requires MVSeed/cache/<scene>/pose/sparse/0 (colmap_first_frames.sh).  Undistorts every registered
# image (max side MAX_SIZE, default 1600) so the neighbour images are available as stereo sources, then
# restricts stereo/patch-match.cfg to the annotated cameras, so only those get a depth map.  Writes
# MVSeed/cache/<scene>/pose/dense/{images,sparse,txt,stereo/depth_maps}, and dense/summary.json.
# Idempotent: a scene with dense/summary.json is skipped.  Log: dense/dense.log
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
MAX_SIZE="${MAX_SIZE:-1600}"
for scene in "$@"; do
  work="MVSeed/cache/$scene/pose"
  [ -d "$work/sparse/0" ] || { echo "== $scene: no sparse model, skipping"; continue; }
  [ -f "$work/dense/summary.json" ] && { echo "== $scene: dense done, skipping"; continue; }
  cams=$(python3 -c "import json;print(' '.join(json.load(open('docs/raw/muvod_object_sets.json'))['$scene']['cameras']))")
  echo "== $scene: dense for [$cams] max_size $MAX_SIZE $(date +%H:%M:%S)"
  t0=$(date +%s)
  docker run --rm --gpus all --shm-size=16g --memory=60g --memory-swap=60g \
    --user "$(id -u):$(id -g)" -v /:/host -w "/host$REPO/$work" \
    -e CAMS="$cams" -e MAX_SIZE="$MAX_SIZE" 3dgs bash -c '
      set -e
      rm -rf dense; mkdir -p dense
      colmap image_undistorter --image_path images --input_path sparse/0 \
        --output_path dense --output_type COLMAP --max_image_size "$MAX_SIZE" > /dev/null
      colmap model_converter --input_path dense/sparse --output_path dense/txt --output_type TXT > /dev/null 2>&1 \
        || { mkdir -p dense/txt && colmap model_converter --input_path dense/sparse --output_path dense/txt --output_type TXT > /dev/null; }
      python - <<PY
import os
cams = set(os.environ["CAMS"].split())
lines = open("dense/stereo/patch-match.cfg").read().splitlines()
keep, i = [], 0
while i < len(lines):
    name, src = lines[i], lines[i + 1]
    if os.path.splitext(name)[0] in cams:
        keep += [name, src]
    i += 2
assert keep, "no annotated camera is registered"
open("dense/stereo/patch-match.cfg", "w").write("\n".join(keep) + "\n")
print("depth maps to compute:", len(keep) // 2)
PY
      colmap patch_match_stereo --workspace_path dense --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size "$MAX_SIZE" 2>&1 | tail -2
      ls dense/stereo/depth_maps/*.geometric.bin | wc -l
    ' > "$work/dense.log" 2>&1
  rc=$?
  mkdir -p "$work/dense"
  python3 - "$work" "$MAX_SIZE" "$((`date +%s` - t0))" "$rc" "$cams" <<'PY'
import glob, json, os, sys
work, mx, secs, rc, cams = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split()
maps = sorted(os.path.basename(p) for p in glob.glob(f"{work}/dense/stereo/depth_maps/*.geometric.bin"))
s = dict(max_image_size=mx, rc=rc, seconds=secs, cameras=cams, depth_maps=maps, n_depth_maps=len(maps))
json.dump(s, open(f"{work}/dense/summary.json", "w"), indent=1)
print(f"   -> {len(maps)} depth maps of {len(cams)} cameras, {secs}s, rc {rc}")
PY
  mv "$work/dense.log" "$work/dense/dense.log" 2>/dev/null || true
done
echo "dense sweep done $(date +%H:%M:%S)"
