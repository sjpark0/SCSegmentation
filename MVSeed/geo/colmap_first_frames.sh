#!/usr/bin/env bash
# S1-E4 plumbing: COLMAP SfM on the start-frame image of every camera of a scene (no calibration input),
# into MVSeed/cache/<scene>/pose/ (gitignored).  Uses the existing `3dgs` image (COLMAP + CUDA).
#
#   MVSeed/geo/colmap_first_frames.sh <scene> [<scene> ...]
#
# Per scene: images/<cam>.jpg (copies of Video/<cam>/<start>.jpg), database.db, sparse/0 (bin + txt),
# summary.json (registered images, points, mean reprojection error, wall time).  Idempotent: a scene
# with summary.json is skipped.  Log: MVSeed/cache/<scene>/pose/colmap.log
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
for scene in "$@"; do
  folder=$(python3 -c "import json;print(json.load(open('SCSam3/demo/MVSeg.json',encoding='utf-8'))['$scene']['folder'])")
  start=$(python3 -c "import json;print(json.load(open('SCSam3/demo/MVSeg.json',encoding='utf-8'))['$scene']['start_frame'])")
  work="MVSeed/cache/$scene/pose"
  if [ -f "$work/summary.json" ]; then echo "== $scene: done, skipping"; continue; fi
  mkdir -p "$work/images"
  for cam in $(ls "Data/MVSeg/$folder/Video"); do
    cp "Data/MVSeg/$folder/Video/$cam/$(printf '%06d' "$start").jpg" "$work/images/$cam.jpg"
  done
  n=$(ls "$work/images" | wc -l)
  echo "== $scene: $n images (frame $start) $(date +%H:%M:%S)"
  t0=$(date +%s)
  docker run --rm --gpus all --shm-size=16g --memory=60g --memory-swap=60g \
    --user "$(id -u):$(id -g)" -v /:/host -w "/host$REPO/$work" 3dgs bash -c '
      set -e
      rm -f database.db; rm -rf sparse; mkdir -p sparse
      # the recipe that registered Carpark 9/9 (SA3D_YOLO/colmap.sh); PINHOLE + guided matching found no initial pair
      colmap feature_extractor --database_path database.db --image_path images \
        --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 \
        --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
      colmap exhaustive_matcher --database_path database.db
      colmap mapper --database_path database.db --image_path images --output_path sparse
      best=""; bestn=0
      for m in sparse/*/; do
        k=$(colmap model_analyzer --path "$m" 2>&1 | grep -oE "Registered images: [0-9]+" | grep -oE "[0-9]+$")
        if [ "${k:-0}" -gt "$bestn" ]; then bestn=$k; best=$m; fi
      done
      echo "best model $best registered $bestn"
      if [ -n "$best" ] && [ "$best" != "sparse/0/" ]; then rm -rf sparse/0; mv "$best" sparse/0; fi
      mkdir -p sparse/0/txt && colmap model_converter --input_path sparse/0 --output_path sparse/0/txt --output_type TXT
      colmap model_analyzer --path sparse/0 2>&1 | tail -12
    ' > "$work/colmap.log" 2>&1
  rc=$?
  t1=$(date +%s)
  python3 - "$work" "$n" "$((t1-t0))" "$rc" <<'PY'
import json, re, sys
work, n, secs, rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
log = open(f"{work}/colmap.log", errors="replace").read()
g = lambda k: (re.findall(k, log) or [None])[-1]
s = dict(n_images=n, rc=rc, seconds=secs,
         registered=int(g(r"Registered images: (\d+)") or 0), points=int(g(r"Points: (\d+)") or 0),
         observations=int(g(r"Observations: (\d+)") or 0), mean_track_length=g(r"Mean track length: ([\d.]+)"),
         mean_reproj_error=g(r"Mean reprojection error: ([\d.]+)px"), n_models=len(re.findall(r"best model", log)))
json.dump(s, open(f"{work}/summary.json", "w"), indent=1)
print(f"   -> registered {s['registered']}/{n}, points {s['points']}, reproj {s['mean_reproj_error']} px, {secs}s, rc {rc}")
PY
done
echo "sweep done $(date +%H:%M:%S)"
