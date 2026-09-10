#!/usr/bin/env bash
# S1-E4c plumbing: a static 3D Gaussian Splatting model per scene from the first-frame COLMAP model.
#
#   MVSeed/geo/train_3dgs.sh <scene> [<scene> ...]
#
# Input  MVSeed/cache/<scene>/pose/{images,sparse/0}   (colmap_first_frames.sh); the original cameras are
#        SIMPLE_RADIAL, which the 3DGS reader refuses, so a FULL-RESOLUTION undistorted workspace
#        MVSeed/cache/<scene>/undist/{images,sparse/0} (PINHOLE) is built first and trained on.
# Output MVSeed/cache/<scene>/gs/point_cloud/iteration_<N>/point_cloud.ply + cameras.json + train.log
# Full resolution (-r 1): the whole point of this arm is that rendering is not resolution limited.
# All images are used for training so the three annotated cameras are reconstructed; the readout is a
# train-view PSNR, stated as such (docs/stage1-E4c-prereg.md section 2).  Idempotent on gs/summary.json.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
GS="${GS:-$HOME/Documents/gaussian-splatting}"
ITERS="${ITERS:-30000}"
for scene in "$@"; do
  work="MVSeed/cache/$scene"
  [ -d "$work/pose/sparse/0" ] || { echo "== $scene: no COLMAP model, skipping"; continue; }
  [ -f "$work/gs/summary.json" ] && { echo "== $scene: gs done, skipping"; continue; }
  if [ ! -f "$work/undist/sparse/0/cameras.bin" ]; then
    echo "   undistorting at full resolution..."
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -v /:/host -w "/host$REPO/$work" 3dgs bash -c '
      set -e
      rm -rf undist; mkdir -p undist
      colmap image_undistorter --image_path pose/images --input_path pose/sparse/0         --output_path undist --output_type COLMAP > /dev/null
      mkdir -p undist/sparse/0 && mv undist/sparse/*.bin undist/sparse/0/
    ' >> "$work/undistort.log" 2>&1 || { echo "   undistort failed"; continue; }
    # this fork of the 3DGS reader also wants the LLFF poses_bounds.npy, produced by its own converter
    docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host$GS" 3dgs       python imgs2poses.py "/host$REPO/$work/undist" >> "$work/undistort.log" 2>&1       || { echo "   imgs2poses failed"; continue; }
  fi
  n=$(ls "$work/undist/images" | wc -l)
  echo "== $scene: 3DGS on $n images, $ITERS iters, full resolution  $(date +%H:%M:%S)"
  t0=$(date +%s)
  mkdir -p "$work/gs"
  docker run --rm --gpus all --shm-size=24g --memory=90g --memory-swap=90g \
    --user "$(id -u):$(id -g)" -v /:/host -w "/host$GS" 3dgs \
    python train.py -s "/host$REPO/$work/undist" -m "/host$REPO/$work/gs" -r 1 \
      --iterations "$ITERS" --save_iterations "$ITERS" --test_iterations "$ITERS" \
      --disable_viewer --quiet > "$work/gs/train.log" 2>&1
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  python3 - "$work" "$ITERS" "$((t1-t0))" "$rc" "$n" <<'PY'
import json, os, re, sys
work, iters, secs, rc, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
log = open(f"{work}/gs/train.log", errors="replace").read()
ply = f"{work}/gs/point_cloud/iteration_{iters}/point_cloud.ply"
psnr = re.findall(r"PSNR\s+([\d.]+)", log)
ok = rc == 0 and os.path.exists(ply)
s = dict(iterations=iters, rc=rc, ok=ok, seconds=secs, n_images=n,
         psnr_train=float(psnr[-1]) if psnr else None, ply=ply if os.path.exists(ply) else None,
         n_gaussians=None)
if ok:
    with open(ply, "rb") as fh:
        head = fh.read(400).decode("latin1")
    m = re.search(r"element vertex (\d+)", head)
    s["n_gaussians"] = int(m.group(1)) if m else None
if ok:
    json.dump(s, open(f"{work}/gs/summary.json", "w"), indent=1)
print(f"   -> ok {ok}, {s['n_gaussians']} gaussians, PSNR(train) {s['psnr_train']}, {secs}s, rc {rc}")
PY
done
echo "3dgs sweep done $(date +%H:%M:%S)"
