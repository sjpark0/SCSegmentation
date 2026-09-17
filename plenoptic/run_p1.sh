#!/usr/bin/env bash
# S3-P runs (docs/plenoptic-P-prereg.md section 1).  Same docker invocation as SCSam3/runForSam2Three.sh;
# the runner, packages, MVSeg.json and eval/ are untouched -- only --config/--data-root point at the rig.
#   plenoptic/run_p1.sh P1-C | P1-H | P2-T | P2-P
set -u
ROOT=/home/sjpark/Documents/SCSegmentation
LOG=${LOG:-/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/9786e830-b342-4be7-bc11-e7701415bd8d/scratchpad}
SHARE=$ROOT/PlenopticServer1/CROOM/SegResult
CTRL="--algo MVOpt --xview-window 0 --track-cams closure --ref-cam muvod --frame0-seed"
HEAD="--algo MVOpt --xview-window 1 --xview-mode C --xview-gate --xview-ptr --xview-tpos-shift 4 --ref-cam muvod --frame0-seed --crop-small 10000 --crop-gate v2"
MEM="--algo MVOpt --xview-window 1 --xview-mode C --xview-gate --xview-ptr --xview-tpos-shift 4 --ref-cam muvod --frame0-seed"   # memory only, no crop retracking (P5)
IMAGE_ID="$(docker images --no-trunc -q scsam3:latest 2>/dev/null | head -n1)"
run() {  # id dataset config data-root flags...
  local id=$1 ds=$2 cfg=$3 root=$4; shift 4
  local start=$SECONDS
  docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
    -e SCSAM3_IMAGE_ID="${IMAGE_ID}" -v /:/host -w "/host${ROOT}/SCSam3" scsam3 \
    python runMVSeg.py "$ds" --config "/host$cfg" --data-root "/host$root" "$@" --overwrite \
    > "$LOG/plen_$id.log" 2>&1
  local st=$?
  docker run --rm -v /:/host scsam3 chown -R "$(id -u):$(id -g)" "/host$root" >/dev/null 2>&1
  printf '%s exit %d  %ds\n' "$id" "$st" $((SECONDS - start)) | tee -a "$LOG/plen_runs.txt"
  return $st
}
case "${1:-}" in
  P1-C) run P1-C T06_flag_ganggang $ROOT/plenoptic/Plenoptic.json $SHARE $CTRL ;;
  P1-H) run P1-H T06_flag_ganggang $ROOT/plenoptic/Plenoptic.json $SHARE $HEAD ;;
  P2-T) run P2-T T06_flag_ganggang_txt $ROOT/plenoptic/Plenoptic_variants.json $ROOT/Data/Plenoptic $CTRL ;;
  P2-P) run P2-P T06_flag_ganggang_pt $ROOT/plenoptic/Plenoptic_variants.json $ROOT/Data/Plenoptic $CTRL ;;
  P3-C) run P3-C T06_flag_ganggang_L $ROOT/plenoptic/Plenoptic_long.json $SHARE $CTRL ;;
  P3-H) run P3-H T06_flag_ganggang_L $ROOT/plenoptic/Plenoptic_long.json $SHARE $HEAD ;;
  P4-C) run P4-C T06_flag_ganggang_L540 $ROOT/plenoptic/Plenoptic_long540.json $SHARE $CTRL ;;
  P4-H) run P4-H T06_flag_ganggang_L540 $ROOT/plenoptic/Plenoptic_long540.json $SHARE $HEAD ;;
  P5-M10) run P5-M10 T06_flag_ganggang_L540 $ROOT/plenoptic/Plenoptic_long540.json $SHARE $MEM ;;
  P5-M5) run P5-M5 T06_flag_ganggang $ROOT/plenoptic/Plenoptic.json $SHARE $MEM ;;
  *) echo "usage: $0 P1-C|P1-H|P2-T|P2-P|P3-C|P3-H|P4-C|P4-H|P5-M10|P5-M5"; exit 2 ;;
esac
