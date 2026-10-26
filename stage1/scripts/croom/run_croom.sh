#!/usr/bin/env bash
# CROOM ring ground-truth runs (docs/croom-annotation.md 6).  One annotated instant x one reference camera = one scene
# folder Data/CROOM_<T>_<frame>_r<ref>/ (scripts/croom/build_scene.py).
#   run_croom.sh build <take> <frame> <ref|auto> [fold]  scene folder(s) (auto = the four quadrant references, build_scene.py;
#                                                     fold = external_fold, default 2 as T06)
#   run_croom.sh pi3 <scene>                          protocol registration (2026-10-27): Pi3X on the 32 images + the 32 input
#                                                     images of the calibration instant T06 #900 (pi3_multi.py, tag t06; only the
#                                                     target cameras' geometry is kept) -> Cache/pi3 -> pi3_multi_t06, then
#                                                     pi3_reproj masks.  CROOM_ANCHOR overrides the anchor scene.
#   run_croom.sh pi3single <scene>                    single-instant Pi3 geometry (the pre-10-27 convention; diagnostics only)
#   run_croom.sh pi3skip <scene>                      pi3_reproj masks from an existing Cache/pi3 (geometry variants)
#   run_croom.sh decode <scene> AB|C0|G0              AB = E15 main preset (Fix A c* + relay 60), C0 = geoset_rep_xs, G0 = sam3_prop
#   run_croom.sh score <scene>                        -> geomem/runs/_croom/scores/<scene>.json (every decoded arm)
#   run_croom.sh prep <take> <frame>                  build auto -> pi3 for the four references (input only; allowed before TEST)
#   run_croom.sh dev <take> <frame> <ref>             build -> pi3 -> AB, C0, G0 -> score   (the DEV instant T04 #1000)
#   run_croom.sh test <take> <frame>                  decode AB, C0, G0 + score on the four prepared scenes (TEST: only after
#                                                     check.py --manifest is committed, docs/croom-annotation.md 5)
# GPU jobs take the E15 lock (geomem/runs/_e15/gpu.lock).  Logs: geomem/runs/_croom/logs/.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$REPO/geomem/runs/_croom
PY=$REPO/.venv/bin/python
ANCHOR=${CROOM_ANCHOR:-T06_flag_ganggang_F}      # calibration instant for the joint registration (T06 #900, input images only)
mkdir -p "$WORK/logs" "$WORK/scores"
ts() { date '+%F %T'; }
gpu_lock() { exec 9>"$REPO/geomem/runs/_e15/gpu.lock"; flock 9; }
cmd=$1; shift
case "$cmd" in
  build)
    take=$1; frame=$2; ref=$3; fold=${4:-2}
    "$PY" "$REPO/scripts/croom/build_scene.py" --take "$take" --frame "$frame" --ref "$ref" --fold "$fold" ;;
  pi3skip)
    scene=$1; gpu_lock
    "$REPO/scripts/run_pi3_reproj.sh" "$scene" --skip-geometry --overwrite ;;
  pi3)
    scene=$1; gpu_lock; cache=$REPO/Data/$scene/Cache; mkdir -p "$cache"
    "$REPO/scripts/docker_run.sh" scgeo:pi3 python scripts/croom/pi3_multi.py --scene "$scene" --tag t06 --anchor "scene:$ANCHOR"
    if [ -d "$cache/pi3" ] && [ ! -L "$cache/pi3" ]; then mv "$cache/pi3" "$cache/pi3_single"; fi
    ln -sfn pi3_multi_t06 "$cache/pi3"
    "$REPO/scripts/run_pi3_reproj.sh" "$scene" --skip-geometry --overwrite ;;
  pi3single)
    scene=$1; gpu_lock
    "$REPO/scripts/run_pi3_reproj.sh" "$scene" --overwrite ;;
  decode)
    scene=$1; arm=$2; gpu_lock
    case "$arm" in
      AB) "$REPO/scripts/docker_run.sh" scsam3 python methods/geomem/run.py --scene "$scene" --out croom_AB --arm geoset_pred_ft \
            --gate reproj --xcl outs --relay 60 --id3d cstar --overwrite ;;
      C0) "$REPO/scripts/docker_run.sh" scsam3 python methods/geomem/run.py --scene "$scene" --out croom_C0 --arm geoset_pred_ft \
            --gate reproj --xcl outs --overwrite ;;
      G0) "$REPO/scripts/run_sam3_prop.sh" "$scene" --out croom_G0 --overwrite ;;
      *) echo "unknown arm $arm" >&2; exit 2 ;;
    esac ;;
  score)
    scene=$1; res=$REPO/Data/$scene/Result; args=()
    for d in "$res"/croom_*; do [ -d "$d" ] || continue; args+=("$(basename "$d" | sed 's/^croom_//')=$d"); done
    rm -f "$WORK/scores/$scene.json"
    "$PY" "$REPO/scripts/croom/score.py" --scene "$scene" --out "$WORK/scores/$scene.json" "${args[@]}" ;;
  prep)
    take=$1; frame=$2
    "$0" build "$take" "$frame" auto | tee -a "$WORK/logs/prep_${take%%_*}_$(printf %06d "$frame").log"
    for d in "$REPO"/Data/CROOM_${take%%_*}_$(printf %06d "$frame")_r[0-9][0-9]; do
      scene=$(basename "$d"); echo "$(ts) pi3 $scene"
      "$0" pi3 "$scene" > "$WORK/logs/pi3_$scene.log" 2>&1 || { echo "$(ts) FAILED pi3 $scene"; exit 1; }
    done; echo "$(ts) prep done" ;;
  test)
    take=$1; frame=$2
    for d in "$REPO"/Data/CROOM_${take%%_*}_$(printf %06d "$frame")_r[0-9][0-9]; do
      scene=$(basename "$d"); [ -d "$d/Result/pi3_reproj" ] || { echo "$scene: run prep first" >&2; exit 1; }
      for arm in AB C0 G0; do
        echo "$(ts) decode $scene $arm"; "$0" decode "$scene" "$arm" > "$WORK/logs/decode_${scene}_$arm.log" 2>&1 || { echo "$(ts) FAILED decode $arm"; exit 1; }
      done
      echo "$(ts) score $scene"; "$0" score "$scene"
    done; echo "$(ts) test ${take%%_*} $frame done" ;;
  dev)
    take=$1; frame=$2; ref=$3
    scene="CROOM_${take%%_*}_$(printf %06d "$frame")_r$(printf %02d "$ref")"
    echo "$(ts) build $scene"; "$0" build "$take" "$frame" "$ref"
    echo "$(ts) pi3 $scene";   "$0" pi3 "$scene"   > "$WORK/logs/pi3_$scene.log" 2>&1 || { echo "$(ts) FAILED pi3"; exit 1; }
    for arm in AB C0 G0; do
      echo "$(ts) decode $scene $arm"; "$0" decode "$scene" "$arm" > "$WORK/logs/decode_${scene}_$arm.log" 2>&1 || { echo "$(ts) FAILED decode $arm"; exit 1; }
    done
    echo "$(ts) score $scene"; "$0" score "$scene"; echo "$(ts) dev $scene done" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
