#!/usr/bin/env bash
# CROOM ring ground-truth runs (docs/croom-annotation.md 6).  One annotated instant x one reference camera = one scene
# folder Data/CROOM_<T>_<frame>_r<ref>/ (scripts/croom/build_scene.py).
#   run_croom.sh build <take> <frame> <ref> [fold]    scene folder (fold = external_fold, default 2 as T06)
#   run_croom.sh pi3 <scene>                          Pi3 geometry cache (scgeo:pi3) + pi3_reproj masks
#   run_croom.sh decode <scene> AB|C0|G0              AB = E15 main preset (Fix A c* + relay 60), C0 = geoset_rep_xs, G0 = sam3_prop
#   run_croom.sh score <scene>                        -> geomem/runs/_croom/scores/<scene>.json (every decoded arm)
#   run_croom.sh dev <take> <frame> <ref>             build -> pi3 -> AB, C0, G0 -> score   (the DEV instant T04 #1000)
# GPU jobs take the E15 lock (geomem/runs/_e15/gpu.lock).  Logs: geomem/runs/_croom/logs/.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$REPO/geomem/runs/_croom
PY=$REPO/.venv/bin/python
mkdir -p "$WORK/logs" "$WORK/scores"
ts() { date '+%F %T'; }
gpu_lock() { exec 9>"$REPO/geomem/runs/_e15/gpu.lock"; flock 9; }
cmd=$1; shift
case "$cmd" in
  build)
    take=$1; frame=$2; ref=$3; fold=${4:-2}
    "$PY" "$REPO/scripts/croom/build_scene.py" --take "$take" --frame "$frame" --ref "$ref" --fold "$fold" ;;
  pi3)
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
