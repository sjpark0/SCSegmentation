#!/usr/bin/env bash
# Library cart take, TCSVT-submitted configuration only (docs/cart.md): source frames 120..2699 of all 24 cameras at
# 1920x1080, prompt = the user-reviewed 17-object mask of cam0005 at frame 120 (Data/Cart/review/f120/).
#   waits for the JPEG staging (24 x 2700 frames), builds the scene, runs, chowns, writes Data/Cart/done/run_f120.
#   LOCK=<file> wraps the GPU run in flock.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="${ROOT}/Data/Cart"; KEY=cart_f120; START=120; NUM=2580
SUB=(--algo MVOpt --xview-window 1 --xview-mode C --xview-gate --xview-ptr --xview-tpos-shift 4 --ref-cam muvod
     --frame0-seed --crop-small 10000 --crop-gate v2)
LR=(--mem-prune --io-cache2 --async-png 8 --crop-stream)
ts() { date '+%F %T'; }
mkdir -p "${C}/logs" "${C}/done"
echo "$(ts) cart run: ${KEY} (source ${START}..$((START + NUM - 1))), HEAD $(git -C "${ROOT}" rev-parse --short HEAD)"
until python3 - "${C}/src" <<'EOF'
import os, sys
s = sys.argv[1]
ok = len(os.listdir(s)) == 24 and all(len([f for f in os.listdir(os.path.join(s, c)) if f.endswith(".jpg") and ".part" not in f]) == 2700
                                      for c in os.listdir(s))
sys.exit(0 if ok else 1)
EOF
do sleep 60; done
echo "$(ts) staging complete"
PYTHONDONTWRITEBYTECODE=1 python3 "${ROOT}/cart/build_cart.py" scene --key "${KEY}" --start "${START}" --num "${NUM}"
mkdir -p "${C}/${KEY}/Mask/cam0005"
cp "${C}/review/f120/cam_ref.png" "${C}/${KEY}/Mask/cam0005/000000.png"
cp "${C}/cart_f120_probe/Mask/objects_labels.json" "${C}/${KEY}/Mask/objects_labels.json"
img="$(docker images --no-trunc -q scsam3:latest 2>/dev/null | head -n1)"
gpu() { if [[ -n "${LOCK:-}" ]]; then flock "${LOCK}" "$@"; else "$@"; fi; }
gpu docker run --rm --name "cart_${KEY}" --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
  -e SCSAM3_TRIM_CACHED_OUTPUTS=1 -e SCSAM3_IMAGE_ID="${img}" -e PYTHONDONTWRITEBYTECODE=1 \
  -v /:/host -w "/host${ROOT}/SCSam3" scsam3 \
  python -u runMVSeg.py "${KEY}" --config "/host${C}/cart.json" --data-root "/host${C}" "${SUB[@]}" "${LR[@]}" --overwrite \
  > "${C}/logs/run_f120.log" 2>&1
st=$?
docker run --rm -v /:/host scsam3 chown -R "$(id -u):$(id -g)" "/host${C}/${KEY}" > /dev/null 2>&1
echo "$(ts) run exit ${st}  $(grep -E 'manifest -> |crop small .* kept' "${C}/logs/run_f120.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
[[ ${st} -eq 0 ]] && grep -q "manifest -> " "${C}/logs/run_f120.log" && touch "${C}/done/run_f120"
