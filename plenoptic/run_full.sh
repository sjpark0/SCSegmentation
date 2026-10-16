#!/usr/bin/env bash
# Full-length CROOM T06 runs (docs/plenoptic-F.md): every frame 0..9937 of all 32 cameras at 3840x2160, results on the NAS.
#   STAGES (default all, in order): sub_F sub_B sub_merge sub_score sub_sync sub_render lat_F lat_B lat_merge lat_score
#                                   lat_sync lat_render sub_clean lat_clean src_clean
#     <cfg>_F      forward run, source frames 900..9937 (scene T06_flag_ganggang_F, start_frame 900, num_frame 9038)
#     <cfg>_B      backward run, source frames 900..0 time-reversed (scene T06_flag_ganggang_B, start 0, num 901)
#     <cfg>_merge  hard-link B's frames into F's tree under source frame numbers 0..899 (plenoptic/build_full.py merge_back)
#     <cfg>_score  frame-900 J/F with the frozen scorer (eval/eval_jf.py, container) + plenoptic/score_full.py readings
#                  -> plenoptic/full/{jf,score}_<cfg>.json (descriptive; a failure is logged and the chain goes on)
#     <cfg>_sync   copy the merged method folder to the NAS (16 parallel rsync) and verify its content digest there
#     <cfg>_render start plenoptic/render_full.py in the background (CPU): overlay MP4s, ring mosaic, contact sheet and
#                  area timeline straight to <NAS>/viz/<method>/; it writes done/<cfg>_render_finished
#     <cfg>_clean  wait for that render, then delete the local result folders (the results live on the NAS)
#     src_clean    after both renders: delete the local copies of the source frames
#   cfg: sub = the TCSVT-submitted configuration, lat = the latest (S1-E14 stage 1 + S2-R6 snap).
#   LOCK=<file> wraps each GPU run in flock; a stage with a done marker is skipped (rerun = resume by stage).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="${P:-${ROOT}/Data/PlenopticFull}"
NAS="${NAS:-${ROOT}/PlenopticServer1/CROOM/SegResult/T06_flag_ganggang_F}"
CONF="${ROOT}/plenoptic/Plenoptic_full.json"
STAGES="${STAGES:-sub_F sub_B sub_merge sub_score sub_sync sub_render lat_F lat_B lat_merge lat_score lat_sync lat_render sub_clean lat_clean src_clean}"
LOGS="${P}/logs"; mkdir -p "${LOGS}" "${P}/done"
SUB=(--algo MVOpt --xview-window 1 --xview-mode C --xview-gate --xview-ptr --xview-tpos-shift 4 --ref-cam muvod
     --frame0-seed --crop-small 10000 --crop-gate v2)
LAT=("${SUB[@]}" --stage1 geoset_rep_xs --ref-snap)
LR=(--mem-prune --io-cache2 --async-png 8 --crop-stream)
declare -A METHOD=([sub]=SegMaskSam3XW1CGPS4MFsCs10kG2 [lat]=SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs)
ts() { date '+%F %T'; }
gpu() { if [[ -n "${LOCK:-}" ]]; then flock "${LOCK}" "$@"; else "$@"; fi; }
mark() { touch "${P}/done/$1"; }
done_() { [[ -f "${P}/done/$1" ]]; }
echo "$(ts) full-run chain  stages: ${STAGES}  HEAD $(git -C "${ROOT}" rev-parse --short HEAD)  lock ${LOCK:-none}"

# the forward runs read the local copies of the source frames: refuse to start before the staging is complete
check_staged() {
  python3 - "${P}" <<'EOF'
import os, sys
p = sys.argv[1]
bad = [c for c in sorted(os.listdir(os.path.join(p, "src")))
       if len([f for f in os.listdir(os.path.join(p, "src", c)) if f.endswith(".jpg")]) != 9938]
print("staged:", "complete" if not bad else f"incomplete {bad[:4]}")
sys.exit(1 if bad or len(os.listdir(os.path.join(p, "src"))) != 32 else 0)
EOF
}

run_one() {   # <cfg> <F|B>
  local cfg=$1 dir=$2 scene="T06_flag_ganggang_$2" st
  local -a A
  [[ ${cfg} == sub ]] && A=("${SUB[@]}") || A=("${LAT[@]}")
  check_staged || { echo "$(ts) ABORT: source frames not fully staged"; exit 1; }
  local img; img="$(docker images --no-trunc -q scsam3:latest 2>/dev/null | head -n1)"
  echo "$(ts) ${cfg}_${dir}: ${METHOD[$cfg]} on ${scene}"
  gpu docker run --rm --name "croom_${cfg}_${dir}" --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
    -e SCSAM3_TRIM_CACHED_OUTPUTS=1 -e SCSAM3_IMAGE_ID="${img}" -e PYTHONDONTWRITEBYTECODE=1 \
    -v /:/host -w "/host${ROOT}/SCSam3" scsam3 \
    python -u runMVSeg.py "${scene}" --config "/host${CONF}" --data-root "/host${P}" "${A[@]}" "${LR[@]}" --overwrite \
    > "${LOGS}/${cfg}_${dir}.log" 2>&1
  st=$?
  docker run --rm -v /:/host scsam3 chown -R "$(id -u):$(id -g)" "/host${P}/${scene}" > /dev/null 2>&1
  echo "$(ts) ${cfg}_${dir} exit ${st}  $(grep -E 'manifest ->|ref snap  |crop small .* kept' "${LOGS}/${cfg}_${dir}.log" | tail -n 3 | tr '\n' ' ' | cut -c1-300)"
  [[ ${st} -eq 0 ]] && grep -q "manifest -> " "${LOGS}/${cfg}_${dir}.log"
}

for stage in ${STAGES}; do
  cfg=${stage%%_*}; what=${stage#*_}
  if done_ "${stage}"; then echo "$(ts) ${stage}: done earlier, skipped"; continue; fi
  if [[ ${stage} == src_clean ]]; then
    until [[ -f "${P}/done/sub_render_finished" && -f "${P}/done/lat_render_finished" ]]; do sleep 60; done
    rm -rf "${P}/src"
    echo "$(ts) src_clean: local source copies removed"; mark "${stage}"; continue
  fi
  case "${what}" in
    F|B) run_one "${cfg}" "${what}" || { echo "$(ts) ABORT: ${stage} failed (log ${LOGS}/${stage}.log)"; exit 1; } ;;
    merge)
      PYTHONDONTWRITEBYTECODE=1 python3 "${ROOT}/plenoptic/build_full.py" merge_back \
        "${P}/T06_flag_ganggang_B/${METHOD[$cfg]}" "${P}/T06_flag_ganggang_F/${METHOD[$cfg]}" \
        > "${LOGS}/${stage}.log" 2>&1 || { echo "$(ts) ABORT: merge failed"; exit 1; }
      cp "${P}/T06_flag_ganggang_B/${METHOD[$cfg]}/MANIFEST.json" "${P}/T06_flag_ganggang_F/${METHOD[$cfg]}/MANIFEST_backward.json"
      echo "$(ts) ${stage}: $(tail -n 1 "${LOGS}/${stage}.log")" ;;
    score)
      mkdir -p "${ROOT}/plenoptic/full"
      if docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${ROOT}" scsam3 \
           python eval/eval_jf.py --root "/host${P}" --config "/host${CONF}" --methods "${METHOD[$cfg]}" --jobs 16 \
           --out "/host${ROOT}/plenoptic/full/jf_${cfg}.json" T06_flag_ganggang_F > "${LOGS}/${stage}.log" 2>&1 &&
         PYTHONDONTWRITEBYTECODE=1 python3 "${ROOT}/plenoptic/score_full.py" --jf "${ROOT}/plenoptic/full/jf_${cfg}.json" \
           --method "${METHOD[$cfg]}" --fwd "${P}/T06_flag_ganggang_F/${METHOD[$cfg]}" \
           --bwd "${P}/T06_flag_ganggang_B/${METHOD[$cfg]}" --out "${ROOT}/plenoptic/full/score_${cfg}.json" \
           >> "${LOGS}/${stage}.log" 2>&1; then
        echo "$(ts) ${stage}: $(tail -n 1 "${LOGS}/${stage}.log")"
      else
        echo "$(ts) WARN: ${stage} failed (log ${LOGS}/${stage}.log); results untouched, chain goes on"; continue
      fi ;;
    sync)
      src="${P}/T06_flag_ganggang_F/${METHOD[$cfg]}"; dst="${NAS}/${METHOD[$cfg]}"
      mkdir -p "${dst}"
      for f in SOURCE.json; do cp "${P}/T06_flag_ganggang_F/${f}" "${NAS}/${f}"; done
      cp -r "${P}/T06_flag_ganggang_F/Mask" "${NAS}/" 2>/dev/null
      cp "${P}/T06_flag_ganggang_B/SOURCE.json" "${NAS}/SOURCE_backward.json"
      mkdir -p "${NAS}/score"; cp "${ROOT}"/plenoptic/full/*_"${cfg}".json "${NAS}/score/" 2>/dev/null
      (cd "${src}" && ls -d cam_* | xargs -P 16 -I{} rsync -rt {} "${dst}/") > "${LOGS}/${stage}.log" 2>&1
      rsync -t "${src}"/MANIFEST*.json "${dst}/" >> "${LOGS}/${stage}.log" 2>&1
      EVAL_DIR="${ROOT}/eval" PYTHONDONTWRITEBYTECODE=1 python3 - "${src}" "${dst}" >> "${LOGS}/${stage}.log" 2>&1 <<'EOF'
import os, sys
sys.path.insert(0, os.environ["EVAL_DIR"])
import manifest as M
a, b = M.content_digest(sys.argv[1], jobs=16), M.content_digest(sys.argv[2], jobs=16)
print("local", a[0], len(a[1]), "nas", b[0], len(b[1]))
sys.exit(0 if a[0] == b[0] and len(a[1]) == len(b[1]) else 3)
EOF
      st=$?
      echo "$(ts) ${stage}: verify exit ${st}  $(tail -n 1 "${LOGS}/${stage}.log")"
      [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: NAS copy differs; local copy kept"; exit 1; } ;;
    render)
      setsid nohup bash -c "nice -n 10 '${ROOT}/stage1/.venv/bin/python' '${ROOT}/plenoptic/render_full.py' \
          '${P}/T06_flag_ganggang_F' '${METHOD[$cfg]}' '${NAS}/viz/${METHOD[$cfg]}' --workers ${RENDER_WORKERS:-12} \
          && touch '${P}/done/${cfg}_render_finished'" > "${LOGS}/${stage}.log" 2>&1 < /dev/null &
      echo "$(ts) ${stage}: started in the background (log ${LOGS}/${stage}.log)" ;;
    clean)
      until [[ -f "${P}/done/${cfg}_render_finished" ]]; do sleep 60; done
      rm -rf "${P}/T06_flag_ganggang_F/${METHOD[$cfg]}" "${P}/T06_flag_ganggang_B/${METHOD[$cfg]}"
      echo "$(ts) ${stage}: local result folders removed (NAS copy verified, render finished)" ;;
  esac
  mark "${stage}"
done
echo "$(ts) full-run chain done"
