#!/usr/bin/env bash
# S2-R7 chain (docs/stage2-R7-prereg.md sections 6-8).  STAGES="sweep check score judge" (default all).
#   sweep  17 one-scene jobs (one GPU job at a time; LOCK=<file> wraps each in flock), each preceded by deleting the
#          arm folder (erratum section 13: --overwrite never clears an output folder), stop on any FAILED scene
#   check  MVSeed/s2r7_check.py (Z1-Z6, GT-free; exit 2 = no verdict)
#   score  eval/eval_jf.py --methods <arm> --out jf_s2r7.json (container)
#   judge  MVSeed/s2r7_judge.py (E8 rule against the E14 headline folder, scored in jf_e14.json)
# Logs: MVSeed/logs/s2r7_*.log, SCSam3/logs/s2r7-*.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/stage1/.venv/bin/python"
STAGES="${STAGES:-sweep check score judge}"
SCENES="${SCENES:-AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast CBABasketball Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF MartialArts Painter PoznanStreet Welder}"
ARM=SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2RmRs
LOG="${ROOT}/MVSeed/logs"; mkdir -p "${LOG}"
ts() { date '+%F %T'; }
has() { [[ " ${STAGES} " == *" $1 "* ]]; }
gpu() { if [[ -n "${LOCK:-}" ]]; then flock "${LOCK}" "$@"; else "$@"; fi; }
folder() { python3 -c "import json;print(json.load(open('${ROOT}/SCSam3/demo/MVSeg.json'))['$1']['folder'])"; }
echo "$(ts) S2-R7 chain start  stages: ${STAGES}  HEAD $(git -C "${ROOT}" rev-parse --short HEAD)  lock ${LOCK:-none}"

if has sweep; then
  for s in ${SCENES}; do
    f=$(folder "${s}") || exit 1
    rm -rf "${ROOT}/Data/MVSeg/${f}/${ARM}"
    out=$(cd "${ROOT}/SCSam3" && gpu env LOGTAG=s2r7 XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod STAGE1=geoset_rep_xs \
          F0SEED=1 CROP=10000 CROPGATE=v2 REFSNAP=1 REFSNAPMEM=1 ./runMVOptThree.sh "${s}" 2>&1)
    st=$?
    echo "${out}" >> "${LOG}/s2r7_sweep.log"
    line=$(echo "${out}" | grep -E "^${s} " | tail -n 1)
    echo "$(ts) ${line}"
    if [[ ${st} -ne 0 || "${line}" != *" ok "* ]]; then echo "$(ts) ABORT: ${s} failed (${line})"; exit 1; fi
  done
fi
if has check; then
  PYTHONDONTWRITEBYTECODE=1 "${PY}" "${ROOT}/MVSeed/s2r7_check.py" > "${LOG}/s2r7_check.log" 2>&1
  st=$?; echo "$(ts) check exit ${st}"; tail -n 3 "${LOG}/s2r7_check.log"
  [[ ${st} -eq 0 ]] || { echo "$(ts) STOP: watch values failed (no verdict; fix, erratum, rerun)"; exit 1; }
fi
if has score; then
  docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${ROOT}" scsam3 \
    python eval/eval_jf.py --methods "${ARM}" --out jf_s2r7.json > "${LOG}/s2r7_score.log" 2>&1
  st=$?; echo "$(ts) score exit ${st}  $(tail -n 1 "${LOG}/s2r7_score.log" | cut -c1-160)"
  [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: scoring failed"; exit 1; }
fi
if has judge; then
  PYTHONDONTWRITEBYTECODE=1 "${PY}" "${ROOT}/MVSeed/s2r7_judge.py" > "${LOG}/s2r7_judge.log" 2>&1
  st=$?; echo "$(ts) judge exit ${st}"; cat "${LOG}/s2r7_judge.log"
fi
echo "$(ts) S2-R7 chain done"
