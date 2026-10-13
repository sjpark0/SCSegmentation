#!/usr/bin/env bash
# S2-R5 chain (docs/stage2-R5-prereg.md section 2): the E8 headline and control with the stage-1
# fine-tuned memory attention in the temporal tracker (--ft-encoder geoset_pred_ft), headline (H-FT)
# first, then control (C-FT); score; judge.  One GPU job at a time.
#   STAGES="sweep score judge" (default all); ONLY="H" restricts the sweeps; SCENES_OVERRIDE="Fencing"
#   restricts the scenes (smoke runs only -- the registered run is all 17).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$(cd "${ROOT}/.." && pwd)/SCSegmentation2D"
STAGES="${STAGES:-sweep score judge}"
# a scene subset is a smoke run: sweep only, so it never writes the registered jf_s2r5.json / judge file
[[ -n "${SCENES_OVERRIDE:-}" ]] && STAGES="sweep"
SCENES="${SCENES_OVERRIDE:-AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast CBABasketball Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF MartialArts Painter PoznanStreet Welder}"
ts() { date '+%F %T'; }
has() { [[ " ${STAGES} " == *" $1 "* ]]; }
echo "$(ts) S2-R5 chain start  stages: ${STAGES}  scenes: ${SCENES}"
echo "$(ts) 2D repo HEAD $(git -C "${SRC}" rev-parse --short HEAD)  checkpoints: $(sha256sum "${SRC}"/geomem/runs/geoset_pred_ft/fold*/ckpt.pt | cut -c1-12 | tr '\n' ' ')"
echo "$(ts) this repo HEAD $(git -C "${ROOT}" rev-parse --short HEAD)  dirty: $(git -C "${ROOT}" status --porcelain --untracked-files=no | wc -l)"
HEAD_ENV="XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod F0SEED=1 CROP=10000 CROPGATE=v2"
CTRL_ENV="XVIEW=0 XREF=muvod F0SEED=1"
RUNLIST="H:H C:C"
if has sweep; then
  for item in ${RUNLIST}; do
    IFS=: read -r rid kind <<< "${item}"
    [[ -n "${ONLY:-}" && " ${ONLY} " != *" ${rid} "* ]] && continue
    envs="${CTRL_ENV}"; [[ ${kind} == H ]] && envs="${HEAD_ENV}"
    echo "$(ts) sweep ${rid}-FT"
    ( cd "${ROOT}/SCSam3" && env ${envs} SEEDS="MVSeed_geoset_rep" FTENC=geoset_pred_ft LOGTAG="s2r5-${rid,,}" ./runMVOptThree.sh ${SCENES} ) 2>&1 | sed "s/^/    /"
  done
fi
METHODS="SegMaskSam3XW1CGPS4MSdgeoset_repFsFtgpCs10kG2 SegMaskSam3XW0MSdgeoset_repFsFtgp"
if has score; then
  echo "$(ts) score"
  docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${ROOT}" scsam3 \
    python eval/eval_jf.py --methods ${METHODS} --out jf_s2r5.json > "${ROOT}/MVSeed/logs/s2r5_score.log" 2>&1
  st=$?; echo "$(ts) score exit ${st}  $(tail -n 2 "${ROOT}/MVSeed/logs/s2r5_score.log" | tr '\n' ' ' | cut -c1-200)"
fi
if has judge; then
  python3 "${ROOT}/MVSeed/s2r5_judge.py" > "${ROOT}/MVSeed/logs/s2r5_judge.log" 2>&1
  st=$?; echo "$(ts) judge exit ${st}"
fi
echo "$(ts) S2-R5 chain done"
