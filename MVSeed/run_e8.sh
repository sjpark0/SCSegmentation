#!/usr/bin/env bash
# S1-E8 chain (docs/stage1-E8-prereg.md section 2-3): regenerate the three GeoSet-family stage-1 outputs
# in the sibling repo SCSegmentation2D, convert them to MVSeed_<tag> seed folders, run the six stage-2
# sweeps (headline A2 B2 C2 first, then control A1 B1 C1), score, judge.  One GPU job at a time.
#   STAGES="regen seeds sweep score judge" (default all); TAGS="geoset_abs" restricts regen/seeds;
#   ONLY="A2 B2" restricts the sweeps.  Stage 1 is checked against the 2D repo (MVSeed/e8_regen_check.py)
#   before the sweeps: run STAGES="regen", check, then STAGES="seeds sweep score judge".
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$(cd "${ROOT}/.." && pwd)/SCSegmentation2D"
STAGES="${STAGES:-regen seeds sweep score judge}"
SCENES="AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast CBABasketball Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF MartialArts Painter PoznanStreet Welder"
ts() { date '+%F %T'; }
has() { [[ " ${STAGES} " == *" $1 "* ]]; }
declare -A REGEN=(
  [geoset_rep]="--arm geoset_pred_ft --gate reproj --out geoset_pred_ft_rep_e8 --r1-out geoset_pred_ft_r1_e8"
  [geoset_abs]="--arm geoset_pred_ft --gate absent --out geoset_pred_ft_abs_e8 --r1-out geoset_pred_ft_r1_e8b"
  [geoalign_rep]="--arm geoalign_ft --gate reproj --out geoalign_ft_x_rep_e8 --r1-out geoalign_ft_x_r1_e8")
declare -A OUT=([geoset_rep]=geoset_pred_ft_rep_e8 [geoset_abs]=geoset_pred_ft_abs_e8 [geoalign_rep]=geoalign_ft_x_rep_e8)
TAGS="${TAGS:-geoset_rep geoset_abs geoalign_rep}"
echo "$(ts) E8 chain start  stages: ${STAGES}"
echo "$(ts) 2D repo HEAD $(git -C "${SRC}" rev-parse --short HEAD)  dirty: $(git -C "${SRC}" status --porcelain --untracked-files=no | wc -l)"

if has regen; then
  for t in ${TAGS}; do
    echo "$(ts) regen ${t}: ${REGEN[$t]}"
    ( cd "${SRC}" && scripts/run_geomem.sh all ${REGEN[$t]} ) > "${ROOT}/MVSeed/logs/e8_regen_${t}.log" 2>&1
    st=$?; echo "$(ts) regen ${t} exit ${st}  ($(tail -n 2 "${ROOT}/MVSeed/logs/e8_regen_${t}.log" | tr '\n' ' '))"
    [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: regen failed"; exit 1; }
  done
fi
if has seeds; then
  for t in ${TAGS}; do
    python3 "${ROOT}/MVSeed/build_geoset_seeds.py" --tag "${t}" --out "${OUT[$t]}" > "${ROOT}/MVSeed/logs/e8_seeds_${t}.log" 2>&1
    st=$?; echo "$(ts) seeds ${t} exit ${st}  $(tail -n 1 "${ROOT}/MVSeed/logs/e8_seeds_${t}.log" | cut -c1-200)"
    [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: seed build failed"; exit 1; }
  done
fi
HEAD_ENV="XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod F0SEED=1 CROP=10000 CROPGATE=v2"
CTRL_ENV="XVIEW=0 XREF=muvod F0SEED=1"
RUNLIST="A2:geoset_rep:H B2:geoset_abs:H C2:geoalign_rep:H A1:geoset_rep:C B1:geoset_abs:C C1:geoalign_rep:C"
if has sweep; then
  for item in ${RUNLIST}; do
    IFS=: read -r rid tag kind <<< "${item}"
    [[ -n "${ONLY:-}" && " ${ONLY} " != *" ${rid} "* ]] && continue
    envs="${CTRL_ENV}"; [[ ${kind} == H ]] && envs="${HEAD_ENV}"
    echo "$(ts) sweep ${rid} (${tag}, ${kind})"
    ( cd "${ROOT}/SCSam3" && env ${envs} SEEDS="MVSeed_${tag}" LOGTAG="e8-${rid,,}" ./runMVOptThree.sh ${SCENES} ) 2>&1 | sed "s/^/    /"
  done
fi
METHODS="SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2 SegMaskSam3XW1CGPS4MSdgeoset_absFsCs10kG2 SegMaskSam3XW1CGPS4MSdgeoalign_repFsCs10kG2 SegMaskSam3XW0MSdgeoset_repFs SegMaskSam3XW0MSdgeoset_absFs SegMaskSam3XW0MSdgeoalign_repFs"
if has score; then
  echo "$(ts) score"
  docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${ROOT}" scsam3 \
    python eval/eval_jf.py --methods ${METHODS} --out jf_e8.json > "${ROOT}/MVSeed/logs/e8_score.log" 2>&1
  echo "$(ts) score exit $?  $(tail -n 2 "${ROOT}/MVSeed/logs/e8_score.log" | tr '\n' ' ' | cut -c1-200)"
fi
if has judge; then
  python3 "${ROOT}/MVSeed/e8_judge.py" > "${ROOT}/MVSeed/logs/e8_judge.log" 2>&1
  echo "$(ts) judge exit $?"
fi
echo "$(ts) E8 chain done"
