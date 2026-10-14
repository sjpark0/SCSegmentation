#!/usr/bin/env bash
# S1-E9 chain (docs/stage1-E9-prereg.md): train the two decoder arms on all five folds, decode the 17 scenes with
# the reproj gate, score the keyframe (stage1/mvseg2d.evaluate) against the adopted arm, judge the stage-1 gate;
# then, for the arms that passed, the in-process stage-2 headline sweep, scoring and the stage-2 judgement.
#   STAGES="train decode eval judge1 sweep score judge2" (default: the stage-1 half "train decode eval judge1");
#   ARMS="geoset_dec_ft geoset_decb_ft"; TAGS="geoset_dec_rep geoset_decb_rep" (sweep: default = judge1's passing tags).
# One GPU job at a time.  Logs: MVSeed/logs/e9_*.log, SCSam3/logs/e9-*.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S1="${ROOT}/stage1"
STAGES="${STAGES:-train decode eval judge1}"
ARMS="${ARMS:-geoset_dec_ft geoset_decb_ft}"
SCENES="AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast CBABasketball Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF MartialArts Painter PoznanStreet Welder"
OUTDIR="${ROOT}/MVSeed/runs/e9"; mkdir -p "${OUTDIR}" "${ROOT}/MVSeed/logs"
ts() { date '+%F %T'; }
has() { [[ " ${STAGES} " == *" $1 "* ]]; }
declare -A TAG_OF=([geoset_dec_ft]=geoset_dec_rep [geoset_decb_ft]=geoset_decb_rep)
declare -A ARM_OF=([geoset_dec_rep]=geoset_dec_ft [geoset_decb_rep]=geoset_decb_ft)
echo "$(ts) E9 chain start  stages: ${STAGES}  arms: ${ARMS}  HEAD $(git -C "${ROOT}" rev-parse --short HEAD)"

if has train; then
  for arm in ${ARMS}; do
    for k in 0 1 2 3 4; do
      if [[ -f "${S1}/geomem/runs/${arm}/fold${k}/ckpt.pt" && -f "${S1}/geomem/runs/${arm}/fold${k}/config.json" ]]; then
        echo "$(ts) train ${arm} fold${k}: ckpt.pt + config.json exist, skip"; continue; fi     # config.json is written after the checkpoint
      echo "$(ts) train ${arm} fold${k}"
      ( cd "${S1}" && scripts/run_geomem_train.sh "${arm}" "${k}" ) > "${ROOT}/MVSeed/logs/e9_train_${arm}_fold${k}.log" 2>&1
      st=$?; echo "$(ts) train ${arm} fold${k} exit ${st}  ($(tail -n 1 "${ROOT}/MVSeed/logs/e9_train_${arm}_fold${k}.log" | cut -c1-160))"
      [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: training failed"; exit 1; }
    done
  done
fi
if has decode; then
  for arm in ${ARMS}; do
    echo "$(ts) decode ${arm} (reproj gate) -> ${arm}_rep_e9 / ${arm}_r1_e9"
    ( cd "${S1}" && scripts/run_geomem.sh all --arm "${arm}" --gate reproj --out "${arm}_rep_e9" --r1-out "${arm}_r1_e9" --overwrite ) \
      > "${ROOT}/MVSeed/logs/e9_decode_${arm}.log" 2>&1
    st=$?; nok=$(grep -c ' ok$' "${ROOT}/MVSeed/logs/e9_decode_${arm}.log")
    echo "$(ts) decode ${arm} exit ${st}  (${nok} scenes ok)"
    [[ ${st} -eq 0 && ${nok} -eq 17 ]] || { echo "$(ts) ABORT: decode failed (run_all records a per-scene error and exits 0)"; exit 1; }
  done
fi
if has eval; then
  for name in geoset_pred_ft_rep_e8 $(for arm in ${ARMS}; do echo "${arm}_rep_e9"; done); do
    ( cd "${S1}" && .venv/bin/python -m mvseg2d.evaluate --data Data --method "${name}" --out "${OUTDIR}/kf_${name}.json" ) \
      > "${ROOT}/MVSeed/logs/e9_eval_${name}.log" 2>&1
    st=$?; echo "$(ts) eval ${name} exit ${st}  $(tail -n 1 "${ROOT}/MVSeed/logs/e9_eval_${name}.log" | cut -c1-160)"
    [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: eval failed"; exit 1; }
  done
  ( cd "${S1}" && .venv/bin/python -m mvseg2d.report "${OUTDIR}"/kf_*.json --out "${OUTDIR}/kf_report.md" ) > /dev/null 2>&1
  for name in $(for arm in ${ARMS}; do echo "${arm}_rep_e9"; done); do        # seed anatomy per arm (docs/stage1-seedfail.md features)
    ( cd "${OUTDIR}" && docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${OUTDIR}" scsam3 \
        python "/host${ROOT}/MVSeed/refloss/seedfail.py" "s1:${name}" "sf_${name}.json" ) > /dev/null 2>&1
  done
fi
if has judge1; then
  ( cd "${S1}" && .venv/bin/python "${ROOT}/MVSeed/e9_judge.py" --stage 1 ) > "${ROOT}/MVSeed/logs/e9_judge1.log" 2>&1
  st=$?; echo "$(ts) judge1 exit ${st}"; cat "${ROOT}/MVSeed/logs/e9_judge1.log"
  [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: judge1 failed"; exit 1; }
fi
if has sweep; then
  TAGS="${TAGS-$(python3 -c "import json;print(' '.join(json.load(open('${OUTDIR}/e9_judge.json'))['stage1']['passing_tags']))")}" || exit 1
  echo "$(ts) sweep tags: ${TAGS:-<none: no arm passed the stage-1 gate>}"
  for tag in ${TAGS}; do
    echo "$(ts) sweep ${tag} (headline, in-process, ref-snap)"
    ( cd "${ROOT}/SCSam3" && env XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod STAGE1="${tag}" F0SEED=1 CROP=10000 CROPGATE=v2 REFSNAP=1 \
        LOGTAG="e9-${tag}" ./runMVOptThree.sh ${SCENES} ) 2>&1 | sed "s/^/    /"
  done
fi
if has score; then
  TAGS="${TAGS-$(python3 -c "import json;print(' '.join(json.load(open('${OUTDIR}/e9_judge.json'))['stage1']['passing_tags']))")}" || exit 1
  [[ -n "${TAGS}" ]] || { echo "$(ts) score: no passing tag, nothing to score"; exit 0; }
  METHODS=$(for tag in ${TAGS}; do echo "SegMaskSam3XW1CGPS4MSd${tag}FsCs10kG2Rs"; done)
  echo "$(ts) score ${METHODS}"
  docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${ROOT}" scsam3 \
    python eval/eval_jf.py --methods ${METHODS} --out jf_e9.json > "${ROOT}/MVSeed/logs/e9_score.log" 2>&1
  st=$?; echo "$(ts) score exit ${st}  $(tail -n 2 "${ROOT}/MVSeed/logs/e9_score.log" | tr '\n' ' ' | cut -c1-200)"
fi
if has judge2; then
  ( cd "${S1}" && .venv/bin/python "${ROOT}/MVSeed/e9_judge.py" --stage 2 ) > "${ROOT}/MVSeed/logs/e9_judge2.log" 2>&1   # numpy for Z3
  st=$?; echo "$(ts) judge2 exit ${st}"; cat "${ROOT}/MVSeed/logs/e9_judge2.log"
fi
echo "$(ts) E9 chain done"
