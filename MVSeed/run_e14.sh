#!/usr/bin/env bash
# S1-E14 chain (docs/stage1-E14-prereg.md section 3).  No training.  One GPU job at a time (LOCK=<file>: each GPU job
# runs under `flock LOCK`).  Stages, in order, each skipped unless listed in STAGES:
#   z4      the adopted path after the code change: keyframe Fencing --xcl out == geoset_pred_ft_rep_x_e11, and DEV-A Dog
#           dev_decode geoset_rep_x (--validate, scratch root) == Data/_dev/geoset_rep_x_A/Dog  (byte comparisons only)
#   decA    dev_decode geoset_rep_xs --split A, dev_score --r1
#   emulA   e14_judge --stage A (Z-emul, invariants on DEV-A) -- commit its result BEFORE any DEV-B stage
#   decB    dev_decode geoset_rep_x / geoset_rep_xs --split B, dev_score --r1 each (DEV-B: once per registration)
#   judgeB  e14_judge --stage B (exit 2 = invariant bug, 3 = pair sets differ, 4 = stop and audit)
#   kf      (only if DEV-B passed) keyframe decode of the arm, mvseg2d.evaluate of C0 and the arm, seedfail anatomy
#   judge1  e14_judge --stage 1
#   sweep   (only if the keyframe gate passed) the in-process headline with STAGE1=geoset_rep_xs
#   score   eval/eval_jf.py --out jf_e14.json
#   judge2  e14_judge --stage 2
# Default STAGES="z4 decA emulA" (everything before DEV-B).  Logs: MVSeed/logs/e14_*.log, SCSam3/logs/e14-*.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S1="${ROOT}/stage1"
PY="${S1}/.venv/bin/python"
STAGES="${STAGES:-z4 decA emulA}"
SCENES="AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast CBABasketball Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF MartialArts Painter PoznanStreet Welder"
OUTDIR="${ROOT}/MVSeed/runs/e14"; LOG="${ROOT}/MVSeed/logs"; mkdir -p "${OUTDIR}" "${LOG}"
JUDGE="${ROOT}/MVSeed/e14_judge.py"
C0=geoset_pred_ft_rep_x_e11; ARM=geoset_pred_ft_rep_xs_e14; ARM_R1=geoset_pred_ft_r1_xs_e14
ts() { date '+%F %T'; }
has() { [[ " ${STAGES} " == *" $1 "* ]]; }
gpu() { if [[ -n "${LOCK:-}" ]]; then flock "${LOCK}" "$@"; else "$@"; fi; }
judge() { PYTHONDONTWRITEBYTECODE=1 "${PY}" "${JUDGE}" "$@"; }
passed() { python3 -c "import json,sys; d=json.load(open('${OUTDIR}/e14_judge.json')); sys.exit(0 if $1 else 1)" 2>/dev/null; }
echo "$(ts) E14 chain start  stages: ${STAGES}  HEAD $(git -C "${ROOT}" rev-parse --short HEAD)  lock ${LOCK:-none}"

cmp_dirs() {   # <a> <b>: PNG bytes of every camera folder identical and the key sets equal
  "${PY}" - "$1" "$2" <<'EOF'
import os, sys
def tree(d):
    out = {}
    for cam in sorted(os.listdir(d)):
        p = os.path.join(d, cam)
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".png"): out[(cam, f)] = open(os.path.join(p, f), "rb").read()
    return out
a, b = tree(sys.argv[1]), tree(sys.argv[2])
same = sum(a[k] == b[k] for k in set(a) & set(b))
print(f"{len(a)} vs {len(b)} PNGs, identical {same}, only-a {len(set(a) - set(b))}, only-b {len(set(b) - set(a))}")
sys.exit(0 if a and set(a) == set(b) and same == len(a) else 1)
EOF
}

dev_decode() {   # <preset> <split> [extra args]
  local preset=$1 split=$2; shift 2
  ( cd "${S1}" && gpu scripts/docker_run.sh scsam3 python -m methods.geomem.dev_decode --preset "${preset}" --split "${split}" "$@" ) \
    > "${LOG}/e14_dec_${preset}_${split}.log" 2>&1
  local st=$? n
  n=$(python3 -c "import json,sys,os; r='${S1}/Data/_dev/${preset}_${split}'; i=json.load(open(r+'/index.json'))['scenes']; print(sum(all(os.path.isfile(f'{r}/{s}/{t:06d}/DONE') for t in v['frames']) for s, v in i.items()))")
  echo "$(ts) dev_decode ${preset} ${split} exit ${st}  (${n}/17 scenes complete)  $(tail -n 1 "${LOG}/e14_dec_${preset}_${split}.log" | cut -c1-120)"
  [[ ${st} -eq 0 && ${n} -eq 17 ]]
}
dev_score() {    # <preset> <split>
  local root="Data/_dev/$1_$2"
  ( cd "${S1}" && gpu scripts/docker_run.sh scsam3 python -m methods.geomem.dev_score --masks "${root}" --out "${root}/dev_pairs.json" --r1 ) \
    > "${LOG}/e14_score_$1_$2.log" 2>&1
  local st=$?; echo "$(ts) dev_score $1 $2 exit ${st}  $(tail -n 1 "${LOG}/e14_score_$1_$2.log" | cut -c1-120)"
  [[ ${st} -eq 0 ]]
}

if has z4; then
  echo "$(ts) z4: keyframe Fencing --xcl out vs ${C0}; DEV-A Dog geoset_rep_x vs Data/_dev/geoset_rep_x_A/Dog"
  ( cd "${S1}" && gpu scripts/run_geomem.sh Fencing --arm geoset_pred_ft --gate reproj --xcl out --out z4e14_check --r1-out z4e14_check_r1 --overwrite ) \
    > "${LOG}/e14_z4.log" 2>&1
  st=$?; echo "$(ts) z4 keyframe decode exit ${st}  $(tail -n 1 "${LOG}/e14_z4.log" | cut -c1-120)"
  F="${S1}/Data/Fencing/Result"; ok=1
  cmp_dirs "${F}/z4e14_check" "${F}/${C0}" || ok=0
  cmp_dirs "${F}/z4e14_check_r1" "${F}/geoset_pred_ft_r1_x_e11" || ok=0
  for j in round1_conf.json xcl_conf.json; do cmp "${F}/z4e14_check/${j}" "${F}/${C0}/${j}" && echo "${j} identical" || ok=0; done
  Z="${S1}/Data/_dev/_z4e14_geoset_rep_x_A"
  ( cd "${S1}" && gpu scripts/docker_run.sh scsam3 python -m methods.geomem.dev_decode --preset geoset_rep_x --split A --scenes Dog \
      --root Data/_dev/_z4e14_geoset_rep_x_A --validate --overwrite ) > "${LOG}/e14_z4_dev.log" 2>&1
  st2=$?; echo "$(ts) z4 DEV-A Dog decode exit ${st2}  $(grep VALIDATE "${LOG}/e14_z4_dev.log")"
  grep -q "VALIDATE True" "${LOG}/e14_z4_dev.log" || ok=0
  for t in $(ls "${S1}/Data/_dev/geoset_rep_x_A/Dog"); do
    A="${Z}/Dog/${t}"; B="${S1}/Data/_dev/geoset_rep_x_A/Dog/${t}"
    cmp_dirs "${A}" "${B}" > /dev/null || { echo "z4 DEV Dog ${t}: final PNGs differ"; ok=0; }
    cmp_dirs "${A}/_r1" "${B}/_r1" > /dev/null || { echo "z4 DEV Dog ${t}: round-1 PNGs differ"; ok=0; }
    for j in round1_conf.json xcl_conf.json; do cmp -s "${A}/${j}" "${B}/${j}" || { echo "z4 DEV Dog ${t}: ${j} differs"; ok=0; }; done
  done
  [[ ${st} -eq 0 && ${st2} -eq 0 && ${ok} -eq 1 ]] || { echo "$(ts) ABORT: Z4 failed (the adopted path is no longer byte-identical)"; exit 1; }
  echo "$(ts) z4 OK"
fi
if has decA; then
  dev_decode geoset_rep_xs A || { echo "$(ts) ABORT: DEV-A arm decode failed"; exit 1; }
  dev_score geoset_rep_xs A || { echo "$(ts) ABORT: DEV-A arm scoring failed"; exit 1; }
fi
if has emulA; then
  judge --stage A > "${LOG}/e14_judgeA.log" 2>&1
  st=$?; echo "$(ts) judge A exit ${st}"; cat "${LOG}/e14_judgeA.log"
  [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: Z-emul / DEV-A invariants failed"; exit 1; }
fi
if has decB; then
  passed "d.get('stageA', {}).get('ok')" || { echo "$(ts) ABORT: stage A is not OK in e14_judge.json"; exit 1; }
  dev_decode geoset_rep_x B || { echo "$(ts) ABORT: DEV-B base decode failed"; exit 1; }
  dev_decode geoset_rep_xs B || { echo "$(ts) ABORT: DEV-B arm decode failed"; exit 1; }
  dev_score geoset_rep_x B || { echo "$(ts) ABORT: DEV-B base scoring failed"; exit 1; }
  dev_score geoset_rep_xs B || { echo "$(ts) ABORT: DEV-B arm scoring failed"; exit 1; }
fi
if has judgeB; then
  judge --stage B ${AUDITED:+--audited} > "${LOG}/e14_judgeB.log" 2>&1
  st=$?; echo "$(ts) judge B exit ${st}"; cat "${LOG}/e14_judgeB.log"
  [[ ${st} -eq 0 ]] || { echo "$(ts) STOP: judge B exit ${st} (2 = invariant bug, 3 = pair sets differ, 4 = stop and audit)"; exit 1; }
fi
if has kf; then
  passed "d['stageB']['rule']['pass']" || { echo "$(ts) DEV-B did not pass: S1-E14 closed, no keyframe run"; exit 0; }
  echo "$(ts) kf decode --xcl outs -> ${ARM} / ${ARM_R1}"
  ( cd "${S1}" && gpu scripts/run_geomem.sh all --arm geoset_pred_ft --gate reproj --xcl outs --out "${ARM}" --r1-out "${ARM_R1}" --overwrite ) \
    > "${LOG}/e14_kf_decode.log" 2>&1
  st=$?; nok=$(grep -c ' ok$' "${LOG}/e14_kf_decode.log")
  echo "$(ts) kf decode exit ${st}  (${nok} scenes ok)"
  [[ ${st} -eq 0 && ${nok} -eq 17 ]] || { echo "$(ts) ABORT: keyframe decode failed"; exit 1; }
  for name in ${C0} ${ARM}; do
    ( cd "${S1}" && "${PY}" -m mvseg2d.evaluate --data Data --method "${name}" --out "${OUTDIR}/kf_${name}.json" ) > "${LOG}/e14_eval_${name}.log" 2>&1
    st=$?; echo "$(ts) eval ${name} exit ${st}  $(tail -n 1 "${LOG}/e14_eval_${name}.log" | cut -c1-160)"
    [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: eval failed"; exit 1; }
    ( cd "${OUTDIR}" && gpu docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${OUTDIR}" scsam3 \
        python "/host${ROOT}/MVSeed/refloss/seedfail.py" "s1:${name}" "sf_${name}.json" ) > "${LOG}/e14_sf_${name}.log" 2>&1
    echo "$(ts) seedfail ${name} exit $?"
  done
  ( cd "${S1}" && "${PY}" -m mvseg2d.report "${OUTDIR}"/kf_*.json --out "${OUTDIR}/kf_report.md" ) > /dev/null 2>&1
fi
if has judge1; then
  passed "d['stageB']['rule']['pass']" || { echo "$(ts) DEV-B did not pass: no keyframe judgement"; exit 0; }
  judge --stage 1 ${AUDITED:+--audited} > "${LOG}/e14_judge1.log" 2>&1
  st=$?; echo "$(ts) judge1 exit ${st}"; cat "${LOG}/e14_judge1.log"
  [[ ${st} -eq 0 ]] || { echo "$(ts) STOP: judge1 exit ${st} (2 = Z1/Z2 failed, 4 = stop and audit)"; exit 1; }
fi
if has sweep; then
  passed "d['stage1']['passing_tags']" || { echo "$(ts) the keyframe gate did not pass: no stage 2"; exit 0; }
  echo "$(ts) sweep geoset_rep_xs (headline, in-process, ref-snap)"
  ( cd "${ROOT}/SCSam3" && gpu env XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod STAGE1=geoset_rep_xs F0SEED=1 CROP=10000 CROPGATE=v2 REFSNAP=1 \
      LOGTAG="e14-geoset_rep_xs" ./runMVOptThree.sh ${SCENES} ) 2>&1 | sed "s/^/    /"
fi
if has score; then
  passed "d['stage1']['passing_tags']" || { echo "$(ts) score: nothing to score"; exit 0; }
  docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${ROOT}" scsam3 \
    python eval/eval_jf.py --methods SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs --out jf_e14.json > "${LOG}/e14_score.log" 2>&1
  st=$?; echo "$(ts) score exit ${st}  $(tail -n 2 "${LOG}/e14_score.log" | tr '\n' ' ' | cut -c1-200)"
fi
if has judge2; then
  judge --stage 2 > "${LOG}/e14_judge2.log" 2>&1
  st=$?; echo "$(ts) judge2 exit ${st}"; cat "${LOG}/e14_judge2.log"
fi
echo "$(ts) E14 chain done"
