#!/usr/bin/env bash
# S1-E12 chain (docs/stage1-E12-prereg.md section 3.1).
#   STAGES="z4 train zt decode eval vis judge1 sweep score judge2" (default: the stage-1 half "z4 train zt decode eval vis judge1");
#   ARMS="geoset_cutall_ft geoset_cut0_ft" (M+ first, so zt stops a broken cut path after M+ fold 0);
#   TAGS (sweep/score: default = judge1's passing tags).  One GPU job at a time.  Logs: MVSeed/logs/e12_*.log, SCSam3/logs/e12-*.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S1="${ROOT}/stage1"
PY="${S1}/.venv/bin/python"
STAGES="${STAGES:-z4 train zt decode eval vis judge1}"
ARMS="${ARMS:-geoset_cutall_ft geoset_cut0_ft}"
SCENES="AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast CBABasketball Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF MartialArts Painter PoznanStreet Welder"
OUTDIR="${ROOT}/MVSeed/runs/e12"; LOG="${ROOT}/MVSeed/logs"; mkdir -p "${OUTDIR}" "${LOG}"
HEAD="$(git -C "${ROOT}" rev-parse --short HEAD)"
ts() { date '+%F %T'; }
has() { [[ " ${STAGES} " == *" $1 "* ]]; }
echo "$(ts) E12 chain start  stages: ${STAGES}  arms: ${ARMS}  HEAD ${HEAD}"

cmp_dirs() {   # <a> <b>: PNG bytes of every camera folder must be identical and the key sets equal
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

if has z4; then
  echo "$(ts) z4: the adopted preset after the code change (Fencing, geoset_pred_ft --gate reproj --xcl out) vs the _x_e11 folders"
  ( cd "${S1}" && scripts/run_geomem.sh Fencing --arm geoset_pred_ft --gate reproj --xcl out --out z4_check --r1-out z4_check_r1 --overwrite ) > "${LOG}/e12_z4.log" 2>&1
  st=$?; echo "$(ts) z4 decode exit ${st}  $(tail -n 1 "${LOG}/e12_z4.log" | cut -c1-120)"
  F="${S1}/Data/Fencing/Result"; ok=1
  cmp_dirs "${F}/z4_check" "${F}/geoset_pred_ft_rep_x_e11" || ok=0
  cmp_dirs "${F}/z4_check_r1" "${F}/geoset_pred_ft_r1_x_e11" || ok=0
  for j in round1_conf.json xcl_conf.json; do cmp "${F}/z4_check/${j}" "${F}/geoset_pred_ft_rep_x_e11/${j}" && echo "${j} identical" || ok=0; done
  [[ ${st} -eq 0 && ${ok} -eq 1 ]] || { echo "$(ts) ABORT: Z4 failed (the adopted path is no longer byte-identical)"; exit 1; }
  echo "$(ts) z4 OK"
fi
if has train; then
  for arm in ${ARMS}; do
    for k in 0 1 2 3 4; do
      if [[ -f "${S1}/geomem/runs/${arm}/fold${k}/ckpt.pt" && -f "${S1}/geomem/runs/${arm}/fold${k}/config.json" ]]; then
        echo "$(ts) train ${arm} fold${k}: ckpt.pt + config.json exist, skip"
      else
        echo "$(ts) train ${arm} fold${k}"
        ( cd "${S1}" && scripts/run_geomem_train.sh "${arm}" "${k}" ) > "${LOG}/e12_train_${arm}_fold${k}.log" 2>&1
        st=$?; echo "$(ts) train ${arm} fold${k} exit ${st}  ($(tail -n 1 "${LOG}/e12_train_${arm}_fold${k}.log" | cut -c1-160))"
        [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: training failed"; exit 1; }
      fi
      if has zt; then
        "${PY}" "${ROOT}/MVSeed/e12_judge.py" --zt "${arm}" "${k}" --head "${HEAD}" > "${LOG}/e12_zt_${arm}_fold${k}.log" 2>&1
        st=$?; echo "$(ts) zt ${arm} fold${k} exit ${st}  $(cut -c1-400 "${LOG}/e12_zt_${arm}_fold${k}.log")"
        [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: zt failed (training invariants, prereg section 7)"; exit 1; }
      fi
    done
  done
fi
if has zt && ! has train; then     # zt over the folds already trained (review 2026-10-15: never skipped silently)
  for arm in ${ARMS}; do
    for k in 0 1 2 3 4; do
      "${PY}" "${ROOT}/MVSeed/e12_judge.py" --zt "${arm}" "${k}" --head "${HEAD}" > "${LOG}/e12_zt_${arm}_fold${k}.log" 2>&1
      st=$?; echo "$(ts) zt ${arm} fold${k} exit ${st}  $(cut -c1-400 "${LOG}/e12_zt_${arm}_fold${k}.log")"
      [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: zt failed (training invariants, prereg section 7)"; exit 1; }
    done
  done
fi
if has decode; then
  for arm in ${ARMS}; do
    echo "$(ts) decode ${arm} (reproj gate, --xcl out) -> ${arm}_rep_x_e12 / ${arm}_r1_x_e12"
    ( cd "${S1}" && scripts/run_geomem.sh all --arm "${arm}" --gate reproj --xcl out --out "${arm}_rep_x_e12" --r1-out "${arm}_r1_x_e12" --overwrite ) \
      > "${LOG}/e12_decode_${arm}.log" 2>&1
    st=$?; nok=$(grep -c ' ok$' "${LOG}/e12_decode_${arm}.log")
    echo "$(ts) decode ${arm} exit ${st}  (${nok} scenes ok)"
    [[ ${st} -eq 0 && ${nok} -eq 17 ]] || { echo "$(ts) ABORT: decode failed (run_all records a per-scene error and exits 0)"; exit 1; }
  done
fi
if has eval; then
  NAMES="geoset_pred_ft_rep_x_e11"
  for arm in ${ARMS}; do NAMES="${NAMES} ${arm}_rep_x_e12"; done
  for name in ${NAMES}; do
    ( cd "${S1}" && "${PY}" -m mvseg2d.evaluate --data Data --method "${name}" --out "${OUTDIR}/kf_${name}.json" ) > "${LOG}/e12_eval_${name}.log" 2>&1
    st=$?; echo "$(ts) eval ${name} exit ${st}  $(tail -n 1 "${LOG}/e12_eval_${name}.log" | cut -c1-160)"
    [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: eval failed"; exit 1; }
  done
  ( cd "${S1}" && "${PY}" -m mvseg2d.report "${OUTDIR}"/kf_*.json --out "${OUTDIR}/kf_report.md" ) > /dev/null 2>&1
  for name in ${NAMES}; do
    [[ ${name} == geoset_pred_ft_rep_x_e11 ]] && continue
    ( cd "${OUTDIR}" && docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${OUTDIR}" scsam3 \
        python "/host${ROOT}/MVSeed/refloss/seedfail.py" "s1:${name}" "sf_${name}.json" ) > "${LOG}/e12_sf_${name}.log" 2>&1
    st=$?; echo "$(ts) seedfail ${name} exit ${st}"
  done
fi
if has vis; then
  echo "$(ts) vis: visibility block on the final and round-1 folders"
  ( cd "${S1}" && scripts/docker_run.sh scsam3 python "/host${ROOT}/MVSeed/refloss/e12/keyframe_vis.py" --out "/host${OUTDIR}/vis.json" \
      geoset_pred_ft_rep_x_e11 geoset_cut0_ft_rep_x_e12 geoset_cutall_ft_rep_x_e12 \
      geoset_pred_ft_r1_x_e11 geoset_cut0_ft_r1_x_e12 geoset_cutall_ft_r1_x_e12 \
      --pair geoset_cutall_ft_rep_x_e12:geoset_cut0_ft_rep_x_e12 --pair geoset_cut0_ft_rep_x_e12:geoset_pred_ft_rep_x_e11 \
      --pair geoset_cutall_ft_r1_x_e12:geoset_cut0_ft_r1_x_e12 --pair geoset_cut0_ft_r1_x_e12:geoset_pred_ft_r1_x_e11 ) > "${LOG}/e12_vis.log" 2>&1
  st=$?; echo "$(ts) vis exit ${st}  $(tail -n 1 "${LOG}/e12_vis.log" | cut -c1-160)"
  [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: vis failed"; exit 1; }
fi
if has judge1; then
  "${PY}" "${ROOT}/MVSeed/e12_judge.py" --stage 1 > "${LOG}/e12_judge1.log" 2>&1
  st=$?; echo "$(ts) judge1 exit ${st}"; cat "${LOG}/e12_judge1.log"
  [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: judge1 failed"; exit 1; }
fi
if has sweep; then
  TAGS="${TAGS-$(python3 -c "import json;print(' '.join(json.load(open('${OUTDIR}/e12_judge.json'))['stage1']['passing_tags']))")}" || exit 1
  echo "$(ts) sweep tags: ${TAGS:-<none: M+ did not pass the stage-1 gate>}"
  for tag in ${TAGS}; do
    echo "$(ts) sweep ${tag} (headline, in-process, ref-snap)"
    ( cd "${ROOT}/SCSam3" && env XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod STAGE1="${tag}" F0SEED=1 CROP=10000 CROPGATE=v2 REFSNAP=1 \
        LOGTAG="e12-${tag}" ./runMVOptThree.sh ${SCENES} ) 2>&1 | sed "s/^/    /"
  done
fi
if has score; then
  TAGS="${TAGS-$(python3 -c "import json;print(' '.join(json.load(open('${OUTDIR}/e12_judge.json'))['stage1']['passing_tags']))")}" || exit 1
  if [[ -z "${TAGS}" ]]; then
    echo "$(ts) score: no passing tag, nothing to score (judge2 still records the outcome)"
  else
    METHODS=$(for tag in ${TAGS}; do echo "SegMaskSam3XW1CGPS4MSd${tag}FsCs10kG2Rs"; done)
    echo "$(ts) score ${METHODS}"
    docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${ROOT}" scsam3 \
      python eval/eval_jf.py --methods ${METHODS} --out jf_e12.json > "${LOG}/e12_score.log" 2>&1
    st=$?; echo "$(ts) score exit ${st}  $(tail -n 2 "${LOG}/e12_score.log" | tr '\n' ' ' | cut -c1-200)"
  fi
fi
if has judge2; then
  "${PY}" "${ROOT}/MVSeed/e12_judge.py" --stage 2 > "${LOG}/e12_judge2.log" 2>&1
  st=$?; echo "$(ts) judge2 exit ${st}"; cat "${LOG}/e12_judge2.log"
fi
echo "$(ts) E12 chain done"
