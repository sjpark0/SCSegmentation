#!/usr/bin/env bash
# S1-E11 chain (docs/stage1-E11-prereg.md section 3.3).  No training.
#   STAGES="z4 decode eval judge1 sweep score judge2" (default: the stage-1 half "z4 decode eval judge1");
#   MODES="out outmem out_nocap" (X1, X2, D); TAGS (sweep/score: default = judge1's passing tags).  One GPU job at a time.
#   Logs: MVSeed/logs/e11_*.log, SCSam3/logs/e11-*.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S1="${ROOT}/stage1"
PY="${S1}/.venv/bin/python"
STAGES="${STAGES:-z4 decode eval judge1}"
MODES="${MODES:-out outmem out_nocap}"
SCENES="AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast CBABasketball Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF MartialArts Painter PoznanStreet Welder"
OUTDIR="${ROOT}/MVSeed/runs/e11"; LOG="${ROOT}/MVSeed/logs"; mkdir -p "${OUTDIR}" "${LOG}"
ts() { date '+%F %T'; }
has() { [[ " ${STAGES} " == *" $1 "* ]]; }
folder_of() { case "$1" in out) echo geoset_pred_ft_rep_x_e11 ;; outmem) echo geoset_pred_ft_rep_xm_e11 ;; out_nocap) echo geoset_pred_ft_rep_xu_e11 ;; esac; }
r1_of() { case "$1" in out) echo geoset_pred_ft_r1_x_e11 ;; outmem) echo geoset_pred_ft_r1_xm_e11 ;; out_nocap) echo geoset_pred_ft_r1_xu_e11 ;; esac; }
echo "$(ts) E11 chain start  stages: ${STAGES}  modes: ${MODES}  HEAD $(git -C "${ROOT}" rev-parse --short HEAD)"

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
  echo "$(ts) z4: the adopted preset after the code change (Fencing, --xcl none) vs the _e8 folders + the in-process byte check"
  ( cd "${S1}" && scripts/run_geomem.sh Fencing --arm geoset_pred_ft --gate reproj --xcl none --out z4_check --r1-out z4_check_r1 --overwrite ) > "${LOG}/e11_z4.log" 2>&1
  st=$?; echo "$(ts) z4 decode exit ${st}  $(tail -n 1 "${LOG}/e11_z4.log" | cut -c1-120)"
  ok=1
  cmp_dirs "${S1}/Data/Fencing/Result/z4_check" "${S1}/Data/Fencing/Result/geoset_pred_ft_rep_e8" || ok=0
  cmp_dirs "${S1}/Data/Fencing/Result/z4_check_r1" "${S1}/Data/Fencing/Result/geoset_pred_ft_r1_e8" || ok=0
  cmp "${S1}/Data/Fencing/Result/z4_check/round1_conf.json" "${S1}/Data/Fencing/Result/geoset_pred_ft_rep_e8/round1_conf.json" && echo "round1_conf.json identical" || ok=0
  "${PY}" "${ROOT}/MVSeed/inproc_check.py" > "${LOG}/e11_inproc_check.log" 2>&1; grep -q "all identical: True" "${LOG}/e11_inproc_check.log" && echo "inproc _ip check: all identical" || { echo "inproc _ip check FAILED"; ok=0; }
  [[ ${st} -eq 0 && ${ok} -eq 1 ]] || { echo "$(ts) ABORT: Z4 failed (the adopted path is no longer byte-identical)"; exit 1; }
  echo "$(ts) z4 OK"
fi
if has decode; then
  for mode in ${MODES}; do
    out=$(folder_of "${mode}"); r1=$(r1_of "${mode}")
    echo "$(ts) decode --xcl ${mode} -> ${out} / ${r1}"
    ( cd "${S1}" && scripts/run_geomem.sh all --arm geoset_pred_ft --gate reproj --xcl "${mode}" --out "${out}" --r1-out "${r1}" --overwrite ) \
      > "${LOG}/e11_decode_${mode}.log" 2>&1
    st=$?; nok=$(grep -c ' ok$' "${LOG}/e11_decode_${mode}.log")
    echo "$(ts) decode ${mode} exit ${st}  (${nok} scenes ok)"
    [[ ${st} -eq 0 && ${nok} -eq 17 ]] || { echo "$(ts) ABORT: decode failed (run_all records a per-scene error and exits 0)"; exit 1; }
  done
fi
if has eval; then
  NAMES="geoset_pred_ft_rep_e8"
  for mode in ${MODES}; do NAMES="${NAMES} $(folder_of "${mode}")"; done
  for name in ${NAMES}; do
    [[ -d "${S1}/Data/Fencing/Result/${name}" ]] || { echo "$(ts) eval: ${name} absent, skipped"; continue; }
    ( cd "${S1}" && "${PY}" -m mvseg2d.evaluate --data Data --method "${name}" --out "${OUTDIR}/kf_${name}.json" ) > "${LOG}/e11_eval_${name}.log" 2>&1
    st=$?; echo "$(ts) eval ${name} exit ${st}  $(tail -n 1 "${LOG}/e11_eval_${name}.log" | cut -c1-160)"
    [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: eval failed"; exit 1; }
  done
  ( cd "${S1}" && "${PY}" -m mvseg2d.report "${OUTDIR}"/kf_*.json --out "${OUTDIR}/kf_report.md" ) > /dev/null 2>&1
  for name in ${NAMES}; do
    [[ ${name} == geoset_pred_ft_rep_e8 ]] && continue
    [[ -d "${S1}/Data/Fencing/Result/${name}" ]] || continue
    ( cd "${OUTDIR}" && docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${OUTDIR}" scsam3 \
        python "/host${ROOT}/MVSeed/refloss/seedfail.py" "s1:${name}" "sf_${name}.json" ) > "${LOG}/e11_sf_${name}.log" 2>&1
    echo "$(ts) seedfail ${name} exit $?"
  done
fi
if has judge1; then
  "${PY}" "${ROOT}/MVSeed/e11_judge.py" --stage 1 > "${LOG}/e11_judge1.log" 2>&1
  st=$?; echo "$(ts) judge1 exit ${st}"; cat "${LOG}/e11_judge1.log"
  [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: judge1 failed"; exit 1; }
fi
if has sweep; then
  TAGS="${TAGS-$(python3 -c "import json;print(' '.join(json.load(open('${OUTDIR}/e11_judge.json'))['stage1']['passing_tags']))")}" || exit 1
  echo "$(ts) sweep tags: ${TAGS:-<none: no arm passed the stage-1 gate>}"
  for tag in ${TAGS}; do
    echo "$(ts) sweep ${tag} (headline, in-process, ref-snap)"
    ( cd "${ROOT}/SCSam3" && env XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod STAGE1="${tag}" F0SEED=1 CROP=10000 CROPGATE=v2 REFSNAP=1 \
        LOGTAG="e11-${tag}" ./runMVOptThree.sh ${SCENES} ) 2>&1 | sed "s/^/    /"
  done
fi
if has score; then
  TAGS="${TAGS-$(python3 -c "import json;print(' '.join(json.load(open('${OUTDIR}/e11_judge.json'))['stage1']['passing_tags']))")}" || exit 1
  if [[ -z "${TAGS}" ]]; then
    echo "$(ts) score: no passing tag, nothing to score (judge2 still records the 'no effect' outcome)"
  else
    METHODS=$(for tag in ${TAGS}; do echo "SegMaskSam3XW1CGPS4MSd${tag}FsCs10kG2Rs"; done)
    echo "$(ts) score ${METHODS}"
    docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${ROOT}" scsam3 \
      python eval/eval_jf.py --methods ${METHODS} --out jf_e11.json > "${LOG}/e11_score.log" 2>&1
    st=$?; echo "$(ts) score exit ${st}  $(tail -n 2 "${LOG}/e11_score.log" | tr '\n' ' ' | cut -c1-200)"
  fi
fi
if has judge2; then
  "${PY}" "${ROOT}/MVSeed/e11_judge.py" --stage 2 > "${LOG}/e11_judge2.log" 2>&1
  st=$?; echo "$(ts) judge2 exit ${st}"; cat "${LOG}/e11_judge2.log"
fi
echo "$(ts) E11 chain done"
