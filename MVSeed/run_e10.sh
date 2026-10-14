#!/usr/bin/env bash
# S1-E10 chain (docs/stage1-E10-prereg.md section 3.3).  No training.
#   STAGES="z4 timing c3 decode eval judge1 sweep score judge2" (default: the stage-1 half "z4 timing c3 decode eval judge1");
#   ROPES="grid remap" (the two arms; the timing stage drops "remap" when a crop costs more than 0.8 s -- registered budget rule);
#   TAGS (sweep/score: default = judge1's passing tags).  One GPU job at a time.  Logs: MVSeed/logs/e10_*.log, SCSam3/logs/e10-*.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S1="${ROOT}/stage1"
PY="${S1}/.venv/bin/python"
STAGES="${STAGES:-z4 timing c3 decode eval judge1}"
if [[ -z "${ROPES+x}" && -f "${ROOT}/MVSeed/runs/e10/timing.json" ]] && python3 -c "import json,sys;sys.exit(0 if json.load(open('${ROOT}/MVSeed/runs/e10/timing.json')).get('drop_remap') else 1)"; then
  ROPES="grid"      # the registered budget rule already fired in an earlier timing stage (timing.json: drop_remap)
fi
ROPES="${ROPES:-grid remap}"
SCENES="AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast CBABasketball Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF MartialArts Painter PoznanStreet Welder"
OUTDIR="${ROOT}/MVSeed/runs/e10"; LOG="${ROOT}/MVSeed/logs"; mkdir -p "${OUTDIR}" "${LOG}"
ts() { date '+%F %T'; }
has() { [[ " ${STAGES} " == *" $1 "* ]]; }
folder_of() { [[ $1 == grid ]] && echo geoset_pred_ft_rep_z2_e10 || echo geoset_pred_ft_rep_z2r_e10; }
r1_of() { [[ $1 == grid ]] && echo geoset_pred_ft_r1_z2_e10 || echo geoset_pred_ft_r1_z2r_e10; }
declare -A ARM_OF=([geoset_rep_z2]=geoset_pred_ft_rep_z2_e10 [geoset_rep_z2r]=geoset_pred_ft_rep_z2r_e10)
echo "$(ts) E10 chain start  stages: ${STAGES}  ropes: ${ROPES}  HEAD $(git -C "${ROOT}" rev-parse --short HEAD)"

cmp_dirs() {   # <a> <b>: PNG bytes of every camera folder (reference included) must be identical and the key sets equal
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
  echo "$(ts) z4: the adopted preset after the code change (Fencing, geoset_pred_ft + reproj, zoom off) vs the _e8 folders"
  ( cd "${S1}" && scripts/run_geomem.sh Fencing --arm geoset_pred_ft --gate reproj --out z4_check --r1-out z4_check_r1 --overwrite ) > "${LOG}/e10_z4.log" 2>&1
  st=$?; echo "$(ts) z4 decode exit ${st}  $(tail -n 1 "${LOG}/e10_z4.log" | cut -c1-120)"
  ok=1
  cmp_dirs "${S1}/Data/Fencing/Result/z4_check" "${S1}/Data/Fencing/Result/geoset_pred_ft_rep_e8" || ok=0
  cmp_dirs "${S1}/Data/Fencing/Result/z4_check_r1" "${S1}/Data/Fencing/Result/geoset_pred_ft_r1_e8" || ok=0
  cmp "${S1}/Data/Fencing/Result/z4_check/round1_conf.json" "${S1}/Data/Fencing/Result/geoset_pred_ft_rep_e8/round1_conf.json" && echo "round1_conf.json identical" || ok=0
  [[ ${st} -eq 0 && ${ok} -eq 1 ]] || { echo "$(ts) ABORT: Z4 failed (the adopted path is no longer byte-identical)"; exit 1; }
  echo "$(ts) z4 OK"
fi
if has timing; then
  echo "$(ts) timing: Fencing Blocks Breakfast with --zoom 2 (grid), outputs timing_z2 (not scored)"
  tot=0; n=0
  for sc in Fencing Blocks Breakfast; do
    ( cd "${S1}" && scripts/run_geomem.sh "${sc}" --arm geoset_pred_ft --gate reproj --zoom 2 --zoom-rope grid --out timing_z2 --r1-out timing_z2_r1 --overwrite ) > "${LOG}/e10_timing_${sc}.log" 2>&1
    st=$?; [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: timing decode of ${sc} failed"; exit 1; }
    read -r zs zr < <("${PY}" -c "import json;c=json.load(open('${S1}/Data/${sc}/Result/timing_z2/RUN.json'))['config'];print(c['zoom_seconds'],c['n_zoom_run'])")
    echo "$(ts) timing ${sc}: ${zr} crops in ${zs} s  ($(tail -n 1 "${LOG}/e10_timing_${sc}.log" | cut -c1-80))"
    tot=$(echo "${tot} + ${zs}" | bc -l); n=$((n + zr))
  done
  per=$(echo "${tot} / ${n}" | bc -l); printf -v per "%.3f" "${per}"
  echo "$(ts) timing: ${n} crops, ${per} s per crop"
  drop=false
  if (( $(echo "${per} > 0.8" | bc -l) )); then
    echo "$(ts) budget rule: ${per} s per crop > 0.8 -> C2 (remap) dropped from this chain (prereg section 3.3)"; ROPES="grid"; drop=true
  fi
  echo "{\"crops\": ${n}, \"seconds\": ${tot}, \"per_crop\": ${per}, \"drop_remap\": ${drop}}" > "${OUTDIR}/timing.json"
fi
if has c3; then
  echo "$(ts) c3: guided-filter control on the C0 PNGs (host)"
  "${PY}" "${ROOT}/MVSeed/refloss/guided_band.py" > "${LOG}/e10_c3.log" 2>&1
  st=$?; echo "$(ts) c3 exit ${st}  $(tail -n 1 "${LOG}/e10_c3.log" | cut -c1-120)"
  [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: c3 failed"; exit 1; }
fi
if has decode; then
  for rope in ${ROPES}; do
    out=$(folder_of "${rope}"); r1=$(r1_of "${rope}")
    echo "$(ts) decode zoom 2 / ${rope} -> ${out} / ${r1}"
    ( cd "${S1}" && scripts/run_geomem.sh all --arm geoset_pred_ft --gate reproj --zoom 2 --zoom-rope "${rope}" --out "${out}" --r1-out "${r1}" --overwrite ) \
      > "${LOG}/e10_decode_${rope}.log" 2>&1
    st=$?; nok=$(grep -c ' ok$' "${LOG}/e10_decode_${rope}.log")
    echo "$(ts) decode ${rope} exit ${st}  (${nok} scenes ok)"
    [[ ${st} -eq 0 && ${nok} -eq 17 ]] || { echo "$(ts) ABORT: decode failed (run_all records a per-scene error and exits 0)"; exit 1; }
  done
fi
if has eval; then
  NAMES="geoset_pred_ft_rep_e8 geoset_pred_ft_rep_gf_e10"
  for rope in ${ROPES}; do NAMES="${NAMES} $(folder_of "${rope}")"; done
  for name in ${NAMES}; do
    [[ -d "${S1}/Data/Fencing/Result/${name}" ]] || { echo "$(ts) eval: ${name} absent, skipped"; continue; }
    ( cd "${S1}" && "${PY}" -m mvseg2d.evaluate --data Data --method "${name}" --out "${OUTDIR}/kf_${name}.json" ) > "${LOG}/e10_eval_${name}.log" 2>&1
    st=$?; echo "$(ts) eval ${name} exit ${st}  $(tail -n 1 "${LOG}/e10_eval_${name}.log" | cut -c1-160)"
    [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: eval failed"; exit 1; }
  done
  ( cd "${S1}" && "${PY}" -m mvseg2d.report "${OUTDIR}"/kf_*.json --out "${OUTDIR}/kf_report.md" ) > /dev/null 2>&1
  for name in ${NAMES}; do
    [[ ${name} == geoset_pred_ft_rep_e8 ]] && continue       # C0's anatomy is MVSeed/runs/seedfail.json
    [[ -d "${S1}/Data/Fencing/Result/${name}" ]] || continue
    ( cd "${OUTDIR}" && docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${OUTDIR}" scsam3 \
        python "/host${ROOT}/MVSeed/refloss/seedfail.py" "s1:${name}" "sf_${name}.json" ) > "${LOG}/e10_sf_${name}.log" 2>&1
    echo "$(ts) seedfail ${name} exit $?"
  done
fi
if has judge1; then
  "${PY}" "${ROOT}/MVSeed/e10_judge.py" --stage 1 > "${LOG}/e10_judge1.log" 2>&1
  st=$?; echo "$(ts) judge1 exit ${st}"; cat "${LOG}/e10_judge1.log"
  [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: judge1 failed"; exit 1; }
fi
if has sweep; then
  TAGS="${TAGS-$(python3 -c "import json;print(' '.join(json.load(open('${OUTDIR}/e10_judge.json'))['stage1']['passing_tags']))")}" || exit 1
  echo "$(ts) sweep tags: ${TAGS:-<none: no arm passed the stage-1 gate>}"
  for tag in ${TAGS}; do
    echo "$(ts) sweep ${tag} (headline, in-process, ref-snap)"
    ( cd "${ROOT}/SCSam3" && env XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod STAGE1="${tag}" F0SEED=1 CROP=10000 CROPGATE=v2 REFSNAP=1 \
        LOGTAG="e10-${tag}" ./runMVOptThree.sh ${SCENES} ) 2>&1 | sed "s/^/    /"
  done
fi
if has score; then
  TAGS="${TAGS-$(python3 -c "import json;print(' '.join(json.load(open('${OUTDIR}/e10_judge.json'))['stage1']['passing_tags']))")}" || exit 1
  if [[ -z "${TAGS}" ]]; then
    echo "$(ts) score: no passing tag, nothing to score (judge2 still records the 'no effect' outcome)"
  else
    METHODS=$(for tag in ${TAGS}; do echo "SegMaskSam3XW1CGPS4MSd${tag}FsCs10kG2Rs"; done)
    echo "$(ts) score ${METHODS}"
    docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${ROOT}" scsam3 \
      python eval/eval_jf.py --methods ${METHODS} --out jf_e10.json > "${LOG}/e10_score.log" 2>&1
    st=$?; echo "$(ts) score exit ${st}  $(tail -n 2 "${LOG}/e10_score.log" | tr '\n' ' ' | cut -c1-200)"
  fi
fi
if has judge2; then
  "${PY}" "${ROOT}/MVSeed/e10_judge.py" --stage 2 > "${LOG}/e10_judge2.log" 2>&1
  st=$?; echo "$(ts) judge2 exit ${st}"; cat "${LOG}/e10_judge2.log"
fi
echo "$(ts) E10 chain done"
