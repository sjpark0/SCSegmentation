#!/usr/bin/env bash
# Stage-1 port check (docs/stage1-port.md): rerun the three E8 stage-1 arms from stage1/ (the code
# ported from SCSegmentation2D) into stage1/Data/<scene>/Result/<E8 out names>; then compare them byte
# for byte with the 2D repo's E8 outputs (MVSeed/stage1_port_check.py).  One GPU job at a time.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ts() { date '+%F %T'; }
echo "$(ts) stage-1 port run  HEAD $(git -C "${ROOT}" rev-parse --short HEAD)  dirty: $(git -C "${ROOT}" status --porcelain --untracked-files=no | wc -l)"
echo "$(ts) image $(docker image inspect scsam3 --format '{{.Id}}' | cut -c1-19)  driver $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"
declare -A REGEN=(
  [geoset_rep]="--arm geoset_pred_ft --gate reproj --out geoset_pred_ft_rep_e8 --r1-out geoset_pred_ft_r1_e8"
  [geoset_abs]="--arm geoset_pred_ft --gate absent --out geoset_pred_ft_abs_e8 --r1-out geoset_pred_ft_r1_e8b"
  [geoalign_rep]="--arm geoalign_ft --gate reproj --out geoalign_ft_x_rep_e8 --r1-out geoalign_ft_x_r1_e8")
for t in ${TAGS:-geoset_rep geoset_abs geoalign_rep}; do
  echo "$(ts) regen ${t}: ${REGEN[$t]}"
  ( cd "${ROOT}/stage1" && scripts/run_geomem.sh all ${REGEN[$t]} ) > "${ROOT}/MVSeed/logs/stage1_port_${t}.log" 2>&1
  st=$?; echo "$(ts) regen ${t} exit ${st}  ($(tail -n 1 "${ROOT}/MVSeed/logs/stage1_port_${t}.log"))"
  [[ ${st} -eq 0 ]] || { echo "$(ts) ABORT"; exit 1; }
done
echo "$(ts) done"
