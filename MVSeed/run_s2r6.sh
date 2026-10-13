#!/usr/bin/env bash
# S2-R6 chain (docs/stage2-R6-prereg.md section 2): reference-camera static snap on the E8 headline
# (H-Rs) and control (C-Rs) output folders; score; judge.  CPU only (scsam3 container for cv2).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ts() { date '+%F %T'; }
echo "$(ts) S2-R6 chain start  this repo HEAD $(git -C "${ROOT}" rev-parse --short HEAD)  dirty: $(git -C "${ROOT}" status --porcelain --untracked-files=no | wc -l)"
DOCKER=(docker run --rm --user "$(id -u):$(id -g)" -v /:/host -w "/host${ROOT}" scsam3)
for base in SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2 SegMaskSam3XW0MSdgeoset_repFs; do
  echo "$(ts) snap ${base}"
  "${DOCKER[@]}" python MVSeed/s2r6_snap.py --method "${base}" 2>&1 | grep -v -i nvidia | sed "s/^/    /"
  st=${PIPESTATUS[0]}
  if [[ ${st} -ne 0 ]]; then echo "$(ts) snap failed (exit ${st}); stopping before scoring"; exit 1; fi
done
echo "$(ts) score"
"${DOCKER[@]}" python eval/eval_jf.py --methods SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs SegMaskSam3XW0MSdgeoset_repFsRs \
  --out jf_s2r6.json > "${ROOT}/MVSeed/logs/s2r6_score.log" 2>&1
st=$?; echo "$(ts) score exit ${st}  $(tail -n 1 "${ROOT}/MVSeed/logs/s2r6_score.log" | cut -c1-200)"
python3 "${ROOT}/MVSeed/s2r6_judge.py" > "${ROOT}/MVSeed/logs/s2r6_judge.log" 2>&1
st=$?; echo "$(ts) judge exit ${st}"
echo "$(ts) S2-R6 chain done"
