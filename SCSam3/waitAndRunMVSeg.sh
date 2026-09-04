#!/usr/bin/env bash
# Wait for the GPU to come free, smoke-test one small dataset, then run the
# whole MVSeg sweep and score it.
#
#   ./waitAndRunMVSeg.sh                    wait for 03:00 and a free GPU
#   NOT_BEFORE=0 ./waitAndRunMVSeg.sh       start as soon as the GPU is free
#   FREE_MIB=30000 ./waitAndRunMVSeg.sh     accept a smaller GPU
#
# The smoke test comes first on purpose: the GPU path has never been executed,
# and Frog is the cheapest dataset to find out on (13 cameras, 5 objects). If it
# fails the sweep is not started, so a broken pipeline costs minutes rather than
# a whole night.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/SCSam3" || exit 1

NOT_BEFORE="${NOT_BEFORE:-$(date -d 'tomorrow 03:00' +%s)}"
GIVE_UP="${GIVE_UP:-$(date -d 'tomorrow 09:00' +%s)}"
FREE_MIB="${FREE_MIB:-40000}"
POLL="${POLL:-300}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="${ROOT}/SCSam3/logs"
mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/auto-${STAMP}.log"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

free_mib() {
	nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

say "waiting: not before $(date -d "@${NOT_BEFORE}" '+%m-%d %H:%M'), "\
"need ${FREE_MIB} MiB free, giving up at $(date -d "@${GIVE_UP}" '+%m-%d %H:%M')"

while :; do
	now=$(date +%s)
	free=$(free_mib)
	free=${free:-0}
	if [[ ${now} -ge ${NOT_BEFORE} && ${free} -ge ${FREE_MIB} ]]; then
		say "GPU free (${free} MiB) - starting"
		break
	fi
	if [[ ${now} -ge ${GIVE_UP} ]]; then
		say "GAVE UP: still only ${free} MiB free at the deadline. Nothing was run."
		nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee -a "${LOG}"
		exit 3
	fi
	sleep "${POLL}"
done

say "smoke test: Frog / OneStageNew"
./runMVSegAll.sh --algo OneStageNew --no-eval --overwrite Frog >>"${LOG}" 2>&1
if [[ $? -ne 0 ]] || ! grep -q "all runs finished" "${LOG}"; then
	say "SMOKE TEST FAILED - sweep not started. See ${LOG}"
	exit 4
fi
say "smoke test ok"

say "full sweep: 15 datasets x 2 algorithms, then J&F"
./runMVSegAll.sh --overwrite >>"${LOG}" 2>&1
status=$?
say "sweep exited with ${status}"
say "log: ${LOG}"
exit ${status}
