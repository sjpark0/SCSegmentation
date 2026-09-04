#!/usr/bin/env bash
# The three datasets no SAM 3 variant could finish, on the memory-fixed package.
# --memory caps the container: host RAM exhaustion is what rebooted the machine
# twice, and the cap turns that into a contained failure.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${ROOT}/SCSam3/logs/mvopt-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOGDIR}"
echo "logs -> ${LOGDIR}"
for ds in "$@"; do
	printf '%-20s ' "${ds}"
	start=$SECONDS
	docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
		-e SCSAM3_TRIM_CACHED_OUTPUTS=1 \
		-v /:/host -w "/host${ROOT}/SCSam3" scsam3 \
		python runMVSeg.py "${ds}" --algo "${ALGO:-MVOpt}" ${OUT:+--out "${OUT}"} --overwrite \
		> "${LOGDIR}/${ds}.log" 2>&1
	st=$?
	if [[ ${st} -eq 0 ]]; then printf 'ok    %5ds\n' $(( SECONDS - start ))
	else printf 'FAILED(%d) %5ds  %s\n' "${st}" $(( SECONDS - start )) "${LOGDIR}/${ds}.log"
		grep -aE "OutOfMemory|Error|Killed" "${LOGDIR}/${ds}.log" | tail -2 | sed 's/^/    /'
	fi
done
docker run --rm -v /:/host scsam3 chown -R "$(id -u):$(id -g)" "/host${ROOT}/Data/MVSeg" >/dev/null 2>&1
echo "all done"
