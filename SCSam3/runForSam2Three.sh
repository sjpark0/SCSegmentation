#!/usr/bin/env bash
# The three datasets the OneStage* demos could not finish, on the ForSam2New
# variant.  --memory caps the container so a runaway is OOM-killed on its own
# instead of taking the host down, which is how the last two attempts ended.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${ROOT}/SCSam3/logs/forsam2-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOGDIR}"
echo "logs -> ${LOGDIR}"
for ds in "$@"; do
	printf '%-20s ' "${ds}"
	start=$SECONDS
	docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
		-v /:/host -w "/host${ROOT}/SCSam3" scsam3 \
		python runMVSegForSam2.py "${ds}" --overwrite \
		> "${LOGDIR}/${ds}.log" 2>&1
	st=$?
	if [[ ${st} -eq 0 ]]; then printf 'ok    %4ds\n' $(( SECONDS - start ))
	else printf 'FAILED(%d) %4ds  %s\n' "${st}" $(( SECONDS - start )) "${LOGDIR}/${ds}.log"
		grep -aE "Error|error|Killed" "${LOGDIR}/${ds}.log" | tail -2 | sed 's/^/    /'
	fi
done
docker run --rm -v /:/host scsam3 chown -R "$(id -u):$(id -g)" "/host${ROOT}/Data/MVSeg" >/dev/null 2>&1
echo "all done"
