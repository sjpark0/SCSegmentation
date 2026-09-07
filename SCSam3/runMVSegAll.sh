#!/usr/bin/env bash
# Run both SCSam3 demos over every MVSeg dataset, then score the results.
#
#   ./runMVSegAll.sh                      every dataset, both algorithms
#   ./runMVSegAll.sh --dry-run            resolve everything, load no model
#   ./runMVSegAll.sh --algo OneStageNew   one algorithm
#   ./runMVSegAll.sh Blocks Frog          named datasets only
#   ./runMVSegAll.sh --no-eval            skip the J&F step at the end
#
# Each dataset runs in its own container so that one failure or OOM does not
# take the rest down, and so GPU memory is fully released in between - the
# 45-camera scenes hold a session per camera.
#
# The GPU has to be free: check with nvidia-smi first.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-scsam3}"
# passed into the container so runMVSeg.py can record it in MANIFEST.json.
# Query one tag: `docker images -q scsam3` lists every tag of the repository,
# so a bare repository name is pinned to :latest, which is what `docker run`
# resolves it to.
IMAGE_REF="${IMAGE}"; [[ "${IMAGE_REF}" == *:* ]] || IMAGE_REF="${IMAGE_REF}:latest"
IMAGE_ID="$(docker images --no-trunc -q "${IMAGE_REF}" 2>/dev/null | head -n1)"
WORKDIR="/host${ROOT}/SCSam3"

ALL_DATASETS=(AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast
              Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF Painter
              PoznanStreet Welder)
# 개발은 MVOpt에서 진행합니다. demoSCSam3OneStageNew는 2026-09-04부로 MVOpt의
# 동결 스냅샷이므로(코드가 동일), 여기서 OneStageNew를 돌리면 패치 코드의 출력이
# 발표 기준선 폴더 SegMaskSam3OneStageNew를 덮어씁니다. 그래서 MVOpt를 씁니다.
# 기준선 마스크를 재생성해야 한다면 git tag baseline-onestagenew를 먼저 체크아웃하십시오.
ALGOS=(OneStage MVOpt)
declare -A OUTNAME=([OneStage]=SegMaskSam3OneStage [MVOpt]=SegMaskSam3MVOpt [OneStageNew]=SegMaskSam3OneStageNew)

DATASETS=()
EXTRA=()
RUN_EVAL=1

while [[ $# -gt 0 ]]; do
	case "$1" in
		--algo)      ALGOS=("$2"); shift 2 ;;
		--dry-run)   EXTRA+=(--dry-run); shift ;;
		--overwrite) EXTRA+=(--overwrite); shift ;;
		--no-eval)   RUN_EVAL=0; shift ;;
		-h|--help)   sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
		-*)          echo "unknown option: $1" >&2; exit 2 ;;
		*)           DATASETS+=("$1"); shift ;;
	esac
done
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=("${ALL_DATASETS[@]}")

STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="${ROOT}/SCSam3/logs/mvseg-${STAMP}"
mkdir -p "${LOGDIR}"
echo "logs -> ${LOGDIR}"

FAILED=()
START_ALL=$SECONDS

for algo in "${ALGOS[@]}"; do
	for ds in "${DATASETS[@]}"; do
		log="${LOGDIR}/${algo}-${ds}.log"
		printf '%-12s %-20s ' "${algo}" "${ds}"
		start=$SECONDS
		docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
			-e SCSAM3_IMAGE_ID="${IMAGE_ID}" \
			-v /:/host -w "${WORKDIR}" "${IMAGE}" \
			python runMVSeg.py "${ds}" --algo "${algo}" "${EXTRA[@]}" \
			> "${log}" 2>&1
		status=$?
		took=$(( SECONDS - start ))
		if [[ ${status} -eq 0 ]]; then
			printf 'ok    %4ds\n' "${took}"
		else
			printf 'FAILED %3ds  (%s)\n' "${took}" "${log}"
			tail -n 3 "${log}" | sed 's/^/    /'
			FAILED+=("${algo}/${ds}")
		fi
	done
done

# chown what the container wrote back to the invoking user
docker run --rm -v /:/host "${IMAGE}" \
	chown -R "$(id -u):$(id -g)" "/host${ROOT}/Data/MVSeg" >/dev/null 2>&1

echo
echo "total $(( SECONDS - START_ALL ))s"
if [[ ${#FAILED[@]} -gt 0 ]]; then
	echo "failed: ${FAILED[*]}"
else
	echo "all runs finished"
fi

# Score whatever finished. A dataset missing one of the methods is skipped by
# eval_jf.py, so a partial run still produces numbers for the rest.
if [[ ${RUN_EVAL} -eq 1 ]] && [[ ! " ${EXTRA[*]} " =~ " --dry-run " ]]; then
	methods=""
	for algo in "${ALGOS[@]}"; do methods="${methods} ${OUTNAME[${algo}]}"; done
	echo
	echo "scoring J&F for:${methods}"
	docker run --rm -v /:/host -w "/host${ROOT}/Data/MVSeg" "${IMAGE}" \
		python eval_jf.py --methods ${methods} --out jf_raw_sam3.json \
		| tail -n 3
	docker run --rm -v /:/host "${IMAGE}" \
		chown "$(id -u):$(id -g)" "/host${ROOT}/Data/MVSeg/jf_raw_sam3.json" >/dev/null 2>&1
	echo
	echo "=== SAM 3 only ==="
	python3 "${ROOT}/Data/MVSeg/report_jf.py" \
		--raw jf_raw_sam3.json --methods ${methods} --no-missing

	# the SAM 2 baselines were scored earlier into jf_raw.json
	if [[ -f "${ROOT}/Data/MVSeg/jf_raw.json" ]]; then
		echo
		echo "=== SAM 2 baselines and SAM 3 side by side ==="
		python3 "${ROOT}/Data/MVSeg/report_jf.py" \
			--raw jf_raw.json jf_raw_sam3.json \
			--methods SegMask1 SegMaskNew1 SegMaskNew2 SegMaskNew3 ${methods} \
			--no-missing
	fi
fi
