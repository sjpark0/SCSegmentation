#!/usr/bin/env bash
# External / control baselines on the stage-1 DEV sets (iccv2027-plan.md WP2): harness validation only (DEV-A, DEV-A');
# TEST runs come after G4.  One GPU job at a time (the E15 lock).
#   run_baselines.sh sam21 <order> <rot>      SAM 2.1 pseudo-video (baselines/sam21_dev.py) on DEV-A (rot 0) / DEV-A' (1, 2)
#                                             -> Data/_dev/sam21_<order>_A[_rot<r>] + dev_score -> geomem/runs/_baselines/scores/
#   run_baselines.sh sam21-queue              both orders x rot 0 1 2 (skips scored ones)
#   run_baselines.sh cutie <order> <rot>      Cutie (baselines/cutie_dev.py; code baselines/third_party/Cutie, weights baselines/weights)
#   run_baselines.sh cutie-queue              both orders x rot 0 1 2
#   run_baselines.sh samv <rot>               SAM-V (baselines/samv_dev.py; image scsamv, weights baselines/weights/samv)
#   run_baselines.sh vggts <rot>              VGGT-Segmentor (baselines/vggts_dev.py; image scsamv, VGGT-S checkpoints)
#   run_baselines.sh samv-queue | vggts-queue rot 0 1 2
#   run_baselines.sh after-e16                wait for the S1-E16 chain, then every queue (host, detached)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MAIN=$(cd "$REPO/.." && pwd)
WORK=$REPO/geomem/runs/_baselines
LOCK=$REPO/geomem/runs/_e15/gpu.lock
cd "$REPO"
mkdir -p "$WORK/logs" "$WORK/scores"
ts() { date '+%F %T'; }
gpu_lock() { exec 9>"$LOCK"; flock 9; }
samv_docker() {   # scsamv = baselines/Dockerfile.samv (torch 2.3.1 cu121); $1 = extra PYTHONPATH entries (container paths)
  local extra=$1; shift
  docker run --rm --gpus all --shm-size=16g --user "$(id -u):$(id -g)" -v "$MAIN:/host$MAIN" -w "/host$REPO" \
    -e PYTHONPATH="/host$REPO:$extra" -e HOME=/tmp -e TQDM_DISABLE=1 -e PYTHONUNBUFFERED=1 --entrypoint python scsamv "$@"
}
sam2_docker() {   # scsam2 = SCSam2/Dockerfile (torch 2.6 cu118, SAM 2 1.0); the repo at the same /host path as docker_run.sh
  docker run --rm --gpus all --shm-size=16g --user "$(id -u):$(id -g)" -v "$MAIN:/host$MAIN" -w "/host$REPO" \
    -e PYTHONPATH="/host$REPO${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA}" -e HOME=/tmp -e TQDM_DISABLE=1 -e PYTHONUNBUFFERED=1 --entrypoint python scsam2 "$@"
}
cmd=${1:-}; shift || true
case "$cmd" in
  sam21)
    order=$1; rot=$2
    name="sam21_${order}_A$([ "$rot" = 0 ] || echo "_rot$rot")"
    gpu_lock
    sam2_docker -m baselines.sam21_dev --order "$order" --split A --ref-rot "$rot"
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_score --masks "/host$REPO/Data/_dev/$name" --out "/host$WORK/scores/$name.json" ;;
  cutie)
    order=$1; rot=$2
    name="cutie_${order}_A$([ "$rot" = 0 ] || echo "_rot$rot")"
    gpu_lock
    PYTHONPATH_EXTRA="/host$REPO/baselines/third_party/Cutie" sam2_docker -m baselines.cutie_dev --order "$order" --split A --ref-rot "$rot"
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_score --masks "/host$REPO/Data/_dev/$name" --out "/host$WORK/scores/$name.json" ;;
  cutie-queue)
    for order in index nn; do for rot in 0 1 2; do
      name="cutie_${order}_A$([ "$rot" = 0 ] || echo "_rot$rot")"
      [ -f "$WORK/scores/$name.json" ] && continue
      echo "$(ts) cutie $order $rot"; "$0" cutie "$order" "$rot" > "$WORK/logs/$name.log" 2>&1 || { echo "$(ts) FAILED cutie $order $rot"; exit 1; }
    done; done
    echo "$(ts) cutie queue done" ;;
  samv)
    rot=$1; name="samv_A$([ "$rot" = 0 ] || echo "_rot$rot")"; S="/host$REPO/baselines/third_party/SAM-V"
    gpu_lock
    samv_docker "$S:$S/submodules/sam-hq:$S/submodules/vggt" -m baselines.samv_dev --split A --ref-rot "$rot"
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_score --masks "/host$REPO/Data/_dev/$name" --out "/host$WORK/scores/$name.json" ;;
  vggts)
    rot=$1; name="vggts_A$([ "$rot" = 0 ] || echo "_rot$rot")"; V="/host$REPO/baselines/third_party/VGGT-S"
    gpu_lock
    samv_docker "$V/src:$V/third_party/vggt_main" -m baselines.vggts_dev --split A --ref-rot "$rot" --eval-only
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_score --masks "/host$REPO/Data/_dev/$name" --out "/host$WORK/scores/$name.json" ;;
  samv-queue|vggts-queue)
    kind=${cmd%-queue}
    for rot in 0 1 2; do
      name="${kind}_A$([ "$rot" = 0 ] || echo "_rot$rot")"
      [ -f "$WORK/scores/$name.json" ] && continue
      echo "$(ts) $kind $rot"; "$0" "$kind" "$rot" > "$WORK/logs/$name.log" 2>&1 || { echo "$(ts) FAILED $kind $rot"; exit 1; }
    done
    echo "$(ts) $kind queue done" ;;
  sam21-queue)
    for order in index nn; do for rot in 0 1 2; do
      name="sam21_${order}_A$([ "$rot" = 0 ] || echo "_rot$rot")"
      [ -f "$WORK/scores/$name.json" ] && continue
      echo "$(ts) sam21 $order $rot"; "$0" sam21 "$order" "$rot" > "$WORK/logs/$name.log" 2>&1 || { echo "$(ts) FAILED sam21 $order $rot"; exit 1; }
    done; done
    echo "$(ts) sam21 queue done" ;;
  after-e16)
    until grep -q 'E16 chain done' "$REPO/geomem/runs/_e16/logs/after_train.log" 2>/dev/null; do
      grep -qE 'FAILED|stop' "$REPO/geomem/runs/_e16/logs/after_train.log" 2>/dev/null && { echo "$(ts) E16 chain failed; stop"; exit 1; }; sleep 300; done
    echo "$(ts) E16 done -> baseline queues"; "$0" sam21-queue && "$0" cutie-queue && "$0" vggts-queue && "$0" samv-queue ;;
  *)
    sed -n '2,/^set -euo pipefail/{/^set -euo/!p}' "$0"; exit 2 ;;
esac
