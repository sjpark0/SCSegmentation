#!/usr/bin/env bash
# One stage-1 decode of T06 frame 900 into ring_d15/<datadir>/T06_flag_ganggang_F/Result/<out>.
#   run_arm.sh <datadir-name> <out> [run.py args...]
# scsam3 image, whole host at /host (the scratchpad lives under /tmp), the repo's stage1 tree as cwd.
set -euo pipefail
D15=/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_d15
MAIN=/home/sjpark/Documents/SCSegmentation
DATA=$1; OUT=$2; shift 2
LOG=$D15/logs/$DATA.$OUT.log
# GPU memory sampler (whole card) while the decode runs
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > "$D15/logs/$DATA.$OUT.gpumem" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
echo "cmd: run.py --scene T06_flag_ganggang_F --data /host$D15/$DATA --out $OUT $*" | tee "$LOG"
T0=$(date +%s)
docker run --rm --gpus all --shm-size=32g --user "$(id -u):$(id -g)" \
  -v /:/host -w "/host$MAIN/stage1" \
  -e HF_HOME="/host$MAIN/SCSam3/hf_cache" -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH="/host$MAIN/stage1" -e HOME=/tmp \
  scsam3 python methods/geomem/run.py --scene T06_flag_ganggang_F --data "/host$D15/$DATA" --out "$OUT" "$@" 2>&1 | tee -a "$LOG"
echo "wall_s $(( $(date +%s) - T0 ))" | tee -a "$LOG"
echo "card_peak_mib $(sort -n "$D15/logs/$DATA.$OUT.gpumem" | tail -1)" | tee -a "$LOG"
