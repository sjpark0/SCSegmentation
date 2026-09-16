#!/usr/bin/env bash
# S3-P: all four registered runs in sequence (GPU is shared, so never in parallel).
cd "$(dirname "$0")/.." || exit 1
for id in P1-C P1-H P2-T P2-P; do
  plenoptic/run_p1.sh "$id" || echo "$id FAILED" >> plenoptic/RUN_FAILED
done
echo ALL_DONE >> "${LOG:-/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/9786e830-b342-4be7-bc11-e7701415bd8d/scratchpad}/plen_runs.txt"
