#!/usr/bin/env bash
# S3-P P3 (docs/plenoptic-P-prereg.md section 7): the 201-frame runs, in sequence.
cd "$(dirname "$0")/.." || exit 1
for id in ${RUN_IDS:-P4-C P4-H}; do plenoptic/run_p1.sh "$id" || echo "$id FAILED" >> plenoptic/RUN_FAILED; done
echo ALL_DONE_P3 >> "${LOG:-/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/9786e830-b342-4be7-bc11-e7701415bd8d/scratchpad}/plen_runs.txt"
