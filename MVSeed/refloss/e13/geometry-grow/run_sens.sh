#!/bin/bash
cd /tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/geometry-grow
PY="env PYTHONDONTWRITEBYTECODE=1 /home/sjpark/Documents/SCSegmentation/stage1/.venv/bin/python -I grow.py --procs 14"
for kv in TAU_2=0.01 TAU_2=0.04 LR=0.5 LR=2 TAU_X=0.01 TAU_X=0.04 M_LINK=1 M_LINK=25 K_LINK=2 K_LINK=8 R_ERODE=0 R_ERODE=4; do
  n=sens_${kv/=/_}
  $PY --out $n --set $kv --variants mv,mv_nosplat > $n.log 2>&1
  echo "$n $(tail -1 $n.log)"
done
for kv in EPS_PLANE=0.005 EPS_PLANE=0.02; do
  n=sens_${kv/=/_}
  $PY --out $n --set $kv --variants mv,mv_plane > $n.log 2>&1
  echo "$n $(tail -1 $n.log)"
done
echo ALLDONE
