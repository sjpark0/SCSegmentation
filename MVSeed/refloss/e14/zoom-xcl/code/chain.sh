#!/usr/bin/env bash
# S1-E14 zoom-xcl chain: 3 decode groups (DEV-A) then dev_score of the 4 arms, each job under the shared GPU lock.
set -u
SP=/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad
cd /home/sjpark/Documents/SCSegmentation/stage1
O=geomem/runs/_e14_zoom-xcl
LOG=$O/logs
mkdir -p $O/scores
G1=CoffeeMartini,AlexaMeadeExhibit,Barn,Frog,Fencing
G2=FlameSteak,Breakfast,MATF,AlexaMeadeFacePaint,Blocks,Dog
G3=PoznanStreet,MartialArts,Painter,CBABasketball,Carpark,Welder
i=0
for G in $G1 $G2 $G3; do
  i=$((i+1)); t0=$(date +%s)
  flock $SP/gpu.lock scripts/docker_run.sh scsam3 python $O/code/zx_decode.py --scenes $G > $LOG/decode_g$i.log 2>&1
  rc=$?; echo "decode group $i exit $rc wall $(( $(date +%s) - t0 ))s $(date +%T)" >> $LOG/chain.log
  [ $rc -eq 0 ] || exit $rc
done
for arm in zx z r2 zxb; do
  t0=$(date +%s)
  flock $SP/gpu.lock scripts/docker_run.sh scsam3 python -m methods.geomem.dev_score --masks $O/${arm}_A --out $O/scores/${arm}_A.json > $LOG/score_$arm.log 2>&1
  rc=$?; echo "score $arm exit $rc wall $(( $(date +%s) - t0 ))s $(date +%T)" >> $LOG/chain.log
done
echo CHAIN_DONE >> $LOG/chain.log
