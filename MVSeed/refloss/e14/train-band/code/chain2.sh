#!/usr/bin/env bash
# S1-E14 train-band, second pass: re-decode E9 D2 / D1 on DEV-A with the token-0 logits saved (<cam>_t0.npz), into
# <preset>_A_t0, checked byte-for-byte against the first pass <preset>_A (determinism).  2 scene groups per preset.
set -u
cd /home/sjpark/Documents/SCSegmentation/stage1
LOCK=/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/gpu.lock
L=geomem/runs/_e14_train-band/logs
R=geomem/runs/_e14_train-band
G1=CoffeeMartini,FlameSteak,MATF,AlexaMeadeExhibit,Fencing,Frog,Dog,AlexaMeadeFacePaint
G2=Breakfast,CBABasketball,Painter,Barn,MartialArts,PoznanStreet,Carpark,Welder,Blocks
until grep -q "chain done" $L/chain.log 2>/dev/null; do sleep 10; done
for P in geoset_decb_rep_x geoset_dec_rep_x; do
  for g in 1 2; do
    if [ $g = 1 ]; then S=$G1; else S=$G2; fi
    t0=$(date +%s)
    PYTHONDONTWRITEBYTECODE=1 flock $LOCK scripts/docker_run.sh scsam3 python $R/code/dump_arm.py \
      --preset $P --scenes $S --tok0 --root $R/${P}_A_t0 --ref-root $R/${P}_A > $L/dump_${P}_t0_g$g.log 2>&1
    rc=$?
    echo "$(date '+%F %T') $P t0 group $g exit $rc wall $(( $(date +%s) - t0 ))s" >> $L/chain2.log
  done
done
echo "$(date '+%F %T') chain2 done" >> $L/chain2.log
