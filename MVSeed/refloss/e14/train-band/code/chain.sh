#!/usr/bin/env bash
# S1-E14 train-band: DEV-A decode + round-2 logit dump of E9 D2 / E9 D1 / E12 C0' with the adopted decode path.
# Each job holds the shared GPU lock; 2 scene groups per preset (each well under 25 min).
set -u
cd /home/sjpark/Documents/SCSegmentation/stage1
LOCK=/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/gpu.lock
L=geomem/runs/_e14_train-band/logs
G1=CoffeeMartini,FlameSteak,MATF,AlexaMeadeExhibit,Fencing,Frog,Dog,AlexaMeadeFacePaint
G2=Breakfast,CBABasketball,Painter,Barn,MartialArts,PoznanStreet,Carpark,Welder,Blocks
for P in ${PRESETS:-geoset_decb_rep_x geoset_dec_rep_x geoset_cut0_rep_x}; do
  for g in 1 2; do
    if [ $g = 1 ]; then S=$G1; else S=$G2; fi
    t0=$(date +%s)
    PYTHONDONTWRITEBYTECODE=1 flock $LOCK scripts/docker_run.sh scsam3 python geomem/runs/_e14_train-band/code/dump_arm.py \
      --preset $P --scenes $S > $L/dump_${P}_g$g.log 2>&1
    rc=$?
    echo "$(date '+%F %T') $P group $g exit $rc wall $(( $(date +%s) - t0 ))s" >> $L/chain.log
  done
done
echo "$(date '+%F %T') chain done" >> $L/chain.log
