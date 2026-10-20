#!/usr/bin/env bash
# D1-D4 arms, one decode at a time (the preset arm itself ran before the --zero-bias code change).
set -uo pipefail
D15=/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_d15
R=$D15/run_arm.sh
$R data preset_post --arm geoset_pred_ft --gate reproj --xcl outs --r1-out preset_post_r1     # default path after the code change
$R data star --arm geomem --untrained                                                         # D2a: registered `star` control
$R data untr --arm geoset_pred_ft --untrained --gate reproj --xcl outs --r1-out untr_r1        # D2b: preset decode, untrained weights
$R data bias0 --arm geoset_pred_ft --zero-bias --gate reproj --xcl outs --r1-out bias0_r1      # D3
$R data xnone --arm geoset_pred_ft --gate reproj --xcl none --r1-out xnone_r1                  # D4: exclusivity off (and L > 0)
$R data xout --arm geoset_pred_ft --gate reproj --xcl out --r1-out xout_r1                     # D4 decomposition: exclusivity, L > 0
echo CHAIN1_DONE
