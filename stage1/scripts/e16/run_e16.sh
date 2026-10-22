#!/usr/bin/env bash
# S1-E16 runner (docs/stage1-E16-prereg.md): the geometry-free control and the seed study.  Measurement only.
#   run_e16.sh train <arm> <seed> <fold>      one training -> geomem/runs/_e16/seed<seed>/<arm>/fold<fold> (scsam3, GPU)
#   run_e16.sh train-queue                    nogeo_pred_ft seeds 0 1 2 x folds 0-4, geoset_pred_ft seeds 1 2 x folds 0-4
#                                             (skips a fold whose ckpt.pt exists; stops at the first failure)
#   run_e16.sh dev <arm> <seed> <rot>         dev_decode --e16-arm <arm> on DEV-A (rot 0) / DEV-A' (rot 1, 2) with the seed's
#                                             checkpoints, then dev_score -> geomem/runs/_e16/scores/<arm>_s<seed>_A[_rot<r>].json
#   run_e16.sh t06 <arm> <seed>               T06 (fold 2) decode of one arm with run.py -> Result/e16_<arm>_s<seed>
#   run_e16.sh decode-queue                   every (arm, seed, rot) of prereg §4 that is not scored yet, then every T06 arm
#   run_e16.sh t06-score | score             T06 scores (ring_diag score.py) / the DEV comparisons vs C0 -> geomem/runs/_e16/summary.json (host)
#   run_e16.sh after-train                    detached chain: wait for the training queue -> decode-queue -> t06-score -> score
# <arm>: NG-idx NG-feat NG-pi3 G G-bias0 AB star.  Seeds: 0 1 2 (star: 0 only; G / AB seed 0 reuse the E14 / E15 decodes).
set -euo pipefail
E16=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$E16/../.." && pwd)
MAIN=$(cd "$REPO/.." && pwd)
WORK=$REPO/geomem/runs/_e16
LOCK=$REPO/geomem/runs/_e15/gpu.lock                    # one GPU job at a time, shared with the E15 runner
PY=$REPO/.venv/bin/python
cd "$REPO"
mkdir -p "$WORK/logs" "$WORK/scores"
ts() { date '+%F %T'; }
gpu_lock() { exec 9>"$LOCK"; flock 9; }
runs_of() { local seed=$1 arm=$2; if [ "$seed" = 0 ] && [ "$arm" != nogeo_pred_ft ]; then echo "$REPO/geomem/runs"; else echo "$WORK/seed$seed"; fi; }
arm_of() { case "$1" in NG-*) echo nogeo_pred_ft ;; G|G-bias0|AB|star) echo geoset_pred_ft ;; *) echo "unknown arm $1" >&2; exit 2 ;; esac; }
t06_flags() {   # run.py flags of an E16 arm (prereg §2)
  case "$1" in
    NG-idx)  echo "--arm nogeo_pred_ft --gate obj --xcl outs --nbr index" ;;
    NG-feat) echo "--arm nogeo_pred_ft --gate obj --xcl outs --nbr feat" ;;
    NG-pi3)  echo "--arm nogeo_pred_ft --gate reproj --xcl outs" ;;
    G)       echo "--arm geoset_pred_ft --gate reproj --xcl outs" ;;
    G-bias0) echo "--arm geoset_pred_ft --gate reproj --xcl outs --zero-bias" ;;
    AB)      echo "--arm geoset_pred_ft --gate reproj --xcl outs --relay 60 --id3d cstar" ;;
    star)    echo "--arm geoset_pred_ft --gate reproj --xcl outs --untrained" ;;
  esac
}
cmd=${1:-}; shift || true
case "$cmd" in
  train)
    arm=$1; seed=$2; fold=$3
    out=$WORK/seed$seed
    [ -f "$out/$arm/fold$fold/ckpt.pt" ] && { echo "exists: $out/$arm/fold$fold"; exit 0; }
    gpu_lock
    scripts/run_geomem_train.sh "$arm" "$fold" --seed "$seed" --out "/host$out" ;;
  train-queue)
    for seed in 0 1 2; do for fold in 0 1 2 3 4; do
      echo "$(ts) train nogeo_pred_ft $seed $fold"; "$0" train nogeo_pred_ft "$seed" "$fold" > "$WORK/logs/train_nogeo_s${seed}_f${fold}.log" 2>&1 || { echo "$(ts) FAILED"; exit 1; }
    done; done
    for seed in 1 2; do for fold in 0 1 2 3 4; do
      echo "$(ts) train geoset_pred_ft $seed $fold"; "$0" train geoset_pred_ft "$seed" "$fold" > "$WORK/logs/train_geoset_s${seed}_f${fold}.log" 2>&1 || { echo "$(ts) FAILED"; exit 1; }
    done; done
    echo "$(ts) train queue done" ;;
  dev)
    arm=$1; seed=$2; rot=$3
    a=$(arm_of "$arm"); runs=$(runs_of "$seed" "$a")
    [ -f "$runs/$a/fold0/ckpt.pt" ] || { echo "no checkpoints: $runs/$a" >&2; exit 2; }
    name="${arm}_s${seed}_A$([ "$rot" = 0 ] || echo "_rot$rot")"
    root="$REPO/Data/_dev/e16/$name"
    extra=(); [ "$arm" = star ] && extra=(--seed-tag s0)
    [ "$arm" = NG-idx ] && [ "$seed" = 0 ] && [ "$rot" = 0 ] && extra+=(--validate)
    gpu_lock
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_decode --e16-arm "$arm" --runs "/host$runs" --seed-tag "s$seed" \
      --split A --ref-rot "$rot" --root "/host$root" "${extra[@]}"
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_score --masks "/host$root" --out "/host$WORK/scores/$name.json" ;;
  t06)
    arm=$1; seed=$2
    a=$(arm_of "$arm"); runs=$(runs_of "$seed" "$a")
    # shellcheck disable=SC2046
    gpu_lock
    scripts/docker_run.sh scsam3 python methods/geomem/run.py --scene T06_flag_ganggang_F --out "e16_${arm}_s${seed}" \
      --runs "/host$runs" $(t06_flags "$arm") --overwrite ;;
  decode-queue)
    for arm in NG-idx NG-feat NG-pi3 G G-bias0 AB star; do
      for seed in 0 1 2; do
        [ "$arm" = star ] && [ "$seed" != 0 ] && continue
        [ "$arm" = G ] && [ "$seed" = 0 ] && continue       # = Data/_dev/geoset_rep_xs_A{,_rot1,_rot2} (E14 / E15)
        [ "$arm" = AB ] && [ "$seed" = 0 ] && continue      # = Data/_dev/e15/AB_60_cstar_A{,_rot1,_rot2} (E15)
        for rot in 0 1 2; do
          name="${arm}_s${seed}_A$([ "$rot" = 0 ] || echo "_rot$rot")"
          [ -f "$WORK/scores/$name.json" ] && continue
          echo "$(ts) dev $arm $seed $rot"; "$0" dev "$arm" "$seed" "$rot" > "$WORK/logs/dev_$name.log" 2>&1 || { echo "$(ts) FAILED dev $arm $seed $rot"; exit 1; }
        done
      done
    done
    for arm in NG-idx NG-feat NG-pi3 G G-bias0 AB star; do
      for seed in 0 1 2; do
        [ "$arm" = star ] && [ "$seed" != 0 ] && continue
        [ -d "$REPO/Data/T06_flag_ganggang_F/Result/e16_${arm}_s${seed}" ] && continue
        echo "$(ts) t06 $arm $seed"; "$0" t06 "$arm" "$seed" > "$WORK/logs/t06_${arm}_s${seed}.log" 2>&1 || { echo "$(ts) FAILED t06 $arm $seed"; exit 1; }
      done
    done
    echo "$(ts) decode queue done" ;;
  t06-score)
    T06_RES=$REPO/Data/T06_flag_ganggang_F/Result
    args=("C0=$T06_RES/e15_C0" "C0_r1=$T06_RES/e15_C0_r1" "AB_60_cstar=$T06_RES/e15_AB_60_cstar")
    for d in "$T06_RES"/e16_*; do b=$(basename "$d"); case "$b" in *_r1) continue ;; esac; args+=("$b=$d"); done
    "$PY" "$REPO/scripts/ring_diag/d1_d5/score.py" "$WORK/t06_scores.json" "${args[@]}" ;;
  score)
    "$PY" "$E16/e16_score.py" "$@" ;;
  after-train)   # wait for the training queue, then decode everything, score T06 and write the summary (host, detached)
    until grep -q 'train queue done' "$WORK/logs/train_queue.log" 2>/dev/null; do
      grep -q '^.*FAILED' "$WORK/logs/train_queue.log" 2>/dev/null && { echo "$(ts) training failed; stop"; exit 1; }; sleep 120; done
    echo "$(ts) training done -> decode queue"
    "$0" decode-queue || exit 1
    "$0" t06-score && "$0" score && echo "$(ts) E16 chain done" ;;
  *)
    sed -n '2,/^set -euo pipefail/{/^set -euo/!p}' "$0"; exit 2 ;;
esac
