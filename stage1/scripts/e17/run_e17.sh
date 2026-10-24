#!/usr/bin/env bash
# S1-E17 runner (docs/stage1-E17-prereg.md, part 1: C-ov on MUVOD only).
#   run_e17.sh angles                         the input-only training angle cache -> scripts/e17/runs/train_angles.json (scsam3, CPU)
#   run_e17.sh train <arm> <seed> <fold>      one training -> geomem/runs/_e17/seed<seed>/<arm>/fold<fold> (scsam3, GPU)
#   run_e17.sh dev <C-ov2|C-ov4> <seed> <rot> dev_decode --e17-arm on DEV-A (rot 0) / DEV-A' (rot 1, 2) with the seed's checkpoints,
#                                             then dev_score -> geomem/runs/_e17/scores/<arm>_s<seed>_A[_rot<r>].json
#   run_e17.sh t06 <arm> <seed>               T06 (fold 2) decode with the main preset's flags -> Result/e17_<arm>_s<seed>{,_r1}
#   run_e17.sh stage1                         seed 0 of both lambdas: 10 trainings, then DEV-A / DEV-A' decodes (6)
#   run_e17.sh select                         the registered lambda rule (§3) on the seed-0 scores -> scripts/e17/runs/select.json
#   run_e17.sh stage2                         seeds 1, 2 of the chosen lambda (10 trainings), their decodes (6), T06 of every trained arm
#   run_e17.sh t06-score | score              T06 scores (ring_diag score.py, round 1 included) / the summary -> geomem/runs/_e17/summary.json
#   run_e17.sh chain                          stage1 -> select -> stage2 -> t06-score -> score (run detached; idempotent)
# <arm>: C-ov2 (geoset_cov2_ft) C-ov4 (geoset_cov4_ft).  DEV-B / DEV-B' are NOT decoded here (reserved for the Fix C confirmation).
set -euo pipefail
E17=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$E17/../.." && pwd)
MAIN=$(cd "$REPO/.." && pwd)
WORK=$REPO/geomem/runs/_e17
LOCK=$REPO/geomem/runs/_e15/gpu.lock                    # one GPU job at a time, shared with the E15 / E16 / baseline runners
PY=$REPO/.venv/bin/python
ANGLES=$E17/runs/train_angles.json
cd "$REPO"
mkdir -p "$WORK/logs" "$WORK/scores"
ts() { date '+%F %T'; }
gpu_lock() { exec 9>"$LOCK"; flock 9; }
cpu_docker() {   # scsam3 without a GPU
  docker run --rm --shm-size=32g --user "$(id -u):$(id -g)" -v "$MAIN:/host$MAIN" \
    -w "/host$REPO" -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONPATH="/host$REPO" -e HOME=/tmp \
    -e CUDA_VISIBLE_DEVICES="" -e PYTHONDONTWRITEBYTECODE=1 scsam3 "$@"
}
arm_of() { case "$1" in C-ov2) echo geoset_cov2_ft ;; C-ov4) echo geoset_cov4_ft ;; *) echo "unknown arm $1" >&2; exit 2 ;; esac; }
need_angles() { [ -f "$ANGLES" ] || { echo "no angle cache $ANGLES (run_e17.sh angles)" >&2; exit 2; }; }
chosen() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['chosen'])" "$E17/runs/select.json"; }
cmd=${1:-}; shift || true
case "$cmd" in
  angles)
    cpu_docker python scripts/e17/train_angles.py "$@" ;;
  train)
    arm=$(arm_of "$1"); seed=$2; fold=$3
    need_angles
    out=$WORK/seed$seed
    [ -f "$out/$arm/fold$fold/ckpt.pt" ] && { echo "exists: $out/$arm/fold$fold"; exit 0; }
    gpu_lock
    scripts/run_geomem_train.sh "$arm" "$fold" --seed "$seed" --out "/host$out" --angles "/host$ANGLES" ;;
  train-seed)   # every fold of one (arm, seed), stop at the first failure
    for fold in 0 1 2 3 4; do
      [ -f "$WORK/seed$2/$(arm_of "$1")/fold$fold/ckpt.pt" ] && continue
      echo "$(ts) train $1 $2 $fold"; "$0" train "$1" "$2" "$fold" > "$WORK/logs/train_$1_s$2_f$fold.log" 2>&1 || { echo "$(ts) FAILED train $1 $2 $fold"; exit 1; }
    done ;;
  dev)
    name_arm=$1; seed=$2; rot=$3
    a=$(arm_of "$name_arm"); runs=$WORK/seed$seed
    [ -f "$runs/$a/fold0/ckpt.pt" ] || { echo "no checkpoints: $runs/$a" >&2; exit 2; }
    name="${name_arm}_s${seed}_A$([ "$rot" = 0 ] || echo "_rot$rot")"
    root="$REPO/Data/_dev/e17/$name"
    extra=(); [ "$name_arm" = C-ov2 ] && [ "$seed" = 0 ] && [ "$rot" = 0 ] && extra+=(--validate)
    gpu_lock
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_decode --e17-arm "$name_arm" --runs "/host$runs" --seed-tag "s$seed" \
      --split A --ref-rot "$rot" --root "/host$root" "${extra[@]}"
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_score --masks "/host$root" --out "/host$WORK/scores/$name.json" ;;
  dev-seed)     # the three DEV decodes of one (arm, seed)
    for rot in 0 1 2; do
      name="$1_s$2_A$([ "$rot" = 0 ] || echo "_rot$rot")"
      [ -f "$WORK/scores/$name.json" ] && continue
      echo "$(ts) dev $1 $2 $rot"; "$0" dev "$1" "$2" "$rot" > "$WORK/logs/dev_$name.log" 2>&1 || { echo "$(ts) FAILED dev $1 $2 $rot"; exit 1; }
    done ;;
  t06)
    name_arm=$1; seed=$2
    a=$(arm_of "$name_arm"); runs=$WORK/seed$seed
    [ -f "$runs/$a/fold2/ckpt.pt" ] || { echo "no fold-2 checkpoint: $runs/$a" >&2; exit 2; }
    gpu_lock
    scripts/docker_run.sh scsam3 python methods/geomem/run.py --scene T06_flag_ganggang_F --out "e17_${name_arm}_s${seed}" \
      --runs "/host$runs" --arm "$a" --gate reproj --xcl outs --relay 60 --id3d cstar --overwrite ;;
  stage1)
    for arm in C-ov2 C-ov4; do "$0" train-seed "$arm" 0 || exit 1; done
    for arm in C-ov2 C-ov4; do "$0" dev-seed "$arm" 0 || exit 1; done
    echo "$(ts) stage1 done" ;;
  select)
    "$PY" "$E17/e17_score.py" --select ;;
  stage2)
    [ -f "$E17/runs/select.json" ] || { echo "no select.json (run_e17.sh select)" >&2; exit 2; }
    c=$(chosen)
    for seed in 1 2; do "$0" train-seed "$c" "$seed" || exit 1; done
    for seed in 1 2; do "$0" dev-seed "$c" "$seed" || exit 1; done
    for arm in C-ov2 C-ov4; do for seed in 0 1 2; do
      [ -f "$WORK/seed$seed/$(arm_of "$arm")/fold2/ckpt.pt" ] || continue
      [ -d "$REPO/Data/T06_flag_ganggang_F/Result/e17_${arm}_s${seed}" ] && continue
      echo "$(ts) t06 $arm $seed"; "$0" t06 "$arm" "$seed" > "$WORK/logs/t06_${arm}_s${seed}.log" 2>&1 || { echo "$(ts) FAILED t06 $arm $seed"; exit 1; }
    done; done
    echo "$(ts) stage2 done" ;;
  t06-score)
    T06_RES=$REPO/Data/T06_flag_ganggang_F/Result
    args=("C0=$T06_RES/e15_C0" "C0_r1=$T06_RES/e15_C0_r1" "AB_60_cstar=$T06_RES/e15_AB_60_cstar" "AB_60_cstar_r1=$T06_RES/e15_AB_60_cstar_r1")
    for d in "$T06_RES"/e17_*; do [ -d "$d" ] || continue; b=$(basename "$d"); args+=("$b=$d"); done
    "$PY" "$REPO/scripts/ring_diag/d1_d5/score.py" "$WORK/t06_scores.json" "${args[@]}" ;;
  score)
    "$PY" "$E17/e17_score.py" "$@" ;;
  chain)
    need_angles
    "$0" stage1 && "$0" select && "$0" stage2 && "$0" t06-score && "$0" score && echo "$(ts) E17 chain done" ;;
  *)
    sed -n '2,/^set -euo pipefail/{/^set -euo/!p}' "$0"; exit 2 ;;
esac
