#!/usr/bin/env bash
# S1-E15 runner (docs/stage1-E15-prereg.md §6, §7, §8, §11.2).  One E15 GPU job at a time (flock on
# stage1/geomem/runs/_e15/gpu.lock; U3: the T06 grid may share the card with the cart job).  T06 = fold 2 only (I9).
#
#   run_e15.sh step0                          Step 0 (§6.1): angles + rows + N2-N6 (scsam3, CPU only, no --gpus)
#   run_e15.sh step1                          Step 1 (§6.2): s*, c_best -> runs/select.json (host)
#   run_e15.sh t06-c0                         T06 C0 re-decode -> Result/e15_C0{,_r1}; must equal the D1 bytes (§6.1)
#   run_e15.sh t06 <arm> <theta> <family>     one T06 decode -> Result/e15_<arm>_<theta>_<family>; I1 (round-1 PNG tree,
#                                             round1_conf.json bytes) and, in scope R, I2 (star-view final PNGs) vs e15_C0
#   run_e15.sh t06-score                      score every Result/e15_* -> geomem/runs/_e15/t06_scores.json (host)
#   run_e15.sh dev <arm> <theta> <family> <rot> [scenes]
#                                             DEV-A (rot 0) / DEV-A' (rot 1, 2) recorded override + dev_score (§4 M2); only
#                                             the §4 I11 bundles (arm, set, theta, family) pass (dev_decode.e15_registered)
#   run_e15.sh dev-c0 <split> <rot>           C0 = geoset_rep_xs on DEV-A' (A 1|2) or, after the 2nd commit, DEV-B' (B 1|2)
#   run_e15.sh dev-main <rot>                 the main preset geoset_rep_xs_ab on DEV-B (0) / DEV-B' (1, 2) -- 2nd commit only
#   run_e15.sh dev-score <root basename>      dev_score of an existing Data/_dev/<root> (e.g. geoset_rep_xs_A_rot1)
#   run_e15.sh c0-rescore <split>             re-score the E14 C0 Data/_dev/geoset_rep_xs_<split> with the current dev_score
#                                             (+ covers / other_obj_mask; scsam3 on CPU) and check J, F, J&F = E14 pair by
#                                             pair -> runs/c0_rescore.json (A: before Step 2; B: after the 2nd commit)
#   run_e15.sh z4-dog                         Z4 (§6.3b): dev_decode geoset_rep_xs --scenes Dog --validate into the
#                                             separate root Data/_dev/e15/_z4_geoset_rep_xs_A (flags off)
#   run_e15.sh zr <theta> <family>            Z-R (§6.3b): AB on DEV-A (restricted), decode only (no scoring): 60 with s*,
#                                             and a theta in [A_E, A_all) when Step 2 asks (need_zcheck)
#   run_e15.sh z4-fencing                     Z4 part 2 (§6.3b): the GPU torch test on the Fencing keyframe (bytes only) ->
#                                             runs/z4_fencing.log (must show 1 passed before Step 2)
#   run_e15.sh zcheck                         the Z stage comparisons Z4 / Z-R / Z-I2 / Z-sim / Z-T06 (e15_zcheck.py, host)
#                                             -> runs/zcheck.json; the stress decode is `dev stress 20 <s*> 0`, G `dev G 60
#                                             <s*> 0`
#   run_e15.sh step2 | step2b                 §6.4 / §6.5 (host)
#   run_e15.sh judge-B [--audited] | judge-T06 | judge-G1     e15_judge.py (host)
# <arm>: AB stress A B-rep B-obj ABc Ac ABe ABnh ABml G GN oracle (the §4 whitelist); <theta>: 20 35 40 45 50 55 60 inf;
# <family>: c0 cstar vote vote_inf hybrid, or off for B-rep / B-obj / oracle.  A Fix-A family needs the committed Step-1
# record scripts/e15/runs/select.json before any GPU decode (§6.2 L5, §10; checked here on the host and again in run.py /
# dev_decode).  The T06 C0 reference = the D1 decode persisted at geomem/runs/_e15/d1_ref (checked against
# design/d1_t06_c0.sha256, prereg §6.1).
set -euo pipefail
E15=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$E15/../.." && pwd)            # stage1
MAIN=$(cd "$REPO/.." && pwd)
WORK=$REPO/geomem/runs/_e15
SCORES=$WORK/scores
T06=T06_flag_ganggang_F
T06_RES=$REPO/Data/$T06/Result
CHAIN="/host$MAIN/PlenopticServer1/CROOM/SegResult/$T06/SegMaskSam3XW1CGPS4MFsCs10kG2/{cam}/900/{obj}.png"
GTDIR="/host$MAIN/Data/PlenopticFull/$T06/Mask/{cam}/000900.png"
D1=$WORK/d1_ref                           # ring-diagnosis D1 decode (preset/, preset_r1/), untracked; sha256 manifest tracked
D1_MANIFEST=$E15/design/d1_t06_c0.sha256
PY=$REPO/.venv/bin/python
mkdir -p "$WORK" "$SCORES"
export PYTHONDONTWRITEBYTECODE=1

gpu_lock() { exec 9>"$WORK/gpu.lock"; flock 9; }
cpu_docker() {   # scsam3 without a GPU (torch on CPU)
  docker run --rm --shm-size=32g --user "$(id -u):$(id -g)" -v "$MAIN:/host$MAIN" \
    -w "/host$REPO" -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONPATH="/host$REPO" -e HOME=/tmp \
    -e CUDA_VISIBLE_DEVICES="" -e PYTHONDONTWRITEBYTECODE=1 scsam3 "$@"
}
gpu_docker() { "$REPO/scripts/docker_run.sh" scsam3 "$@"; }

arm_flags() {    # the §4 whitelist row of <arm> as run.py flags
  local arm=$1 fam=$2
  case "$arm" in
    B-rep|B-obj|oracle) [ "$fam" = off ] || { echo "$arm takes family off" >&2; exit 2; } ;;
    *) [ "$fam" != off ] || { echo "$arm needs a Fix A family" >&2; exit 2; } ;;
  esac
  case "$arm" in
    AB|stress) echo "--id3d $fam" ;;
    A)      echo "--id3d $fam --relay-mode jacobi" ;;
    B-rep)  echo "" ;;
    B-obj)  echo "--relay-trust obj" ;;
    ABc)    echo "--id3d $fam --flagged chain --chain-dir $CHAIN" ;;
    Ac)     echo "--id3d $fam --relay-mode jacobi --flagged chain --chain-dir $CHAIN" ;;
    ABe)    echo "--id3d $fam --flagged empty" ;;
    ABnh)   echo "--id3d $fam --relay-order angle" ;;
    ABml)   echo "--id3d $fam --relay-mem latest" ;;
    G)      echo "--id3d $fam --id3d-scope global" ;;
    GN)     echo "--id3d $fam --id3d-scope global_nf" ;;
    oracle) echo "--relay-trust gt --gt-dir $GTDIR" ;;
    *) echo "unknown arm $arm (the §4 whitelist)" >&2; exit 2 ;;
  esac
}

step1_guard() {  # §6.2 L5 / §10: a Fix-A family decodes only after the Step-1 commit (run.e15_require_step1)
  [ "$1" = off ] || "$PY" "$E15/e15_select.py" --step check-step1 --family "$1"
}

check_d1() {     # the persisted D1 decode is byte for byte the manifest's (prereg §6.1)
  [ -d "$D1/preset" ] && [ -d "$D1/preset_r1" ] || { echo "no D1 decode at $D1 (prereg §6.1)" >&2; exit 2; }
  (cd "$D1" && sha256sum --quiet --strict -c "$D1_MANIFEST") \
    || { echo "$D1 differs from $D1_MANIFEST: STOP (prereg §6.1)" >&2; exit 2; }
  local n
  n=$(cd "$D1" && find preset preset_r1 -type f | wc -l)
  [ "$n" = "$(grep -c . "$D1_MANIFEST")" ] || { echo "$D1 holds $n files, the manifest $(grep -c . "$D1_MANIFEST")" >&2; exit 2; }
}

check_t06_fold2() {
  "$PY" - "$REPO/Data/$T06/meta.json" <<'EOF'
import json, sys
f = json.load(open(sys.argv[1])).get("external_fold")
sys.exit(0 if f == 2 else f"T06 meta external_fold is {f}, not 2 (prereg I9)")
EOF
}

main_preset_exists() {   # the second registration commit appended geoset_rep_xs_ab (prereg §6.7)
  "$PY" -c "import sys; sys.path.insert(0, '$REPO'); from methods.geomem.stage1 import PRESETS; sys.exit(0 if 'geoset_rep_xs_ab' in PRESETS else 1)"
}

same_tree() {    # PNG trees byte-identical (RUN.json etc. ignored)
  diff -rq -x 'RUN.json' -x '*.json' "$1" "$2" >/dev/null
}

cmd=${1:-}; shift || true
case "$cmd" in
  step0)
    check_d1
    cpu_docker python scripts/e15/e15_select.py --step 0 "$@" ;;
  step1|step2|step2b)
    "$PY" "$E15/e15_select.py" --step "${cmd#step}" "$@" ;;
  t06-c0)
    check_t06_fold2; check_d1; gpu_lock
    gpu_docker python methods/geomem/run.py --scene "$T06" --out e15_C0 --arm geoset_pred_ft --gate reproj --xcl outs --overwrite
    { same_tree "$T06_RES/e15_C0" "$D1/preset" && same_tree "$T06_RES/e15_C0_r1" "$D1/preset_r1" \
      && cmp -s "$T06_RES/e15_C0/round1_conf.json" "$D1/preset/round1_conf.json" \
      && cmp -s "$T06_RES/e15_C0/xcl_conf.json" "$D1/preset/xcl_conf.json"; } \
      || { echo "T06 C0 re-decode differs from the D1 bytes: STOP (prereg §6.1)" >&2; exit 2; } ;;
  t06)
    arm=$1; theta=$2; fam=$3; shift 3
    case "$arm:$theta" in
      stress:*|*:20) echo "theta 20 / the stress arm is registered on DEV-A only, not T06 (prereg §4)" >&2; exit 2 ;;
    esac
    check_t06_fold2
    [ -d "$T06_RES/e15_C0_r1" ] || { echo "run t06-c0 first" >&2; exit 2; }
    name="${arm}_${theta}_${fam}"
    # shellcheck disable=SC2046
    flags=$(arm_flags "$arm" "$fam")
    step1_guard "$fam"
    gpu_lock
    # shellcheck disable=SC2086
    gpu_docker python methods/geomem/run.py --scene "$T06" --out "e15_$name" --arm geoset_pred_ft --gate reproj --xcl outs \
      --relay "$theta" $flags --overwrite "$@"
    "$PY" "$E15/e15_zcheck.py" --t06-invariants "$name" || { echo "I1 / I2 differ from e15_C0: a bug (prereg §8)" >&2; exit 2; } ;;
  t06-score)
    args=("C0=$T06_RES/e15_C0" "C0_r1=$T06_RES/e15_C0_r1")
    for d in "$T06_RES"/e15_*; do
      b=$(basename "$d"); n=${b#e15_}
      case "$n" in C0|C0_r1|*_r1) continue ;; esac
      args+=("$n=$d")
    done
    "$PY" "$REPO/scripts/ring_diag/d1_d5/score.py" "$WORK/t06_scores.json" "${args[@]}" ;;
  dev)
    arm=$1; theta=$2; fam=$3; rot=$4; scenes=${5:-all}
    # shellcheck disable=SC2046
    flags=$(arm_flags "$arm" "$fam")
    step1_guard "$fam"
    root="e15/${arm}_${theta}_${fam}_A$([ "$rot" = 0 ] || echo "_rot$rot")"
    gpu_lock
    # shellcheck disable=SC2086
    gpu_docker python -m methods.geomem.dev_decode --preset geoset_rep_xs --split A --ref-rot "$rot" --e15-arm "$arm" \
      --relay "$theta" $flags --scenes "$scenes"
    gpu_docker python -m methods.geomem.dev_score --masks "Data/_dev/$root" --scenes "$scenes" \
      --out "geomem/runs/_e15/scores/$(basename "$root").json" ;;
  dev-c0)
    split=$1; rot=$2
    case "$rot" in 1|2) ;; *) echo "dev-c0: rot 1 or 2 (the unrotated C0 roots exist)" >&2; exit 2 ;; esac
    root="geoset_rep_xs_${split}_rot${rot}"
    gpu_lock
    gpu_docker python -m methods.geomem.dev_decode --preset geoset_rep_xs --split "$split" --ref-rot "$rot"
    gpu_docker python -m methods.geomem.dev_score --masks "Data/_dev/$root" --out "geomem/runs/_e15/scores/$root.json" ;;
  dev-main)
    rot=$1
    root="geoset_rep_xs_ab_B$([ "$rot" = 0 ] || echo "_rot$rot")"
    gpu_lock
    gpu_docker python -m methods.geomem.dev_decode --preset geoset_rep_xs_ab --split B --ref-rot "$rot"
    gpu_docker python -m methods.geomem.dev_score --masks "Data/_dev/$root" --out "geomem/runs/_e15/scores/$root.json" ;;
  dev-score)
    root=$1
    case "$root" in
      *_B|*_B_rot*) main_preset_exists || { echo "dev-score $root: DEV-B / DEV-B' is scored after the second registration commit (prereg §7.1, §10)" >&2; exit 2; } ;;
    esac
    gpu_lock
    gpu_docker python -m methods.geomem.dev_score --masks "Data/_dev/$root" --out "geomem/runs/_e15/scores/$(basename "$root").json" ;;
  c0-rescore)
    split=$1
    case "$split" in
      A) ;;
      B) main_preset_exists || { echo "c0-rescore B: DEV-B is read after the second registration commit (prereg §7.1)" >&2; exit 2; } ;;
      *) echo "c0-rescore: split A or B" >&2; exit 2 ;;
    esac
    # CPU (no GPU, no lock): J / F / J&F do not depend on the device -- a CPU re-score of DEV-A Dog + Blocks reproduced
    # every field of the E14 (GPU) rows (2026-10-20); about 17 min for DEV-A
    cpu_docker python -m methods.geomem.dev_score --masks "Data/_dev/geoset_rep_xs_$split" --r1 \
      --out "geomem/runs/_e15/scores/geoset_rep_xs_$split.json"
    "$PY" "$E15/e15_select.py" --step rescore-check --split "$split" ;;
  zr)
    theta=$1; fam=$2
    step1_guard "$fam"
    gpu_lock
    gpu_docker python -m methods.geomem.dev_decode --preset geoset_rep_xs --split A --ref-rot 0 --e15-arm AB \
      --relay "$theta" --id3d "$fam" ;;
  z4-dog)
    gpu_lock
    gpu_docker python -m methods.geomem.dev_decode --preset geoset_rep_xs --split A --scenes Dog --validate \
      --root Data/_dev/e15/_z4_geoset_rep_xs_A --overwrite ;;
  z4-fencing)   # the Fencing keyframe byte test of §6.3b Z4 (GPU torch test) -> runs/z4_fencing.log; Z4 counts only with it passing
    gpu_lock
    gpu_docker python -m pytest -q -p no:cacheprovider tests_torch/test_e15_torch.py -k test_z4_zr_fencing_keyframe_bytes \
      2>&1 | tee "$E15/runs/z4_fencing.log"
    grep -qE "1 passed" "$E15/runs/z4_fencing.log" || { echo "z4-fencing: not passed" >&2; exit 2; } ;;
  zcheck)
    "$PY" "$E15/e15_zcheck.py" "$@" ;;
  judge-B)
    "$PY" "$E15/e15_judge.py" --stage B "$@" ;;
  judge-T06)
    "$PY" "$E15/e15_judge.py" --stage T06 "$@" ;;
  judge-G1)
    "$PY" "$E15/e15_judge.py" --stage G1 "$@" ;;
  *)
    sed -n '2,/^set -euo pipefail/{/^set -euo/!p}' "$0"; exit 2 ;;
esac
