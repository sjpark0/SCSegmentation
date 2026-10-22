# S1-E16 사전 등록 — 기하 없는 미세조정 대조(NG)와 편향 0 디코딩, 시드 3 (측정만, 선택 없음)

> 2026-10-22 작성·커밋(코드와 같은 커밋). 계획 [iccv2027-plan.md](iccv2027-plan.md) WP4 절제 1번("미리 선언한 측정이며, 무엇을 고르는 데 쓰지 않습니다"). 개발 집합 규약 [stage1-dev-protocol.md](stage1-dev-protocol.md). S1-E15 판정([stage1-E15.md](stage1-E15.md), G1 통과) 뒤 첫 실험.
> 이 문서는 **측정 등록**입니다. 주 팔(`geoset_rep_xs_ab`, θ_s\* 60, s\* A-c\*)과 어떤 상수도 바꾸지 않고, DEV-B·DEV-B′·TEST는 쓰지 않습니다. 결과는 논문의 절제 표(WP4)와 재학습 잡음 추정에 씁니다. 키프레임(TEST) 수치는 계획 §4의 TEST 1회(12-28~01-22)에서 다른 팔과 함께 한 번만 냅니다.

## 1. 질문

학습형 1단계(`geoset_pred_ft` = SAM 3 기억 어텐션 미세조정 + 3D 어텐션 편향 GeoBias, π³ 이웃·reproj 게이트로 디코딩)의 이득 중 **기하 몫**과 **미세조정 몫**을 가릅니다.
- 기하가 전혀 없는 같은 레시피(NG)를 같은 예산·시드로 학습하면 얼마나 나오는가.
- 학습한 편향을 디코딩에서 0으로 두면(G-bias0) 얼마나 잃는가(계획의 예상 편향 효과 +0.008).
- 재학습 잡음(시드)의 크기는 얼마인가(계획 추정 약 0.0017). E15 주 팔이 시드에 따라 흔들리는가.

## 2. 팔 (모두 측정; 디코딩 옵션은 `run.py`·`dev_decode.py`의 등록된 값만)

| 팔 | 학습 | 디코딩 | 시드 |
|---|---|---|---|
| **NG-idx** `nogeo_pred_ft` | geoset_pred_ft와 같은 레시피(예측 기억 두 번째 뷰, 3 epoch, stride 2, lr 1e-3/1e-5, wd 0.01, 폴드 5개), `use_bias = False`, `use_geometry = False` | 이웃 k = 3을 **리그 번호** 거리로(`--nbr index`), `--gate obj`, `--xcl outs` | 0, 1, 2 |
| **NG-feat** | 같은 체크포인트 | 이웃을 **SAM 3 전역 특징**(backbone 최저 해상도 맵의 공간 평균, 단위 벡터) 코사인 유사도로(`--nbr feat`), `--gate obj`, `--xcl outs` | 0, 1, 2 |
| **NG+π³** (NG-pi3) | 같은 체크포인트 | π³ 이웃 + `--gate reproj`, `--xcl outs`(C0와 같은 디코딩; 모델만 기하 없음) | 0, 1, 2 |
| **G** | `geoset_pred_ft` 현행(시드 0 = `geomem/runs/geoset_pred_ft`) + 시드 1, 2 재학습 | `geoset_rep_xs`(= C0) | 0(있음), 1, 2 |
| **G-bias0** | G 체크포인트 | `geoset_rep_xs` + `--zero-bias`(GeoBias 매개변수 0; 링 진단 D3와 같은 코드) | 0, 1, 2 |
| **AB** | G 체크포인트 | `geoset_rep_xs_ab`(E15 주 팔: Fix A c\* + 릴레이 θ 60) | 0(E15에 있음), 1, 2 |
| **별** | 학습 없음 | `geoset_rep_xs` + `--untrained`(0 초기화 머리; E13의 별 대조) | — |

- NG는 편향 팔이 아니고(BIAS_ARMS 밖) 3D 임베딩도 없습니다(NO_EMBED_ARMS). 기억 어텐션은 G와 같이 미세조정합니다(FT_ARMS). 디코딩 때 π³ 캐시는 읽히지만 모델이 쓰지 않습니다(`use_bias`·`use_geometry` 꺼짐). NG-idx·NG-feat는 π³ 없이 이웃을 고르고 게이트도 모델 예측(`obj`: presence ∧ IoU ≥ 0.7)만 씁니다.
- `--nbr`는 기하 없는 팔에만 받습니다(코드 단언). 기본값 `pi3`는 바꾸지 않은 경로이며 config에 키를 남기지 않습니다.
- G-bias0의 `zero_bias`와 NG의 `nbr`는 config.json에 기록됩니다. DEV 디코딩은 `dev_decode --e16-arm <팔> --runs <시드 디렉터리>`(기록된 덮어쓰기: index.json·config.json에 `e16_arm`, `runs`, `train_seed`). PRESETS에는 아무것도 더하지 않습니다(E15 I5 유지).

## 3. 학습 (GPU, 25회)

- `nogeo_pred_ft` 시드 0·1·2 × 폴드 0–4 = 15회, `geoset_pred_ft` 시드 1·2 × 폴드 0–4 = 10회. 출력 `stage1/geomem/runs/_e16/seed<k>/<팔>/fold<j>/`(시드 0의 G는 기존 `geomem/runs/geoset_pred_ft`). 명령 `scripts/run_geomem_train.sh <팔> <폴드> --seed k --out geomem/runs/_e16/seed<k>`.
- 시드는 머리 초기화·표본 순서·두 번째 기억 카메라 추출(`memory_rng`)에 들어갑니다. 레시피의 다른 값은 전부 같습니다(config.json으로 확인: epochs 3, frame_stride 2, steps 2 772, lr, weight_decay).
- 한 폴드 약 37분(E14 기록 2 227 s) → 약 15.5 h. 큐 `stage1/scripts/e16/run_e16.sh train-queue`(하나 실패하면 멈춤).

## 4. 디코딩·채점 (GPU, 측정만)

- 집합: **DEV-A**(기준 0)와 **DEV-A′**(기준 회전 1·2). 모든 팔·시드. C0 = `geoset_rep_xs` 시드 0의 기존 디코딩(`Data/_dev/geoset_rep_xs_A{,_rot1,_rot2}`)을 대조로 짝지어 비교합니다(`dev_compare`, 쌍 → 뷰 → 장면 부트스트랩 10 000, seed 0). 루트 `Data/_dev/e16/<팔>_s<시드>_A[_rot<r>]`. AB 시드 0은 E15의 `Data/_dev/e15/AB_60_cstar_A{,_rot1,_rot2}`를 그대로 읽습니다.
- **T06**(fold 2 체크포인트, 900번, `ring_diag/d1_d5/score.py`, 184쌍): 같은 팔·시드를 `run.py`로 디코딩(`--out e16_<팔>_s<시드>`, `--runs`), J_all·wrong_all·Jfull·먼 절반 J. 먼 뷰의 측정이 목적이므로 포함합니다(E15와 같은 개발 용도).
- 제한 디코딩(규약 §2)은 NG-idx·NG-feat에서도 같은 이웃 규칙으로 후보를 고릅니다(`restricted_cams(coords=...)`) — 전체 카메라에서 고른 k개가 모두 제한 집합에 들어 있어 결과가 같습니다(규약의 π³ 경우와 같은 논증). `--validate`로 장면마다 첫 프레임에서 확인합니다(NG-idx 시드 0).
- 채점 `dev_score`(covers·other_obj_mask 포함), 비교 `dev_compare`. 보고 단위: 팔 × 시드, 그리고 시드 평균 ± 표준편차.

## 5. 보고 (규칙 없음, 판정 없음)

- 팔·시드별 DEV-A·DEV-A′: ΔJ&F [CI], ΔJ, ΔF, 장면별, 넓은 쌍(a_v ≥ 55°, DEV-A′) ΔJ, 다른 객체 마스크 수, 환각, 시간. T06: J_all, wrong_all, Jfull, 먼 절반 J.
- 파생량(시드별로 내고 평균): **편향 효과** = G − G-bias0; **기하 학습 효과** = G − NG+π³(같은 디코딩, 학습만 다름); **이웃·게이트 효과** = NG+π³ − NG-idx, NG+π³ − NG-feat; **미세조정 효과** = NG-idx − 별; **재학습 잡음** = G 시드 3개의 표준편차(DEV-A J&F), AB 시드 3개의 표준편차.
- 결과 문서 [stage1-E16.md](stage1-E16.md). 계획 WP4 절제 표에 들어갑니다.

## 6. 예측 (실행 전; 맞고 틀림을 보고만 함)

1. G 시드 1·2의 DEV-A J&F는 시드 0(0.9174 키프레임이 아니라 DEV-A C0 값)과 ±0.003 안. 재학습 잡음 표준편차 ≈ 0.002.
2. G − G-bias0 = +0.004 ~ +0.010(DEV-A), DEV-A′ 넓은 쌍에서는 더 큼(+0.01 이상).
3. NG-idx·NG-feat는 G보다 DEV-A에서 0.01 이상 낮고, T06 J_all < 0.85, wrong_all ≥ 10. NG+π³는 그 사이(DEV-A에서 G와 0.005 안, T06에서는 reproj 게이트 덕에 NG-idx보다 높음).
4. AB 시드 1·2: T06 J_all ≥ 0.92, wrong_all ≤ 5; DEV-A′ 넓은 쌍 ΔJ > 0.
5. 별 < NG-idx < NG+π³ ≤ G 순서가 DEV-A·T06 모두에서 유지.

## 7. 하지 않을 것

- DEV-B·DEV-B′·TEST 디코딩 없음. 주 팔·상수·프리셋 변경 없음. 이 결과로 무엇도 고르지 않습니다.
- `--nbr`·게이트·k = 3·τ = 0.7·xcl outs는 위 표대로 고정. 새 변형을 사후에 돌리지 않습니다(돌리면 별도 등록).
- 시드 3개로 끝냅니다(더 돌리지 않음).

## 8. 예산 (RTX 6000 Ada)

학습 25 × 37분 ≈ 15.5 h. 디코딩: DEV-A 약 15분 + DEV-A′ 2 × 15분 = 45분 × (NG-idx 3 + NG-feat 3 + NG+π³ 3 + G 2 + G-bias0 3 + AB 2 + 별 1 = 17) ≈ 13 h(NG-feat는 프레임마다 전체 카메라 특징 계산이 더해져 조금 더). T06 20회 × 1분. 합계 약 29 h(계획의 20–25 h보다 약간 많음: 계획에 없던 AB 시드와 DEV-A′가 더해짐).

## 9. 코드 (이 커밋)

- `stage1/geomem/arms.py` `NOGEO_ARMS`(PRED_MEM·NO_EMBED에 포함, SET2·BIAS 아님); `geomem/train.py` ARMS·SET_ARMS·FT_ARMS; `methods/geomem/run.py` ARMS·SET_ARMS, `--nbr {pi3,index,feat}`(`camera_coords`, `camera_index_coords`, `camera_feat_coords`; `check_gate_out`의 `nbr`), `config["nbr"]`; `methods/geomem/dev_decode.py` `--e16-arm`, `--runs`, `--seed-tag`, `E16_ARMS`, `restricted_cams(coords=)`, config 기록; `scripts/e16/run_e16.sh`.
- 시험 `tests/test_e16.py`(호스트; 학습 플래그는 torch 컨테이너에서), 기존 시험의 프리셋 목록 단언은 E15 2차 커밋의 주 프리셋을 반영해 고침(`test_e12_cut`, `test_xcl`, `test_zoom`; 규칙 변경 아님).
- 끄면 바이트 동일: `--nbr pi3`(기본)·`--e16-arm` 없음이면 경로와 config가 그대로입니다(E15 Z4와 같은 뜻; `dev_decode --validate`가 NG-idx 시드 0에서 제한/전체 동일을 확인).

## 10. 순서

1. 이 문서 + 코드 커밋 → 2. 학습 큐(약 15.5 h) → 3. 디코딩·채점 큐(DEV-A, DEV-A′, T06) → 4. `stage1-E16.md` 보고(§5), ROADMAP·노션 갱신.

## 11. 이력

| 일자 | 내용 |
|---|---|
| 2026-10-22 | 작성·커밋(G1 통과 당일, 계획 11-16 항목을 당김). |
