# GeoSet-2 — 객체 단위 게이트와 예측 기억 학습으로 2라운드 기억 집합을 본선으로 만드는 A-3 설계

> 2026-10-09. GeoMem-2([2026-10-07-geomem2-design.md](2026-10-07-geomem2-design.md), 결과 [results-geomem2.md](../../results-geomem2.md))의
> 2라운드 기억 집합(`geoset_ft`)을 다듬는 단계. B-1(GeoQuery, [results-geoquery.md](../../results-geoquery.md) §8)은 두 번의 널 결과로 닫았다.
> 벤치마크 규약은 같다(17장면 × 주석 카메라 2대 × 객체 = 664쌍, 기준 프레임, J·DAVIS F, 장면 부트스트랩 10,000회).

## 0. 한 문장

`geoset_ft`(0.909)의 2라운드는 핀홀 장면(Blocks +0.024, CBABasketball +0.045)을 올리지만 어안 장면 Dog(−0.036)·Welder(−0.023)를 깎아 1라운드 대비
유의하지 않다(J +0.004 [−0.004, +0.013]). 손실은 **몇 개 객체**에 집중된다(Welder camera_0003 metal_part 477 px: J 0.565 → 0.079, 예측이 4,715 px로
부풂). 원인 후보 둘을 함께 고친다: (1) 게이트가 **뷰 단위**라 객체 하나의 오류를 못 거른다 → **객체 단위 게이트**; (2) 학습의 두 번째 기억은 GT
마스크인데 추론은 **예측** 마스크라 모델이 틀린 기억을 본 적이 없다 → **예측 기억으로 학습**. 토대는 1라운드 최고인 `geobias_only`로 통일하고,
대조군을 같이 돌려 두 변경의 효과를 각각 쌍체 비교로 가른다.

## 1. 배경 (확인된 사실)

- `geoset_ft`는 `geobias_ft`(3D 임베딩 + 바이어스) 위에 기억 집합을 얹은 것이라 1라운드가 0.906으로, 임베딩 없는 `geobias_only_ft`(0.908)보다 낮다.
- 2라운드 게이트 τ = 0.7(존재 객체의 예측 IoU 평균)은 거의 거르지 않는다(Dog 31/40, Welder 38/45, 나머지 장면은 전부 통과). 2라운드는 사실상
  "π³ 카메라 중심 최근접 3뷰를 항상 추가"로 동작했다(results-geomem2 §6).
- 기억 토큰은 객체별 배치다: `MemoryView.feats [5184, B, 64]`, 포인터 토큰 `[4, B, 64]`. 기억 어텐션 패치(`geomem/attn_patch.py`)의 질의는
  `[B, heads, Lq, d]`이고 가산 마스크 `tracker._geo_attn_mask`는 마지막 두 차원만 검사하므로 `[B, 1, Lq, ΣL]` 마스크가 그대로 브로드캐스트된다.
- 학습(`geomem/train.py:train_one_step`)의 두 번째 기억은 `encode_reference(mem_image, mem_backbone, mem_masks=GT)`이고, 추론
  (`methods/geomem/run.py`)의 두 번째 기억은 `encode_memory_from_prediction(round1 logits, presence, obj_ptr)`이다.
- 전 뷰 백본 특징 캐시(`sam3_feats`, 17장면 × 11프레임 × 전 카메라 4,059뷰)와 프레임별 π³ 격자(`pi3_frames`)는 GeoQuery 때 만들어 두었다.
  비주석 카메라를 학습의 기억 출처로 쓰는 데 데이터 준비가 필요 없다.
- 어안 장면의 π³ 기하는 핀홀 가정으로 복원돼 신뢰도가 낮고(Welder camera_0003의 최근접 3대가 38·10·11번), 데이터셋에 GT 캘리브레이션이
  없어 이웃 선택 기준의 옳고 그름은 검증할 수 없다. **이번 설계는 이웃 선택 기준을 바꾸지 않는다**(6절).
- 함정: 프로세스 전역 autocast 캐시 off 필수(`methods/sam3_single.load_model`); GPU 작업 동시 실행 금지; 상류 파일 수정 금지.

## 2. 객체 단위 게이트 (합의 2절)

### 2.1 확신 벡터

- `MemoryView`에 `conf: torch.Tensor | None` (`[B]` bool)을 추가한다. `None`은 전부 참. `encode_reference`(GT 기억)는 `None`,
  `encode_memory_from_prediction`은 `conf = (presence > 0) & (ious >= tau)`를 넣는다(`tau` 인자, 기본 `TAU_CONF = 0.7`).
- 기준 뷰(집합의 첫 뷰)는 항상 전부 참이어야 한다(`MemorySet.gate_mask`가 단언).

### 2.2 마스크

- `MemorySet.gate_mask(dtype, device) -> Tensor | None`: 어떤 뷰에도 거짓이 없으면 `None`. 아니면 `[B, 1, 1, ΣL]` 가산 마스크로, 뷰 v의 특징 토큰
  5,184개 열(`prompt()`의 앞부분, 뷰 순서대로)과 포인터 토큰 4개 열(뒷부분, 뷰 순서대로)에 대해 `conf_v[b]`가 거짓인 객체 b의 행에 `-inf`,
  나머지 0. 기준 뷰 열은 언제나 0이므로 어떤 행도 전부 `-inf`가 되지 않는다.
- `GeoMemStar.decode_view`: `mask = geo_bias(...)` (`[1, 1, Lq, ΣL]`)에 `gate_mask`를 더해 `[B, 1, Lq, ΣL]`로 넘긴다. `use_bias`가 거짓인 팔
  (`geomem*`)은 게이트가 있을 때만 게이트 마스크를 단독으로 넘긴다(이번 팔들은 모두 `use_bias`가 참이라 경로가 바뀌지 않는다).
- 확신 객체가 하나도 없는 뷰는 집합에 넣지 않는다(추론 3.2, 학습 4.2). 뷰를 빼는 것과 모든 객체를 막는 것은 어텐션상 동치지만 메모리를 아낀다.
- 전부 참이면 마스크가 `None`이라 기존 경로와 **바이트 동일**하다(5절 테스트).

## 3. 추론 (`methods/geomem/run.py`)

### 3.1 게이트 선택지

- `--gate {view,obj}` (기본 `view`). 결과 이름: 1라운드 `<arm>_r1`, 2라운드 `view`는 `<arm>`(기존과 같음), `obj`는 `<arm>_obj`.
- `view`: 기존 그대로. 뷰 평균 IoU ≥ τ인 뷰만 확신 뷰로 두고, 표적마다 최근접 k = 3 중 확신 뷰를 넣는다. 확신 벡터는 쓰지 않는다(`conf=None`).
- `obj`: 뷰 필터 없이 표적마다 최근접 k = 3 뷰를 후보로 하고, 각 후보의 1라운드 예측으로 `conf_v`를 만든다. `conf_v`가 전부 거짓인 뷰는 뺀다.
  기억 집합 = c_ini + 남은 후보(확신 벡터 포함). τ = 0.7은 뷰 게이트와 같은 값이며 테스트 점수로 고르지 않는다.

### 3.2 진단 기록

- 1라운드가 끝나면 장면마다 **2라운드 결과 폴더** `Result/<out>/round1_conf.json`에 `{cam: {obj_id: {"iou": float, "presence": float}}}`를 쓴다(두 게이트 공통).
  `--r1-out NAME|none`(기본 `<out>_r1`)으로 1라운드 폴더 이름을 정하며, 뷰 게이트 실행이 이미 `<arm>_r1`을 썼으면 객체 게이트 실행은 `--r1-out none`으로 돌려 같은 결과를 다시 쓰지 않는다.
- 2라운드 `RUN.json`의 `config`에 `gate`, `tau`, `k`, `mem_per_view_mean`, `views_with_ref_only_frac`, 그리고 `obj`일 때
  `gated_pairs_frac`(후보 뷰 × 객체 중 막힌 비율)·`views_dropped`(확신 객체가 없어 뺀 뷰 수)를 더한다.

### 3.3 평가 행렬

| 팔 | r1 | r2 view | r2 obj |
|---|---|---|---|
| `geoset_ft` (기존 체크포인트) | 있음 | 있음 | 새로 |
| `geoset_only_ft` | 새로 | 새로 | 새로 |
| `geoset_pred_ft` | 새로 | 새로 | 새로 |

새 채점 7회. 모두 하나의 백그라운드 작업 안에서 순차 실행한다.

## 4. 학습 (`geomem/train.py`, `geomem/data.py`) (합의 1절)

### 4.1 공통 레시피

두 팔 모두 `geobias_only` 토대(`use_bias=True, use_geometry=False, no_rope=False`), SAM 3 초기값에서 3 epoch, 프레임 간격 2, 객체 ≤ 16,
AdamW(GeoBias 1e-3, 기억 어텐션 1e-5, wd 0.01), 코사인 감쇠, bf16 autocast(캐시 off), 모델 선택 없음, 두 번째 기억 확률
`EXTRA_MEM_PROB = 0.5`, 5-fold 장면 교차검증(`geomem/folds.json`). 기존 `geoset_ft`와 다른 점은 토대뿐이다.

### 4.2 팔

- `geoset_only_ft` (대조군): 두 번째 기억 = 세 번째 주석 카메라의 GT 마스크(`encode_reference`, `conf=None`). 기존 `geoset_ft` 레시피 그대로.
- `geoset_pred_ft` (처리군): 두 번째 기억의 출처 카메라 m을 **같은 프레임의 기준·표적 이외 모든 카메라**(비주석 포함)에서 균등 추출한다
  (`data.random_memory_camera(data, sample, rng)`; 특징 캐시가 없으면 `FileNotFoundError`로 멈춘다). 배치에는 m의 백본과 격자만 싣는다
  (`load_sample(..., extra_mem_cam=m, extra_mem_gt=False)`). 스텝 안에서:
  1. `torch.no_grad()` + 현재 가중치로 `decode_view(mem_backbone, B, ref, p_mem, valid_mem)` → `logits, presence, ious, obj_ptr`.
  2. `encode_memory_from_prediction(..., tau)` → `conf`. `conf`가 전부 거짓이면 그 스텝은 기억 하나로 간다(`n_mem = 1`).
  3. 기억 집합 c_ini + m으로 표적을 디코딩하고 손실은 기존과 같다(`losses.supervised_loss`).
  예측 경로에는 기울기가 흐르지 않는다(1단계 전체가 `no_grad`, 기억 텐서는 분리). 학습 로그 `log.csv`에 `n_mem`과 함께 `gated_frac`(그 스텝에서
  막힌 객체 비율; 기억이 하나면 빈칸)을 더한다.
- 이름은 `geomem/train.py:ARMS`·`BIAS_ARMS`·`arm_flags`, `methods/geomem/run.py:SET_ARMS`에 등록한다. `arm_flags`에 `extra_mem_pred: bool`을 둔다.

### 4.3 비용

`geoset_ft`가 fold당 30~34분(1,818 s)이었다. `geoset_only_ft`는 같고, `geoset_pred_ft`는 스텝의 절반에서 디코딩이 하나 늘어 약 45분으로 본다.
합계 약 6.5 GPU-시간, 순차 실행.

## 5. 테스트

- 호스트(`tests/`): `MemorySet.gate_mask`의 모양 `[B,1,1,ΣL]`, 거짓 객체의 행에서 그 뷰의 5,184 + 4열만 `-inf`, 기준 뷰 열은 0, 전부 참이면 `None`,
  기준 뷰에 거짓이 있으면 단언 실패. `random_memory_camera`가 기준·표적을 제외한 카메라만 뽑는다.
- 토치(`tests_torch/`): (a) `conf=None`인 두 뷰 집합의 `decode_view` 출력이 게이트 코드 전후로 바이트 동일; (b) 한 객체를 두 번째 뷰에서 막으면
  그 객체의 출력은 "c_ini만 있는 집합"의 출력과 부호 일치 > 99.99 %·상대 최대 차 < 5 %(SDPA 커널 차이 허용 기준, results-geomem2 §7)이고 다른 객체는
  바뀌지 않는다; (c) `geoset_pred_ft` 스모크 학습(2스텝)에서 손실이 유한하고 기울기가 GeoBias·기억 어텐션에만 있으며 `gated_frac`이 기록된다;
  (d) `run.py --arm geoset_ft --gate obj`를 장면 하나(Blocks)에 돌려 `_r1`·`_obj` 결과와 `round1_conf.json`이 쓰이고 `RUN.json`에 3.2의 키가 있다.

## 6. 범위 밖

- 이웃 선택 기준(π³ 중심 거리, k = 3)과 τ 값의 변경. 어안 기하 보정. 세 번째 라운드. 디코더 학습(B-2). 10 epoch 전체 프레임 재학습.
- `geoset_ft`(옛 토대)의 재학습. 옛 체크포인트는 r2 obj 평가에만 쓴다.

## 7. 사전 등록한 비교와 성공 기준

| 비교 | 가르는 것 |
|---|---|
| `geoset_ft_obj − geoset_ft` | 객체 게이트만의 효과(학습 없음) |
| `geoset_only_ft_r1 − geobias_only_ft` | 토대 동등성(≈ 0이어야 함; 아니면 레시피 차이를 먼저 찾는다) |
| `geoset_pred_ft_obj − geoset_only_ft_obj` | 예측 기억 학습의 효과(게이트 동일) |
| `geoset_pred_ft_obj − geoset_pred_ft_r1` | 2라운드 이득이 유의한가 |
| `geoset_pred_ft_obj − geobias_only_ft`, `− geoset_ft` | 헤드라인 |

성공: 어떤 장면도 자기 r1보다 J&F 0.01 넘게 떨어지지 않고, 전체 J&F ≥ 0.909, 2라운드 − 1라운드의 J CI가 0을 뺀다. 결과 문서
`docs/results-geoset2.md`에는 위 비교 전부, 장면별·축별 표, 주석 카메라에서 1라운드 예측 IoU 대 실제 J의 상관(게이트의 신뢰도), 게이트 통계
(어안 4장면 별도)를 적는다. 실패해도 같은 문서에 적고 다음 단계는 정하지 않는다.

## 8. 위험

- 예측 기억 학습 초기에는 예측이 나빠 대부분 막히고, 모델이 "두 번째 기억은 무시"를 배울 수 있다. 그래도 추론 분포와 같으므로 해가 되지는 않는다.
  `gated_frac` 궤적으로 확인한다.
- `-inf` 마스크와 bf16: 기준 뷰 열이 항상 0이라 전부 `-inf`인 행은 없다. 그래도 (b) 테스트가 NaN을 잡는다.
- 2라운드는 1라운드 결과 전부를 GPU에 들고 있어 최대 19.5 GB였다. 확신 벡터는 작아 메모리 변화가 없다.
