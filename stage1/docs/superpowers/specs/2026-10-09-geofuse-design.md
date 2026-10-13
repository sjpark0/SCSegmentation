# GeoFuse — 다중시점 특징을 표적 격자에 정렬·융합하는 B 설계 (A-3 위에 얹는 공동 입력)

> 2026-10-09. 전편: A-3 [results-geoset2.md](../../results-geoset2.md)(`geoset_pred_ft` + 재투영 게이트 0.914, 부재 게이트 0.915), B-1 [results-geoquery.md](../../results-geoquery.md)(객체 쿼리 허브, 닫음).
> 벤치마크 규약은 같다(17장면 × 주석 카메라 2대 × 객체 = 664쌍, 기준 프레임, J·DAVIS F, 장면 부트스트랩 10,000회).

## 0. 한 문장

표적 뷰를 디코딩할 때 **다른 모든 뷰의 백본 특징을 π³ 기하로 표적의 72×72 격자에 정렬해 셀 단위 교차뷰 어텐션으로 융합**하고, 융합된 특징을 기억 어텐션에 넣는다
(`geofuse_ft`). 이것이 B의 정의("다중시점 집합이 한 번에 입력으로 들어가 공동으로 디코딩")를 만족하는 가장 작은 형태다. 융합은 0 초기화 게이트 뒤에 있어 학습 전에는 A-3와
바이트 동일하고, 학습 레시피는 `geoset_pred_ft`와 같으므로 그 팔이 그대로 대조군이다. 한 번의 학습으로 두 답을 낸다: 1라운드끼리의 차이가 **융합의 순수 효과**, 2라운드(재투영 게이트)끼리의 차이가
**최종 헤드라인**이다.

## 1. 배경 (설계를 묶는 사실)

- B-1이 실패한 이유: 객체당 벡터 하나로 요약한 다른 뷰 정보는 동결 디코더가 쓰지 못했다. 반면 A-3의 **공간 정렬된** 기억(픽셀 단위 특징 맵)은 효과가 있었다. 이 설계는 다른 뷰 정보를
  **셀 단위로 정렬한 특징**으로 넣고, 그것을 **학습이 증명된 경로**(기억 어텐션의 입력 `src`; A의 미세조정이 효과를 낸 바로 그 경로)에 넣는다.
- 기억 어텐션 입력: `GeoMemStar.decode_view`는 `sam3_io.expand_features`로 `vision_feats`(3단) 를 만들고 `vision_feats[-1]`(`[5184, B, 256]`, 72×72)을 `src`로 넣는다. 고해상도 2단은 SAM 마스크
  디코더로만 간다. 융합은 `src`에만 적용한다.
- 기하: `geomem/pointmap.view_grid(cache, cam, ref_cam)`은 뷰의 72×72 셀마다 **기준 좌표계·장면 스케일로 나눈** 3D 점 `p`와 `valid`, 그리고 `scale`을 준다. π³ 캐시(`Cache/pi3/<cam>.npz`:
  `depth, conf, K, c2w`, 작업 해상도 378×672, 픽셀 인덱스 주점)는 기준 프레임과 학습 프레임(`Cache/pi3_frames/<t>/`) 모두 전 카메라에 있다. 가림 허용치 `OCC_TOL = 0.01`(`geomem/warp_torch.py`).
- 특징 캐시: `Cache/sam3_feats/<t>/<cam>.pt`(이미지 + fpn 3단 fp16, 뷰당 16.7 MB)가 17장면 × 11프레임 × 전 카메라에 있다. 전 뷰의 `fpn[-1]`만 쓰므로 뷰당 2.6 MB면 되고, 스텝마다 46파일 ×
  16.7 MB를 읽지 않도록 **72×72 단만 모은 프레임별 캐시**를 한 번 만든다(2.5절).
- 함정: 프로세스 전역 autocast 캐시 off; GPU 작업 동시 실행 금지; 상류 파일 수정 금지; 테스트 점수로 하이퍼파라미터를 고르지 않는다.

## 2. 모델

### 2.1 정렬 (`geomem/align.py`)

`align_views(p_tgt, valid_tgt, scale, c2w_ref, views) -> (W [N, 5184, 256], valid [N, 5184])`

- 입력: 표적의 격자 `p_tgt [5184,3]`(기준 좌표계/스케일), `valid_tgt [5184]`, `scale`(float), 기준 `c2w_ref`; 뷰마다 `(F_v [256,72,72], depth_v [h_w,w_w], K_v, c2w_v)`.
- 표적 셀의 3D 점 `P = p_tgt · scale`(기준 좌표계) → 뷰 v의 카메라 좌표 `X = inv(c2w_v) · c2w_ref · P` → 작업 해상도 픽셀 `u = K_v[0,0]·X_x/X_z + K_v[0,2]`, `v = K_v[1,1]·X_y/X_z + K_v[1,2]`
  (`geomem/warp_torch.reproject_coords`와 같은 규약; 정규화 좌표 `gx = (u+0.5)/w_w·2−1`).
- 특징 보간: `F.grid_sample(F_v[None], grid, bilinear, zeros, align_corners=False)` → `W_v [5184, 256]`. 72×72 격자는 전체 이미지를 늘린 것이라 정규화 좌표가 그대로 맞는다.
- 유효: `valid_tgt ∧ X_z > 0 ∧ 범위 안 ∧ depth_v(u,v) > 0 ∧ X_z ≤ depth_v(u,v)·(1 + OCC_TOL)`. 깊이는 최근접 샘플.
- 뷰 집합 = 장면의 **전 카메라**(기준 포함, 표적 자신 포함 — 자기 뷰의 정렬은 항등에 가깝고 유효 마스크로 자연히 처리된다; 자기 뷰를 빼지 않는 이유는 구현 단순성이며 융합 모듈의 자기 토큰과는 별개).
- 비용: N·5184·256 fp16 — 46뷰에서 122 MB. 장면·표적마다 한 번.

### 2.2 융합 (`geomem/fusion.py`, `ViewFusion`)

`ViewFusion(dim=256, heads=4)`; `forward(F_t [5184,256], W [N,5184,256], valid [N,5184]) -> (F' [5184,256], stats)`

- 셀마다 토큰 집합 = [자기 특징 `F_t[c]`, `W_1[c] … W_N[c]`] (N+1개). 쿼리 `q = W_q F_t[c]`, 키·값 `k_i = W_k x_i, v_i = W_v x_i`, `W_q, W_k, W_v: 256→256`(4헤드 × 64), 무효 뷰는 `−inf`.
  자기 토큰은 항상 유효하므로 전부 무효여도 행이 비지 않는다. pre-LayerNorm 없음(입력은 백본 특징 그대로).
- 출력 `F' = F_t + g · W_o(attn)`, `g` 스칼라 0 초기화, `W_o: 256→256`. 학습 전 `F' == F_t`(바이트 동일).
- 통계(`torch.no_grad`, 헤드 평균): 자기 토큰 질량의 평균, 타 뷰 질량의 뷰 단위 정규화 엔트로피, 유효 뷰 수의 셀 평균. 추론 시 장면마다 `RUN.json`에 기록.
- 파라미터 ≈ 263k. B-1의 교훈대로 "열렸는지"(`g`, 자기 대 타 뷰 질량)를 fold별로 보고한다.

### 2.3 래퍼 (`GeoFuse(GeoMemStar)`, `geomem/wrapper.py`)

- `decode_view(backbone_out, B, memory, p_v, valid_v, aligned=None, chunk=None)`: `aligned = (W, valid)`가 주어지면 `vision_feats[-1]`의 `[5184, 256]`(B개 복제본은 동일)에 융합을 적용하고
  B로 확장해 `src`로 넣는다. 나머지(GeoBias, 게이트, 청크, SAM 헤드)는 그대로. `aligned=None`이면 A-3와 동일.
- 융합은 **표적 디코딩에만**. `encode_reference`(c_ini + GT)와 `encode_memory_from_prediction`(예측 기억)은 원래 특징을 쓴다 — 동결 기억 인코더의 입력 분포를 바꾸지 않기 위해서다.
  단, 예측 기억 뷰를 **디코딩**할 때(학습의 1단계, 추론의 1라운드)는 융합을 쓴다(그 뷰도 표적이므로).
- `freeze("geofuse_ft")`: `geobias_only_ft`와 같고 융합 모듈 전부를 학습한다. `trainable_parameters`에 융합 포함. 체크포인트에 `fusion` 상태 추가.

### 2.4 팔

| 팔 | 토대 | 두 번째 기억 | 융합 | 역할 |
|---|---|---|---|---|
| `geoset_pred_ft` (기존) | geobias_only | 예측(게이트 τ 0.7), p 0.5 | 없음 | 대조군 |
| **`geofuse_ft`** | geobias_only | 예측(게이트 τ 0.7), p 0.5 | 표적·예측 기억 뷰 디코딩에 전 뷰 정렬 융합 | 처리군 |

### 2.5 캐시 (`geomem/cache_feats.py --level72`)

`Cache/sam3_feats72/<t>.pt` = `{"cams": [...], "feats": fp16 [N,256,72,72]}`를 기존 `sam3_feats/<t>/<cam>.pt`에서 모아 쓴다(백본 재계산 없음). 17장면 × 11프레임, 총 약 11 GB, 수 분.
없으면 학습은 `FileNotFoundError`로 멈춘다(파일 이름을 적는다).

## 3. 학습 (`geomem/train.py`)

- 레시피는 `geoset_pred_ft`와 **동일**: geobias_only 토대, SAM 3 초기값(warm start 없음), 3 epoch, 프레임 간격 2, 객체 ≤ 16, 예측 기억 확률 0.5(`memory_rng`로 출처 추출), 객체 게이트
  `PRED_MEM_TAU` 0.7, AdamW(GeoBias·융합 1e-3, 기억 어텐션 1e-5, wd 0.01), 코사인, bf16 autocast(캐시 off), 5-fold, 모델 선택 없음. 다른 점은 융합뿐이다.
- 스텝: `load_sample(..., aligned=True)`가 프레임 `t`의 `sam3_feats72` 한 파일과 전 카메라의 `pi3_frames` 기하를 싣고, 표적과 (있다면) 예측 기억 뷰 각각에 대해 `align_views`를 돌린다.
  예측 기억 뷰 디코딩은 `no_grad` + 융합 사용, 표적 디코딩은 융합 사용(기울기 흐름). `log.csv`에 `g`(게이트 값)·`self_mass`(자기 토큰 질량 평균) 열을 더한다.
- 비용 추정: 정렬 2회 × 46뷰는 ms 단위; `sam3_feats72` 읽기 120 MB/스텝. fold당 35~45분, 5 fold 약 3.5시간.

## 4. 추론 (`methods/geomem/run.py`, 팔 `geofuse_ft`)

- 장면당 전 뷰의 백본을 먼저 계산해 GPU에 둔다(46뷰 × 5.3M × 2바이트 ≈ 0.5 GB; `fpn[-1]`과 고해상도 2단 모두 — 각 뷰가 표적이 될 때 다시 계산하지 않는다). 기하는 `Cache/pi3`.
- 1라운드: 표적마다 `align_views` → `decode_view(..., aligned=)`. 2라운드: A-3와 같은 기억 집합(예측 기억 뷰는 원래 특징으로 인코딩) + `--gate reproj`; 표적 디코딩에 다시 `aligned=`.
  결과 이름 `geofuse_ft_r1`, `geofuse_ft_rep`(`--gate reproj --out geofuse_ft_rep --r1-out geofuse_ft_r1`).
- `RUN.json`에 융합 통계(2.2)와 유효 정렬 비율(셀 × 뷰 중 유효, 어안 별도 집계용).

## 5. 사전 등록한 비교와 성공 기준

| 비교 | 가르는 것 |
|---|---|
| **`geofuse_ft_r1 − geoset_pred_ft_r1`** | 융합의 순수 효과(기억 집합 없이) |
| **`geofuse_ft_rep − geoset_pred_ft_rep`** | 최종 헤드라인(A-3 + B 대 A-3) |
| `geofuse_ft_rep − geofuse_ft_r1` | 융합 위에서 2라운드가 아직 더 주는가 |
| `geofuse_ft_rep − geobias_only_ft`, `− sam3_prop` | A 최고·기존 1단계 대비 |

성공: 첫 비교의 J CI가 0을 빼고, `geofuse_ft_rep` J&F ≥ 0.915. 열림 확인: fold별 `g`가 0에서 벗어나고 자기 토큰 질량이 1 미만으로 내려와야 한다 — 둘 다 아니면 B-1처럼 "닫힘"으로 적는다.
결과는 `docs/results-geofuse.md`에 헤드라인, 쌍체 표, 장면별·축별 표, 융합 통계(fold별 `g`, 자기 질량, 뷰 엔트로피, 유효 정렬 비율 — 어안 별도), 학습 기록, 한계, 다음 질문으로 적는다.

## 6. 테스트

- 정렬(`tests_torch/test_align.py`, CPU 합성 기하): 자기 뷰 정렬이 항등(보간 오차 안), 가림 셀 무효, 범위 밖 무효, 표적 무효 셀은 전 뷰 무효.
- 융합(`tests_torch/test_fusion.py`): `g=0`이면 `F' == F_t` 바이트 동일, `g≠0`이면 변화, 전부 무효여도 유한, 통계 범위(자기 질량 ∈ [0,1], 엔트로피 ∈ [0,1]).
- 래퍼(GPU, Blocks): `aligned=None`과 `g=0`의 `decode_view`가 A-3 경로와 바이트 동일; `g`를 흔들면 출력 변화.
- 학습 스모크(20스텝): 융합·GeoBias·기억 어텐션에 기울기, 예측 경로에 기울기 없음, 로그에 `g`·`self_mass`.
- 추론 스모크(Blocks, untrained): `geofuse_ft` 1·2라운드 실행, `RUN.json`에 융합 통계 키, 결과 PNG가 A-3 untrained와 동일(게이트 0).
- 캐시: `--level72`가 만든 파일의 모양·카메라 순서가 `meta["cams"]`와 같다.

## 7. 범위 밖

기준·기억 뷰 특징의 융합, 고해상도 특징 융합, 이웃 셀 어텐션, 가시성 가중 평균 ablation, 뷰 부분집합 선택, 디코더 학습(B-2), τ·θ 변경.

## 8. 위험

- **열리지 않음**(B-1 재현): `g`는 자기 토큰 뒤의 잔차 게이트라 B-1의 프롬프트 토큰과 달리 기울기가 살아 있는 경로에 있다(A의 미세조정이 `src`를 통해 효과를 냈다). 그래도 열리지 않으면 통계가 그것을
  보여 주고, 그 자체가 결과다.
- **기하 오차**: 어안 장면의 π³ 정렬은 틀릴 수 있다. 유효 마스크와 학습된 어텐션이 걸러야 하며, 유효 정렬 비율과 어안 축의 점수를 따로 적는다.
- **시간**: 전 뷰 백본을 미리 계산하므로 1라운드 시간이 A-3 r1(146 s)보다 약간 늘지만, 2라운드가 이미 전 뷰를 디코딩하므로 총량은 비슷하다.
