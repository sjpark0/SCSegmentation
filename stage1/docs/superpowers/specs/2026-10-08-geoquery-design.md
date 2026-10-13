# GeoQuery — 객체 쿼리 허브로 다중뷰를 공동 디코딩하는 B-1 설계

> 2026-10-08. GeoMem-2([2026-10-07-geomem2-design.md](2026-10-07-geomem2-design.md), 결과 [results-geomem2.md](../../results-geomem2.md))의
> 다음 단계이자 "B(다중뷰 공동 디코더)"의 첫 형태. 벤치마크 규약은 1차와 같다(17장면 × 주석 카메라 2대 × 객체 = 664쌍, 기준 프레임, J·DAVIS F).

## 0. 한 문장

A의 최고 팔 `geobias_only_ft`(J&F 0.908) 위에 **객체 쿼리 허브**를 덧붙인다: 기준 뷰의 객체 포인터를 객체 쿼리로 삼아 **모든 뷰의 토큰을
3D 객체 점구름 거리 바이어스로 한 번에 읽고**, 갱신된 쿼리를 **모든 뷰의 SAM 디코더에 같은 프롬프트 토큰으로 넣어** N뷰 마스크를 한 번에
낸다. 허브를 끄면 정확히 A가 되므로, 같은 벤치마크·같은 예산에서 허브의 순수 효과를 `geobias_only_ft`(+ 같은 추가 학습을 한 대조군) 대비 잰다.

## 1. 배경 (A에서 이어지는 사실)

- `geobias_only_ft` 0.908 > `geomem_ft` 0.900 > `sam3_prop` 0.890: 3D 신호는 **어텐션 내적 자리의 바이어스**로 넣어야 효과가 있고, 더해지는
  임베딩은 필요 없다(results-geomem2 §1).
- `geoset_ft`(이웃 뷰의 1라운드 **예측**을 기억에 추가) 0.909는 1라운드 대비 유의하지 않고, Dog·Welder에서는 예측 오류가 기억으로 전파돼
  나빠졌다(§6). 비주석 뷰는 **예측이 아니라 특징**으로 기여해야 한다.
- 뷰 v는 지금 c_ini 기억만 본다(뷰 v ← c_ini). 뷰-뷰 쌍별 어텐션은 N·k·5184²으로 비싸고 "쿼리 하나 → N 마스크"의 공동 디코딩이 아니다
  (brainstorm 1문). 선택: **객체 쿼리 허브** — 모든 정보가 M개 객체 쿼리를 거쳐 흐르는 바퀴 모양(뷰 v ← q_m ← 모든 뷰), 비용은 N에 선형.
- 코드 사실: `_forward_sam_heads`(sam3_tracker_base.py:217-)는 `sam_prompt_encoder(points, boxes=None, masks)`로 희소 프롬프트
  `[B, P, 256]`(점이 없으면 `not_a_point_embed` 패딩 토큰 1개)과 밀집 프롬프트를 만들고, `sam_mask_decoder(image_embeddings, image_pe,
  sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output, repeat_image=False, high_res_features)`를 불러
  `(low_res_multimasks, ious, sam_output_tokens, object_score_logits)`를 받은 뒤 존재 판정·업샘플·`obj_ptr_proj`·`no_obj_ptr` 처리를 한다
  (:262-379). 희소 프롬프트에 토큰을 하나 더 붙이는 데 업스트림 수정은 필요 없다(래퍼가 두 호출과 꼬리 처리를 복제).
- 함정: 프로세스 전역 autocast 캐시 off 필수(`methods/sam3_single.load_model`); GPU 작업 동시 실행 금지; 상류 파일 수정 금지.

## 2. 모델 (합의 1절)

구성: `GeoQuery = geobias_only_ft 경로 + 객체 점구름 + QueryHub + 쿼리 프롬프트 주입`. 이미지 인코더·프롬프트 인코더·SAM 마스크 디코더 동결.

### 2.1 입력

- N뷰 백본 특징(캐시, 뷰당 `[256, 72, 72]` + 고해상도 2단), 뷰별 3D 격자 `p_v [72,72,3]`·`valid_v`(π³ 포인트맵, 기준 좌표계, 장면 스케일 정규화 —
  `geomem/pointmap.view_grid` 그대로), c_ini 정답 마스크(객체 M ≤ 16).

### 2.2 기준 인코딩과 객체 점구름 (`geomem/objcloud.py`)

- A 그대로: c_ini + 정답 마스크 → `MemorySet`(뷰 1개) + `obj_ptr_m [256]`.
- 객체 점구름 `P_m`: 기준 72×72 격자에서 객체 m의 마스크 커버리지(1008 → 72 평균 풀링) ≥ 0.5이고 `valid`인 셀의 3D 점. 256점을 넘으면
  고정 seed의 무작위 부표본. 점이 0개인 객체(아주 작거나 무효 영역)는 `P_m = ∅`로 두고 바이어스 전부 `b_none`.
- 토큰 거리 `d_m(v, cell) = min_{x∈P_m} ‖p_v[cell] − x‖` (유효 셀만, 무효는 ∞). 장면당 한 번, `torch.no_grad`, `[N, 5184, M]` fp32.
  비용: 46뷰 × 5184 × 256점 × 16객체 ≈ 1e9 거리, GPU에서 1초 안.

### 2.3 QueryHub (`geomem/query_hub.py`)

- `QueryHub(dim=256, layers=2, heads=8, ffn=1024)`. 입력 `q0 = obj_ptr + proj0(obj_ptr)`(`proj0` 0 초기화), 뷰 토큰
  `T = concat_v feats_v` `[N·5184, 256]`(백본 특징 + SAM 위치 인코딩; 학습 안 함), 거리 `d [M, N·5184]`, 유효 마스크.
- 층 = 교차 어텐션(q ← T, 로짓 + `ObjBias(d)`) → 쿼리 자기 어텐션(M개끼리) → FFN. 세 블록 모두 출력에 스칼라 게이트 `g = 0` 초기화
  (`x + g·block(x)`), LayerNorm pre-norm.
- `ObjBias`: A의 `GeoBias`와 같은 꼴(RBF 16개, 중심 [0, 2] 등간격, σ 0.125, `w` 0 초기화, 무효 셀 `b_none`). 헤드 공유(1 스칼라를 전 헤드에).
- 출력: `q [M, 256]`과 **뷰별 어텐션 질량** `a [L, M, N]`(각 층에서 쿼리 m이 뷰 v의 토큰에 준 어텐션 합; 헤드 평균) — 통계용.
- 부분집합: 학습에서는 K개 뷰의 토큰만 `T`에 넣는다(2절). 추론은 전 뷰.

### 2.4 쿼리 프롬프트 주입 (`GeoQuery.decode_view`, `geomem/wrapper.py`)

- 뷰 v의 기억 어텐션은 A와 동일(`MemorySet` = c_ini 1개, `GeoBias`, 패치) → `pix_feat_with_mem`.
- `_forward_sam_heads` 대신: `sparse, dense = sam_prompt_encoder(points=(zeros[B,1,2], −ones[B,1]), boxes=None, masks=None)`;
  `tok = proj_q(q_m) + type_embed` (`proj_q` 0 초기화, `type_embed [256]` 0 초기화) → `sparse' = cat([tok[:, None], sparse], 1)` `[B, 2, 256]`;
  `sam_mask_decoder(pix_feat_with_mem, image_pe, sparse', dense, multimask_output=False, repeat_image=False, high_res_features)` →
  꼬리 처리(존재 판정 `object_score_logits > 0`, `NO_OBJ_SCORE` 대체, 1008 업샘플, `obj_ptr_proj`, `no_obj_ptr`)를 업스트림 :320-370과 같게 복제.
- 반환은 A의 `decode_view`와 같은 dict(`logits_1008, ious, presence, low_res, obj_ptr`).
- **구현 정정(Task 3 ruling):** 토큰을 하나 더 붙이면 디코더 어텐션 정규화가 바뀌어 학습 전 부호 일치가 99.989 %에 그쳤으므로, 실제 구현은 `tok`을
  빈 프롬프트의 첫 패딩 토큰(`not_a_point_embed`)에 **더한다**(토큰 수 불변, 학습 전 출력이 A와 정확히 같음).
- 게이트·투영이 0이면 `tok = 0` 상수 토큰 하나가 더 들어갈 뿐이다. 학습 전 출력은 `geobias_only_ft`와 **부호 일치 > 99.99 %**(바이트 동일은
  요구하지 않음; 토큰 수가 달라 어텐션 정규화가 미세하게 바뀐다). 이 동치가 대조군 테스트다.

### 2.5 추론 (`methods/geomem/run.py`, 팔 `geoquery*`)

1. 전 뷰 백본 특징(캐시 또는 즉석 계산), 전 뷰 3D 격자, 기준 인코딩, 객체 점구름, 거리.
2. `QueryHub`로 `q`와 `a`를 한 번 계산(전 뷰 토큰, 46뷰 × 5184 = 24만 키).
3. 뷰마다 `decode_view` → 마스크 기록. 라운드 1회. 2라운드·예측 기억 없음.
4. RUN.json `config`에 허브 통계: 뷰별 어텐션 질량의 **엔트로피**(뷰 분포, 최대 log N으로 정규화, 층·객체 평균), **기준 뷰 비율**, **상위 3뷰
   비율**, 객체 점구름 크기 평균, `P_m = ∅` 객체 수.

### 2.6 팔

| 팔 | 허브 | ObjBias | 초기화 | 학습 |
|---|---|---|---|---|
| `geoquery_ft` | 있음 | 있음 | 같은 fold의 `geobias_only_ft` ckpt(기억 어텐션·GeoBias) + 새 모듈 0 | 허브·proj·type·ObjBias 1e-3, 기억 어텐션·GeoBias 1e-5 |
| `geoquery_nobias_ft` | 있음 | 없음(로짓에 0) | 〃 | 〃 |
| `geobias_only_ft2` | 없음 | — | 〃 | 기억 어텐션·GeoBias 1e-5만(같은 예산으로 이어 학습한 대조군) |

## 3. 학습 (합의 2절)

- 규약은 A와 동일: 5-fold 장면 교차검증(`geomem/folds.json`), 21프레임 중 stride 2(11장), 객체 ≤ 16, 3 epoch, AdamW(wd 0.01), 코사인,
  bf16 autocast(캐시 off), 모델 선택 없음, seed 0. 체크포인트 `config.json`에 팔·fold·보지 않은 장면·학습 커밋·SHA256·`init_from`(A ckpt SHA256).
- 샘플 = (장면, 프레임, 기준 카메라) — 기준은 A처럼 주석 3대를 번갈아 쓰고, **표적은 나머지 주석 카메라 2대 모두**(두 뷰 디코딩, 손실 합).
  A의 (장면, 프레임, 기준, 표적 1대) 열거를 표적 2대로 묶으므로 fold당 샘플 수는 A의 절반(429~462), 스텝당 디코딩은 2뷰.
- 허브 입력 부분집합 K = 8: 기준 + 표적 2 + 전 뷰에서 무작위 5(뷰가 8개 미만인 장면은 전부). 무작위 선택은 seed 고정 rng.
- 손실 = A와 같은 focal + dice + IoU + 존재(`geomem/losses.py`), 표적 2뷰 합. 재투영 자기지도 없음.
- 학습 파라미터: 허브 2층 ≈ 2.1M(층당 교차 어텐션 0.26M + 자기 어텐션 0.26M + FFN 0.52M), proj0·proj_q ≈ 0.13M, type 256, ObjBias 17,
  기억 어텐션 ≈ 5.9M, GeoBias 17. `geobias_only_ft2`는 기억 어텐션·GeoBias만.
- 선행: **전 뷰 특징 캐시** — `geomem.cache_feats --all-views`로 학습 프레임 11장 × 전 뷰(약 65 GB, 디스크 1.4 TB 여유, GPU 약 15분). 기존 캐시
  파일은 건너뛴다. 캐시가 끝난 뒤 학습을 시작하고, 학습 중 다른 GPU 작업은 없다.
- 예산: 3팔 × 5fold × 약 30~35분 ≈ 8 h(순차, 백그라운드). 학습 로그 `log.csv`에 step, epoch, 손실 항, lr, seconds, n_views(허브 입력 뷰 수).

## 4. 평가 (합의 3절)

- `scripts/run_geomem.sh all --arm geoquery_ft` 등 → `mvseg2d.evaluate` → `mvseg2d.report` over
  `sam3_prop star geomem_ft geobias_only_ft geobias_only_ft2 geoquery_nobias_ft geoquery_ft`.
- 쌍체 CI(장면 부트스트랩 10,000회, seed 0): `geoquery_ft − geobias_only_ft2`(허브의 순수 효과), `geoquery_ft − geobias_only_ft`,
  `geoquery_ft − sam3_prop`, `geoquery_ft − geoquery_nobias_ft`(ObjBias 효과), `geobias_only_ft2 − geobias_only_ft`(추가 학습 효과),
  `geoquery_nobias_ft − geobias_only_ft2`.
- 축별(크기·리그·시점차)·장면별 표, 학습 표(스텝·시간·epoch 손실), 허브 통계 표(장면별 엔트로피·기준 뷰 비율·상위 3뷰 비율).
- 결과 문서 `docs/results-geoquery.md`: 헤드라인, CI, 표, 학습된 ObjBias 모양, 읽기, 다음 단계(디코더 미세조정 여부)에 대한 함의 — 결정은 하지 않음.

## 5. 코드 구조

| 파일 | 역할 |
|---|---|
| `geomem/objcloud.py` (신규) | `object_clouds(masks_bool, p_ref, valid_ref, max_points=256, seed=0) -> list[Tensor[n_m,3]]`, `cloud_distances(p_views, valid_views, clouds) -> Tensor[M, N*5184]`(∞ for invalid/empty) |
| `geomem/query_hub.py` (신규) | `ObjBias`(RBF 16), `QueryHub(dim, layers, heads, ffn)`: `forward(q0, tokens, dist, valid) -> (q, attn_mass[L,M,N])` |
| `geomem/wrapper.py` | `class GeoQuery(GeoMemStar)`: `run_hub(view_feats, p_views, valid_views, clouds, obj_ptr) -> (q, mass)`, `decode_view(..., query=None)`(query가 None이면 A 경로); 꼬리 처리 복제 `_sam_heads_with_query` |
| `geomem/cache_feats.py` | `--all-views`(학습 프레임 × 전 뷰) |
| `geomem/data.py` | `enumerate_pairs(data, scenes, frame_stride)`(장면·프레임 단위), `load_sample_pair(...)`(표적 2대), `hub_views(scene, ref, targets, k_random=5, rng)` |
| `geomem/train.py` | 팔 `geoquery_ft`, `geoquery_nobias_ft`, `geobias_only_ft2`; `init_from` 적재; 스텝당 표적 2뷰 |
| `methods/geomem/run.py` | `QUERY_ARMS`; 전 뷰 허브 1회 → 뷰별 디코딩; 허브 통계 기록 |
| `tests/` (호스트) | 점구름·거리(numpy 참조 구현 대조, 빈 객체, 무효 셀), `enumerate_pairs`·`hub_views` |
| `tests_torch/` | `ObjBias` 0 초기화·`b_none`, `QueryHub` 형상·게이트 0에서 `q == q0`·어텐션 질량 합 1, `GeoQuery` 게이트 0 출력 ≈ `geobias_only_ft`(부호 일치 > 99.99 %), 학습 smoke(3팔), 전 뷰 추론 1장면(Blocks) |

## 6. 범위 밖

SAM 디코더 교체(2단계 공동 디코더), 이미지 인코더·프롬프트 인코더·마스크 디코더 미세조정, 외부 데이터, 재투영 자기지도, 어안 보정,
학습 예산 확대, 뷰-뷰 직접 어텐션, 잠재 3D 볼륨 허브, 다중 프레임(시간) 입력.

## 7. 위험

- SAM 디코더는 희소 프롬프트에 "점·상자"만 보고 학습됐다. 동결 디코더가 쿼리 토큰을 활용하지 못하면 `geoquery_ft ≈ geobias_only_ft2`가
  된다. 그 경우 "허브 효과 없음(동결 디코더)"으로 기록하고 디코더 미세조정의 근거로 삼는다.
- 허브가 기준 뷰에만 어텐션을 몰면(엔트로피 ≈ 0, 기준 뷰 비율 ≈ 1) 다중뷰 읽기가 일어나지 않은 것이다. 2.5의 통계가 이를 드러낸다.
- 객체 점구름은 π³ 포인트맵에 묶인다. 어안 장면과 작은 객체(≤ 셀 1개)에서는 점구름이 비거나 왜곡된다 — A와 같은 한계이며 보정하지 않는다.
