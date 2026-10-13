# GeoAlign — 다른 뷰의 특징을 기준 격자에 정렬해 GT 마스크를 붙인 "예측 없는" 기억 집합 (B의 (가) 형태)

> 2026-10-10. 전편: A-3 [results-geoset2.md](../../results-geoset2.md)(`geoset_pred_ft` + 재투영 게이트 0.914), B-1 [results-geoquery.md](../../results-geoquery.md)(닫음),
> B-2 [results-geofuse.md](../../results-geofuse.md)(표적 쪽 특징 융합, 열렸으나 일반화하지 않음, 닫음). 벤치마크 규약은 같다(17장면 × 주석 카메라 2대 × 객체 = 664쌍, 기준 프레임,
> J·DAVIS F, 장면 부트스트랩 10,000회).

## 0. 한 문장

A-3의 2라운드는 이웃 뷰의 **예측** 마스크를 기억으로 쓰므로 틀린 기억이 생기고 게이트가 필요했다. 이 설계는 이웃 뷰 v의 72×72 백본 특징을 **π³로 기준 격자에 정렬**해 "v에서 본 c_ini"를
만들고, 거기에 **c_ini의 GT 마스크를 그대로 붙여** 기존 기억 인코더(`encode_reference`)로 기억 뷰를 만든다(`geoalign_ft`). 예측도 게이트도 없고, 1라운드에서 바로 여러 뷰의 외관을 기억에
넣는다. B-1·B-2가 실패하고 A-3가 성공한 공통 설명("다른 뷰 정보는 마스크가 붙은 기억으로 넣을 때만 동결 디코더가 안전하게 쓴다")을 전제로 하는 가장 단순한 형태이며, 핵심 질문은
**예측 없는 기억 집합이 A-3의 2라운드(0.914)에 닿는가**다.

## 1. 배경 (설계를 묶는 사실)

- 마스크 프롬프트 프레임(`track_step(mask_inputs=…)`)에서 기억 인코더 `_encode_new_memory`는 **72×72 단 특징(`current_vision_feats[-1]`)과 마스크**만 쓴다. 고해상도 2단은
  `_use_mask_as_output`의 SAM 헤드(포인터 토큰·존재 점수)에만 들어간다. 따라서 정렬은 72단만 하면 되고, 고해상도 2단과 이미지는 기준 뷰의 것을 그대로 쓴다.
- `geomem/align.py`(GeoFuse)가 뷰 v의 72단 특징을 임의 표적 격자에 정렬한다(`align_views`). 표적을 기준 격자(`view_grid(cache, ref, ref)`)로 두면 그대로 쓰인다. 유효 규칙은 현재 한쪽만
  본다(가림만). 리뷰에서 지적된 대로 표면 **앞에** 뜬 점은 다른 표면의 특징을 가져오므로, 이 설계는 **양방향 규칙** `|z / depth_v − 1| ≤ 0.05`를 쓴다(`alignment_depth_ratio` 기준).
- 72단 캐시 `Cache/sam3_feats72/<t>.pt`(학습 프레임, 전 카메라)와 π³ 기하(`pi3`, `pi3_frames`)가 있다. 기준 프레임의 72단 특징은 추론 시 백본에서 바로 나온다.
- `MemorySet`은 뷰마다 3D 격자 `(p, valid)`를 들고 GeoBias가 그 거리로 바이어스를 만든다. 정렬 기억 뷰의 격자는 **기준 격자**다(같은 셀 = 같은 3D 점).
- 함정: 프로세스 전역 autocast 캐시 off; GPU 작업 동시 실행 금지; 상류 파일 수정 금지; 테스트 점수로 하이퍼파라미터를 고르지 않는다.

## 2. 모델

### 2.1 정렬 기억 뷰 (`geomem/align.py` 확장, `GeoMemStar.encode_aligned`)

- `align_views(..., two_sided=False)`: `two_sided=True`이면 유효 = `valid_tgt ∧ z>0 ∧ 범위 안 ∧ depth>0 ∧ |z/depth − 1| ≤ tol2`(`tol2 = 0.05`, `TWO_SIDED_TOL`). 기존 호출(GeoFuse)은 바뀌지 않는다.
- `fill_aligned(W_v [5184,256], valid_v [5184], F_ref [5184,256]) -> [5184,256]`: 유효 셀은 정렬 특징, 무효 셀은 **기준 뷰 자신의 72단 특징**. 메운 비율 `1 − valid_v.mean()`을 통계로 남긴다.
- `GeoMemStar.encode_aligned(image_ref, backbone_ref, masks_bool, p_ref, valid_ref, W_v, valid_v, name) -> MemoryView`: `backbone_ref`의 `backbone_fpn[2]`를 `fill_aligned` 결과(`[1,256,72,72]`로
  복원)로 바꾼 사전을 만들어 `encode_reference(image_ref, backbone_v, masks_bool, p_ref, valid_ref, name)`를 부르고 그 뷰(`conf=None`)를 돌려준다. 고해상도 2단·이미지·격자는 기준의 것.
  자기 뷰(v = 기준)를 정렬하면 항등이므로 결과는 기준 기억과 **바이트 동일**해야 한다(테스트).

### 2.2 팔

| 팔 | 토대 | 두 번째 기억(학습) | 역할 |
|---|---|---|---|
| `geoset_only_ft` (기존) | geobias_only | 세 번째 주석 카메라의 GT 기억 | 대조군(GT 기억) |
| `geoset_pred_ft` (기존) | geobias_only | 임의 카메라의 **예측** 기억 + 객체 게이트 | 대조군(A-3) |
| **`geoalign_ft`** | geobias_only | 임의 카메라(비주석 포함, `memory_rng`)의 **정렬 기억 뷰** | 처리군 |

## 3. 학습 (`geomem/train.py`, `geomem/data.py`)

- 레시피는 `geoset_only_ft`와 같다: geobias_only 토대(`use_bias=True, use_geometry=False`), SAM 3 초기값, 3 epoch, 프레임 간격 2, 객체 ≤ 16, 두 번째 기억 확률 `EXTRA_MEM_PROB` 0.5, AdamW(GeoBias 1e-3,
  기억 어텐션 1e-5, wd 0.01), 코사인, bf16 autocast(캐시 off), 5-fold, 모델 선택 없음. 학습 파라미터는 `geobias_only_ft`와 같다(정렬·메움에 파라미터 없음).
- `arm_flags("geoalign_ft")`: `extra_mem_prob 0.5`, `extra_mem_pred False`, `extra_mem_aligned True`, `use_fusion False`.
- `load_sample(..., extra_mem_cam=m, extra_mem_gt=False, aligned_mem=True)`: m의 72단 특징(`load_level72`에서 해당 카메라 행)과 π³ 기하, 기준의 `scale`·`c2w_ref`를 싣는다. 이미지·마스크는 기준의 것을 쓴다.
- 스텝: `W_m, valid_m = align_views(p_ref, valid_ref, scale, c2w_ref, [view_m], two_sided=True)` → `view = wrapper.encode_aligned(ref_image, ref_backbone, ref_masks, p_ref, valid_ref, W_m[0], valid_m[0], name=m)`
  → `ref = ref.with_view(view)` → 표적 디코딩, 손실은 기존. `log.csv`에 `align_valid`(유효 셀 비율; 기억이 하나면 빈칸)를 더한다. 기울기는 GeoBias·기억 어텐션에만 흐른다(기준 인코딩은 `no_grad`).
- 비용: `geoset_only_ft`와 같은 fold당 30~34분, 5 fold 약 2.8시간.

## 4. 추론 (`methods/geomem/run.py`, 팔 `geoalign_ft`; `SET_ARMS`에 추가)

- 1라운드: 표적마다 **같은** 기억 집합 = c_ini + 최근접 k = 3 이웃(π³ 카메라 중심 거리, 기준 제외; A-3의 후보 규칙과 같음)의 정렬 기억 뷰. 이웃의 72단 특징은 그 뷰의 백본에서(전 뷰 백본을 먼저
  계산해 두는 GeoFuse 경로 재사용). 정렬 기억 뷰는 장면당 한 번 만들어 모든 표적이 공유한다(표적과 무관하게 기준 격자에 정렬되므로).
  결과 `geoalign_ft_r1`.
- 2라운드: 1라운드 결과에 A-3와 같은 예측 기억(최근접 3대, `--gate reproj`)을 **더해** 기억 = c_ini + 정렬 뷰 3 + 예측 뷰 ≤ 3 → `geoalign_ft_rep`. `--gate` 기본 `reproj`로 돌린다.
- `RUN.json`: `align_valid_mean`(정렬 뷰의 유효 셀 비율 평균), `align_views`(=3), 기존 게이트 통계.
- 다른 팔은 바뀌지 않는다(`geoset_*`는 정렬 뷰를 만들지 않음).

## 5. 사전 등록한 비교와 성공 기준

| 비교 | 가르는 것 |
|---|---|
| **`geoalign_ft_r1 − geoset_pred_ft_r1`** | 정렬 기억의 순수 효과(예측 없이, 1라운드끼리) |
| **`geoalign_ft_r1 − geoset_pred_ft_rep`** | **예측 없는 기억 집합이 A-3 2라운드에 닿는가** |
| `geoalign_ft_rep − geoset_pred_ft_rep` | 정렬 + 예측 기억 합산 헤드라인 |
| `geoalign_ft_r1 − geoset_only_ft_r1` | 토대 동등성(≈ 0이어야 함) |
| `geoalign_ft_rep − geoalign_ft_r1` | 정렬 기억 위에서 예측 기억이 아직 더 주는가 |

성공(최소): 첫 비교의 J CI가 0을 뺀다. 성공(강): `geoalign_ft_r1` J&F ≥ 0.914(2라운드 대체). 결과는 `docs/results-geoalign.md`에 헤드라인, 쌍체 표, 장면별·축별, 정렬 통계(뷰당 유효 셀 비율,
어안 별도; 기억 어텐션에서 정렬 뷰 토큰이 받는 질량 — `attn_patch`에 훅을 두지 않고 GeoBias 마스크 없이 재는 간단한 방법이 없으면 생략하고 그렇게 적는다), 학습 기록, 한계, 다음 질문으로 적는다.

## 6. 테스트

- 정렬(`tests_torch/test_align.py`): `two_sided=True`가 표면 앞에 뜬 점을 무효로 함(기존 `test_depth_consistency_two_sided`의 기하 재사용); `fill_aligned`가 유효 셀은 정렬 특징, 무효 셀은 기준 특징.
- 래퍼(GPU, Blocks): 자기 뷰 정렬 기억 뷰가 기준 기억과 바이트 동일(`feats`, `obj_ptr_tokens`, `object_score_logits`); 다른 뷰(cam9)의 정렬 기억 뷰는 `feats`가 다르고 유한; 기억 집합 2뷰로 디코딩이 돈다.
- 학습 스모크(20스텝): `geoalign_ft`가 `n_mem ∈ {1, 2}`, `align_valid` 기록, 기울기가 GeoBias·기억 어텐션에만, 학습 파라미터 수 = `geobias_only_ft`.
- 추론 스모크(Blocks, untrained): `geoalign_ft` 1라운드가 기억 4뷰(c_ini + 3)로 돌고 `RUN.json`에 `align_valid_mean`; 2라운드가 돈다.

## 7. 범위 밖

무효 토큰 마스킹(접근 2), 고해상도 2단 정렬(접근 3), 뷰 선택 기준·k 변경, GeoFuse 판별 실험, 기준 뷰 자체의 특징 교체.

## 8. 위험

- 정렬 기억 뷰의 메운 셀은 기준 기억과 중복된 토큰이다. 어안(유효 10~16 %)에서는 정렬 뷰가 거의 기준의 복사본이 되어 효과가 없을 수 있다 — 유효 비율을 어안 별도로 적는다.
- 학습 시 두 번째 기억을 임의 카메라에서 뽑으므로 추론(최근접 3대)과 분포가 다르다. A-3도 같은 차이를 가졌고 효과가 있었다.
- 정렬 오차가 큰 셀이 GT 마스크 아래 놓이면 "틀린 외관에 맞는 마스크"가 기억이 된다. 양방향 규칙이 이를 줄이지만 없애지는 못한다.
