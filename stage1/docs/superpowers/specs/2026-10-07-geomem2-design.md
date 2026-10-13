# GeoMem-2 — 3D 거리 어텐션 바이어스(A-2)와 점진 집합 기억(A-3) 설계

> 2026-10-07. GeoMem-star([2026-10-06-geomem-design.md](2026-10-06-geomem-design.md))의 다음 단계. 벤치마크 규약은 1차와 같다
> ([2026-10-06-mvseg2d-benchmark-design.md](2026-10-06-mvseg2d-benchmark-design.md)). 이 문서는 대화에서 절별로 합의한 설계다.

## 0. 한 문장

GeoMem-star(`geomem_ft`, J&F 0.900)에서 3D 위치가 **어텐션이 실제로 비교하는 자리(로짓)**에 들어가지 않았던 배선을 **3D 거리 어텐션
바이어스**로 고치고(A-2), 기억을 c_ini 하나에서 **c_ini + 확신 있는 이웃 뷰들의 1라운드 예측**으로 넓혀(A-3), 같은 벤치마크·같은 예산에서
`geomem_ft`·`star`·`sam3_prop` 대비 이득을 잰다. 결과는 B(다중뷰 공동 디코더)로 갈지 결정하는 근거가 된다.

## 1. 배경 (GeoMem 결과에서 이어지는 것)

- `geomem_ft` 0.900 > `sam3_prop` 0.890 ≈ `star` 0.888. 임베딩만 학습한 `geomem`·`geomem_ssl`은 `star`와 차이 없음.
- 원인 진단(리뷰): SAM 3 추적기의 기억 어텐션은 `pos_enc_at_input=True`, `pos_enc_at_cross_attn_queries=False`로 빌드돼 질의 위치 인코딩이
  인코더 입력에 `0.1×`로 더해질 뿐이고, 교차 어텐션 로짓에는 키 쪽 위치만 들어간다. 2D RoPE는 여전히 기준 (i,j)와 대상 (i,j)를 묶는다.
- 기억이 c_ini 하나뿐이라 먼 뷰로의 직접 전이가 필요한 장면(Blocks, CBABasketball)에서 사슬 전파에 뒤진다.
- 코드 사실: 교차 어텐션은 `RoPEAttention`(1헤드, 질의 256, 키 64, `rope_k_repeat=True`, `use_fa3=False`)이고 RoPE 적용 뒤
  `F.scaled_dot_product_attention(q, k, v)`를 부른다 → `attn_mask`(가산 바이어스)를 넣을 자리가 있다. 포인터 토큰 4개는 `num_k_exclude_rope`로
  RoPE에서 제외된다. 기억 인코더(`_encode_new_memory`)는 예측 마스크와 존재 로짓으로도 기억을 만든다.
- 함정: 추적기가 프로세스 전역 autocast를 닫지 않으므로 `torch.set_autocast_cache_enabled(False)`가 필수(`methods/sam3_single.load_model`).

## 2. 모델 (합의 1절)

### 2.1 A-2 3D 거리 어텐션 바이어스 (`GeoBias`)

- 대상 뷰 셀 i(5184개)와 기억 뷰 m의 셀 j(5184개)의 정규화 3D 좌표 p_i, p^m_j(GeoMem의 `view_grid`, c_ini 좌표계·장면 스케일)에서
  d_ij = ‖p_i − p^m_j‖. 두 셀 다 유효할 때만.
- RBF 특징 φ(d) = exp(−(d − c_k)² / 2σ²), k = 1..16, 중심 c_k는 [0, 2] 균등, σ = 0.125(스케일 정규화 뒤 2는 장면 중앙값 거리의 2배).
- 바이어스 b_ij = w·φ(d_ij) (w ∈ R^16, 0 초기화). 한쪽이라도 무효인 쌍은 학습 상수 b_none(0 초기화). 포인터 토큰 열(기억 뷰당 4개)은 0.
- 바이어스 행렬 [B, 1, 5184, Σ_m(5184 + 4)]를 `attn_mask`(float, 가산)로 교차 어텐션의 SDPA에 전달. 배치(객체)에 걸쳐 같은 값(브로드캐스트).
- 0 초기화 → 학습 전 모델은 `geomem_ft`의 학습 전(= `star`)과 출력이 같다. 테스트로 확인.
- `--no-rope`: 교차 어텐션의 `num_k_exclude_rope`를 전체 키 수로 두어 2D RoPE를 끈다(절제 팔).
- 메모리: 기억 1개 ≈ 5184² × B × 2바이트 = B=16에서 0.86 GB, 기억 4개 ≈ 3.4 GB. 플래시 커널은 attn_mask를 지원하지 않아 메모리 효율/수학
  커널로 떨어진다(속도 저하 허용).

### 2.2 A-3 점진 집합 기억 (`MemorySet`)

- `RefMemory`를 기억 뷰의 **목록**으로 일반화: 뷰마다 토큰 [5184,B,64], 위치(+조건 행 + 3D 키 임베딩), 포인터 토큰 4개와 위치, 그 뷰의 3D 격자
  (p, valid). 디코딩 때 토큰·위치를 뷰 순서로 이어 붙이고(`prompt`, `prompt_pos`), 포인터 토큰은 뷰별 블록 끝에 둔다; `num_obj_ptr_tokens`는
  마지막 뷰의 4개만 RoPE 제외가 되므로 **포인터 토큰을 모두 맨 뒤에 모아** 4M개를 제외한다.
- 추론 2라운드:
  1. 1라운드: 기억 = {c_ini}로 전 뷰 디코딩(지금의 GeoMem). 뷰마다 마스크 로짓·존재 로짓·IoU 예측을 보관하고, 예측 마스크로 그 뷰의 기억을
     `_encode_new_memory`로 만든다(`track_step`의 `run_mem_encoder` 경로와 같은 입력: 1008 해상도 마스크 로짓, 존재 로짓).
  2. 뷰 v의 기억 집합 = {c_ini} ∪ {v를 제외한 뷰 중 **π³ 카메라 중심 거리가 가장 가까운 3개** 가운데 1라운드가 확신 있는 것: 객체별이 아니라
     뷰별 판정으로, 존재 로짓 > 0인 객체의 IoU 예측 평균 ≥ 0.7}. 고정 규칙, 튜닝 없음.
  3. 2라운드: 집합 기억으로 뷰 v를 다시 디코딩. 1라운드(`<method>_r1`)와 2라운드(`<method>`) 결과를 둘 다 저장.
- 학습: 샘플 (r, v)에 세 번째 주석 카메라 m의 **정답 마스크**로 만든 기억을 50 % 확률로 추가(나머지 50 %는 c_ini만). 추론(예측 기억)과 학습
  (정답 기억)의 출처 차이는 알고 감수한다.

### 2.3 팔

| 팔 | 3D 임베딩 | 바이어스 | RoPE(교차) | 집합 기억 | 학습 파라미터 |
|---|---|---|---|---|---|
| `geobias_ft` | ○ | ○ | ○ | × | 임베딩 + 바이어스 + 어텐션 4층 |
| `geobias_norope_ft` | ○ | ○ | × | × | 〃 |
| `geobias_only_ft` | × | ○ | ○ | × | 바이어스 + 어텐션 |
| `geoset_ft` | ○ | ○ | ○ | ○ | 임베딩 + 바이어스 + 어텐션 |

비교 기준: `geomem_ft`(재사용), `star`, `sam3_prop`. 임베딩만 학습하는 팔은 GeoMem에서 효과 없음이 확인돼 두지 않는다.

## 3. 학습 (합의 2절)

GeoMem과 동일: 5-fold(`geomem/folds.json`), 샘플 (장면, t, r, v) with stride 2, 객체 ≤16, focal 20 + dice + IoU L1 + 존재 BCE, AdamW(임베딩·
바이어스 1e-3, 어텐션 1e-5, wd 0.01), bf16 autocast(캐시 끔), 코사인, **3 epoch 고정, 모델 선택 없음**, 시드 0. `geoset_ft`만 2.2의 50 % 세 번째
카메라 기억을 추가. 체크포인트에 `geo_embed`, `geo_bias`, `memory_attention`과 config(팔, fold, 장면, 커밋, SHA256).

## 4. 평가 (합의 2절)

- 방법 `geobias_ft`, `geobias_norope_ft`, `geobias_only_ft`, `geoset_ft`(+ `geoset_ft_r1`)를 1차 벤치마크 형식으로. 장면마다 그 장면을 보지 않은
  fold의 체크포인트만(누출 거부 유지).
- `geoset_ft`의 RUN.json에 뷰당 기억 수 평균, 기억이 c_ini뿐인 뷰 비율, 선택된 이웃 분포를 기록.
- 읽는 값: 헤드라인, 네 축, 짝지은 차이(각 팔 − `geomem_ft`, − `star`, − `sam3_prop`; `geoset_ft − geoset_ft_r1`). 핵심은
  `geobias_ft − geomem_ft`(배선 효과)와 `geoset_ft − geoset_ft_r1`(집합 기억 효과).
- GPU 작업은 직렬로만.

## 5. 코드 구조

```
geomem/bias.py        GeoBias(nn.Module): rbf(d), forward(p_q, valid_q, [(p_m, valid_m)]...) -> attn_mask [1,1,Lq,Lk]; 0 초기화
geomem/attn_patch.py  patch_cross_attention(tracker, bias_provider, no_rope): RoPEAttention 인스턴스의 forward를 감싸 attn_mask 전달
geomem/wrapper.py     MemorySet, encode_reference -> encode_memory(view, masks|logits, presence), decode_view(backbone, B, memory_set, p_v, valid_v)
geomem/train.py       --arm 추가, 세 번째 카메라 기억 50 %
methods/geomem/run.py 2라운드 추론, 이웃 선택, r1/r2 저장
tests/, tests_torch/  2.1·2.2의 테스트(바이어스 모양·0 초기화·블록 결합·무효 처리, 패치 어텐션 바이트 동일성, 이웃 선택 규칙, 학습 스모크)
```

## 6. 범위 밖

3D RoPE 대체(iii), 전 뷰 집합 기억(b), 예측 기억으로 학습(노출 편향 해소), B(공동 디코더), 어안 보정, 학습 예산 확대.
