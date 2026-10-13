# GeoMem-star — SAM 3 추적기 기억을 3D 기하 기억으로 바꾸는 어댑터 설계

> 2026-10-06. 대화에서 절별로 합의한 설계. 벤치마크 스펙 [2026-10-06-mvseg2d-benchmark-design.md](2026-10-06-mvseg2d-benchmark-design.md)의
> 과제·채점 규약을 그대로 따른다. 구현 계획은 이 문서를 근거로 따로 쓴다.

## 0. 한 문장

SAM 3 추적기가 "몇 프레임 전인가"로 기억을 읽던 시간 위치 인코딩을 **π³ 포인트맵에서 만든 3D 위치 임베딩**으로
바꾸고, 기억 집합을 "직전 프레임들" 대신 **기준 카메라 하나(별 모양)**로 두어, 다중시점 집합을 순서 없이 한 번에
분리하는 모델로 확장한다. 바뀐 부분(작은 MLP, 선택적으로 기억 어텐션)만 MVSeg 장면 교차검증으로 학습해, 1차
벤치마크 기준선 `sam3_prop`(J&F 0.890)을 넘는지 본다.

## 1. 배경과 동기

- 1차 벤치마크(`docs/results-stage1.md`): 뷰=비디오 SAM 3 전파 0.890이 포즈 없는 재투영(0.622)·대응(0.575)·정제(0.688)를
  모두 크게 앞섰다. 남은 손실은 정상 쌍의 경계 오차, 작은 객체(<2k px 0.740), 어안(0.862)에 있다.
- 문헌(`../SCSegmentation/docs/mvimage-seg-survey.md`)에서 뷰=비디오 전파를 같은 과제로 이긴 유일한 가족은 "포인트맵
  3D 위치를 SAM 2 기억/디코더에 주입하고 학습"한 MV-SAM·3AM·G²TAM이다. SAM 3 위에서, 리그 데이터로 평가한 것은 없다.
- 이전 프로젝트의 교차시점 메모리(`SCSam3/demoSCSam3MVOpt/xview_gather.py`)는 이웃 카메라 기억 토큰을 **시간 거리
  행을 재활용**해 붙였다(카메라 번호 거리 = 시간 거리). 기하도 학습도 없었고 이득은 +0.005였다. 이 설계는 그 빈자리
  (기하 + 학습)를 채운다.
- 사용자 결정: C(장면별 3D 필드 최적화)는 π³ 기하가 상한이라 기각. A(기하 기억 어댑터)를 B(다중뷰 공동 디코더)보다
  먼저 구현해 "기존보다 좋아지는가"를 확인한다.

## 2. SAM 3 추적기의 관련 구조 (읽기 결과, 수정 지점)

경로: `UP = ../SCSegmentation/SCSam3/sam3/sam3`, `SC = ../SCSegmentation/SCSam3/demoSCSam3MVOpt`.

| 항목 | 위치 | 사실 |
|---|---|---|
| 기억 토큰 | `UP/model/memory.py:160-203`, `sam3_tracker_base.py:800-849` | 뷰·객체당 `maskmem_features` [B,64,72,72] + `maskmem_pos_enc`(2D sine, 64). 객체는 배치 차원 |
| 시간 위치 행 | `sam3_tracker_base.py:103-106`, 더하는 곳 `:673-677` | `maskmem_tpos_enc` [7,1,1,64]. 조건 프레임은 6행, 직전 k프레임은 6−k행 |
| 기억 프레임 선택 | `:586-650` | 조건 프레임(최대 4) + 직전 6프레임(`num_maskmem=7`) |
| 기억 어텐션 | `:785-797`, `UP/model/decoder.py:611-710` | 4층 교차 어텐션, 질의 256차원(현재 뷰 특징 + `current_vision_pos_embeds`), 키 64차원(기억 + 위치), 키에 프레임 내부 2D RoPE |
| 객체 포인터 | `:695-773` | 기억 프레임당 256차원 포인터를 64차원 토큰 4개로 쪼개 기억 뒤에 붙임, 시간 거리 sine→Linear |
| 조립 함수 | `_prepare_memory_conditioned_features` `:559-798`, 호출 `track_step :974-984` | 기억 없음(초기 조건 프레임)이면 `no_mem_embed`로 조기 반환 |
| 학습 인프라 | `UP/train/` | 추적기 학습 설정·손실·로더 **없음**. `forward_tracking`(`:851-924`), `loss_fns.py`의 dice/focal/iou만 재사용 가능. 동결은 `requires_grad_(False)` 직접 |
| 주의 | `:681-693` | `prob_to_dropout_spatial_mem`, `rng`가 `__init__`에 없어 `training=True`로 그 경로를 타면 AttributeError. 래퍼는 `eval()` 모드로 두고 autograd만 켠다 |

## 3. 모델 수정부 (합의 1절)

SAM 3 패키지 파일은 수정하지 않는다. `geomem/`의 **래퍼**가 추적기 내부 함수를 호출해 두 가지만 바꾼다.

### 3.1 기억 집합: 별 모양

대상 뷰 v를 디코딩할 때 기억은 기준 카메라 c_ini 하나다. c_ini의 (학습: 정답, 추론: 정답) 마스크를 기억 인코더에
넣어 만든 객체별 토큰(72×72×64)과 객체 포인터(4토큰)가 전부. 시간 위치 행은 조건 프레임 행(6행) 하나로 고정하고,
직전 프레임 선택 규칙은 우회한다. 포인터의 시간 거리는 상수 1. 뷰 처리 순서와 결과가 무관하다.

### 3.2 3D 위치 임베딩 (학습 대상)

1. 뷰 v의 π³ 포인트맵(작업 해상도, c_ini를 입력 0번으로 둔 π³ 1회 실행이라 **전 뷰가 공통 좌표계**)을 c_ini 카메라
   좌표계로 옮기고(π³ c2w), 유효점(신뢰도 > 0.1)의 c_ini 원점 거리 중앙값으로 나눠 스케일을 정규화한다.
2. SAM 3가 이미지를 1008×1008로 비율 없이 리사이즈하므로, 포인트맵도 같은 방식으로 72×72 격자에 **면적 평균**(유효점만)
   풀링해 셀마다 3D 좌표 `p ∈ R³`와 유효 비율을 얻는다. 유효 비율 < 0.25인 셀은 "기하 없음".
3. `φ(p) = [p, sin(2^k π p), cos(2^k π p)]_{k=0..7}` (51차원) → 공유 MLP(51→128→128, GELU) → 키 머리 Linear(128→64),
   질의 머리 Linear(128→256). "기하 없음" 셀은 학습되는 벡터 `e_none`(128) 하나를 MLP 출력 대신 쓴다.
4. 키 머리 출력은 c_ini 기억 토큰의 `maskmem_pos_enc`에 더하고(+ 고정 시간 행), 질의 머리 출력은 대상 뷰의
   `current_vision_pos_embeds`(최상위 72×72, 256)에 더한다. 양쪽에 같은 3D 임베딩이 들어가므로 어텐션이 같은 3D 위치를
   직접 맞출 수 있다. 3D 임베딩은 객체에 무관하므로 배치(객체) 차원에 브로드캐스트한다.
5. **0 초기화**: 두 머리의 가중치·편향을 0으로 둔다. 학습 전 모델 = 기하 없는 별 모양 SAM 3(대조군 `star`)과
   출력이 바이트 단위로 같아야 한다(테스트로 확인).

### 3.3 그대로 두는 것

이미지 인코더(ViT), 기억 인코더, 마스크 디코더(SAM 헤드, IoU·존재 로짓), 객체 포인터 경로, 키 RoPE, 1008 입력 해상도,
`fill_hole_area = 0`.

### 3.4 학습 범위 — 세 팔

| 팔 | 학습 파라미터 | 손실 |
|---|---|---|
| `geomem` | MLP(51→128→128) + 두 머리 + `e_none` (≈ 10만) | 지도(3.5) |
| `geomem_ft` | 위 + 기억 어텐션 4층(`TransformerEncoderCrossAttention`, 수백만) | 지도 |
| `geomem_ssl` | `geomem`과 같음 | 지도 + 비주석 뷰 재투영 일관성(4.4) |

### 3.5 전방 계산 (학습·추론 공통)

```
ref: pix_feat(c_ini) [캐시] + GT 마스크 → 기억 인코더 → maskmem_features, maskmem_pos_enc, obj_ptr   (객체 B개)
     pos_k = maskmem_pos_enc + tpos[6] + KeyHead(MLP(φ(P_ref)))                                  (P_ref: c_ini 72×72 3D)
tgt: vision_feats(v) [캐시], pos_q = current_vision_pos_embeds + QueryHead(MLP(φ(P_v)))
     memory_attention(src=vision_feats(v), src_pos=pos_q, prompt=[mem_ref; obj_ptr 토큰], prompt_pos=[pos_k; ptr_pos])
     → SAM 헤드 → 마스크 로짓 [B,1,H,W], IoU 예측 [B], 존재 로짓 [B]
```

## 4. 학습 파이프라인 (합의 2절)

### 4.1 데이터

- `prepare_data.py --train-frames`: `../SCSegmentation/Data/MVSeg/<scene>/`에서 라벨 21프레임(start..start+20)의 이미지(전
  카메라)와 라벨맵(주석 3대)을 `Data/<scene>/frames/<t>/{images,masks}/`에 복사(≈2.5 GB). 벤치마크 입력 `images/`,
  `masks/`는 손대지 않는다.
- π³: (장면, t)마다 전 카메라로 실행해 `Data/<scene>/Cache/pi3_frames/<t>/`에 캐시(추론과 같은 조건). 기준 프레임
  캐시 `Cache/pi3/`는 이미 있다.
- 백본 특징: 주석 3대 × 21프레임의 `vision_feats`(3레벨)·`current_vision_pos_embeds`를 fp16으로 `Cache/sam3_feats/<t>/<cam>.pt`에
  캐시(≈1,071장, 10 GB 안팎). 비주석 뷰(팔 3용)는 샘플마다 2~4개만 쓰므로 **즉석 계산**한다.

### 4.2 샘플과 fold

- 샘플 = (장면, t, 기준 r, 대상 v), r ≠ v ∈ 주석 3대. (장면, t)당 6쌍, 총 2,142. 객체 = r 라벨맵의 id 전부, 한 스텝 최대
  16개 무작위 부분집합. 증강은 기준 카메라 선택뿐(반전 없음).
- 5-fold, 장면 단위, 리그 층화: 어안 4장면이 서로 다른 fold에 하나씩, 핀홀 13장면을 고정 시드로 배치. `geomem/folds.json`에
  적어 커밋. fold k 모델은 fold k 장면을 보지 않는다.

### 4.3 지도 손실

SAM 2 추적 학습 조합, 상류 `UP/train/loss/loss_fns.py`의 함수 사용. 대상 뷰 마스크 로짓에 sigmoid focal(가중 20) +
dice(1), IoU 예측에 L1(1, 목표 = 예측·정답 IoU), 존재 로짓에 BCE(1, 목표 = 정답 비어 있지 않음). 대상 뷰에서 안
보이는 객체도 손실에 포함(빈 마스크가 정답).

### 4.4 재투영 일관성 손실 (`geomem_ssl`)

샘플마다 비주석 뷰 u를 2~4개 무작위로 추가한다. u의 예측 로짓 `L_u`(원 해상도 시그모이드)를 u의 포인트맵으로
3D에 올려 c_ini에 투영하고, c_ini 정답 마스크 `G_ref`를 투영 좌표에서 `grid_sample`로 읽어 `soft-dice(σ(L_u)[valid], G_ref∘proj[valid])`
를 계산한다. `valid` = u의 깊이 유효 ∧ c_ini에서 가시(`warp.visible`, 가림 허용 1%). 가중치 0.5 고정. 반대 방향(c_ini →
u)은 쓰지 않는다(정답이 c_ini에만 있으므로).

### 4.5 최적화

AdamW, bf16 autocast, 배치 1샘플, 코사인 감쇠, **10 epoch 고정, 모델 선택 없음**(검증으로 epoch을 고르지 않는다).
학습률: MLP·머리·`e_none` 1e-3, 기억 어텐션(팔 2) 1e-5, weight decay 0.01. 시드 고정. 체크포인트·손실 곡선 CSV를
`geomem/runs/<arm>/fold<k>/`에 저장. 예상 비용 fold당 30분(팔 3은 1.5배), 전체 ≈ 7시간, 48 GB 1장.

## 5. 평가와 대조군 (합의 3절)

- 벤치마크 방법 `geomem`, `geomem_ft`, `geomem_ssl`: 장면마다 그 장면이 속한 fold의 체크포인트로, 기준 프레임에서 c_ini
  기억으로 N−1 뷰를 디코딩해 기존 형식(`Result/<method>/<cam>/<id>.png`, RUN.json)으로 쓴다. RUN.json config에 fold,
  체크포인트 SHA256, 학습 커밋을 기록.
- 대조군 `star`: 같은 코드 경로를 학습 전(0 초기화) 가중치로 실행(`--untrained`). 기하 없는 별 모양 SAM 3.
- 기준선 `sam3_prop` 0.890(1차).
- 읽는 값: 전체 J&F와 네 축(장면·크기·리그·시점차). 핵심 차 = `geomem − star`(기하 효과), `geomem − sam3_prop`(기존
  1단계 대비). 사전 기대값은 적되 임계값으로 쓰지 않는다. 학습 시간은 별도 표.
- 모든 방법은 `scsam3` 이미지에서 돈다(torch 2.10). 학습도 같은 이미지.

## 6. 코드 구조

```
geomem/
  pointmap.py      # π³ 캐시 → c_ini 좌표계 정규화 → 72×72 풀링 (p, valid)   [numpy, 테스트 가능]
  embed.py         # φ(p), MLP, 두 머리, e_none; 0 초기화                      [torch]
  wrapper.py       # SAM 3 추적기 내부 함수 호출: 기억 인코딩, 별 모양 조립, 어텐션, SAM 헤드
  data.py          # 샘플 열거, fold 분할, 캐시 로더
  losses.py        # 지도 손실 묶음, 재투영 일관성(differentiable warp)
  cache_feats.py   # 백본 특징 캐시 생성 (scsam3)
  train.py         # 학습 루프 (scsam3)
  folds.json
methods/geomem/run.py   # 벤치마크 방법 (--arm, --untrained)
scripts/run_geomem_*.sh
tests/test_geomem_*.py  # 풀링·정규화·fold·손실·0초기화 동일성(CPU 가능한 부분)
```

## 7. 범위 밖

- B(다중뷰 공동 디코더), 2라운드 점진 기억(GeoMem-progressive).
- 무라벨 리그(EBS 19대 등)의 자기지도 학습 — `geomem_ssl`이 효과를 보이면 다음 후보.
- 어안 보정, 작은 객체 크롭. 결과는 축으로 보고만 한다.
- 상류 SAM 3 패키지 수정. 불가피하면 복사본을 `geomem/` 안에 두고 기록한다.
