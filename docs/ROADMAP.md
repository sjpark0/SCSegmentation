# 구현 계획 — 살아있는 문서

**기준**: [analysis/REPORT.md](analysis/REPORT.md)의 개선 제안 P1~P14와 실험 계획 11행.
**갱신**: 상태가 바뀔 때마다 이 파일과 노션(개발 › SCSegmentation › 구현 계획)을 함께 갱신합니다.
**최종 갱신**: 2026-09-08 (P12 절제 완료) · 갱신 이력은 문서 끝. 결과: [phase1-measurement.md](phase1-measurement.md) · [phase2-control.md](phase2-control.md) · [phase3-neighbourhood.md](phase3-neighbourhood.md)

상태 기호 — ✅ 완료 · 🔶 부분 · ⬜ 미착수 · ❌ 기각(근거 있음) · 🔒 선행 조건 대기

---

## 한눈에 보기

REPORT.md가 낸 제안 14개 중 **5개 완료(P1·P2·P3·P6 + G1), 4개 부분(P5·P7·P8·P13)**입니다. 평가 프로토콜(P3)과 대조군(P2)이 끝났고, 정확도 항목(P4·P5·P8·P12)이 남았습니다.
실험 계획 11행 중 **행 3·4·5·6을 돌렸습니다**(행 2는 보류). 핵심 결과: 패키지 내 W=0 대조군이 OneStage와 바이트 동일, 교차시점 메모리의 순효과는 사전 등록 기준 미달(+0.0057, CI [−0.0002, +0.0140]).

| P | 제안 | 상태 | 단계 | 비고 |
|---|---|---|---|---|
| P1 | seed/프레임 마스크 고정 해제 | ✅ | — | S1·S2·S3·S7로 반영. (d) 트림은 env-gated, `propagation_full` 가드는 이 패키지에서 발동 안 함 |
| P2 | 패키지 내 window ablation + 위생 3건 + closure | ✅ | 2 | XW 계열(커밋 44e5199). XW0 ≡ OneStage 바이트 동일, closure == all 실측, W 곡선은 W=1에서 포화 ([phase2-control.md](phase2-control.md)) |
| P3 | 주장을 담을 수 있는 평가 프로토콜 | ✅ | 1 | `eval/report_jf.py` v2 (`--paired --split --bin-by-nb --area-weighted --aggregation --ceiling --paper`). 엔드포인트 E0/E1/E2 **확정** (2026-09-07, [phase1-measurement.md](phase1-measurement.md) §5) |
| P4 | cross-view gather 재작성(양측 창·클리핑·t−1 fallback) | 🔶 | 3 | 모드 A~E 구현·절제 완료(09-08). 1차 기준 통과 모드 없음 → 규칙상 A 유지; C는 Fencing v0 +0.079·헤드라인 0.8468. **채택 여부 사용자 결정** ([phase3-neighbourhood.md](phase3-neighbourhood.md) §5) |
| P5 | seed 품질 검사와 복구 | 🔶 | 1(진단 ✅) → 3(복구 ⬜) | 진단 완료: 퇴화 시드 SAM 3 20개, 상한 +0.015, donor 가능분 +0.009, Welder +0.040 ([raw/seed_census.md](raw/seed_census.md)) |
| P6 | non-overlap int64 승격 제거 | ✅ | — | S4 |
| P7 | spatial predictor 은퇴 + autocast 정리 + fp16 프레임 | 🔶 | — | 은퇴·autocast ✅(S6). **fp16 ❌** — 출력 변경 확인, fp32(S5)로 대체 |
| P8 | reference 규칙(객체 수) + ceiling 열 | 🔶 | 1(ceiling ✅) → 3(규칙 ⬜) | ceiling 0.9375(max-id) → 0.9594(객체 수). Blocks +0.128 / MATF +0.087 / FacePaint +0.061 재현 |
| P9 | 하나의 propagation 계약 | ⬜ | 5 | 벤치마크 숫자 불변. P5 재전파를 안전하게 |
| P10 | seeding 응답 생략 | ⬜ | 5 | SCHEDULE H2. 시간만 절감(120~260초 → 수초) |
| P11 | 뷰당 detector 잔재 축소 | ⬜ | 5 | 안전한 부분집합만 |
| P12 | 이웃 토큰 조건화 ablation | ✅ | 3 | 완료(09-08). 단독 5 + 조합 2 + 꼬리표 곡선 6. **꼬리표 최적은 s=4(이웃을 5프레임 전으로), s=5에서 다시 나빠짐**. 조합은 가산 아닌 중복(GPS4 +0.00053). 적응 규칙은 영상 측정과 무상관(r=−0.19)이라 상수 s=4가 답. 헤드라인 0.8450→0.8454 ([phase3-conditioning.md](phase3-conditioning.md) §8) |
| P13 | 엔지니어링 위생 묶음 | 🔶 | 0·1·2·5 | 문서·형제 폴더 패치·`.dockerignore`·매니페스트·**CPU 테스트 62개** ✅. **dead API·배너 ⬜** |
| P14 | 이벤트 구동 cross-view 재시딩 | 🔒 | 5 | P4·P5 결과 본 뒤 |
| G1 | 토큰 폐기(0순위) | ✅ | — | 두 토큰 모두 외부 유출 없음 확인. 이미지는 토큰 없이 재빌드 |

---

## 왜 이 순서인가 — 세 가지 사실

`docs/` 전체를 세 관점(논문화·정확도·안정성)으로 독립 평가했고, 셋 다 같은 결론이었습니다.

**1. 현재 수치는 무승부입니다.** 헤드라인 +0.0017(0.8451 vs 0.8434) 대 데이터셋 표준오차 ±0.006.
짝지은 부트스트랩은 15개 중 8승, CI [−0.008, +0.011]로 0을 넘습니다. 면적 가중 J에서는 효과가 사라집니다.

**2. 핵심 메커니즘이 절제된 적이 없습니다.** 교차시점 메모리를 끈 패키지 내 대조군이 없습니다.
OneStage 비교는 패키지·요청 형태·세션 수가 달라 대조군이 아니고, `range(-4,0)` 윈도는 한 번도 바꿔 돌린 적이 없습니다.

**3. SAM 2 비교가 동일 조건이 아닙니다.** `fill_hole_area` 8 대 0, non-overlap 끔 대 켬, 저해상도 288 대 256.
우위가 전부 F(+0.006)인데 설정 차이가 같은 크기입니다.

그래서 **다음 목표는 J&F 0.001이 아니라, 관측된 차이를 귀속 가능한 메커니즘으로 바꾸는 것**입니다.
기존 파일만으로 계산해 보면 비기준 카메라에서 Δ≈+0.009로 헤드라인의 5배입니다 — 논문의 주장이 있어야 할 자리는 거기일 수 있습니다.

---

## 단계

기간은 1인 · RTX 6000 Ada 1대 · 15개 스윕 35~90분 기준입니다.

### Phase 0 — 보존 (1일, GPU 불필요) · P13 일부 — ✅ 완료 2026-09-07

| 할 일 | 상태 |
|---|---|
| `runMVSegAll.sh`·`launch_container.sh`에 `--memory=90g --memory-swap=90g` | ✅ 09-07 |
| `/tmp/sibling_backup`을 git 태그 `baseline-siblings`로 | ✅ 09-07 — `/tmp`는 재부팅으로 이미 소실, 태그는 `99c05c7`(마지막 무패치 푸시본)에 |
| 미푸시 커밋 푸시, `scsam3:pre-secret-backup` 삭제 | ✅ 09-07 — 이미지 삭제, 전체 이미지 이력에 토큰 레이어 0개 |

**완료 조건**: `git status -sb`에 ahead 없음 · 실행 스크립트 4개 전부 메모리 상한 보유 · 토큰 보유 이미지 없음.

### Phase 1 — 정직한 측정 (GPU 불필요) · P3 · P5 진단 · P8 ceiling — ✅ 완료 2026-09-07

결과와 엔드포인트 제안: **[phase1-measurement.md](phase1-measurement.md)**. 도구는 `eval/` (채점기·집계기 v2, 시드 집계, 매니페스트). 정본 원시 점수는 `Data/MVSeg/jf_v2.json`, 논문 표 47개는 `docs/raw/paper_tables.md`.

| 할 일 | P | 예상 판독 (REPORT 기준) | 결과 (15개 기준) |
|---|---|---|---|
| `report_jf.py --paired` | P3 | 과장 → 방어 가능한 주장 | ✅ MVOpt−OneStage +0.0024, 7/8, CI [−0.0003, +0.0065]; vs SAM 2 +0.0016, 8/7, CI [−0.008, +0.011]. **유의하지 않음** |
| `--split ref\|nonref` | P3 | nonref 0.8052 / 0.8093 / 0.8144 | ✅ 12개 값 재현. 15개 nonref 0.7976 / 0.7973 / 0.8012 (Δ +0.0039); ref에서는 Δ −0.0005 |
| `--bin-by-nb` | P3 | nb=4 +0.0047, nb=0 −0.0001 | ✅ 재현(ΔJ). nb≥4 ΔJ&F +0.0043 (13/12), nb=0 −0.0001 |
| `--area-weighted` | P3 | Δ≈0 | ✅ MVOpt−OneStage +0.0001. 새 사실: SAM 3 vs SAM 2 +0.030 (큰 객체 우위) |
| 집계 방식 명시, `--common` 상시 | P3 | OneStage는 시퀀스 집계에서 동타 | ✅ sequence 0.8358/0.8357/0.8381, 순위 불변. `--common` 기본값 |
| 퇴화 시드 집계 | P5 | Welder 손실의 상한 | ✅ SAM 3 20개(missing 15·tiny 4·misaligned 1). 상한 +0.015, donor 가능분 +0.009, Welder +0.040 |
| `--ceiling` 열 | P8 | Blocks +0.128, MATF +0.087, FacePaint +0.061 | ✅ 정확히 재현. 평균 ceiling 0.9375 → 0.9594 |
| 마스크 폴더 매니페스트 | P13 | 지금은 mtime뿐 | ✅ 117개 폴더 `MANIFEST.json`(내용 해시·추정 출처). `runMVSeg.py`가 새 실행마다 자동 기록 |
| 논문이 보고할 SAM 3 열 하나 확정 | — | `experiments.md` (2) | ✅ `SegMaskSam3MVOpt` 하나 (E0, 확정 2026-09-07) |

**완료 조건**: `report_jf.py` 하나로 논문의 모든 표 재생성 ✅ · 기존 폴더 재채점이 0.8449/0.8465/0.8497 재현 ✅ (396/396 항목 동일) · 어느 엔드포인트가 주장을 감당하는지 문서로 결정 ✅ (E0/E1/E2, phase1-measurement.md §5, 2026-09-07 확정).

**확정된 엔드포인트** — E0 헤드라인: 15개·pooled·as-is·전체 프레임, SAM 3 열은 MVOpt 하나, 짝지은 CI 병기 · E1 메커니즘: 비기준 카메라 J&F와 nb≥4 구간의 짝지은 Δ, 대조군은 패키지 내 W=0(행 3) · E2 보조: 면적 가중, ceiling, 퇴화 시드, 시퀀스 집계 · 쓰지 않음: 12개 부분집합 p값, OneStageNew 별도 열, exported-only.
**Phase 2에 사전 등록된 합격 기준 (E1)**: nb≥4 구간 Δ(XW4 − XW0)의 클러스터 부트스트랩 95% CI가 0을 제외 · nb=0 구간 |Δ| < 0.001. 못 넘으면 "효과 없음"으로 보고합니다.
**남은 결정 2건**: P5 misaligned 임계값 0.10 vs 0.20 · 수 px 프롬프트 객체의 프로토콜 처리.

### Phase 2 — 없는 대조군 · P2 · P13 테스트 — ✅ 완료 2026-09-07

결과와 계열 결정 입력: **[phase2-control.md](phase2-control.md)**. 코드는 커밋 44e5199, 마스크는 `SegMaskSam3XW{0,1,2,4,6}`·`XW4all`, 원시 점수 `Data/MVSeg/jf_xw.json`, 표 `docs/raw/phase2/`.

| 할 일 | 세부 |
|---|---|
| 위생 수정 3건 | (1) `range(-4,0)` → `range(-W,0)` + 생성자 인자 + env · (2) `if prev_spatial_idx < 0: continue` · (3) `_, unselected_cond_outputs1 = ...` |
| `--track-cams closure` | `track_idx = range(0, max(scored_idx)+1)`. Welder 553초 → ~60초 |
| `--xview-window N` → `SegMaskSam3XW{N}` | 실험 행 3·4의 배관 |
| CPU-only pytest | 메모리 불변식 7종, N=3 IndexError/wrap/clobber, ref=N−1, 21프레임 |
| **행 3·4 실행**: XW0 vs XW4, 15개 | 이 방법이 가진 적 없는 첫 유효 대조군 |

| 할 일 | 결과 |
|---|---|
| 위생 수정 3건 | ✅ `cross_view_window`·`cross_view_hygiene`(기본 off, 플래그 없이는 바이트 동일 — 가드 Blocks·Fencing diff 0) |
| `--track-cams closure` | ✅ closure == all: Welder(4 vs 46세션)·Dog(4 vs 41)·Blocks diff 0. Welder 433 → 86초 |
| `--xview-window N` | ✅ `SegMaskSam3XW{N}`, 위생 자동 on, 거부 규칙 6종, 모델 read-back |
| CPU pytest | ✅ 62개(수집 함수 등가 20k 구성, 실제 메서드 golden, closure 보조정리, 러너 해석표, 메모리 불변식 S1~S7) |
| 행 3·4 | ✅ **XW0 ≡ OneStage 바이트 동일(19,848 PNG)**. XW4 − XW0: 15개 +0.0024, 비기준 +0.0039, nb≥4 +0.0057 (12/8), 클러스터 CI [−0.0002, +0.0140] → **사전 등록 기준 미달, "효과 없음"**. nb=0 정확히 0 |
| C2 버그 효과 | MVOpt − XW4 = +0.0001 (CI [−0.0001, +0.0002]): 마스크는 바뀌고 점수는 안 바뀜 |
| 행 5 (W 곡선, 6개) | ✅ XW0 0.8620 / XW1 0.8681 / XW2 0.8685 / XW4 0.8686 / XW6 0.8688 — W=1에서 포화 |

**주의**: 위생 수정은 출력을 바꿉니다. **0.8451과 다른 시스템**이 되므로 논문은 한 계열을 택해 끝까지 그것으로 보고합니다.

**완료 조건**: closure = all ✅ · XW0/XW4 15개 존재·채점 ✅ · 순효과가 nonref·nb 구간·paired CI로 보고 ✅ · 어느 계열이 헤드라인인지 문서로 결정 🔶 (phase2-control.md §6: XW 계열 권장, **사용자 확정 대기**).

### Phase 3 — 측정된 손실 회복 (3~4주) · P5 · P4 · P8 · P12

각 항목은 **실행 전에 합격 기준을 적고**, 못 넘으면 측정값과 함께 닫습니다.

| 할 일 | P | 기대값 | 근거 성격 |
|---|---|---|---|
| `RepairSeeds()` — 퇴화 시드 감지·donor 재전파 | P5 | Welder 0.8437 → ~0.879, 헤드라인 +0.0023~0.0027 | **정량화됨** |
| 양측 창 + 경계 클리핑 + t−1 fallback | P4 | Fencing ~+0.02 | **측정됨(09-08)**: C에서 Fencing +0.052(v0 +0.079), 15개 +0.0018 vs A, 비기준은 불변. 1차 기준 미달 |
| `--reference count` | P8 | 모든 방법 동반 상승, 순위 불변 | 프로토콜 |
| 이웃 obj_ptr 4토큰 + 게이트 + 행 이동 | P12 | 기준: **이웃 받는 36대(기준 카메라 포함)에서 V−XW1의 클러스터 CI가 0 제외** & 기준 카메라 무손실 ([phase3-conditioning.md](phase3-conditioning.md) §1.3; REPORT의 옛 'nonref +0.005' 기준을 대체) | 진단 실험 |

**선행**: P5 구현 전 `SCSam3VideoInference.py:1855-1870` refine 분기가 같은 tracker state에 cond frame을 추가하는지 로그로 확인 (심사자 이견 지점).

### Phase 4 — 공정한 비교 (1~2주) · 실험 행 11

| 할 일 | 기대 |
|---|---|
| SAM 2 `apply_postprocessing=False`, `non_overlap_masks=True` 재실행 | F +0.006이 후처리 몫일 수 있음 |
| 인용 중인 SAM 2 열(`SegMaskNew1`)의 스크립트·설정 특정, 재현 | 재현 실험은 `New3`를 재현했음 |
| SAM 3 `fill_hole_area` ∈ {0, 8, 16} | 배너에 값 출력 |
| 기준 시점 3개로 강제해 분산 | 논문의 오차 막대 |

### Phase 5 — 부채 (여유 시간) · P9 · P10 · P11 · P13 나머지 · P14

`return_output=False`(P10, 단독 패치로 재심사) · propagation 계약(P9) · detector 잔재(P11 안전 부분집합) ·
dead API·`copy.py`·MultiGPU 이름(P13) · TwoStage 등록하거나 되돌리거나(중간 상태 금지) ·
ForSam2New 경로 분석 · 재시딩(P14, P4·P5 후).

---

## 실험 계획 상태 (REPORT.md 11행)

| # | 구성 | 폴더 | 상태 | 단계 |
|---|---|---|---|---|
| 0 | SAM 2 baseline | `SegMaskNew1` | ✅ 기존 | — |
| 1 | OneStage / written | `SegMaskSam3OneStage` | ✅ 기존 | — |
| 2 | OneStage / closure (타이밍) | `SegMaskSam3OneStageC` | 보류 | 2 — XW0(= OneStage 출력)의 closure 타이밍으로 대체 |
| 3 | **XW0 대조군** | `SegMaskSam3XW0` | ✅ 09-07 | 2 — OneStage와 바이트 동일 |
| 4 | **XW4 위생 수정** | `SegMaskSam3XW4` | ✅ 09-07 | 2 — 0.8451 (= MVOpt) |
| 5 | W ∈ {1,2,6} 곡선 (6개 데이터셋) | `SegMaskSam3XW{1,2,6}` | ✅ 09-07 | W=1 포화 |
| 6 | closure = all 검증 | `SegMaskSam3XW4all` | ✅ 09-07 | Welder·Dog·Blocks diff 0 |
| 7 | 양측 창 (P4) | `SegMaskSam3XW1{B,C,D,E}` | ✅ 09-08 | 3 — 규칙상 A 유지, C 채택은 결정 대기 |
| 8 | seed 복구 (P5) | `SegMaskSeedRepair` | ⬜ | 3 |
| 9 | reference 규칙 (P8) | `*_refcount` | ⬜ | 3 |
| 10 | 이웃 토큰 (P12) | `SegMaskSam3XW1{G,P,GP,S1..S5,PS4,GPS4}` | ✅ 09-08 | 3 |
| 11 | 설정 매칭 (E6) | `SegMaskNew1_matched` | ⬜ | 4 |

행 3·4가 나왔습니다. 이후 정확도 제안(P4·P5·P8·P12)은 **XW 계열 위에서, E1 엔드포인트로** 판정합니다.

---

## 하지 말 것 (근거: SCHEDULE.md, investigations-closed.md)

| 항목 | 이유 |
|---|---|
| fp16 프레임 저장 | 소비자의 fp32 캐스트가 가수 비트를 복구 못 함 → **마스크 변경** (H3) |
| bf16 autocast 컨텍스트 변경 | `decoder.py:71` fp32 구간이 깨짐 → **모든 마스크 변경** (H4) |
| 프레임마다 `clear_autocast_cache()` | 피크 감소 없이 ~5 GiB/프레임 HBM 트래픽. 경계에서 한 번만 |
| 시점 전파 시작 인덱스 명시 | Blocks 0.7484 → 0.2827 (investigations-closed 4번) |
| `--track-cams written`을 NewMem 계열에 | 윈도가 3대 리스트를 감아 **다른 모델**이 됨 (operations 함정 1) |
| OneStageNew 폴더에서 개발 | 동결 스냅샷. 개발은 MVOpt (`FROZEN.md`) |

---

## 갱신 이력

| 일자 | 내용 |
|---|---|
| 2026-09-04 | v1 — 단계별 계획 초안 (세 관점 비교) |
| 2026-09-07 | v2 — REPORT.md P1~P14 기준으로 재구성. 상태표·실험 행 상태·갱신 이력 추가. 노션 개발 › SCSegmentation 아래 동기화 |
| 2026-09-07 | **Phase 0 완료.** 메모리 상한 4개 스크립트 전부, `baseline-siblings` 태그, 토큰 보유 이미지 삭제. open-items #1(토큰) 해소로 정정 |
| 2026-09-07 | **Phase 1 완료.** `eval/` 채점기·집계기 v2(발표 수치 396/396 재현), 퇴화 시드 집계, 매니페스트 117개. P3 ✅, P5·P8 진단/ceiling ✅, P13 매니페스트 ✅. 결과·엔드포인트 제안은 phase1-measurement.md — 엔드포인트·misaligned 임계값·수 px 프롬프트 처리는 사용자 확정 대기 |
| 2026-09-07 | **엔드포인트 확정** (E0 헤드라인 · E1 메커니즘 = 비기준·nb≥4, 대조군 W=0 · E2 보조; 12개 부분집합 p값·OneStageNew 별도 열·exported-only는 사용 안 함). P3 ✅ 확정. 남은 결정: misaligned 임계값, 수 px 프롬프트 처리 |
| 2026-09-07 | **Phase 2 완료.** XW 계열 구현(44e5199, 기본 경로 바이트 동일, 테스트 62개), GPU 56회. XW0 ≡ OneStage, closure == all, E1 **미달**(+0.0057, CI [−0.0002, +0.0140]), C2 효과 0, W=1 포화. P2 ✅. 계열 결정(XW 권장)은 사용자 확정 대기 |
| 2026-09-08 | **P12 조합·곡선 완료** (사전 등록 0955845 → GPU 75회). 꼬리표 곡선 s=0..5: +0.0000/+0.0001/+0.0002/+0.0004/**+0.00043**/+0.0003 — **s=4 최적, s=5에서 하락**(단조 아님). 조합 GPS4 +0.00053(가산 예측 +0.00085, 중복 예측 +0.00042 중 중복 쪽), 게이트 기여는 조합 안에서도 +0.0001. 데이터셋별 최적 s(2~5)와 영상 측정 비율의 상관 r=−0.19 → **적응 규칙 근거 없음, 상수 s=4**. 권장 GPS4(또는 최소 변경 S4), 채택은 사용자 결정 |
| 2026-09-08 | **P12 절제 완료** (사전 등록 011fa86 → 구현 a72d462 → GPU 81회). 감시값 전부 통과. G 미달, P·GP·S2·S4 CI 통과하나 크기 +0.0002~0.0004(실용 기준의 1/10)이고 상위 셋 동률 → **규칙상 손잡이 없음 유지**. 게이트는 이웃 토큰 26%를 버려도 Δ+0.0001; **정보를 더하지 않는 S4가 포인터 채널 P와 동등** — 공간축 채널이 내용으로 기여하지 않는다는 증거. 채택 여부는 사용자 결정 |
| 2026-09-08 | **P4 절제 완료** (사전 등록 2847fd2 → 구현 e1bba33 → GPU 80회). 모드 A~E, 감시값 전부 통과. 1차 기준(비기준 CI>0) 통과 모드 없음 → **규칙상 A 유지**. B·C·E는 S1(Fencing v0 ≥0.85) 통과, C·E 헤드라인 0.8468(+0.0018 vs A), E는 C 대비 이득 없음. C 채택 여부는 사용자 결정 |
