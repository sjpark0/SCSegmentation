# 구현 계획 — 살아있는 문서

**기준**: [analysis/REPORT.md](analysis/REPORT.md)의 개선 제안 P1~P14와 실험 계획 11행.
**갱신**: 상태가 바뀔 때마다 이 파일과 노션(개발 › SCSegmentation › 구현 계획)을 함께 갱신합니다.
**최종 갱신**: 2026-09-07 · 갱신 이력은 문서 끝.

상태 기호 — ✅ 완료 · 🔶 부분 · ⬜ 미착수 · ❌ 기각(근거 있음) · 🔒 선행 조건 대기

---

## 한눈에 보기

REPORT.md가 낸 제안 14개 중 **3개가 끝났고 전부 메모리 항목**입니다. 정확도·평가·대조군 항목은 하나도 시작하지 않았습니다.
실험 계획 11행 중 **새로 돌린 행은 0개**입니다(행 0·1은 기존 결과).

| P | 제안 | 상태 | 단계 | 비고 |
|---|---|---|---|---|
| P1 | seed/프레임 마스크 고정 해제 | ✅ | — | S1·S2·S3·S7로 반영. (d) 트림은 env-gated, `propagation_full` 가드는 이 패키지에서 발동 안 함 |
| P2 | 패키지 내 window ablation + 위생 3건 + closure | ⬜ | **2** | 심사 3/3 합의 1순위. 대조군의 전부 |
| P3 | 주장을 담을 수 있는 평가 프로토콜 | ⬜ | **1** | GPU 불필요. 이후 모든 판정의 엔드포인트 |
| P4 | cross-view gather 재작성(양측 창·클리핑·t−1 fallback) | ⬜ | 3 | P2 (1)~(3) 위에 얹음 |
| P5 | seed 품질 검사와 복구 | ⬜ | 1(진단) → 3(복구) | 유일하게 정량화된 정확도 이득 |
| P6 | non-overlap int64 승격 제거 | ✅ | — | S4 |
| P7 | spatial predictor 은퇴 + autocast 정리 + fp16 프레임 | 🔶 | — | 은퇴·autocast ✅(S6). **fp16 ❌** — 출력 변경 확인, fp32(S5)로 대체 |
| P8 | reference 규칙(객체 수) + ceiling 열 | ⬜ | 1(ceiling) → 3(규칙) | 프로토콜 변경, 순위 불변 |
| P9 | 하나의 propagation 계약 | ⬜ | 5 | 벤치마크 숫자 불변. P5 재전파를 안전하게 |
| P10 | seeding 응답 생략 | ⬜ | 5 | SCHEDULE H2. 시간만 절감(120~260초 → 수초) |
| P11 | 뷰당 detector 잔재 축소 | ⬜ | 5 | 안전한 부분집합만 |
| P12 | 이웃 토큰 조건화 ablation | ⬜ | 3 | 이웃 obj_ptr이 가장 유망. 사전 선언 기준으로 채택/기각 |
| P13 | 엔지니어링 위생 묶음 | 🔶 | 0·2·5 | 문서·형제 폴더 패치·`.dockerignore` ✅. **테스트·dead API·배너 ⬜** |
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

### Phase 1 — 정직한 측정 (1주, GPU 불필요) · P3 · P5 진단 · P8 ceiling ★

**이 단계가 이후 전부를 바꾸거나 죽일 수 있고, 아무것도 망가뜨릴 수 없습니다.** 디스크에 있는 `jf_*.json`과 마스크만 씁니다.

| 할 일 | P | 예상 판독 (REPORT 기준) |
|---|---|---|
| `report_jf.py --paired` (36캠 부호검정 + 부트스트랩 CI) | P3 | 과장 → 방어 가능한 주장 |
| `--split ref\|nonref` | P3 | nonref 0.8052 / 0.8093 / 0.8144, Δ≈+0.009 |
| `--bin-by-nb` (min(view_idx, W)) | P3 | nb=4 +0.0047, nb=0 −0.0001 |
| `--area-weighted` (`eval_jf.py`에 `gt_area`) | P3 | Δ≈0 |
| 집계 방식 명시(object-pooled vs per-sequence), `--common` 상시 | P3 | OneStage는 시퀀스 집계에서 동타 |
| 퇴화 시드 집계 (기존 `<cam>/0/*.png`, 데이터셋별) | P5 | Welder 손실의 상한 파악, 비용 0 |
| `--ceiling` 열 (GT 시드 상한) | P8 | Blocks +0.128, MATF +0.087, FacePaint +0.061 |
| 마스크 폴더마다 매니페스트(git rev·인자·플래그) | P13 | `Data/`가 gitignore라 지금은 mtime뿐 |
| 논문이 보고할 SAM 3 열 하나 확정 | — | `experiments.md` (2) |

**완료 조건**: `report_jf.py` 하나로 논문의 모든 표 재생성 · 기존 폴더 재채점이 0.8449/0.8465/0.8497 재현 · **어느 엔드포인트가 주장을 감당하는지 문서로 결정**.

### Phase 2 — 없는 대조군 (2주) · P2 · P13 테스트

| 할 일 | 세부 |
|---|---|
| 위생 수정 3건 | (1) `range(-4,0)` → `range(-W,0)` + 생성자 인자 + env · (2) `if prev_spatial_idx < 0: continue` · (3) `_, unselected_cond_outputs1 = ...` |
| `--track-cams closure` | `track_idx = range(0, max(scored_idx)+1)`. Welder 553초 → ~60초 |
| `--xview-window N` → `SegMaskSam3XW{N}` | 실험 행 3·4의 배관 |
| CPU-only pytest | 메모리 불변식 7종, N=3 IndexError/wrap/clobber, ref=N−1, 21프레임 |
| **행 3·4 실행**: XW0 vs XW4, 15개 | 이 방법이 가진 적 없는 첫 유효 대조군 |

**주의**: 위생 수정은 출력을 바꿉니다. **0.8451과 다른 시스템**이 되므로 논문은 한 계열을 택해 끝까지 그것으로 보고합니다.

**완료 조건**: closure = all (`diff -rq` 0, Blocks·Fencing·Welder) · XW0/XW4 15개 존재·채점 · 순효과가 nonref·nb 구간·paired CI로 보고 · 어느 계열이 헤드라인인지 문서로 결정.

### Phase 3 — 측정된 손실 회복 (3~4주) · P5 · P4 · P8 · P12

각 항목은 **실행 전에 합격 기준을 적고**, 못 넘으면 측정값과 함께 닫습니다.

| 할 일 | P | 기대값 | 근거 성격 |
|---|---|---|---|
| `RepairSeeds()` — 퇴화 시드 감지·donor 재전파 | P5 | Welder 0.8437 → ~0.879, 헤드라인 +0.0023~0.0027 | **정량화됨** |
| 양측 창 + 경계 클리핑 + t−1 fallback | P4 | Fencing ~+0.02 | 추론, 미측정 |
| `--reference count` | P8 | 모든 방법 동반 상승, 순위 불변 | 프로토콜 |
| 이웃 obj_ptr 4토큰 | P12 | 기준: nonref +0.005 초과 & reference 무손실 | 진단 실험 |

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
| 2 | OneStage / closure (타이밍) | `SegMaskSam3OneStageC` | ⬜ | 2 |
| 3 | **XW0 대조군** | `SegMaskSam3XW0` | ⬜ | **2** |
| 4 | **XW4 위생 수정** | `SegMaskSam3XW4` | ⬜ | **2** |
| 5 | W ∈ {1,2,6} 곡선 (6개 데이터셋) | `SegMaskSam3XW{1,2,6}` | ⬜ | 2~3 |
| 6 | closure = all 검증 | `SegMaskSam3XW4all` | ⬜ | 2 |
| 7 | 양측 창 (P4) | `SegMaskSam3XV_two{2,4}` | ⬜ | 3 |
| 8 | seed 복구 (P5) | `SegMaskSeedRepair` | ⬜ | 3 |
| 9 | reference 규칙 (P8) | `*_refcount` | ⬜ | 3 |
| 10 | 이웃 토큰 (P12) | `SegMaskSam3X_<tag>` | ⬜ | 3 |
| 11 | 설정 매칭 (E6) | `SegMaskNew1_matched` | ⬜ | 4 |

REPORT.md의 결론 그대로: **행 3·4가 나오기 전에는 어떤 정확도 제안도 판정할 수 없습니다.**

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
