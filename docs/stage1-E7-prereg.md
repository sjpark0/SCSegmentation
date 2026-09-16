# S1-E7 사전 등록 — 시드를 2단계에 실제로 통과시켜 실측: 정답 시드 오라클 · E6 투표 시드 · E6 오라클 선택 시드

> 2026-09-16. 층: **확정(측정)**. 계기: TCSVT 초고 외부 리뷰 지적 4 — "Table VI(+1.11, 최종 출력 GT 대체)와
> §VII-C(+2.967 / −0.061, 시드만 바꾸고 2단계 미실행 환산값)는 서로 다른 오라클이며, 환산값은 실제 최종 성능도
> 보장된 상한도 아니다. 가장 큰 분석력 향상은 **모든 카메라의 초기 시드를 GT로 바꾸고 같은 2단계를 다시 실행**하는 것."
> 이 문서는 GPU 실행 **전에** 커밋됩니다. §5 이후는 실행 뒤에 채웁니다.

## 1. 질문 셋

| # | 질문 | 답이 바꾸는 것 |
|---|---|---|
| Q1 | 1단계(시드)가 완벽하면 같은 2단계는 몇 점을 내는가 | 원고 §VI "seed repair 상한"을 환산이 아닌 실측으로 교체; "초기화 오류 대 이후 추적 오류" 분리 |
| Q2 | E6 투표 시드를 실제로 전파하면 −0.061(환산)이 몇 점이 되는가 | §VII-C의 환산값을 실측으로 교체 |
| Q3 | E6 오라클 선택 시드를 실제로 전파하면 +2.967(환산)이 몇 점이 되는가 | 〃; 환산식이 낙관 상한이라는 E6 등록의 주장 검증 |

## 2. 시드 집합 (전부 `Data/MVSeg/<scene>/MVSeed_<tag>/`, MVSeed/README.md R2)

| 태그 | 내용 | 만드는 것 |
|---|---|---|
| `MVSeed_gt` | **주석 카메라 3대**(c_ini 포함)의 프레임 0 시드 = 그 카메라의 **정답 마스크**. 객체 = basic 집합(c_ini 프레임 0의 id). 그 카메라 정답 프레임 0에 없는 basic 객체는 **빈 PNG**(면적 0 = "없음"을 명시). 비주석 카메라는 G0 유지 | `MVSeed/build_seed_sets.py --set gt` |
| `MVSeed_e6_vote` | E6이 쓴 그대로(등록 규칙 선택, 28쌍 교체). **재생성하지 않음** | 기존 |
| `MVSeed_e6_oracle` | G0 폴더 복사 후, 정답 있는 비기준 664행(동결 `F0.json`∪`N0.json`)마다 E6 일곱 스트림(`seed_vote.STREAMS`, 같은 `mask_path`) 중 **정답 IoU 최대** 후보로 교체. 동점·전부 0이면 G0 유지. c_ini 미변경 | `MVSeed/build_seed_sets.py --set oracle` |

각 폴더에 `SEED_MANIFEST.json`(`written_views` = 쓴 카메라, `dataset`, `start_frame`)을 넣어 러너의 coverage 검사를 받습니다.

## 3. 실행 (17장면, `SCSam3/runMVOptThree.sh`, 기존 플래그만; 코드 변경 없음)

| # | 구성 | 명령 (env) | 출력 폴더 | 기저 |
|---|---|---|---|---|
| R1 | 대조군 + GT 시드 | `XVIEW=0 XREF=muvod SEEDS=MVSeed_gt F0SEED=1` | `SegMaskSam3XW0MSdgtFs` | `SegMaskSam3XW0MFs` (89.549) |
| R2 | 헤드라인 + GT 시드 | `XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod SEEDS=MVSeed_gt F0SEED=1 CROP=10000 CROPGATE=v2` | `SegMaskSam3XW1CGPS4MSdgtFsCs10kG2` | `SegMaskSam3XW1CGPS4MFsCs10kG2` (91.015) |
| R3 | 대조군 + E6 투표 시드 | `XVIEW=0 XREF=muvod SEEDS=MVSeed_e6_vote F0SEED=1` | `SegMaskSam3XW0MSde6_voteFs` | `SegMaskSam3XW0MFs` |
| R4 | 대조군 + E6 오라클 시드 | `XVIEW=0 XREF=muvod SEEDS=MVSeed_e6_oracle F0SEED=1` | `SegMaskSam3XW0MSde6_oracleFs` | `SegMaskSam3XW0MFs` |

채점: `eval/eval_jf.py --methods <4개> --out jf_e7.json`(`--root` 상대, 함정 13) → `eval/report_jf.py --raw jf_e7.json jf_r4.json --muvod --paired <기저> <처치>` (장면 클러스터 부트스트랩, seed 0, 10 000회 — 기존 도구).

**모드 C의 단서**: R2에서 채점 카메라의 이웃(closure 안의 비주석 카메라)은 G0 시드로 시작하므로, R2는 "주석 카메라의 시드만 완벽"한 조건입니다. 이것은 결과 해석에 적습니다.

## 4. 예측 (실행 전)

| # | 예측 | 근거 |
|---|---|---|
| P1 | R1 basic J&F³ = 기저 +3.0 ~ +5.0 | 비기준 정상 쌍(정답 시드 아님)이 기준 카메라(정답 시드) 수준 0.95로 오르면 쌍의 2/3에 ≈+0.06 → ≈+4점; 오라클 "비기준 완벽" +7.40이 절대 상한 |
| P2 | R2 = 기저 +2.0 ~ +4.0 (R1의 Δ보다 작음) | 크롭이 작은 객체의 손실 일부를 이미 회수; 이웃(비주석)은 G0 그대로 |
| P3 | R1·R2 모두 **프레임 0 J = 1.0 정확히**(정답 있는 쌍 전부) | `--frame0-seed`가 시드 PNG를 그대로 씀 — 실패하면 배관 오류 |
| P4 | R3 = 기저 −0.3 ~ +0.1, 그리고 R3 − 기저 ≤ E6 환산값(−0.061)과 같은 부호 | E6: F0 회복 10, N0 파괴 11; 2단계 회귀(0.964)가 시드 차이를 줄임 |
| P5 | R4 = 기저 +1.0 ~ +2.5, **그리고 +2.967보다 작음** | E6 등록의 "환산은 낙관 상한" 주장. 틀리면(실측 ≥ 환산) 환산식 정의를 다시 봐야 함 |
| P6 | R1 − 기저 > R4 − 기저 > R3 − 기저 | 정답 시드 ⊇ 오라클 선택(후보 집합 안의 최선) ⊇ 규칙 선택 |

## 5. 읽기 규칙 (실행 전 고정)

1. **R1·R2·R4는 오라클이지 방법이 아니다** — 채택 판정 없음. 값은 원고 Table VI를 **오라클 표**(모집단 · 실패 기준 · 무엇을 GT로 바꾸는지 · 2단계 재실행 여부 · 이득)로 재구성하는 데 쓴다.
2. **R3(투표)는 E6 관문 실패가 이미 확정**이므로, 실측이 +0.3 이상 나와도 채택하지 않는다(사후 채택 금지). 보고만.
3. P3 실패(프레임 0 J ≠ 1.0) 시 결과를 읽기 전에 배관을 조사한다.
4. P5 실패(R4 ≥ +2.967) 시 E6 환산식의 "낙관 상한" 문구를 원고와 E6 문서에서 철회한다.
5. 장면별 표·쌍별 Δ 분포(F0/N0 별)를 함께 보고한다. 다중비교 보정 없음(4개 비교, 명시).

## 6. 감시값

- Z1 R3·R4의 프레임 0 PNG는 시드 폴더 PNG와 **바이트 동일**(`--frame0-seed`).
- Z2 각 실행 `MANIFEST.json`의 `provenance.seeds_from.png_digest`가 시드 폴더 digest와 일치.
- Z3 R1 대 기저: 비주석 카메라의 `view_areas.seed`(있는 경우) 불변.

## 7. 이력

| 일자 | 내용 |
|---|---|
| 2026-09-16 | 등록 (GPU 실행 전 커밋) |
