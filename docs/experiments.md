> **2026-09-04 상태 변경.** `demoSCSam3OneStageNew`는 이제 `demoSCSam3MVOpt`의 바이트 동일
> 동결 스냅샷입니다. 아래에서 "OneStageNew"가 *무패치 원본 코드*를 뜻하는 서술은
> **git 태그 `baseline-onestagenew`** 기준으로 읽으십시오. 배경: [README.md](README.md)

# 실험 — MVSeg J&F

> **2026-09-07 Phase 1.** 채점기·집계기가 저장소의 [`eval/`](../eval/)로 옮겨져 v2가 됐습니다(프레임별 점수, 짝지은 통계, ref/nonref, nb 구간, 면적 가중, ceiling, `--paper`).
> 정본 원시 점수는 `Data/MVSeg/jf_v2.json`(9개 방법 × 15개, 저장본과 396/396 항목 동일), 논문 표는 `docs/raw/paper_tables.md`,
> 판독과 엔드포인트 제안은 [phase1-measurement.md](phase1-measurement.md)입니다. 아래 표의 숫자는 그대로 유효합니다.

## 평가 방법

DAVIS J&F. `eval/eval_jf.py`가 원시 점수를, `eval/report_jf.py`가 집계를 담당합니다 (`Data/MVSeg/`의 원본 두 파일은 발표 당시 그대로 보존; git 이력은 `eval/` 첫 커밋 b54e7b6).

| 항목 | 내용 | 근거 |
|---|---|---|
| 시퀀스 단위 | `(데이터셋, 카메라)` 한 쌍 = DAVIS 시퀀스 하나 | `eval_jf.py:29-33` |
| J | 마스크 IoU. `union == 0 → 1.0` | `eval_jf.py:36-39` |
| F | DAVIS 경계 F. `BOUND_TH = 0.008`, `bound_pix = ceil(0.008 · ‖(h,w)‖)` | `eval_jf.py:33`, `:128-129` |
| 마스크 파일 없음 | **건너뛰지 않고 빈 예측으로 채점**하며 `missing_files`에 집계 | `eval_jf.py:158-161` |
| `inner` | 첫·마지막 프레임 제외 (`slice(1,-1)`) — DAVIS 원본과 동일 | `eval_jf.py:174` |

**입력 레이아웃.** GT는 `Mask/<카메라>/<프레임>.png` (그레이스케일, 픽셀값 = 객체 ID).
예측은 `<method>/<카메라>/<프레임>/<객체ID>.png` (이진). 객체 ID 폴더가 **프레임 아래**입니다.

**집계 두 가지.**
- `as-is` — 내보내지 않은 객체를 빈 예측으로 계산. **DAVIS 규약이고, 이것을 인용하십시오.**
- `exported only` — 출력 파일이 아예 없는 객체를 제외. 디렉터리 목록에 의존하므로 폴더를 지우면 값이 바뀝니다.

`eval_jf.py`는 `cv2`가 필요해 컨테이너 안에서 돌려야 합니다. `report_jf.py`는 표준 라이브러리만 씁니다.

## 결과 — 15개 데이터셋, `as-is`, 전체 프레임

`M`=SegMask, `M1`=SegMask1, `MN`=SegMaskNew, `MN1~3`=SegMaskNew1~3 (SAM 2)
`S3-1S`=OneStage, `S3-1SN`=OneStageNew, `S3-MVO`=MVOpt (SAM 3)

| 데이터셋 | M | M1 | MN | MN1 | MN2 | MN3 | S3‑1S | S3‑1SN | S3‑MVO |
|---|---|---|---|---|---|---|---|---|---|
| AlexaMeadeExhibit | 0.8910 | 0.8910 | 0.8913 | 0.8913 | 0.8916 | 0.8919 | 0.8618 | 0.8618 | 0.8618 |
| AlexaMeadeFacePaint | 0.8385 | 0.7843 | 0.8384 | 0.7845 | 0.7844 | 0.7844 | 0.7927 | 0.7928 | 0.7928 |
| Barn | 0.8567 | 0.8567 | 0.8569 | 0.8569 | 0.8569 | 0.8569 | 0.8858 | 0.8865 | 0.8865 |
| Blocks | 0.8248 | 0.7487 | 0.8243 | 0.7488 | 0.7487 | 0.7487 | 0.7484 | 0.7478 | 0.7478 |
| Breakfast | 0.7178 | 0.7178 | 0.7138 | 0.7138 | 0.7160 | 0.7167 | 0.7419 | 0.7421 | 0.7421 |
| Carpark | 0.9417 | 0.9417 | 0.9416 | 0.9416 | 0.9420 | 0.9420 | 0.9438 | 0.9434 | 0.9434 |
| CoffeeMartini | 0.8362 | 0.8362 | 0.8380 | 0.8380 | 0.8365 | 0.8352 | 0.8461 | 0.8440 | 0.8440 |
| Dog | 0.7662 | 0.9089 | 0.7654 | 0.9075 | 0.9086 | 0.9087 | 0.9241 | 0.9241 | 0.9241 |
| Fencing | 0.9273 | 0.9273 | 0.9321 | 0.9321 | 0.9337 | 0.9337 | 0.8862 | 0.9112 | 0.9112 |
| FlameSteak | 0.4627 | 0.7835 | 0.4627 | 0.7832 | 0.7833 | 0.7833 | 0.7736 | 0.7734 | 0.7734 |
| Frog | 0.9734 | 0.9734 | 0.9754 | 0.9754 | 0.9756 | 0.9755 | 0.9753 | 0.9743 | 0.9743 |
| MATF | 0.6108 | 0.6649 | 0.6649 | 0.6649 | 0.6650 | 0.6649 | 0.6840 | 0.6841 | 0.6841 |
| Painter | 0.8341 | 0.8618 | 0.8329 | 0.8713 | 0.8619 | 0.8618 | 0.8682 | 0.8681 | 0.8681 |
| PoznanStreet | 0.8543 | 0.8543 | 0.8646 | 0.8646 | 0.8672 | 0.8672 | 0.8662 | 0.8787 | 0.8787 |
| Welder | 0.8781 | 0.8781 | 0.8778 | 0.8778 | 0.8781 | 0.8784 | 0.8419 | 0.8437 | 0.8437 |
| **평균 J&F** | **0.8142** | **0.8419** | **0.8187** | **0.8434** | **0.8433** | **0.8433** | **0.8427** | **0.8451** | **0.8451** |
| **평균 J** | 0.7840 | 0.8110 | 0.7879 | 0.8125 | 0.8124 | 0.8124 | 0.8081 | 0.8104 | 0.8104 |
| **평균 F** | 0.8445 | 0.8728 | 0.8495 | 0.8744 | 0.8743 | 0.8742 | 0.8772 | 0.8797 | 0.8797 |

## 결과 — 12개 공통 부분집합

제외: AlexaMeadeExhibit, CoffeeMartini, FlameSteak (**당시** OneStageNew가 완주하지 못한 것들.
2026-09-04 메모리 수정 반영 이후에는 완주합니다).
데이터셋별 값은 위 표와 동일합니다 — 평균만 달라집니다.

| | M | M1 | MN | MN1 | MN2 | MN3 | S3‑1S | S3‑1SN | S3‑MVO |
|---|---|---|---|---|---|---|---|---|---|
| **평균 J&F** | **0.8353** | **0.8431** | **0.8407** | **0.8449** | **0.8449** | **0.8449** | **0.8465** | **0.8497** | **0.8497** |
| 평균 J | 0.8071 | 0.8153 | 0.8118 | 0.8171 | 0.8170 | 0.8171 | 0.8145 | 0.8175 | 0.8175 |
| 평균 F | 0.8635 | 0.8709 | 0.8695 | 0.8727 | 0.8727 | 0.8728 | 0.8786 | 0.8820 | 0.8820 |

**★ 0.8451과 0.8497은 둘 다 맞습니다.** 전자는 15개 전체, 후자는 12개 공통 부분집합입니다.
감사 보고서 헤더의 0.8497이 12개짜리인 것을 몰라 모순으로 오인하기 쉽습니다.

## 반드시 알아야 할 세 가지 ★

### (1) MVOpt와 OneStageNew는 같은 결과입니다 — 열이 둘이지만 결과는 하나

`jf_sam3_onestagenew_full.json`과 `jf_sam3_mvopt_all.json`은 45개 항목의 `result` 블록이 **전부 동일**합니다(차이 0건).
마스크 PNG도 12개 데이터셋 전부 파일 단위로 동일합니다.

논문에는 **한 열로만** 쓰십시오. 두 열로 보고하면 독립적인 두 방법처럼 보입니다.

### (2) 15개짜리 OneStageNew 열은 MVOpt 출력입니다

**원본** OneStageNew 코드는 무거운 3개를 완주한 적이 없고, 해당 마스크 폴더가 디스크에 존재하지 않습니다.
(2026-09-04 이후의 `demoSCSam3OneStageNew` 폴더 코드는 완주할 수 있습니다 — 아래 ★ 주의)
`jf_sam3_onestagenew_full.json`의 그 9개 항목은 MVOpt 출력을 OneStageNew 이름으로 채점한 것입니다.

정당화는 가능합니다 — MVOpt는 같은 알고리즘이고 12개에서 바이트 동일이 증명됐습니다.
하지만 **"OneStageNew가 15개를 돌았다"고 쓰면 사실이 아닙니다.**

권장: 정본으로 `jf_sam3_mvopt_all.json`을 쓰고, 방법 이름은 알고리즘 이름으로 하나만 쓰십시오.

**★ 주의 — 기본 스윕이 조용히 마스크를 만들 수 있습니다.**
무거운 3개에는 `SegMaskSam3OneStageNew` 폴더가 없으므로, `runMVSeg.py:294-295`의
"이미 존재함" 가드가 걸리지 않습니다. `--algo OneStageNew`로 스윕을 돌리면
**논문이 완주 불가라고 적은 바로 그 3개에 마스크가 생성**되어 `jf_sam3_onestagenew_full.json`을
사후적으로 "검증"해 버립니다. `runMVSegAll.sh`는 이 때문에 MVOpt를 향하도록 바꿨습니다.

### (3) `exported only` 집계는 OneStageNew에서 `nan`입니다

`report_jf.py:27-49`가 "없는 객체"를 **디스크 디렉터리 목록**으로 판정합니다.
OneStageNew의 3개 폴더가 없으므로 그 9개 카메라의 모든 객체가 없는 것으로 표시되고 `mean([]) → nan`이 됩니다.

`as-is`를 쓰면 이 문제가 없습니다. 저장된 점수만으로 계산되기 때문입니다.

## 재현 명령 (2026-09-07 이후)

```bash
cd /home/sjpark/Documents/SCSegmentation
M="SegMaskNew1 SegMaskSam3OneStage SegMaskSam3MVOpt"

# 15개 전체 (--common 이 기본값; 12개는 --datasets 로 명시)
python3 eval/report_jf.py --methods $M
python3 eval/report_jf.py --methods $M --datasets AlexaMeadeFacePaint Barn Blocks Breakfast Carpark Dog Fencing Frog MATF Painter PoznanStreet Welder

# 논문 표 전부 → docs/raw/paper_tables.md
python3 eval/report_jf.py --paper docs/raw/paper_tables.md
```

원시 점수를 다시 만들려면 (컨테이너, CPU만, 약 3분):

```bash
docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host/$PWD scsam3 \
  python eval/eval_jf.py --methods SegMask SegMask1 SegMaskNew SegMaskNew1 SegMaskNew2 SegMaskNew3 \
      SegMaskSam3OneStage SegMaskSam3OneStageNew SegMaskSam3MVOpt --out /host$PWD/Data/MVSeg/jf_v2.json
```

옛 명령(`Data/MVSeg/report_jf.py --raw jf_raw.json jf_sam3_onestage.json jf_sam3_mvopt_all.json ...`)도 그대로 동작하며 같은 숫자를 냅니다.

## 파일 목록

| 파일 | 상태 | 내용 |
|---|---|---|
| `jf_v2.json` | **정본 (2026-09-07)** | 9개 방법 × 15개, 프레임별 J·F·면적·메타 포함. 아래 4개 정본과 항목 단위로 동일 |
| `seed_census.json` | 진단 | 퇴화 시드 집계 원자료 (`eval/seed_census.py`) |
| `jf_xw.json` | **정본 (XW 계열)** | XW0·XW4·XW4all·XW1·XW2·XW6 원시 점수 (v2 형식) |
| `jf_p4.json` | **정본 (P4)** | XW1(15개)·XW1B·XW1C·XW1D·XW1E 원시 점수 — `jf_xw.json`과 함께 읽음(뒤 파일이 우선) |
| `jf_raw.json` | **정본** | SAM 2 6종, 15개 데이터셋 |
| `jf_sam3_onestage.json` | **정본** | OneStage 15개 |
| `jf_sam3_mvopt_all.json` | **정본** | MVOpt 15개 — SAM 3 최종 결과의 정본 |
| `jf_sam3_onestagenew_full.json` | 주의 | MVOpt 출력을 OneStageNew 이름으로 채점. 위 (2) 참조 |
| `jf_sam3_onestagenew.json` | 보조 | OneStageNew가 실제로 돌린 12개. `--common`이 12개를 뽑도록 하는 용도 |
| `jf_sam3_mvopt.json` | 중복 | 무거운 3개만. `_all`에 포함됨 |
| `jf_raw_sam3_onestage.STALE-prefix-bug.json` | **폐기** | 이름이 오해를 부릅니다. [investigations-closed.md](investigations-closed.md) 6번 참조 |
| `jf_recheck.json` | 보조 | SAM 2 재현 검증 (CoffeeMartini) |
| `jf_summary.json`, `missing.json`, `jf_report.txt` | 구자료 | 2026-09-02 SAM 2 작업 산출물 |

`Data/`는 git 제외 대상이라 이 파일들에는 git 이력이 없습니다. 2026-09-07부터는 각 마스크 폴더의 `MANIFEST.json`이 내용 해시(sha256)와 출처(git rev·인자·플래그, 기존 폴더는 로그에서 추정)를 담습니다 — `python3 eval/manifest.py verify <폴더>`. `diff -rq`로 폴더를 비교할 때는 `-x MANIFEST.json`을 붙이십시오.

## 출력 디렉토리 목록

`Data/MVSeg/<데이터셋>/<폴더>/<카메라>/<프레임>/<객체ID>.png`

| 폴더 | 데이터셋 | 성격 |
|---|---|---|
| `Mask` | 15 | **GT** — 그레이스케일, 픽셀값 = 객체 ID |
| `SegMask` `SegMask1` `SegMaskNew` `SegMaskNew1~3` | 15 | SAM 2 결과 (2025) |
| `SegMaskSam3OneStage` | 15 | SAM 3 OneStage |
| `SegMaskSam3MVOpt` | 15 | **SAM 3 최종 결과 정본** |
| `SegMaskSam3OneStageNew` | 12 | **원본 코드** 출력. 무거운 3개는 당시 완주 불가 |
| `SegMaskSam3XW0` `SegMaskSam3XW4` | 15 | **XW 계열**(2026-09-07, 위생 on, closure). XW0 = 패키지 내 W=0 대조군 = OneStage와 바이트 동일; XW4 = 0.8451 (= MVOpt) — [phase2-control.md](phase2-control.md) |
| `SegMaskSam3XW1` `XW2` `XW6` | 6 | W 곡선 (Barn·Blocks·Carpark·Fencing·PoznanStreet·Welder) |
| `SegMaskSam3XW4all` | 3 | closure == all 검증 (Welder·Dog·Blocks, XW4와 diff 0) |
| `SegMaskSam3XW1{B,C,D,E}` | 15 | P4 이웃 방향 절제(W=1): B 양쪽 t−1, C 이전 t·이후 t−1, D 이전 t−1, E 2회 계산 — [phase3-neighbourhood.md](phase3-neighbourhood.md) |
| `SegMaskSam3XW1{B,C,E}all`, `SegMaskSam3XW1E_rerun` | 2·1 | 원뿔 closure == all(Welder·Dog)·E 결정성 검증 부산물 (diff 0) |
| `SegMask*_SA3D` `SegMask*_SAM2` | 9 | 타 방법 비교군 |
| `SegMaskSam3OneStage_recheck` | 15 | 검증 부산물 — 메모리 수정 전후 대조용. **채점된 적 없음** |
| `SegMaskSam3ForSam2New` | 1 | ForSam2New 시험 실행 (CoffeeMartini). 채점된 적 없음 |
| `SegMaskSam2Recheck` | 1 | SAM 2 재현 검증 (CoffeeMartini). `jf_recheck.json` |

아래 둘은 **검증 부산물이라 삭제해도 결과에 영향이 없습니다** — 다만 재실행하려면 GPU 시간이 듭니다:
`SegMaskSam3OneStage_recheck`, `SegMaskSam3ForSam2New`.
(`SegMaskSam3OneStageNew_verify`와 `SegMaskSam3MVOpt_notrim`은 2026-09-04에 삭제했습니다.)

`SegMask (Copy)` 류가 보이면 실수로 만들어진 사본입니다. 어떤 `jf_*.json`도 참조하지 않습니다.
