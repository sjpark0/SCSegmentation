# Phase 3 / P4 — 이웃 메모리의 방향과 프레임: 사전 등록과 결과

**일자**: 2026-09-08 (사전 등록) · **계열**: XW (위생 on) · **대조군**: `SegMaskSam3XW0` (= OneStage, 바이트 동일)
**배경**: 현재 설계(이전 시점, 같은 프레임 t)는 과거 실험에서 경험적으로 고른 것이고, 그때 코드에는 C1/C2 버그가 있었습니다. Phase 2에서 창 폭은 무관(W=1 포화)하고 노이즈 바닥은 0임이 확인됐으므로, 남은 설계 변수인 **방향**과 **프레임 오프셋**을 깨끗한 절제로 다시 결정합니다.

> 이 문서의 §1은 GPU 실행 **전에** 작성·커밋됐습니다. §2 이후는 실행 후에 채웁니다.

---

## 1. 사전 등록

### 1.1 변형 (W=1, 15개 데이터셋 전부)

| 모드 | 이전 시점 v−k | 이후 시점 v+k | 출력 폴더 | 비고 |
|---|---|---|---|---|
| A | 프레임 t | — | `SegMaskSam3XW1` | 현재 설계 (6개 있음, 9개 추가 실행) |
| B | 프레임 t−1 | 프레임 t−1 | `SegMaskSam3XW1B` | 대칭, 한 프레임 오래됨 |
| C | 프레임 t | 프레임 t−1 | `SegMaskSam3XW1C` | REPORT P4 제안 |
| D | 프레임 t−1 | — | `SegMaskSam3XW1D` | A와 B의 차이를 방향/신선도로 가르는 대조군 |
| E | 2회 계산: 1차 = B, 2차 = 전부 프레임 t | | `SegMaskSam3XW1E` | 1차 출력은 버리고 2차 출력을 메모리·마스크로 |

고정 규칙: 이웃 토큰의 시간 위치 부호는 모든 모드·양쪽·모든 지연에서 `maskmem_tpos_enc[|offset|−1]` (A와 동일, 한 번에 한 요인만 바꿈). 첫 추적 프레임(t = start+1)의 t−1 조회는 이웃의 시드 cond 항목을 그대로 씀(zero-mask 시드 이웃은 no-object 메모리를 넣음; 게이팅은 P12로 미룸). 경계는 클리핑(wrap 없음). closure는 정확한 의존 원뿔: A/D는 0..max(scored), B/C는 0..min(N−1, max(scored)+W·20), E는 0..min(N−1, max(scored)+2W·20).

### 1.2 1차 기준 (E1, 모드별)

비기준 카메라, 카메라 수준 짝지은 ΔJ&F(모드 − XW0), **이웃을 실제로 받는 구간**에서, 클러스터 부트스트랩 95% CI(데이터셋 클러스터, `random.Random(0)`, 10,000회). **합격 = CI 하한 > 0.**

- A·D (`--nb-mode lower`): nb = min(v, 1) ≥ 1인 비기준 카메라 28대 → nb 표의 `nb>=1` 행.
- B·C·E (`--nb-mode both`): nb = min(v,1)+min(N−1−v,1) ≥ 1인 비기준 카메라 30대 전부 → `--split nonref`의 카메라 수준 paired 행(= nb 표의 누적 `nb>=1` 행).

```bash
R="--raw Data/MVSeg/jf_xw.json Data/MVSeg/jf_p4.json"
for M in "" B C D E; do NB=lower; [ -n "$M" ] && [ "$M" != D ] && NB=both
  python3 eval/report_jf.py $R --methods SegMaskSam3XW0 SegMaskSam3XW1$M --split nonref --window 1 \
      --nb-mode $NB --bin-by-nb --paired SegMaskSam3XW0 SegMaskSam3XW1$M > docs/raw/phase3/e1_nonref_XW1$M.txt; done
```

### 1.3 2차 기준

- **S1** Fencing v0(기준 카메라, 인덱스 0): 객체별 프레임 1..20 J 평균의 평균. **XW0 기준값 = 0.7936** (jf_xw.json에서 실행 전 기록). 기대(REPORT P4): B/C/E 중 하나 이상이 **≥ 0.85**. A·D는 정의상 못 움직임.
- **S2** 기준 카메라 무손실: `--split ref` 카메라 수준 paired(15대), 클러스터 CI가 전부 0 아래에 있으면 안 됨(CI 상한 ≥ 0).
- **S3** 비교 가능성: 다섯 Δ 모두를 고정 분할(`--nb-mode lower --window 1`, nb≥1, 28대)에서도 나란히 보고. 전체 카메라 nb 표를 두 정의로.
- **S4** 면적 가중, inner 변형.

### 1.4 감시값 (Phase 2의 nb=0 감시값을 대체)

- **Z1** 인덱스 0 카메라 폴더: XW1D == XW1 == XW0 바이트 동일(채점 인덱스 0 카메라가 있는 9개 데이터셋). 차이가 있으면 누수·비결정성.
- **Z2** closure(원뿔) == all: B·C·E × Welder·Dog, diff 0 (Dog E는 원뿔 = N이라 결정성 검사).
- **Z3** E 결정성: Blocks XW1E 재실행 diff 0.
- **Z4** A vs XW0의 nb=0 카메라 Δ = 정확히 0 (새로 도는 9개 데이터셋).

감시값 하나라도 실패한 모드는 자격 없음이며, 표를 읽기 전에 원인을 조사합니다.

### 1.5 결정 규칙

자격 = 감시값 전부 통과 + S2 통과. 자격 있는 모드 중 **1차 기준 CI가 0을 제외하면서 비기준 카메라 평균 ΔJ&F가 가장 큰 모드**가 승자. 0.001 이내 동률이거나 아무 모드도 CI 조건을 못 넘으면 **다섯 행 전부 보고하고 현재 설계 A 유지**. 승자는 W=4로 한 번 더 돌려 확인(기대: 15개 평균 변화 < 0.002, 보고만 하고 재선택에 쓰지 않음). 비교 5개·대조군 1개·규칙 1개, 다중비교 보정 없음(명시).

**판정에 쓰지 않는 것**: 6개 부분집합, 객체 단위 일화, `--split all` 평균, 실행 시간.

### 1.6 실행 계획

가드(legacy Blocks·Fencing, XW1·XW4 Blocks → 기존 폴더와 diff 0) → 스모크(Blocks, 4모드) → Z2 → W=1 스윕(A 9개 33분, D 45분, B 71분, C 71분, E 144분) → 채점 → 승자 W=4. 가드 대상 파일이 바뀌면 채점 전에 가드를 다시 돌립니다.
