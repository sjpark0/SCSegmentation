# P5 사전 등록 — 퇴화 시드 복구 (RepairSeeds)

> **이 문서는 GPU 실행 전에 커밋됩니다.** 판정 규칙·감시값·기대값을 여기 적힌 대로 읽고, 실행 뒤에
> 바꾸지 않습니다. 못 넘으면 측정값과 함께 "효과 없음"으로 닫습니다. P4(2847fd2)·P12(011fa86)와 같은
> 규율입니다. 준비 작업과 진단 근거는 [phase5-seed-repair-prep.md](phase5-seed-repair-prep.md).

## 0. 한 줄 요약

시점 전파가 만든 첫 프레임 마스크가 **비정상적으로 작은 (시점, 객체) 쌍**을 정답 없이 찾아, 옆 시점의
마스크를 두 번째 조건 프레임으로 넣고 **새 시점 세션에서 다시 전파**해 그 쌍만 바꿔 넣습니다.
그다음은 평소와 같이 카메라별 시간 추적입니다.

- 처치: `SegMaskSam3XW1GPS4MRp` (채택 구성 + `--repair-seeds`)
- 대조: `SegMaskSam3XW1GPS4M` (이미 있음)
- 프로토콜: MUVOD basic, 17개 장면, c_ini 시딩 ([muvod-protocol.md](muvod-protocol.md))

## 1. 규칙 (구현이 이것과 다르면 구현이 틀린 것)

### 1.1 감지 — 정답을 보지 않습니다

(시점 v, 객체 o)를 **퇴화**로 봅니다, 다음 넷이 모두 참일 때:

1. v가 기준 시점(c_ini)이 아닙니다. 기준 시점의 시드는 정답에서 온 것이라 손대지 않습니다.
2. o의 c_ini 프롬프트 면적 ≥ 64 px. 프롬프트 자체가 없거나 몇 픽셀이면 전파 실패가 아니라
   프롬프트 문제이고, 이 단계의 몫이 아닙니다.
3. v에서 o의 시드 면적 < **T(v) = max(64, 0.05 × v의 0이 아닌 시드 면적들의 중앙값)**.
   0이 아닌 시드가 3개 미만이면 T(v) = 64.
4. **donor가 있습니다** (1.2).

3·4가 [prep §3](phase5-seed-repair-prep.md)의 정정입니다. REPORT 원안(3만)은 오탐이 90%였고,
가장 큰 오탐 부류(c_ini 첫 프레임에 없는 객체 73개)를 2가 제거합니다.

**이 감지기가 원리상 못 잡는 것.** 면적은 크지만 자리가 틀린 시드(prep의 `misaligned`, 예: MATF
S1_CAM_1 obj 26, Welder camera_0004 obj 12). 겹침은 정답 없이 계산할 수 없습니다. **이 실험의
대상이 아니며**, 결과 표에 "감지 범위 밖"으로 따로 적습니다.

### 1.2 donor 선택

v에서 ±4 시점 안(경계는 잘라냄)에서, 같은 객체의 시드 면적이 **max(64, 0.25 × c_ini 프롬프트 면적)
이상**인 시점 중 **면적이 가장 큰 시점**. 후보가 없으면 ±8로 넓히고, 그래도 없으면 **복구 불가**로
기록하고 건드리지 않습니다. 같은 객체로 이미 퇴화 판정된 시점과 지난 라운드에 쓴 donor는 후보에서
뺍니다. 동점이면 가까운 시점, 다시 동점이면 낮은 인덱스.

### 1.3 복구

한 장면의 퇴화 쌍 전부를 **라운드마다 한 번의 새 시점 세션**으로 처리합니다.

1. 1차 전파 결과(`masks_spatial`)를 손에 든 채 **새 세션**을 엽니다 (같은 N장 첫 프레임).
2. 퇴화 쌍이 있는 객체마다: c_ini에 정답 마스크를, 그 객체의 각 donor 시점에 donor 시드 마스크를
   프롬프트로 넣습니다. 둘 다 조건 프레임이 됩니다.
3. 양방향으로 전파합니다. **시작 인덱스는 명시하지 않습니다**
   ([investigations-closed 4번](investigations-closed.md), `SPATIAL_START_IMPLICIT=1`).
4. **퇴화로 판정된 (v, o)만** 새 마스크로 바꿔 넣습니다. 다른 쌍은 1차 결과 그대로입니다.
5. 세션을 닫습니다. 바꿔 넣은 뒤 다시 감지해, 여전히 퇴화이고 아직 안 쓴 donor가 있으면 한 번 더.
   **최대 2라운드.**
6. 그다음 평소대로 시점 모델 폐기 → 시간 추적.

**왜 기존 세션이 아니라 새 세션인가.** 1차 전파가 끝난 세션은 모든 시점에 출력 표식이 남아 암묵
시작점이 0으로 바뀌고, 이미 추적된 시점에 넣는 프롬프트가 조건 프레임이 되는지가 설정에 따릅니다.
새 세션에서는 두 프롬프트가 모두 초기 조건 프레임이고 시작점이 min(프롬프트 시점)으로 정의됩니다.
비용은 1차 전파 한 번과 같고(특징 추출이 지배적), 퇴화 쌍이 있는 장면에서만 듭니다.

**순서 제약.** 복구는 `PropagateAcrossViews()` 뒤, `RetireSpatialPredictor()` 앞. 테스트
`test_repair_must_run_before_the_spatial_model_is_retired`가 고정합니다.

## 2. 엔드포인트

| 이름 | 읽기 | 대조 |
|---|---|---|
| **E-P5-0 (1차)** | MUVOD basic J&F³, 17개 장면, `Rp − M`, 장면 짝지음, 클러스터 CI | `SegMaskSam3XW1GPS4M` |
| **E-P5-1 (메커니즘)** | Welder camera_0004의 obj 8·14 J_all | 각각 0.000 |
| 보조 | complete 표, 장면·카메라·(시점,객체) 단위 Δ 표 전부, 라운드별 복구 수 | |
| 감지 범위 밖 | Welder camera_0004 obj 12, MATF S1_CAM_1 obj 26의 J_all | 보고만, 기준 아님 |

## 3. 판정 규칙 — 실행 전에 고정

**채택**(헤드라인을 `Rp`로 바꿈)은 아래 **셋이 모두** 참일 때:

- **(a) E-P5-1**: obj 8 J_all ≥ 0.8 **그리고** obj 14 J_all ≥ 0.8.
- **(b) E-P5-0**: 17개 장면 짝지은 평균 Δ ≥ 0 **그리고** 어느 장면도 Δ < −0.005가 아님
  (basic 기준). 이것이 "오탐 47개에 잘못 주입"의 방어선입니다.
- **(c) S1**: 퇴화 쌍이 0인 장면은 `M`과 **바이트 동일** (`diff -rq -x MANIFEST.json`).

(a)만 실패 → "복구는 다른 곳에서 작동하지만 표적은 못 고쳤다". Welder 서술은 쓰지 않습니다.
(b) 실패 → 채택하지 않고 (시점,객체) 표를 전부 싣습니다.
(c) 실패 → 구현 결함. 결과를 해석하지 않고 원인부터 찾습니다.

## 4. 감시값

| | 기대 | 어긋나면 |
|---|---|---|
| S1 | 플래그 0인 장면 바이트 동일. 대리 계산으로는 **Carpark·CoffeeMartini·FlameSteak·Frog·MartialArts** 5개. 실제 목록은 매니페스트의 플래그 수로 확정 | 구현 결함 (복구가 새어 나감) |
| S2 | 15개 진짜 퇴화 중 감지기 범위 안의 **14개가 전부 플래그됨** (MATF obj 26 제외). 매니페스트로 확인 | 감지기가 등록 규칙과 다름 |
| S3 | 채점 카메라 기준 플래그 수 ≈ 49 (대리 계산; 첫 기록 프레임 PNG로 셌으므로 정확히 같지 않아도 됨). 실제는 추적하는 모든 시점(closure 집합)에서 세므로 더 많음 | — (기록만) |
| S4 | 기준 시점의 시드는 어느 장면에서도 바뀌지 않음 | 구현 결함 |

## 5. 기대값 — 모형값이지 주장이 아닙니다

집계 도구의 복구 모형(걸린 객체에 `max(저장값, 중앙값)`)이 준 값입니다. 실제 재전파가 이 값을
낼 이유는 없습니다.

| | 저장값 | 모형 복구 | Δ |
|---|---|---|---|
| 17개 basic (`M`) | 0.8544 | 0.8655 (임계 0.10) / 0.8668 (0.20) | +0.0111 / +0.0125 |
| Welder | 0.8418 | 0.8815 / 0.8905 | +0.040 / +0.049 |

감지기가 misaligned를 못 잡으므로 **현실적 상한은 0.10 쪽**입니다. D2=0.20은 진단 범위를
정하는 값이고 이 실험의 감지기에는 영향이 없습니다. 둘 다 표에 싣습니다.

**대리 플래그 표** (채점 카메라, 1차 결과 PNG 기준, 구현이 대략 재현해야 함):

| 장면 | 플래그 | 진짜 | | 장면 | 플래그 | 진짜 |
|---|---|---|---|---|---|---|
| AlexaMeadeExhibit | 8 | 3 | | Fencing | 4 | 0 |
| AlexaMeadeFacePaint | 1 | 0 | | FlameSteak | **0** | 0 |
| Barn | 2 | 0 | | Frog | **0** | 0 |
| Blocks | 6 | 3 | | MATF | 6 | 1 |
| Breakfast | 10 | 3 | | MartialArts | **0** | 0 |
| CBABasketball | 3 | 1 | | Painter | 3 | 1 |
| Carpark | **0** | 0 | | PoznanStreet | 1 | 0 |
| CoffeeMartini | **0** | 0 | | Welder | 4 | 2 |
| Dog | 1 | 0 | | | | |

Barn·FacePaint·Dog·Fencing·PoznanStreet는 **플래그는 있으나 진짜 퇴화가 없는** 장면입니다.
(b)의 "−0.005" 방어선이 이들을 지켜보는 자리입니다.

## 6. 실행

```bash
# 사전 등록 커밋 뒤에만
cd SCSam3
XVIEW=1 XGATE=1 XPTR=1 XSHIFT=4 TRACK=closure XREF=muvod XREPAIR=1 LOGTAG=p5-rp \
  ./runMVOptThree.sh AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast CBABasketball \
    Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF MartialArts Painter PoznanStreet Welder
```

채점과 판정:

```bash
docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \
  python eval/eval_jf.py --methods SegMaskSam3XW1GPS4M SegMaskSam3XW1GPS4MRp \
    --out /host$PWD/Data/MVSeg/jf_p5.json
python3 eval/report_jf.py --raw jf_p5.json --muvod                                  # E-P5-0
python3 eval/report_jf.py --raw jf_p5.json --aggregation sequence --ref-rule muvod \
  --objects basic --paired SegMaskSam3XW1GPS4M SegMaskSam3XW1GPS4MRp                # (b)
for d in Carpark CoffeeMartini FlameSteak Frog MartialArts; do                       # (c)
  diff -rq -x MANIFEST.json Data/MVSeg/$d/SegMaskSam3XW1GPS4M Data/MVSeg/$d/SegMaskSam3XW1GPS4MRp && echo "$d identical"
done
```

E-P5-1과 (시점,객체) 표는 `jf_p5.json`의 `per_frame`과 각 `MANIFEST.json`의 `seed_repair` 블록에서
읽습니다. GPU 한 스윕(약 1시간 + 복구 세션).

## 7. 이 실험이 답하지 않는 것

- misaligned 시드(감지 범위 밖). 정답 없는 대리 신호(이웃과의 면적 비)는 별도 등록이 필요합니다.
- 복구가 교차시점 메모리(XW1GPS4)와 독립인지 — `XW0MRp`는 돌리지 않습니다. 채택되면 후속.
- 다중 카메라 시딩(정답을 3배 쓰는 것) — MUVOD 규약 밖이라 별도 표로만 가능합니다.
