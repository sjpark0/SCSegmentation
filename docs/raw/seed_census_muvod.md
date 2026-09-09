# Seed census — cross-view seed 품질 진단 (P5, 비용 0)

`eval/seed_census.py`가 생성. 마스크 PNG와 `jf_*.json`만 읽고 GPU를 쓰지 않습니다. REPORT.md P5의 "먼저 비용 0 진단" 항목이며 ROADMAP Phase 1의 퇴화 시드 집계입니다.

**seed란.** `runMVSeg.py`의 `TrackForward()`는 cross-view pass가 만든 마스크를 각 카메라의 `start_frame`에 mask prompt로 넣고(없으면 zero mask) 앞으로만 추적합니다. 따라서 각 scored 카메라의 **첫 기록 프레임** `<method>/<cam>/<start_frame>/<obj>.png`이 temporal tracker가 21프레임 내내 조건화된 seed 그 자체입니다 (reference 카메라에서는 GT prompt를 재예측한 것). 객체 집합은 `eval_jf.py`가 채점하는 것과 같이 그 카메라 GT의 어느 프레임에든 등장하는 id 전부입니다.

**분류 규칙** (우선순위 순):

| 클래스 | 조건 |
|---|---|
| no_folder | 그 데이터셋에 `<method>/` 폴더 자체가 없음. 분류할 것이 없으므로 그 (데이터셋, 방법)은 경고와 함께 건너뛰고 평균에서 제외. 퇴화가 아님 |
| unreachable | 객체 id가 reference 카메라(max-id 규칙, `eval_jf.dataset_meta`)의 seed GT에 없음 → zero prompt, tracker가 한 픽셀도 예측하지 않음. GT에 객체가 있는 프레임은 J=F=0, 없는 프레임은 1.0(DAVIS 관례, `eval_jf`: union=0 → 1.0)이므로 저장된 J=F는 GT가 빈 프레임의 비율(예: Breakfast v9 obj 17은 21프레임 중 7프레임 → 0.3333). 구조적이며 어떤 방법도 못 고침 (P8의 몫) |
| gt_empty | 도달 가능하지만 이 카메라의 seed 프레임 GT에 0 px (나중에 등장). 퇴화가 아님, 별도 집계 |
| missing | seed 파일 없음 |
| empty | seed 파일은 있으나 0 px |
| tiny | 0 < pred_area < max(64, 0.05 × gt_area). P5 면적 검사의 **GT-oracle 변형** — REPORT.md P5의 런타임 검사는 GT 없이 0.05 × (뷰 전체 예측 면적의 중앙값)과 비교합니다 |
| misaligned | tiny가 아니고 gt_area > 0인데 IoU < 0.10 |
| ok | 나머지 |

**퇴화(degenerate)** = missing + empty + tiny + misaligned. SAM 2 러너는 0 px 파일을 쓰고 SAM 3 러너는 파일을 안 쓰므로 `SegMaskNew1`의 empty와 SAM 3 폴더의 missing은 같은 실패 양상(0-px 파일 vs 파일 없음)이지만 같은 객체 집합은 아닙니다 — empty+missing 기준으로 `SegMaskSam3XW1GPS4M` 7개 vs `SegMaskSam3XW0M` 7개 중 7개가 겹칩니다(`SegMaskSam3XW1GPS4M`에만: 없음; `SegMaskSam3XW0M`에만: 없음). **저장된 J&F** = `(cam, obj)` 쌍의 (J+F)/2 평균 = `report_jf.py` as-is/all 표(`docs/experiments.md`). **복구 목표 J&F** = 퇴화 객체에 max(저장된 (J+F)/2, 그 데이터셋·방법의 ok 객체 중앙값 J와 중앙값 F의 평균)을 준 값 — 저장값이 이미 중앙값보다 높은 객체는 낮추지 않습니다. **상한 J&F** = 퇴화 객체를 J=F=1로 둔 값. no_folder·unreachable·gt_empty는 두 계산에서 건드리지 않습니다.

점수 출처: `SegMaskSam3XW1GPS4M` ← `jf_muvod.json`, `SegMaskSam3XW0M` ← `jf_muvod.json`. 데이터셋별 reference 카메라: AlexaMeadeExhibit `camera_0001`, AlexaMeadeFacePaint `camera_0007`, Barn `v7`, Blocks `cam4`, Breakfast `v7`, CBABasketball `v20`, Carpark `v4`, CoffeeMartini `cam16`, Dog `camera_0002`, Fencing `v4`, FlameSteak `cam16`, Frog `v7`, MATF `S1_CAM_4`, MartialArts `v9`, Painter `v6`, PoznanStreet `v4`, Welder `camera_0001`.

## SegMaskSam3XW1GPS4M

| 데이터셋 | 객체 수 | unreachable | missing | empty | tiny | misaligned | 저장된 J&F | 복구 목표 J&F | 상한 J&F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AlexaMeadeExhibit | 93 | 0 | 3 | 0 | 0 | 0 | 0.8624 | 0.8928 | 0.8947 |
| AlexaMeadeFacePaint | 33 | 3 | 0 | 0 | 0 | 0 | 0.8451 | 0.8451 | 0.8451 |
| Barn | 90 | 0 | 0 | 0 | 0 | 0 | 0.8870 | 0.8870 | 0.8870 |
| Blocks | 52 | 6 | 0 | 0 | 2 | 1 | 0.7831 | 0.8268 | 0.8278 |
| Breakfast | 85 | 7 | 1 | 0 | 2 | 0 | 0.8026 | 0.8270 | 0.8296 |
| CBABasketball | 58 | 1 | 1 | 0 | 0 | 0 | 0.8757 | 0.8919 | 0.8929 |
| Carpark | 66 | 0 | 0 | 0 | 0 | 0 | 0.9421 | 0.9421 | 0.9421 |
| CoffeeMartini | 143 | 16 | 0 | 0 | 0 | 0 | 0.8343 | 0.8343 | 0.8343 |
| Dog | 23 | 3 | 0 | 0 | 0 | 0 | 0.7716 | 0.7716 | 0.7716 |
| Fencing | 24 | 0 | 0 | 0 | 0 | 0 | 0.9033 | 0.9033 | 0.9033 |
| FlameSteak | 134 | 20 | 0 | 0 | 0 | 0 | 0.7822 | 0.7822 | 0.7822 |
| Frog | 15 | 0 | 0 | 0 | 0 | 0 | 0.9773 | 0.9773 | 0.9773 |
| MATF | 77 | 11 | 1 | 0 | 0 | 1 | 0.7284 | 0.7505 | 0.7523 |
| MartialArts | 50 | 0 | 0 | 0 | 0 | 0 | 0.9388 | 0.9388 | 0.9388 |
| Painter | 78 | 6 | 1 | 0 | 0 | 0 | 0.8512 | 0.8635 | 0.8640 |
| PoznanStreet | 71 | 0 | 0 | 0 | 0 | 0 | 0.8977 | 0.8977 | 0.8977 |
| Welder | 47 | 0 | 0 | 0 | 2 | 0 | 0.8418 | 0.8815 | 0.8837 |
| **평균 (17개)** | 1139 | 73 | 7 | 0 | 6 | 2 | **0.8544** | **0.8655** | **0.8661** |

gt_empty 4개, ok 1047개 (합계 1139개 중). 평균 행의 J&F는 데이터셋별 값의 평균이고 객체 수 열은 합계입니다.

## SegMaskSam3XW0M

| 데이터셋 | 객체 수 | unreachable | missing | empty | tiny | misaligned | 저장된 J&F | 복구 목표 J&F | 상한 J&F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AlexaMeadeExhibit | 93 | 0 | 3 | 0 | 0 | 0 | 0.8618 | 0.8922 | 0.8941 |
| AlexaMeadeFacePaint | 33 | 3 | 0 | 0 | 0 | 0 | 0.8457 | 0.8457 | 0.8457 |
| Barn | 90 | 0 | 0 | 0 | 0 | 0 | 0.8874 | 0.8874 | 0.8874 |
| Blocks | 52 | 6 | 0 | 0 | 2 | 1 | 0.7767 | 0.8204 | 0.8214 |
| Breakfast | 85 | 7 | 1 | 0 | 2 | 0 | 0.8012 | 0.8266 | 0.8291 |
| CBABasketball | 58 | 1 | 1 | 0 | 0 | 0 | 0.8751 | 0.8913 | 0.8923 |
| Carpark | 66 | 0 | 0 | 0 | 0 | 0 | 0.9424 | 0.9424 | 0.9424 |
| CoffeeMartini | 143 | 16 | 0 | 0 | 0 | 0 | 0.8343 | 0.8343 | 0.8343 |
| Dog | 23 | 3 | 0 | 0 | 0 | 0 | 0.7716 | 0.7716 | 0.7716 |
| Fencing | 24 | 0 | 0 | 0 | 0 | 0 | 0.8768 | 0.8768 | 0.8768 |
| FlameSteak | 134 | 20 | 0 | 0 | 0 | 0 | 0.7819 | 0.7819 | 0.7819 |
| Frog | 15 | 0 | 0 | 0 | 0 | 0 | 0.9758 | 0.9758 | 0.9758 |
| MATF | 77 | 11 | 1 | 0 | 0 | 1 | 0.7284 | 0.7506 | 0.7523 |
| MartialArts | 50 | 0 | 0 | 0 | 0 | 0 | 0.9384 | 0.9384 | 0.9384 |
| Painter | 78 | 6 | 1 | 0 | 0 | 0 | 0.8494 | 0.8617 | 0.8622 |
| PoznanStreet | 71 | 0 | 0 | 0 | 0 | 0 | 0.8869 | 0.8869 | 0.8869 |
| Welder | 47 | 0 | 0 | 0 | 2 | 0 | 0.8419 | 0.8818 | 0.8840 |
| **평균 (17개)** | 1139 | 73 | 7 | 0 | 6 | 2 | **0.8515** | **0.8627** | **0.8633** |

gt_empty 4개, ok 1047개 (합계 1139개 중). 평균 행의 J&F는 데이터셋별 값의 평균이고 객체 수 열은 합계입니다.

## 17개 데이터셋 평균 — 방법 비교

| 방법 | 퇴화 객체 | 그중 ref 프롬프트 <64 px | 저장된 J&F | 복구 목표 J&F | Δ복구 | 복구 목표(프롬프트 ≥64 px만) | Δ | 상한 J&F | Δ상한 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SegMaskSam3XW1GPS4M | 15 | 0 | 0.8544 | 0.8655 | +0.0111 | 0.8655 | +0.0111 | 0.8661 | +0.0118 |
| SegMaskSam3XW0M | 15 | 0 | 0.8515 | 0.8627 | +0.0112 | 0.8627 | +0.0112 | 0.8633 | +0.0118 |

"ref 프롬프트 <64 px"는 reference 카메라의 seed GT에서 그 객체가 64 px 미만인 경우입니다 — 프롬프트 자체가 tracker 입력 해상도에서 사라지므로 어떤 뷰에도 donor가 없고 P5로 복구할 수 없습니다.

## misaligned 임계값 민감도

| IoU 임계값 | 방법 | 퇴화 객체 | misaligned | 평균 저장 | 평균 복구 목표 | 평균 상한 | Welder 저장 → 복구 → 상한 |
|---:|---|---:|---:|---:|---:|---:|---|
| 0.10 | SegMaskSam3XW1GPS4M | 15 | 2 | 0.8544 | 0.8655 | 0.8661 | 0.8418 → 0.8815 → 0.8837 |
| 0.10 | SegMaskSam3XW0M | 15 | 2 | 0.8515 | 0.8627 | 0.8633 | 0.8419 → 0.8818 → 0.8840 |
| 0.20 | SegMaskSam3XW1GPS4M | 19 | 6 | 0.8544 | 0.8668 | 0.8676 | 0.8418 → 0.8905 → 0.8936 |
| 0.20 | SegMaskSam3XW0M | 19 | 6 | 0.8515 | 0.8640 | 0.8648 | 0.8419 → 0.8913 → 0.8944 |
| 0.30 | SegMaskSam3XW1GPS4M | 26 | 13 | 0.8544 | 0.8691 | 0.8702 | 0.8418 → 0.8992 → 0.9030 |
| 0.30 | SegMaskSam3XW0M | 26 | 13 | 0.8515 | 0.8664 | 0.8674 | 0.8419 → 0.8996 → 0.9034 |

## REPORT.md 앵커 (A3 / P5) — Welder camera_0004

| 방법 | obj | 클래스 | gt_area | pred_area | IoU | J | F |
|---|---:|---|---:|---:|---:|---:|---:|
| SegMaskSam3XW1GPS4M | 8 | tiny | 13268 | 4 | 0.000 | 0.0001 | 0.0325 |
| SegMaskSam3XW1GPS4M | 12 | ok | 14345 | 2633 | 0.184 | 0.1698 | 0.8974 |
| SegMaskSam3XW1GPS4M | 14 | tiny | 18899 | 7 | 0.000 | 0.0000 | 0.0305 |
| SegMaskSam3XW0M | 8 | tiny | 13268 | 4 | 0.000 | 0.0001 | 0.0317 |
| SegMaskSam3XW0M | 12 | ok | 14345 | 2633 | 0.184 | 0.1492 | 0.8746 |
| SegMaskSam3XW0M | 14 | tiny | 18899 | 7 | 0.000 | 0.0000 | 0.0099 |

Welder 카메라별 저장값 J / F / J&F (퇴화 seed 수), 그리고 `SegMaskSam3XW1GPS4M` 대비 데이터셋 J&F 차이에 대한 카메라별 기여 (= ΔJ&F<sub>cam</sub> × n<sub>cam</sub> / 47):

| 방법 | camera_0001 | camera_0003 | camera_0004 | 데이터셋 | 기여 camera_0001 | 기여 camera_0003 | 기여 camera_0004 | 합 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegMaskSam3XW1GPS4M | 0.8800 / 0.9638 / 0.9219 (0) | 0.7929 / 0.9040 / 0.8485 (0) | 0.7093 / 0.7889 / 0.7491 (2) | 0.8418 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| SegMaskSam3XW0M | 0.8800 / 0.9638 / 0.9219 (0) | 0.7983 / 0.9038 / 0.8510 (0) | 0.7067 / 0.7868 / 0.7468 (2) | 0.8419 | +0.0000 | +0.0009 | -0.0007 | +0.0001 |

## 퇴화 seed 전체 목록

### SegMaskSam3XW1GPS4M — 15개

| 데이터셋 | 카메라 | obj | 클래스 | gt_area | pred_area | ref 프롬프트 px | IoU | J | F | ref |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| AlexaMeadeExhibit | camera_0004 | 17 | missing | 955 | 0 | 1058 | 0.000 | 0.0000 | 0.0000 |  |
| AlexaMeadeExhibit | camera_0004 | 21 | missing | 444 | 0 | 924 | 0.000 | 0.0000 | 0.0000 |  |
| AlexaMeadeExhibit | camera_0004 | 24 | missing | 1683 | 0 | 557 | 0.000 | 0.0000 | 0.0000 |  |
| Blocks | cam0 | 4 | tiny | 442 | 18 | 466 | 0.041 | 0.1926 | 0.2803 |  |
| Blocks | cam0 | 14 | misaligned | 1821 | 503 | 37237 | 0.050 | 0.1798 | 0.4389 |  |
| Blocks | cam9 | 3 | tiny | 343 | 27 | 555 | 0.079 | 0.0252 | 0.2381 |  |
| Breakfast | v5 | 19 | missing | 2 | 0 | 181 | 0.000 | 0.0003 | 0.0253 |  |
| Breakfast | v5 | 24 | tiny | 6 | 4 | 588 | 0.000 | 0.1117 | 0.4163 |  |
| Breakfast | v9 | 24 | tiny | 872 | 48 | 588 | 0.055 | 0.0424 | 0.8083 |  |
| CBABasketball | v06 | 20 | missing | 137380 | 0 | 86577 | 0.000 | 0.0000 | 0.0000 |  |
| MATF | S1_CAM_1 | 26 | misaligned | 4603 | 2168 | 1154 | 0.000 | 0.0000 | 0.3162 |  |
| MATF | S1_CAM_10 | 22 | missing | 1342 | 0 | 327 | 0.000 | 0.0000 | 0.0000 |  |
| Painter | v0 | 9 | missing | 1696 | 0 | 11143 | 0.000 | 0.0000 | 0.0000 |  |
| Welder | camera_0004 | 8 | tiny | 13268 | 4 | 5983 | 0.000 | 0.0001 | 0.0325 |  |
| Welder | camera_0004 | 14 | tiny | 18899 | 7 | 6766 | 0.000 | 0.0000 | 0.0305 |  |

### SegMaskSam3XW0M — 15개

| 데이터셋 | 카메라 | obj | 클래스 | gt_area | pred_area | ref 프롬프트 px | IoU | J | F | ref |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| AlexaMeadeExhibit | camera_0004 | 17 | missing | 955 | 0 | 1058 | 0.000 | 0.0000 | 0.0000 |  |
| AlexaMeadeExhibit | camera_0004 | 21 | missing | 444 | 0 | 924 | 0.000 | 0.0000 | 0.0000 |  |
| AlexaMeadeExhibit | camera_0004 | 24 | missing | 1683 | 0 | 557 | 0.000 | 0.0000 | 0.0000 |  |
| Blocks | cam0 | 4 | tiny | 442 | 18 | 466 | 0.041 | 0.1926 | 0.2803 |  |
| Blocks | cam0 | 14 | misaligned | 1821 | 503 | 37237 | 0.050 | 0.1798 | 0.4389 |  |
| Blocks | cam9 | 3 | tiny | 343 | 27 | 555 | 0.079 | 0.0219 | 0.2381 |  |
| Breakfast | v5 | 19 | missing | 2 | 0 | 181 | 0.000 | 0.0000 | 0.0000 |  |
| Breakfast | v5 | 24 | tiny | 6 | 4 | 588 | 0.000 | 0.1067 | 0.3652 |  |
| Breakfast | v9 | 24 | tiny | 872 | 48 | 588 | 0.055 | 0.0318 | 0.7538 |  |
| CBABasketball | v06 | 20 | missing | 137380 | 0 | 86577 | 0.000 | 0.0000 | 0.0000 |  |
| MATF | S1_CAM_1 | 26 | misaligned | 4603 | 2168 | 1154 | 0.000 | 0.0000 | 0.3162 |  |
| MATF | S1_CAM_10 | 22 | missing | 1342 | 0 | 327 | 0.000 | 0.0000 | 0.0000 |  |
| Painter | v0 | 9 | missing | 1696 | 0 | 11143 | 0.000 | 0.0000 | 0.0000 |  |
| Welder | camera_0004 | 8 | tiny | 13268 | 4 | 5983 | 0.000 | 0.0001 | 0.0317 |  |
| Welder | camera_0004 | 14 | tiny | 18899 | 7 | 6766 | 0.000 | 0.0000 | 0.0099 |  |

`ref` ●는 reference 카메라(seed = GT prompt 재예측). "ref 프롬프트 px"는 reference 카메라 seed GT에서의 면적.

## 경계 사례 — ok이지만 IoU < 0.30

분류상 ok지만 seed IoU가 낮아 REPORT.md가 obj 12를 "어긋난" seed라 부른 것과 같은 부류입니다. IoU < 0.10 규칙은 이들을 잡지 않습니다.

| 데이터셋 | 카메라 | obj | 방법 | gt_area | pred_area | IoU | J | F |
|---|---|---:|---|---:|---:|---:|---:|---:|
| AlexaMeadeExhibit | camera_0003 | 17 | SegMaskSam3XW1GPS4M | 1118 | 302 | 0.261 | 0.0911 | 0.8132 |
| AlexaMeadeExhibit | camera_0003 | 17 | SegMaskSam3XW0M | 1118 | 302 | 0.261 | 0.0731 | 0.8404 |
| AlexaMeadeExhibit | camera_0003 | 21 | SegMaskSam3XW1GPS4M | 644 | 171 | 0.266 | 0.0195 | 0.1905 |
| AlexaMeadeExhibit | camera_0003 | 21 | SegMaskSam3XW0M | 644 | 171 | 0.266 | 0.0158 | 0.0952 |
| AlexaMeadeExhibit | camera_0004 | 30 | SegMaskSam3XW1GPS4M | 569 | 121 | 0.146 | 0.0402 | 0.5186 |
| AlexaMeadeExhibit | camera_0004 | 30 | SegMaskSam3XW0M | 569 | 121 | 0.146 | 0.0598 | 0.5608 |
| Barn | v0 | 1 | SegMaskSam3XW1GPS4M | 3905 | 1532 | 0.295 | 0.3233 | 0.8591 |
| Barn | v0 | 1 | SegMaskSam3XW0M | 3905 | 1532 | 0.295 | 0.3233 | 0.8591 |
| Breakfast | v5 | 14 | SegMaskSam3XW1GPS4M | 347 | 66 | 0.190 | 0.3809 | 0.8618 |
| Breakfast | v5 | 14 | SegMaskSam3XW0M | 347 | 66 | 0.190 | 0.3812 | 0.8618 |
| FlameSteak | cam01 | 5 | SegMaskSam3XW1GPS4M | 6987 | 1934 | 0.248 | 0.2169 | 0.8032 |
| FlameSteak | cam01 | 5 | SegMaskSam3XW0M | 6987 | 1934 | 0.248 | 0.2088 | 0.8005 |
| FlameSteak | cam10 | 14 | SegMaskSam3XW1GPS4M | 2064 | 1057 | 0.189 | 0.1904 | 0.9197 |
| FlameSteak | cam10 | 14 | SegMaskSam3XW0M | 2064 | 1057 | 0.189 | 0.1871 | 0.9196 |
| FlameSteak | cam10 | 37 | SegMaskSam3XW1GPS4M | 149179 | 46809 | 0.241 | 0.2404 | 0.6536 |
| FlameSteak | cam10 | 37 | SegMaskSam3XW0M | 149179 | 46809 | 0.241 | 0.2404 | 0.6537 |
| MATF | S1_CAM_10 | 10 | SegMaskSam3XW1GPS4M | 514 | 1713 | 0.204 | 0.2272 | 0.8697 |
| MATF | S1_CAM_10 | 10 | SegMaskSam3XW0M | 514 | 1713 | 0.204 | 0.2260 | 0.8713 |
| Welder | camera_0003 | 4 | SegMaskSam3XW1GPS4M | 477 | 112 | 0.235 | 0.1711 | 0.9390 |
| Welder | camera_0003 | 4 | SegMaskSam3XW0M | 477 | 112 | 0.235 | 0.2224 | 0.9288 |
| Welder | camera_0004 | 12 | SegMaskSam3XW1GPS4M | 14345 | 2633 | 0.184 | 0.1698 | 0.8974 |
| Welder | camera_0004 | 12 | SegMaskSam3XW0M | 14345 | 2633 | 0.184 | 0.1492 | 0.8746 |

## gt_empty (참고 — 퇴화 아님)

- **SegMaskSam3XW1GPS4M**: 4개 (seed가 0 px가 아닌 것 0개) — Blocks/cam9/4, MATF/S1_CAM_1/6, MATF/S1_CAM_1/7, PoznanStreet/v8/2
- **SegMaskSam3XW0M**: 4개 (seed가 0 px가 아닌 것 0개) — Blocks/cam9/4, MATF/S1_CAM_1/6, MATF/S1_CAM_1/7, PoznanStreet/v8/2

## 판독 — P5의 기대 이득

- **SegMaskSam3XW1GPS4M**: 퇴화 seed 15개 / 도달 가능 객체 1066개. 복구 목표 0.8544 → 0.8655 (+0.0111), 프롬프트 ≥64 px만 0.8655 (+0.0111), 상한 0.8661 (+0.0118). 데이터셋별(저장 → 복구 → 상한): Blocks 3개 0.7831 → 0.8268 → 0.8278; Welder 2개 0.8418 → 0.8815 → 0.8837; AlexaMeadeExhibit 3개 0.8624 → 0.8928 → 0.8947; Breakfast 3개 0.8026 → 0.8270 → 0.8296; MATF 2개 0.7284 → 0.7505 → 0.7523; CBABasketball 1개 0.8757 → 0.8919 → 0.8929; Painter 1개 0.8512 → 0.8635 → 0.8640.
- **SegMaskSam3XW0M**: 퇴화 seed 15개 / 도달 가능 객체 1066개. 복구 목표 0.8515 → 0.8627 (+0.0112), 프롬프트 ≥64 px만 0.8627 (+0.0112), 상한 0.8633 (+0.0118). 데이터셋별(저장 → 복구 → 상한): Blocks 3개 0.7767 → 0.8204 → 0.8214; Welder 2개 0.8419 → 0.8818 → 0.8840; AlexaMeadeExhibit 3개 0.8618 → 0.8922 → 0.8941; Breakfast 3개 0.8012 → 0.8266 → 0.8291; MATF 2개 0.7284 → 0.7506 → 0.7523; CBABasketball 1개 0.8751 → 0.8913 → 0.8923; Painter 1개 0.8494 → 0.8617 → 0.8622.

**Welder 앵커 재현.** `SegMaskSam3XW0M`의 camera_0004에서 obj 8/14는 4 px / 7 px seed(GT 13,268 / 18,899 px)로 tiny / tiny, J 0.00 / 0.00. obj 12는 2,633 px, IoU 0.184, J 0.15 — REPORT.md는 이를 "어긋난" seed라 했고, 이 census의 규칙(IoU < 0.10)으로는 **ok**입니다 (IoU 0.184보다 높은 임계값이면 misaligned로 잡힙니다). camera_0004의 J&F는 `SegMaskSam3XW1GPS4M` 0.7491 vs `SegMaskSam3XW0M` 0.7468(REPORT.md의 0.861 vs 0.707은 J&F가 아니라 J: 0.7093 vs 0.7067), 데이터셋 격차 +0.0001 중 이 카메라의 기여가 -0.0007 — "Welder 손실 전부가 camera_0004"는 성립하지 않습니다 (camera_0003에서는 오히려 `SegMaskSam3XW0M`가 앞섭니다).

**P5의 기대 이득.** `SegMaskSam3XW1GPS4M`: 퇴화 15개는 도달 가능 객체 1066개의 1.4%, 데이터셋 17개 중 10개는 0개. 복구 목표 이득 +0.0111 중 +0.0000는 ref 프롬프트 <64 px 객체의 몫이라 P5가 닿을 수 있는 이득은 +0.0111, 기여 상위: Blocks +0.0026, Welder +0.0023, AlexaMeadeExhibit +0.0018. `SegMaskSam3XW0M`: 퇴화 15개는 도달 가능 객체 1066개의 1.4%, 데이터셋 17개 중 10개는 0개. 복구 목표 이득 +0.0112 중 +0.0000는 ref 프롬프트 <64 px 객체의 몫이라 P5가 닿을 수 있는 이득은 +0.0112, 기여 상위: Blocks +0.0026, Welder +0.0023, AlexaMeadeExhibit +0.0018. ref 프롬프트 <64 px인 퇴화 객체는 없습니다. `SegMaskSam3XW0M`에서 P5가 닿는 이득의 출처. 첫째: Blocks cam0 obj 14 misaligned; cam0 obj 4 tiny; cam9 obj 3 tiny — 데이터셋 +0.0437 = 헤드라인 +0.0026; `SegMaskSam3XW1GPS4M` 대비 Blocks 격차 -0.0064 → 복구 후 +0.0373 (소거). 둘째: Welder camera_0004 obj 8/14 tiny — 데이터셋 +0.0399 = 헤드라인 +0.0023; `SegMaskSam3XW1GPS4M` 대비 Welder 격차 +0.0001 → 복구 후 +0.0401 (소거); REPORT.md P5의 "Welder 0.8437 → ~0.879, 헤드라인 +0.0023~0.0027"와 비교. `SegMaskSam3XW1GPS4M`도 같은 종류의 seed 실패를 15개(missing 7, tiny 6, misaligned 2) 갖고 있고, 그중 empty+missing 7개 가운데 7개는 `SegMaskSam3XW0M`의 missing+empty 7개와 같은 (카메라, 객체)입니다. P5를 `SegMaskSam3XW1GPS4M` 러너에도 적용하면 두 방법이 함께 오르고 저장값 순위(`SegMaskSam3XW1GPS4M` > `SegMaskSam3XW0M`)는 복구 목표(프롬프트 ≥64 px)에서도 그대로이므로, P5는 순위를 바꾸는 항목이 아니라 두 방법의 바닥을 올리는 항목입니다. misaligned 임계값을 0.10에서 0.20~0.30(으)로 올리면 퇴화 수가 늘고(`SegMaskSam3XW1GPS4M` 15 → 26, `SegMaskSam3XW0M` 15 → 26) 복구 목표도 위 민감도 표만큼 오르지만(Welder camera_0004 obj 12, IoU 0.184 부류), 그 seed들은 정의상 tiny가 아니라 면적 검사로는 잡히지 않아 P5의 감지기가 IoU 계열 신호(donor 마스크와의 IoU)를 함께 봐야 합니다. unreachable은 1139개 scored 객체 중 73개로 모든 방법에 동일하며 P5가 아니라 P8(reference 규칙)의 몫입니다.
