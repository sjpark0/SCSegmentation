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

**퇴화(degenerate)** = missing + empty + tiny + misaligned. SAM 2 러너는 0 px 파일을 쓰고 SAM 3 러너는 파일을 안 쓰므로 `SegMaskNew1`의 empty와 SAM 3 폴더의 missing은 같은 실패 양상(0-px 파일 vs 파일 없음)이지만 같은 객체 집합은 아닙니다 — empty+missing 기준으로 `SegMaskNew1` 13개 vs `SegMaskSam3OneStage` 15개 중 11개가 겹칩니다(`SegMaskNew1`에만: Breakfast/v5/14, Breakfast/v7/14; `SegMaskSam3OneStage`에만: AlexaMeadeExhibit/camera_0004/17, AlexaMeadeExhibit/camera_0004/21, AlexaMeadeExhibit/camera_0004/24, CoffeeMartini/cam16/19); `SegMaskNew1` 13개 vs `SegMaskSam3MVOpt` 15개 중 11개가 겹칩니다(`SegMaskNew1`에만: Breakfast/v5/14, Breakfast/v7/14; `SegMaskSam3MVOpt`에만: AlexaMeadeExhibit/camera_0004/17, AlexaMeadeExhibit/camera_0004/21, AlexaMeadeExhibit/camera_0004/24, CoffeeMartini/cam16/19). **저장된 J&F** = `(cam, obj)` 쌍의 (J+F)/2 평균 = `report_jf.py` as-is/all 표(`docs/experiments.md`). **복구 목표 J&F** = 퇴화 객체에 max(저장된 (J+F)/2, 그 데이터셋·방법의 ok 객체 중앙값 J와 중앙값 F의 평균)을 준 값 — 저장값이 이미 중앙값보다 높은 객체는 낮추지 않습니다. **상한 J&F** = 퇴화 객체를 J=F=1로 둔 값. no_folder·unreachable·gt_empty는 두 계산에서 건드리지 않습니다.

점수 출처: `SegMaskNew1` ← `jf_raw.json`, `SegMaskSam3OneStage` ← `jf_sam3_onestage.json`, `SegMaskSam3MVOpt` ← `jf_sam3_mvopt_all.json`. 데이터셋별 reference 카메라: AlexaMeadeExhibit `camera_0001`, AlexaMeadeFacePaint `camera_0008`, Barn `v0`, Blocks `cam9`, Breakfast `v5`, Carpark `v0`, CoffeeMartini `cam02`, Dog `camera_0003`, Fencing `v0`, FlameSteak `cam10`, Frog `v4`, MATF `S1_CAM_1`, Painter `v15`, PoznanStreet `v0`, Welder `camera_0001`.

## SegMaskNew1

| 데이터셋 | 객체 수 | unreachable | missing | empty | tiny | misaligned | 저장된 J&F | 복구 목표 J&F | 상한 J&F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AlexaMeadeExhibit | 93 | 0 | 0 | 0 | 0 | 0 | 0.8913 | 0.8913 | 0.8913 |
| AlexaMeadeFacePaint | 33 | 5 | 0 | 0 | 0 | 0 | 0.7845 | 0.7845 | 0.7845 |
| Barn | 90 | 0 | 0 | 0 | 0 | 1 | 0.8569 | 0.8670 | 0.8677 |
| Blocks | 52 | 10 | 0 | 0 | 0 | 1 | 0.7488 | 0.7617 | 0.7621 |
| Breakfast | 85 | 5 | 0 | 10 | 0 | 0 | 0.7138 | 0.8118 | 0.8191 |
| Carpark | 66 | 0 | 0 | 0 | 0 | 0 | 0.9416 | 0.9416 | 0.9416 |
| CoffeeMartini | 143 | 13 | 0 | 0 | 0 | 2 | 0.8380 | 0.8516 | 0.8519 |
| Dog | 23 | 0 | 0 | 0 | 0 | 0 | 0.9075 | 0.9075 | 0.9075 |
| Fencing | 24 | 0 | 0 | 0 | 0 | 0 | 0.9321 | 0.9321 | 0.9321 |
| FlameSteak | 134 | 19 | 0 | 1 | 0 | 0 | 0.7832 | 0.7905 | 0.7907 |
| Frog | 15 | 0 | 0 | 0 | 0 | 0 | 0.9754 | 0.9754 | 0.9754 |
| MATF | 77 | 18 | 0 | 1 | 0 | 2 | 0.6649 | 0.7007 | 0.7031 |
| Painter | 78 | 5 | 0 | 1 | 0 | 0 | 0.8713 | 0.8786 | 0.8791 |
| PoznanStreet | 71 | 2 | 0 | 0 | 0 | 0 | 0.8646 | 0.8646 | 0.8646 |
| Welder | 47 | 0 | 0 | 0 | 0 | 0 | 0.8778 | 0.8778 | 0.8778 |
| **평균 (15개)** | 1031 | 77 | 0 | 13 | 0 | 6 | **0.8434** | **0.8558** | **0.8566** |

gt_empty 3개, ok 932개 (합계 1031개 중). 평균 행의 J&F는 데이터셋별 값의 평균이고 객체 수 열은 합계입니다.

## SegMaskSam3OneStage

| 데이터셋 | 객체 수 | unreachable | missing | empty | tiny | misaligned | 저장된 J&F | 복구 목표 J&F | 상한 J&F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AlexaMeadeExhibit | 93 | 0 | 3 | 0 | 0 | 0 | 0.8618 | 0.8922 | 0.8941 |
| AlexaMeadeFacePaint | 33 | 5 | 0 | 0 | 0 | 0 | 0.7927 | 0.7927 | 0.7927 |
| Barn | 90 | 0 | 0 | 0 | 0 | 0 | 0.8858 | 0.8858 | 0.8858 |
| Blocks | 52 | 10 | 0 | 0 | 0 | 1 | 0.7484 | 0.7616 | 0.7619 |
| Breakfast | 85 | 5 | 8 | 0 | 0 | 0 | 0.7419 | 0.8200 | 0.8259 |
| Carpark | 66 | 0 | 0 | 0 | 0 | 0 | 0.9438 | 0.9438 | 0.9438 |
| CoffeeMartini | 143 | 13 | 1 | 0 | 1 | 0 | 0.8461 | 0.8576 | 0.8579 |
| Dog | 23 | 0 | 0 | 0 | 0 | 0 | 0.9241 | 0.9241 | 0.9241 |
| Fencing | 24 | 0 | 0 | 0 | 0 | 0 | 0.8862 | 0.8862 | 0.8862 |
| FlameSteak | 134 | 19 | 1 | 0 | 0 | 0 | 0.7736 | 0.7808 | 0.7811 |
| Frog | 15 | 0 | 0 | 0 | 0 | 0 | 0.9753 | 0.9753 | 0.9753 |
| MATF | 77 | 18 | 1 | 0 | 1 | 0 | 0.6840 | 0.7035 | 0.7051 |
| Painter | 78 | 5 | 1 | 0 | 0 | 0 | 0.8682 | 0.8806 | 0.8810 |
| PoznanStreet | 71 | 2 | 0 | 0 | 0 | 0 | 0.8662 | 0.8662 | 0.8662 |
| Welder | 47 | 0 | 0 | 0 | 2 | 0 | 0.8419 | 0.8818 | 0.8840 |
| **평균 (15개)** | 1031 | 77 | 15 | 0 | 4 | 1 | **0.8427** | **0.8568** | **0.8577** |

gt_empty 3개, ok 931개 (합계 1031개 중). 평균 행의 J&F는 데이터셋별 값의 평균이고 객체 수 열은 합계입니다.

## SegMaskSam3MVOpt

| 데이터셋 | 객체 수 | unreachable | missing | empty | tiny | misaligned | 저장된 J&F | 복구 목표 J&F | 상한 J&F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AlexaMeadeExhibit | 93 | 0 | 3 | 0 | 0 | 0 | 0.8618 | 0.8921 | 0.8940 |
| AlexaMeadeFacePaint | 33 | 5 | 0 | 0 | 0 | 0 | 0.7928 | 0.7928 | 0.7928 |
| Barn | 90 | 0 | 0 | 0 | 0 | 0 | 0.8865 | 0.8865 | 0.8865 |
| Blocks | 52 | 10 | 0 | 0 | 0 | 1 | 0.7478 | 0.7609 | 0.7613 |
| Breakfast | 85 | 5 | 8 | 0 | 0 | 0 | 0.7421 | 0.8202 | 0.8262 |
| Carpark | 66 | 0 | 0 | 0 | 0 | 0 | 0.9434 | 0.9434 | 0.9434 |
| CoffeeMartini | 143 | 13 | 1 | 0 | 1 | 0 | 0.8440 | 0.8573 | 0.8576 |
| Dog | 23 | 0 | 0 | 0 | 0 | 0 | 0.9241 | 0.9241 | 0.9241 |
| Fencing | 24 | 0 | 0 | 0 | 0 | 0 | 0.9112 | 0.9112 | 0.9112 |
| FlameSteak | 134 | 19 | 1 | 0 | 0 | 0 | 0.7734 | 0.7806 | 0.7809 |
| Frog | 15 | 0 | 0 | 0 | 0 | 0 | 0.9743 | 0.9743 | 0.9743 |
| MATF | 77 | 18 | 1 | 0 | 1 | 0 | 0.6841 | 0.7036 | 0.7052 |
| Painter | 78 | 5 | 1 | 0 | 0 | 0 | 0.8681 | 0.8805 | 0.8810 |
| PoznanStreet | 71 | 2 | 0 | 0 | 0 | 0 | 0.8787 | 0.8787 | 0.8787 |
| Welder | 47 | 0 | 0 | 0 | 2 | 0 | 0.8437 | 0.8839 | 0.8860 |
| **평균 (15개)** | 1031 | 77 | 15 | 0 | 4 | 1 | **0.8451** | **0.8593** | **0.8602** |

gt_empty 3개, ok 931개 (합계 1031개 중). 평균 행의 J&F는 데이터셋별 값의 평균이고 객체 수 열은 합계입니다.

## 15개 데이터셋 평균 — 방법 비교

| 방법 | 퇴화 객체 | 그중 ref 프롬프트 <64 px | 저장된 J&F | 복구 목표 J&F | Δ복구 | 복구 목표(프롬프트 ≥64 px만) | Δ | 상한 J&F | Δ상한 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SegMaskNew1 | 19 | 7 | 0.8434 | 0.8558 | +0.0123 | 0.8507 | +0.0073 | 0.8566 | +0.0131 |
| SegMaskSam3OneStage | 20 | 7 | 0.8427 | 0.8568 | +0.0141 | 0.8517 | +0.0091 | 0.8577 | +0.0150 |
| SegMaskSam3MVOpt | 20 | 7 | 0.8451 | 0.8593 | +0.0143 | 0.8543 | +0.0092 | 0.8602 | +0.0151 |

"ref 프롬프트 <64 px"는 reference 카메라의 seed GT에서 그 객체가 64 px 미만인 경우입니다 — 프롬프트 자체가 tracker 입력 해상도에서 사라지므로 어떤 뷰에도 donor가 없고 P5로 복구할 수 없습니다.

## misaligned 임계값 민감도

| IoU 임계값 | 방법 | 퇴화 객체 | misaligned | 평균 저장 | 평균 복구 목표 | 평균 상한 | Welder 저장 → 복구 → 상한 |
|---:|---|---:|---:|---:|---:|---:|---|
| 0.10 | SegMaskNew1 | 19 | 6 | 0.8434 | 0.8558 | 0.8566 | 0.8778 → 0.8778 → 0.8778 |
| 0.10 | SegMaskSam3OneStage | 20 | 1 | 0.8427 | 0.8568 | 0.8577 | 0.8419 → 0.8818 → 0.8840 |
| 0.10 | SegMaskSam3MVOpt | 20 | 1 | 0.8451 | 0.8593 | 0.8602 | 0.8437 → 0.8839 → 0.8860 |
| 0.20 | SegMaskNew1 | 24 | 11 | 0.8434 | 0.8591 | 0.8601 | 0.8778 → 0.8930 → 0.8938 |
| 0.20 | SegMaskSam3OneStage | 25 | 6 | 0.8427 | 0.8598 | 0.8608 | 0.8419 → 0.8913 → 0.8944 |
| 0.20 | SegMaskSam3MVOpt | 25 | 6 | 0.8451 | 0.8623 | 0.8634 | 0.8437 → 0.8937 → 0.8967 |
| 0.30 | SegMaskNew1 | 32 | 19 | 0.8434 | 0.8622 | 0.8636 | 0.8778 → 0.8930 → 0.8938 |
| 0.30 | SegMaskSam3OneStage | 30 | 11 | 0.8427 | 0.8619 | 0.8631 | 0.8419 → 0.8996 → 0.9034 |
| 0.30 | SegMaskSam3MVOpt | 30 | 11 | 0.8451 | 0.8643 | 0.8656 | 0.8437 → 0.9003 → 0.9040 |

## REPORT.md 앵커 (A3 / P5) — Welder camera_0004

| 방법 | obj | 클래스 | gt_area | pred_area | IoU | J | F |
|---|---:|---|---:|---:|---:|---:|---:|
| SegMaskNew1 | 8 | ok | 13268 | 13197 | 0.943 | 0.9321 | 1.0000 |
| SegMaskNew1 | 12 | ok | 14345 | 13733 | 0.897 | 0.9023 | 1.0000 |
| SegMaskNew1 | 14 | ok | 18899 | 17903 | 0.931 | 0.9251 | 1.0000 |
| SegMaskSam3OneStage | 8 | tiny | 13268 | 4 | 0.000 | 0.0001 | 0.0317 |
| SegMaskSam3OneStage | 12 | ok | 14345 | 2633 | 0.184 | 0.1492 | 0.8746 |
| SegMaskSam3OneStage | 14 | tiny | 18899 | 7 | 0.000 | 0.0000 | 0.0099 |
| SegMaskSam3MVOpt | 8 | tiny | 13268 | 4 | 0.000 | 0.0000 | 0.0105 |
| SegMaskSam3MVOpt | 12 | ok | 14345 | 2633 | 0.184 | 0.1369 | 0.8586 |
| SegMaskSam3MVOpt | 14 | tiny | 18899 | 7 | 0.000 | 0.0000 | 0.0099 |

Welder 카메라별 저장값 J / F / J&F (퇴화 seed 수), 그리고 `SegMaskNew1` 대비 데이터셋 J&F 차이에 대한 카메라별 기여 (= ΔJ&F<sub>cam</sub> × n<sub>cam</sub> / 47):

| 방법 | camera_0001 | camera_0003 | camera_0004 | 데이터셋 | 기여 camera_0001 | 기여 camera_0003 | 기여 camera_0004 | 합 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegMaskNew1 | 0.8882 / 0.9630 / 0.9256 (0) | 0.8186 / 0.8621 / 0.8403 (0) | 0.8613 / 0.8722 / 0.8668 (0) | 0.8778 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| SegMaskSam3OneStage | 0.8800 / 0.9638 / 0.9219 (0) | 0.7983 / 0.9038 / 0.8510 (0) | 0.7067 / 0.7868 / 0.7468 (2) | 0.8419 | -0.0012 | +0.0036 | -0.0383 | -0.0359 |
| SegMaskSam3MVOpt | 0.8804 / 0.9643 / 0.9223 (0) | 0.8090 / 0.9050 / 0.8570 (0) | 0.7071 / 0.7841 / 0.7456 (2) | 0.8437 | -0.0011 | +0.0057 | -0.0387 | -0.0341 |

## 퇴화 seed 전체 목록

### SegMaskNew1 — 19개

| 데이터셋 | 카메라 | obj | 클래스 | gt_area | pred_area | ref 프롬프트 px | IoU | J | F | ref |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| Barn | v7 | 30 | misaligned | 1049 | 1452 | 1256 | 0.000 | 0.0000 | 0.0433 |  |
| Blocks | cam0 | 14 | misaligned | 1821 | 579 | 64451 | 0.077 | 0.1857 | 0.4325 |  |
| Breakfast | v5 | 14 | empty | 347 | 0 | 347 | 0.000 | 0.1429 | 0.1429 | ● |
| Breakfast | v5 | 16 | empty | 7 | 0 | 7 | 0.000 | 0.0000 | 0.0000 | ● |
| Breakfast | v5 | 19 | empty | 2 | 0 | 2 | 0.000 | 0.0000 | 0.0000 | ● |
| Breakfast | v5 | 24 | empty | 6 | 0 | 6 | 0.000 | 0.0952 | 0.0952 | ● |
| Breakfast | v7 | 14 | empty | 1116 | 0 | 347 | 0.000 | 0.0476 | 0.0476 |  |
| Breakfast | v7 | 19 | empty | 181 | 0 | 2 | 0.000 | 0.0000 | 0.0000 |  |
| Breakfast | v7 | 24 | empty | 588 | 0 | 6 | 0.000 | 0.0000 | 0.0000 |  |
| Breakfast | v9 | 14 | empty | 457 | 0 | 347 | 0.000 | 0.7619 | 0.7619 |  |
| Breakfast | v9 | 19 | empty | 357 | 0 | 2 | 0.000 | 0.0000 | 0.0000 |  |
| Breakfast | v9 | 24 | empty | 872 | 0 | 6 | 0.000 | 0.0000 | 0.0000 |  |
| CoffeeMartini | cam10 | 19 | misaligned | 6547 | 33638 | 18867 | 0.000 | 0.0000 | 0.0000 |  |
| CoffeeMartini | cam16 | 19 | misaligned | 2409 | 78672 | 18867 | 0.000 | 0.0000 | 0.0000 |  |
| FlameSteak | cam01 | 60 | empty | 5518 | 0 | 15613 | 0.000 | 0.0000 | 0.0000 |  |
| MATF | S1_CAM_10 | 21 | misaligned | 6100 | 55781 | 466 | 0.000 | 0.0000 | 0.0000 |  |
| MATF | S1_CAM_4 | 21 | misaligned | 6224 | 57504 | 466 | 0.000 | 0.0006 | 0.1117 |  |
| MATF | S1_CAM_4 | 26 | empty | 1154 | 0 | 4603 | 0.000 | 0.0000 | 0.0000 |  |
| Painter | v0 | 9 | empty | 1696 | 0 | 48489 | 0.000 | 0.3981 | 0.3873 |  |

### SegMaskSam3OneStage — 20개

| 데이터셋 | 카메라 | obj | 클래스 | gt_area | pred_area | ref 프롬프트 px | IoU | J | F | ref |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| AlexaMeadeExhibit | camera_0004 | 17 | missing | 955 | 0 | 1058 | 0.000 | 0.0000 | 0.0000 |  |
| AlexaMeadeExhibit | camera_0004 | 21 | missing | 444 | 0 | 924 | 0.000 | 0.0000 | 0.0000 |  |
| AlexaMeadeExhibit | camera_0004 | 24 | missing | 1683 | 0 | 557 | 0.000 | 0.0000 | 0.0000 |  |
| Blocks | cam0 | 14 | misaligned | 1821 | 356 | 64451 | 0.030 | 0.1627 | 0.4314 |  |
| Breakfast | v5 | 16 | missing | 7 | 0 | 7 | 0.000 | 0.0000 | 0.0000 | ● |
| Breakfast | v5 | 19 | missing | 2 | 0 | 2 | 0.000 | 0.0000 | 0.0000 | ● |
| Breakfast | v5 | 24 | missing | 6 | 0 | 6 | 0.000 | 0.0952 | 0.0952 | ● |
| Breakfast | v7 | 19 | missing | 181 | 0 | 2 | 0.000 | 0.0000 | 0.0000 |  |
| Breakfast | v7 | 24 | missing | 588 | 0 | 6 | 0.000 | 0.0000 | 0.0000 |  |
| Breakfast | v9 | 14 | missing | 457 | 0 | 347 | 0.000 | 0.7619 | 0.7619 |  |
| Breakfast | v9 | 19 | missing | 357 | 0 | 2 | 0.000 | 0.0000 | 0.0000 |  |
| Breakfast | v9 | 24 | missing | 872 | 0 | 6 | 0.000 | 0.0000 | 0.0000 |  |
| CoffeeMartini | cam16 | 15 | tiny | 3356 | 13 | 15595 | 0.004 | 0.0309 | 0.5734 |  |
| CoffeeMartini | cam16 | 19 | missing | 2409 | 0 | 18867 | 0.000 | 0.0000 | 0.0000 |  |
| FlameSteak | cam01 | 60 | missing | 5518 | 0 | 15613 | 0.000 | 0.0000 | 0.0000 |  |
| MATF | S1_CAM_10 | 10 | tiny | 514 | 19 | 702 | 0.035 | 0.1050 | 0.6496 |  |
| MATF | S1_CAM_4 | 26 | missing | 1154 | 0 | 4603 | 0.000 | 0.0000 | 0.0000 |  |
| Painter | v0 | 9 | missing | 1696 | 0 | 48489 | 0.000 | 0.0000 | 0.0000 |  |
| Welder | camera_0004 | 8 | tiny | 13268 | 4 | 5983 | 0.000 | 0.0001 | 0.0317 |  |
| Welder | camera_0004 | 14 | tiny | 18899 | 7 | 6766 | 0.000 | 0.0000 | 0.0099 |  |

### SegMaskSam3MVOpt — 20개

| 데이터셋 | 카메라 | obj | 클래스 | gt_area | pred_area | ref 프롬프트 px | IoU | J | F | ref |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| AlexaMeadeExhibit | camera_0004 | 17 | missing | 955 | 0 | 1058 | 0.000 | 0.0000 | 0.0000 |  |
| AlexaMeadeExhibit | camera_0004 | 21 | missing | 444 | 0 | 924 | 0.000 | 0.0000 | 0.0000 |  |
| AlexaMeadeExhibit | camera_0004 | 24 | missing | 1683 | 0 | 557 | 0.000 | 0.0000 | 0.0000 |  |
| Blocks | cam0 | 14 | misaligned | 1821 | 356 | 64451 | 0.030 | 0.1645 | 0.4312 |  |
| Breakfast | v5 | 16 | missing | 7 | 0 | 7 | 0.000 | 0.0000 | 0.0000 | ● |
| Breakfast | v5 | 19 | missing | 2 | 0 | 2 | 0.000 | 0.0000 | 0.0000 | ● |
| Breakfast | v5 | 24 | missing | 6 | 0 | 6 | 0.000 | 0.0952 | 0.0952 | ● |
| Breakfast | v7 | 19 | missing | 181 | 0 | 2 | 0.000 | 0.0000 | 0.0000 |  |
| Breakfast | v7 | 24 | missing | 588 | 0 | 6 | 0.000 | 0.0000 | 0.0000 |  |
| Breakfast | v9 | 14 | missing | 457 | 0 | 347 | 0.000 | 0.7619 | 0.7619 |  |
| Breakfast | v9 | 19 | missing | 357 | 0 | 2 | 0.000 | 0.0000 | 0.0000 |  |
| Breakfast | v9 | 24 | missing | 872 | 0 | 6 | 0.000 | 0.0000 | 0.0000 |  |
| CoffeeMartini | cam16 | 15 | tiny | 3356 | 13 | 15595 | 0.004 | 0.0050 | 0.1059 |  |
| CoffeeMartini | cam16 | 19 | missing | 2409 | 0 | 18867 | 0.000 | 0.0000 | 0.0000 |  |
| FlameSteak | cam01 | 60 | missing | 5518 | 0 | 15613 | 0.000 | 0.0000 | 0.0000 |  |
| MATF | S1_CAM_10 | 10 | tiny | 514 | 19 | 702 | 0.035 | 0.1055 | 0.6506 |  |
| MATF | S1_CAM_4 | 26 | missing | 1154 | 0 | 4603 | 0.000 | 0.0000 | 0.0000 |  |
| Painter | v0 | 9 | missing | 1696 | 0 | 48489 | 0.000 | 0.0000 | 0.0000 |  |
| Welder | camera_0004 | 8 | tiny | 13268 | 4 | 5983 | 0.000 | 0.0000 | 0.0105 |  |
| Welder | camera_0004 | 14 | tiny | 18899 | 7 | 6766 | 0.000 | 0.0000 | 0.0099 |  |

`ref` ●는 reference 카메라(seed = GT prompt 재예측). "ref 프롬프트 px"는 reference 카메라 seed GT에서의 면적.

## 경계 사례 — ok이지만 IoU < 0.30

분류상 ok지만 seed IoU가 낮아 REPORT.md가 obj 12를 "어긋난" seed라 부른 것과 같은 부류입니다. IoU < 0.10 규칙은 이들을 잡지 않습니다.

| 데이터셋 | 카메라 | obj | 방법 | gt_area | pred_area | IoU | J | F |
|---|---|---:|---|---:|---:|---:|---:|---:|
| AlexaMeadeExhibit | camera_0003 | 17 | SegMaskSam3OneStage | 1118 | 302 | 0.261 | 0.0731 | 0.8404 |
| AlexaMeadeExhibit | camera_0003 | 17 | SegMaskSam3MVOpt | 1118 | 302 | 0.261 | 0.0783 | 0.8434 |
| AlexaMeadeExhibit | camera_0003 | 21 | SegMaskSam3OneStage | 644 | 171 | 0.266 | 0.0158 | 0.0952 |
| AlexaMeadeExhibit | camera_0003 | 21 | SegMaskSam3MVOpt | 644 | 171 | 0.266 | 0.0166 | 0.0952 |
| AlexaMeadeExhibit | camera_0003 | 24 | SegMaskNew1 | 1602 | 607 | 0.227 | 0.2214 | 0.8345 |
| AlexaMeadeExhibit | camera_0004 | 9 | SegMaskNew1 | 869377 | 192644 | 0.218 | 0.2175 | 0.4526 |
| AlexaMeadeExhibit | camera_0004 | 30 | SegMaskSam3OneStage | 569 | 121 | 0.146 | 0.0598 | 0.5608 |
| AlexaMeadeExhibit | camera_0004 | 30 | SegMaskSam3MVOpt | 569 | 121 | 0.146 | 0.0410 | 0.5169 |
| Barn | v10 | 23 | SegMaskNew1 | 222749 | 36277 | 0.154 | 0.1665 | 0.5210 |
| Barn | v10 | 26 | SegMaskNew1 | 904376 | 241781 | 0.267 | 0.2718 | 0.3147 |
| Barn | v7 | 6 | SegMaskNew1 | 1977 | 2446 | 0.262 | 0.3027 | 0.6233 |
| Barn | v7 | 23 | SegMaskNew1 | 211671 | 43585 | 0.205 | 0.2168 | 0.4707 |
| Barn | v7 | 26 | SegMaskNew1 | 869845 | 201214 | 0.230 | 0.2364 | 0.2970 |
| Barn | v7 | 30 | SegMaskSam3OneStage | 1049 | 250 | 0.232 | 0.2482 | 0.8910 |
| Barn | v7 | 30 | SegMaskSam3MVOpt | 1049 | 250 | 0.232 | 0.2478 | 0.8892 |
| Blocks | cam4 | 3 | SegMaskSam3OneStage | 555 | 114 | 0.122 | 0.0081 | 0.4852 |
| Blocks | cam4 | 3 | SegMaskSam3MVOpt | 555 | 114 | 0.122 | 0.0076 | 0.4738 |
| Breakfast | v9 | 8 | SegMaskSam3OneStage | 27815 | 4994 | 0.180 | 0.1858 | 0.0000 |
| Breakfast | v9 | 8 | SegMaskSam3MVOpt | 27815 | 4994 | 0.180 | 0.1844 | 0.0000 |
| Breakfast | v9 | 26 | SegMaskNew1 | 8907 | 2110 | 0.187 | 0.1875 | 0.5072 |
| FlameSteak | cam01 | 19 | SegMaskSam3OneStage | 4123 | 893 | 0.174 | 0.1736 | 0.6969 |
| FlameSteak | cam01 | 19 | SegMaskSam3MVOpt | 4123 | 893 | 0.174 | 0.1748 | 0.6981 |
| FlameSteak | cam01 | 61 | SegMaskNew1 | 366265 | 122801 | 0.292 | 0.2934 | 0.5506 |
| FlameSteak | cam01 | 61 | SegMaskSam3OneStage | 366265 | 125424 | 0.295 | 0.2947 | 0.5605 |
| FlameSteak | cam01 | 61 | SegMaskSam3MVOpt | 366265 | 125424 | 0.295 | 0.2949 | 0.5611 |
| MATF | S1_CAM_10 | 10 | SegMaskNew1 | 514 | 1801 | 0.201 | 0.2004 | 0.8591 |
| PoznanStreet | v4 | 16 | SegMaskNew1 | 12007 | 3555 | 0.160 | 0.0971 | 0.3414 |
| PoznanStreet | v8 | 16 | SegMaskNew1 | 14238 | 2463 | 0.132 | 0.1080 | 0.3546 |
| Welder | camera_0003 | 4 | SegMaskSam3OneStage | 477 | 112 | 0.235 | 0.2224 | 0.9288 |
| Welder | camera_0003 | 4 | SegMaskSam3MVOpt | 477 | 112 | 0.235 | 0.3608 | 0.9499 |
| Welder | camera_0003 | 6 | SegMaskNew1 | 432631 | 55735 | 0.125 | 0.1273 | 0.3645 |
| Welder | camera_0004 | 12 | SegMaskSam3OneStage | 14345 | 2633 | 0.184 | 0.1492 | 0.8746 |
| Welder | camera_0004 | 12 | SegMaskSam3MVOpt | 14345 | 2633 | 0.184 | 0.1369 | 0.8586 |

## gt_empty (참고 — 퇴화 아님)

- **SegMaskNew1**: 3개 (seed가 0 px가 아닌 것 0개) — Breakfast/v7/16, Breakfast/v9/16, PoznanStreet/v8/2
- **SegMaskSam3OneStage**: 3개 (seed가 0 px가 아닌 것 0개) — Breakfast/v7/16, Breakfast/v9/16, PoznanStreet/v8/2
- **SegMaskSam3MVOpt**: 3개 (seed가 0 px가 아닌 것 0개) — Breakfast/v7/16, Breakfast/v9/16, PoznanStreet/v8/2

## 판독 — P5의 기대 이득

- **SegMaskNew1**: 퇴화 seed 19개 / 도달 가능 객체 954개. 복구 목표 0.8434 → 0.8558 (+0.0123), 프롬프트 ≥64 px만 0.8507 (+0.0073), 상한 0.8566 (+0.0131). 데이터셋별(저장 → 복구 → 상한): Breakfast 10개(프롬프트 <64 px 7) 0.7138 → 0.8118 → 0.8191; MATF 3개 0.6649 → 0.7007 → 0.7031; CoffeeMartini 2개 0.8380 → 0.8516 → 0.8519; Blocks 1개 0.7488 → 0.7617 → 0.7621; Barn 1개 0.8569 → 0.8670 → 0.8677; Painter 1개 0.8713 → 0.8786 → 0.8791; FlameSteak 1개 0.7832 → 0.7905 → 0.7907.
- **SegMaskSam3OneStage**: 퇴화 seed 20개 / 도달 가능 객체 954개. 복구 목표 0.8427 → 0.8568 (+0.0141), 프롬프트 ≥64 px만 0.8517 (+0.0091), 상한 0.8577 (+0.0150). 데이터셋별(저장 → 복구 → 상한): Breakfast 8개(프롬프트 <64 px 7) 0.7419 → 0.8200 → 0.8259; Welder 2개 0.8419 → 0.8818 → 0.8840; AlexaMeadeExhibit 3개 0.8618 → 0.8922 → 0.8941; MATF 2개 0.6840 → 0.7035 → 0.7051; Blocks 1개 0.7484 → 0.7616 → 0.7619; Painter 1개 0.8682 → 0.8806 → 0.8810; CoffeeMartini 2개 0.8461 → 0.8576 → 0.8579; FlameSteak 1개 0.7736 → 0.7808 → 0.7811.
- **SegMaskSam3MVOpt**: 퇴화 seed 20개 / 도달 가능 객체 954개. 복구 목표 0.8451 → 0.8593 (+0.0143), 프롬프트 ≥64 px만 0.8543 (+0.0092), 상한 0.8602 (+0.0151). 데이터셋별(저장 → 복구 → 상한): Breakfast 8개(프롬프트 <64 px 7) 0.7421 → 0.8202 → 0.8262; Welder 2개 0.8437 → 0.8839 → 0.8860; AlexaMeadeExhibit 3개 0.8618 → 0.8921 → 0.8940; MATF 2개 0.6841 → 0.7036 → 0.7052; CoffeeMartini 2개 0.8440 → 0.8573 → 0.8576; Blocks 1개 0.7478 → 0.7609 → 0.7613; Painter 1개 0.8681 → 0.8805 → 0.8810; FlameSteak 1개 0.7734 → 0.7806 → 0.7809.

**Welder 앵커 재현.** `SegMaskSam3MVOpt`의 camera_0004에서 obj 8/14는 4 px / 7 px seed(GT 13,268 / 18,899 px)로 tiny / tiny, J 0.00 / 0.00. obj 12는 2,633 px, IoU 0.184, J 0.14 — REPORT.md는 이를 "어긋난" seed라 했고, 이 census의 규칙(IoU < 0.10)으로는 **ok**입니다 (IoU 0.184보다 높은 임계값이면 misaligned로 잡힙니다). camera_0004의 J&F는 `SegMaskNew1` 0.8668 vs `SegMaskSam3MVOpt` 0.7456(REPORT.md의 0.861 vs 0.707은 J&F가 아니라 J: 0.8613 vs 0.7071), 데이터셋 격차 -0.0341 중 이 카메라의 기여가 -0.0387 — "Welder 손실 전부가 camera_0004"는 성립합니다 (camera_0003에서는 오히려 `SegMaskSam3MVOpt`가 앞섭니다).

**P5의 기대 이득.** `SegMaskNew1`: 퇴화 19개는 도달 가능 객체 954개의 2.0%, 데이터셋 15개 중 8개는 0개. 복구 목표 이득 +0.0123 중 +0.0051는 ref 프롬프트 <64 px 객체의 몫이라 P5가 닿을 수 있는 이득은 +0.0073, 기여 상위: MATF +0.0024, Breakfast +0.0015, CoffeeMartini +0.0009. `SegMaskSam3OneStage`: 퇴화 20개는 도달 가능 객체 954개의 2.1%, 데이터셋 15개 중 7개는 0개. 복구 목표 이득 +0.0141 중 +0.0051는 ref 프롬프트 <64 px 객체의 몫이라 P5가 닿을 수 있는 이득은 +0.0091, 기여 상위: Welder +0.0027, AlexaMeadeExhibit +0.0020, MATF +0.0013. `SegMaskSam3MVOpt`: 퇴화 20개는 도달 가능 객체 954개의 2.1%, 데이터셋 15개 중 7개는 0개. 복구 목표 이득 +0.0143 중 +0.0051는 ref 프롬프트 <64 px 객체의 몫이라 P5가 닿을 수 있는 이득은 +0.0092, 기여 상위: Welder +0.0027, AlexaMeadeExhibit +0.0020, MATF +0.0013. ref 프롬프트 <64 px 객체: Breakfast obj 16(ref v5 GT 7 px; v5 empty(`SegMaskNew1`), v5 missing(`SegMaskSam3OneStage`·`SegMaskSam3MVOpt`)), Breakfast obj 19(ref v5 GT 2 px; v5/v7/v9 empty(`SegMaskNew1`), v5/v7/v9 missing(`SegMaskSam3OneStage`·`SegMaskSam3MVOpt`)), Breakfast obj 24(ref v5 GT 6 px; v5/v7/v9 empty(`SegMaskNew1`), v5/v7/v9 missing(`SegMaskSam3OneStage`·`SegMaskSam3MVOpt`)). 이들은 tracker 입력 해상도에서 프롬프트 자체가 사라져 donor 뷰가 있을 수 없으므로 P5의 재전파가 아니라 프롬프트 해상도(또는 라벨 자체)의 문제입니다. `SegMaskSam3MVOpt`에서 P5가 닿는 이득의 출처. 첫째: Welder camera_0004 obj 8/14 tiny — 데이터셋 +0.0402 = 헤드라인 +0.0027; `SegMaskNew1` 대비 Welder 격차 -0.0341 → 복구 후 +0.0061 (소거); REPORT.md P5의 "Welder 0.8437 → ~0.879, 헤드라인 +0.0023~0.0027"와 비교. 둘째: AlexaMeadeExhibit camera_0004 obj 17/21/24 missing — 데이터셋 +0.0304 = 헤드라인 +0.0020; `SegMaskNew1` 대비 AlexaMeadeExhibit 격차 -0.0296 → 복구 후 +0.0008 (소거). `SegMaskNew1`도 같은 종류의 seed 실패를 19개(empty 13, misaligned 6) 갖고 있고, 그중 empty+missing 13개 가운데 11개는 `SegMaskSam3MVOpt`의 missing+empty 15개와 같은 (카메라, 객체)입니다. 복구 목표(프롬프트 ≥64 px)에서는 저장값 순위 `SegMaskSam3MVOpt` > `SegMaskNew1` > `SegMaskSam3OneStage`가 `SegMaskSam3MVOpt` > `SegMaskSam3OneStage` > `SegMaskNew1`로 바뀝니다. misaligned 임계값을 0.10에서 0.20~0.30(으)로 올리면 퇴화 수가 늘고(`SegMaskNew1` 19 → 32, `SegMaskSam3OneStage` 20 → 30, `SegMaskSam3MVOpt` 20 → 30) 복구 목표도 위 민감도 표만큼 오르지만(Welder camera_0004 obj 12, IoU 0.184 부류), 그 seed들은 정의상 tiny가 아니라 면적 검사로는 잡히지 않아 P5의 감지기가 IoU 계열 신호(donor 마스크와의 IoU)를 함께 봐야 합니다. unreachable은 1031개 scored 객체 중 77개로 모든 방법에 동일하며 P5가 아니라 P8(reference 규칙)의 몫입니다.
