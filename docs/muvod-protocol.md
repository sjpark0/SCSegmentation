# MUVOD 평가 프로토콜 채택

우리 데이터셋은 MUVOD 벤치마크가 배포한 것이고, 그 논문에 기존 방법의 J&F가 이미 실려 있습니다.
그래서 자체 평가 규약을 버리고 **MUVOD의 평가 방식으로 갈아탑니다.** 이 문서는 그 규약이 무엇이고,
논문이 적어두지 않은 부분을 우리가 어떻게 확정했는지를 남깁니다.

- 논문: Ashkani Chenarlogh 외, *MUVOD: A Novel Multi-view Video Object Segmentation Dataset
  and A Benchmark for 3D Segmentation*, arXiv:2507.07519.
- 데이터: <https://volumetric-repository.labs.b-com.com/#/muvod> (등록 없이 내려받힙니다.)
- 공개된 코드는 없습니다. 평가 스크립트도, 장면별 카메라 지정표도 배포본에 들어 있지 않습니다.

---

## 1. 논문이 정한 것

| 항목 | 논문의 표현 |
|---|---|
| 채점 카메라 | "evaluation is conducted on videos from three selected cameras: the initial camera c_ini and two additional cameras" |
| 기준 카메라 | "we select an initial camera c_ini positioned near the center of the rig, providing comprehensive coverage of the dynamic objects and maximizing visibility of others" |
| basic 평가 | "The scores are calculated only with the objects visible on the input reference frame" |
| complete 평가 | "The complete evaluation takes all the labelled objects into consideration" |
| 집계 | J&F³ = Σ (J&F)_ci / N, N = 3 |
| 프레임 | 영상당 30프레임 주석 중 수작업으로 다듬은 21프레임 |
| 기존 방법 | XMem을 공간축과 시간축 양쪽에 적용 |
| 결과 | basic 전체 79.4 %, complete 전체 75.6 % (17개 장면) |

우리 쪽 구현은 `eval/report_jf.py --muvod`입니다. 카메라별로 객체 평균을 내고, 세 카메라를 평균해
장면 점수를 만들고, 장면을 평균해 전체를 냅니다(`--aggregation sequence`). J와 F 자체는 DAVIS 정의
그대로이고 이 저장소에서 2026-09-02 이후 한 줄도 바뀌지 않았습니다.

---

## 2. 논문이 정하지 않은 것: 장면별 c_ini

논문은 c_ini를 **이름으로 밝히지 않습니다.** 그런데 basic 평가의 객체 집합이 c_ini의 첫 프레임으로
정해지므로, 이걸 모르면 비교 자체가 성립하지 않습니다. 그래서 세 가지 증거를 모아 장면마다 하나씩
확정했고, 그 값을 `SCSam3/demo/MVSeg.json`의 `c_ini` 항목에 적어 두었습니다.

### 증거 A — 배포본이 알려주는 리그 형상

`muvod.json`(배포 색인)에 장면마다 `rig` 항목이 있습니다. 직선·호·평면 리그에서는 주석된 세 카메라 중
**번호가 가운데인 것**이 곧 리그 중앙입니다. 예를 들어 Carpark는 9대 직선 리그에 0·4·8이 주석돼 있어
4번이 정확히 중앙이고, Painter는 4×4 평면 16대에 0·6·15가 주석돼 있어 6번이 중앙에 가장 가깝습니다.

### 증거 B — 원본 데이터셋의 실제 카메라 좌표

번호가 공간 순서와 무관한 리그가 여섯 장면 있습니다. 이들은 원본 데이터셋의 보정 파일을 직접 받아
좌표로 판정했습니다(아카이브 전체가 아니라 HTTP range 요청으로 해당 파일만 수 KB씩 받았습니다).

- **구글 반구 리그 4장면**(AlexaMeadeExhibit, AlexaMeadeFacePaint, Dog, Welder) — Immersive Light
  Field Video의 `models.json`. 46대가 극점에서 바깥으로 고리를 그리며 번호가 매겨집니다:
  `camera_0001`이 리그 축 위 극점(0.7°), 0002~0006이 21° 고리, 0007·0009 등 홀수가 38° 고리,
  0008 등 짝수가 44° 고리입니다. 즉 **가운데 번호 규칙이 여기서는 틀립니다** — AlexaMeadeFacePaint의
  주석 카메라 0007·0008·0009 중 중앙에 가장 가까운 것은 0008이 아니라 0007입니다.
- **메타 2행 평면 리그 2장면**(CoffeeMartini, FlameSteak) — Neural 3D Video의 `poses_bounds.npy`.
  cam00이 중앙이지만 테스트용으로 제외된 카메라라 주석 대상이 아니고, 남은 것 중 중앙에 가장 가까운
  것은 윗줄 가운데인 **cam16**입니다. 주석된 셋 중 2.6~3.1배 차이로 앞섭니다. 여기서도 가운데 번호
  규칙(cam10)은 오히려 셋 중 가장 먼 쪽에 가깝습니다.

### 증거 C — 논문 표가 스스로 거는 제약 ★

Table III(basic)와 Table IV(complete)의 값이 **같은** 장면이 여덟 개 있습니다. 두 평가의 차이는
객체 집합뿐이므로, 값이 같다는 것은 **c_ini의 첫 프레임이 그 장면의 모든 라벨 객체를 이미 담고 있다**는
뜻입니다. 우리 정답 마스크로 카메라별 첫 프레임 객체 집합을 세어 보면 이 조건을 만족하는 카메라가
장면에 따라 하나뿐인 경우가 있고, 그때 c_ini가 유일하게 결정됩니다.

| 장면 | basic = complete? | 모든 객체를 담은 카메라 | 결론 |
|---|---|---|---|
| AlexaMeadeExhibit | 예 (82.8) | `camera_0001`만 (33/33) | **c_ini = camera_0001**, 증거 B와 일치 |
| MartialArts | 예 (83.7) | `v9`만 (19/19) | **c_ini = v9**, 증거 A와 일치 |
| PoznanStreet | 예 (80.2) | `v4`만 (24/24) | **c_ini = v4**, 증거 A와 일치 |
| Welder | 예 (85.2) | `camera_0001`, `camera_0003` | 증거 B가 극점인 0001로 좁힘 |
| Barn·Carpark·Fencing·Frog | 예 | 셋 다 | 제약 없음, 증거 A를 따름 |
| Dog | **아니오** (75.5 / 67.2) | `camera_0003` | **c_ini ≠ camera_0003** — 0002 또는 0004 |

세 장면에서 서로 독립적인 증거 두 개가 같은 카메라를 가리킵니다. 이것이 c_ini 지정의 주된 근거입니다.

### 남은 불확실성 — Dog

Dog는 주석 카메라 0002·0003·0004가 모두 같은 21° 고리에 있어 좌표로 우열을 가릴 수 없습니다
(장면마다 순서가 뒤바뀔 정도의 차이입니다). 다만 증거 C가 **0003을 배제**합니다. 0003이 c_ini였다면
basic과 complete가 같아야 하는데 논문은 75.5와 67.2로 다르게 적고 있기 때문입니다. 남은 둘 중
Dog 장면 자체의 좌표에서 중심에 조금 더 가까운 **camera_0002**를 채택하고, `camera_0004`로도 한 번
더 돌려 감도 확인 결과를 함께 싣습니다.

### 최종 표

| 장면 | 리그 | 주석 카메라 | c_ini | 근거 |
|---|---|---|---|---|
| AlexaMeadeExhibit | 반구 45 | 1, 3, 4 | **1** | B + C(유일) |
| AlexaMeadeFacePaint | 반구 46 | 7, 8, 9 | **7** | B(38.4° 대 44.1°) |
| Barn | 5×3 평면 15 | 0, 7, 10 | **7** | A(격자 중앙) |
| Blocks | 10 직선호 | 0, 4, 9 | **4** | A |
| Breakfast | 5×3 평면 15 | 5, 7, 9 | **7** | A(격자 중앙) |
| CBABasketball | 직선호 30 | 6, 20, 24 | **20** | A |
| Carpark | 9 직선 | 0, 4, 8 | **4** | A(정중앙) |
| CoffeeMartini | 2행 평면 18 | 2, 10, 16 | **16** | B(cam00 제외 후 최근접) |
| Dog | 반구 41 | 2, 3, 4 | **2** | C가 3을 배제, B로 2 선택 (불확실) |
| Fencing | 10 직선호 | 0, 4, 9 | **4** | A |
| FlameSteak | 2행 평면 21 | 1, 10, 16 | **16** | B |
| Frog | 13 직선 | 4, 7, 10 | **7** | A |
| MATF | 10 스테레오 직선 | 1, 4, 10 | **4** | A |
| MartialArts | 이중호 15 | 5, 9, 13 | **9** | C(유일) + A |
| Painter | 4×4 평면 16 | 0, 6, 15 | **6** | A |
| PoznanStreet | 9 직선 | 0, 4, 8 | **4** | C(유일) + A |
| Welder | 반구 41 | 1, 3, 4 | **1** | B + C |

---

## 3. 우리가 쓰던 기준 카메라와 무엇이 다른가

기존 러너는 `pick_reference`로 **첫 프레임의 객체 번호 최댓값이 가장 큰 카메라**를 골랐고, 동점이면
`cam_list`의 첫 항목이 이겼습니다. 이 규칙은 17개 장면 중 **14개에서 c_ini와 다른 카메라**를 고릅니다.
동점이 잦아 사실상 "번호가 가장 작은 카메라", 즉 리그 가장자리를 고르는 일이 많았습니다.

그래서 기준 카메라를 실행 시점에 지정할 수 있게 했습니다.

```bash
python SCSam3/runMVSeg.py <장면> --xview-window 1 --xview-gate --xview-ptr \
       --xview-tpos-shift 4 --track-cams closure --ref-cam muvod
```

`--ref-cam muvod`는 설정의 `c_ini`를 읽고, 출력 폴더 이름 끝에 **M**을 붙입니다(`SegMaskSam3XW1GPS4M`).
장면마다 c_ini의 순위가 달라도 폴더 이름은 하나로 유지되므로, 벤치마크 전체를 한 이름으로 채점할 수
있습니다. `--ref-cam center`는 정렬한 `cam_list`의 가운데 항목, `--ref-cam <번호>`는 그 카메라이고
둘 다 `R<순위>` 접미사를 씁니다. 플래그를 주지 않으면 예전 규칙 그대로이고 출력도 예전 그대로입니다.

## 4. 채점

```bash
docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \
  python eval/eval_jf.py --methods SegMaskSam3XW1GPS4M SegMaskSam3XW0M \
    --out Data/MVSeg/jf_muvod.json
python3 eval/report_jf.py --raw jf_muvod.json --muvod
```

`--muvod`는 프로토콜을 고정합니다 — 카메라 평균 후 장면 평균, 기준 카메라는 c_ini, 21프레임 전부.
이 설정을 바꾸는 플래그를 같이 주면 실행을 거부합니다. 출력은 basic 표와 complete 표, 그리고 장면별
객체 집합 표입니다. 마지막 표의 마지막 두 열은 **반드시 일치해야 합니다** — 우리 데이터에서 basic과
complete 객체 집합이 같은 장면은, 논문에서도 두 점수가 같아야 하기 때문입니다. 어긋나면 그 장면의
c_ini가 틀린 것입니다.

## 5. 버리는 것

기존 자체 규약(pooled 집계, ref/nonref 분리, 면적 가중, ceiling 열, 12개 장면 부분집합)은 내부 절제
실험을 비교할 때만 씁니다. 외부와 견주는 수치는 전부 `--muvod`로 냅니다.
