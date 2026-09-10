> **2026-09-04 상태 변경.** `demoSCSam3OneStageNew`는 이제 `demoSCSam3MVOpt`의 바이트 동일
> 동결 스냅샷입니다. 아래에서 "OneStageNew"가 *무패치 원본 코드*를 뜻하는 서술은
> **git 태그 `baseline-onestagenew`** 기준으로 읽으십시오. 배경: [README.md](README.md)

# 실행 — 방법과 함정

## 하드웨어

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 6000 Ada, **49140 MiB (~48 GiB)**, cc 8.9, 1대 |
| 드라이버 / CUDA | 580.173.02 / 13.0 |
| 호스트 RAM | **125 GiB**, 스왑 8 GiB |

GPU 48 GiB가 상한입니다. SAM 3의 무거운 데이터셋은 45~46 GiB에서 터집니다.

## 실행 스크립트

| 스크립트 | 용도 | 메모리 상한 |
|---|---|---|
| `runMVSegAll.sh` | 전체 스윕 | **없음 ★** |
| `runMVOptThree.sh` | 데이터셋 지정 실행. `ALGO`·`OUT`·`XVIEW`·`TRACK`·`LOGTAG` 환경변수로 알고리즘/출력/교차시점 창/추적 모드/로그 폴더 지정 | `--memory=90g` |
| `runForSam2Three.sh` | ForSam2New 판 | `--memory=90g` |
| `waitAndRunMVSeg.sh` | GPU가 빌 때까지 대기 후 스윕. Frog 스모크 테스트 먼저 | — |
| `launch_container.sh` | X11 대화형 셸 | **없음 ★** |

**★ 상한 없는 두 스크립트를 무거운 데이터셋에 쓰지 마십시오.** 호스트 RAM 고갈로 머신이 두 번 재부팅됐습니다.
`--memory=90g --memory-swap=90g`는 스왑을 0으로 만들어 cgroup OOM killer가 90 GiB에서 발동하게 합니다 —
머신이 죽는 대신 컨테이너만 죽습니다.

한 데이터셋 = 한 컨테이너입니다. 실패가 번지지 않고, GPU 메모리가 확실히 반환됩니다.

```bash
# 특정 알고리즘으로 특정 데이터셋
ALGO=MVOpt OUT=SegMaskSam3MVOpt ./runMVOptThree.sh Blocks Painter

# 컨테이너가 만든 파일 소유권 복구 (스크립트가 자동으로 함)
docker run --rm -v /:/host scsam3 chown -R "$(id -u):$(id -g)" "/host$PWD/Data/MVSeg"
```

### 마스크 폴더 매니페스트 (`MANIFEST.json`)

`Data/`는 git 제외 대상이라 마스크 폴더의 출처 정보가 mtime뿐이었습니다 (REPORT.md P13). 2026-09-07부터
`<데이터셋>/<method>/MANIFEST.json`이 그 역할을 합니다 — **method 폴더 바로 아래의 파일 하나**이고,
카메라 폴더 안에는 아무것도 들어가지 않습니다. 도구는 `eval/manifest.py`(호스트 python, 표준 라이브러리만).

- **내용**: 카메라 목록, 카메라별 프레임 범위·개수, PNG 수, `content_digest`
  (PNG 전체의 (상대경로, sha256)를 정렬해 다시 sha256 — mtime·`MANIFEST.json` 자체·PNG가 아닌 파일에 무관),
  PNG mtime 범위, `provenance`(git rev·dirty, `algo`, `argv`, `track_cams`,
  `SPATIAL_START_IMPLICIT`·`SCSAM3_TRIM_CACHED_OUTPUTS`, 도커 이미지 ID, 패키지 `*.py`의 `source_digest`,
  torch/cuda, 호스트명) + `provenance_basis`(실행 시 기록인지 사후 추정인지, 근거).
- **자동 기록**: `runMVSeg.py`가 `done -> <out_dir>`를 찍은 **뒤에** 씁니다. 마스크 출력에는 영향이 없고,
  실패해도 실행은 실패하지 않으며 `manifest not written (...)`만 찍습니다. `--dry-run`에서는 쓰지 않습니다.
  이미지 ID는 실행 스크립트 4개가 `-e SCSAM3_IMAGE_ID=$(docker images --no-trunc -q scsam3:latest | head -n1)`로
  넘깁니다 — `docker run`을 손으로 치면 `"unknown"`으로 남습니다.
- **기존 폴더**: 2026-09-07에 `eval/manifest.py sweep --legacy`로 15개 데이터셋 × 8 method(117개)를 채웠습니다.
  SAM 3 폴더의 provenance는 로그·mtime·git 이력에서 **추정**한 값이고(`recorded_at_run_time: false`,
  근거는 `provenance_basis`에 인용), SAM 2 폴더(`SegMask1`, `SegMaskNew1~3`)는 거의 전부 `unknown`입니다.
  같은 스윕에서 `SegMaskSam3MVOpt`와 `SegMaskSam3OneStageNew`의 다이제스트가 12개 데이터셋 전부 같음을 확인했습니다.

```bash
python3 eval/manifest.py show   Data/MVSeg/Barn/SegMaskSam3MVOpt     # 보기
python3 eval/manifest.py verify Data/MVSeg/Barn/SegMaskSam3MVOpt     # 다이제스트 재계산, 불일치면 exit 1
python3 eval/manifest.py write  Data/MVSeg/Barn/SegMaskSam3XW0 \
        --live --package SCSam3/demoSCSam3MVOpt --algo MVOpt          # 수동 실행 뒤 직접 기록
python3 eval/manifest.py sweep  --root Data/MVSeg --methods SegMaskSam3MVOpt SegMaskSam3XW0   # 있는 폴더 전부
```

**다른 도구와의 공존.** `eval_jf.py`는 `<method>/<카메라>/`만, `report_jf.py`의 `find_absent`는
`<method>/<카메라>/<프레임>/`만 나열하므로 method 폴더에 놓인 파일은 보지 못합니다
(2026-09-07 확인: 매니페스트 117개를 쓴 전후로 `report_jf.py` 출력이 동일, 컨테이너에서 돌린 `eval_jf.py`의
Frog 점수가 `jf_sam3_mvopt_all.json`과 일치). `waitAndRunMVSeg.sh`·`runMVSegAll.sh`·`runMVOptThree.sh`는
폴더 내용을 읽지 않고, 끝의 `chown -R`이 매니페스트까지 덮습니다.
두 가지만 주의하십시오:

- 폴더를 **바이트 비교할 때는 `diff -rq -x MANIFEST.json A B`** — 매니페스트는 `written_at`·`path`가
  달라 항상 차이로 잡힙니다.
- `runMVSeg.py`의 "이미 존재함" 가드는 매니페스트만 남은 폴더도 비어 있지 않다고 보므로 `--overwrite`가 필요합니다.

## 러너 등록표

`SCSam3/runMVSeg.py`:

| `--algo` | 폴더 | 기본 출력 | 시점 추적 기본값 |
|---|---|---|---|
| `OneStage` | `demoSCSam3OneStage` | `SegMaskSam3OneStage` | `written` |
| `OneStageNew` | `demoSCSam3OneStageNew` | `SegMaskSam3OneStageNew` | **`all`** |
| `MVOpt` | `demoSCSam3MVOpt` | `SegMaskSam3MVOpt` | **`all`** |
| `MVOpt --xview-window W` (0..6) | `demoSCSam3MVOpt` | `SegMaskSam3XW{W}` (`--track-cams all`이면 `SegMaskSam3XW{W}all`) | **`closure`** |
| `MVOpt --xview-hygiene` | `demoSCSam3MVOpt` | `SegMaskSam3XW4` | **`closure`** |
| `MVOpt --xview-window W --xview-mode M` (M ∈ B,C,D,E) | `demoSCSam3MVOpt` | `SegMaskSam3XW{W}{M}` | **`closure`** = 의존 원뿔 (아래 함정 10) |
| `OneStage --track-cams closure` | `demoSCSam3OneStage` | `SegMaskSam3OneStageC` (실험 행 2) | `closure` |

`NEEDS_ALL_VIEWS = ("OneStageNew", "MVOpt")`. `--track-cams closure`는 카메라 `0..max(채점 시점 인덱스)`만 세션을 엽니다.

### XW 계열 — 교차시점 창 파라미터와 위생 수정 (2026-09-07, Phase 2)

`demoSCSam3MVOpt`의 트래커에 생성자 인자 `cross_view_window`(기본 4)·`cross_view_hygiene`(기본 off)가 생겼고,
교차시점 수집 루프는 `xview_gather.py`의 순수 함수입니다. **플래그 없이 돌리면 발표본(`SegMaskSam3MVOpt`)과 바이트 동일**합니다
(legacy 경로는 음수 인덱스 wrap·`IndexError`·`selected_cond_outputs` 덮어쓰기(REPORT C1/C2)까지 그대로 재현).

- `--xview-window W`: 이웃 창 W(0 = 이웃 메모리 없음, 패키지 내 대조군), **위생 수정 2건 자동 on**
  (음수 인덱스 건너뜀 → wrap·IndexError 없음 / cond pointer 덮어쓰기 제거), 추적 기본값 `closure`, 출력 `SegMaskSam3XW{W}`.
  W ≠ 4는 위생 없이 허용되지 않습니다(다른 이웃 집합으로 wrap하므로).
- `--xview-hygiene`: W=4에 위생만 켠 것(= `--xview-window 4`).
- **거부되는 조합** (`sys.exit`): OneStage/OneStageNew에 `--xview-*`; MVOpt에 위생 없는 `--track-cams closure`(legacy는 wrap 때문에
  closure ≠ all); XW + `--track-cams written`(이웃이 카메라 인덱스로 정의됨); XW 출력을 기본 폴더(`SegMaskSam3MVOpt` 등)에;
  legacy 실행을 `SegMaskSam3XW*` 이름에; `SCSAM3_XVIEW_*` 환경변수만 있고 플래그 없음(함정 9).
- **공간축 메모리 손잡이** (2026-09-08, P12): `--xview-gate`(이웃 메모리에도 자기 프레임과 같은 `eff_iou_score > 0.01` 검사 —
  단 must-include 없음, 점수 키가 없으면 통과) · `--xview-ptr`(이웃의 `obj_ptr`을 자기 포인터 뒤에 추가) ·
  `--xview-tpos-shift S`(이웃 토큰의 시간 위치 행을 S만큼 밀어 "덜 최근"으로; `1..5`, 경계 W+S ≤ 6).
  폴더 이름 문법은 `SegMaskSam3XW{W}{모드}{G}{P}{S<s>}[all]`입니다(예: `SegMaskSam3XW1GP`, `SegMaskSam3XW1S2`).
  거부: XW 플래그 없이 손잡이만 · W=0에 손잡이 · W+S > 6 · `--xview-tpos-shift 0`(argparse) · OneStage/OneStageNew에 손잡이.
- 실행 시 모델이 실제로 든 값을 읽어 `cross-view     window W, hygiene H, mode M, gate G, ptr P, tpos-shift S`로 찍고,
  명령줄과 다르면 종료합니다. 매니페스트에는 `lineage`, `xview_window`, `xview_hygiene`, `xview_mode`, `xview_gate`,
  `xview_ptr`, `xview_tpos_shift`, `xview_gate_stats`(게이트 진단), `track_cams_requested`, `track_idx`, `n_sessions`가
  **`provenance` 블록 아래에** 기록됩니다(최상위가 아닙니다).
- **기준 카메라 `--ref-cam`** (2026-09-09, MUVOD): 정답 마스크를 어느 카메라에 주고 시작할지를 정합니다. **채점 옵션이 아니라
  실행 자체를 바꿉니다.** `muvod`는 `MVSeg.json`의 `c_ini`(장면별로 리그 중앙 카메라, 근거는
  [muvod-protocol.md](muvod-protocol.md)), `center`는 정렬한 `cam_list`의 가운데, 숫자는 그 카메라입니다.
  자동 폴더 이름에 접미사가 붙습니다 — `muvod`는 **M**, 나머지는 `R<순위>`. 장면마다 c_ini의 순위가 달라도 M은 하나로
  유지되므로 벤치마크 전체를 한 method 이름으로 채점할 수 있습니다. 플래그가 없으면 예전 `pick_reference` 규칙 그대로이고
  출력도 예전 그대로입니다. `runMVOptThree.sh`는 `XREF`로 넘깁니다.
- **시점별 면적 계측 `view_areas`** (2026-09-09, 항상 켜짐): 매니페스트 `provenance.view_areas`에
  `seed`(불러온 모든 시점의 첫 프레임 마스크 면적)와 `tracked`(추적하는 모든 시점의 프레임별 객체 면적)를
  남깁니다. **채점 카메라 셋만이 아니라 추적 세션 전부**입니다 — 오늘 확인한 피해가 채점되지 않는 시점에서
  왔고(Blocks cam8, CBABasketball v05) 그 시점의 마스크는 디스크에 없기 때문입니다. 나중에 "이 이웃이
  건강한가"를 묻는 실험을 GPU 없이 검증하려면 이 궤적이 필요합니다. **순수 기록이라 출력은 바뀌지
  않습니다**(Fencing 504장 PNG 바이트 동일로 확인). 크기는 장면당 20~30 KB. 모드 E는 2차 계산이 채점
  카메라만 내므로 `tracked`가 **1차 잠정값**입니다 — 같은 매니페스트의 `two_pass`로 구분하십시오.
- **시드 복구 `--repair-seeds`** (2026-09-09, P5): 시점 전파가 만든 첫 프레임 마스크 중 비정상적으로 작은
  (시점, 객체) 쌍을 **정답 없이** 찾아, 옆 시점 마스크를 두 번째 조건 프레임으로 넣고 **새 시점 세션**에서
  다시 전파해 그 쌍만 바꿔 넣습니다. 규칙·판정은 [phase5-seed-repair-prereg.md](phase5-seed-repair-prereg.md)
  §1에 고정돼 있고 `demoSCSam3MVOpt/seed_repair.py`(torch 없음)가 감지·donor 선택을, 러너의
  `RepairSeeds()`가 세션 작업을 맡습니다. 폴더 접미사 **Rp**(예: `SegMaskSam3XW1GPS4MRp`). 시점 전파가 있는
  패키지(MVOpt·OneStageNew)에서만 받고 OneStage는 거부합니다. 실행 로그에 `seed repair    round N: ...`와
  쌍별 `전 px <- donor -> 후 px` 줄이 찍히고, 매니페스트 `provenance.seed_repair`에 전부 남습니다.
  **순서가 중요합니다** — `PropagateAcrossViews()` 뒤, `RetireSpatialPredictor()` 앞. 뒤에 두면 에러 없이
  조용히 아무것도 고치지 않습니다(테스트가 고정). `runMVOptThree.sh`는 `XREPAIR`로 넘깁니다.
- `runMVOptThree.sh`의 `XGATE`·`XPTR`·`XSHIFT`는 **비어 있지 않기만 하면 켜집니다** — `XGATE=0`도 플래그를 켭니다
  (`TRACK`과 같은 함정). `XSHIFT=0`은 `--xview-tpos-shift 0`을 만들어 argparse가 거부하므로 실행이 죽습니다.
  `XREF`는 값이 그대로 `--ref-cam`에 들어가므로 오타가 나면 실행이 죽습니다(무시되지 않습니다).
  `XREPAIR`도 `XGATE`처럼 비어 있지 않기만 하면 켜집니다 — `XREPAIR=0`도 복구를 켭니다.
- **closure == all 보조정리**: 위생 on이면 시점 v는 시점 0..v만 읽으므로 closure와 all의 출력이 같습니다(실험 행 6이 실측 검증).
  nb=0 카메라는 W에 무관하게 입력이 같으므로 XW0 vs XW4에서 정확히 0이어야 하며, 0이 아니면 비결정성·누수 신호입니다.
- **이웃 모드 `--xview-mode`** (2026-09-08, P4): A = 이전 시점의 프레임 t(기본, 폴더 이름에 글자 없음) · B = 양쪽 시점의 t−1 ·
  C = 이전 시점 t + 이후 시점 t−1 · D = 이전 시점 t−1 · E = 프레임당 2회 계산(1차 B, 2차 양쪽 시점의 t; 1차 출력은 버림).
  W=0에는 모드를 줄 수 없고(대조군), `--track-cams written`과 함께 쓸 수 없습니다. 이웃 토큰의 시간 위치 부호는 모든 모드에서
  `|offset|−1` 행으로 같습니다. 사전 등록·결과는 [phase3-neighbourhood.md](phase3-neighbourhood.md).

```bash
XVIEW=0 LOGTAG=xw0 ./runMVOptThree.sh Blocks              # SegMaskSam3XW0, closure
XVIEW=4 TRACK=all OUT=SegMaskSam3XW4all ./runMVOptThree.sh Welder
OUT=SegMaskSam3MVOpt_guard LOGTAG=guard ./runMVOptThree.sh Blocks Fencing   # 바이트 동일성 가드
diff -rq -x MANIFEST.json ../Data/MVSeg/Blocks/SegMaskSam3MVOpt ../Data/MVSeg/Blocks/SegMaskSam3MVOpt_guard   # 출력 0줄이어야 함
```

**가드 규칙.** `SCSam3TrackerPredictorNewMem.py`, `xview_gather.py`, `build_scsam3.py`, `SCSam3VideoPredictorNewMem.py`,
`SCSam3Video.py`, `runMVSeg.py` 중 하나라도 바뀌면 XW 폴더를 채점하기 전에 위 가드(Blocks·Fencing)를 다시 돌립니다.

### 테스트 (CPU, 컨테이너)

`tests/`에 pytest 188개가 있습니다 — 수집 함수 등가(무작위 20,000 구성), 실제 메서드 golden, closure 보조정리, MUVOD 프로토콜(객체 필터·집계·가드), 러너 해석표(17개 데이터셋),
집계기 CI 열, 메모리 불변식 S1~S7. 실제 트래커는 CPU에서 생성되지 않으므로(`PositionEmbeddingSine`이 cuda 할당) bare-instance 하니스를 씁니다.

```bash
cd /home/sjpark/Documents/SCSegmentation
docker run --rm --user $(id -u):$(id -g) -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH= \
  -v /:/host -w /host$PWD scsam3 python -m pytest tests -q -p no:cacheprovider
```

`SCSam3/` 안에서 python을 띄우지 마십시오(그 안의 `sam3` 체크아웃이 설치된 패키지를 가립니다).
`demoSCSam3ForSam2*`는 별도 러너 `runMVSegForSam2.py`, `demoSCSam3TwoStage*`는 **미등록**입니다.

---

# 함정

## 1. `--track-cams written`은 NewMem 계열에서 다른 모델이 됩니다 ★★

단순한 속도 옵션이 아닙니다. NewMem 트래커의 메모리 어텐션은 **리스트 인덱스 기준 앞 4개 시점**을 끌어옵니다
(`SCSam3TrackerPredictorNewMem.py:1262-1274`):

```python
for s_pos in range(-4, 0):
    prev_spatial_idx = spatial_idx + s_pos
```

음수 인덱스는 파이썬 규칙대로 **리스트 끝으로 감깁니다.** 경계 검사가 없습니다.
`written`을 쓰면 리스트에 채점 대상 3대만 남으므로, 각 시점이 나머지 두 대를 감아서 참조합니다 — **다른 모델입니다.**

`OneStage`는 이런 결합이 없어(단일 `session_id`) 기본값이 `written`입니다. 45대 데이터셋의 GPU OOM을 이걸로 해결했습니다.

## 2. 시점 전파 시작 인덱스를 명시하면 안 됩니다 ★★

`SPATIAL_START_IMPLICIT`(기본 1)을 끄지 마십시오. 근거는
[investigations-closed.md](investigations-closed.md) 4번, 메커니즘은 이렇습니다:

- 예측기가 읽는 키는 `start_frame_index`인데 데모 기반 클래스는 다른 키를 넘깁니다
  (`SCSam3Video.py:216-221`) — 즉 **원래부터 암묵적으로 동작해 왔습니다.**
- 처리 순서 자체는 양쪽이 같습니다. 차이는 **action history**입니다.
  `propagation_direction="both"`에서 역방향 패스가 `parse_action_history_for_propagation`에 다시 들어가는데,
  기록된 `frame_idx`가 `0`이나 `num_frames-1`이면 `propagation_fetch`로 격하됩니다
  (`SCSam3VideoInference.py:1265-1272`). `fetch`는 캐시만 읽으므로 계산된 적 없는 시점들은 빈 마스크가 됩니다.
- 암묵(기본)이면 기록값이 `None`이라 이 조건이 성립하지 않습니다.

**잠재 위험.** 현재 기준시점이 마지막 인덱스인 것은 Blocks·Painter뿐이지만,
`cam_list`에 마지막 인덱스가 들어 있는 데이터셋은 6개입니다 —
Blocks(9/10), Carpark(8/9), Fencing(9/10), MATF(9/10), Painter(15/16), PoznanStreet(8/9).
`pick_reference`가 다른 카메라를 고르게 되면 이들도 같은 방식으로 깨집니다.

**소스의 모순.** `PropagateAcrossViews`의 docstring(`runMVSeg.py:200-204`)은
"시작 시점을 여기서 명시한다"고 되어 있습니다 — 기본 동작과 **반대**입니다. docstring이 낡았습니다.

## 3. `sys.path` 수술은 필수입니다 ★

```python
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != HERE]
sys.path.insert(0, algo_dir)
os.chdir(algo_dir)
```

`SCSam3/sam3/`에 `__init__.py`가 없어 PEP 420 네임스페이스 조각이 됩니다.
컨테이너에서 `sam3`는 `/opt/sam3`에 editable 설치되어 있고, `_EditableFinder`가 `PathFinder` **뒤에** 등록됩니다.
따라서 `SCSam3`가 경로에 있으면 `PathFinder`가 먼저 빈 네임스페이스를 잡습니다:

```
sam3.__file__ = None
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

이 `TypeError`는 BPE 토크나이저를 찾는 `pkg_resources.resource_filename`에서 납니다.

- 필터의 `if p`는 빈 문자열(cwd) 항목도 제거합니다. **이것도 필요합니다** — `python runMVSeg.py`는 스크립트 디렉터리를 `sys.path[0]`에 넣습니다.
- `os.chdir`은 임포트 때문이 아니라 데모 스크립트들이 상대 경로를 쓰기 때문입니다
  (`demoSCSam3*/` 안의 숫자 폴더 `0`~`31`이 그 산물입니다).
  두 러너 모두 chdir 전에 자기 경로를 절대화하므로 안전하지만, **이후에 상대 경로를 추가하면 깨집니다.**

## 4. 요청 형태는 속성이 아니라 플래그로 판별합니다

두 예측기의 요청 키가 상호 배타적이라 잘못 고르면 `KeyError`입니다.

| 패키지 | `predictor_spatial` | `uses_spatial_predictor` | `RetireSpatialPredictor` |
|---|---|---|---|
| OneStage | 없음 | 없음 | **있음** |
| OneStageNew | 있음 | **있음** | **있음** | ← 2026-09-04 이후. 이전에는 `없음 / 없음`
| MVOpt | 있음 | **있음** | **있음** |

MVOpt는 폐기 시 `predictor_spatial = None`으로 만들기 때문에, 폐기 후에는 속성 검사가 **틀린 분기를 고릅니다.**
그래서 플래그를 봅니다. 2026-09-04 이후 OneStageNew도 플래그를 가지므로 같은 경로입니다
(그 이전에는 속성 대체 경로로 같은 분기를 탔습니다).

## 5. `spatial model retired` 로그는 거짓일 수 있습니다 ★

`runMVSeg.py:335-338`이 메서드 존재만 확인하고 무조건 출력합니다.

- **OneStage**는 `predictor_spatial`을 애초에 만들지 않아 메서드가 즉시 `return`합니다 —
  **아무것도 반환하지 않고 "retired"를 찍습니다.**
- **OneStageNew**는 2026-09-04 이전에 메서드 자체가 없어 약 3.2 GiB 교차시점 모델이 시간 추적 내내 상주했습니다.
  이것이 무거운 데이터셋에서 터지던 이유 중 하나였습니다. **지금은 메서드가 있어 실제로 폐기합니다** —
  Welder가 90 GiB 상한 안에서 완주하게 된 직접적 원인입니다.
- 실제로 회수하려면 `torch.clear_autocast_cache()`가 반드시 함께 호출되어야 합니다.
  autocast 캐시가 죽은 fp32 파라미터에 대한 약한 참조로 bf16 사본을 붙들고 있기 때문입니다.

## 6. `PYTORCH_CUDA_ALLOC_CONF`

파이썬에서만 설정되며, 항상 `expandable_segments:True`, 항상 torch 임포트 전입니다.
쉘 스크립트나 `docker run`에서는 설정하지 않습니다. 러너 두 개는 `setdefault`라 밖에서 덮어쓸 수 있습니다.

OOM의 원인이 아닙니다 — [investigations-closed.md](investigations-closed.md) 3번.

## 7. Docker 이미지와 빌드 — 토큰은 더 이상 쓰지 않습니다

`scsam3:latest`(`9064e3dc8548`, 2026-09-04 빌드)는 토큰 없는 빌드본입니다. `HF_TOKEN` 레이어 0개,
가중치는 `SCSam3/hf_cache/`를 `COPY`해서 들어갑니다. 토큰 레이어 3개를 갖고 있던 2026-02-25 이미지는
Phase 0(2026-09-07)에서 삭제했습니다. `scsam3:notoken`은 같은 이미지의 중복 태그입니다.

```bash
DOCKER_BUILDKIT=1 docker build -t scsam3 SCSam3/     # SCSam3/hf_cache/ 가 채워져 있으면 토큰 불필요
```

`hf_cache/`가 비어 있을 때만 `--secret id=hf_token,src=<파일>`로 내려받습니다 (루트 README 참조).
`SCSam3/hf_cache/*`는 gitignore 대상이고 `.gitkeep`만 추적됩니다 — 6.5 GB 가중치를 커밋하지 마십시오.

## 8. 워크플로 원본 출력(`docs/raw/*.json`)에 비밀값이 섞일 수 있습니다 ★

2026-09-07, `docs/raw/roadmap_raw.json`에 옛 이미지의 HF 토큰 전문이 그대로 들어간 채 커밋됐고
(에이전트가 `docker history` 출력을 인용), GitHub 푸시 보호가 푸시를 막았습니다.
푸시되기 전이라 로컬 커밋 4개를 `git filter-branch --index-filter`로 다시 써서 값을 `hf_ShVKFypk[REDACTED]`로 가렸습니다.
토큰은 원격에 간 적이 없습니다.

- **증상**: `! [remote rejected] main -> main (push declined due to repository rule violations)`.
  원인은 그 위 `remote: error: GH013 …` 블록에 커밋 해시와 `경로:행`으로 나옵니다. 태그까지 함께 거부됩니다.
- **대응**: "allow the secret" 링크는 쓰지 마십시오 — 토큰이 원격 이력에 영구히 남습니다.
  미푸시 커밋이면 다시 쓰고, 이미 푸시됐다면 토큰을 폐기하십시오.
- **예방**: `docs/tools/pre-commit-secrets.sh`가 스테이징된 추가 줄에서 토큰 패턴을 찾으면 커밋을 막습니다.
  훅은 저장소에 포함되지 않으므로 클론마다 설치해야 합니다:

  ```bash
  cp docs/tools/pre-commit-secrets.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
  ```

## 9. `SCSAM3_XVIEW_WINDOW` / `SCSAM3_XVIEW_HYGIENE` 환경변수는 데모 경로를 조용히 바꿉니다

`build_tracker_newmem`은 생성자 인자가 None일 때만 이 두 변수를 읽습니다(`sam3_demoVideo.py`처럼 인자 없이 `SCSam3Video(device)`를 부르는 경로용).
셸 프로파일에 남아 있으면 데모가 다른 모델로 돌고, `runMVSeg.py`는 플래그 없이 이 변수가 설정돼 있으면 **실행을 거부**합니다
(설정은 명령줄에서만 받습니다). 파싱은 엄격합니다: 빈 값은 미설정, `4`·`1`/`true`/`on`·`0`/`false`/`off`만 허용, 그 외는 `ValueError`.
`SCSAM3_TRIM_CACHED_OUTPUTS`와 달리 `"false"`가 on을 뜻하지 않습니다.

## 10. 양측 모드(B·C·E)의 closure는 "max(채점)+W"가 아니라 의존 원뿔입니다

이웃이 이후 시점에서도 오면 시점 v의 프레임 start+n은 시점 v+nW(E는 v+2nW)까지 의존합니다. 그래서 `--track-cams closure`는
모드별로 `0..min(N−1, max(채점 인덱스)+W·(num_frame−1))`(B·C), `+2W·(num_frame−1)`(E)로 계산됩니다(`runMVSeg.closure_reach`).
W=1이면 Welder·Dog·AlexaMeadeExhibit이 24세션(B·C)·44~41세션(E), FacePaint 29·46세션이라 Phase 2의 속도 이득은 대부분 사라집니다.
`max(채점)+1+W`로 자르면 채점 시점의 출력이 세션 수에 따라 달라지는 **다른 모델**이 됩니다(CPU 하니스로 확인). 원뿔 == all은 Welder·Dog에서 실측 검증합니다.

E 모드는 `runMVSeg.run_two_pass`만 사용합니다. 프레임 t의 1차 패스(모든 세션 `next()`) 뒤, 어떤 세션이든 t+1로 넘어가기 **전에**
`recompute_frame` 요청을 한 번 보내야 합니다 — 넘어간 뒤에는 t의 특징 캐시가 비워져 `RuntimeError`가 납니다(조용히 틀리지 않고 크게 실패하도록 둔 것).



## 11. `mount_*.sh`는 저장소에 들어가지 않습니다 ★

`mount_250620.sh`에 내부 호스트 주소와 SMB 자격증명이 평문으로 있었고, 2025-11-27부터
2026-09-09까지 공개 원격에 올라가 있었습니다. 지금은 `.gitignore`의 `mount_*.sh`가 막습니다.

- **로컬 파일은 지우지 마십시오.** 추적만 꺼져 있고 마운트에 그대로 씁니다.
- **자격증명 교체는 아직 안 됐습니다** — 이력에서 여전히 읽힙니다
  ([open-items.md](open-items.md) 0번).
- 같은 성격의 파일(자격증명·내부 주소가 든 셸 스크립트)을 새로 만들 때는 `.gitignore`를
  먼저 확인하십시오. 커밋 훅(`docs/tools/pre-commit-secrets.sh`)은 **토큰 모양 문자열만**
  잡습니다. 평문 비밀번호는 잡지 못합니다 — 함정 8의 훅이 이걸 놓친 이유입니다.

## 함정 12 (2026-09-10) — 모드 C 큰 장면 + 크롭 추적은 한 호출 600초를 넘김

하네스 배경 태스크는 메모리 감시에 죽지만 **컨테이너는 완주**합니다(CoffeeMartini에서 확인: 래퍼가 죽어도 MANIFEST·21프레임 정상). 600초를 넘길 장면(CoffeeMartini·FlameSteak·반구 3장면)은 `setsid nohup <드라이버>.sh > log &`로 하네스 밖에서 한 장면씩 순서대로 돌리고(앞 컨테이너 종료를 `docker ps`로 기다림) 로그를 Monitor로 지켜보십시오. 완주는 `MANIFEST.json`의 `crop_small`·프레임 폴더 수로 확인.
