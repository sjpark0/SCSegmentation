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
| `runMVOptThree.sh` | 데이터셋 지정 실행. `ALGO`·`OUT` 환경변수로 알고리즘/출력 지정 | `--memory=90g` |
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

## 러너 등록표

`SCSam3/runMVSeg.py`:

| `--algo` | 폴더 | 기본 출력 | 시점 추적 기본값 |
|---|---|---|---|
| `OneStage` | `demoSCSam3OneStage` | `SegMaskSam3OneStage` | `written` |
| `OneStageNew` | `demoSCSam3OneStageNew` | `SegMaskSam3OneStageNew` | **`all`** |
| `MVOpt` | `demoSCSam3MVOpt` | `SegMaskSam3MVOpt` | **`all`** |

`NEEDS_ALL_VIEWS = ("OneStageNew", "MVOpt")`.
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
| OneStageNew | 있음 | 없음 | 없음 |
| MVOpt | 있음 | **있음** | **있음** |

MVOpt는 폐기 시 `predictor_spatial = None`으로 만들기 때문에, 폐기 후에는 속성 검사가 **틀린 분기를 고릅니다.**
그래서 플래그를 봅니다. OneStageNew는 플래그가 없어 속성 대체 경로로 같은 분기를 탑니다.

## 5. `spatial model retired` 로그는 거짓일 수 있습니다 ★

`runMVSeg.py:335-338`이 메서드 존재만 확인하고 무조건 출력합니다.

- **OneStage**는 `predictor_spatial`을 애초에 만들지 않아 메서드가 즉시 `return`합니다 —
  **아무것도 반환하지 않고 "retired"를 찍습니다.**
- **OneStageNew**는 메서드 자체가 없어 약 3.2 GiB 교차시점 모델이 시간 추적 내내 상주합니다.
  이것이 OneStageNew가 무거운 데이터셋에서 터지는 이유 중 하나입니다.
- 실제로 회수하려면 `torch.clear_autocast_cache()`가 반드시 함께 호출되어야 합니다.
  autocast 캐시가 죽은 fp32 파라미터에 대한 약한 참조로 bf16 사본을 붙들고 있기 때문입니다.

## 6. `PYTORCH_CUDA_ALLOC_CONF`

파이썬에서만 설정되며, 항상 `expandable_segments:True`, 항상 torch 임포트 전입니다.
쉘 스크립트나 `docker run`에서는 설정하지 않습니다. 러너 두 개는 `setdefault`라 밖에서 덮어쓸 수 있습니다.

OOM의 원인이 아닙니다 — [investigations-closed.md](investigations-closed.md) 3번.

## 7. Docker 이미지가 Dockerfile보다 오래됐습니다 ★

`scsam3:latest`는 2026-02-25 빌드본으로, `HF_TOKEN` 레이어 3개를 갖고 있고 secret mount 레이어는 0개입니다.
즉 **현재 Dockerfile(secret mount)로 빌드된 적이 없습니다.** 모든 스크립트가 이 옛 이미지를 씁니다.

빌드하려면:

```bash
echo -n "<토큰>" > /tmp/hf_token
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,src=/tmp/hf_token -t scsam3 SCSam3/
rm /tmp/hf_token
```

[open-items.md](open-items.md) 1번 참조.
