> **2026-09-04 상태 변경.** `demoSCSam3OneStageNew`는 이제 `demoSCSam3MVOpt`의 바이트 동일
> 동결 스냅샷입니다. 아래에서 "OneStageNew"가 *무패치 원본 코드*를 뜻하는 서술은
> **git 태그 `baseline-onestagenew`** 기준으로 읽으십시오. 배경: [README.md](README.md)

# 메모리 최적화 — 무엇을 왜 바꿨나

**목적.** MVSeg 15개 중 3개(AlexaMeadeExhibit, CoffeeMartini, FlameSteak)가 OOM으로 완주하지 못했습니다.
알고리즘은 그대로 두고 메모리만 줄여서 완주시키는 것이 목표였습니다.

**불변 조건.** 완주하던 데이터셋의 출력은 **바이트 단위로 동일해야 합니다.** 발표된 J&F 수치가 여기에 걸려 있기 때문입니다.
이 조건은 실측으로 지켰습니다 (12개 데이터셋 12,692장, 차이 0건).

**작업 폴더.** `SCSam3/demoSCSam3MVOpt/` — `demoSCSam3OneStageNew`의 바이트 사본에서 시작했습니다.

작업 당시에는 원본을 손대지 않았으나, **2026-09-04에 검증을 마치고 원본에 반영했습니다.**
지금은 두 폴더가 동일하고, OneStageNew는 동결 스냅샷입니다.

상세 설계 근거는 [analysis/SCHEDULE.md](analysis/SCHEDULE.md)에 있습니다. 아래는 실제로 들어간 것의 요약입니다.

---

## 7개 변경

전체 diff: [diffs/mvopt_vs_onestagenew.diff](diffs/mvopt_vs_onestagenew.diff)

### S1 — `previous_stages_out`에 sentinel 문자열 저장

**대상.** `SCSam3VideoInference.py` 2곳, `SCSam3VideoInferenceNewMem.py` 2곳

**이전.**
```python
if "previous_stages_out" not in inference_state:
    inference_state["previous_stages_out"] = {}
if frame_idx not in inference_state["previous_stages_out"]:
    inference_state["previous_stages_out"][frame_idx] = out
else:
    inference_state["previous_stages_out"][frame_idx].update(out)
```

**이후.**
```python
inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
```

**근거.** `previous_stages_out`은 `[None] * num_frames`로 초기화되는 **리스트**입니다 (`:164`).
따라서 제거된 가드 두 개는 원래 도달 불가였습니다:

- `"previous_stages_out" not in inference_state` — 키는 항상 존재하므로 **항상 거짓**
- `frame_idx not in <list>` — 리스트의 `in`은 값 검사입니다. 내용물은 `None`이나 문자열이므로 정수 `frame_idx`와 절대 일치하지 않아 **항상 참** → `.update()` 분기는 죽은 코드

읽는 쪽은 `is None` / `is not None` 검사뿐입니다 (`:188`, `:225`, `:233`).
그리고 이 sentinel은 **upstream SAM 3가 하는 것과 글자 그대로 같습니다** (`sam3/sam3/model/sam3_video_inference.py:400`).

**절감.** `out` 딕셔너리가 붙들고 있던 video-resolution GPU 마스크가 세션 내내 상주하던 것을 해제.

### S2 — 비디오 레벨 `mask_inputs_per_obj` 쓰기 제거

**대상.** `SCSam3VideoInference.py` 1곳, `SCSam3VideoInferenceNewMem.py` 1곳

삭제된 줄:
```python
inference_state["mask_inputs_per_obj"][obj_id][frame_idx] = mask
```

**근거.** 이 비디오 레벨 딕셔너리는 **쓰기 전용**입니다. 읽는 코드가 없습니다.
트래커는 자기 사본을 `obj_idx` 키로 따로 보관합니다 (`SCSam3TrackerPredictor.py:407`).
같은 이름이 두 곳에 있어 혼동하기 쉬우니 주의하십시오 — 비디오 레벨과 트래커 레벨은 다른 딕셔너리입니다.

**절감.** 프롬프트된 객체당 fp32 video-res 텐서 1개(호스트 RAM)를 세션 내내 붙들던 것을 해제.

### S3 — `mask_inputs_per_frame`에 meta 자리표시자 저장

**대상.** `SCSam3TrackerPredictor.py:407`, `SCSam3TrackerPredictorNewMem.py:407`

```python
mask_inputs_per_frame[frame_idx] = torch.empty(
    mask_inputs_video_res.shape, dtype=torch.bool, device="meta"
)
```

**근거.** 이 딕셔너리는 **키만** 읽힙니다 — `.pop()`(반환값 버림), `.keys()`, `in`, `set.update(dict)`(키 순회), `_map_keys`(값을 그대로 통과).
meta 텐서는 shape·dtype이 맞고 저장소가 없습니다.

**중요.** 진짜 마스크는 지역변수 `mask_inputs_video_res`에 그대로 남아 있고, 실제로 필요한 `:447`/`:459`에서 그것을 씁니다.

**절감.** 33객체 × 45시점 × 2560×1920 기준 **6.8 GiB GPU**.

### S4 — `_apply_non_overlapping_constraints`의 bool 빠른 경로

**대상.** `SCSam3TrackerPredictor.py:~1398`, `SCSam3TrackerPredictorNewMem.py:~1933`

```python
if pred_masks.dtype == torch.bool and background_value == 0:
    del pred_masks_single_score
    keep = pixel_level_non_overlapping_masks > 0
    keep &= pred_masks
    return keep
```

**근거.** bool 입력에서 `torch.clamp(pred_masks, max=0)`은 전부 0입니다.
따라서 일반 경로의 `torch.where`는 `pred_masks & (mask > 0)`으로 환원됩니다 —
다만 파이썬 정수 `max`가 bool을 int64로 승격시켜 **int64 (N,1,H,W) 임시 텐서 2개**를 만듭니다. 그게 이 함수의 피크입니다.

**주의.** 이 분기는 int64 대신 bool을 반환합니다. 두 호출 지점 모두 결과를 즉시 `> 0`으로 임계처리하므로 출력은 동일합니다.
호출 지점을 늘릴 때는 이 전제를 다시 확인하십시오.

### S5 — 정규화된 이미지를 float32로 캐스트

**대상.** `io_utils.py:~206`

```python
img -= self.img_mean
img /= self.img_std
img = img.to(torch.float32)     # 추가
```

**근거.** `img_np / 255.0`이 uint8을 numpy 경유로 **float64**로 승격시키고, 이후 in-place 연산은 좌변 dtype을 유지합니다.
값을 읽는 유일한 소비자가 어차피 float32로 캐스트합니다 (`sam3/model/sam3_image.py:152`).
float64→float32 반올림은 멱등이므로 모델 입력은 비트 단위로 같습니다.

**하지 말 것 두 가지.**
- 캐스트를 mean/std 연산 **앞으로** 옮기지 마십시오. 연산이 float32로 바뀌어 이중 반올림으로 1 ulp 차이가 날 수 있습니다.
- float16을 쓰지 마십시오. 소비자의 float32 캐스트가 버려진 가수 비트를 복구하지 못해 **마스크가 바뀝니다.**

**절감.** 시점당 상주 사본이 23.26 → 11.63 MiB. 시점 수만큼, 그리고 시점 전파용 유사 비디오에 N개가 더 있습니다.

### S6 — `RetireSpatialPredictor` + `uses_spatial_predictor` ★ 순수 수정 아님

**대상.** `SCSam3Video.py`

시점 전파가 끝나면 교차시점 모델을 통째로 버립니다. 세션을 닫고, `predictor.model = None`, `self.predictor_spatial = None`,
`gc.collect()`, `torch.cuda.empty_cache()`.

**`.cpu()`가 아니라 버리는 이유.** 호스트 사본은 약 3.2 GiB이고, 머신을 죽여온 자원이 바로 호스트 RAM입니다.
bf16 캐스트본은 전역 autocast 캐시가 붙들고 있으므로 **호출자가 `torch.clear_autocast_cache()`도 같이 실행해야** 실제로 회수됩니다.
`runMVSeg.py`가 그렇게 합니다.

**`uses_spatial_predictor` 플래그가 따로 있는 이유.** 폐기 후에는 `hasattr(self, "predictor_spatial")`가
"이 패키지가 NewMem 계열인가"의 대용이 될 수 없습니다. 호출자는 이 플래그를 봐야 합니다.

**분류.** 메모리 수정 + **새 공개 API**입니다. `runMVSeg.py`가 `hasattr(sc, "RetireSpatialPredictor")`로 분기하므로,
이 메서드를 어떤 폴더에 넣는 순간 **러너가 그 폴더에 대해 다르게 동작합니다.**

### S7 — `TRIM_CACHED_OUTPUTS` (환경변수, 기본 꺼짐) ★ 순수 수정 아님

**대상.** `SCSam3VideoInferenceNewMem.py`

```python
TRIM_CACHED_OUTPUTS = os.environ.get("SCSAM3_TRIM_CACHED_OUTPUTS", "").strip() not in ("", "0")
```

전파 중 이미 소비된 과거 프레임의 `cached_frame_outputs` 항목을 버립니다.

**두 가지 안전장치 — 하나는 실제로 작동하지 않습니다.**
- 엄격히 과거인 프레임만 버립니다. 아직 도달하지 않은 프레임은 건드리지 않습니다. **(유효)**
- `action_history`에 `propagation_full`이 있으면 아무것도 하지 않습니다. **★ 이 가드는 이 패키지에서 절대 발동하지 않습니다.**
  `add_prompt`가 항상 첫 전파보다 먼저 실행되어 `action_history`가 비어 있지 않으므로,
  `parse_action_history_for_propagation`이 `propagation_full`을 반환할 일이 없습니다. docstring의 주장이 틀렸습니다.

**★ "기본 꺼짐"은 실제 실행 조건이 아닙니다.** `runMVOptThree.sh:14`가 `-e SCSAM3_TRIM_CACHED_OUTPUTS=1`을 넘깁니다.
**MVOpt 결과 전부가 이 스크립트로 만들어졌습니다.**

직접 확인 (2026-09-04): Blocks를 트림 명시적 OFF로 다시 돌려 저장본과 대조 → **882 대 882 PNG, 완전 일치.**
배치 경로에서는 ON/OFF가 같은 출력을 냅니다.

값 파싱 함정: 거짓인 것은 `""`·공백·`"0"`뿐입니다. **`"false"`·`"off"`·`"no"`·`"00"`은 전부 켜집니다.**
플래그는 import 시점에 한 번만 읽으므로 프로세스 안에서 `os.environ`을 바꿔도 효과가 없습니다.

**분류.** 새 기능입니다. 대화형 경로에는 조용한 손상 위험이 있습니다 —
[mvopt-audit.md](mvopt-audit.md) 참조. 배치 실행에서만 쓰십시오.

---

## 동등성 증거

| 대조 | 규모 | 차이 |
|---|---|---|
| MVOpt vs OneStageNew — 12개 데이터셋 | PNG 12,692장 | **0** |
| 카메라 단위 J·F 값 집합 | 36개 | **0** |
| 트림 ON vs OFF (Blocks) | PNG 882장 | **0** |
| **반영 후 OneStageNew vs MVOpt** — Welder + 무거운 3개 | PNG 8,064장 | **0** |

마지막 줄은 2026-09-04 반영 직후 실측입니다. 이 넷은 **원본 OneStageNew가 90 GiB 상한 안에서
완주하지 못하던 것들**입니다 (Welder는 OOM 중단, 나머지 셋은 한 번도 완주한 적 없음).
반영 후 전부 완주하고 MVOpt와 바이트 동일합니다.
| OneStage 패치 전/후 — 15개 데이터셋 | 14개 바이트 동일 | Painter만 14파일, **점수 동일** |
| ForSam2New 패치 전/후 — Frog | PNG 315장 | **0** |

변경별 반증 검증(에이전트 32개) 결과는 [mvopt-audit.md](mvopt-audit.md)에 있습니다.

Painter 건은 패치가 원인이 아닙니다. [investigations-closed.md](investigations-closed.md) 1번 참조.

---

## 폴더별 적용 현황

| 폴더 | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---|---|---|---|---|---|---|
| `demoSCSam3MVOpt` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `demoSCSam3OneStage` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (무효) | 해당없음 |
| `demoSCSam3TwoStage` | ✓ | ✓ | ✓ | ✓ | ✓ | — | 해당없음 |
| `demoSCSam3TwoStageNew` | ✓ | ✓ | ✓ | ✓ | ✓ | — | 해당없음 |
| `demoSCSam3ForSam2` | 해당없음 | 해당없음 | ✓ | ✓ | ✓ | — | 해당없음 |
| `demoSCSam3ForSam2New` | 해당없음 | 해당없음 | ✓ | ✓ | ✓ | — | 해당없음 |
| `demoSCSam3OneStageNew` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**MVOpt와 OneStageNew가 7종 전부를 갖습니다** — 2026-09-04부로 두 폴더는 바이트 동일합니다.
무패치 원본은 git 태그 `baseline-onestagenew`에만 있습니다.

### "해당없음"의 이유

**S1·S2가 `ForSam2*`에 없는 이유 — 앵커가 존재하지 않습니다.**
두 폴더에는 `def add_tracker_new_mask`가 아예 없고, `add_tracker_new_points`도 `previous_stages_out`을 건드리지 않습니다.
S2가 패치하는 객체 등록 블록도 없습니다. SAM 2 스타일 패키지라
`_build_tracker_output` 다음에 바로 `_cache_frame_outputs`로 넘어갑니다 — 중간 장부가 없습니다.

**S7이 5개 폴더에 없는 이유** — 이 변경은 `SCSam3VideoInferenceNewMem.py`에 사는데,
그 파일은 MVOpt와 OneStageNew에만 있습니다. 나머지는 다중 세션 전파 루프가 없습니다.
(단일 세션 쪽에도 `cached_frame_outputs`는 있으므로 이식은 가능합니다.)

**OneStage의 S6이 무효인 이유** — `predictor_spatial`을 애초에 만들지 않아 메서드가 즉시 반환합니다.
그런데 `spatial model retired` 로그는 찍힙니다. [operations.md](operations.md) 함정 5번 참조.

### S1 개수 세는 법 (함정)

`_THIS_FRAME_HAS_OUTPUTS_` 문자열은 **upstream에도 1개 있습니다**(`_run_single_frame_inference`).
S1 변경은 2개를 **추가**합니다. 따라서:

| 개수 | 의미 |
|---|---|
| 1 | **미적용** (upstream 기본값) |
| 3 | **적용됨** |

개수 1을 보고 "부분 적용"으로 오독하기 쉽습니다.

## 새 폴더에 적용할 때

1. **먼저 기준선을 만드십시오.** 패치 전 코드로 데이터셋 하나를 돌려 출력을 보관합니다.
   기준선 없이 패치하면 동등성을 증명할 방법이 없습니다.
2. 앵커가 **정확히 한 번** 일치할 때만 적용하십시오. 일치하지 않으면 억지로 넣지 마십시오 —
   구조가 다르다는 신호이고, 그 폴더에서는 그 수정이 다른 자리에 있어야 합니다.
3. S6·S7은 순수 수정이 아닙니다. 넣을지 따로 판단하십시오. 특히 S6은 러너 동작을 바꿉니다.
4. 패치 후 같은 데이터셋을 다시 돌려 `diff -rq`로 대조하십시오.

**주의.** ForSam2New처럼 SAM 2 스타일 API인 폴더는 S1·S2의 앵커가 없어 절감이 거의 없습니다.
"복사하면 된다"는 가정은 이미 반증됐습니다. [investigations-closed.md](investigations-closed.md) 10번 참조.
