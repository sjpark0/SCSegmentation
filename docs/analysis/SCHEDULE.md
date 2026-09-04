> **시점 안내 (2026-09-04).** 이 문서는 `demoSCSam3OneStageNew`가 **무패치 원본**이던 시점의
> 조사 결과입니다. 이후 그 폴더는 `demoSCSam3MVOpt`의 동결 스냅샷으로 바뀌었으므로,
> "OneStageNew에는 없다 / 무패치다" 류의 서술은 **현재 상태가 아닙니다.**
> 원본은 git 태그 `baseline-onestagenew`. 배경: `docs/README.md`

# SCSam3 `demoSCSam3MVOpt` 메모리 최적화 실행 계획

## 목적과 원칙

### 이 폴더가 존재하는 이유

`/home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3MVOpt`는 `demoSCSam3OneStageNew`의 **신선한 복사본**이다. 12개 `.py` 파일이 원본과 byte-identical이며, 중복 파일 `SCSam3TrackerPredictorNewMem copy.py`만 제외했다 (`diff -rq demoSCSam3MVOpt demoSCSam3OneStageNew`로 확인: 차이는 `copy.py`, `__pycache__`, 데모 산출물 디렉터리 `0`~`31`뿐이며 이들은 `runMVSeg.py` 경로에서 읽히지 않는다 — HF 캐시에서 체크포인트를 받으므로 `build_scsam3.py:693/709/835/975`).

**원본(`demoSCSam3OneStageNew`, `demoSCSam3OneStage`, 그 외 sibling 폴더)은 절대 건드리지 않는다.** 게시된 J&F 숫자가 그 폴더들의 출력에 묶여 있고, 문제가 생기면 즉시 되돌아갈 기준점이 필요하다.

### 불변 조건

> **완료되는 데이터셋의 출력이 bit-identical 해야 한다.**

출력이 조금이라도 바뀌면 그것은 트레이드오프가 아니라 **실패**다. 유일한 예외는 사전에 "출력 변경"이라고 명시적으로 표시하고 별도로 결정한 항목이며, 단계 1에는 그런 항목이 하나도 없다.

### 확인 방법

완료되는 데이터셋(예: `Blocks`)을 새 폴더로 돌린 뒤 기존 출력과 재귀 비교한다.

```bash
diff -rq /home/sjpark/Documents/SCSegmentation/Data/MVSeg/Blocks/SegMaskSam3OneStageNew \
         /home/sjpark/Documents/SCSegmentation/Data/MVSeg/Blocks/SegMaskSam3MVOpt_s1
```

**출력이 0줄이어야 한다.** PNG는 바이너리이므로 `diff -rq`가 그대로 bit-identity 검사가 된다. 상세 절차는 마지막 절.

각 사이트를 적용할 때마다 이 검사를 돌린다. 여러 개를 몰아서 적용한 뒤 diff가 깨지면 어느 사이트가 원인인지 이분 탐색해야 한다.

### 공유 파일 취급 원칙

`runMVSeg.py`는 `OneStage`/`OneStageNew`와 **공유**된다. 여기에 가하는 모든 편집은 기존 두 알고리즘에 대해 출력 중립이어야 하며, 반드시 `getattr`/`hasattr` 가드 뒤에 둔다 (S6 참조).

---

## 단계 0 — 러너 등록 (선행 필수, 코드 변경 없음)

`runMVSeg.py:38-40`의 `ALGOS`/`DEFAULT_OUT`에 `MVOpt`가 없어서 **현재 이 폴더는 러너에서 선택 자체가 불가능하다.** `argparse`의 `choices=sorted(ALGOS)`(`runMVSeg.py:45`)가 거부하고, `DEFAULT_OUT[args.algo]`(`:253`)는 KeyError를 낸다. 등록 전까지 아래 모든 절감은 **0**이다.

```python
# runMVSeg.py:38-40
ALGOS = {"OneStage": "demoSCSam3OneStage",
         "OneStageNew": "demoSCSam3OneStageNew",
         "MVOpt": "demoSCSam3MVOpt"}
DEFAULT_OUT = {"OneStage": "SegMaskSam3OneStage",
               "OneStageNew": "SegMaskSam3OneStageNew",
               "MVOpt": "SegMaskSam3MVOpt"}
```

### 함정 — `--track-cams` 기본값

`runMVSeg.py:292`:

```python
track_mode = args.track_cams or ("all" if args.algo == "OneStageNew" else "written")
```

`MVOpt`를 그냥 등록하면 기본값이 **`written`**이 된다. 이것은 (a) 세션 수가 달라져 출력이 바뀌고, (b) scored 카메라가 4개 미만인 데이터셋에서는 `output_dicts[-4]`가 IndexError를 낸다(감사 보고서 C1, `SCSam3TrackerPredictorNewMem.py:1254`). 반드시 둘 중 하나:

```python
track_mode = args.track_cams or ("all" if args.algo in ("OneStageNew", "MVOpt") else "written")
```

또는 모든 실행에 `--track-cams all`을 명시한다. **검증 절차에서는 두 가지를 다 한다** (명시 플래그 + 기본값 수정).

---

## 단계 1 — 메모리 (지금 착수)

적용 순서는 아래 번호 순이다. 앞의 셋은 순수 삭제/치환이고, 뒤의 둘이 실패 데이터셋을 실제로 여는 항목이다.

---

### S1. `previous_stages_out`에 `out` dict 대신 sentinel 문자열 저장

**무엇을** — 저자 패치 4곳을 upstream과 동일한 sentinel 한 줄로 교체:

| 파일 | 라인 | 함수 |
|---|---|---|
| `demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py` | 1743-1752 | `add_tracker_new_points` |
| `demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py` | 1992-2001 | `add_tracker_new_mask` ← **MVSeg 핵심** |
| `demoSCSam3MVOpt/SCSam3VideoInference.py` | 1656-1665 | `add_tracker_new_points` |
| `demoSCSam3MVOpt/SCSam3VideoInference.py` | 1905-1914 | `add_tracker_new_mask` |

각 블록을 아래로 대체:

```python
inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
```

**왜** — 저장되는 `out["obj_id_to_mask"]`는 객체별 `(1, H_video, W_video)` bool **CUDA** 텐서 dict다. 이것이 세션 수명 내내 GPU에 고정된다. upstream SAM 3는 같은 자리에 문자열 sentinel만 넣는다 (`SCSam3VideoInferenceNewMem.py:400`, `SCSam3VideoInference.py:398`, upstream `sam3/sam3/model/sam3_video_inference.py:399-400`).

**bit-identical 근거** — `previous_stages_out`의 **모든** 리더가 None 여부만 본다:
- `SCSam3VideoInferenceNewMem.py:189` (`is None`), `:226` (`all(out is None ...)`), `:234` (`min(t ... if out is not None)`)
- 같은 것이 `SCSam3VideoInference.py:188`, `:225`, `:233`

전체 grep으로 확인했다. 내용을 순회/직렬화/deepcopy하는 코드는 패키지 어디에도 없다. dict와 문자열은 이 세 리더에게 구별 불가능하다. 또한 `previous_stages_out`은 **list**(`:165`/`:164`에서 `[None] * num_frames`)이므로 저자 코드의 `frame_idx not in ...`은 값 멤버십 스캔이고 항상 True — `.update(out)` 분기는 이미 죽은 코드다. 즉 기존 코드도 이미 단순 덮어쓰기였다.

**절감 (GPU)** — DS1 6.95 GiB / DS2 6.40 GiB / DS3 7.64 GiB. (points 경로 2곳은 MVSeg에서 호출되지 않으므로 절감 0이나 일관성을 위해 함께 적용.)

**검증** — `Blocks` 실행 후 `diff -rq` 0줄.

**의존관계** — 없음. **단, S2보다 반드시 먼저.** S1의 패치 앵커는 `inference_state["mask_inputs_per_obj"][obj_id][frame_idx] = mask` 줄(`NewMem:1967` / `plain:1880`)에서 시작하는데, S2가 바로 그 줄을 삭제한다. S1을 먼저 적용하면 그 줄이 그대로 남아 S2의 앵커가 유효하다. 순서를 바꾸면 S1을 다시 앵커링해야 한다.

---

### S2. video-level `mask_inputs_per_obj` 저장 삭제

**무엇을** — 아래 두 줄을 삭제(주석으로 대체):

- `demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py:1967`
- `demoSCSam3MVOpt/SCSam3VideoInference.py:1880`

```python
inference_state["mask_inputs_per_obj"][obj_id][frame_idx] = mask
```

dict 생성 라인(`NewMem:1955-1964`, `plain:1868-1877`)은 **그대로 둔다** → 키는 빈 dict로 계속 존재하므로 어떤 동적 접근도 KeyError를 새로 만들지 않는다.

**왜** — 호출자의 fp32 video-res 시드 마스크(`runMVSeg.py:186`, `:222-227`)를 세션 끝까지 CPU에 고정한다. **읽는 코드가 하나도 없다.** `mask_inputs_per_obj`의 모든 read는 *tracker_state*를 받는다: `SCSam3VideoInferenceNewMem.py:2091`, `SCSam3VideoInference.py:2004`, 그리고 `SCSam3TrackerPredictor{,NewMem}.py`의 전 사이트. tracker_state는 `_init_new_tracker_state`(`NewMem:997-1003`)가 `self.tracker.init_state(...)`로 새로 만든 별개의 dict이며 video-level state와 절대 alias되지 않는다. tracker는 자기 사본(`SCSam3TrackerPredictorNewMem.py:407`의 `mask_inputs_video_res`)을 따로 갖는다.

**절감 (host RAM)** — fp32 video-res, `H*W*4` 바이트:
- DS1 (45×33 @2560×1920): 1485 × 18.75 MiB = **27.80 GiB**
- DS2 (18×66 @2704×2028): 1188 × 20.92 MiB = **25.62 GiB**
- DS3 (21×68 @2704×2028): 1428 × 20.92 MiB = **30.56 GiB**

GPU 절감 0 (MVSeg 러너는 CPU 텐서를 넘긴다). `masks_spatial`에 마스크가 없는 (view,obj) 쌍은 `runMVSeg.py:218-219`의 공유 `zero` 텐서를 재사용하므로 실측은 이 상한보다 약간 낮다 (Blocks 실측 141 storage / 1.089 GiB).

**검증** — `Blocks` `diff -rq` 0줄 + `/proc/<pid>/status` RSS 피크 비교.

**의존관계** — **S1 이후.** (같은 rank-0 블록을 건드림.)

---

### S3. tracker의 시드 마스크를 meta 플레이스홀더로

**무엇을** — 두 파일의 `add_new_mask` 안, 각각 **:407**:

- `demoSCSam3MVOpt/SCSam3TrackerPredictorNewMem.py:407`
- `demoSCSam3MVOpt/SCSam3TrackerPredictor.py:407`

```python
mask_inputs_per_frame[frame_idx] = torch.empty(
    mask_inputs_video_res.shape, dtype=torch.bool, device="meta"
)
```

**지역 변수 `mask_inputs_video_res`는 절대 재바인딩하지 않는다** — `:447`/`:459`의 `torch.where`가 실제 마스크를 계속 소비해야 한다.

**왜** — video-res bool 마스크가 **compute device(cuda)** 에 (객체 × 프롬프트 프레임)마다 고정된다(`:377`에서 device로 이동). `runMVSeg.py` `TrackForward`는 뷰 × 객체마다 한 번씩 프롬프트하므로 항목 수 = views × objects.

**bit-identical 근거** — `mask_inputs_per_obj[obj_idx]`의 모든 리더가 **키만** 쓴다:
- `:267`, `:951` — `pop(frame_idx, None)` 반환값 폐기
- `:759-760` — `.keys()` → `input_frames_inds` → `:761` assert, `:767` `min()`
- `:964` — `in` 멤버십 → `frame_has_input`
- `:1775` — `set.update(dict)` (키 순회)
- `:1808` — `_map_keys`(`:1800-1806`)는 `v = container.pop(k)` 후 새 키로 재삽입, 값 미검사
- 내용을 보는 유일한 줄 `:845`는 **주석 처리됨**

plain 스택도 동일 구조 (`:267`, `:759-760`, `:940`, `:953`, `:1241`, `:1274`). 키 집합과 `len()`이 동일하므로 assert·min·스캔·remove_object 장부가 전부 동일하게 동작한다.

**절감 (GPU)** — DS1 6.95 GiB / DS2 6.40 GiB / DS3 7.64 GiB. (NewMem 6.80/6.07/7.29 + plain 0.15/0.34/0.35)

**하지 말 것** — `inference_state["storage_device"]`(= cpu)로 옮기는 대안은 **기각**. `_init_new_tracker_state`가 `offload_state_to_cpu` 기본 True로 호출하므로 6~7 GiB가 host RAM으로 넘어가는데, host RAM이 바로 지금 죽고 있는 자원이다.

**검증** — `Blocks` `diff -rq` 0줄. Dockerfile 베이스가 `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel`이므로 meta 디바이스는 안전하나, 첫 실행 시 `torch.empty(..., device="meta")`가 `@torch.inference_mode()` + bf16 autocast 안에서 뜨는지 로그로 확인할 것 (호스트에 torch가 없어 정적 검증만 마침).

**의존관계** — 없음. S1/S2와 다른 파일이므로 병렬 적용 가능.

---

### S4. non-overlap 후처리의 int64 승격 제거

**무엇을** — 두 파일의 `_apply_object_wise_non_overlapping_constraints` 마지막 `torch.where` **앞에** fast path를 추가:

- `demoSCSam3MVOpt/SCSam3TrackerPredictorNewMem.py:1911-1931` (블록 1925-1931)
- `demoSCSam3MVOpt/SCSam3TrackerPredictor.py:1377-1397` (블록 1391-1397)

```python
if pred_masks.dtype == torch.bool and background_value == 0:
    del pred_masks_single_score
    keep = pixel_level_non_overlapping_masks > 0
    keep &= pred_masks
    return keep
# (기존 torch.where 블록은 그대로 남긴다)
```

**왜** — `torch.clamp(bool_tensor, max=0)`은 Python int `max` 때문에 **int64**로 승격된다(컨테이너 torch 2.10.0+cu128에서 실측). 즉 clamp 결과와 `torch.where` 결과가 각각 int64 `(N,1,H,W)`다. N=26 @2560×1920이면 각각 **975 MiB** — 이것이 `logs/mvseg-20260903-104557/OneStageNew-AlexaMeadeExhibit.log:298`의 "Tried to allocate 976 MiB"와 정확히 일치한다.

**bit-identical 근거** — bool 입력 + `background_value == 0`이면 `clamp(pred_masks, max=0)`은 전부 0이므로

```
torch.where(plnom > 0, pred_masks, clamp(pred_masks, max=0))  ==  pred_masks & (plnom > 0)
```

이고, 그 결과는 int64로 materialize될 뿐 값은 같다. **호출자 두 곳이 결과를 즉시 `> 0`으로 임계화한다**:
- `SCSam3VideoInferenceNewMem.py:505-513` — `(...).squeeze(1) > 0`, `background_value=0`
- `SCSam3VideoInference.py:503-511` — 동일

입력 dtype은 `NewMem:465` / `plain:463`의 `assert out_binary_masks.dtype == torch.bool`로 런타임 고정된다. 컨테이너에서 무작위 300+2800 trial (N 2~6, 랜덤/전0/동점/음수/NaN 스코어, `background_value` `0`과 `0.0` 양쪽)로 `ref > 0 == fast > 0`을 확인, 불일치 0.

**출력 영향 표시** — **반환 dtype이 int64 → bool로 바뀐다.** 두 호출자 모두 `> 0`을 씌우므로 written PNG는 bit-identical이지만, 이것이 이 항목의 유일한 관측 가능 변화이므로 여기에 명시해 둔다. **조건: 향후 이 함수의 반환값을 크기(magnitude)로 쓰는 호출자를 추가하지 말 것.**

**절감 (GPU 피크)** — 해당 statement 피크:

| | 전 | 후 | statement 절감 | 함수 피크 절감 |
|---|---|---|---|---|
| N=26 @2560×1920 | 3168.8 MiB | 731.3 MiB | 2437.5 MiB | **1425.0 MiB (1.39 GiB)** |
| N=66 @2704×2028 | 8975.0 MiB | 2071.2 MiB | 6903.8 MiB | **4100.4 MiB (4.00 GiB)** |
| N=68 @2704×2028 | 9247.4 MiB | 2134.1 MiB | 7113.3 MiB | **4224.4 MiB (4.13 GiB)** |

함수 피크는 `super()._apply_non_overlapping_constraints`(`sam3_tracker_base.py:1115-1132`)로 이동한다 — 이 함수 내부는 건드리지 않는다(별개의 위험한 사이트).

**검증** — `Blocks` `diff -rq` 0줄. 추가로 컨테이너에서 원본/패치 모듈을 나란히 import 해 무작위 differential test를 다시 한 번 돌릴 것.

**의존관계** — 없음. **AlexaMeadeExhibit의 OOM을 직접 일으킨 할당이 여기다.**

**주의** — 패치 후 이 두 함수는 upstream `sam3/sam3/model/sam3_tracking_predictor.py:1349-1367`과 갈라진다. 나중에 upstream을 재동기화하면 조용히 revert된다.

---

### S5는 보류 — 아래 「보류 항목」 참조

---

### S6. spatial predictor 은퇴 + autocast 캐시 정리 + fp32 프레임

세 개의 하위 항목이며 **(a)와 (b)는 반드시 함께 나간다.**

#### (a) cross-view 모델 은퇴 — `demoSCSam3MVOpt/SCSam3Video.py`

- `__init__`(`:13-17`)에 `self.uses_spatial_predictor = True` 추가
- `RetireSpatialPredictor()` 메서드 신설(`:29-31` 뒤): spatial 세션 `close_session` → `predictor.model = None` → `self.predictor_spatial = None` → `gc.collect()` → `torch.cuda.empty_cache()`
- 속성은 **`del`이 아니라 `None` 대입** (구식 호출자의 `hasattr` 스니핑을 깨지 않기 위해)

#### (a') 러너 배관 — `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSeg.py`

- `:240` `hasattr(self, "predictor_spatial")` → `getattr(self, "uses_spatial_predictor", getattr(self, "predictor_spatial", None) is not None)`
- `PropagateAcrossViews` 직후(`:309-311`)에 `if hasattr(sc, "RetireSpatialPredictor"): sc.RetireSpatialPredictor(); torch.clear_autocast_cache()`

**왜 안전한가** — `predictor_spatial`/`session_id_statial`의 마지막 읽기는 `PropagateAcrossViews`(`runMVSeg.py:192-210`)다. 이후 `TrackForward`(`:212-247`)와 write loop(`:314-330`)는 `self.predictor`(NewMem)와 `masks_spatial`만 만진다. `masks_spatial` 값은 `out["out_binary_masks"][i] > 0.0`(`:208`)로 새로 할당된 bool 텐서이며 세션 상태의 view가 아니다. 두 predictor는 별개 클래스·별개 `_ALL_INFERENCE_STATES`(`SCSam3VideoPredictor.py:26` vs `SCSam3VideoPredictorNewMem.py:26`)·별개 모델 인스턴스다.

**기존 알고리즘 중립성** — `OneStage`는 `predictor_spatial` 속성 자체가 없어 `False` → `session_id` 분기(기존과 동일). `OneStageNew`는 속성이 있고 `None`이 아니며 `RetireSpatialPredictor`가 없으므로 `hasattr` 가드에 걸려 은퇴 코드가 실행되지 않는다 → 완전히 동일.

#### (b) `torch.clear_autocast_cache()` — (a)와 **동시 필수**

트래커가 `torch.autocast(...).__enter__()`를 하고 절대 나오지 않으므로(`SCSam3TrackerPredictorNewMem.py:51-52`, `SCSam3TrackerPredictor.py:51-52`) ATen이 자동으로 캐시를 비우지 않는다. 캐시는 bf16 사본에 **강한 참조**, 원본 fp32 파라미터에 **약한 참조**를 갖는다. (a)만 하고 (b)를 안 하면 은퇴한 모델의 bf16 사본 ~1.6 GiB가 GPU에 그대로 남는다. 또한 **순서가 correctness에 걸려 있다** — 파라미터 free 직후 곧바로 clear해야 한다(중간에 새 fp32 leaf 텐서를 만들지 않는다).

값 중립인 이유: 캐시 항목은 `param.to(bfloat16)`의 결정론적 함수이고 파라미터는 추론 중 불변(`load_state_dict`/`nn.init`은 빌드 시점, `eval()` + `@torch.inference_mode()`, optimizer 없음). 지운 항목은 bit-for-bit 재계산된다. `sam3/model/decoder.py:71`의 autocast-disabled FFN 영역은 캐시를 조회조차 하지 않으므로 무관.

**프레임마다 clear하지 말 것.** 프레임 내부에서는 살아 있는 가중치 전체가 어차피 캐시되어 있어 피크가 내려가지 않고, 매 프레임 ~5 GiB의 HBM 트래픽만 추가된다. **cross-view → temporal 경계에서 딱 한 번.**

#### (c) 프레임 dtype float64 → float32 — `demoSCSam3MVOpt/io_utils.py:204-206`

```python
img -= self.img_mean
img /= self.img_std
img = img.to(torch.float32)     # ← 추가
self.image = img
```

**왜** — `img_np / 255.0`(`io_utils.py:32`)에서 uint8이 numpy 승격 규칙에 따라 **float64**가 되고, `from_numpy` → `permute`가 이를 유지하며, in-place `-=`/`/=`는 LHS dtype을 유지한다. 프레임 하나가 3×1008×1008 = 3,048,192 elem → fp64 **23.26 MiB** vs fp32 **11.63 MiB**.

**bit-identical 근거** — dtype에 민감한 소비자는 단 하나: `sam3/model/sam3_image.py:152`의 `image.to(dtype=torch.float32, device=self.device)`, 그것도 `backbone.forward_image` 직전. 그 외 경로는 raw 프레임을 버린다 — `feature_cache`(`sam3_video_base.py:394-397`)는 참조 저장, `_get_image_feature`(`SCSam3TrackerPredictorNewMem.py:1073`)는 expand 후 `image=`로 흘려보내는데 `sam3_tracker_base.py:832`가 `isinstance(maskmem_backbone, SimpleMaskEncoder)` 분기를 타므로(`build_scsam3.py:365`) 이미지 인자를 받지 않는 `SimpleMaskEncoder.forward(pix_feat, masks, skip_mask_sigmoid)`가 호출된다. `img.cuda().float()` 캐시 미스 분기(`:1063-1064`)는 `backbone=None`(`build_scsam3.py:441/495/507/517`)이라 `:1053-1058`에서 먼저 raise되므로 **죽은 코드**다. 따라서 모델이 보는 값은 `round_to_f32(v)`이고, 미리 f32로 반올림해도 `round_to_f32(round_to_f32(v)) = round_to_f32(v)` — 반올림은 idempotent다.

**두 가지 금지 사항:**
1. 캐스팅은 반드시 `-=`/`/=` **뒤에** 온다. 앞에 두면 뺄셈이 fp32에서 일어나 double rounding으로 1 ulp 차이가 날 수 있다.
2. **float16 금지.** 감사 보고서 `REPORT.md:183`은 fp16을 "결과 동일"로 권하지만 **그 주장은 틀렸다.** fp16은 mantissa 10비트라 소비자의 fp32 캐스트로 복구 불가능하고, ~1e-3 상대 오차가 mask logit을 거쳐 `> 0.0` 임계(`runMVSeg.py:328`)에서 픽셀을 뒤집는다. 보류 항목에 명시.

**절감**

| | (a) GPU | (b) GPU | (c) GPU | (a) host | (c) host |
|---|---|---|---|---|---|
| DS1 45×33 | ~4.4 GiB | ~1.6 GiB | 0.51 GiB | ~1.27 GiB | 0.51 GiB |
| DS2 18×66 | ~3.8 GiB | ~1.6 GiB | 0.20 GiB | ~1.02 GiB | 0.20 GiB |
| DS3 21×68 | ~3.9 GiB | ~1.6 GiB | 0.24 GiB | ~1.22 GiB | 0.24 GiB |

((a) GPU = 3.2 GiB 파라미터 + N뷰 pseudo-video + ~0.2 GiB detector 캐시. 체크포인트 `sam3.pt` = 3,450,062,241 B ≈ 3.21 GiB, 인스턴스 2개.)

**검증** — `Blocks` `diff -rq` 0줄. 추가로 `torch.clear_autocast_cache()` 호출이 실제 존재하는지 첫 실행 로그로 확인(호스트에 torch가 없어 실행 검증 불가). 은퇴 직후 프레임에서 ~1.6 GiB의 bf16 재할당 transient가 생기므로 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(`runMVSeg.py:32`)가 켜져 있는지 확인.

**의존관계** — 단계 0(등록) 필수. (a)와 (b)는 분리 불가. (c)는 독립. **단일 GPU 전제** — `SCSam3Video.py:15`가 `range(torch.cuda.device_count())`를 넘기는데 world_size>1이면 두 번째 predictor의 `_start_nccl_process_group`이 `assert not torch.distributed.is_initialized()`(`SCSam3VideoPredictorNewMem.py:429`)에서 이미 죽는다(패치 이전부터). `RetireSpatialPredictor`는 의도적으로 `shutdown()`을 호출하지 않는다 — world_size>1에서 프로세스 전역 `destroy_process_group()`이 temporal predictor의 collective까지 죽이기 때문.

---

### 요약 표

| 항목 | 대상 (file:line) | 절감 (DS1 / DS2 / DS3) | 출력 영향 | 검증 |
|---|---|---|---|---|
| **S0** 등록 | `runMVSeg.py:38-40`, `:292` | — (선행 조건) | 없음 (기존 algo 분기 불변) | `--dry-run`으로 경로 해석 확인 |
| **S1** sentinel | `SCSam3VideoInferenceNewMem.py:1743-1752`, `:1992-2001`; `SCSam3VideoInference.py:1656-1665`, `:1905-1914` | GPU 6.95 / 6.40 / 7.64 GiB | **없음** (리더 전원 None 검사) | Blocks `diff -rq` 0 |
| **S2** video-level mask 삭제 | `SCSam3VideoInferenceNewMem.py:1967`; `SCSam3VideoInference.py:1880` | host 27.80 / 25.62 / 30.56 GiB | **없음** (리더 부재) | Blocks `diff -rq` 0 + RSS |
| **S3** tracker meta 시드 | `SCSam3TrackerPredictorNewMem.py:407`; `SCSam3TrackerPredictor.py:407` | GPU 6.95 / 6.40 / 7.64 GiB | **없음** (키만 읽힘) | Blocks `diff -rq` 0 |
| **S4** int64 승격 제거 | `SCSam3TrackerPredictorNewMem.py:1925-1931`; `SCSam3TrackerPredictor.py:1391-1397` | GPU 피크 1.39 / 4.00 / 4.13 GiB | **반환 dtype int64→bool** (호출자 2곳 모두 `> 0` 임계화 → PNG 동일) | Blocks `diff -rq` 0 + differential test |
| **S6a+b** 모델 은퇴 + autocast clear | `SCSam3Video.py:13-17`, `:29-31`; `runMVSeg.py:240`, `:309-311` | GPU 6.0 / 5.4 / 5.5 GiB, host 1.27 / 1.02 / 1.22 GiB | **없음** (은퇴 시점 이후 미참조) | Blocks `diff -rq` 0 |
| **S6c** fp64 → fp32 프레임 | `io_utils.py:204-206` | GPU 0.51 / 0.20 / 0.24 GiB, host 동량 | **없음** (반올림 idempotent) | Blocks `diff -rq` 0 |

---

## 보류 항목

### H1. S5 — `cached_frame_outputs` 트림 (**결함 확인, 재설계 필요**)

**무엇이었나** — `SCSam3VideoInferenceNewMem.py:1274-1279`의 partial-propagation 루프에서 `_cache_frame_outputs` 직후 오래된 프레임 항목을 삭제하는 env-gated 트림(`SCSAM3_TRIM_CACHED_OUTPUTS`).

**왜 보류인가** — 두 번째 검토자가 실제 결함을 찾았다. 제안된 창이 `abs(f - keep_frame_idx) >= keep`라서 **아직 방문하지 않은 앞쪽 프레임까지 삭제한다.** 그리고 partial-propagation 루프는 설계상 "리파인된 객체를 각 프레임에 이미 캐시된 VG 예측과 **머지**"하는 것이다(`:1121` docstring, `:1248-1249` 주석) — `_build_tracker_output`(`:1261` → `:559-572`)이 **모든** 프레임에서 그 프레임의 기존 캐시 항목을 머지 베이스로 읽는다. 트림이 앞쪽을 지우면 `:1245-1246`이 빈 dict를 새로 만들고, 그 프레임부터 리파인된 객체 하나만 남고 **나머지 객체 마스크가 조용히 사라진다.** 추가된 `propagation_fetch` assert는 이 경로를 잡지 못한다(fetch가 일어나지 않으므로).

MVSeg 러너 경로에서는 재현되지 않는다(temporal 세션에 full VG propagation을 돌리지 않아 start 이후 프레임 항목이 애초에 없다). 하지만 데모/인터랙티브 경로(text prompt → propagate → refine → propagate)는 깨진다.

**손대기 전에 확립해야 할 것** — 두 가지를 모두 만족하는 재설계:
1. 창을 **엄격히 과거 방향**으로만 (`keep_frame_idx - f >= keep`, reverse면 부호 반전)
2. 해당 세션이 **한 번이라도 `propagation_full`을 돈 적이 있으면 트림을 완전히 끈다** (`action_history` 검사, 또는 항목이 이 partial 루프에서 만들어졌는지 표시)

그리고 데모 경로(`SCSam3Video.RunNaiveTracking` → `propagation_direction` 기본 `"both"` → `start_frame=0`이면 `propagation_fetch`)에서 assert가 뜨지 않는지 실행 확인.

**부가 결함** — `int(os.environ.get("SCSAM3_TRIM_CACHED_OUTPUTS", "0"))`은 변수가 빈 문자열로 export되면 import 시 ValueError. `cached_frame_outputs_trimmed` 플래그가 `reset_state`(`:110`)에서 지워지지 않음.

**중요도** — 이것이 세 실패 데이터셋의 host RAM을 여는 **유일한** 항목이다. 「예상 결과」 참조. **단계 2의 1순위.**

### H2. S5b — seeding 응답 생략 (`return_output=False`) (**분리 후 재심사 가능**)

`add_tracker_new_mask`에 `return_output` 플래그를 넣어 `_build_tracker_output` + `_postprocess_output`의 O(objects²) 재구축을 건너뛰는 부분. **두 검토자 모두 이 절반은 output-identical로 판정했고, 발견된 결함은 H1(트림) 쪽에만 있다.** 다만 하나의 패치 묶음으로 제출되어 함께 보류되었다.

손대기 전에 확립할 것: H1과 완전히 분리된 패치로 다시 제출하고 두 검토자를 다시 통과시킬 것. 주의 사항 — 응답 형태가 `{"frame_index":..., "outputs": None}`으로 바뀌므로 **cross-view 세션(`AddReferenceMask`, `runMVSeg.py:184-190`)에는 절대 켜지 않는다**(`:190`이 `out_obj_ids`를 읽는데, 그것은 생략되는 필터링의 산물이다). 지속적 GPU 절감은 S1과 중복이며, 고유한 이득은 시간(측정: AlexaMeade 120 s / CoffeeMartini 217 s / FlameSteak 260 s의 seeding)과 GPU transient(호출당 ~1.7 GiB → ~5 MiB)다.

### H3. fp16 프레임 저장 (**출력 변경 — 적용 금지**)

감사 보고서 `REPORT.md:183`은 `io_utils.py`에서 fp16을 권하며 "결과 동일"이라 단정하지만 **그 주장은 검증에서 반증되었다.** upstream(`sam3/model/io_utils.py:59, 102, 197, 396`)이 fp16을 쓴다는 것도 근거가 되지 않는다 — upstream은 PIL 리사이즈 + fp16 정규화라 이 fork와 이미 수치적으로 다르며, dtype만 맞춰도 upstream 숫자가 재현되지 않고 게시된 J&F만 무효화된다. S6(c)의 float32가 여기서 얻을 수 있는 전부다.

### H4. bf16 autocast 컨텍스트 자체 (**do-not-touch**)

`SCSam3TrackerPredictorNewMem.py:51-52`, `SCSam3TrackerPredictor.py:51-52`의 영구 열린 `torch.autocast`를 종료하거나 범위를 좁히는 것, 그리고 `model.bfloat16()`으로 가중치를 미리 캐스팅하는 것 — 둘 다 어떤 op이 bf16으로 도는지를 바꾸고, `sam3/model/decoder.py:71`이 의도적으로 fp32로 돌리는 FFN을 깨뜨린다. **모든 마스크가 바뀐다.**

### H5. `super()._apply_non_overlapping_constraints` 내부 (`sam3_tracker_base.py:1115-1132`)

S4 이후 함수 피크가 이쪽으로 이동한다(DS1 1743.8 MiB, DS3 5023.0 MiB). 여기를 줄이려면 argmax/clamp 의미론을 건드려야 하므로 별개의, 훨씬 위험한 사이트다. 손대기 전에: fp32 스코어 맵의 argmax 타이브레이킹(최저 obj index)이 유지되는지 differential test로 확립할 것.

### H6. tracker 시드 마스크를 `storage_device`(cpu)로 이동 (**기각**)

S3의 대안. `offload_state_to_cpu` 기본 True이므로 6~7 GiB가 host RAM으로 넘어간다 — 지금 호스트를 죽이고 있는 바로 그 자원. 게다가 1485회의 device→host 복사가 추가된다.

### H7. C1 wrap 가드 / C2 clobber 수정 / window 인자화 (**출력 변경, 의도적**)

`SCSam3TrackerPredictorNewMem.py:1253-1262`의 음수 인덱스 wrap 가드, `selected_cond_outputs` 덮어쓰기 수정, `range(-4,0)` 인자화. 전부 **의도적으로 출력을 바꾸는** 알고리즘 수정이다. 단계 2로. 단계 1과 절대 섞지 않는다 — 섞으면 `diff -rq` 검증이 무의미해진다.

### H8. `--track-cams written` / `closure`

세션 수를 바꾸므로 출력이 바뀐다. `written`(3세션)은 `output_dicts[-4]` IndexError까지 낸다. 단계 1 실행은 항상 `--track-cams all`.

### H9. sibling 폴더 미적용

`demoSCSam3OneStageNew`, `demoSCSam3TwoStageNew`, `demoSCSam3ForSam2*`, `demoSCSam3OneStage`, `demoSCSam3TwoStage`가 같은 저자 패치를 갖고 있으나 **의도적으로 건드리지 않는다.** 원본 보존이 목적이다.

### H10. `add_tracker_new_points` 경로의 O(N²) seeding

`SCSam3VideoInferenceNewMem.py:1520-1765`. MVSeg는 mask 프롬프트만 쓰므로 미적용. 데모의 `AddPoint` 경로에만 해당.

### H11. 다중 GPU

`SCSam3Video.py:15`가 `range(torch.cuda.device_count())`를 넘기는데, GPU가 2개 이상이면 두 번째 predictor 생성 시 `assert not torch.distributed.is_initialized()`(`SCSam3VideoPredictor.py:423`, `SCSam3VideoPredictorNewMem.py:429`)에서 죽는다. **패치 이전부터 그렇다.** 단계 1의 모든 분석은 단일 GPU 전제이며, world_size>1은 정적 추론만 했다. 프로덕션은 단일 GPU 유지.

---

## 예상 결과

호스트: **GPU NVIDIA RTX 6000 Ada, 49,140 MiB = 47.99 GiB** / **시스템 RAM 125 GiB (가용 ~111 GiB)**.

마스크 1장 바이트 (bool): 2560×1920 = 4.6875 MiB, 2704×2028 = 5.2296 MiB. fp32 video-res는 ×4.

### GPU

| | AlexaMeadeExhibit 45×33 | CoffeeMartini 18×66 | FlameSteak 21×68 |
|---|---|---|---|
| 현재 피크 | **~45.8 GiB**에서 OOM (44.87 GiB allocated + 976 MiB 요청, 첫 프레임 0/22) | 미측정 (host RAM으로 먼저 사망) | 미측정 (host RAM으로 먼저 사망) |
| S1 | −6.95 | −6.40 | −7.64 |
| S3 | −6.95 | −6.40 | −7.64 |
| S6a+b+c | −6.5 | −5.6 | −5.7 |
| S4 (피크) | −1.39 | −4.00 | −4.13 |
| **합계** | **−21.8 GiB** | **−22.4 GiB** | **−25.1 GiB** |
| **예상 피크** | **~24.0 GiB** | (기준선 미상, 여유 −22.4) | (기준선 미상, 여유 −25.1) |

**GPU 판정: 세 데이터셋 모두 47 GiB 안에 들어간다.** AlexaMeadeExhibit는 산술이 명확하다 — 45.8 → 24.0 GiB, 카드 용량의 절반. CoffeeMartini/FlameSteak는 GPU 기준선이 측정된 적이 없으나(host RAM으로 먼저 죽었다), 뷰 수가 18/21로 45보다 훨씬 적고 S4 절감이 오히려 더 크므로(4.0/4.1 GiB) GPU가 병목이 될 가능성은 낮다.

### Host RAM — **여기서 문제가 남는다**

| | AlexaMeadeExhibit | CoffeeMartini | FlameSteak |
|---|---|---|---|
| S2 절감 (시드, 1회성) | −27.80 GiB | −25.62 GiB | −30.56 GiB |
| S6 절감 | −1.8 GiB | −1.2 GiB | −1.5 GiB |
| **단계 1 합계** | **−29.6 GiB** | **−26.8 GiB** | **−32.1 GiB** |
| 남는 프레임당 성장 (`cached_frame_outputs`) | 45 × 33 × 4.6875 MiB = **6.80 GiB/frame** | 18 × 66 × 5.2296 MiB = **6.07 GiB/frame** | 21 × 68 × 5.2296 MiB = **7.29 GiB/frame** |
| 22 프레임 누적 | **149.6 GiB** | **133.5 GiB** | **160.4 GiB** |
| 가용 111 GiB 대비 | **초과** (~frame 15에서 사망) | **초과** (~frame 17) | **초과** (~frame 14) |

**Host RAM 판정: 세 데이터셋 모두 여전히 들어가지 않는다.** 단계 1은 시드 고정분(25~31 GiB)을 없애지만, `cached_frame_outputs`의 프레임당 6~7 GiB 성장은 그대로다. 이것이 `SCSam3VideoInferenceNewMem.py:554`(저자의 `.cpu()` 패치 `:549-552`가 GPU 누수를 host 누수로 전환한 자리)이고, `logs/mvseg-20260903-114432/OneStageNew-FlameSteak.log`가 `11/22 [25:21<1:14:43, 407.58s/it]`에서 traceback 없이 끊긴 원인이다.

등록 객체 수는 zero-area filter(`SCSam3VideoInference.py:464`) 때문에 GT 객체 수보다 적을 수 있다(감사 보고서는 FlameSteak를 53 등록으로 측정). 53개로 계산해도 21 × 53 × 5.2296 MiB = 5.68 GiB/frame × 22 = **125 GiB** — 여전히 가용 111 GiB를 넘는다. 객체 수 축소로는 해결되지 않는다.

### 결론

- **단계 1은 GPU OOM을 확실히 제거한다.** AlexaMeadeExhibit의 CUDA OOM은 사라진다.
- **단계 1만으로는 세 데이터셋 중 어느 것도 완주하지 못한다.** 실패 모드가 CUDA OOM에서 host RAM thrash로 옮겨갈 뿐이며, 사망 프레임이 11 → 14~17 정도로 늦어진다. AlexaMeadeExhibit는 지금까지 host RAM 사망을 겪지 않았지만(GPU에서 먼저 죽었으므로) 단계 1 이후에는 겪게 된다 — 즉 **AlexaMeadeExhibit의 실패 모드가 바뀐다.**
- **완주에 필요한 것은 H1(`cached_frame_outputs` 트림)의 재설계다.** 트림이 프레임당 1개 항목만 유지하면 resident set은 DS1 6.80 GiB / DS2 6.07 GiB / DS3 7.29 GiB로 평탄해지고, 여기에 단계 1의 절감을 더하면 세 데이터셋 모두 여유 있게 완주한다.
- **호스트 안정성:** 단계 1 적용 후에도 실패 데이터셋을 돌릴 때는 `runMVSegAll.sh`의 `docker run`에 `--memory` 캡을 반드시 걸 것(현재 `runForSam2Three.sh:13`에만 있고 `runMVSegAll.sh:58`에는 없다). 캡 없이 돌리면 host RAM 고갈로 머신이 다시 내려간다.

---

## 단계 2 이후 (착수 전)

단계 1이 끝나고 `diff -rq`가 통과한 뒤에 순서대로.

**2-1. `cached_frame_outputs` 트림 재설계 (H1).** 엄격 과거 방향 창 + `propagation_full` 이력이 있는 세션에서는 비활성화. 세 실패 데이터셋을 실제로 여는 유일한 항목. 단계 1 완료가 선행 조건(그래야 남은 성장이 이 사이트 하나로 격리되어 측정 가능하다).

**2-2. seeding 응답 생략 분리 재심사 (H2, P10).** `return_output=False`를 H1과 완전히 분리한 패치로 재제출. 2-1과 독립이지만 같은 함수를 건드리므로 순서를 정해 적용.

**2-3. 음수 인덱스 wrap 수정 (감사 C1 / P2-(2)).** `SCSam3TrackerPredictorNewMem.py:1254` 뒤에 `if prev_spatial_idx < 0: continue`. **출력 변경** — index 0~3 카메라(scored 36개 중 8개, 그중 6개가 reference)가 받는 이웃 수가 바뀐다. 단계 1 이후에 적용해야 bit-identity 기준선이 확보된 상태에서 변화량을 귀속할 수 있다.

**2-4. `selected_cond_outputs` clobber 수정 (감사 C2 / P2-(3)).** `:1262`를 `_, unselected_cond_outputs1 = ...`로. 한 줄. **출력 변경** — 뷰 0..3이 마지막 카메라의 seed object pointer를 쓰는 결함을 없앤다. 2-3과 함께 나가야 의미가 있다(2-3의 wrap이 이 fallback을 매 프레임 실행시키는 원인).

**2-5. 패키지 내 window ablation (P2-(1)/(5)).** `range(-4,0)` → `range(-self.cross_view_window, 0)`, `build_scsam3.py:494-548`에 인자 + `SCSAM3_XVIEW_WINDOW` env, `--xview-window N`이 `--out`을 `SegMaskSam3XW{N}`으로. W=0이 유일하게 올바른 대조군이다(감사 E2). 2-3/2-4 선행 필수 — 위생 수정 없이 측정한 W 곡선은 wrap/clobber 효과와 섞인다.

**2-6. seed 복구 (P5).** `PropagateAcrossViews` 뒤 `RepairSeeds()`: (view, obj) 면적 < max(64 px, 0.05×median)이면 degenerate 판정, ±4 인덱스 내 최대 면적 뷰를 donor로 spatial 세션에 두 번째 cond frame 추가 후 재전파. Welder −0.034 전부가 camera_0004의 4/7 px seed(감사 A3). 착수 전 무비용 진단부터: 기존 `SegMaskSam3OneStageNew/<cam>/0/*.png`에서 degenerate seed 개수를 데이터셋별로 집계해 상한을 파악할 것. 구현 전 `SCSam3VideoInference.py:1855-1870`의 refine 분기가 새 tracker state를 만들지 않는지 로그로 확인할 것.

---

## 검증 절차

경로 상수: `ROOT=/home/sjpark/Documents/SCSegmentation`, 컨테이너 이미지 `scsam3`.

### 0. 등록 확인 (모델 로드 없음)

```bash
cd /home/sjpark/Documents/SCSegmentation/SCSam3
python3 runMVSeg.py Blocks --algo MVOpt --track-cams all --dry-run
```

배너의 `algorithm      MVOpt  (demoSCSam3MVOpt)`와 출력 경로를 눈으로 확인.

### 1. 사이트 하나 적용 → 완료 데이터셋 하나 → diff

`S1` 적용 후:

```bash
docker run --rm --gpus all --shm-size=32g --memory=100g \
  -v /:/host -w /host/home/sjpark/Documents/SCSegmentation/SCSam3 scsam3 \
  python runMVSeg.py Blocks --algo MVOpt --out SegMaskSam3MVOpt_s1 \
                     --track-cams all --overwrite

docker run --rm -v /:/host scsam3 \
  chown -R "$(id -u):$(id -g)" /host/home/sjpark/Documents/SCSegmentation/Data/MVSeg

diff -rq /home/sjpark/Documents/SCSegmentation/Data/MVSeg/Blocks/SegMaskSam3OneStageNew \
         /home/sjpark/Documents/SCSegmentation/Data/MVSeg/Blocks/SegMaskSam3MVOpt_s1
```

**출력 0줄이 통과 조건.** 한 줄이라도 나오면 그 사이트를 되돌리고 원인을 찾는다.

`S2`, `S3`, `S4`, `S6`도 각각 `--out SegMaskSam3MVOpt_s2` … `_s6`로 같은 절차를 반복한다. 사이트별로 폴더를 나누면 실패 시 이분 탐색이 필요 없다.

### 2. 완료 데이터셋 확대 (전체 단계 1 적용 후)

`Blocks` 하나로는 부족하다. 최소 아래 넷을 돌린다 — 뷰 수, 객체 수, degenerate seed 유무가 서로 다르다:

```bash
for ds in Blocks Welder Painter Fencing; do
  docker run --rm --gpus all --shm-size=32g --memory=100g \
    -v /:/host -w /host/home/sjpark/Documents/SCSegmentation/SCSam3 scsam3 \
    python runMVSeg.py "$ds" --algo MVOpt --out SegMaskSam3MVOpt \
                       --track-cams all --overwrite
done
docker run --rm -v /:/host scsam3 \
  chown -R "$(id -u):$(id -g)" /host/home/sjpark/Documents/SCSegmentation/Data/MVSeg

for ds in Blocks Welder Painter Fencing; do
  echo "== $ds"
  diff -rq "/home/sjpark/Documents/SCSegmentation/Data/MVSeg/$ds/SegMaskSam3OneStageNew" \
           "/home/sjpark/Documents/SCSegmentation/Data/MVSeg/$ds/SegMaskSam3MVOpt"
done
```

여유가 있으면 나머지 8개(`AlexaMeadeFacePaint Barn Breakfast Carpark Dog Frog MATF PoznanStreet`)까지 12개 전부. **12개 전부 0줄이면 bit-identity가 확립된다.**

### 3. 실패 데이터셋 세 개

```bash
for ds in AlexaMeadeExhibit CoffeeMartini FlameSteak; do
  docker run --rm --gpus all --shm-size=32g --memory=100g \
    -v /:/host -w /host/home/sjpark/Documents/SCSegmentation/SCSam3 scsam3 \
    python runMVSeg.py "$ds" --algo MVOpt --out SegMaskSam3MVOpt \
                       --track-cams all --overwrite \
    2>&1 | tee "/home/sjpark/Documents/SCSegmentation/SCSam3/logs/mvopt-${ds}.log"
done
```

`--memory=100g`는 필수다 — 「예상 결과」대로 단계 1만으로는 세 개 모두 host RAM을 초과하므로, 캡이 없으면 호스트가 다시 내려간다. 캡이 있으면 컨테이너만 OOM-kill되고 로그가 남는다.

기대 결과: **세 개 모두 CUDA OOM은 사라지고, host RAM 캡에서 멈춘다.** 로그의 마지막 `frame N written`을 기록할 것 — 이것이 H1 재설계의 기준선이 된다. 완주하면 계산이 보수적이었다는 뜻이므로 그것도 그대로 기록한다.

### 4. J&F

```bash
docker run --rm -v /:/host \
  -w /host/home/sjpark/Documents/SCSegmentation/Data/MVSeg scsam3 \
  python eval_jf.py \
    --methods SegMaskSam3OneStageNew SegMaskSam3MVOpt \
    --out jf_raw_mvopt.json

docker run --rm -v /:/host scsam3 \
  chown "$(id -u):$(id -g)" \
    /host/home/sjpark/Documents/SCSegmentation/Data/MVSeg/jf_raw_mvopt.json

python3 /home/sjpark/Documents/SCSegmentation/Data/MVSeg/report_jf.py \
  --raw jf_raw_mvopt.json \
  --methods SegMaskSam3OneStageNew SegMaskSam3MVOpt \
  --common --no-missing
```

`--common`은 두 방법이 **모두** 결과를 가진 (dataset, camera)만 집계하므로, MVOpt만 완주한 데이터셋이 섞여 비교가 오염되는 것을 막는다.

**기대 결과: `SegMaskSam3OneStageNew`와 `SegMaskSam3MVOpt`의 J&F가 소수점 이하 전 자리까지 동일.** `diff -rq`가 0줄이었으므로 당연히 그래야 하며, 다르게 나오면 diff 절차 자체를 의심할 것(폴더 누락, 부분 실행 등).

SAM 2 기준선과 나란히 보려면:

```bash
python3 /home/sjpark/Documents/SCSegmentation/Data/MVSeg/report_jf.py \
  --raw jf_raw.json jf_raw_mvopt.json \
  --methods SegMask1 SegMaskNew1 SegMaskNew2 SegMaskNew3 \
            SegMaskSam3OneStageNew SegMaskSam3MVOpt \
  --common --no-missing
```

### 5. 메모리 실측 (선택, 권장)

각 실행에서 `docker stats --no-stream` 또는 컨테이너 내부 `/proc/<pid>/status`의 `VmHWM`을 프레임마다 찍어 「예상 결과」의 산술과 대조한다. 특히 확인할 것:
- 프레임당 host RSS 증가분이 표의 6.80 / 6.07 / 7.29 GiB/frame과 맞는가 (맞으면 남은 성장이 `cached_frame_outputs` 하나로 격리된 것이 확인되고, H1 재설계의 기대 효과를 그대로 신뢰할 수 있다)
- seeding 직후 GPU `torch.cuda.memory_allocated()`가 S1+S3+S6 합계만큼 내려갔는가