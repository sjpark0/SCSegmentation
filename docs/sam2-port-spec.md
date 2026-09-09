> **보류 중인 사양서입니다 (2026-09-09).** 이 문서는 2026-09-09 새벽에 작성됐고, 그날 오후
> MUVOD 벤치마크의 기준선(XMem)이 확보되면서 **착수를 보류했습니다.** SAM 2 열이 필요했던 이유가
> "외부와 견줄 비교군이 없다"였는데, 이제 데이터셋 출처 논문의 공개 수치가 그 자리를 대신합니다
> ([muvod-comparison.md](muvod-comparison.md)).
>
> 그럼에도 저장소에 남기는 이유는 셋입니다. ① Part A가 담은 측정 결과(설정 매칭 가능 여부,
> `fill_hole` 무효 증명, `SegMaskNew1` 재현 불가와 그 출처 결함)는 보류와 무관하게 유효한 사실입니다.
> ② Part B의 이식 계획은 나중에 SAM 2로 같은 실험을 돌릴 때 그대로 쓸 수 있습니다.
> ③ 이 문서가 임시 디렉터리에만 있어 재부팅 한 번으로 사라질 상태였습니다.
>
> **아래 본문의 "다음 단계"와 GPU 계획은 실행되지 않았습니다.** 착수 전에 §0의 불변 규칙과
> 기준 커밋(`3b292f7`)이 아직 유효한지 다시 확인하십시오.

---

# SPEC_SAM2 — SAM 2 실험 사양서 (A: 설정 매칭 재실행, B: XW 이식)

작성 2026-09-09 · 기준 커밋 `3b292f7` (clean) · 저장소 `/home/sjpark/Documents/SCSegmentation`
근거 문서: `docs/analysis/REPORT.md`(E6·C1·C2·행 11), `docs/experiments.md`, `docs/operations.md`,
`docs/phase2-control.md`, `docs/phase3-neighbourhood.md`, `docs/phase3-conditioning.md` §8·§9(GPS4 채택).

**이 사양서 하나만 읽고 구현할 수 있게 씁니다.** 모든 경로는 절대경로 또는 저장소 루트 기준입니다.
저장소 루트를 아래에서 `$ROOT = /home/sjpark/Documents/SCSegmentation` 로 씁니다.

---

## 0. 불변 규칙 (어기면 결과가 무효)

| # | 규칙 | 이유 |
|---|---|---|
| R1 | `SCSam2/demo/**`, `SCSam2/sam2/**` 를 **절대 수정하지 않습니다**. 새 폴더 `SCSam2/demoSCSam2MVOpt/` 에서만 작업합니다. | 발표본 출처 보존. `demoSCSam3MVOpt` 가 만들어진 방식과 동일 |
| R2 | 도커 이미지 **`scsam2` 를 재빌드하지 않습니다**. | 재빌드하면 `/opt/sam2` 가 바뀌어 바이트 동일성 가드가 무효 |
| R3 | SAM 2는 **`scsam3` 이미지에서 절대 돌지 않습니다**. (측정: `scsam3` 에 `sam2` 패키지 없음 — `importlib.util.find_spec('sam2') is None`) SAM 2는 `scsam2`(torch 2.6.0+cu118, py3.11.11), SAM 3는 `scsam3`(torch 2.10.0+cu128, py3.12.3). | 실행 스택 분리 |
| R4 | 파이썬을 `SCSam2/` 또는 `SCSam3/` **안에서** 띄우지 않습니다(각각 `sam2/`, `sam3/` 체크아웃이 설치 패키지를 가림). 항상 `$ROOT` 또는 새 폴더에서 실행. `SCSam2/demo/build_sam.py:20-32` 가 이 상황을 명시적으로 `RuntimeError` 로 막고 있습니다. | 함정 3 |
| R5 | `Data/` 의 기존 폴더를 **삭제·수정하지 않습니다**. 새 출력은 새 이름으로만. | 발표 수치 보존 |
| R6 | 폴더 바이트 비교는 항상 `diff -rq -x MANIFEST.json A B`. | 매니페스트는 `written_at`·`path` 때문에 항상 다름 |

---

## 1. 실행 스택 — 측정으로 확정한 사실

| 항목 | 측정값 | 확인 방법 |
|---|---|---|
| SAM 2 코드의 `sam2` 임포트 | **`/opt/sam2/sam2`** (이미지 빌드 시 `COPY ./sam2 /opt/sam2` + `pip install -e .`) | `docker run … scsam2 python -c "import sam2; print(sam2.__path__)"` → `['/opt/sam2/sam2']` |
| `/opt/sam2/sam2` vs `SCSam2/sam2/sam2` | **바이트 동일** (`diff -rq --exclude=__pycache__` exit 0) | 컨테이너 안에서 직접 diff |
| ⚠️ 함정 | 저장소의 `SCSam2/sam2/**` 를 고쳐도 **실행에 반영되지 않습니다**(이미지 안 복사본이 돕니다). R2와 합쳐 "업스트림 SAM 2 코드는 손댈 수 없다"가 됩니다 | — |
| `sam2._C` (CUDA 확장) | **없음** — `ImportError: cannot import name '_C' from 'sam2'` | `docker run … scsam2 python -c "from sam2 import _C"` |
| 모델 설정 경로 | 하이드라 config module `sam2` (`/opt/sam2/sam2/__init__.py` 의 `initialize_config_module("sam2")`). `"./configs/sam2.1/sam2.1_hiera_l.yaml"` 은 **설치 패키지 안**에서 풀립니다 | — |
| 체크포인트 | `"../models/sam2.1_hiera_large.pt"` = `$ROOT/SCSam2/models/sam2.1_hiera_large.pt` (898 MB, 존재 확인) — 새 폴더도 `SCSam2/` 바로 아래이므로 상대경로가 그대로 맞습니다 | — |
| pytest | `scsam2` 이미지에 **없음**. 네트워크는 열려 있어 `pip install --target` 으로 벤더링 가능(검증 완료) | §B.5 |

---

# PART A — 설정 매칭 재실행 (실험 행 11 / E6)

## A.1 설정 표 — SAM 2 vs SAM 3, 매칭 가능 여부

발표된 SAM 2 열은 `Data/MVSeg/<ds>/SegMaskNew1` (15개, `jf_raw.json`·`jf_v2.json`).
발표된(채택된) SAM 3 열은 `SegMaskSam3XW1GPS4` (phase3-conditioning §9), 대조군 `SegMaskSam3XW0`.

| # | 설정 | SAM 2 (발표본) | SAM 3 (XW1GPS4) | 매칭? | A에서 하는 일 |
|---|---|---|---|---|---|
| 1 | `fill_hole_area` **설정값** | **8** — `SCSam2/demo/build_sam.py:266` (살아 있는 두 번째 `build_sam2_video_predictor_new`) | **0** — `SCSam3/demoSCSam3MVOpt/SCSam3Video.py:18,35` 가 런타임에 0으로 덮어씀 (빌더 기본 `build_scsam3.py:516,576` 도 0) | 가능 | 새 폴더에서 0으로 |
| 1b | `fill_hole` **실제 효과** | **없음** — `sam2._C` 미빌드 → `get_connected_components` 가 `ImportError`, `fill_holes_in_mask_scores` 가 `except Exception` 으로 삼키고 **입력을 그대로 반환** (`SCSam2/sam2/sam2/utils/misc.py:312-334`) | 없음 | **이미 매칭됨** | 증명만 하고 지나감 (§A.2 F-test, §A.5 G2) |
| 2 | 출력 non-overlap | **없음** — `non_overlap_masks=False` (`sam2_video_predictor.py:26`), 그래서 `_get_orig_video_res_output` (:390-402) 은 원시 per-object 로짓을 그대로 반환 | **항상 적용** — `SCSam3VideoInferenceNewMem.py:510-517`: **이진화된** 마스크에 `_apply_object_wise_non_overlapping_constraints(masks, tracker_probs, background_value=0)`. 플래그와 무관하게 `if out_binary_masks.shape[0] > 1` 이면 무조건 (참고: `non_overlap_masks_for_output` 은 빌더에서 **False** 이고 :538-539 경로는 안 탑니다) | 가능하지만 **`non_overlap_masks=True` 로는 안 됩니다** | SAM 3의 **object-wise** 규칙을 이식 (§A.3) |
| 3 | 메모리 인코더 non-overlap | False (`sam2_base.py:63`) | False (`build_scsam3.py:505,565`) | 이미 매칭 | — |
| 4 | `dynamic_multimask_via_stability` / delta / thresh | True / 0.05 / 0.98 (`build_sam.py:259-262`) | True / 0.05 / 0.98 (`build_scsam3.py:511-513`) | 이미 매칭 | **`apply_postprocessing=True` 를 유지**해야 합니다 |
| 5 | `binarize_mask_from_pts_for_mem_enc` | True (`build_sam.py:264`) | SAM 3 트래커에 해당 인자 없음(다른 기제) | 매칭 대상 아님 | 손대지 않음 |
| 6 | `max_cond_frames_in_attn` | **−1** (SAM2Base 기본, yaml 미설정) | **4** (`build_scsam3.py:507,567`) | 가능, 그러나 **이 데이터셋에서는 무의미** — 세션당 조건 프레임이 정확히 1개이므로 `select_closest_cond_frames` 의 조기반환 분기(`sam2_utils.py:32-34`)가 두 값에서 동일 | 매칭 목록에 넣지 않음 (근거 기록) |
| 7 | `num_maskmem` | **7** (`sam2.1_hiera_l.yaml:88`) | **7** (`build_scsam3.py:488,548`) | 이미 매칭 | — |
| 8 | `max_obj_ptrs_in_encoder` | 16 (yaml 미설정, 기본) | 16 (`sam3_tracker_base.py` 기본) | 이미 매칭 | — |
| 9 | `memory_temporal_stride_for_eval` | 1 | 1 | 이미 매칭 | — |
| 10 | 메모리 선택(`use_memory_selection` / `frame_filter` / `eff_iou_score`) | **없음** — 다섯 이름 전부 `SCSam2/sam2/sam2/**` 와 `SCSam2/demo/*.py` 에서 0건 | **켜짐** (`build_scsam3.py:517,577` `use_memory_selection=apply_temporal_disambiguation`, 기본 True), `mf_threshold=0.01` | **매칭 불가**(아키텍처) | A의 범위 밖. B의 G 손잡이 해석에 결정적 (§B.4.3) |
| 11 | **low-res 마스크 격자** | **256** = `image_size 1024 // 4` (`sam2.1_hiera_l.yaml:89`, `sam2_video_predictor.py:430`) | **288** = `1008 // 14 * 4` (`build_scsam3.py:547,550`) | **매칭 불가** — 백본 stride/체크포인트가 바뀝니다 | 논문에 잔여 교란항으로 명시 (§A.7) |
| 12 | 기준 카메라 규칙 | GT id 최대 카메라 (`sam2_demoVideoNew_maskSingleInputMVSeg.py:44-56`) | 동일 (`runMVSeg.py:159-175 pick_reference`) | 이미 매칭 | — |
| 13 | 채점 대상/프레임 | `cam_list` 3대, `start_frame`+21 | 동일 | 이미 매칭 | — |

**행 11의 문구를 문자 그대로 실행하면 안 됩니다.** `REPORT.md:233` / `ROADMAP.md:161` 은
"SAM 2 `apply_postprocessing=False`, `non_overlap_masks=True`" 라고 적혀 있는데, 이 두 가지는 각각 틀렸습니다.

* `apply_postprocessing=False` 는 `fill_hole_area` 뿐 아니라 **`dynamic_multimask_via_stability` 와
  `binarize_mask_from_pts_for_mem_enc` 까지 꺼 버립니다** (`build_sam.py:259-266`). SAM 3는
  `dynamic_multimask` 를 **켜고** 있으므로(4번 행), 이건 매칭이 아니라 새로운 불일치를 만듭니다.
* `non_overlap_masks=True` 는 **다른 규칙**입니다: SAM 2의 `_apply_non_overlapping_constraints`
  (`sam2_base.py:891-909`)는 video-res **로짓**에 대한 픽셀별 argmax 이고, SAM 3의 것은
  **이진 마스크 + 객체 점수** argmax 입니다. 겹치는 픽셀에서 승자가 다릅니다.

따라서 매칭은 **정확히 두 개**입니다: `fill_hole_area 8 → 0`, 그리고 **SAM 3의 object-wise
non-overlap 규칙을 출력단에 이식**. `apply_postprocessing` 은 True 로 둡니다.

## A.2 fill_hole 절반은 no-op임을 증명 (측정 완료)

세 가지 증거가 이미 있습니다.

1. **코드**: `_C` 임포트 실패 → `fill_holes_in_mask_scores` 가 `except Exception` 으로 삼키고 `mask = input_mask` 반환.
2. **로그**: `SCSam3/logs/sam2-recheck-CoffeeMartini.log` 에 두 경로 모두에서 경고 —
   `/opt/sam2/sam2/sam2_video_predictor.py:786` (spatial pass) 와 `SCSam2VideoPredictorNew.py:818` (temporal pass).
3. **발표본 픽셀 측정 (이번에 새로 측정, CPU만)**: 발표본 마스크에 fill_hole=8이 지웠어야 할 크기의
   **구멍이 그대로 남아 있습니다**. (low-res 256²에서 면적 8 → full-res 환산 임계)

| 데이터셋 | 해상도 | full-res 환산 임계 | `SegMaskNew1` 내부 구멍 수 / 최소면적 / ≤임계 | `SegMaskNew3` |
|---|---|---|---|---|
| CoffeeMartini | 2028×2704 | ~669 px | 43 / **1 px** / 31 | 45 / 1 px / 33 |
| Fencing | 1080×1920 | ~253 px | 45 / **2 px** / 35 | 67 / 1 px / 45 |

(각 400장 무작위 표본, 경계에 닿지 않는 배경 연결성분만 카운트. 스크립트는
`/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/9786e830-b342-4be7-bc11-e7701415bd8d/scratchpad/sam2/holes2.py` 에 있습니다.)

**결론**: E6가 전제한 "SAM 2는 8픽셀 구멍을 메웠고 SAM 3는 안 메웠다"는 **효과 면에서 거짓**입니다.
두 스택은 이미 "구멍 안 메움"으로 일치하며, **F +0.006 의 어떤 부분도 fill_hole 에 귀속될 수 없습니다.**
A는 이 사실을 (i) CPU 테스트로 고정하고 (ii) 1개 데이터셋 GPU A/B로 재확인한 뒤,
**실질적으로는 non-overlap 한 변수만 조작하는 실험**이 됩니다.

## A.3 매칭 컬럼을 어떻게 만드는가 — 오프라인 유도 (GPU 1회)

핵심 관찰: SAM 2·SAM 3 **양쪽 모두 non-overlap 을 출력단에서만** 적용하고 메모리로 되먹이지 않습니다.
SAM 2가 저장하는 `pred_masks` 는 제약 이전의 low-res 로짓(`SCSam2VideoPredictorNew.py:815-834`)이고,
SAM 3의 제약은 응답 빌더 안에서만 돕니다. 따라서

> **매칭 컬럼 = (제약 없는 실행이 만든 이진 마스크) + (per-object 점수) 를 CPU에서 후처리한 것**

이며, **GPU 재실행은 한 번이면 됩니다**. 그 한 번의 실행이 동시에 §A.4의 바이트 동일성 가드입니다.

### A.3.1 이식할 규칙 (SAM 3와 동일한 연산)

```python
# 입력: masks_bool [N, H, W] (out_mask_logits > 0.0),  scores [N] (per-object)
# SAM 3: SCSam3TrackerPredictorNewMem.py:2040-2076, background_value=0, 이진 입력
pred = torch.where(masks_bool[:, None], scores[:, None, None, None], 0.0)   # [N,1,H,W]
keep = torch.argmax(pred, dim=0, keepdim=True) == torch.arange(N)[:, None, None, None]
out  = masks_bool[:, None] & keep                                           # bool
```
`argmax` 의 동점 처리(첫 최대 채택)까지 `sam2_base.py:891-909` 의 `torch.argmax` 와 동일합니다.
`N == 1` 이면 SAM 3도 건너뛰므로(`if out_binary_masks.shape[0] > 1`) SAM 2 쪽도 건너뜁니다.

### A.3.2 SAM 2의 점수는 무엇인가

SAM 3는 `out_tracker_probs`(트래커 per-object 점수)를 씁니다. SAM 2의 대응물은
`object_score_logits` (모든 저장 출력에 이미 있음, `SCSam2VideoPredictorNew.py:826-834`).
**단조 변환은 argmax를 바꾸지 않으므로** 로짓을 그대로 쓰든 `sigmoid` 를 씌우든 결과가 동일합니다
— 이 동치는 사양의 일부이고 CPU 테스트로 고정합니다(§B.5 T-A3).

### A.3.3 구현 위치

새 폴더 `SCSam2/demoSCSam2MVOpt/` 안, 다음 두 곳:

1. **점수 덤프** — 러너 `SCSam2/runMVSeg.py` 의 쓰기 루프에서, `--dump-scores` 가 켜져 있을 때
   **마스크 폴더 바깥**의 형제 디렉터리에 씁니다:
   `Data/MVSeg/<ds>/<OUT>_meta/<cam>/<frame>.json` = `{"obj_ids": [...], "object_score_logits": [...]}`.
   *마스크 폴더 안에 아무 파일도 넣지 않습니다* — `diff -rq -x MANIFEST.json` 가드를 약화시키지 않기 위해서입니다.
2. **오프라인 후처리** — 새 파일 `SCSam2/demoSCSam2MVOpt/apply_nonoverlap.py`
   (호스트 python + numpy + cv2 필요 → `scsam2` 컨테이너에서 CPU로 실행):
   ```
   python apply_nonoverlap.py --src Data/MVSeg/<ds>/SegMaskSam2Legacy \
                              --meta Data/MVSeg/<ds>/SegMaskSam2Legacy_meta \
                              --dst  Data/MVSeg/<ds>/SegMaskSam2Matched
   ```
   프레임마다 존재하는 객체 PNG를 모두 읽어 `masks_bool`, 메타에서 `scores` 를 만들고 위 규칙을 적용해
   같은 파일 이름으로 다시 씁니다. **객체 집합·파일 이름·해상도는 그대로**입니다.

### A.3.4 온라인 경로(교차검증용)

같은 계산을 모델 안에서도 할 수 있게 러너에 `--nonoverlap {off,objectwise}` 를 둡니다
(`off` 가 기본이고 발표본 경로). `objectwise` 는 새 폴더의 `SCSam2VideoNew.py` 쓰기 지점 직전에서
`out_mask_logits > 0.0` 과 `object_score_logits` 로 A.3.1을 적용합니다.
**두 경로는 반드시 같은 PNG를 내야 합니다** — 2개 데이터셋에서 `diff -rq` 0 으로 확인(§A.5 G4).
오프라인 경로를 정본으로 삼는 이유는 GPU 시간이 절반이고, 매칭 전/후가 **같은 한 번의 실행**에서
나오므로 재현 오차가 비교에 들어오지 않기 때문입니다.

### A.3.5 fill_hole 매칭

러너에 `--fill-hole-area N` (기본 `None` = 건드리지 않음 = 빌더의 8). `--fill-hole-area 0` 이면
`SCSam2VideoNew.__init__` 에서 `self.predictor.model.fill_hole_area = 0`,
`self.predictor_spatial.model.fill_hole_area = 0` 로 **생성 후 대입** 합니다
(SCSam3Video.py:18,35 와 같은 방식). 하이드라 오버라이드로는 못 합니다 —
`build_sam.py:257-268` 이 호출자 오버라이드 **뒤에** 자기 것을 붙이므로 `++model.fill_hole_area=0` 이
`=8` 에 집니다. `fill_hole_area` 는 호출 시점에 읽히므로(`:817`) 사후 대입이 동등합니다.

## A.4 재현 가드 — 오늘의 환경이 발표본을 만드는가

### A.4.1 무엇을 재현할 수 있는가 (측정으로 확정)

| 폴더 | 오늘 재현 가능? | 근거 |
|---|---|---|
| `SegMaskNew3` | **예, 바이트 단위로** | `SegMaskSam2Recheck`(2026-09-03, `sam2_MVSeg_recheck.py`, 폴더 리터럴 2줄만 다름) vs `SegMaskNew3` → `diff -rq -x MANIFEST.json` **0줄** (CoffeeMartini 4158장). 이번에 재확인함 |
| `SegMaskNew1` (**발표 인용 열**) | **아니오** | 같은 recheck vs `SegMaskNew1` → **2731줄** 차이. `SegMaskNew1` vs `SegMaskNew3` → 2731줄 |
| `SegMaskNew2` | 아니오 | 1345줄 차이 |

`SegMaskNew1` 을 만든 교차시점 수집 루프는 **git 에 없습니다**. 파일이 git 에 처음 들어온 것이
`68b92ca`(2025-11-27)로 모든 발표 폴더가 쓰인 **뒤**이고, `68b92ca` 는 그 이전 상태
(`ec81251`, 2025-07-25: 양측 `range(-num_maskmem//2, num_maskmem//2)`, `frame_idx-1` 읽기,
tpos 행 `num_maskmem-|s|-1`, 경계 가드 있음)를 현재의 `range(-4,0)`·`frame_idx` 읽기·가드 없음·
C2 fallback 으로 **통째로 교체**했습니다. PNG mtime: New1 2025-10-15, New2 2025-11-11~21, New3 2025-11-21.

**따라서 A가 만드는 것은 New1 계열이 아니라 New3 계열입니다.** REPORT 행 11의 출력 이름
`SegMaskNew1_matched` 는 오해를 부르므로 **`SegMaskSam2Matched` 로 바꿉니다**(§A.6).

### A.4.2 새로 발견한 출처 결함 — `AlexaMeadeExhibit/SegMaskNew1` 은 두 실행의 이어붙이기

`find -printf '%TY-%Tm-%Td'` 집계:

```
AlexaMeadeExhibit: 1947장 2025-10-15  +  132장 2025-11-11      ← 유일하게 섞여 있음
그 외 14개 데이터셋: 전부 2025-10-15 단일
```
2025-11-11 에 다시 쓰인 132장 = `camera_0001/0`(33) + `camera_0001/1`(33) + `camera_0003/0`(33) + `camera_0004/0`(33),
즉 **세 카메라의 프레임 0 전부와 camera_0001 의 프레임 1**. 프레임 0은 `--frames all` 집계에 들어가므로
발표된 AlexaMeadeExhibit 숫자는 한 번의 실행 산출물이 아닙니다.
(`--frames inner` 는 첫·마지막 프레임을 빼므로 프레임 0의 영향은 없지만 camera_0001 프레임 1은 남습니다.)
→ **리스크 목록에 올리고, A의 헤드라인은 `--frames all` 과 `inner` 를 함께 보고합니다.**

### A.4.3 가드 절차 (A의 첫 GPU 작업)

```
G1  새 폴더에서 아무 플래그 없이 CoffeeMartini 1개 실행 → --out SegMaskSam2Guard
    diff -rq -x MANIFEST.json Data/MVSeg/CoffeeMartini/SegMaskSam2Guard \
                              Data/MVSeg/CoffeeMartini/SegMaskNew3      ⇒ 0줄 (필수)
G2  Fencing 1개를 --fill-hole-area 0 로 실행 → SegMaskSam2FH0
    diff -rq -x MANIFEST.json <FH0> Data/MVSeg/Fencing/SegMaskNew3      ⇒ 0줄 (fill_hole no-op 실증)
G3  15개 전체 legacy 실행 → SegMaskSam2Legacy (+ _meta 점수 덤프)
    for ds: diff -rq -x MANIFEST.json Data/MVSeg/$ds/SegMaskSam2Legacy \
                                      Data/MVSeg/$ds/SegMaskNew3        ⇒ 전부 0줄 (기대)
G4  Fencing·Blocks 를 --nonoverlap objectwise 로 실행 → SegMaskSam2MatchedOnline
    diff -rq -x MANIFEST.json <online> <오프라인 산출 SegMaskSam2Matched> ⇒ 0줄
```
G1 은 **다른 어떤 것도 하기 전에** 통과해야 합니다.

### A.4.4 아무것도 바이트 재현하지 못하면 (폴백 사다리)

| 단계 | 조건 | 대응 |
|---|---|---|
| F0 | G1·G3 전부 0줄 | 최선. 기준선 = 기존 `SegMaskNew3` 점수(`jf_v2.json`), 매칭 컬럼은 같은 실행에서 유도 |
| F1 | G3 에서 일부 데이터셋이 다름 | 그 폴더를 `SegMaskSam2Legacy` 라는 **독립 method 로 채점**하고 `SegMaskNew3` 대비 재현 오차를 보고. 카메라 단위 \|ΔJ&F\| **< 0.001** 이면 진행 |
| F2 | 재현 오차 ≥ 0.001 인 카메라가 있음 | 진행하되 **그 데이터셋을 A의 1차 판독에서 제외**하고 별도 표로. 원인 후보: 드라이버/이미지 ID(매니페스트에 기록), 비결정성(같은 데이터셋 2회 실행 diff로 분리) |
| F3 | 재현 자체가 실패(실행 오류·대량 불일치) | **A를 중단**하고, E6는 "환경 드리프트로 재현 불가"로 보고. 논문에는 F 차이의 후처리 몫을 **미해결**로 남김 |
| **F0~F2 공통** | — | **1차 판독은 언제나 같은 실행 안의 짝지은 비교** `SegMaskSam2Matched − SegMaskSam2Legacy` 입니다. New1/New3 계열 차이는 이 비교에 **들어오지 않습니다**. `SegMaskNew1` 과의 비교는 헤드라인 표 연속성 용도로만 부차 보고 |

## A.5 출력 폴더 이름

| 폴더 | 데이터셋 | 내용 |
|---|---|---|
| `SegMaskSam2Guard` | CoffeeMartini | G1 가드 산출물 (채점 안 함, 검증 후 삭제 가능) |
| `SegMaskSam2FH0` | Fencing | G2 fill_hole no-op 실증 (채점 안 함) |
| **`SegMaskSam2Legacy`** | 15 | 새 폴더의 무플래그 실행. `SegMaskNew3` 와 바이트 동일이어야 함. **A의 기준선** |
| `SegMaskSam2Legacy_meta` | 15 | per-(cam,frame) `object_score_logits` JSON. 마스크 폴더가 **아님** |
| **`SegMaskSam2Matched`** | 15 | fill_hole 0 + SAM 3 object-wise non-overlap. **A의 처리군** |
| `SegMaskSam2MatchedOnline` | Fencing, Blocks | G4 교차검증 (채점 안 함) |

`SegMaskNew1_matched` 라는 이름은 쓰지 않습니다(§A.4.1). `docs/experiments.md` 의 "출력 디렉토리 목록"과
`docs/ROADMAP.md:161`, `docs/analysis/REPORT.md:233` 을 이 이름으로 갱신합니다.

## A.6 GPU 계획과 소요 시간

측정 근거: 기존 `SegMaskNew3` 폴더의 PNG mtime 범위(= 마스크 쓰기 구간). 여기에 모델 빌드·프레임 로딩
오버헤드를 데이터셋당 +1.5분으로 잡았습니다. 실행은 `--track-cams all`(legacy 필수, §B.3.4).

| 데이터셋 | 카메라 | 측정 마스크 시간 | 예상 총시간 |
|---|---|---|---|
| AlexaMeadeExhibit | 45 | 22.0분 | 24 |
| AlexaMeadeFacePaint | 46 | 10.7 | 12 |
| Barn | 15 | 6.8 | 8 |
| Blocks | 10 | 3.5 | 5 |
| Breakfast | 15 | 6.7 | 8 |
| Carpark | 9 | 3.1 | 5 |
| CoffeeMartini | 18 | 18.1 | 20 |
| Dog | 41 | 6.7 | 8 |
| Fencing | 10 | 1.5 | 3 |
| FlameSteak | 21 | 21.6 | 23 |
| Frog | 13 | 1.5 | 3 |
| MATF | 10 | 6.4 | 8 |
| Painter | 16 | 6.9 | 8 |
| PoznanStreet | 9 | 3.4 | 5 |
| Welder | 46 | 11.8 | 13 |
| **합계** | | **130.7분** | **≈ 2시간 35분 / 스윕** |

| 작업 | GPU 시간 |
|---|---|
| G1 CoffeeMartini 가드 | 20분 |
| G2 Fencing fill_hole A/B | 3분 |
| **G3 15개 legacy 스윕 (+점수 덤프)** | **2시간 35분** |
| G4 Fencing·Blocks 온라인 교차검증 | 8분 |
| 오프라인 매칭 생성 (CPU) | ~5분 |
| **A 합계** | **≈ 3시간 6분** |

한 데이터셋 = 한 컨테이너, `--memory=90g --memory-swap=90g` (operations.md 하드웨어 절: 호스트 RAM 고갈로
머신이 두 번 재부팅됨). SAM 2는 46카메라에서도 완주한 이력이 있어 GPU 48 GiB 는 문제되지 않습니다.

## A.7 채점·집계 명령과 사전 등록 판독

### A.7.1 채점

```bash
cd /home/sjpark/Documents/SCSegmentation
docker run --rm --user $(id -u):$(id -g) -e PYTHONDONTWRITEBYTECODE=1 -v /:/host -w /host$PWD scsam2 \
  python eval/eval_jf.py --methods SegMaskSam2Legacy SegMaskSam2Matched \
      --out /host$PWD/Data/MVSeg/jf_sam2_matched.json
```
(`eval_jf.py` 는 `cv2` 가 필요합니다. `scsam3` 로 돌려도 같은 숫자가 나오지만, SAM 2 산출물은
SAM 2 스택에서 채점하는 편이 출처가 깔끔합니다 — 두 이미지 결과가 같은지 Frog 1개로 확인해 두십시오.)

### A.7.2 집계 (호스트 python, 표준 라이브러리만)

```bash
R="--raw Data/MVSeg/jf_v2.json Data/MVSeg/jf_xw.json Data/MVSeg/jf_p12.json \
   Data/MVSeg/jf_p12b.json Data/MVSeg/jf_sam2_matched.json"

# (a) 1차: 같은 실행 안의 짝지은 매칭 효과
python3 eval/report_jf.py $R --methods SegMaskSam2Legacy SegMaskSam2Matched \
    --paired SegMaskSam2Legacy SegMaskSam2Matched > docs/raw/sam2/a_e6_matched.txt
python3 eval/report_jf.py $R --methods SegMaskSam2Legacy SegMaskSam2Matched \
    --paired SegMaskSam2Legacy SegMaskSam2Matched --frames inner > docs/raw/sam2/a_e6_matched_inner.txt

# (b) 재현 가드의 수치 확인 (F0이면 델타가 정확히 0)
python3 eval/report_jf.py $R --methods SegMaskNew3 SegMaskSam2Legacy \
    --paired SegMaskNew3 SegMaskSam2Legacy > docs/raw/sam2/a_repro.txt

# (c) 헤드라인 표 — 매칭 전/후 SAM 2 와 SAM 3 채택본
python3 eval/report_jf.py $R --methods SegMaskNew1 SegMaskNew3 SegMaskSam2Legacy \
    SegMaskSam2Matched SegMaskSam3XW0 SegMaskSam3XW1GPS4 > docs/raw/sam2/a_headline.txt

# (d) 최종 귀속: SAM 3 − 매칭된 SAM 2
python3 eval/report_jf.py $R --methods SegMaskSam2Matched SegMaskSam3XW1GPS4 \
    --paired SegMaskSam2Matched SegMaskSam3XW1GPS4 > docs/raw/sam2/a_attribution.txt
```

### A.7.3 실행 전에 기록하는 기준값 (오늘 측정, 15개·pooled·all frames·as-is)

| 지표 | `SegMaskNew1` | `SegMaskNew3` | `SegMaskSam3XW0` | `SegMaskSam3XW1GPS4` |
|---|---|---|---|---|
| J&F | 0.8434 | 0.8433 | 0.8427 | **0.8454** |
| J | 0.8125 | 0.8124 | 0.8081 | 0.8104 |
| **F** | **0.8744** | **0.8742** | 0.8772 | **0.8805** |

* **ΔF(SAM3 채택본 − SAM2 발표본) = 0.8805 − 0.8744 = +0.0061** ← REPORT E6 가 말한 "+0.006"의 정체.
* ΔJ = 0.8104 − 0.8125 = **−0.0021** (SAM 3가 J 에서 짐).
* ΔJ&F = +0.0020. 즉 **SAM 3의 우위는 전부, 그리고 그 이상이 경계 F 에서 나옵니다.**
* 계열 보정: New1 → New3 로 기준선을 바꾸면 ΔF 는 +0.0061 → **+0.0063** 이 됩니다(New3 F = 0.8742).
* 짝지은 통계(카메라 45대, `SegMaskSam3XW1GPS4 − SegMaskNew3`): 평균 ΔJ&F +0.0028,
  29승 16패, sign p 0.072, Wilcoxon p 0.045, 클러스터 CI [−0.0076, +0.0126].

### A.7.4 사전 등록 판독 — "SAM 3의 경계 F 우위는 설정 매칭 후에도 살아남는가"

**1차 지표**: `F(SegMaskSam2Matched) − F(SegMaskSam2Legacy)`, 15개 데이터셋 pooled 평균 = **Δ_post**.
**2차 지표**: 카메라 45대 짝지은 ΔF, 클러스터 부트스트랩 95% CI (`random.Random(0)`, 10,000회).

세 갈래로 미리 갈라 둡니다.

| 관측 | 해석 (사전 등록) |
|---|---|
| **Δ_post ≥ +0.005** | non-overlap 이 SAM 2의 F 를 SAM 3 수준으로 끌어올림. 남은 격차 `+0.0063 − Δ_post` 가 **모델·해상도 몫**. "SAM 3의 F 우위는 후처리 규칙의 산물"이라고 쓸 수 있음 |
| **+0.001 ≤ Δ_post < +0.005** | 부분 설명. 논문에는 "F 격차 +0.006 중 Δ_post 는 출력단 non-overlap, 나머지는 low-res 해상도(256 vs 288)와 모델 차이로 남으며 후자는 **매칭 불가**" 로 기술 |
| **Δ_post < +0.001** (CI 가 0 포함) | **F 우위는 설정 불일치로 설명되지 않음.** E6 는 기각되고, SAM 3의 경계 F 우위는 실질적 결과로 인정 |
| Δ_post < 0 | non-overlap 이 SAM 2의 F 를 **떨어뜨림** → E6 격차는 오히려 과소평가였음. 그대로 보고 |

**예상(실행 전 기록)**: non-overlap 은 겹치는 픽셀만 건드리므로 J 변화는 −0.001 안쪽, F 변화는 그보다
크되 +0.005 를 넘기 어렵습니다. 객체가 3~15개로 적고 대부분 겹치지 않는 장면이 많기 때문입니다.
따라서 **두 번째 칸(부분 설명)이 가장 그럴듯하다**고 미리 적어 둡니다.

**판정에 쓰지 않는 것**: 12개 부분집합(전체와 부호가 다름, experiments.md ★), 객체 단위 일화, 실행 시간.

### A.7.5 매칭할 수 없는 것 (논문에 반드시 명시)

1. **low-res 마스크 격자 256 vs 288** — `image_size 1024 / stride 4` vs `1008 / 14 × 4`. 백본 stride 를
   바꾸면 다른 체크포인트가 됩니다. **A 이후에도 남는 유일한 후처리 외 교란항**이고, `fill_hole` 이
   적용되는 격자이기도 합니다(효과는 §A.2 로 무효화되었지만 마스크 업샘플 경로는 남습니다).
2. **메모리 선택(temporal disambiguation)** — SAM 3는 `use_memory_selection=True` + `mf_threshold=0.01`
   로 자기 과거 기억을 걸러 씁니다. SAM 2에는 그 기제 자체가 없습니다.
3. **백본·디코더 가중치 전체** — 당연하지만 명시. A는 "후처리 몫"만 분리합니다.
4. **`SegMaskNew1` 자체** — 그 실행을 만든 코드가 git 에 없으므로(§A.4.1), A는 New3 계열 위에서만
   성립합니다. 논문 표에 `SegMaskNew1` 을 계속 인용한다면 New1↔New3 격차(J&F 0.0001, F 0.0002)를
   각주로 달아야 합니다.

---

# PART B — XW(교차시점) 작업의 SAM 2 이식

## B.1 왜 되는가 — SAM 2의 수집 루프는 SAM 3와 문장 단위로 같습니다

`SCSam2/demo/SCSam2VideoPredictorNew.py:501-508` (working tree 행번호; 과제 브리프의 487-505 는
2025-11-27 이전 번호):

```python
501  for s_pos in range(-4, 0):
502      prev_spatial_idx = spatial_idx + s_pos
503      if spatial_idx != prev_spatial_idx:                     # 항상 참 (s_pos != 0)
504          out = output_dicts[prev_spatial_idx]["non_cond_frame_outputs"].get(frame_idx, None)
505          if out is None:
506              selected_cond_outputs, unselected_cond_outputs1 = select_closest_cond_frames(...)   # C2
507              out = unselected_cond_outputs1.get(frame_idx, None)                                 # 항상 None
508          s_pos_and_prevs.append((s_pos, out))
```

* :486-492, :493-499 의 두 블록은 **삼중따옴표 문자열 안**이라 죽어 있습니다(각각 mode D `lower_tm1`,
  mode A without fallback). 살아 있는 것은 :501-508 하나입니다.
* 이것은 `xview_gather.gather_cross_view_memories(window=4, hygiene=False, mode="lower_t")` 와
  **문장 대 문장으로 같습니다** — 음수 인덱스 wrap(C1), `len(output_dicts) < 4` 일 때 IndexError,
  죽은 cond fallback 과 그 유일한 효과인 `selected_cond_outputs` 재바인딩(C2)까지.
* C2 재바인딩의 소비처도 같습니다: :443 에서 자기 뷰로 묶인 이름이 :506 에서 이웃 뷰로 덮이고,
  :548-567 의 `ptr_cond_outputs`/`pos_and_ptrs` 가 그걸 읽습니다. `unselected_cond_outputs`(:443)는
  **덮이지 않습니다**(:506 은 `unselected_cond_outputs1` 에 씁니다) — :574 는 여전히 자기 뷰를 읽습니다.
* `max_cond_frames_in_attn = -1` 이므로 `unselected_cond_outputs1` 은 항상 `{}`,
  즉 fallback 은 메모리를 공급하지 않고 **재바인딩만** 합니다.

**따라서 REPORT C1/C2 는 SAM 3 이식본의 결함이 아니라 두 스택이 공유하는 설계의 성질입니다.**

구조적 전제도 전부 같습니다.

| 항목 | SAM 2 | SAM 3 |
|---|---|---|
| `num_maskmem` | 7 (`sam2.1_hiera_l.yaml:88`) | 7 |
| 자기 메모리 tpos 행 | `num_maskmem - t_pos - 1` (:523) | 같음 (:1391) |
| 이웃 메모리 tpos 행 | `abs(s_pos) - 1` (:539) | 같음 (:1412) |
| 경계 | `0 ≤ W ≤ 6`, `W + S ≤ 6` | 같음 |
| `output_dicts` 모양 | 객체별·카메라별 dict 리스트, 인덱스 = 카메라 (:942-976) | 같음 (세션 순) |
| 포인터 채널 | `use_obj_ptrs_in_encoder/add_tpos_enc/proj_tpos_enc/use_signed_tpos` 전부 true (yaml:104-108), 포인터 1개 → `C//mem_dim = 256/64 = 4` 토큰 | 같은 4-way split |
| lockstep | 데모가 카메라 순서로 카메라마다 `next()` 1회 (`sam2_demoVideoNew_maskSingleInputMVSeg.py:67-71`) | `runMVSeg.py:615-617` 과 동형 |

## B.2 새 폴더 — 내용과 만드는 법

### B.2.1 만드는 법

```bash
cd /home/sjpark/Documents/SCSegmentation
mkdir -p SCSam2/demoSCSam2MVOpt
for f in build_sam.py SCSam2ImagePredictor.py SCSam2VideoNew.py SCSam2VideoPredictorNew.py \
         misc.py pose.py colmap_read_model.py MVSeg.json \
         sam2_demoVideoNew_maskSingleInputMVSeg.py sam2_demoVideo_maskSingleInputMVSeg.py; do
  cp -p SCSam2/demo/$f SCSam2/demoSCSam2MVOpt/$f
done
cp -p SCSam3/demoSCSam3MVOpt/xview_gather.py SCSam2/demoSCSam2MVOpt/xview_gather.py   # 바이트 동일 사본
```
그리고 **복사 직후** 원본 sha256 목록을 `SCSam2/demoSCSam2MVOpt/ORIGIN.md` 에 기록합니다
(`sha256sum` 출력 + 커밋 `3b292f7` + 날짜 + R1 규칙 문구). 이 커밋이 "바이트 동일 사본" 커밋이고,
이후 커밋에서만 편집합니다 — `demoSCSam3MVOpt` 가 만들어진 방식과 같습니다.

### B.2.2 왜 이 파일들인가

| 파일 | 필요한 이유 |
|---|---|
| `SCSam2VideoPredictorNew.py` | **개발 대상**. 교차시점 수집·토큰 루프·포인터 블록이 여기 |
| `SCSam2VideoNew.py` | 세션 관리·프롬프트·트래킹 제너레이터. `from build_sam import …`, `from SCSam2VideoPredictorNew import …`, `from misc import *` |
| `build_sam.py` | 두 빌더(`_spatial`, `_new`). 하이드라 `_target_` 문자열이 모듈 이름으로 이 폴더를 가리키게 됩니다 |
| `SCSam2ImagePredictor.py` | `build_sam2_video_predictor_spatial` 의 `_target_`(`SCSam2ImagePredictor.SAM2VideoPredictorSpatial`) |
| `misc.py` | `SCSam2VideoNew.py` 가 `from misc import *` |
| `pose.py`, `colmap_read_model.py` | 발표 데모 스크립트가 `from pose import *` 를 하므로 스냅샷 완결성을 위해. 러너는 쓰지 않음 |
| `MVSeg.json` | 데이터셋 설정. 동결 사본(§B.5 T-B1 이 `SCSam2/demo/MVSeg.json`·`SCSam3/demo/MVSeg.json` 과 동일함을 검사 — 오늘 확인: 셋 다 동일) |
| `sam2_demoVideoNew_maskSingleInputMVSeg.py` | **발표본을 만든 드라이버 원본**. 실행하지 않고, 러너가 이것과 의미상 동등함을 문서·테스트로 고정하기 위한 참조 |
| `sam2_demoVideo_maskSingleInputMVSeg.py` | 평면(교차시점 없음) 경로 참조. `SegMask1` 계열 |
| `xview_gather.py` | SAM 3와 **바이트 동일**. 이웃 기하(창·모드·위생·wrap 재현)를 두 스택이 문자 그대로 공유한다는 주장을 테스트 가능하게 만듦 |

**추가로 새로 만드는 파일** (복사본이 아님):

| 파일 | 내용 |
|---|---|
| `SCSam2/demoSCSam2MVOpt/xview_gate_sam2.py` | SAM 2용 게이트 술어와 게이팅 함수 (§B.4.3). `xview_gather` 에서 `new_xview_stats`·`record_xview` 를 그대로 임포트 |
| `SCSam2/demoSCSam2MVOpt/apply_nonoverlap.py` | §A.3.3 의 오프라인 후처리 |
| `SCSam2/demoSCSam2MVOpt/ORIGIN.md` | 복사 출처·해시·규칙 |
| `SCSam2/runMVSeg.py` | 러너 (§B.3) |
| `SCSam2/runSam2Three.sh` | 컨테이너 구동 스크립트 (§B.3.5) |
| `tests_sam2/` | scsam2 이미지에서 도는 CPU 테스트 (§B.5) |
| `tests/test_sam2_parity.py` | scsam3 pytest 스위트에 붙는 torch-free 검사 (§B.5) |

### B.2.3 복사본에 가하는 유일한 **비기능** 편집

`build_sam.py:145-188` 의 **죽은 첫 번째 `build_sam2_video_predictor_new`** 를 삭제합니다.
그것은 존재하지 않는 클래스 `SCSam2VideoPredictorNew.SAM2VideoPredictorNew` 를 겨냥하고 있고
:235-278 의 동명 정의에 가려집니다(파이썬은 마지막 정의만 남깁니다). 로그의 배너로 구분됩니다:
죽은 쪽 `build_sam2_video_predictor_new`, 산 쪽 `build_sam2_video_predictorNew`(공백 없음).
**함정**: `fill_hole_area` 를 grep 해서 첫 히트(:176)를 고치면 아무 일도 안 일어나고 조용히
매칭되지 않은 실행이 나옵니다. 살아 있는 앵커는 **:266** 입니다.
삭제가 출력 중립임은 §B.5 T-B2(정의 1개, `_target_` 이 `SAM2VideoPredictorCustom`)로 고정하고,
G1 가드가 실증합니다.

## B.3 러너 `SCSam2/runMVSeg.py`

`SCSam3/runMVSeg.py` 를 형태·플래그·거부 규칙·매니페스트 기록까지 **그대로 옮깁니다**. 차이만 적습니다.

### B.3.1 구조

```
ALGOS   = {"MVOpt": "demoSCSam2MVOpt"}          # SAM 2는 개발 대상 패키지가 하나뿐
DEFAULT_OUT = {"MVOpt": "SegMaskSam2Legacy"}
XVIEW_ALGOS = ("MVOpt",)
XVIEW_MAX_WINDOW = 6
XVIEW_OUT_PREFIX = "SegMaskSam2XW"
XVIEW_MODES = ("A","B","C","D","E")             # 초판은 A만 구현, 나머지는 sys.exit("not implemented")
XVIEW_TPOS_SHIFTS = tuple(range(1, 6))          # 1..5
```

`build_runner(algo)` 는 SAM 3판과 같은 `sys.path` 수술을 합니다 — `SCSam2/` 자신을 sys.path 에서
빼고(`SCSam2/sam2` 가 설치 패키지를 가리므로), 패키지 폴더를 맨 앞에 넣고, `os.chdir(algo_dir)`
(체크포인트 상대경로 `../models/...` 와 하이드라 config 이름이 이걸 전제합니다).

### B.3.2 플래그

| 플래그 | 기본 | 의미 |
|---|---|---|
| `dataset` | — | `MVSeg.json` 키 |
| `--out` | 계산됨 | 출력 폴더 |
| `--config` | 패키지 안 `MVSeg.json` | |
| `--track-cams {all,closure}` | legacy=`all`, XW=`closure` | §B.3.4 |
| `--xview-window W` (0..6) | 없음 | 켜면 **위생 자동 on**, `closure` 기본, 출력 `SegMaskSam2XW{W}` |
| `--xview-hygiene` | 없음 | `--xview-window 4` 와 같음 |
| `--xview-mode M` | A | 초판 A만 |
| `--xview-gate` | off | §B.4.3 |
| `--xview-gate-mode {memscore,occl}` | `memscore` | §B.4.3 |
| `--xview-ptr` | off | §B.4.2 |
| `--xview-tpos-shift S` (1..5) | 0 | §B.4.1 |
| `--fill-hole-area N` | None(=8) | §A.3.5 |
| `--nonoverlap {off,objectwise}` | off | §A.3.4 |
| `--dump-scores` | off | §A.3.3 |
| `--overwrite`, `--dry-run`, `--device` | | SAM 3판과 동일 |

**거부 규칙**(전부 `sys.exit`, SAM 3판 `resolve_run` 과 동형):
XW 플래그 없이 손잡이만 · `W=0` 에 손잡이 · `W + S > 6` · `--xview-tpos-shift 0` (argparse) ·
위생 없는 `--track-cams closure` · XW 실행을 `SegMaskSam2Legacy`/`SegMaskNew*` 이름에 ·
legacy 실행을 `SegMaskSam2XW*` 이름에 · `--nonoverlap objectwise` 를 `SegMaskSam2Legacy` 이름에.

### B.3.3 실행 루프 — 발표본과 문장 단위로 같아야 합니다

`sam2_demoVideoNew_maskSingleInputMVSeg.py:39-80` 을 그대로 옮깁니다:

1. `SCSam2VideoNew(device, **xview_kwargs)`;
2. `sc.LoadVideo_Folder_MVSeg(folder+"/Video/", perms, start_frame, prefix, prefix1)` — **모든** 카메라에
   `init_state`(변경 금지, §B.3.4);
3. 기준 카메라 = `cam_list` 중 GT `np.max` 가 최대인 카메라, `obj_ids = np.max(img)`, `id = perms.index(max_mask)`;
4. `for i in range(obj_ids): sc.AddMaskSingle(id, (img == i+1)*255, i+1)`;
5. `if id != 0: sc.InitializeSegmentation(refCamID=id, reverse=True)` 그리고 `sc.InitializeSegmentation(refCamID=id)`
   — **두 줄 다, 이 순서로**;
6. `sc.RunNaiveTracking(start_frame)`;
7. `for idx in range(50): if idx >= num_frame: break; for spatial_idx in range(len(perms)): next(...)`.
   `if perms[spatial_idx] in cam_list` 는 **PNG 쓰기만** 거릅니다 — `next()` 는 모든 카메라에 대해
   호출되어야 합니다(그게 lockstep 계약입니다).
8. PNG: `(out_mask_logits[i] > 0.0).cpu().numpy()[0,...] * 255` → `<out>/<cam>/<frame>/<objid>.png`.

`num_frame` 은 15개 전부 21 이라 `range(50)` 은 걸리지 않습니다.

### B.3.4 `--track-cams`

* **legacy(위생 off)에는 `all` 만 허용합니다.** wrap 때문에 뷰 0..3 이 뷰 N-4..N-1 을 읽으므로
  카메라를 줄이면 **다른 모델**이 됩니다(operations.md 함정 1과 동일한 논리).
* **위생 on 이면 `closure`**: 뷰 v 는 뷰 v-W..v-1 만 읽으므로 `0..max(채점 시점 인덱스)` 만 추적하면
  채점 뷰의 출력이 `all` 과 같습니다.
* **SAM 2의 closure 는 SAM 3보다 안전합니다** — 인덱스가 전혀 이동하지 않기 때문입니다. 구현은
  "추적하지 않을 카메라의 **제너레이터를 만들지 않고 `next()` 도 하지 않는다**" 뿐입니다.
  `init_state`(2단계)와 `RunNaiveTracking` 의 seeding 은 **모든 카메라에 그대로 둡니다.**

  **이유(중요)**: 교차시점 pass 의 유사 비디오는 `original_states[i]["images"][start_frame]` 로 만들어지고
  (`SCSam2ImagePredictor.py:29-49`), 그 `inference_state["num_frames"] = len(images)` 가
  `max_obj_ptrs_in_encoder = min(num_frames, 16)` 에 들어갑니다. 카메라 수를 줄이면 **spatial pass 자체가
  달라집니다**(예: Welder 46→4 이면 포인터 16개 → 4개). 그러므로 카메라 로딩·spatial pass·seeding 은
  건드리지 않고, **temporal 추적만** 줄입니다. 그래도 비용의 대부분(21프레임 × 객체 × 카메라)이 사라집니다.
* `closure == all` 을 Welder·Dog 에서 `diff -rq -x MANIFEST.json` 0 으로 실측 확인합니다(감시값 Z2).

데이터셋별 closure 세션 수 (= max(채점 인덱스)+1):

```
AlexaMeadeExhibit 45→4   FacePaint 46→9   Barn 15→11   Blocks 10→10   Breakfast 15→10
Carpark 9→9   CoffeeMartini 18→15   Dog 41→4   Fencing 10→10   FlameSteak 21→17
Frog 13→10   MATF 10→10   Painter 16→16   PoznanStreet 9→9   Welder 46→4
```

### B.3.5 매니페스트와 구동 스크립트

`eval/manifest.py:write_run_manifest` 를 그대로 호출합니다(수정하지 않습니다).
`package_dir=SCSam2/demoSCSam2MVOpt`, `algo="MVOpt"`, `torch=torch`,
`extra` 에 `lineage`, `xview_window/_hygiene/_mode/_gate/_gate_mode/_ptr/_tpos_shift`,
`xview_gate_stats`, `fill_hole_area`, `nonoverlap`, `track_cams_requested`, `track_idx`,
`n_sessions`, `scored_view_idx`, `reference`, `stack="sam2"`, `sam2_C_available=False`.

이미지 ID는 `manifest.py:426` 이 `SCSAM3_IMAGE_ID` 를 읽으므로 그 이름 그대로 **scsam2 이미지 ID**를
넘기고, `extra["docker_image"] = "scsam2:latest"` 를 함께 적어 오해를 막습니다.

`SCSam2/runSam2Three.sh` (= `SCSam3/runMVOptThree.sh` 의 SAM 2판):

```bash
#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${ROOT}/SCSam2/logs/${LOGTAG:-sam2}-$(date +%Y%m%d-%H%M%S)"; mkdir -p "${LOGDIR}"
IMAGE_ID="$(docker images --no-trunc -q scsam2:latest | head -n1)"
for ds in "$@"; do
  docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \
    -e SCSAM3_IMAGE_ID="${IMAGE_ID}" -v /:/host -w "/host${ROOT}/SCSam2" scsam2 \
    python runMVSeg.py "${ds}" ${OUT:+--out "${OUT}"} \
      ${XVIEW:+--xview-window "${XVIEW}"} ${XGATE:+--xview-gate} ${XPTR:+--xview-ptr} \
      ${XSHIFT:+--xview-tpos-shift "${XSHIFT}"} ${TRACK:+--track-cams "${TRACK}"} \
      ${FILLHOLE:+--fill-hole-area "${FILLHOLE}"} ${NONOVL:+--nonoverlap "${NONOVL}"} \
      ${DUMP:+--dump-scores} --overwrite > "${LOGDIR}/${ds}.log" 2>&1
done
docker run --rm -v /:/host scsam2 chown -R "$(id -u):$(id -g)" "/host${ROOT}/Data/MVSeg" >/dev/null 2>&1
```
**함정(SAM 3판과 같음)**: `XGATE`·`XPTR`·`XSHIFT`·`DUMP` 는 **비어 있지 않기만 하면 켜집니다** —
`XGATE=0` 도 플래그를 켭니다. `XSHIFT=0` 은 argparse 가 거부해 실행이 죽습니다.
`FILLHOLE=0` 은 빈 문자열이 아니므로 정상 동작합니다(`--fill-hole-area 0`).

## B.4 이식 — 정확한 편집 지점

전부 `SCSam2/demoSCSam2MVOpt/SCSam2VideoPredictorNew.py` 안입니다. 행번호는 **복사 직후**(=원본) 기준.

### B.4.0 임포트와 생성자

`:21-23` 아래에 추가:
```python
from xview_gather import (gather_cross_view_memories, gather_mode_for, new_xview_stats,
                          record_xview, resolve_cross_view, resolve_cross_view_knobs,
                          resolve_cross_view_mode, UNCHANGED)
from xview_gate_sam2 import gate_cross_view_sam2
```

`:308-310` 의 `__init__` 을 다음으로 바꿉니다(하이드라가 kwargs 로 넣어 줍니다):
```python
def __init__(self, *args, cross_view_window=None, cross_view_hygiene=None,
             cross_view_mode=None, cross_view_gate=None, cross_view_gate_mode=None,
             cross_view_ptr=None, cross_view_tpos_shift=None, **kwargs):
    print("SAM2VideoPredictorCustom")
    super().__init__(*args, **kwargs)          # num_maskmem 을 세팅
    self.cross_view_window, self.cross_view_hygiene = resolve_cross_view(
        cross_view_window, cross_view_hygiene, self.num_maskmem)
    self.cross_view_mode = resolve_cross_view_mode(cross_view_mode, self.cross_view_hygiene)
    self.cross_view_gate, self.cross_view_ptr, self.cross_view_tpos_shift = \
        resolve_cross_view_knobs(cross_view_gate, cross_view_ptr, cross_view_tpos_shift,
                                 self.cross_view_window, self.cross_view_hygiene, self.num_maskmem)
    self.cross_view_gate_mode = cross_view_gate_mode or "memscore"
    self.mf_threshold = 0.01                   # SAM 3 와 같은 임계값 (sam3_tracker_base.py 기본)
    self.xview_stats = new_xview_stats()
```
**전달 경로**: `build_sam.py` 사본의 살아 있는 빌더(:235-278)에 같은 이름의 파라미터를 더하고,
`hydra_overrides.extend(hydra_overrides_extra)` **뒤에** `++model.cross_view_*=…` 를 붙입니다
(값이 `None` 이 아닌 것만). 그래야 `apply_postprocessing` 블록에 지지 않습니다.
`SCSam2VideoNew.__init__` 도 같은 인자를 받아 `build_sam2_video_predictor_new(...)` 로 넘깁니다.

### B.4.1 수집 루프 교체 (창 W · 위생 · 모드 · 행 이동 S)

`:484-509` (즉 `s_pos_and_prevs = []` 부터 죽은 두 블록과 살아 있는 루프까지)를 통째로 다음으로:

```python
            # spatial추가 -- cross-view gather, see xview_gather.py.  legacy (4, False)
            # 이면 :501-508 을 문장 단위로 재현합니다 (wrap, IndexError, C2 재바인딩 포함).
            s_pos_and_prevs, rebound = gather_cross_view_memories(
                output_dicts, spatial_idx, frame_idx,
                window=self.cross_view_window, hygiene=self.cross_view_hygiene,
                max_cond_frames_in_attn=self.max_cond_frames_in_attn,
                select_fn=select_closest_cond_frames,
                mode=gather_mode_for(self.cross_view_mode, None),
                track_in_reverse=track_in_reverse,
            )
            if rebound is not UNCHANGED:
                selected_cond_outputs = rebound          # C2: 원래 :506 이 하던 그 일
            if self.cross_view_gate or self.cross_view_ptr or self.cross_view_tpos_shift:
                s_pos_and_prevs, n_seen, n_fail = gate_cross_view_sam2(
                    s_pos_and_prevs, self.mf_threshold, apply=self.cross_view_gate,
                    mode=self.cross_view_gate_mode)
                record_xview(self.xview_stats, spatial_idx, frame_idx, n_seen, n_fail)
```
죽은 두 문자열 블록은 삭제합니다(모드 D·A-without-fallback 로 `xview_gather` 안에 이미 있습니다).

`:539` 의 이웃 tpos 행에 S 를 더합니다:
```python
                    maskmem_enc + self.maskmem_tpos_enc[abs(s_pos) - 1 + self.cross_view_tpos_shift]
```
**S 의 경계**: `resolve_cross_view_knobs` 가 `W + S ≤ num_maskmem - 1 = 6` 을 강제합니다
(행 6 은 cond 프레임 행). SAM 3와 같은 식·같은 값.

### B.4.2 P — 이웃 객체 포인터

`:577` (`pos_and_ptrs.append((t_diff, out["obj_ptr"]))`) 과 `:578` 사이에 삽입:

```python
                # P12 P: 이웃의 obj_ptr 을 자기 포인터 **뒤에** 추가.  자기 목록의 상한
                # (max_obj_ptrs_in_encoder) 과 tpos 정규화 상수는 그대로 두고, 이웃은
                # 이웃 메모리가 num_maskmem 을 넘어 추가되듯 추가 토큰으로 들어갑니다.
                # 시간 위치 = 이웃 메모리 토큰이 갖는 별칭: v-k 는 "k 프레임 전" -> t_diff = k + S.
                # 두 방향 추적 모두에서 양수 (자기 non-cond t_diff 와 같은 부호 규약).
                if self.cross_view_ptr:
                    for s_pos, prev in s_pos_and_prevs:
                        if prev is None:
                            continue
                        pos_and_ptrs.append(
                            (abs(s_pos) + self.cross_view_tpos_shift,
                             prev["obj_ptr"].to(device)))
```
SAM 3 (`SCSam3TrackerPredictorNewMem.py:1486-1492`)와 의미가 같습니다. 아래는 손대지 않습니다:
`torch.stack`, `get_1d_sine_pe(obj_pos / t_diff_max)`(`t_diff_max = max_obj_ptrs_in_encoder - 1 = 15`),
`obj_ptr_tpos_proj`, `mem_dim` 4-way split, `num_obj_ptr_tokens = obj_ptrs.shape[0]` (:605).
`add_tpos_enc_to_obj_ptrs=true` 이므로 S 는 P 채널에서도 실제로 위치를 움직입니다.
이웃 메모리 토큰은 포인터 토큰 **앞에** 붙으므로 `memory_attention` 의 `num_obj_ptr_tokens` 불변식은
구조상 유지됩니다.

**SAM 2 고유의 상한 주의**: W=1 + P 면 포인터가 최대 1(cond) + 15(자기 non-cond) + 1(이웃) = 17개,
토큰 68개. SAM 3와 같은 성질이고(공칭 16을 하나 넘김), 인코더에 상한이 없어 무해합니다.

### B.4.3 G — 게이트: **SAM 2에는 정확한 대응물이 없습니다** (명시적 이탈)

**사실**: SAM 2에는 메모리 선택 기제가 **아예 없습니다**. `use_memory_selection`, `frame_filter`,
`eff_iou_score`, `mf_threshold`, `cal_mem_score` 다섯 이름 모두 `SCSam2/sam2/sam2/**` 와
`SCSam2/demo/*.py` 에서 **0건**입니다. SAM 2의 시간축 루프는 `range(1, num_maskmem)` 를 고정 stride 로
걸을 뿐입니다.

**그래서 정당화가 뒤집힙니다.** SAM 3에서 G 는 *대칭 회복*입니다 — 자기 기억은 이미
`eff_iou_score > 0.01` 로 걸러지는데 이웃만 무조건 들어오고 있었으므로 같은 검사를 겁니다.
**SAM 2에서 G 는 대칭을 만드는 게 아니라 없던 비대칭을 만듭니다** — 자기 기억은 안 걸리는데
이웃만 걸리게 됩니다. 논문·보고서에 **이 문장을 그대로** 실어야 합니다. G 는 SAM 2 쪽에서
"SAM 3의 원칙 이식"이 아니라 **순수 절제(ablation)** 입니다.

두 가지 대체물을 구현하고 둘 다 보고합니다.

| 모드 | 술어 | 필요한 추가 상태 |
|---|---|---|
| **`memscore`** (기본) | SAM 3 `cal_mem_score` 의 복제: `object_score_norm = where(l > 0, sigmoid(l)*2 - 1, 0)`; `score = (object_score_norm * iou).mean()`; 통과 = `score > 0.01` | `ious` 저장 필요 (아래) |
| `occl` | `object_score_logits > 0` — SAM 2 자신의 `is_obj_appearing` (`sam2_base.py:360, :719`) | 없음 (이미 저장됨) |

**"키 없음" 규약은 SAM 3와 같게** 합니다: 필요한 키가 없는 항목은 **통과시킵니다**
(`xview_gather.admits` 의 주석 (ii)). must-include 는 두지 않습니다(주석 (i)).
모드 A에서 이웃은 항상 `non_cond_frame_outputs` 에서 오므로 키가 있습니다. t−1 계열 모드에서만
seed cond 항목이 잡히고 그건 `ious` 가 없어 통과합니다.

`ious` 저장 (게이트가 켜졌을 때만, 기본 경로의 저장 상태를 바꾸지 않기 위해):

* `:735-743` 의 언팩에서 세 번째 슬롯이 `ious` 입니다
  (`_forward_sam_heads` 와 `_use_mask_as_output` 양쪽 반환 튜플의 2번 인덱스, `sam2_base.py:404-412, :456-464`).
  `_, _, ious, low_res_masks, …` 로 받고 `if self.cross_view_gate: current_out["ious"] = ious`.
* `:828-834` 의 `compact_current_out` 에 `if self.cross_view_gate: compact_current_out["ious"] = ious`.
* **선택된 마스크의 IoU** 는 `ious.max(dim=-1, keepdim=True).values` 입니다 —
  `sam2_base.py:383` 이 `best_iou_inds = argmax(ious, dim=-1)` 로 고르므로 최대값과 같고,
  `multimask_output=False` 면 항목이 하나뿐이라 역시 같습니다.

`xview_gate_sam2.py`:
```python
from xview_gather import new_xview_stats, record_xview   # 공유 (재export만)

def admits_sam2(out, threshold, mode="memscore"):
    """SAM 3 frame_filter 점수 검사의 SAM 2 대체물.  키가 없으면 True(통과)."""
    l = out.get("object_score_logits", None)
    if l is None:
        return True
    if mode == "occl":
        return bool((l > 0).all())
    iou = out.get("ious", None)
    if iou is None:
        return True
    import torch
    norm = torch.where(l > 0, torch.sigmoid(l) * 2 - 1, torch.zeros_like(l))
    return bool((norm * iou.max(dim=-1, keepdim=True).values).mean() > threshold)

def gate_cross_view_sam2(s_pos_and_prevs, threshold, apply, mode="memscore"):
    n_seen = n_fail = 0; kept = []
    for s_pos, prev in s_pos_and_prevs:
        if prev is None:
            continue
        n_seen += 1
        if admits_sam2(prev, threshold, mode):
            kept.append((s_pos, prev))
        else:
            n_fail += 1
    return (kept if apply else s_pos_and_prevs), n_seen, n_fail
```
`xview_gather.gate_cross_view` 와 카운터 규약(`n_seen` = None 아닌 항목, `apply=False` 면 입력을
그대로 반환)이 동일합니다.

**예상**: SAM 3 쪽에서도 게이트는 자주 no-op 이었습니다(Fencing `XW1G` 의
`provenance.xview_gate_stats` = `{calls 1600, nb_seen 1440, nb_fail 0, nb_dropped 0}`; 탈락 0인
데이터셋 Barn·Carpark·Fencing·Frog). SAM 2 쪽에서는 더 드물게 걸릴 것으로 봅니다.
**탈락이 0이면 `XW1G` 는 `XW1` 과 바이트 동일해야 하고**, 그것 자체가 보고할 결과입니다.

### B.4.4 손잡이 전부 끄면 오늘의 코드와 문장 단위로 같음

`(window, hygiene, mode, gate, ptr, shift) = (4, False, "A", False, False, 0)` 이면:
`gather_cross_view_memories` 의 legacy 분기가 :501-508 을 재현하고, `rebound` 가 :506 의 재바인딩을
그대로 전달하며, `+ self.cross_view_tpos_shift` 는 `+0`, P 블록은 실행되지 않고,
게이트 블록도 조건식이 거짓이라 실행되지 않습니다(카운터도 안 돕니다).

## B.5 CPU 테스트

### B.5.1 어디서 도는가

* **`tests/` (기존 scsam3 pytest 스위트)** — torch-free 검사 하나만 추가:
  `tests/test_sam2_parity.py`. `xview_gather.py` 두 사본의 **sha256 동일**, `MVSeg.json` 세 사본의 동일,
  그리고 SAM 2 러너의 `resolve_run` 해석표(파일 경로로 로드, `import torch` 없음).
  주의: `tests/conftest.py` 가 `SCSam3/demoSCSam3MVOpt` 를 `sys.path[0]` 에 넣으므로
  **SAM 2 사본은 반드시 파일 경로로, 다른 모듈 이름으로 로드**합니다
  (`importlib.util.spec_from_file_location("xview_gather_sam2", …)`).
* **`tests_sam2/` (신규, scsam2 이미지 전용)** — torch·sam2 가 필요한 검사. 자체 `conftest.py` 를 두어
  SCSam3 경로를 **넣지 않습니다**.

pytest 는 이미지에 없으므로 한 번 벤더링합니다(검증 완료):
```bash
mkdir -p $ROOT/SCSam2/.pytest-libs      # .gitignore 에 추가
docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$ROOT scsam2 \
  python -m pip install --quiet --no-input --target /host$ROOT/SCSam2/.pytest-libs pytest
docker run --rm --user $(id -u):$(id -g) -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/host$ROOT/SCSam2/.pytest-libs -v /:/host -w /host$ROOT scsam2 \
  python -m pytest tests_sam2 -q -p no:cacheprovider
```
(`$ROOT` 는 `/home/sjpark/Documents/SCSegmentation`. 컨테이너 안 작업 디렉터리는 `/host$ROOT` 이며
`SCSam2/` 안이 아니므로 R4 를 지킵니다.)

### B.5.2 테스트 목록

| id | 무엇을 | 어디서 |
|---|---|---|
| T-B1 | `SCSam2/demoSCSam2MVOpt/MVSeg.json` == `SCSam2/demo/MVSeg.json` == `SCSam3/demo/MVSeg.json` (오늘 확인: 동일) | tests/ |
| T-B2 | 패키지 사본의 `build_sam.py` 에 `build_sam2_video_predictor_new` 정의가 **정확히 1개**, `_target_` 이 `SCSam2VideoPredictorNew.SAM2VideoPredictorCustom` (AST) | tests/ |
| T-B3 | `xview_gather.py` 두 사본 sha256 동일 | tests/ |
| T-B4 | SAM 2 러너 `resolve_run` 해석표: 15개 데이터셋 × (legacy / XW0 / XW1 / XW1GPS4 / closure / all) 의 out 이름·track_idx·거부 조합 | tests/ |
| T-B5 | **수집 등가**: `gather_cross_view_memories(4, hygiene=False, mode="lower_t")` 가 원본 :501-508 루프의 재구현과 무작위 20,000 구성에서 동일 (`tests/legacy_ref.py` 의 SAM 2판) | tests/ |
| T-A1 | `from sam2 import _C` 가 `ImportError` (환경 사실 고정) | tests_sam2/ |
| T-A2 | `fill_holes_in_mask_scores(x, 8)` 이 `torch.equal` 로 입력과 동일 (오늘 통과 확인) | tests_sam2/ |
| T-A3 | object-wise non-overlap 의 **단조 불변**: 점수를 로짓으로 주든 `sigmoid` 로 주든 결과 마스크가 동일; `N==1` 이면 no-op; 동점은 첫 인덱스 승 | tests_sam2/ |
| T-A4 | `apply_nonoverlap.py` 골든: 손으로 만든 3객체 겹침 케이스가 SAM 3의 `_apply_object_wise_non_overlapping_constraints` 정의와 일치 | tests_sam2/ |
| T-B6 | **실제 메서드 골든**: bare-instance 하니스로 `SAM2VideoPredictorCustom._prepare_memory_conditioned_features_multiple` 를 CPU 에서 호출(생성자 우회 — `__new__` + `torch.nn.Module.__init__` 로 가능함을 오늘 확인). 스텁 `memory_attention` 이 받은 `memory`·`memory_pos_embed`·`num_obj_ptr_tokens` 를 읽어, (a) legacy 설정에서 토큰 출처·tpos 행이 원본과 동일, (b) W=0 이면 이웃 토큰 0개, (c) S 가 행을 정확히 S 만큼 민다, (d) P 가 포인터 토큰을 4개 늘린다, (e) `num_obj_ptr_tokens` 불변식 | tests_sam2/ |
| T-B7 | 게이트: `admits_sam2` 의 키 없음 통과, `memscore` 가 SAM 3 `cal_mem_score` 공식과 수치 일치, `apply=False` 여도 카운터는 돈다 | tests_sam2/ |
| T-B8 | lockstep 계약: 채점 안 하는 카메라도 `next()` 가 호출되는지 (러너의 쓰기 루프를 스텁 제너레이터로 구동) | tests_sam2/ |

## B.6 바이트 동일성 가드 (B 쪽)

| id | 내용 | 통과 조건 |
|---|---|---|
| **B-G1** | 새 폴더, 플래그 없음, Blocks·Fencing | `diff -rq -x MANIFEST.json` vs `SegMaskNew3` **0줄** |
| B-G2 | 리드백: 실행 배너가 `cross-view window 4, hygiene False, mode A, gate False, ptr False, tpos-shift 0` 을 찍고 명령줄과 다르면 `sys.exit` (SAM 3 러너 :566-579 와 동형) | — |
| B-G3 | W=0 이 평면 예측기와 수학적으로 같음: `SegMaskSam2XW0` 를 `SegMask1`(평면 계열)과 채점 비교 | 카메라 단위 \|ΔJ&F\| < 1e-4. **바이트 동일은 요구하지 않습니다** — `SegMask1` 은 다른 드라이버(카메라별 순차 소비, lockstep 아님)로 만들어졌으므로 GPU 커널 순서까지 같다고 보장할 수 없습니다 |
| B-G4 | Z1(감시값): 인덱스 0 채점 카메라를 가진 9개 데이터셋에서, 모든 손잡이 변형의 그 카메라 폴더가 `XW1` 과 바이트 동일 (이웃이 없어 손잡이가 닿지 않음) | 9 × 변형수 쌍 전부 0줄 |
| B-G5 | Z2: closure == all, Welder·Dog, `XW1` 과 `XW1GPS4` | diff 0 |
| B-G6 | Z3: 결정성, Blocks `XW1GPS4` 재실행 | diff 0 |
| B-G7 | Z4: 게이트 탈락 0 인 데이터셋에서 `XW1G` == `XW1` | diff 0 |
| **가드 규칙** | `SCSam2VideoPredictorNew.py`, `xview_gather.py`, `xview_gate_sam2.py`, `build_sam.py`, `SCSam2VideoNew.py`, `runMVSeg.py` 중 하나라도 바뀌면 **B-G1 을 다시 돌린 뒤에만** XW 폴더를 채점합니다 | operations.md 의 XW 가드 규칙과 동일 |

**Z1 이 성립하는 이유(사전 등록)**: 위생 on + 아래쪽 전용 창이면 인덱스 0 카메라는 **어떤 W 에서도**
이웃이 없으므로 인코더 입력이 모든 구성에서 같습니다. 0 이 아닌 Δ 는 창의 효과가 아니라
**비결정성이거나 세션 간 누수** 신호입니다 (phase2 SPEC 개정 F4 와 동일).
15개 중 인덱스 0 채점 카메라가 있는 데이터셋은 9개
(AlexaMeadeExhibit, Barn, Blocks, Carpark, Fencing, MATF, Painter, PoznanStreet, Welder) — SAM 3와 같습니다.

## B.7 실험 집합

폴더 문법: **`SegMaskSam2XW{W}{모드}{G}{P}{S<s>}[all]`** (SAM 3 문법에서 `Sam3`→`Sam2`).

| # | 구성 | 플래그 | 출력 | 데이터셋 | 무엇을 분리 |
|---|---|---|---|---|---|
| B0 | 발표본 재현 | 없음 | `SegMaskSam2Legacy` | 15 | 가드 + A의 기준선 |
| **B1** | **패키지 내 대조군** | `--xview-window 0` | `SegMaskSam2XW0` | 15 | 교차시점 메모리 없음. **B의 대조군** |
| **B2** | **SAM 3가 고른 창** | `--xview-window 1` | `SegMaskSam2XW1` | 15 | cross-view memory 순효과 = B2 − B1 |
| B3 | 발표 창 + 위생 | `--xview-window 4` | `SegMaskSam2XW4` | 15 | C1/C2 결함의 몫 = B3 − B0 |
| **B4** | **채택 구성 이식** | `--xview-window 1 --xview-gate --xview-ptr --xview-tpos-shift 4` | `SegMaskSam2XW1GPS4` | 15 | GPS4 가 설계의 성질인가 |
| B5 | 손잡이 단독 G | `… --xview-gate` | `SegMaskSam2XW1G` | 15 | G 기여 (SAM 2에서는 절제) |
| B6 | 손잡이 단독 P | `… --xview-ptr` | `SegMaskSam2XW1P` | 15 | P 기여 |
| B7 | 손잡이 단독 S4 | `… --xview-tpos-shift 4` | `SegMaskSam2XW1S4` | 15 | S 기여 |
| B8 | closure==all | `--xview-window 1 --track-cams all` (그리고 GPS4판) | `SegMaskSam2XW1all`, `SegMaskSam2XW1GPS4all` | Welder, Dog | Z2 |
| B9 | 결정성 | B4 재실행 | `SegMaskSam2XW1GPS4_rerun` | Blocks | Z3 |
| B10 | 게이트 모드 감도 | `--xview-gate-mode occl` | `SegMaskSam2XW1Goccl` | 게이트 탈락이 있었던 데이터셋만 | G 대체물 선택의 민감도 |

**우선순위**: B0 → B1 → B2 → B4 가 핵심 4개입니다. 이것만 나오면 B의 질문에 답할 수 있습니다.
B3·B5~B7 은 여유가 있을 때. B8~B10 은 감시값이라 반드시 돌립니다(비용이 작음).

## B.8 사전 등록 엔드포인트 — SAM 3 쪽을 그대로 거울

### B.8.1 왜 그대로 옮길 수 있는가

**측정으로 확인**: SAM 2의 채점 카메라 45대 중 이웃을 받는(시점 인덱스 ≥ 1) 카메라가
**정확히 36대**, 인덱스 0 카메라가 **9대**입니다 — SAM 3와 **같은 36/9 분할**입니다
(같은 데이터셋·같은 `cam_list`·같은 아래쪽 전용 기하·같은 기준 규칙).
`eval/report_jf.py --split all --window 1 --nb-mode lower --bin-by-nb` 가 SAM 2 method 에도
그대로 동작함을 오늘 확인했습니다.

### B.8.2 1차 기준 (phase3-conditioning §1.3 과 동일한 형태)

> **이웃을 실제로 받는 카메라 36대**에서 **카메라 수준 짝지은 ΔJ&F**,
> 데이터셋 클러스터 부트스트랩 95% CI (`random.Random(0)`, 10,000회).
> **합격 = CI 하한 > 0.**

두 개의 1차 대조를 미리 등록합니다.

| 대조 | 질문 | SAM 3 쪽 값(비교 대상) |
|---|---|---|
| **E1-Sam2** = `XW1 − XW0` | 교차시점 메모리 자체가 SAM 2에서도 효과가 있는가 | A − XW0 = **+0.0029**, 15승 21패, 클러스터 CI **[−0.0002, +0.0085]** |
| **E2-Sam2** = `XW1GPS4 − XW1` | 채택된 세 손잡이가 SAM 2에서도 같은 방향인가 | GPS4 − XW1 1차 평균 **+0.00053**, CI 0 제외 |

```bash
R="--raw Data/MVSeg/jf_v2.json Data/MVSeg/jf_sam2_xw.json"
for P in "SegMaskSam2XW0 SegMaskSam2XW1" "SegMaskSam2XW1 SegMaskSam2XW1GPS4"; do
  set -- $P
  python3 eval/report_jf.py $R --methods $1 $2 \
    --split all --window 1 --nb-mode lower --bin-by-nb --paired $1 $2 \
    > docs/raw/sam2/b_e1_${2}.txt
done
```

### B.8.3 2차 기준

* **S1** 비기준 카메라(`--split nonref`)에서 같은 비교.
* **S2** 기준 카메라 무손실: `--split ref`, 카메라 수준 CI **상한 ≥ 0**.
* **S3** 헤드라인 15개 J&F (`SegMaskNew3`, `SegMaskSam2Legacy`, `XW0`, `XW1`, `XW4`, `XW1GPS4`,
  그리고 SAM 3의 `XW0`·`XW1GPS4`). 판정에는 쓰지 않습니다.
* **S4** 면적 가중(`--area-weighted`), inner 프레임(`--frames inner`).
* **S5** 게이트 진단: 각 폴더 `MANIFEST.json` 의 `provenance.xview_gate_stats` 를 데이터셋별로.
  탈락 0인 데이터셋에서는 `XW1G == XW1` 이어야 합니다(B-G7).
* **S6** C1/C2 결함의 몫: `XW4 − Legacy`(=`XW4 − SegMaskNew3`). SAM 3 쪽 대응값은
  phase2 §4.3 의 "MVOpt(legacy W=4) vs XW4(위생 W=4)".

### B.8.4 결정 규칙과 미리 적는 판독

**결정 규칙**: 감시값(B-G4~B-G7) 전부 통과 + S2 통과가 자격. 1차 평균이 가장 크면서 CI 가 0을
제외하는 구성이 승자. 0.001 이내 동률이거나 아무 구성도 CI 조건을 못 넘으면
**전부 보고하고 "SAM 2에서는 검출되지 않음"으로 씁니다.** 다중비교 보정 없음(명시).
CI 조건을 넘더라도 1차 평균 < +0.005 면 "통계적으로 검출되나 실질적으로 작다"를 함께 적습니다.

**핵심 판독표 — "공간축 메모리 발견은 SAM 3의 성질인가 설계의 성질인가"**

| 관측 | 사전 등록 결론 |
|---|---|
| E1-Sam2 가 SAM 3의 +0.0029 와 **같은 부호·같은 자릿수**(대략 +0.001 ~ +0.006) | **설계의 성질.** 두 백본에서 재현되므로 "이웃 뷰 메모리를 시간축 memory attention 에 붙이면 아주 작은 이득이 난다"는 일반 진술이 가능. 논문의 주장은 크기가 아니라 구조에 실어야 한다는 phase3 §8.4 결론이 강화됨 |
| E1-Sam2 ≈ 0 (CI 가 0 포함이고 평균 \|Δ\| < 0.001) | **SAM 3 쪽 +0.0029 도 SAM 3 고유거나 잡음.** 두 스택에서 모두 "검출 안 됨"이면 결론은 **이 채널로는 공간축 정보가 거의 전달되지 않는다** 로 수렴 — Phase 2·P4 의 결론과 일치 |
| E1-Sam2 가 **음수**이고 CI 상한 < 0 | **백본 의존.** SAM 2에서는 이웃 메모리가 해롭다. E6 와 무관한 새로운 발견이므로 별도로 보고 |
| E2-Sam2 가 SAM 3와 같은 방향(+) | 세 손잡이의 정당화(특히 S: "시점 한 칸은 프레임 한 칸보다 멀다")가 백본 독립. **단, G 는 §B.4.3 의 이탈 때문에 SAM 2에서 "대칭 회복"으로 해석할 수 없음을 반드시 병기** |
| S6 (`XW4 − Legacy`) 가 크다 | 발표된 SAM 2 열이 C1/C2 결함의 영향을 받고 있었다는 뜻. 그 크기를 SAM 3 쪽 phase2 §4.3 값과 나란히 보고 |

**미리 적는 예상**: SAM 3 쪽 E1 은 CI 하한이 −0.0002 로 0을 아슬아슬하게 포함했습니다.
SAM 2 쪽도 같은 자릿수·같은 결론(검출 경계)일 것으로 봅니다. **B의 가치는 "이겼다"가 아니라
"두 백본에서 같은 크기·같은 경계"라는 재현성 진술**에 있습니다.

**판정에 쓰지 않는 것**: 12개 부분집합, 객체 단위 일화, bin 없는 전체 평균, 실행 시간.

## B.9 GPU 계획

`--track-cams closure`(위생 on 이므로 허용)로 추정. closure 세션 수는 §B.3.4 표.
스윕당 마스크 시간 ≈ 74분 + 데이터셋당 오버헤드 1.5분 × 15 ≈ **1시간 36분**.
(참고: `--track-cams all` 이면 §A.6 처럼 2시간 35분.)

| 작업 | 스윕 | GPU 시간 |
|---|---|---|
| B-G1 가드 (Blocks·Fencing, all) | — | 8분 |
| **B1 XW0** | 15, closure | 1시간 36분 |
| **B2 XW1** | 15, closure | 1시간 36분 |
| **B4 XW1GPS4** | 15, closure | 1시간 36분 |
| B3 XW4 | 15, closure | 1시간 36분 |
| **핵심 4개 소계** | | **약 6시간 32분** |
| B5·B6·B7 단독 손잡이 | 15 × 3, closure | 4시간 48분 |
| B8 closure==all (Welder·Dog × 2구성, all) | 4회 | 42분 |
| B9 결정성 (Blocks) | 1회 | 5분 |
| B10 게이트 모드 (탈락 데이터셋만, 최대 4개) | ~4회 | 25분 |
| **B 합계 (전부)** | | **약 12시간 12분** |

**A + B 전체 ≈ 15시간 20분.** 한 데이터셋 = 한 컨테이너, `--memory=90g --memory-swap=90g`.
SAM 2는 46카메라에서도 완주 이력이 있으므로 OOM 위험은 SAM 3보다 낮습니다.

## B.10 실행 순서

```
1.  새 폴더 생성 + ORIGIN.md  (커밋 1: "바이트 동일 사본")
2.  T-A1·T-A2 (CPU)                                     ← fill_hole no-op 고정
3.  A/G1  CoffeeMartini 가드                            ← 여기서 막히면 전부 중단
4.  러너 + 매니페스트 + T-B1~T-B4 (CPU)                 (커밋 2: "러너")
5.  A/G2  Fencing fill_hole A/B
6.  A/G3  15개 legacy 스윕 + 점수 덤프
7.  apply_nonoverlap.py + T-A3·T-A4 → SegMaskSam2Matched
8.  A/G4  온라인 교차검증
9.  A 채점·집계·판독 (§A.7)                              (커밋 3: "A 결과")
10. xview 이식 + T-B5~T-B8 (CPU)                        (커밋 4: "XW 이식")
11. B-G1 가드 (Blocks·Fencing)
12. B1·B2·B4 스윕, 이어서 B3
13. B8·B9·B10 감시값
14. B 채점·집계·판독 (§B.8)                              (커밋 5: "B 결과")
15. 문서 갱신: experiments.md(폴더 목록·jf 파일), operations.md(SAM 2 러너 등록표·함정),
    ROADMAP 행 11(이름·상태), REPORT E6(§A.2 의 fill_hole 반증)
```

---

# 리스크와 명시적 한계

## 리스크

| # | 리스크 | 영향 | 완화 |
|---|---|---|---|
| R-1 | **발표 인용 열 `SegMaskNew1` 은 재현 불가**. 그 코드가 git 에 없음(`68b92ca` 가 수집 루프를 통째로 교체) | A의 산출물이 New1 계열이 아님 | 1차 판독을 **같은 실행 안의 짝지은 비교**(Matched − Legacy)로 잡아 계열 차이를 제거. New1↔New3 격차(J&F 0.0001, F 0.0002)를 각주 |
| R-2 | **`AlexaMeadeExhibit/SegMaskNew1` 이 두 실행의 이어붙이기** (1947장 2025-10-15 + 132장 2025-11-11: 세 카메라 프레임 0 전부 + camera_0001 프레임 1) | 그 데이터셋의 발표 수치가 단일 실행 산출이 아님 | 리스크로 공개. `--frames all` 과 `inner` 를 함께 보고. 재현 대상은 `SegMaskNew3`(단일 날짜) |
| R-3 | `sam2._C` 부재가 **오늘의 환경 사실**이고, 2025-10 에도 그랬다는 건 간접 증거(§A.2 구멍 측정 + New1 이 New3 보다 마스크가 작다는 Lens A 측정) | fill_hole 무효화 주장의 강도 | 구멍 측정이 사실상 결정적(면적 1~2 px 구멍이 New1 에 존재). 그래도 "이미지가 재빌드되면 이 성질이 바뀐다"를 매니페스트에 `sam2_C_available=False` 로 기록 |
| R-4 | **`build_sam.py` 의 죽은 중복 정의**(:145-188)를 고쳐 조용히 매칭 실패 | A 전체가 무효 | 살아 있는 앵커는 **:266**. 사본에서 죽은 정의를 삭제하고 T-B2 로 고정 |
| R-5 | **하이드라 오버라이드 순서** — `apply_postprocessing` 블록이 호출자 오버라이드 뒤에 붙어 `++model.fill_hole_area=0` 이 짐 | 매칭 실패 | 생성 후 속성 대입(§A.3.5) + 리드백 로그(B-G2) |
| R-6 | **`--track-cams closure` 로 spatial pass 가 바뀔 위험** — 카메라 수를 줄이면 유사 비디오의 `num_frames` 가 줄어 `max_obj_ptrs_in_encoder = min(num_frames, 16)` 이 바뀜 | closure 실행이 다른 모델이 됨 | `init_state`·spatial pass·seeding 은 **모든 카메라에 그대로** 두고 temporal 제너레이터만 줄임(§B.3.4). B8 로 실측 확인 |
| R-7 | legacy 에 closure 를 쓰면 wrap 때문에 다른 모델 | 대조 무효 | 러너가 `sys.exit` 로 거부 |
| R-8 | `tests/conftest.py` 가 SCSam3 패키지를 `sys.path[0]` 에 넣어 SAM 2 사본의 `xview_gather` 를 가림 | 테스트가 엉뚱한 파일을 검사 | SAM 2 테스트는 파일 경로 + 별도 모듈 이름으로 로드. torch 테스트는 별도 디렉터리 `tests_sam2/` + 자체 conftest |
| R-9 | `scsam2` 에 pytest 없음 | CPU 테스트 불가 | `--target` 벤더링(검증 완료). `.gitignore` 에 추가 |
| R-10 | 매니페스트가 `SCSAM3_IMAGE_ID` 를 읽음 | SAM 2 실행에 SAM 3 변수명 | 그 이름으로 scsam2 ID 를 넘기고 `extra["docker_image"]="scsam2:latest"` 병기. `eval/manifest.py` 는 수정하지 않음 |
| R-11 | G 게이트가 `ious` 저장을 요구 → 저장 상태가 바뀜 | 기본 경로 바이트 변화 | `if self.cross_view_gate` 로만 저장. B-G1 이 잡음 |
| R-12 | 손잡이 조합이 SAM 3와 다른 방향으로 나오면 "재현 실패"로 오독될 수 있음 | 해석 오류 | §B.8.4 에 네 갈래를 **실행 전에** 등록 |
| R-13 | 15시간 GPU. 중간에 이미지·드라이버가 바뀌면 계열이 갈라짐 | 비교 무효 | 매 실행 매니페스트에 이미지 ID·torch·cuda 기록. 가드 규칙(B-G7 아래)대로 코드 변경 시 B-G1 재실행 |
| R-14 | `eval_jf.py` 를 `scsam2` 로 돌리면 cv2 4.12/numpy 2.2 로 `scsam3` 와 부동소수 차이가 날 수 있음 | 점수 미세 불일치 | Frog 1개를 두 이미지에서 채점해 동일함을 먼저 확인. 다르면 `scsam3` 로 통일 |

## 매칭할 수 없는 것 (논문에 과장하지 않기 위해)

1. **low-res 마스크 격자 256 (SAM 2) vs 288 (SAM 3)** — `image_size 1024 / stride 4` vs
   `image_size 1008 / backbone_stride 14 × 4`. 백본 stride 를 바꾸면 다른 체크포인트입니다.
   **A 이후 남는 유일한 후처리 외 설정 교란항**이며, F 귀속은 **이 항에 조건부**임을 명시해야 합니다.
2. **메모리 선택 (temporal disambiguation)** — SAM 3는 자기 과거 기억을 `eff_iou_score > 0.01` 로
   걸러 쓰고 SAM 2는 그 기제가 없습니다. 그래서 **B의 G 손잡이는 SAM 2에서 "대칭 회복"이 아니라
   새로 만드는 비대칭**이고, 순수 절제로만 해석해야 합니다.
3. **백본·디코더 가중치와 학습 데이터** — 당연하지만 명시. A는 후처리 몫만 분리합니다.
4. **`SegMaskNew1` 을 만든 실행 자체** — 코드가 git 에 없습니다. 재현 가능한 SAM 2 계열은 New3 뿐입니다.
5. **`fill_hole_area` 의 "의도된" 효과** — 두 스택 모두 실제로는 구멍을 메우지 않았으므로(§A.2)
   "SAM 2가 구멍을 메웠다"는 서술은 **REPORT E6 에서 삭제**해야 합니다. 설정값 차이는 남지만 효과는 0입니다.
6. **`AlexaMeadeExhibit` 의 발표 수치** — 이어붙이기(R-2). 단일 실행 재현과 1:1 대응하지 않습니다.
