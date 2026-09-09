# SCSam3 OneStageNew (multi-view cross-view memory tracker) 분석 보고서

대상: `/home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3OneStageNew/` (SAM 3 fork), 실행기 `SCSam3/runMVSeg.py`, 평가기 `Data/MVSeg/eval_jf.py`, `report_jf.py`.
아래의 모든 수치는 BRIEF.md 및 이번 세션의 검증 결과(컨테이너 내 재실행, per-frame 재계산, 코드 대조)에서 나온 것이며, 메커니즘 서술은 전부 코드 라인을 직접 확인한 것입니다.

---

# 알고리즘 개요

설계 의도(저자의 용어로): N대의 동기화된 카메라에 대해 **한 기준 뷰(reference view)의 시작 프레임 한 장에만 프롬프트**를 주고, (1) 카메라를 "프레임"으로 취급한 pseudo-video로 마스크를 모든 뷰에 전파(spatial pass)한 뒤, (2) 각 뷰의 temporal tracker가 자기 뷰의 memory bank(`num_maskmem=7`)에 더해 **이웃 뷰의 `maskmem_features`를 memory attention prompt에 그대로 이어 붙여**(NewMem) 뷰 간 정보를 공유하며 lockstep으로 추적하는 방식입니다.

```
[프롬프트: ref view, frame t0]
        │
        ▼
 (1) spatial pass  ── predictor_spatial (plain SAM 3 tracker)
     pseudo-video [view0 @t0, view1 @t0, ..., viewN-1 @t0]
     propagate 'both' from ref  ──►  masks_spatial[view][obj]
        │  (zero mask if absent)             runMVSeg.py:192-230
        ▼
 (2) temporal pass ── predictor (NewMem), 뷰당 세션 1개, 객체당 tracker state 1개
     for t:                                     runMVSeg.py:314-316 (lockstep)
       for v in 0..N-1:
         prompt = [own cond] + [own t-1..t-6]
                + [view v-1..v-4 의 maskmem @ frame t]   ◄── 기여의 핵심
                + [own obj_ptr]                  SCSam3TrackerPredictorNewMem.py:1253-1264, 1296-1314
         memory attention ─► SAM decoder ─► memory encoder ─► output_dict[v][t]
```

- 이웃 창은 `for s_pos in range(-4, 0)` (SCSam3TrackerPredictorNewMem.py:1253)로 하드코딩, **같은 프레임 t**의 non-cond 출력만 조회(:1260), 이웃 토큰은 자기 뷰 t-1..t-4용 temporal embedding `maskmem_tpos_enc[abs(s_pos)-1]`(:1312)을 그대로 씀.
- tracker 코어(`_prepare_memory_conditioned_features_multiple`, `track_step_multiple`, `_run_single_frame_inference_multiple`)는 difflib 대조 결과 upstream `sam3_tracker_base.py:559/929`, `sam3_tracking_predictor.py:1050`과 `output_dict -> output_dicts[spatial_idx]` 치환 + 위 30줄 블록을 제외하면 동일. 이 블록은 SAM 2 버전(`SCSam2VideoPredictorNew.py:501-541`)에서 그대로 옮겨온 것입니다.
- 실질적 기여 코드는 ~600줄(spatial 블록, `add_tracker_new_mask`, `_propogate_tracker_one_frame_local_gpu_multiple`, session/request 배관)이며, 나머지 ~1,700줄짜리 파일 6개는 upstream 복사본입니다.

---

# 강점

1. **파이프라인이 실제로 동작하고, 12개 데이터셋에서 SAM 2 버전과 동등 이상.** 12-dataset J&F: SAM2 0.8449 / OneStage 0.8465 / OneStageNew 0.8497. Barn +0.030, Breakfast +0.028, MATF +0.019, Dog +0.017 등에서 SAM 2 대비 우세. F(경계)가 SAM 2보다 일관되게 높습니다(0.8727 → 0.8820).
2. **cross-view memory가 "죽은 트랙을 되살리는" 실제 사례가 있음.** Fencing v9 obj 3(평균 3,380 px, 프레임의 0.16%): OneStage J 0.128 → OneStageNew 0.547(frame 8 이후 IoU 0.63 유지), obj 4 0.473 → 0.579; PoznanStreet v8 obj 2: 0.219 → 0.901. 이웃 뷰에서 얇은 물체가 살아 있으면 자기 뷰의 memory bank가 무너져도 복구된다는 메커니즘의 존재 증거입니다.
3. **뷰 간 object 정합을 `obj_id`로 수행하는 것은 SAM 2 버전보다 견고함.** `_get_tracker_inference_states_by_obj_ids`(SCSam3VideoInferenceNewMem.py:1428-1439)로 각 뷰에서 같은 id의 tracker state를 찾아 `iss` 리스트를 만듭니다(:1029-1035). SAM 2 버전은 `obj_idx` 위치 정합(SCSam2VideoPredictorNew.py:945-947)이라 등록 순서가 다르면 객체가 섞였습니다.
4. **재현성이 좋음.** OneStage 두 번의 sweep(logs/mvseg-20260903-085851 vs -104557)에서 45개 카메라 중 41개가 bit-identical(4개는 bug #2 케이스). 노이즈 바닥이 낮아 ablation을 정확히 할 수 있는 조건입니다.
5. **cross-view memory 자체의 GPU 비용은 작음.** 스텝당 prompt는 최대 11 × 5184 × 64 bf16 ≈ 7 MB, K/V ≈ 29 MB/layer, flash SDPA로 score matrix 미실체화. Blocks에서 측정한 프레임당 transient 피크는 1.0 GiB 전체. 뷰-프레임당 wall time도 OneStage 0.56 s vs OneStageNew 0.44 s(Welder)로 **추가 비용이 측정되지 않음**. 즉 OOM/속도 문제는 알고리즘의 본질적 한계가 아니라 배관과 상태 관리 문제입니다(아래 참조).

---

# 검증된 문제점

## 정확성

**C1. 음수 인덱스 wrap + lockstep → 뷰 0..3은 이웃을 0..3개만 받고, `--track-cams written`(3세션)에서는 IndexError.**
`prev_spatial_idx = spatial_idx + s_pos`(SCSam3TrackerPredictorNewMem.py:1254)가 음수일 때 `output_dicts[...]`(:1256, :1260)는 Python list이므로 뷰 N-1..N-4로 wrap. lockstep(runMVSeg.py:314-316)에서 그 뷰들은 아직 프레임 t를 만들지 않았으므로 `.get(frame_idx)`가 None → 스킵(:1297-1298). 결과: 유효 이웃 수 = min(v, 4). 스코어링되는 36개 카메라 중 8개(Barn v0, Blocks cam0, Carpark v0, Fencing v0, MATF S1_CAM_1, Painter v0, PoznanStreet v0, Welder camera_0001)가 index 0이고, 그중 6개가 **reference 카메라**입니다. 세션이 4개 미만이면 `output_dicts[-4]`가 IndexError(`runMVSeg.py --algo OneStageNew --track-cams written`이 이 케이스). 링(ring) wrap은 설계된 것이 아니라 우연히 생긴 것이며, 실제로는 동작하지 않습니다.

**C2. 죽은 cond-frame fallback이 `selected_cond_outputs`를 덮어써서, 뷰 0..3이 마지막 카메라(view N-1)의 seed object pointer를 자기 것 대신 사용.**
:1262 `selected_cond_outputs, unselected_cond_outputs1 = select_closest_cond_frames(...)`가 자기 뷰 cond dict(:1193)를 이웃 뷰 것으로 재바인딩. 이 값은 :1335-1341에서 cond object pointer로 소비됩니다. fallback 자체는 절대 hit하지 않음(`sam3_tracker_utils.py:286-288`: cond frame ≤ `max_cond_frames_in_attn=4`면 `unselected={}`). C1의 wrap 때문에 뷰 0..3에서 매 프레임 실행되며 마지막으로 실행되는 것은 항상 view N-1. zero-mask로 seed된 객체(runMVSeg.py:228-229)는 그 뷰에서 `obj_ptr = no_obj_ptr`, `object_score_logits = -10`(sam3_tracker_base.py:419-425)이므로, **자기 뷰에 있는 객체에 "없음" pointer가 주입**됩니다. SAM 2 버전에도 같은 결함(SCSam2VideoPredictorNew.py:506)이 있어 SAM 2 대비 손실의 단독 원인은 아니지만, OneStage-vs-OneStageNew 비교를 오염시킵니다(E2 참조). 수정은 한 줄: `_, unselected_cond_outputs1 = ...`.

**C3. [brief #2 root cause 확정] 기준 뷰가 마지막 index일 때 explicit `start_frame_index`가 cross-view pass를 무너뜨림.**
`propagation_direction='both'`는 같은 state에 forward → backward를 같은 start로 두 번 호출(SCSam3VideoPredictor.py:207-224). backward leg에서 `parse_action_history_for_propagation`(SCSam3VideoInference.py:1264-1274)이 `action_history[-1]["frame_idx"] in [0, num_frames-1]`이면 `"propagation_fetch"`를 반환 → 캐시(:1052-1054)만 읽으므로 방문 안 한 뷰는 `{}` → zero mask seed → Blocks/Painter 비기준 뷰 J≈0.003. hotstart/`tracking_bounds`는 원인이 아님(`tracking_bounds`는 `sam3_image.py:714`의 `**kwargs`에 삼켜져 upstream에서도 dead). start=None이면 `None in [0, N-1]`이 False라 정상 동작. upstream 로직이 그대로 복사된 것입니다.

**C4. [brief #1] 요청 키 불일치.** `SCSam3Video.py:177, 228`은 `start_frame_idx`를 보내고 predictor는 `start_frame_index`를 읽음(SCSam3VideoPredictorNewMem.py:103, SCSam3VideoPredictor.py:102). 무시되어 None + direction `'both'` 기본값(:102) → 데모는 전체 영상을 양방향 추적(Frog 180 프레임 낭비). 이 우연한 None 덕분에 데모는 C3를 피해감.

**C5. [brief #3] SAM 2 API 잔재가 호출 시 예외.** `LoadVideo_Folder`(SCSam3Video.py:31-40), `LoadVideo_Folder_MVSeg`(:42-51), `AddMaskSingle`(:156-163), `AddMask`(:165-171), `RunTracking`(:189-192) 모두 존재하지 않는 메서드/속성 호출. `from urllib import response`(:1) stray import. tracker의 `_reset_tracking_results`(SCSam3TrackerPredictorNewMem.py:140-153)는 :1027의 upstream 정의에 shadow되는 dead code(살아나면 `frames_tracked_per_obj` KeyError).

**C6. MultiGPU 서브클래스가 import되지 않은 upstream 이름을 참조.** `Sam3VideoPredictorMultiGPU`(SCSam3VideoPredictorNewMem.py:397, :476; SCSam3VideoPredictor.py:391, :470). `SCSam3Video.py:15`가 `gpus_to_use=range(device_count())`를 넘기므로 GPU 2개 이상 호스트에서는 생성자에서 NameError. 현재 결과는 1-GPU 호스트라 무관하지만, "OOM이면 큰 머신으로"라는 자연스러운 우회가 막혀 있습니다. 설계상 lockstep은 어차피 single-GPU(worker는 stream을 eager하게 소진, :500-502).

**C7. 텍스트/박스 프롬프트를 NewMem temporal 세션에 주면 upstream 내부에서 TypeError.** 빈 action_history → `propagation_full` → base `_propogate_tracker_one_frame_local_gpu`(sam3_video_base.py:1118)가 `tracker.propagate_in_video(inference_state, ...)`를 호출하나 NewMem 시그니처는 `(inference_states, spatial_idx, ...)`(SCSam3TrackerPredictorNewMem.py:808-819). 따라서 hotstart(15/8/8), masklet confirmation, recondition_every_nth_frame=16 등 build_scsam3.py:927-937의 파라미터는 NewMem 경로에서 **전부 inert**입니다. mask-seeded 경로에서는 spatial pass도 `propagation_partial`이라 detector 결과를 쓰지 않습니다.

**C8. 사소하지만 잠재적 결함.** `seq_len`이 temporal 루프의 stale 값 재사용(:1304); `previous_stages_out`는 list인데 dict처럼 `frame_idx not in ...`(SCSam3VideoInferenceNewMem.py:1997)로 검사해 항상 True(우연히 동작); `print(past_out.get('eff_iou_score'))`(:1590)는 `trim_past_non_cond_mem_for_eval`을 켜면 stdout 폭주; `io_utils.py:202`의 `offload_video_to_cpu=False` 경로는 UnboundLocalError; `AsyncVideoFrameLoaderFile.__getitem__`(io_utils.py:158)은 `cap.read()`의 `ret`를 검사 안 함.

## 알고리즘 설계

**A1. 정보는 세션 리스트 순서로 위쪽으로만 흐른다.** 뷰 v는 뷰 0..v-1의 프레임 t만 보고, 상위 뷰는 절대 하위 뷰에 피드백하지 않음(:1253, :1260 같은 frame_idx). 결과가 카메라 기하가 아니라 `perms` 순서를 인코딩합니다(Dog perms는 카메라를 건너뜀; `sam3_demoVideo.py:17`은 `[0..7,16..31,8..15]`). 측정: 효과가 있는 두 데이터셋에서 모두 고인덱스 카메라만 이득(Fencing v4 +0.001 vs v9 +0.066; PoznanStreet v4 +0.005 vs v8 +0.029). nb 구간별 ΔJ: nb=0 −0.0001(8캠), nb=1 −0.0004, nb=2 +0.0053, nb=3 +0.0002, nb=4 +0.0047(21캠).

**A2. Fencing 손실은 전부 v0(reference, index 0, 이웃 없음)에서 발생.** v0 J1-20 0.794 vs SAM 2 0.890(F 0.924 vs 0.987); obj 3/4 J 0.32/0.23 vs SAM 2 0.63/0.68. 같은 두 객체가 이웃 4개를 받는 v9에서는 0.55/0.58로 복구됨. v4, v9는 SAM 2 이상. 즉 **복구 메커니즘이 정확히 필요한 카메라에 구조적으로 닿지 않습니다.**

**A3. Welder 손실은 전부 camera_0004(index 3)의 cross-view seed 실패.** spatial pass가 obj 8/14에 4 px / 7 px seed(GT 13,268 / 18,899 px), obj 12 seed IoU 0.184를 넘김. SAM 2 pass는 같은 객체를 IoU 0.94/0.93/0.90으로 seed하고 J 0.93/0.93/0.90으로 추적. SAM 3 두 변형 모두 J 0.0/0.0/0.14 (21프레임). seed는 cond memory(:1291, tpos row num_maskmem−1)와 cond pointer(:1335-1341)로 영구 고정되며 이웃 memory(뷰 0/1/2는 객체를 보유)로도 뒤집히지 않음(OneStageNew ΔJ +0.0004). **temporal cross-view memory는 잘못된 seed를 구조적으로 복구할 수 없습니다.** camera_0004 하나(0.861 vs 0.707)가 Welder −0.034 전체를 설명.

**A4. 이웃 토큰은 "같은 카메라의 t−k 프레임"으로 위장되어 들어간다.** `maskmem_tpos_enc[abs(s_pos)-1]`(:1312)은 자기 뷰 t−1..t−4의 embedding(:1291)과 byte-identical. `cond_frame_spatial_embedding`은 이 체크포인트에 없음(:1283 getattr). memory cross-attention은 `rope_k_repeat=True`(build_scsam3.py:388-400; sam/rope.py:76-78)라 이웃 토큰도 자기 뷰 72×72 픽셀 그리드 RoPE를 받음 → 쿼리 픽셀 (i,j)가 이웃 카메라의 (i,j)를 편향 조회. 즉 **이웃 카메라가 픽셀 정렬되어 있고 이웃 마스크가 "한 프레임 전 물체"라는 가정**이 암묵적으로 들어가 있습니다. 관측과 일치: 잘 추적되는 카메라에서는 중립~약한 음수(Blocks cam9 −0.003, Frog −0.0005..−0.0008, Breakfast v5 F −0.0046), 자기 트랙이 죽었을 때만 양수.

**A5. 이웃 memory는 memory selection을 우회.** 자기 뷰 non-cond memory는 `frame_filter`(eff_iou > 0.01, sam3_tracker_base.py:517-557)로 걸러지지만 :1260은 무조건 취함. zero-mask seed된 이웃은 매 프레임 `no_obj_embed_spatial` memory를 recency code 0(s_pos=−1)으로 주입. 다만 검증 결과 **관측 가능한 회귀는 없음**(OneStageNew에서 GT-nonempty missing pair가 오히려 1963 → 1203으로 감소; Painter v6의 추가 missing 14건은 전부 GT-empty 프레임). 위생 항목이지 손실 원인은 아닙니다.

**A6. 이웃 `obj_ptr`(256-d, GPU 상주)은 사용되지 않음**(:1335-1368은 `output_dicts[spatial_idx]`만). SAM 3에서 pointer가 object/no-object 결정을 담으므로 cross-view 존재 신호로 가장 자연스러운 채널이 비어 있습니다.

**A7. [brief #4, #7] reference 규칙이 docstring("most objects")과 달리 max id를 택함(runMVSeg.py:102 `np.max(img)`; SAM 2 스크립트 :44-50도 동일).** reference seed 프레임에 없는 객체는 zero prompt → zero-area filter(SCSam3VideoInference.py:464) → 미등록 → 모든 프레임 J=F=0. GT만으로 계산: 21,651개 scored (obj,frame) 쌍 중 1,579개(7.3%)가 **어떤 방법으로도 도달 불가**. MATF 372/1611, FlameSteak 399/2814, CoffeeMartini 273/3003, Blocks 203/1085(cam0 34%), FlameSteak cam01 42%. count 규칙이면 Blocks 203→63, MATF 372→231, PoznanStreet 42→0, Painter 99→61, FacePaint 105→63. MATF/Blocks의 낮은 점수는 상당 부분 추적이 아닌 규칙의 결과입니다.

## 메모리·성능

**M1. 46 GB OOM의 실제 구성 (측정, Blocks 10뷰×15객체 @1080p에서 walk 후 스케일링).** 모델 6.706 GiB(2모델, params 6.425) + 뷰당 0.344 GiB (detector cache 0.277 + 2×O×H×W bool + pos-enc 클론) + spatial 세션. AlexaMeadeExhibit(45×33 @2560×1920) 환산 ≈ 38 GiB resident. 로그(logs/mvseg-20260903-104557/OneStageNew-AlexaMeadeExhibit.log:298)의 OOM은 **첫 프레임(0/22)**에서 `_postprocess_output` → `_apply_object_wise_non_overlapping_constraints`(SCSam3VideoInferenceNewMem.py:508 → TrackerNewMem :1926)에서 "Tried to allocate 976 MiB", 44.87 GiB allocated. **cross-view memory prompt는 이 예산의 수십 MB에 불과**하므로 window/num_maskmem을 줄여도 OOM은 그대로입니다. OOM은 "모델 크기"가 아니라 "모든 N뷰에 세션을 열고 뷰·객체당 full-res 상태를 고정하는 것"입니다.

**M2. seed 마스크가 뷰·객체당 2번 full-res로 GPU에 고정.** (a) `previous_stages_out[frame_idx] = out`(SCSam3VideoInferenceNewMem.py:1998; points 경로 :1749; spatial :1911)이 `_build_tracker_output`(:569-572)이 GPU로 되올린 전체 객체 마스크 dict를 보관. 저자의 `.cpu()` 패치(:549-552)는 `.copy()`한 사본만 옮기므로 원본은 그대로. 읽는 곳은 `is None` 검사뿐(:189, :226, :234); upstream은 문자열 sentinel(:400). (b) tracker의 `mask_inputs_per_frame[frame_idx] = mask_inputs_video_res`(TrackerNewMem :407, bool video-res, device=cuda)는 이후 key 조회(:759, :951, :964, :1775, :1808)에만 쓰임. 합계 AlexaMeade ~13.6 GiB, CoffeeMartini ~12 GiB, FlameSteak ~14.6 GiB. (b)는 upstream 상속이고 SAM 2도 1024² float를 보관했으므로 SAM 2 대비 회귀는 아니지만, 낭비 자체는 사실입니다.

**M3. 호스트 RAM: CoffeeMartini/FlameSteak는 CUDA OOM이 아니라 host-RAM 사망.** 로그(mvseg-20260903-114432/OneStageNew-FlameSteak.log)는 `11/22 [25:21<1:14:43, 407.58s/it]`에서 traceback 없이 끊기고, it/s가 38→44→75→221→407 s로 폭주(thrash 시그니처). CoffeeMartini도 12-13/22에서 동일. 원인: (a) `cached_frame_outputs`가 뷰·객체·프레임당 H×W bool을 CPU에 영구 보관(:554, 저자 패치가 GPU 누수를 host 누수로 전환); (b) `inference_state["mask_inputs_per_obj"][obj_id][frame_idx] = mask`(:1967)가 호출자의 fp32 video-res seed(21.9 MB)를 보관, **어디서도 읽지 않음**(Blocks에서 141개 고유 storage 1.089 GiB 측정); (c) maskmem bf16 + pred_masks가 CPU에 무제한 축적(`trim_past_non_cond_mem_for_eval=False`). FlameSteak(21뷰×53 등록객체): seed ~24 GB + 프레임당 ~7 GB → 11프레임에 ~100 GB. 해당 sweep(114432)은 `--memory` 캡 없이 돌았습니다(runMVSegAll.sh:58; 캡은 runForSam2Three.sh:13에만). Brief의 "clean CUDA OOM" 서술은 AlexaMeade에만 맞습니다.

**M4. non-overlap 후처리가 bool을 int64로 승격.** `torch.clamp(pred_masks, max=background_value)`(TrackerNewMem :1929, bool + Python int 0)가 int64 (N,1,H,W) → `torch.where` 결과도 int64. 26×1920×2560×8 B = 975 MiB = 로그의 976 MiB. 블록 피크 3.46 GiB(26 마스크), 68 마스크 @2704×2028는 ~10 GiB. 매 yield(V×21회) + 매 seeding add에서 실행. upstream 상속 코드(sam3_tracking_predictor.py:1349-1369)이며, bool 입력·background 0이면 수식이 `pred_masks & (pixel_level>0)`로 축약됩니다.

**M5. 뷰당 detector 잔재 0.277 GiB.** `_prepare_backbone_feats`(SCSam3VideoInferenceNewMem.py:1441-1458)가 매 뷰·프레임에 full detector(backbone+grounding+seg head+NMS)를 돌리고 `_ =`로 버림(:1450). `forward_video_grounding_multigpu`는 world_size=1에서도 다음 프레임을 prefetch(sam3_image.py:763-786)하여 2프레임 상주; 프레임당 pred_masks(1,200,288,288) bf16 33 MB + FPN 55.7 MB + pos-enc 55.7 MB(매 호출 `.repeat` 신규 할당, position_encoding.py:95). 45뷰 = 12.5 GiB. 시간 측면: ViT trunk(60 ms)는 tracker feature에 필요하므로 낭비가 아니고, grounding head 23 ms/뷰-프레임만 낭비(Welder 전체의 ~4%).

**M6. 두 모델 + 종료되지 않는 autocast 캐시.** `SCSam3Video.py:16, 19`가 동일 체크포인트로 모델 2개(6.4 GiB). spatial 모델은 `PropagateAcrossViews` 이후 미사용이나 세션도 닫지 않음(`close_session` 호출 없음; `_ALL_INFERENCE_STATES`는 클래스 속성). `bf16_context.__enter__()`(TrackerNewMem :51-52, upstream 동일)가 영구 → autocast weight-cast 캐시가 절대 비워지지 않아 모델당 ~1.6 GiB(측정: 3.18 GiB unreachable). spatial pseudo-video는 `img/255.0`(io_utils.py:32) float64로 N×24.4 MB GPU 상주(upstream은 fp16, 6 MB).

**M7. seeding이 객체 수 제곱.** add_prompt마다 `_build_tracker_output`이 캐시 전체를 GPU로 재업로드(:569-572), `_cache_frame_outputs`가 전체를 다시 `.cpu()`(:549-552), `_postprocess_output`이 누적 객체 전체에 full-res non-overlap. 컨테이너 재현: AlexaMeade 120 s(OOM까지 186 s 중), CoffeeMartini 217 s, FlameSteak 260 s; 완료 데이터셋에서 1-11%. 응답은 `TrackForward`가 버림(runMVSeg.py:221-230).

**M8. 뷰·객체당 tracker 호출 B=1.** `_propogate_tracker_one_frame_local_gpu_multiple`(:1022-1052)이 객체마다 별도 1-frame generator; SAM 2 버전은 카메라당 state 하나에 전 객체 배치(SCSam2VideoNew.py:123-142). FLOP은 어차피 객체별(query가 객체별 expand, :1072-1086)이라 배치가 FLOP을 줄이진 않지만 launch/전송 오버헤드(뷰-프레임당 pageable H2D 13×O회)는 줄어듭니다. 또한 `clear_non_cond_mem_around_input`이 batch_size≤1 조건으로 현재 항상 켜져 있어(:728-733) 배치화하면 동작이 바뀝니다.

**M9. maskmem_pos_enc가 tracker state마다 클론**(:1727-1731; 0.66 MB × V × O ≈ 1 GB on AlexaMeade). 소소.

## 실험 설계

**E1. +0.0032 J&F는 3개 카메라의 3-4개 작은 객체.** 36개 scored 카메라 중 31-32개가 |ΔJ|≤0.004. Fencing +0.025 → 12-dataset 평균 +0.0021, PoznanStreet +0.0125 → +0.0010, 합 0.0031/0.0032. 객체 수준: Fencing v9 id3 ΔJ +0.419(3,380 px), id4 +0.106; PoznanStreet v8 id2 +0.681; Welder camera_0003 id4 +0.138(1,344 px). `eval_jf.py:183-188`은 객체별 평균(면적 무가중)이라 3k px 객체가 111k px 객체(Fencing id7)와 같은 가중치; 면적가중 J는 0.9253 vs 0.9254. 12-dataset paired sign test 7+/5−, p≈0.39. 세 객체를 OneStage 값으로 고정하면 0.8497 → 0.8468(≈OneStage 0.8465).

**E2. OneStage vs OneStageNew는 통제된 ablation이 아님.** 다른 폴더(sys.path 교체, runMVSeg.py:117-119), 다른 request 형태(hasattr 스니핑 :240), 다른 세션 수(`track_mode`: OneStage=written 3캠, OneStageNew=all, :292). **이웃을 0개 받는 index-0 카메라 8개도 seed 프레임 이후 매 프레임 PNG가 다름**(per-(obj,frame) |ΔJ| 최대 0.49 Fencing v0 f17, 0.43 Barn v0, 카메라 평균은 ±0.0015 이내). 그 원인은 C2(마지막 뷰의 pointer 주입)로 확정되었습니다. 즉 index-0 카메라는 "cross-view 없음" 대조군이 아니며, ±0.005 이하의 어떤 카메라 차이도 창의 효과로 귀속할 수 없습니다. 같은 패키지에서 `range(-4,0)`을 비운 대조군은 한 번도 돌지 않았습니다.

**E3. 12-dataset 부분집합이 SAM 3 vs SAM 2 부호를 뒤집음.** 15-dataset 전체: SAM2 0.8434 vs OneStage 0.8427 (J 0.8125 vs 0.8081). 제외된 3개가 가장 무거운 장면(45/18/21뷰)이고 AlexaMeade −0.030, FlameSteak −0.010, CoffeeMartini +0.008. OneStageNew는 그 셋에 숫자가 없음. 두 차이 모두 노이즈 수준(SE ≈0.006)이므로 정직한 결론은 "동률". 추가로 `runMVSegAll.sh:107-110`은 `report_jf.py`를 `--common` 없이 호출해 AVERAGE 행이 12-dataset과 15-dataset 분모를 섞습니다(`--common`은 report_jf.py:104-106에 이미 존재).

**E4. reference 카메라(점수의 1/3)는 GT seed에서 시작하는 single-view VOS.** seed 프레임은 저장된 cond 출력(GT를 288² low-res로 왕복, TrackerNewMem :446-448, :866-872)이라 J0 0.962(SAM 2 0.951). 비기준 카메라만: 0.8052 / 0.8093 / 0.8144 → reference가 모든 방법을 ~0.04 올리고 multi-view 신호를 1/3로 희석. seed 프레임 자체는 +0.0012..0.0016 균일(순위 불변; `J_inner`는 이미 계산됨 :185-186).

**E5. 타이밍 비교는 세션 수 비교.** Welder OneStage 3세션 vs OneStageNew 46세션. tqdm으로 분리하면 temporal 35 s/63 = 0.56 s vs 427 s/966 = 0.44 s per view-frame; cross-view pass 16 s vs 15 s. "2-3× 느림"은 cross-view memory 비용이 아니라 "46뷰 vs 3뷰"입니다.

**E6. SAM 2 vs SAM 3 설정 불일치.** — **2026-09-09 정정: 아래 세 변수 중 `fill_hole_area`는 처음부터
무효였습니다.** `scsam2` 이미지에 CUDA 확장 `sam2._C`가 없어 `fill_holes_in_mask_scores`가 실행된 적이
없습니다([sam2-baseline.md](../sam2-baseline.md)). 따라서 남는 교란은 **출력 겹침 제거 하나**(사후에
객체별 점수만으로 적용 가능)와 **맞출 수 없는 저해상도 288 대 256**입니다. 원문: SAM 2는 `fill_hole_area=8`(SCSam2/demo/build_sam.py:130, low-res 256²에서 적용 = full-res ~250 px)이고 SAM 3는 `SCSam3Video.py:17, 20`에서 0으로 강제(builder 기본 16, build_scsam3.py:936); SAM 3는 non-overlap을 출력에 적용하고 SAM 2는 `non_overlap_masks=False`; low-res 288 vs 256. F +0.006 차이는 이 변수들과 같은 크기입니다. 등록 객체 22 vs 15는 로그 규칙 차이(SAM 2는 zero-mask id도 카운트)일 뿐 점수 무관.

**E7. 빠진 ablation.** window 크기(:1253 리터럴, 생성자 인자 없음; maskmem_tpos_enc 7행이라 W≤6만 가능), 패키지 내 on/off, 양측 창(lockstep 때문에 현재 구조로는 불가능), 뷰 순서/링, reference 선택, seed 모달리티(텍스트는 C7로 불가), horizon(21프레임, num_maskmem=7이면 steady state는 7프레임부터), scored 카메라는 GT로 고정(변경 불가).

## 엔지니어링

**G1. [즉시 조치] HuggingFace 토큰 노출.** `README.md:3`(uncommitted working copy, `git diff`에 `hf_...` 34자)과 `SCSam3/Dockerfile:35-36`(`ARG HF_TOKEN` + `RUN hf download --token`). 후자는 로컬 `scsam3:latest` 이미지의 `docker history --no-trunc`에 **다른 두 번째 토큰**이 3개 레이어에 기록되어 있음이 확인됨. 커밋 습관(`git add -A` 식 "추가/수정")상 다음 커밋이 public origin에 올립니다. 두 토큰 모두 폐기, README 줄 삭제, `--mount=type=secret` 또는 런타임 env로 교체, 이미지 재빌드.

**G2. 포크-편집 구조.** 1,712줄 upstream 파일 대비 1,645-1,687줄 공유; import한 upstream 클래스(SCSam3VideoInference.py:22, NewMem :23)는 미사용; 동일 스택이 6개 폴더(ForSam2, ForSam2New, OneStage, OneStageNew, TwoStage, TwoStageNew)에 복제, `SCSam3VideoInference.py`는 4개 폴더에서 md5 동일. C3 수정은 OneStageNew 안에서만 2곳(SCSam3VideoInference.py:1269, NewMem :1352-1361) + 다른 폴더. OneStageNew 안에서 plain/NewMem 두 스택이 동시에 live(build_scsam3.py:39-45). `SCSam3TrackerPredictorNewMem copy.py`(brief #5)는 live 파일과 한 줄 차이.

**G3. 테스트 부재.** `test.py`는 dict 길이 출력 4줄. 유일한 게이트 `waitAndRunMVSeg.sh:54-59`는 `--no-eval` + `grep "all runs finished"`라 all-black 마스크도 통과. C3/C4는 모델 없이 밀리초에 테스트 가능한 순수 Python 로직.

**G4. 설정이 monkey-patch/리터럴.** `fill_hole_area=0` 런타임 덮어쓰기(SCSam3Video.py:17, 20; runMVSeg 배너 :271-279에 안 보임), window `range(-4,0)`(:1253, TwoStageNew 복사본과 md5 동일), `perms`/경로/프롬프트/프레임 수가 데모 리터럴(sam3_demoVideo.py:17, 20, 24, 50).

**G5. 상속된 upstream 파라미터의 오해 소지.** hotstart 15/8/8, recondition 16, keep_alive 30(build_scsam3.py:927-937)은 NewMem 경로에서 inert(C7), spatial pass에서도 mask 프롬프트면 inert. 텍스트 프롬프트 spatial pass에서만 살아나며 그때는 "8개 카메라에서 detection 미매칭이면 객체 삭제", "뷰 0/16/32를 cond frame으로 재시딩" 같은 뷰-간 의미로 작동합니다.

---

# 개선 제안 (우선순위순)

세 심사자의 top-5 교집합: **(P1) seed/프레임 마스크 고정 해제**와 **(P2) 패키지 내 window ablation + closure 트랙 모드**는 3/3, **(P3) 평가 프로토콜**은 2/3(나머지 1명도 3/5/5), **(P4) cross-view gather 재작성**은 세 심사자가 각기 다른 변형을 top에 올렸으나 같은 메커니즘, **(P5) seed 복구**도 2/3. 그 아래는 점수순.

> **0순위(제안 아님, 즉시): G1 토큰 폐기.** 코드 변경 전에 두 토큰을 huggingface.co에서 revoke하고 `README.md` 2-3행을 삭제하십시오.

### P1. seed/프레임 마스크 고정 해제 (심사 3/3 합의, impact 5·feasibility 5·evidence 5)
- **무엇을**: (a) `SCSam3VideoInferenceNewMem.py:1998`(및 :1749, SCSam3VideoInference.py:1911)에서 `previous_stages_out[frame_idx] = out` → upstream sentinel 문자열(:400과 동일). (b) :1967 `mask_inputs_per_obj[obj_id][frame_idx] = mask` 삭제(또는 `True`). (c) TrackerNewMem :407을 `.to(storage_device)` 또는 key만 저장(로컬 변수는 :446-461에서 계속 사용). (d) partial 루프에서 `_cache_frame_outputs`(:1274-1279) 직후 `cached_frame_outputs[frame_idx-1]`를 pop하는 request-gated 플래그(`_build_tracker_output` :556-582는 현재 프레임 항목만 읽음).
- **왜**: M2, M3. CoffeeMartini/FlameSteak를 죽인 host-RAM 성장(프레임당 ~7-9 GB)의 유일한 처방.
- **예상 효과**: GPU −13.6/−12/−14.6 GiB(Alexa/Coffee/Flame) seeding 직후; host −24~29 GB seed + 프레임당 성장 ~1 GB로 감소 → FlameSteak 21프레임이 ~35 GB 이내. 출력 bit-identical(제거되는 텐서는 어디서도 읽히지 않음).
- **난이도**: 시간 단위. (d)는 데모의 `propagation_fetch`(:1138)/`remove_object`(:1412-1416)를 깨므로 request-gated, 기본 off.
- **검증**: `runMVSeg.py Blocks --algo OneStageNew --out ..._p1 --overwrite` 후 `diff -rq` 기존 출력과 0 차이; `measure_mem.py`의 walk 라인 `previous_stages_out/obj_id_to_mask`, `mask_inputs_per_obj` 소멸, RSS 프레임당 +0.43 GB 성장 정지; FlameSteak `--memory=90g`에서 완주.

### P2. 패키지 내 window ablation(W=0/1/2/4/6) + 위생 수정 3건 + closure/prefix 트랙 모드 (3/3 합의, 5·5·5)
- **무엇을**: (1) :1253 `range(-4,0)` → `range(-self.cross_view_window, 0)`, `build_tracker_newmem`(build_scsam3.py:494-548) 인자 + `SCSAM3_XVIEW_WINDOW` env; (2) :1254 뒤 `if prev_spatial_idx < 0: continue`; (3) :1262 `_, unselected_cond_outputs1 = ...`; (4) runMVSeg.py:292-294에 `--track-cams closure`: `track_idx = range(0, max(scored_idx)+1)` (m == 카메라 index 유지, None 배관 불필요); (5) `--xview-window N`이 `--out`을 `SegMaskSam3XW{N}`으로. OneStage도 `--track-cams all`로 한 번 돌려 타이밍 열 확보.
- **왜**: E2(패키지 내 W=0이 유일한 올바른 대조군), E5, C1/C2, E3(AlexaMeade/Welder scored idx [0,2,3] → closure면 4세션이라 15-dataset 표 완성 가능). 창이 one-sided이고 lockstep이 오름차순이라 scored 뷰 v는 뷰 0..v에만 의존 — (3) 적용 후 closure는 all과 출력 동일.
- **예상 효과**: 정확도 이득 없음; 대신 노이즈 바닥(XW0 vs OneStage), W 곡선, 공정한 per-view-frame 타이밍, AlexaMeade OneStageNew 완주(38 → ~13 GiB, 945 → 84 view-frames), Welder 553 s → ~60 s. Coffee/Flame은 closure가 15/17세션이라 P1 선행 필요.
- **난이도**: 시간 단위. **심사 이견**: 한 심사자는 5×15 그리드 비용을 들어 W∈{0,4} 전체 후 곡선을 권함.
- **검증**: 아래 실험 계획표.

### P3. 주장을 담을 수 있는 평가 프로토콜 (2/3 top; 4·5·5 / 3·5·5 / 4·5·5)
- **무엇을**: `report_jf.py`에 `--split ref|nonref`, `--bin-by-nb`(min(view_idx, W)), `--area-weighted`(eval_jf.py:180-188에 `gt_area` 추가), `--paired`(36캠 per-camera Δ + sign test/bootstrap), `runMVSegAll.sh:100-110`에 `--common` 상시. `J_inner`도 표에.
- **왜**: E1, E3, E4. 현재 헤드라인은 1/3이 single-view VOS이고 3개 객체가 지배.
- **예상 효과**: 모델 변경 없음. 현 출력에서 예상 판독: nonref Δ +0.005, 면적가중 Δ≈0, nb=4 +0.0047 vs nb=0 −0.0001, 15-dataset OneStage < SAM2 0.0007. 이후 모든 제안의 1차 endpoint를 nb 구간·nonref 행으로.
- **난이도**: 시간 단위, GPU 불필요. gt_area 추가로 캐시된 jf_*.json 재계산(분 단위).
- **검증**: 기존 폴더 재채점이 0.8449/0.8465/0.8497, nonref 0.8052/0.8093/0.8144, Fencing v9 id3 ΔJ +0.419를 재현.

### P4. cross-view gather 재작성: 경계 클리핑, 양측 창, t−1 fallback, clobber 제거, 생성자 인자 (심사 3인이 각기 다른 변형을 상위에; 5·4·5 / 4·4·4 / 4·5·4)
- **무엇을**: :1252-1264와 :1296-1314를 `_select_cross_view_memories(output_dicts, spatial_idx, frame_idx, track_in_reverse)`로 추출. s_pos ∈ [−W..−1, +1..+W], 인덱스 범위 밖은 skip(`ring=True`일 때만 modulo), `non_cond.get(t)` 없으면 `.get(t−1)`(reverse면 t+1) — lockstep에서 상위 뷰의 t−1은 두 번째 프레임부터 항상 존재; `lag` 플래그 기록; :1262-1263 삭제; :1304 `seq_len`을 이웃 feats에서 재계산. 기본값은 one-sided/no-ring으로 두어 기존 숫자 재현.
- **왜**: A1, A2, C1, C2. 복구 메커니즘이 index-0 카메라(reference 6/12 포함)에 닿게 하는 유일한 방법.
- **예상 효과**: Fencing v0 obj 3/4가 v9와 같은 복구를 받으면 v0 J 0.80 → ~0.86, Fencing −0.021 대부분 해소. 이것은 **유추이지 측정이 아님**. 양측 W=4는 memory token +115%(transient만). ring은 LLFF(Blocks/Fencing/Carpark, poses_bounds가 open arc)에서 off, 46캠 돔은 링 여부 미확인이라 데이터셋별 플래그.
- **난이도**: 일 단위(배관 포함). **이견**: 한 심사자는 env-var 변형(P2)과 중복이니 하나만 구현하라고 함 — P2의 (1)-(3)을 먼저, P4는 그 위에.
- **검증**: Fencing/PoznanStreet/Welder/Blocks/Barn/Carpark에 `XV_WINDOW ∈ {none, -4..-1, two2, two4}`; nb=0 카메라의 ΔJ가 0에서 벗어나는지, Fencing v0 obj 3/4 per-frame IoU, idx≥4 카메라 손실 ≤0.005.

### P5. cross-view pass의 seed 품질 검사와 복구 (2/3 top; 4·3·4 / 4·3·4 / 5·3·4)
- **무엇을**: runMVSeg `PropagateAcrossViews` 뒤 `RepairSeeds()`: (view, obj)별 area < max(64 px, 0.05×median)이면 degenerate; donor = ±4 인덱스 내 최대 면적 뷰; spatial 세션에 `add_prompt(frame_index=donor, mask=masks_spatial[donor][obj], obj_id=obj)`로 두 번째 cond frame 추가 후 재전파(implicit start로 C3 회피), 통과하면 교체, 최대 2라운드. 옵션: `Mask/objects_labels.json`의 라벨로 detector 재실행(score ≥0.7, IoU ≥0.1 vs donor mask). SAM 2 러너(runMVSegForSam2.py)에도 동일 적용.
- **왜**: A3. Welder −0.034 전부가 camera_0004의 4/7 px seed.
- **예상 효과**: camera_0004 obj 8/14/12 J 0/0/0.14 → ≥0.8이면 Welder J&F +0.035~0.04, SAM 2 대비 손실 소거. 다른 데이터셋은 트리거 드묾.
- **난이도**: 일 단위. **심사 이견**: 한 심사자는 spatial 세션에서 기존 obj_id 재프롬프트가 새 tracker state를 만들지 걱정(feasibility 2); 다른 둘은 `use_stateless_refinement=False`라 refine 분기로 같은 state에 cond frame이 추가됨을 확인. 구현 전 이 분기(SCSam3VideoInference.py:1855-1870)를 로그로 확인할 것. 면적 검사는 obj 12 같은 "어긋난" seed(2,633 px, IoU 0.18)를 놓침.
- **검증**: 먼저 비용 0 진단 — 기존 `SegMaskSam3OneStageNew/<cam>/0/*.png`에서 degenerate seed 수를 데이터셋별로 집계해 상한 파악. 그 후 Welder/Fencing/Blocks/PoznanStreet/MATF/Barn + 대조 Carpark/Frog.

### P6. full-res non-overlap의 int64 승격 제거 (1/3 top; 4·5·5 / 4·5·5 / 4·5·5)
- **무엇을**: TrackerNewMem :1911-1930, bool 입력·background 0일 때 `pred_masks & (pixel_level_non_overlapping_masks > 0)`로 축약(호출부 :513의 `> 0`는 no-op이 됨); 또는 running argmax(오름차순 strict `>`로 `torch.argmax` 첫 최대 tie-break 재현).
- **왜**: M4. AlexaMeade에서 실제로 raise한 할당.
- **예상 효과**: transient 3.46 → ~0.25 GiB(26 마스크), 68 마스크에서 ~10 → ~0.4 GiB; 출력 bit-identical. 단독으로는 어떤 OOM도 못 살림(resident 38-44 GiB).
- **난이도**: 시간 단위. 검증: 합성 마스크로 `torch.equal` + `max_memory_allocated`; Blocks diff 0.

### P7. spatial predictor 은퇴 + autocast 캐시 정리 + fp16 프레임 (1/3 top; 3·4·5 / 3·4·4 / 4·4·5)
- **무엇을**: `PropagateAcrossViews` 후 `close_session(session_id_statial)` + `predictor_spatial.model.cpu()`/`del`(runMVSeg.py:240의 hasattr 스니핑은 `is_newmem` 플래그로 대체); 각 스텝 후 `torch.clear_autocast_cache()`(bf16 가중치 적재는 decoder.py:71의 autocast-off 영역 때문에 수치가 바뀌므로 금지); io_utils.py:32에 upstream처럼 `.to(float16)`(fp32로 캐스팅되므로 결과 동일). 장기: 두 predictor가 nn.Module 공유.
- **왜**: M6. **예상 효과**: seeding 전 ~6-8 GiB 회수. 난이도 시간 단위.

### P8. reference 규칙을 docstring대로(객체 수) + 도달 불가 ceiling 열 (4·5·5 / 3·5·5 / 4·5·5)
- **무엇을**: runMVSeg.py:102 `len(np.unique(img))-1`, `--reference {count,maxid,<cam>}`(기본 maxid로 재현성 유지), `seed_coverage.json` 출력, `report_jf.py --ceiling`. runMVSegForSam2.py:78-79에도 동일.
- **왜**: A7. **예상 효과**: 달성 가능 J 상한 Blocks +0.128, MATF +0.087, FacePaint +0.061 — 모든 방법에 동일하게 적용되는 **프로토콜 변경**이라 방법 간 순위는 안 바뀜; MATF/Blocks가 "추적 실패"로 읽히는 것을 막음. **이견**: 두 심사자는 "reference가 cross-view memory를 받게 중앙 뷰로 tie-break"라는 동기는 근거가 약하다고 봄(이웃이 있는 reference 카메라도 이득 없음) — count 규칙만 채택.

### P9. 하나의 propagation 계약: 엄격한 request 키, 방향 기록, `[0, N−1]` fetch 휴리스틱 교체, 정확한 프레임 수 (4·3·5 / 3·3·5 / 4·3·5)
- **무엇을**: 두 predictor의 `handle_stream_request`에서 미지의 키는 ValueError, `propagation_direction` 필수; `add_action_history(..., reverse=reverse)`; `parse_action_history_for_propagation`(SCSam3VideoInference.py:1264-1274, NewMem :1350-1361)을 방향별 완료 검사로; `_get_processing_order`(:244-246)를 "정확히 n프레임"으로(tracker 쪽 :777-806의 max+1 규약도 함께, 안 그러면 :1057 assert). 세션 close.
- **왜**: C3, C4, 미종료 generator. **효과**: 벤치마크 숫자는 불변(runMVSeg가 이미 우회), 데모 경로와 P5의 재전파가 안전해짐. 난이도 일 단위, upstream 복사본 2곳+OneStage 폴더 수정.

### P10. seeding 응답 생략 (3·4·5 / 3·5·5 / 3·5·5)
`add_tracker_new_mask`에 `return_output=False`를 뚫어 `_build_tracker_output`/`_postprocess_output`을 건너뛰고 새 마스크만 `cached_frame_outputs[frame_idx][obj_id] = new_mask.cpu()`. spatial 세션의 `AddReferenceMask`는 `out_obj_ids`가 필요하므로 유지. M7 → 120/217/260 s가 수 초로, 출력 동일.

### P11. 뷰당 detector 잔재 축소 (3·3·5 / 3·3·4 / 4·3·5)
`_prepare_backbone_feats` 후 buffer에서 `pred_*` 키 삭제 + pos-enc 공유 + `maskmem_pos_enc` 클론 hoist(:1727-1731)는 안전(mask 경로 gating 필요; text 경로는 buffer를 읽음). prefetch 억제는 `sam3_image.py:763-786`(untracked upstream 체크아웃) 수정이 필요하며, **wrapper에서 pop하면 다음 프레임에 재계산되어 오히려 detector forward가 추가됨**(한 심사자 지적). 안전한 부분집합만 먼저.

### P12. 이웃 토큰 조건화 ablation: gating / tpos 행 / 이웃 obj_ptr (3·4·3 / 2·4·3 / 3·3·3)
:1260에서 `object_score_logits<=0 or eff_iou<=mf_threshold` skip; :1312 행을 {alias(현재), cond row 6, rows 3..5, mean}에서 선택; :1355-1367 뒤에 이웃 `obj_ptr`을 4토큰으로 추가(`num_obj_ptr_tokens` 앞). A4/A5/A6 대상. 심사자 모두 "gating 회귀 증거는 반박됨, tpos 변형은 frozen encoder에 OOD, 이웃 pointer가 가장 유망"에 동의. Fencing/PoznanStreet/Welder/Blocks 4개 데이터셋 그리드로 "이웃 내용을 쓰는가, 위치만 쓰는가"를 판별하는 진단 실험으로 가치.

### P13. 엔지니어링 위생 묶음 (2~3·4~5·5)
CPU-only pytest(request 키, parse_action_history ref=N−1, processing_order 21프레임, N=3 IndexError/wrap/clobber, seeding parity, AST 미정의 이름) + `waitAndRunMVSeg.sh`에 eval + J&F 하한; dead API(C5)·`copy.py`·shadowed 메서드 삭제, MultiGPU 이름 수정 + `gpus_to_use=[current_device]`, 텍스트 프롬프트에 NotImplementedError(C7); window/fill_hole_area/track_cams를 인자화하고 배너에 출력(G4). 숫자에 영향 없음.

### P14. 이벤트 구동 cross-view 재시딩 (4·3·3 / 3·2·3 / 4·3·3)
자기 트랙 score ≤0 k프레임 연속 & 이웃 ≥2개 보유 시, 2-frame pseudo-video로 마스크 전이 후 `add_prompt`로 새 cond frame. A3의 구조적 한계("attention만으로는 고정된 absent 상태를 못 뒤집음")를 mask prompt 채널로 우회하는 유일한 설계. 단 Fencing v0는 하위 이웃이 없어 P4 없이는 트리거 불가, 중간 add_prompt가 새 tracker state를 만드는 문제(:1828-1831)와 `iss[m]=tmp[0]` 정합(:1027-1035)이 미검증. P4·P5 결과를 본 뒤.

### 낮은 우선순위 (심사자 평균 ≤2.5)
- spatial pass memory chain 단축/mf_threshold 상향: reference cond frame은 매 hop에서 항상 attend되고 SAM 2는 같은 chain으로 Welder cam4를 0.94로 seed하므로 전제가 약함. `max_mem_frames` cap으로 frame-0 IoU proxy만 빠르게 볼 가치.
- 양방향/링 pseudo-video seeding: scored Welder 카메라에 효과 0(자인), 링 여부 미기록, hop-decay 전제 반박됨.
- 기하 기반 이웃 선택/homography warp: poses는 3개 데이터셋뿐, frozen encoder에 warped feature는 OOD, 근거리 물체에 평면 가정 부적합. 주 단위 노력 대비 음성 결과 가능성 높음.

---

# 제안하는 실험 계획

전제: P2의 위생 수정 3건(:1254 guard, :1262 rename, W 인자) 적용, `--track-cams closure` 추가, `report_jf.py --common` 상시, P3의 `--split nonref --bin-by-nb --paired`. 모든 행은 `runMVSeg.py <ds> --algo <A> --track-cams <T> --xview-window <W> --out <OUT> --overwrite` → `eval_jf.py <ds...> --methods <OUTs> --out jf_abl.json` → `report_jf.py --raw jf_raw.json jf_abl.json --methods ... --common`.

| # | 구성 | algo / track-cams / W | out 폴더 | 데이터셋 | 분리하려는 것 | 성공 판독 |
|---|---|---|---|---|---|---|
| 0 | SAM 2 baseline | runMVSegForSam2.py | SegMaskNew1 (기존) | 15 | 외부 기준 | — |
| 1 | 기존 OneStage | OneStage / written / — | SegMaskSam3OneStage (기존) | 15 | 커버리지 열 | — |
| 2 | OneStage, 세션 수 동일 | OneStage / closure / — | SegMaskSam3OneStageC | 15 | 세션 수 vs 창 (타이밍) | 1과 PNG 동일; per-view-frame 시간 |
| 3 | **패키지 내 대조군** | OneStageNew / closure / 0 | SegMaskSam3XW0 | 15 | 노이즈 바닥, 패키지 효과 | 36캠 전부 \|ΔJ\| vs 행 2 < 0.001 (P1 적용 후) |
| 4 | 현재 창 (위생 수정) | OneStageNew / closure / 4 | SegMaskSam3XW4 | 15 | **cross-view memory의 순효과** = 행4 − 행3 | nb 구간별 Δ, nonref Δ, paired test; AlexaMeade/Welder 처음으로 완주 |
| 5 | 창 크기 곡선 | OneStageNew / closure / 1, 2, 6 | SegMaskSam3XW{1,2,6} | Fencing, PoznanStreet, Welder, Blocks, Barn, Carpark (+전체는 여유 시) | W 감도 | 단조 여부; W=6은 tpos 7행 내 |
| 6 | closure = all 검증 | OneStageNew / all / 4 | SegMaskSam3XW4all | Blocks, Fencing, Welder | closure 정확성 | 행 4와 `diff -rq` 0 (P1·P6·P7 후 Welder 46세션도 fit) |
| 7 | 양측 창 (P4) | OneStageNew / closure / two2, two4, (ring on Welder/Dog만) | SegMaskSam3XV_two2/4 | 행 5와 동일 6개 | index-0/reference 카메라 복구 | Fencing v0 J1-20 0.79 → ≥0.85; nb=0 구간 Δ ≠ 0 |
| 8 | seed 복구 (P5) | OneStageNew / closure / 4, `--repair-seeds` | SegMaskSeedRepair | Welder, Blocks, MATF, PoznanStreet, Painter + 대조 Carpark, Frog | spatial pass 실패의 기여 | Welder cam4 obj 8/14 J ≥0.8; 대조군 ±0.001 |
| 9 | reference 규칙 (P8) | 세 방법 모두 `--reference count` | *_refcount | Blocks, MATF, PoznanStreet, Painter, FacePaint | 도달 불가 zero의 기여 | 구조적 zero 203→63 등 재현; 세 방법 J ≥+0.03 동반 상승, 순위 불변 |
| 10 | 이웃 토큰 조건화 (P12) | OneStageNew / closure / 4, gate/tpos/ptr 그리드 | SegMaskSam3X_<tag> | Fencing, PoznanStreet, Welder, Blocks | 내용 vs 위치 코드 | 어떤 변형이든 nonref에서 +0.005 초과 & reference 무손실 |
| 11 | 설정 매칭 (E6) | SAM 2 `apply_postprocessing=False`, `non_overlap_masks=True` | SegMaskSam2Matched | 12 | F +0.006의 후처리 몫 | SAM2 vs OneStage F 차이 재계산 |

보고할 표: (a) 15-dataset `--common` 헤드라인(행 0/1/3/4), (b) nonref-only 및 nb 구간별(행 3 vs 4 vs 7), (c) per-object 산점(Fencing v0/v9, PoznanStreet v8, Welder cam4)으로 "어디서 이득이 나는가", (d) per-view-frame 타이밍(행 2 vs 4), (e) ceiling 열(행 9). 순서는 표 번호대로: 행 3-4가 나오기 전에는 어떤 정확도 제안도 판정할 수 없습니다.

---

# 결론

이 코드는 "이웃 카메라의 memory를 SAM 3 memory attention에 이어 붙인다"는 하나의 아이디어를 SAM 2 버전에서 그대로 옮긴 것이며, 아이디어 자체는 Fencing v9·PoznanStreet v8에서 죽은 트랙을 되살리는 실제 사례로 존재를 증명했지만, 현재 구현은 창이 one-sided·리스트 순서 기반이고 index-0(reference 6/12 포함) 카메라에는 아예 닿지 않으며, 이웃 토큰을 "같은 카메라의 이전 프레임"으로 위장해 넣기 때문에 효과가 작고(+0.003, 3-4개 작은 객체) 카메라 순서에 의존합니다. SAM 2 대비 두 손실(Welder, Fencing)은 각각 cross-view seed 실패와 이웃 없는 reference 카메라의 단일 뷰 drift로, 지금의 temporal cross-view memory가 구조적으로 손댈 수 없는 지점에 있습니다. OOM과 속도 문제는 알고리즘이 아니라 상태 관리입니다 — cross-view prompt는 수십 MB인 반면, 뷰·객체당 두 번 고정되는 full-res seed 마스크, 영구 축적되는 host-side 마스크, 두 개의 모델과 종료되지 않는 autocast 캐시, int64로 승격되는 후처리가 46 GB와 host 사망을 만들며, 이들 대부분은 출력을 바꾸지 않고 제거할 수 있습니다. 코드로서는 upstream 1,700줄 파일을 포크-편집한 복사본 6벌, 테스트 0개, 하드코딩된 창, 그리고 working copy와 Docker 레이어에 노출된 HF 토큰(즉시 폐기 필요)이 가장 큰 부채입니다. 실험 설계로서는 OneStage-vs-OneStageNew가 패키지·세션 수·numerics가 모두 다른 비교라 창의 효과를 분리하지 못하며, 12-dataset 부분집합은 15-dataset 전체와 부호가 다릅니다. **가장 가치 있는 다음 한 걸음은 P1(마스크 고정 해제)과 P2(위생 수정 3건 + W 인자 + closure 모드)를 적용한 뒤 실험 계획의 행 3·4(패키지 내 W=0 vs W=4, 15개 데이터셋, `--common`, nonref/nb 구간 분리)를 돌리는 것**입니다 — 이것이 나오기 전까지는 이 방법이 무엇을 기여하는지 숫자로 말할 수 없고, 나온 뒤에는 P4(양측 창)와 P5(seed 복구)가 각각 Fencing과 Welder를 겨냥한 명확한 다음 실험이 됩니다.