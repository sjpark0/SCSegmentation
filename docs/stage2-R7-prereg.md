# S2-R7 사전 등록 — S2-R6의 교체 결정을 기준 카메라 추적기 기억에도 넣기 (목록 3번, 실시간 한 팔 `Rm`)

> 2026-10-16. **팔 출력이 생기기 전 커밋**(S2-R6H 등록과 같은 커밋). 사용자 지시 "3번과 4번으로 넘어가자"의 3번 "S2-R6 확장: 유지한 마스크를 추적기 기억에도 넣기".
> 설계는 워크플로 wf_32ecd2d7(지도 3·설계 3·심사 2·종합 1), 근거 `MVSeed/refloss/s2r6x/`(GPU 탐침은 MUVOD 정답을 읽지 않음).
> **2단계에는 MUVOD 안에 개발 집합이 없습니다**(정답이 시험 구간 21프레임뿐). 그래서 팔 하나, 새 상수 없음, 한 번 실행, E8 §5 규칙, 멈춤 규칙(§8)으로 등록합니다. 기대는 "채택 안 됨"(P ≈ 0.25)이며, 이 실행의 가치는 3번을 추론이 아니라 측정으로 닫는 데 있습니다.

## 1. 질문

고정한 S2-R6 규칙이 교체하는 프레임에서, 프레임 0 프롬프트 마스크를 출력 PNG뿐 아니라 c_ini 추적기 기억에도 쓰면 MUVOD 2단계 J&F가 바뀌는가.

## 2. 알려진 사실 (등록 전)

- 코드 경로: c_ini 객체마다 SAM 3 추적기 상태가 따로 있고, 프레임 t의 기억은 디코딩 직후 생성기 안에서 부드러운 디코더 로짓(sigmoid × 20 − 10)으로 인코딩됩니다(크롭 패스 전, 객체 간 겹침 처리 없음). 세션은 프레임마다 카메라 순서로 함께 진행하고(c_ini를 먼저 다 돌지 않음), 모드 C에서 세션 r+1은 같은 프레임 t의, r−1은 t−1의 c_ini 항목을 **같은 객체로** 읽습니다.
- 탐침(정답 읽지 않음, Blocks·CoffeeMartini): 저장된 마스크를 다시 인코딩하면 기억이 비트 단위로 같음(320/320, 1 000/1 000); 주입 시 기억 변화 중앙값 2.1 %(p90 11 %); 출력 PNG XOR/합집합 0.03–0.51 %, 바뀐 PNG의 IoU 중앙값 0.995–0.9997; 주입 비용 객체·프레임당 약 2 ms; 끄면 헤드라인과 바이트 동일(Blocks 951/951, 두 번).
- 정답을 쓴 상한(기존 출력만): c_ini에서 아직 교체되지 않은 손실 0.343(천장), 현실적으로 +0.01 ~ +0.04; 다른 카메라 약 0.005 이하(c_ini에서 거리 1 / 2 / ≥ 3인 채점 시점이 2 / 6 / 26개, 교차 시점 기억을 끄고 켜도 거리 1 쌍이 0.0008만 움직임).
- 헤드라인 H의 교체 4 580 중 1 399가 c_ini 작은 물체 크롭 대상(유지 1 249, 되돌림 150)이며, 이들의 오프라인 결정은 크롭 마스크로 내린 것입니다.

## 3. 팔과 폴더

| 팔 | 폴더 | 내용 |
|---|---|---|
| 기저 H | `SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs` | E14 헤드라인(basic 92.766 / complete 88.086, `Data/MVSeg/jf_e14.json`, docker sha256:9064e3dc8548…). **다시 돌리지 않음**(결정성은 G2·Z5로 확인) |
| 팔 M | `SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2RmRs` | 헤드라인 명령 + `--ref-snap-mem`(스윕 `REFSNAPMEM=1`). 접미사 `Rm`은 `Rs` 바로 앞(`Rs`는 맨 끝이라는 규약) |

**돌리지 않음으로 지금 선언**(MUVOD에서 돌리려면 새 등록): 헤드라인 교체 목록 재생(sha256 55a68c32…a8a4f6; 두 번 돌아야 하고 크롭 대상 결정이 인과적이지 않음), 크롭 대상 주입, obj_ptr 교체(M2; 포인터 코사인 0.33 대 연속 프레임 0.998, 분포 밖), 조건 프레임 삽입(M3; 프레임 0 정답 조건 프레임이 4번 교체 뒤 주의에서 빠짐), 인코딩 전 훅(겹침 처리 전 결정), 그림자 팔(자기 채널을 분리하지 못하고 시험 GPU만 두 배).

## 4. 규칙 (새 상수 없음)

- **T** = c_ini 크롭 대상: 0 < area(GT_start == o) < 10 000 px(`--crop-small` 값)인 c_ini 객체, `crop_track`의 판정과 같음(CropSmall이 프레임 0 프롬프트에서 만드는 집합). 크롭 패스가 꺼져 있으면 T = ∅.
- **언제**: 함께 진행하는 쓰기 루프에서 c_ini 세션(r = reference_view_index)이 프레임 t > start를 내놓고 그 PNG를 쓴 직후, 세션 r+1을 진행하기 전.
- **누구**: T에 없고 프레임 start PNG m0가 비어 있지 않은 모든 c_ini 객체 o.
- **결정**: snap = `ref_snap.decide(m0, mask_t, I_start, I_t)`(TAU 0.8, DELTA 4, DILATE 31). mask_t = 방금 쓴 프레임 t의 마스크, I = `Video/<c_ini>/%06d.jpg` 회색조를 `snap_in_place`와 똑같이 읽고 크기 조정(INTER_AREA).
- **교체하면**: o의 추적기 상태에서 프레임 t의 캐시된 영상 특징으로 `mf = tracker._encode_new_memory(…)`; pred_masks_high_res = bilinear(bilinear_antialias(m0 → 288×288) × 20 − 10 → 1008×1008), align_corners False(프레임 0 프롬프트가 가는 길), object_score_logits = 저장된 항목의 것, is_mask_from_pts = True. 그리고 `non_cond_frame_outputs[t]['maskmem_features'].copy_(mf)` — 저장 dtype·장치 그대로, **제자리에서**(객체별 조각이 저장소를 공유).
- **그대로**: 항목의 obj_ptr, maskmem_pos_enc, object_score_logits, iou_score, eff_iou_score, pred_masks; 모든 조건 항목과 다른 세션; 크롭 패스; `--ref-snap`(크롭 패스 뒤 `snap_in_place`는 그대로 돌아 PNG 규칙 불변).
- 결과: T 밖 객체에서 기억 결정과 최종 PNG 결정은 같은 입력의 같은 함수 → 주입 집합 = 이 실행 자신의 PNG 교체 집합 − T, 정확히.

## 5. 거부

`--ref-snap` 없음, `--ref-cam`이 muvod가 아님, `--algo`가 MVOpt가 아님, `--xview-mode E`/두 번 패스, GPU 둘 이상, `RmRs`가 없는 명시적 `--out` 중 하나라도 있으면 거부합니다.

## 6. 스윕 전 관문 (정답 없음)

연기 시험 출력은 별도 `--data-root` 아래에 두고 채점하지 않으며 끝나면 지웁니다.
- **G0** 이 문서가 커밋됨.
- **G1** 호스트·컨테이너 테스트 통과.
- **G2** 플래그를 끈 Blocks·Fencing 재실행의 PNG content_digest = H.
- **G3** Dog(c_ini 크롭 대상 0)에서 팔 M이 탐침(`MVSeed/refloss/s2r6x/mem/probe_mem.py`, PROBE_MODE=inject_mem)과 content_digest가 같고, 주입 (t, 객체) 목록이 같고, 특징 캐시 누락 0.
- 관문이 실패하면 고치고 이 문서 §13에 정오표를 적은 뒤 스윕합니다.

## 7. 감시값 (정답 없음; 17장면, 채점 전 `MVSeed/s2r7_check.py`)

실패하면 판정하지 않습니다: 고치고 정오표를 적고 실패한 장면을 다시 돌립니다(코드가 바뀌면 전 장면).
- **Z1** 모든 카메라의 프레임 start PNG가 H와 바이트 동일; crop_small의 대상·유지·이유·면적 동일, 유지된 크롭 PNG 모두 바이트 동일; T = H의 c_ini 크롭 대상.
- **Z2** (a) 주입 = 이 실행의 ref_snap.snapped_files − T, 정확히; (b) 주입 ⊆ H의 비대상 교체 3 181(δ는 영상과 m0에만 달렸고 H에서 δ를 통과한 비대상 프레임은 모두 교체됨); (c) 프레임 start+1에서 주입 = H의 그 프레임 비대상 교체, 합 210(AME 18, AMFP 11, Barn 14, Blocks 11, Breakfast 10, CBA 7, Carpark 9, CoffeeMartini 27, Dog 7, Fencing 4, FlameSteak 36, Frog 0, MATF 13, MartialArts 11, Painter 12, PoznanStreet 8, Welder 12); (d) 주입한 항목은 모두 non_cond[t]에 있었고 object_score_logits > 0; (e) 캐시 누락 0.
- **Z3** 교체된 c_ini PNG는 모두 자기 프레임 start PNG와 같음; MANIFEST에 규칙 (0.8, 4, 31).
- **Z4**(원뿔) c_ini 세션 위치 r, f1 = start+1일 때 세션 r+d의 PNG나 view_areas가 H와 처음 달라지는 프레임이 d ≥ 1이면 ≥ f1, d = 0이면 ≥ f1+1, d ≤ −1이면 ≥ f1+|d|(모드 C: v−1은 t, v+1은 t−1을 읽음).
- **Z5** Frog(주입 0)가 H와 바이트 동일.
- **Z6** docker 이미지 id와 demoSCSam3MVOpt source_digest = H; argv = H + `--ref-snap-mem`; git HEAD 기록.

## 8. 채택 (E8 §5)과 멈춤 규칙

`MVSeed/s2r7_judge.py`(e7_judge의 load / rows / scene_scores / paired, e14_judge --stage 2와 같은 방식), 반올림 전 값:
- 17장면이 basic·complete 모두 채점됨; Z1–Z6 통과;
- basic Δ(M − H) > 0 그리고 95 % 장면 부트스트랩 CI 하한 > 0(10 000회, Random(0));
- 어떤 장면도 basic Δ < −1.0 아님; complete Δ ≥ 0.

모두 만족하면 **채택**, 아니면 "채택 안 함: S2-R6 결정을 기억에 넣어도 측정 가능한 이득 없음"(CI 상한 < 0이면 "손해"). 헤드라인은 그대로.
**멈춤 규칙**: 다른 기억 변형(재생, 크롭 대상, M2, M3, 훅, 다른 인코딩)은 새 등록 없이 MUVOD에서 돌리지 않으며, 그 등록은 같은 시험 집합을 두 번째로 보는 것임을 밝히고 97.5 % CI로 판정합니다.

## 9. 채택 시 공개

헤드라인은 `…RmRs`가 되고, 언급할 때마다 다음을 붙입니다: "Rs는 시험 정답으로 설계한 후처리(LOSO +0.387, PNG 교체에만 해당). Rm은 같은 실시간 결정을 c_ini 추적기 기억에도 넣는 것으로, 2단계 개발 집합 없이 같은 17장면에서 사전 등록한 한 번의 실행: Δ x [CI]. 다른 카메라로 번지는 몫은 LOSO가 덮지 않음." 한 번 패스 `…Rs` 수치도 함께 보고합니다.

## 10. 예측 (관문 아님)

| # | 예측 |
|---|---|
| P1 | 주입 수 ∈ [3 150, 3 181](3 181이 상한) |
| P2 | 이 실행 자신의 Rs 교체 수 ∈ [4 560, 4 600](H 4 580) |
| P3 | 주입 항목의 상대 기억 변화 중앙값 1–4 %, p90 ≤ 15 % |
| P4 | 채점 카메라·장면별 PNG XOR/합집합 ≤ 1 %; 바뀐 PNG의 IoU 중앙값 ≥ 0.99; c_ini PNG는 start+2부터만 바뀜 |
| P5 | basic Δ ∈ [−0.03, +0.04](80 %), 점 추정 +0.01; c_ini 몫 [−0.02, +0.03], 다른 카메라 [−0.01, +0.01]; \|complete Δ − basic Δ\| ≤ 0.03; 최악 장면 ≥ −0.2 |
| P6 | P(채택) ≈ 0.25 — 검정력보다 부호가 불확실해서(아주 작지만 고른 효과는 부트스트랩을 통과할 수 있음) |
| P7 | 장면당 시간 ≤ 1.05 × H |

## 11. 서술만

c_ini와 다른 카메라 기여, 다른 카메라의 c_ini 거리별 기여; c_ini 프레임 등급(양쪽 모두 교체 = Δ 정확히 0, 새로 교체·잃은 교체, 첫 주입 뒤 교체 안 됨, 주입 없는 객체, T 객체) × 프레임 0 복사 등급(≥ 0.98 / 거의 정지 / 움직임); 카메라별 PNG XOR/합집합; 기억 변화 통계; 장면별 시간.

## 12. 구현 (MVOpt 패키지와 sam3는 고치지 않음 → source_digest 불변)

1. `SCSam3/ref_snap.py`(torch 없음): `read_grey`, `region`을 분리해 `snap_in_place`·`decide`가 그대로 쓰게(바이트 불변, `tests/test_pipeline_inprocess.py`의 고정 테스트 통과); `OnlineSnap(out_dir, video_dir, cam, start, exclude, np, cv2)`가 첫 t > start에서 프레임 start PNG와 I_start를 읽고 객체별 팽창 영역을 한 번 계산, `decide_frame(t, {obj: mask_t})`는 `decide()`와 결과가 같아야 함; :11의 존재하지 않는 `tests/test_ref_snap.py` 언급 바로잡기.
2. `SCSam3/runMVSeg.py`: `--ref-snap-mem`; `resolve_run`의 거부(§5)와 `Rm` 접미사; 쓰기 루프 전에 T와 `OnlineSnap`; c_ini PNG를 쓴 직후 결정 → `InjectRefMemory`(탐침과 같은 인코딩, 제자리 복사, 상대 변화 기록); CropSmall 뒤 T가 c_ini 크롭 대상과 다르면 중단; MANIFEST에 `ref_snap_mem`(끄면 null).
3. `SCSam3/runMVOptThree.sh`: `${REFSNAPMEM:+--ref-snap-mem}`.
4. 테스트: 호스트(접미사, 거부, 끈 경우 resolve_run 불변, `OnlineSnap` = `snap_in_place`, T 계산), 컨테이너(제자리 기록, obj_ptr·점수·조건 항목 불변).
5. `MVSeed/s2r7_check.py`(Z1–Z6), `MVSeed/s2r7_judge.py`.

## 13. 이력

| 일자 | 내용 |
|---|---|
| 2026-10-16 | 등록(팔 출력 전, S2-R6H와 같은 커밋). 설계 워크플로 wf_32ecd2d7, 근거 `MVSeed/refloss/s2r6x/` |
