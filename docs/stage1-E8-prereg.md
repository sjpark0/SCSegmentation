# S1-E8 사전 등록 — 새 1단계(GeoSet-2 계열 세 구성) 시드를 본류 2단계에 통과시켜 MUVOD에서 실측

> 2026-10-12. 층: **확정(측정·채택 판정)**. 계기: 논문 제출(태그 `tcsvt-submitted`) 뒤 1단계(키프레임 초기 마스크
> 생성)를 별도 저장소 `SCSegmentation2D`(브랜치 `geomem`, HEAD `b089577`)에서 새로 만들었고, 사용자가 그것을 본류
> 1단계로 바꿔 MUVOD에서 돌려 기존 방법과 비교하라고 요청했습니다. 사용자는 2D 저장소가 최선으로 꼽은 **세 구성 전부**를
> 보기를 원했습니다. MVSeed README 규칙 R4·R5: 다른 곳에서 좋았던 것은 **동기**일 뿐이고, 본류 2단계까지 붙여 c_ini·MUVOD
> 규약으로 다시 잰 것만 인용합니다.
> 이 문서는 시드 재생성과 2단계 GPU 실행 **전에** 커밋됩니다. §8 이후는 실행 뒤에 채웁니다.

## 1. 새 1단계 — 세 구성 (팔)

공통: 입력은 c_ini의 키프레임 정답 라벨맵 + 키프레임의 **모든 카메라 이미지** + π³ 기하 캐시(`Data/<scene>/Cache/pi3/<cam>.npz`),
텍스트 없음. 뷰를 비디오로 보는 사슬 전파 대신 c_ini 기억 하나로 각 뷰를 직접 디코딩(star)하고, 기억 어텐션에 π³ 3D 거리 RBF
바이어스를 더해 그 어텐션만 미세조정(백본·마스크 디코더 동결). 2라운드에서 π³ 카메라 중심 거리 최근접 3대의 1라운드 예측을 기억으로
추가하고 객체별 게이트로 거름. SAM 3는 본류와 같은 패키지·가중치(상류 수정 없음).

| 팔 | 태그 | 2D 구성 | 2D 저장소 키프레임 J&F / F (비기준 주석 664쌍) | 2D 저장소의 장면별 안전 | 1단계 시간(17장면) |
|---|---|---|---|---|---|
| **A** | `geoset_rep` | `--arm geoset_pred_ft --gate reproj` (재투영 게이트) | 0.914 / 0.951 | 통과 | 376 s |
| B | `geoset_abs` | `--arm geoset_pred_ft --gate absent` (부재 게이트) | 0.915 / 0.951 | 실패 (Welder −0.02) | 366 s |
| C | `geoalign_rep` | `--arm geoalign_ft --gate reproj` (정렬 기억 뷰, 표적 자기 제외 = `_x`) | 0.915 / 0.953 | 통과 | 729 s |

현행 1단계 계열(`sam3_prop`)은 같은 채점에서 0.890입니다. 2D 저장소는 A를 기본 추천으로 꼽았습니다.

**학습 데이터에 관한 단서(원고에 반드시 적음).** 어텐션 미세조정에 MUVOD 정답(각 장면 키프레임부터 21프레임, stride 2)을 썼습니다.
체크포인트는 **장면 단위 5-fold**(`geomem/folds.json`)이고 채점 장면 자신은 그 장면의 fold에서 빠지므로 **같은 장면의 누출은 없습니다**.
그러나 (a) 같은 데이터셋의 다른 장면 정답으로 학습한 모델이고, (b) MUVOD XMem 기준선과 우리 제출본은 MUVOD로 아무것도 학습하지 않았으므로,
비교표의 이 열들은 "같은 조건"이 아닙니다.

## 2. 시드 집합 `MVSeed_<태그>` (`Data/MVSeg/<scene>/MVSeed_<태그>/`, MVSeed README R2)

1. **재생성.** 2D 저장소 HEAD `b089577`(작업 트리 깨끗한 상태)에서 세 팔을 **새 출력 폴더**(`*_e8`)로 다시 만듭니다. 기존 출력 중 A·B는
   `5efa3fd`, C는 `6f739ca`에서 만들어졌고 그 뒤 추론 코드(`methods/geomem/run.py` 등)가 바뀌었기 때문입니다. 기존 2D 출력은 건드리지 않습니다.
   GPU는 한 번에 하나(2D 세션 작업과 동시 실행 금지).
   - A `scripts/run_geomem.sh all --arm geoset_pred_ft --gate reproj --out geoset_pred_ft_rep_e8 --r1-out geoset_pred_ft_r1_e8`
   - B `scripts/run_geomem.sh all --arm geoset_pred_ft --gate absent --out geoset_pred_ft_abs_e8 --r1-out geoset_pred_ft_r1_e8b`
   - C `scripts/run_geomem.sh all --arm geoalign_ft --gate reproj --out geoalign_ft_x_rep_e8 --r1-out geoalign_ft_x_r1_e8`
   - 감시값 Z0: 재생성 PNG와 기존 출력(`geoset_pred_ft_rep`, `geoset_pred_ft_abs`, `geoalign_ft_x_rep`)의 바이트 동일 비율(보고만; 시드는 **재생성본**).
2. **변환** `MVSeed/build_geoset_seeds.py --tag <태그> --out <출력>`: 장면의 **모든 카메라**(`Video/` 폴더) × **basic 객체**(`seed_ids_c_ini`)마다
   - 2D 출력에 `<cam>/<obj>.png`가 있으면 → 형식 검사(모드 L, 장면 해상도, 값 ⊆ {0,255}) 후 **바이트 그대로 복사**
   - 없으면 → **빈 PNG**(면적 0 = 새 1단계가 "이 뷰에 없음"이라고 판단). G0로 되돌리지 않습니다 — 그러면 처치가 섞입니다.
   - 레이아웃 `<cam>/<start_frame>/<obj>.png`, `SEED_MANIFEST.json`(`written_views` = 모든 카메라, `dataset`, `start_frame`, `c_ini`,
     출처 저장소·HEAD·출력 폴더·`RUN.json` sha256·fold·체크포인트 sha256).
   - 사전 점검(10-09 출력으로 변환만 시험, 폴더는 지움): 17장면 8 243 PNG 중 6 436 복사·1 807 빈 PNG, 형식 위반 0.
3. 비주석 카메라 시드도 교체합니다. 헤드라인 구성은 그것을 교차 시점 메모리로 읽으므로, 이것이 "1단계를 통째로 바꾼" 조건입니다.

## 3. 실행 (17장면, `SCSam3/runMVOptThree.sh`, 기존 플래그만; 본류 코드 변경 없음)

공통 env — 대조군: `XVIEW=0 XREF=muvod F0SEED=1`, 헤드라인: `XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod F0SEED=1 CROP=10000 CROPGATE=v2`.

| # | 구성 | 시드 | 출력 폴더 | 기저 (현행 1단계 G0) |
|---|---|---|---|---|
| A2 | 헤드라인 | `MVSeed_geoset_rep` | `SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2` | `SegMaskSam3XW1CGPS4MFsCs10kG2` (91.015, 제출본) |
| B2 | 헤드라인 | `MVSeed_geoset_abs` | `SegMaskSam3XW1CGPS4MSdgeoset_absFsCs10kG2` | 〃 |
| C2 | 헤드라인 | `MVSeed_geoalign_rep` | `SegMaskSam3XW1CGPS4MSdgeoalign_repFsCs10kG2` | 〃 |
| A1 | 대조군 | `MVSeed_geoset_rep` | `SegMaskSam3XW0MSdgeoset_repFs` | `SegMaskSam3XW0MFs` (89.549) |
| B1 | 대조군 | `MVSeed_geoset_abs` | `SegMaskSam3XW0MSdgeoset_absFs` | 〃 |
| C1 | 대조군 | `MVSeed_geoalign_rep` | `SegMaskSam3XW0MSdgeoalign_repFs` | 〃 |

순서 A2 → B2 → C2 → A1 → B1 → C1(판정 대상인 헤드라인 먼저). 예상 GPU 약 7시간. 채점 `eval/eval_jf.py --methods <6개> --out jf_e8.json`(컨테이너),
판정 `MVSeed/e8_judge.py` → `MVSeed/runs/e8_judge.json`(E7 판정과 같은 집계: 객체 → 카메라 → 장면 평균 J&F³, 장면 클러스터 부트스트랩 seed 0·10 000회).

## 4. 예측 (실행 전)

| # | 예측 | 근거 |
|---|---|---|
| P1 | A1 basic Δ = **+0.3 ~ +1.5** | 키프레임 시드 격차의 약 22 %를 메움(2D 저장소 추정 +0.3~+1.0); E7 정답 시드 +4.38, E6 오라클 환산 2.7배 낙관 보정 |
| P2 | A2 basic Δ = **+0.2 ~ +1.5** | 크롭·교차 시점 메모리가 일부를 이미 회수(E7에서 헤드라인 Δ ≈ 대조군 Δ); 비주석 이웃도 바뀌어 증폭·확산 둘 다 가능 |
| P3 | 세 팔의 헤드라인 Δ 차이는 각각 0.3점 이내 | 키프레임 J&F 차이가 0.001(0.914 대 0.915) |
| P4 | B2의 Welder Δ < A2의 Welder Δ | 키프레임에서 부재 게이트가 Welder를 깎음 |
| P5 | A1 F0(41쌍) 회복 > 파괴, N0(623쌍) 파괴 ≤ 3 | 키프레임에서 실패 33/45 회복, 정상 2/619 파괴 |
| P6 | 여섯 실행 모두 프레임 0 PNG = 시드 PNG 바이트 동일(시드 PNG가 있는 채점 쌍 전부) | `--frame0-seed` 배관 |

## 5. 채택 규칙 (실행 전 고정)

**헤드라인 실행(A2·B2·C2)이 판정 대상입니다.** 팔마다 다음 셋을 모두 만족하면 "통과"입니다.

1. basic Δ(대 `SegMaskSam3XW1CGPS4MFsCs10kG2`) > 0 이고 **95% CI 하한 > 0**.
2. **방어선**: 어느 장면도 basic Δ < −1.0 이 아님(P5 기각 때와 같은 선).
3. complete Δ ≥ 0(점 추정).

- **선택**: 통과한 팔이 둘 이상이면 basic 점 추정이 가장 높은 팔을 새 헤드라인으로 채택합니다. 동률(차이 < 0.05점)이면 A > C > B
  순(2D 저장소 추천 순: 안전 통과·싼 것 먼저).
- 통과한 팔이 없으면 헤드라인을 바꾸지 않고 "효과 없음"으로 보고합니다. 다른 2D 구성을 추가로 돌려 다시 판정하지 않습니다(사후 선택 금지) —
  더 돌리려면 새 등록이 필요합니다.
- 대조군 실행(A1·B1·C1)은 진단입니다(카메라 독립 추적에서 1단계만의 효과). 대조군만 통과 기준을 넘고 헤드라인이 못 넘으면 "1단계 개선이
  교차 시점·크롭과 중복"으로 읽습니다.
- 다중비교: 판정 비교 3개(+진단 3개)에 보정은 하지 않습니다. 민감도로 **98.33 % CI**(본페로니 3)도 함께 보고하고, 채택 팔의 그 하한이
  0 이하이면 원고에 그렇게 적습니다.

## 6. 보고할 비교표

basic·complete J&F³, 17장면: MUVOD XMem(논문 Table III·IV, 79.35 / 75.63) · 제출본 헤드라인(91.015 / 86.399) · 대조군 G0(89.549 / 84.987) ·
**A1·B1·C1·A2·B2·C2** · 정답 시드 상한(E7 R1 93.933 · R2 95.273). 장면별 Δ, F0/N0 쌍 분해, 벤치마크 가중 집단 기여(`e7_contrib.py` 방식).

## 7. 감시값

- Z0 재생성 대 기존 2D 출력 바이트 동일 비율(팔별).
- Z1 프레임 0 PNG = 시드 PNG 바이트 동일(P6).
- Z2 각 실행 `MANIFEST.json`의 `provenance.seeds_from.png_digest` = 시드 폴더 digest(17/17).

## 8. 이력

| 일자 | 내용 |
|---|---|
| 2026-10-12 | 등록 (시드 재생성·GPU 실행 전 커밋). 초안은 A 하나였으나 사용자 요청으로 실행 전에 B·C를 추가 |
| 2026-10-12 | Z0: 재생성 세 팔이 기존 2D 출력과 PNG 바이트 동일(6 436 · 6 471 · 6 442), 키프레임 664쌍 J 전부 동일(`MVSeed/runs/e8_regen_check.json`) |
| 2026-10-12 | 시드 변환 수정(2단계 실행 전): C의 AlexaMeadeFacePaint `camera_0042`는 모든 객체 예측이 비어 2D 작성기가 폴더를 만들지 않음(폴더는 비어 있지 않은 예측이 있을 때만 생김). 변환기가 "카메라 폴더 집합 = Video/ 카메라"를 요구해 멈췄으므로, 출처 폴더 ⊆ Video/ 카메라로 완화하고 폴더 없는 카메라는 §2-2 규칙대로 전부 빈 PNG. 매니페스트에 `cams_without_folder` 기록. 규칙 변경 없음 |
