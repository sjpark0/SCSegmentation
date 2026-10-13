# MVSeg2D — 기준 프레임 다중시점 객체분리 방법론 비교 벤치마크 설계

> 2026-10-06. 대화에서 절별로 합의한 설계를 그대로 적은 문서입니다. 구현 계획은 이 문서를 근거로 따로 씁니다.

## 0. 한 문장

MVSeg 17장면의 기준 프레임에서 **기준 카메라(c_ini) 정답 마스크 1장을 고정**하고, 나머지 N−1 뷰의
객체별 마스크를 추출하는 과제 하나에 대해 **서로 다른 가족의 방법들을 같은 입력·같은 출력 형식·같은
채점기로 비교**한다. SAM 계열에 묶이지 않는다.

## 1. 배경과 동기

이전 프로젝트(`../SCSegmentation`)는 다중시점 **비디오** 객체분리를 다뤘고, 그 2단계 파이프라인에서
1단계(기준 카메라 마스크 → 같은 프레임의 다른 뷰, "시드")가 결과를 거의 결정한다는 것이 측정됐다.

- 시드 J와 이후 21프레임 평균 J의 관계: 기울기 0.964, R² 0.933 (`docs/stage1-headroom.md`)
- 주석 카메라 3대의 시드를 정답으로 바꾸면 MUVOD basic J&F +4.4 (`docs/stage1-E7.md`)
- 실패는 뷰 순서가 아니라 **객체 크기와 시점차**를 따라가고, 이득의 절반 이상은 실패 쌍이 아니라
  정상 쌍의 작은 경계 오차에서 온다
- 기하 투영 계열은 MVSeed에서 세 번 관문 미달: COLMAP+MVS 깊이(E4a, 구멍), π³ 포인트맵(E4b, 518 px
  작업 해상도에서 작은 객체 소실), 3DGS 렌더 깊이(E4c, 자신 있게 틀린 깊이)
- 반구 어안 4장면(AlexaMeadeExhibit, AlexaMeadeFacePaint, Dog, Welder)은 원본이 어안이라 핀홀 기하가
  성립하지 않음. COLMAP은 PoznanStreet·Painter·Frog에서 실패

이 프로젝트는 그 1단계 문제만 떼어내어 **방법론 비교**를 목적으로 한다. 이전 규약(보정 없음·학습 없음·
SAM 3만)에 묶이지 않고, 포즈 기반 3D 방법과 사전학습 다중뷰 분할 모델의 추론도 비교 대상에 넣는다.

## 2. 과제 정의와 평가 규약 (합의 1절)

### 2.1 입력

장면당, `Data/<scene>/`에 `prepare_data.py`가 만든 것만 쓴다.

| 파일 | 내용 |
|---|---|
| `images/<cam>.jpg` | 기준 프레임의 모든 카메라 이미지 (9~46장) |
| `masks/<cam>.png` | 기준 프레임 정답 라벨맵, 주석 카메라 3대. 8비트 그레이, 픽셀값 = 객체 id, 0 = 배경 |
| `objects_labels.json` | 객체 id → {label, type} |
| `meta.json` | `ref_frame`, `ref_cam`(= MUVOD c_ini), `cams`, `mask_cams`, `image_size` |

기준 프레임은 `MVSeed_gt/SEED_MANIFEST.json`의 `start_frame`(대부분 0; CBABasketball·PoznanStreet 90,
Frog 180, MATF 30, Painter 40). 방법은 **`masks/<ref_cam>.png` 하나와 `images/*.jpg`만** 읽는다.
다른 주석 카메라의 정답은 채점기만 읽는다. 카메라 포즈는 제공하지 않는다. 포즈 기반 방법은 이미지에서
스스로 추정한다(COLMAP 등).

### 2.2 출력

```
Data/<scene>/Result/<method>/<cam>/<id>.png    이진 PNG(0/255), 이미지와 같은 해상도
Data/<scene>/Result/<method>/RUN.json          실행 시간(초), 최대 GPU 메모리, 설정, 코드 git 해시, 날짜
```

모든 카메라에 대해 쓴다. 기준 카메라는 정답을 그대로 복사한다(채점하지 않음). 객체가 그 뷰에 없다고
판단하면 PNG를 쓰지 않거나 빈 마스크를 쓴다. 둘 다 "빈 예측"으로 채점한다.

### 2.3 채점

- **대상 뷰**: 기준 카메라를 뺀 나머지 주석 카메라 2대 × 17장면 = 34뷰.
- **객체 집합**: MUVOD basic — 기준 카메라 기준 프레임 정답에 나타나는 객체 id. 대상 뷰 정답에 그
  객체가 없으면 (정답 빈 마스크) 그 쌍은 집계에서 뺀다. 단, 그 쌍에 예측이 있으면 "환각" 수로 따로 센다.
- **지표**: 쌍(scene, cam, id)마다 J(IoU)와 F(DAVIS 경계 F, `eval_jf.py`의 `db_eval_boundary`
  이식). 뷰 평균 → 장면 평균 → 전체 평균. J&F = (J+F)/2.
- **행 기록**: 쌍마다 J, F, 정답 면적(px), 예측 면적, 시점차(카메라 번호 거리; 포즈가 생기면 각도 추가),
  장면, 리그 종류(핀홀/어안).

### 2.4 보고 축

전체 평균 외에 반드시 다음 축으로 나눠 보고한다. 이 축이 없으면 방법 간 차이를 해석할 수 없다.

1. 장면별(17)
2. 객체 크기 구간: < 2,000 px / 2,000~10,000 / > 10,000 (대상 뷰 정답 면적 기준)
3. 핀홀 13장면 vs 어안 4장면
4. 시점차 구간
5. 실행 시간과 GPU 메모리

### 2.5 공정성 규칙

- 모든 방법은 같은 입력만 받는다 (2.1).
- 임계값·하이퍼파라미터는 데이터를 보지 않고 고정하거나 장면 단위 leave-one-out으로 정한다. 34개
  채점 뷰의 곡선을 보고 고르는 것은 금지.
- 실행 시간·메모리를 RUN.json에 기록한다.
- 방법의 비결정성(GPU)은 그대로 둔다. 재현 확인은 같은 설정 2회 실행의 차이를 보고하는 것으로 한다.

## 3. 비교 방법 (합의 2절)

가족별 대표를 고른다. 1차는 기존 Docker 이미지로 바로 돌릴 수 있는 것, 2차는 새 의존성이 필요한 것. 전파·정제에 쓰는 분할 모델은 모두 SAM 3로 통일한다.

| 단계 | 가족 | 방법 id | 내용 | 포즈 | 학습 | 이미지 |
|---|---|---|---|---|---|---|
| 1차 | 뷰=비디오 전파 | `sam3_prop` | 기준 프레임 N장을 비디오로 보고 SAM 3로 전파. `--order index`(기준선)와 `ref_outward` | 없음 | 없음 | `scsam3` |
| 1차 | 기하 재투영 | `pi3_reproj` | π³ 포인트맵으로 c_ini 마스크를 3D로 올려 각 뷰에 재투영. 변형 `--pixel-limit`로 작업 해상도 상향(255k → 1M px; 타일링은 타일마다 좌표계가 달라 불가) | 없음 | 없음 | `scgeo:pi3` |
| 1차 | 기하 + 정제 | `pi3_sam3` | `pi3_reproj` 결과를 점·박스 프롬프트로 SAM 3 이미지 예측 정제 | 없음 | 없음 | `scgeo:pi3` → `scsam3` |
| 1차 | 특징 대응 + 정제 | `dino_sam3` | DINOv2 밀집 대응으로 마스크 전이 → SAM 3 정제 (Matcher/PerSAM 방식) | 없음 | 없음 | `scgeo:dino` → `scsam3` |
| 2차 | 기하 기초 모델 | `vggt_reproj` 등 | VGGT 또는 MapAnything로 π³ 대체 비교 | 없음 | 없음 | 새 이미지 1개 |
| 2차 | 사전학습 다중뷰 분할 | 공개된 것 | MV-SAM, 3AM, SegMASt3R 중 코드·가중치가 공개된 것. 추론만 | 없음 | 추론만 | 새 이미지 |
| 선택 | 뷰=비디오 전파 | `sam2_prop` | 같은 전파를 SAM 2로. 어안 장면에서 SAM 3 실패 13쌍 중 10쌍을 SAM 2가 성공시킨 기록(`mvvos-algorithm-analysis.md` §2.4)이 있어, 어안 실패 원인 분석이 필요할 때만 대조 팔로 추가 | 없음 | 없음 | 기존 SCSam2 이미지 |
| 2차 | 포즈 기반 3D | `gs_lift` | COLMAP + 3DGS + FlashSplat식 역스플래팅 리프팅 → 전 뷰 렌더 | COLMAP | 장면별 3DGS | `3dgs`/`sa3d_gs` |

**제외**: 장면별 특징 필드 학습(SAGA, Gaussian Grouping 등)은 비용 대비 리프팅형과 포화 수준이 같아
`gs_lift` 하나로 대표. 다중카메라 다객체 추적 계열은 입력 형식이 다름.

**검증용 방법**: `oracle`(정답을 그대로 복사)은 채점기가 J=F=1을 내는지 확인하는 용도. `empty`(전부 빈
예측)는 하한.

## 4. 코드 구조 (합의 3절)

```
SCSegmentation2D/
  prepare_data.py              # ../SCSegmentation/Data/MVSeg → Data/ (작성됨)
  Data/<scene>/{images,masks,meta.json,objects_labels.json}
  Data/<scene>/Result/<method>/<cam>/<id>.png
  Data/<scene>/Result/<method>/RUN.json
  Data/<scene>/Cache/<provider>/   # π³ 깊이·포즈, DINO 특징, COLMAP 모델 등 중간 산출물
  mvseg2d/                     # 호스트 Python, torch 불필요 (numpy·Pillow·scipy 정도)
    io.py                      # meta/라벨맵/결과 읽기·쓰기, 라벨맵 ↔ 객체별 이진 마스크
    metrics.py                 # J, F (DAVIS 경계 코드 이식)
    evaluate.py                # Result/<method> → scores/<method>.json (쌍 단위 행)
    report.py                  # scores/*.json → 표 (2.4의 축), Markdown 출력
  methods/
    common.py                  # 인자 파싱, 입력 로드, 출력·RUN.json 쓰기. 각 컨테이너에 마운트
    oracle/run.py, empty/run.py
    sam3_prop/run.py
    pi3_reproj/run.py, pi3_sam3/run.py, dino_sam3/run.py
  scripts/run_<method>.sh      # docker run 래퍼 (이미지·마운트·HF_HOME·--user 고정)
  tests/                       # mvseg2d 단위 테스트 (합성 마스크)
  scores/                      # evaluate.py 출력 (gitignore)
  docs/
```

### 4.1 방법 인터페이스

```
python methods/<method>/run.py --scene <name> --data Data --out <method> [방법별 옵션]
```

- `common.load_scene(data, scene)` → 이미지 경로들, `ref_cam`, 기준 라벨맵, 객체 id 목록.
- `common.write_result(data, scene, method, cam, obj_id, mask_bool)`.
- `common.write_run(data, scene, method, seconds, gpu_mb, config)`.
- 기준 카메라 출력은 `common`이 정답을 복사해 준다 (방법이 신경 쓸 필요 없음).
- `--scene all`은 17장면 순회.

### 4.2 2단 방법의 캐시

`pi3_sam3`와 `dino_sam3`는 이미지가 다르다. 1단(기하/특징)은 `Data/<scene>/Cache/pi3/`(카메라별 깊이,
신뢰도, K, c2w, 작업 해상도)·`Cache/dino/`(카메라별 특징 맵)에 npz로 쓰고, 2단은 `scsam3` 컨테이너에서
캐시를 읽는다. `pi3_reproj`의 재투영 마스크도 `Result/pi3_reproj/`에 있으므로 `pi3_sam3`는 그것을
프롬프트 원천으로 읽는다(재계산 없음).

### 4.3 재사용 (이전 저장소는 읽기만, 수정 금지)

| 가져올 것 | 출처 | 용도 |
|---|---|---|
| `db_eval_boundary`, `_seg2bmap` | `../SCSegmentation/Data/MVSeg/eval_jf.py` | F |
| `unproject/transfer/project/visible`, 전·후방 워프 | `../SCSegmentation/MVSeed/geo/warp.py` | `pi3_reproj` |
| Pi3X 호출·`target_size`·`cameras_from_pi3` | `../SCSegmentation/MVSeed/geo/e4b_project.py` | `pi3_reproj` 1단 |
| `view_order` 와 전파 호출 | `../SCSegmentation/MVSeed/run_seed.py` | `sam3_prop` |
| COLMAP 첫 프레임 스크립트, 3DGS 학습 | `../SCSegmentation/MVSeed/geo/{colmap_first_frames.sh,train_3dgs.sh}` | 2차 `gs_lift` |

복사해 와서 이 저장소 안에 둔다. 경로 상수는 `Data/` 기준으로 바꾼다.

### 4.4 오류 처리

- 방법이 한 장면에서 실패하면 그 장면의 `RUN.json`에 `error`를 쓰고 다음 장면으로 간다. 채점기는
  `error`가 있는 (method, scene)을 표에 "실패"로 표시하고 평균에서 뺀 뒤 각주로 적는다.
- 채점기는 결과 PNG 해상도가 정답과 다르면 **거부**한다(리사이즈하지 않음). 방법이 원 해상도로 올려 써야
  한다.
- 라벨맵에 `objects_labels.json`에 없는 id가 있으면 경고만 하고 채점한다 (정답은 그대로 신뢰).

### 4.5 테스트

- `tests/test_metrics.py`: 합성 사각형 마스크로 J·F 값(완전 일치 1, 분리 0, 절반 겹침 등)과 DAVIS
  코드와의 일치.
- `tests/test_io.py`: 라벨맵 ↔ 이진 마스크 왕복, 빈 예측 처리, 해상도 불일치 거부.
- `tests/test_evaluate.py`: 임시 `Data/`에 `oracle`을 흉내 내어 J=F=1, `empty`로 J=F=0, 환각 계수.
- 방법 스크립트: Blocks(10대, 핀홀) 한 장면 스모크 → 전 장면.

### 4.6 실행 순서

1. `mvseg2d/` 채점기·보고서 + 테스트
2. `oracle`·`empty`로 채점기 검증
3. `sam3_prop` (기준선) 17장면 → 첫 보고서
4. `pi3_reproj` → `pi3_sam3` → `dino_sam3`
5. 1차 보고서 작성 후 2차 방법 선정 (공개 여부 확인)

## 5. 범위 밖

- 시간축 추적(2단계). 이 벤치마크는 기준 프레임 한 장만 본다.
- 이전 저장소 수정. `../SCSegmentation`은 읽기만.
- 논문 수치 산출. 여기 결과는 방법 선택의 근거이고, 본류에 옮길 때는 본류 규약으로 다시 돈다.
- 어안 4장면용 기하 적응(Fisheye3R 등). 어안은 별도 축으로 보고만 한다.
