# 외부 데이터 (Hypersim · Replica · MOVi-MC-AC) — 출처·형식·준비 상태

> 2026-10-27 사용자 결정(ScanNet++ 접근 반려 뒤): **D1 두 번째 벤치마크 = Hypersim + Replica**, **D5 Fix C 데이터 = Hypersim 학습 분할 + MOVi-MC-AC**([iccv2027-plan.md](iccv2027-plan.md) §G2·D1·D2·D5). 이 문서는 세 데이터의 출처·라이선스·원본 형식·다운로드 상태를 적습니다. 평가 규약(어느 프레임을 기준으로, 어느 프레임을 평가하는지, 점수)은 **별도 사전 등록**으로 고정하며 여기서는 정하지 않습니다. 원본은 `stage1/Data/_ext/<데이터>/raw/`(git 밖), 도구·목록은 `stage1/Data/_ext/<데이터>/tools/`와 `stage1/scripts/ext/`.

## 1. Hypersim (Apple, CC BY-SA 3.0)

- 출처 `https://github.com/apple-aiml-research/ml-hypersim`(구 `apple/ml-hypersim`). 장면별 ZIP(1–20 GB, 총 약 1.9 TB) `https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/<scene>.zip`. 필요한 파일만 HTTP Range로 뽑는 기여 스크립트 `contrib/99991/download.py`(MIT)를 `tools/`에 복사해 씁니다(`--scene`, `--contains`는 AND).
- **받는 것(궤적마다)**: `images/scene_cam_XX_final_preview/frame.NNNN.color.jpg`(톤맵 색, 약 1024×768), `images/scene_cam_XX_geometry_hdf5/frame.NNNN.semantic_instance.hdf5`(장면 전역 인스턴스 ID — 모든 프레임·궤적에서 같은 ID = 같은 물체), `_detail/`(`cam_XX/camera_keyframe_{positions,orientations}.hdf5`, `metadata_scene.csv`의 `meters_per_asset_unit`, 카메라 목록). 내재 파라미터는 저장소의 `contrib/mikeroberts3000/metadata_camera_parameters.csv`(장면별 투영 행렬; `tools/`에 복사). 깊이(`depth_meters.hdf5`)는 받지 않음(필요하면 같은 스크립트로 추가).
- **분할**: `evermotion_dataset/analysis/metadata_images_split_scene_v1.csv`(장면 단위; `tools/`에 복사, 장면 목록 `tools/{train,val,test}_scenes.txt`). 공개 이미지 74 619장: **train 365장면/614궤적/59 543장, val 46/77/7 386, test 46/78/7 690**. 궤적(장면, cam_XX)마다 최대 100프레임.
- **SAM-V의 Hypersim 규약(Table 1, 프롬프트 조건 단일 객체)**: `stage1/baselines/third_party/SAM-V/README.md` — 한 "장면" = (장면, 카메라 궤적) 쌍 `ai_001_001_00`; `color/NNNNN.jpg`, `instance/NNNNN.png`(uint16), `pose/NNNNN.txt`(4×4 c2w, m); test 분할, 프레임 샘플러 `pose_near`(연속)·`pose_diverse`(다양) 16프레임, 기준 마스크에서 점 10개(seed 0); `benchmarks/compare_baseline_sam2.py`. 우리 하네스 `stage1/baselines/samv_dev.py`는 이미 이 점 규약을 따릅니다. **같은 데이터·같은 규약으로 SAM-V와 직접 비교**하는 것이 D1의 근거입니다.
- 상태: 2026-10-27 10:36 다운로드 큐 시작(`tools/download_queue.sh`, test → val → train; 장면당 약 20 s·35 MB, 전체 약 3 h·17 GB 추정; 로그 `logs/download_queue.log`, 장면마다 `raw/<scene>/.done`).

## 2. Replica (Meta, Replica Dataset Research Terms; 렌더는 Semantic-NeRF, 비상업 연구)

- 원본 메시 `https://github.com/facebookresearch/Replica-Dataset`(18장면; 연구·교육 목적, 받는 사람도 약관에 동의). 평가에 쓰는 것은 **Semantic-NeRF(Zhi et al., ICCV 2021)의 사전 렌더**: 8장면(`office_0–4`, `room_0–2`), `Sequence_1`에 900프레임, `rgb/rgb_N.png`, `depth/depth_N.png`(mm), `semantic_class/`, `traj_w_c.txt`(4×4 c2w, OpenCV), 인스턴스 맵은 별도 `Replica_Instance_Segmentation.zip`(`semantic_instance/semantic_instance_N.png`, uint16). 배포 Dropbox `https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija`. Gaga(CVPR 2025)의 재패키지(Google Drive `1lGD0ubFeGxi-8kFgX0PD754bHdq8gBtx`)는 같은 렌더의 train/test 분할본.
- **공개 규약(Gaga·Semantic-NeRF)**: 900프레임 중 `range(0, 900, 5)` 180장이 train, `+2` 180장이 test; 정답 = Semantic-NeRF 인스턴스 맵의 test 프레임; Gaga는 포즈를 COLMAP으로 다시 구함(우리는 π³). 가우시안 분할 계열(Gaussian Grouping·Gaga 등)의 공개 수치는 이 분할 위의 값이라 "제로샷 보조 벤치마크"로 인용 가능.
- 상태: 이 호스트에서 `www.dropbox.com` 연결이 자주 끊기고(`Recv failure`), Gaga Drive는 `gdown`이 "public link를 가져올 수 없음"(권한/접근 제한)으로 실패. 10-27 10:45 Dropbox 폴더 zip 다운로드를 재시도 중(`logs/dropbox.log`). **안 되면 대안**: (a) 사용자가 다른 망에서 Dropbox 폴더를 받아 NAS에 두기, (b) Replica 메시(GitHub 릴리스, 접근됨) + Habitat-Sim으로 Semantic-NeRF의 렌더 스크립트(`SSR/data_generation/`)를 돌려 직접 렌더(궤적은 직접 생성, 공개 분할과 달라져 공개 수치와 직접 비교 불가).

## 3. MOVi-MC-AC (LLNL, CC BY 4.0)

- 출처 `https://huggingface.co/datasets/Amar-S/MOVi-MC-AC`(arXiv 2507.00339; Kubric/MOVi로 생성, Google Scanned Objects). 장면별 `train/<id>.tar.gz`(1 651장면, 약 1.2 TB), `test/<id>.tar.gz`(390장면, 277 GB); 장면당 87 MB–2.1 GB(중앙값 약 0.7 GB). 목록·크기 `tools/{train,test}_scenes.txt`(HF API; train은 첫 1 000개만 받음).
- **형식(표본 `test/b5d6d5…`로 확인)**: `camera_0000–0005/`에 `rgba_NNNNN.png`(**256×256** RGBA), `segmentation_NNNNN.png`(uint8, 0 = 배경, k = k번째 인스턴스; 6대·24프레임에서 ID 일관), `depth_NNNNN.tiff`(float32), `metadata.json`(`camera`: 정규화 K, 프레임별 positions·quaternions, 센서 폭 32 mm·초점 35 mm; `instances`: asset_id·category·is_dynamic·프레임별 가시율·3D 상자), `obj_XXXX/`(객체별 아모달 rgba·segmentation(0/255)·depth). 카메라 6대는 정지·직선 이동·패닝이 섞이고 쌍별 시점차 20–134°(표본). 24프레임·12 fps.
- 용도: Fix C(C-ext)의 "넓은 시점차 + 가림" 학습 표본. **주의**: 해상도 256²는 MUVOD(1080p)·Hypersim(1024×768)보다 훨씬 낮고 물체가 작아, 학습 전에 업샘플 비율과 표본 수(장면 수)를 2부 등록에서 정해야 합니다. 평가에는 쓰지 않습니다.
- 상태: README·`test_obj_descriptors.json`·표본 장면 1개 받음(`raw/test/`). 학습용 부분집합(장면 수)은 E17 2부 등록 때 정한 뒤 받습니다(장면당 약 0.7 GB).

## 4. 남은 일

1. Hypersim 큐 완료 확인 → SAM-V 형식(`color/instance/pose`)과 우리 1단계 장면 형식으로 변환하는 스크립트(`stage1/scripts/ext/`) + 분할별 DEV 집합 정의(사전 등록).
2. Replica 렌더 확보(위 대안) → 8장면 변환.
3. E17 2부 등록 초안에 Hypersim train + MOVi-MC-AC 표본 추출 규칙(장면 수·프레임 쌍·시점차 분포) 반영.
4. G5 조건 ②를 "Hypersim 시험 분할에서 G0 대비 J&F CI 하한 > 0"으로 둔 상태(계획 §4). 평가 규약 등록 전에는 어떤 팔도 Hypersim test를 디코딩하지 않습니다(val·train만 개발용).

## 5. 이력

| 일자 | 내용 |
|---|---|
| 2026-10-27 | 문서 작성. Hypersim 큐 시작, MOVi-MC-AC 표본 확인, Replica 다운로드 경로 문제 기록. |
