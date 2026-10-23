# stage1/baselines — 외부·대조 기준선 하네스 (iccv2027-plan.md WP2)

모든 하네스는 **1단계만**(기준 카메라 c_ini의 정답 마스크 → 다른 카메라의 마스크) 돌리고, 출력은 `methods.geomem.dev_decode`와 같은 DEV 형식(`Data/_dev/<이름>/<장면>/<t:06d>/<카메라>/<객체>.png` + `index.json` + `DONE`)이라 `methods.geomem.dev_score`가 그대로 채점합니다. 하네스 검증은 DEV-A·DEV-A′에서만 하고(DEV-B는 등록의 확인용), TEST(키프레임·CROOM·ScanNet++)는 G4 뒤 한 번 돌립니다(계획 §4).

| 하네스 | 방법 | 이미지 | 코드·가중치 | 입력 규약 |
|---|---|---|---|---|
| `sam21_dev.py` | SAM 2.1 hiera-large 유사동영상 전파(G0와 같은 꼴, SAM 3 대신 SAM 2.1) | `scsam2` (= `SCSam2/Dockerfile`, torch 2.6 cu118, SAM 2 1.0) | `SCSam2/models/sam2.1_hiera_large.pt` | 기준 위치에 정답 마스크 프롬프트, 양방향 전파. 순서 `index`(카메라 번호) / `nn`(π³ 중심 최근접 이웃 사슬) |
| `cutie_dev.py` | Cutie(CVPR 2024, MIT) 유사동영상 추적 | `scsam2` + `PYTHONPATH`에 `third_party/Cutie` | `third_party/Cutie`(커밋 기록), `weights/cutie-base-mega.pth`(v1.0) | 기준 프레임 마스크로 메모리 초기화, 기준에서 앞·뒤 두 번 추적(새 메모리). `max_internal_size 480`, 나머지 `eval_config` |
| `vggts_dev.py` | VGGT-Segmentor(CVPR 2026, Apache-2.0) 쌍별 교차 뷰 분할 | `scsamv` + `PYTHONPATH`에 `third_party/VGGT-S/src`·`third_party/vggt_main` | `third_party/VGGT-S`(+ 공식 VGGT 클론, README의 트랙 헤드 수정 적용), `official_ckpts/main_exp.pth`, VGGT-1B(`weights/samv/vggt_model.pt` 링크) | 기준 마스크에서 k-means 5점, 기준·대상 518² 전체 프레임(크롭 없음), 예측기 refine 1회, 대상 8장 묶음 |
| `samv_dev.py` | SAM-V(arXiv 2609.25490, Apache-2.0 코드·CC BY-NC 가중치) 프롬프트 조건 단일 객체 | `scsamv` (`Dockerfile.samv`: torch 2.3.1 cu121 + 저장소 pin) | `third_party/SAM-V`(+ `submodules/vggt`, `submodules/sam-hq`), `weights/samv/{sam_vit_h_4b8939.pth, vggt_model.pt, sam_v_stage2.pth}` | 객체마다 기준 마스크에서 균등 10점(SAM-V 벤치마크 규칙, seed 0), 프레임 집합 = 기준 + 평가 카메라 + 기준에 가까운 카메라(최대 16), 모델 입력 1024² |

- `third_party/`·`weights/`는 git에 넣지 않습니다(`.gitignore`). 커밋 해시·가중치 sha256은 각 실행의 `index.json`에 적힙니다.
- 실행·채점 큐: `scripts/baselines/run_baselines.sh`(E15 GPU 잠금 공유, 점수 `geomem/runs/_baselines/scores/`).
- 등록 원칙: 순서·점 수·프레임 수 같은 규약은 TEST 전에 이 README와 계획 문서에 고정합니다. 하네스는 DEV에서 "돈다"는 것만 확인하고 수치로 무엇도 고르지 않습니다.
- 다음 후보(계획 순위): V²-SAM(4, mmengine·xtuner 스택), SegMASt3R/MuViSeg(5, uv + torch 2.10, MASt3R/VGGT 체크아웃) — 저장소는 `third_party/`에 받아 두었고 환경은 아직. 코드 없는 MV-SAM·G²TAM·3AM·S4VY·DOMR은 인용만.

## DEV 하네스 검증 결과 (2026-10-23 밤, `scripts/baselines/summarize.py` → `scripts/baselines/runs/dev_summary.json`)

C0 = `geoset_rep_xs`(시드 0): DEV-A 0.9107, DEV-A′ 0.9039. 수치는 "하네스가 돈다"의 확인이며 비교 주장에 쓰지 않습니다(TEST는 G4 뒤 한 번).

| 하네스 | DEV-A J&F | ΔJ&F vs C0 [CI] | DEV-A′ J&F | 넓은 쌍 ΔJ | 비고 |
|---|---|---|---|---|---|
| SAM 2.1 번호순 | 0.8837 | −0.027 [−0.039, −0.017] | 0.8831 | +0.012 | 전파 기반; 넓은 쌍에서는 C0보다 낫다(C0의 먼 뷰 붕괴, E15 참고) |
| SAM 2.1 이웃순 | 0.8704 | −0.040 | 0.8819 | +0.012 | |
| Cutie 번호순 / 이웃순 | 0.8105 / 0.7914 | −0.100 / −0.119 | 0.7858 / 0.7759 | −0.141 | `max_internal_size 480` |
| VGGT-Segmentor (쌍별) | 0.5237 | −0.387 | 0.4827 | −0.452 | 크롭 없는 전체 프레임 입력; 큰 객체에서 약함 |
| SAM-V (10점, 16프레임) | 0.5197 | −0.391 | — | — | DEV-A′ 두 회전은 생략(객체 50개 장면에서 프레임당 5분, 세 회전 약 10 h; 하네스 검증에는 DEV-A로 충분) |

- 제로샷 쌍별 방법(SAM-V·VGGT-S)의 낮은 값은 MUVOD(큰 객체·넓은 장면·점 프롬프트)와 그 방법들의 설계 조건(Hypersim·Ego-Exo 작은 객체) 차이가 큽니다. TEST 보고에서는 "제로샷" 열로 나누고(계획 WP2 공정성), 프롬프트 규약(점 수·크롭)을 그대로 밝힙니다.
