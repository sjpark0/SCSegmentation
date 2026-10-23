# stage1/baselines — 외부·대조 기준선 하네스 (iccv2027-plan.md WP2)

모든 하네스는 **1단계만**(기준 카메라 c_ini의 정답 마스크 → 다른 카메라의 마스크) 돌리고, 출력은 `methods.geomem.dev_decode`와 같은 DEV 형식(`Data/_dev/<이름>/<장면>/<t:06d>/<카메라>/<객체>.png` + `index.json` + `DONE`)이라 `methods.geomem.dev_score`가 그대로 채점합니다. 하네스 검증은 DEV-A·DEV-A′에서만 하고(DEV-B는 등록의 확인용), TEST(키프레임·CROOM·ScanNet++)는 G4 뒤 한 번 돌립니다(계획 §4).

| 하네스 | 방법 | 이미지 | 코드·가중치 | 입력 규약 |
|---|---|---|---|---|
| `sam21_dev.py` | SAM 2.1 hiera-large 유사동영상 전파(G0와 같은 꼴, SAM 3 대신 SAM 2.1) | `scsam2` (= `SCSam2/Dockerfile`, torch 2.6 cu118, SAM 2 1.0) | `SCSam2/models/sam2.1_hiera_large.pt` | 기준 위치에 정답 마스크 프롬프트, 양방향 전파. 순서 `index`(카메라 번호) / `nn`(π³ 중심 최근접 이웃 사슬) |
| `cutie_dev.py` | Cutie(CVPR 2024, MIT) 유사동영상 추적 | `scsam2` + `PYTHONPATH`에 `third_party/Cutie` | `third_party/Cutie`(커밋 기록), `weights/cutie-base-mega.pth`(v1.0) | 기준 프레임 마스크로 메모리 초기화, 기준에서 앞·뒤 두 번 추적(새 메모리). `max_internal_size 480`, 나머지 `eval_config` |
| `samv_dev.py` | SAM-V(arXiv 2609.25490, Apache-2.0 코드·CC BY-NC 가중치) 프롬프트 조건 단일 객체 | `scsamv` (`Dockerfile.samv`: torch 2.3.1 cu121 + 저장소 pin) | `third_party/SAM-V`(+ `submodules/vggt`, `submodules/sam-hq`), `weights/samv/{sam_vit_h_4b8939.pth, vggt_model.pt, sam_v_stage2.pth}` | 객체마다 기준 마스크에서 균등 10점(SAM-V 벤치마크 규칙, seed 0), 프레임 집합 = 기준 + 평가 카메라 + 기준에 가까운 카메라(최대 16), 모델 입력 1024² |

- `third_party/`·`weights/`는 git에 넣지 않습니다(`.gitignore`). 커밋 해시·가중치 sha256은 각 실행의 `index.json`에 적힙니다.
- 실행·채점 큐: `scripts/baselines/run_baselines.sh`(E15 GPU 잠금 공유, 점수 `geomem/runs/_baselines/scores/`).
- 등록 원칙: 순서·점 수·프레임 수 같은 규약은 TEST 전에 이 README와 계획 문서에 고정합니다. 하네스는 DEV에서 "돈다"는 것만 확인하고 수치로 무엇도 고르지 않습니다.
- 다음 후보(계획 순위): VGGT-Segmentor(3), V²-SAM(4), SegMASt3R/MuViSeg(5). 코드 없는 MV-SAM·G²TAM·3AM·S4VY·DOMR은 인용만.
