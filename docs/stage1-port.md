# 1단계(GeoSet-2) 코드 이전 — SCSegmentation2D → `stage1/`, 재실행 결과 **바이트 동일**

> 2026-10-13. 사용자 지시: "1단계쪽 코드내용을 여기로 가지고와서 실행해줘. SCSegmentation2D는 … 이제부턴 여기서 진행".
> 이전 커밋 2c2b7bc, 재현 확인 `MVSeed/run_stage1_port.sh` → `MVSeed/stage1_port_check.py` → `MVSeed/runs/stage1_port_check.json`.

## 1. 무엇을 옮겼나

- **코드**: `SCSegmentation2D` 브랜치 `geomem` 커밋 `5f40bf4`의 `geomem/ methods/ mvseg2d/ scripts/ tests/ tests_torch/ docs/ prepare_data.py pyproject.toml requirements-host.txt README.md`를 `git archive`로 `stage1/`에 그대로.
  E8 출력은 `b089577`에서 만들었지만, 그 뒤 커밋(카메라 순서 수정 8532969 등)은 GeoSet 추론 경로(`methods/geomem/run.py`, `geomem/*`, `methods/common.py`, `methods/sam3_single.py`)를 건드리지 않습니다.
- **바꾼 것은 경로뿐**:
  - `stage1/methods/common.py`: `SIBLING` = 본 저장소 루트(SAM 3 패키지 `SCSam3/demoSCSam3MVOpt`·`SCSam3/hf_cache`), `RUN.json`의 git dirty 판정은 추적 파일만(`--untracked-files=no`).
  - `stage1/scripts/docker_run.sh`: 본 저장소 하나를 같은 `/host…` 경로로 마운트(형제 저장소 읽기 전용 마운트 제거).
  - `stage1/prepare_data.py`, `stage1/scripts/import_g0_seeds.py`: 원본 경로를 본 저장소 `Data/MVSeg`로.
- **git 밖 복사본**(`.gitignore`: `Data`, `stage1/geomem/runs/`, `*.pt`):
  - 입력 `stage1/Data/<scene>/{meta.json, objects_labels.json, images, masks, Cache/pi3}` 17장면 838 MB. π³ 캐시는 다시 만들지 않고 복사(카메라 순서에 의존하는 결합 추정이라 재생성하면 달라질 수 있음).
  - 체크포인트 `stage1/geomem/runs/{geoset_pred_ft, geoalign_ft}/fold0~4` 231 MB. 10개 모두 `config.json`의 `ckpt_sha256`과 일치. `folds.json` sha256 `ab12356d…` 동일.
  - 재학습에 필요한 특징 캐시(`Cache/sam3_feats*` 84 GB, `frames/`, `pi3_frames` 등)와 다른 팔의 체크포인트는 옮기지 않았습니다. 재학습할 때 가져옵니다.
- **본 저장소 호출부 재지정**: `MVSeed/build_geoset_seeds.py`(기본 원본 = `stage1`, 검증용 `--dst-suffix` 추가), `MVSeed/run_e8.sh`, `MVSeed/run_s2r5.sh`, `SCSam3/ft_encoder.py`(체크포인트·folds), `SCSam3/runMVSeg.py` 도움말, `tests/test_ft_encoder.py`. `MVSeed/e8_regen_check.py`는 2D 저장소 안의 과거 대조라 그대로 둠.

## 2. 재현 확인

환경: 이미지 `scsam3` `sha256:9064e3dc8548`(E8과 같음), 드라이버 580.178.04, RTX 6000 Ada. 세 구성 GPU 27분.

| 폴더 (`stage1/Data/<scene>/Result/…` vs 2D `Data/<scene>/Result/…`) | 동일 PNG | 다름 | 한쪽만 | `round1_conf.json` | `RUN.json`(시간·git 제외) |
|---|---|---|---|---|---|
| `geoset_pred_ft_rep_e8` (E8 채택) | **6,436** | 0 | 0 | 17/17 동일 | 17/17 동일 |
| `geoset_pred_ft_r1_e8` | 6,370 | 0 | 0 | — | 17/17 |
| `geoset_pred_ft_abs_e8` | 6,471 | 0 | 0 | 17/17 | 17/17 |
| `geoset_pred_ft_r1_e8b` | 6,370 | 0 | 0 | — | 17/17 |
| `geoalign_ft_x_rep_e8` | 6,442 | 0 | 0 | 17/17 | 17/17 |
| `geoalign_ft_x_r1_e8` | 6,397 | 0 | 0 | — | 17/17 |

2단계 시드: 새 출력으로 `build_geoset_seeds.py --dst-suffix _port`를 돌려 기존 `Data/MVSeg/<scene>/MVSeed_{geoset_rep, geoset_abs, geoalign_rep}`와 비교 → 세 집합 각 **8,243 PNG 전부 동일**(`SEED_MANIFEST.json`만 다름). 임시 폴더는 지움.
시드가 같으므로 S1-E8·S2-R5·S2-R6의 2단계 결과와 헤드라인 92.538은 이 이전으로 바뀌지 않습니다.

테스트: `stage1/` 호스트 111 통과·3 건너뜀(`stage1/.venv`), 토치 114 통과(`stage1/scripts/test_torch.sh`, scsam3), 본 저장소 `tests/test_ft_encoder.py` 27 통과(등록 체크포인트 표·누출 검사 포함, 이제 `stage1/`을 읽음).

## 3. 이제부터

- 1단계 실행: `cd stage1 && scripts/run_geomem.sh all --arm geoset_pred_ft --gate reproj --out <이름> --r1-out <이름>` → `stage1/Data/<scene>/Result/<이름>`, 그다음 `python3 MVSeed/build_geoset_seeds.py --tag … --out <이름>`로 2단계 시드.
- 1단계 개발 이력(GeoSet·GeoAlign·GeoFuse 결과 문서)은 `stage1/docs/`에 있습니다.
- `SCSegmentation2D`는 참조용으로 남겨 둡니다(E8 원본 출력 보관). 새 작업은 하지 않습니다.
