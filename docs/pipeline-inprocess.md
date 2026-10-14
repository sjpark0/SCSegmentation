# 한 프로세스 파이프라인 — 1단계 → 2단계 → S2-R6를 파일 없이 한 번에

> 2026-10-13. 사용자 지시: "원래의 내 코드는 하나의 프로세스안에 들어가서 별도로 마스크저장은 없고 … 한번에 다 돌아갔는데 그렇게해주고, 필요한 경우 단계별로 저장할 수 있게해줘. 당연히 stage1과 stage2는 별도의 클래스나 파일로 존재할수는 있고".
> 그 전까지는 1단계 CLI(`stage1/scripts/run_geomem.sh`) → 시드 변환(`MVSeed/build_geoset_seeds.py`) → 2단계(`runMVSeg.py --seeds-from MVSeed_<tag>`) → 후처리(`MVSeed/s2r6_snap.py`)를 각각 따로 돌렸고, 단계 사이를 PNG 폴더가 이었습니다.

## 1. 사용법

헤드라인 한 장면, 한 프로세스(컨테이너 `scsam3`):

```
cd SCSam3
XVIEW=1 XMODE=C XGATE=1 XPTR=1 XSHIFT=4 XREF=muvod STAGE1=geoset_rep F0SEED=1 CROP=10000 CROPGATE=v2 REFSNAP=1 \
    ./runMVOptThree.sh Fencing             # -> Data/MVSeg/Fencing/SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs
```

`runMVSeg.py` 플래그로는 `--stage1 geoset_rep --ref-snap`(+ 기존 헤드라인 플래그)입니다. (2026-10-15 S1-E11 채택 뒤 헤드라인 프리셋은 `geoset_rep_x`입니다 — 같은 경로에 1단계 뷰 안 배타가 더해진 것; 아래의 바이트 검증은 `geoset_rep`로 한 것이고 `geoset_rep_x`의 CLI 대 한 프로세스 일치는 [stage1-E11.md](stage1-E11.md) Z3/Z3b로 확인했습니다.) 중간 결과가 필요하면 `SAVESTAGES=1`(`--save-stages`):

| `<dataset>/<out>_stages/` | 내용 |
|---|---|
| `stage1/final/<cam>/<obj>.png`, `stage1/round1/…`, `round1_conf.json`, `config.json` | 1단계 출력. 1단계 CLI의 `Result/<out>`, `<r1_out>`과 같은 배치·같은 바이트 |
| `seeds/<cam>/<start>/<obj>.png` + `SEED_MANIFEST.json` | 2단계에 들어간 시드(빈 시드 포함). `MVSeed_` 폴더와 같은 배치라 `--seeds-from`으로 다시 쓸 수 있음 |
| `stage2/` | `--ref-snap` 직전의 2단계 출력(하드 링크) |

저장하지 않으면 디스크에 쓰는 것은 최종 출력 폴더(채점 대상)와 `MANIFEST.json`뿐입니다.

## 2. 구조

| 파일 | 역할 |
|---|---|
| `stage1/methods/geomem/stage1.py` | **`Stage1` 클래스**: 프리셋(`geoset_rep` = S1-E8 채택, `geoset_abs`, `geoalign_rep`) → `run(scene)`이 마스크 딕셔너리를 돌려줌, `save()`(선택), `close()`(모델 해제·autocast 캐시 설정 복원) |
| `stage1/methods/geomem/run.py` | `decode_scene(scene, args, sink=None)`: 출력 대상을 `FileSink`(CLI, 기존 그대로) / `MemorySink`(한 프로세스)로 분리. 디코딩 코드는 그대로 |
| `SCSam3/stage1_supply.py` | 2단계 쪽 연결: 사전 점검(입력·체크포인트·누출·키프레임/c_ini 일치), 시드 변환(`build_geoset_seeds.py` 규칙을 메모리에서), 마스크 다이제스트, 저장 |
| `SCSam3/demoSCSam3MVOpt/seeds_from.py` | `apply_seeds()`: `apply_seed_folder()`와 같은 본체(`_apply`)를 메모리 시드로 |
| `SCSam3/ref_snap.py` | S2-R6 규칙을 출력 폴더에 제자리 적용(`s2r6_snap.py`와 같은 상수·판정 함수, 시험으로 고정) |
| `SCSam3/runMVSeg.py` | `--stage1`, `--ref-snap`, `--save-stages` |

실행 순서(`runMVSeg.py main`): 파일 사전 점검 → `build_runner`(SAM 3 패키지 경로 정리) → **1단계**(모델 적재 → 디코딩 → 해제) → 2단계 모델 생성 → 기준 카메라 프롬프트·교차 시점 전파 → **시드 슬롯에 1단계 마스크 대입**(`--seeds-from`과 같은 자리) → 시간 추적·기록 → 작은 물체 crop → **S2-R6 스냅** → `MANIFEST.json`.

- 폴더 이름은 파일 경로와 같습니다(`Sd<tag>`): 같은 시드이기 때문입니다. `MANIFEST.json`의 `seeds_from.mode = "stage1-inprocess"`와 `mask_digest`, `stage1`(팔·게이트·폴드·체크포인트 sha256)로 구분합니다.
- 1단계 입력은 그대로 `stage1/Data/<scene>/`(프레임 0 영상, 기준 GT, **미리 계산한 π³ 기하 캐시**)와 `stage1/geomem/runs/<arm>/fold<k>/`입니다. π³는 다른 이미지(`scgeo:pi3`)의 모델이라 이 프로세스에 넣지 않았고, 카메라 보정처럼 미리 만든 입력으로 둡니다.
- 1단계 모델은 2단계 모델을 만들기 전에 해제하므로 GPU 최대 사용량은 두 단계 중 큰 쪽입니다.
- 1단계 CLI(`run_geomem.sh`)와 파일 경로(`--seeds-from`, `s2r6_snap.py`)는 그대로 남아 있습니다. 1단계만 개발·평가할 때 씁니다.

## 3. 재현 확인

`MVSeed/inproc_check.py` → `MVSeed/runs/inproc_check.json`. 17장면을 `--out …Rs_ip --save-stages`로 돌려 단계마다 파일 경로 결과와 대조:

| 단계 | 대조 대상 | 기준 |
|---|---|---|
| 1단계 최종 | `stage1/Data/<scene>/Result/geoset_pred_ft_rep_e8` | PNG 바이트 |
| 1단계 1라운드 | `…/geoset_pred_ft_r1_e8` | PNG 바이트 |
| `round1_conf.json` | `…/geoset_pred_ft_rep_e8/round1_conf.json` | 바이트 |
| 시드 | `Data/MVSeg/<folder>/MVSeed_geoset_rep` | 화소(> 127), 모든 (카메라, 객체) |
| 2단계(스냅 전) | `…/SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2` | PNG 바이트 |
| 최종 | `…/SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs` | PNG 바이트 |

## 4. 결과

17장면 한 프로세스 실행(GPU 101.5분, 장면당 110~856초) → **모든 단계 바이트 동일**(`all_ok: true`):

| 단계 | 같음 | 다름·한쪽만 |
|---|---|---|
| 1단계 최종 PNG | 6,436 | 0 |
| 1단계 1라운드 PNG | 6,370 | 0 |
| `round1_conf.json` | 17 / 17 | 0 |
| 시드(화소) | 8,243 | 0 |
| 2단계 출력(스냅 전) PNG | 22,729 | 0 |
| 최종 출력 PNG | 22,729 | 0 |

최종 출력이 `…Cs10kG2Rs`와 바이트 동일하므로 헤드라인 **92.538 / 87.869**는 한 프로세스 실행에서도 그대로입니다(다시 채점할 필요 없음). S2-R6 교체 수도 장면마다 같습니다(예: Fencing 33 / 160).
같은 프로세스에서 1단계 모델을 먼저 적재·해제해도 2단계의 수치가 바뀌지 않는다는 것(autocast 캐시 설정 복원 포함)도 이것으로 확인됩니다.

테스트: 본 저장소 546 통과(새 `tests/test_pipeline_inprocess.py` 포함: 폴더 이름 = `--seeds-from`, 거부 조건, `apply_seeds` = `apply_seed_folder`, 시드 변환 규칙, 스냅 규칙 = `s2r6_snap.py` 200개 무작위 사례, 제자리 스냅과 사전 사본 보존), `stage1/` 호스트 111·토치 95(26 건너뜀).
검증 출력 `Data/MVSeg/<folder>/SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs_ip{,_stages}`는 git 밖에 남겨 둡니다(17장면 약 0.6 GB).
