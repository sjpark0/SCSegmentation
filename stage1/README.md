# SCSegmentation2D — 기준 프레임 다중시점 객체분리 벤치마크

> **2026-10-13 본 저장소로 이전.** 이 트리는 `SCSegmentation2D`(브랜치 `geomem`, 커밋 `5f40bf4`)의 1단계 코드를 그대로 옮긴 것입니다.
> 바뀐 것은 경로뿐입니다: `methods/common.py`의 `SIBLING`(이제 본 저장소 루트)·`git status --untracked-files=no`, `scripts/docker_run.sh`의 마운트(본 저장소 하나), `prepare_data.py`·`scripts/import_g0_seeds.py`의 원본 경로.
> 입력(`Data/<scene>/{images,masks,meta.json,Cache/pi3}`)과 체크포인트(`geomem/runs/{geoset_pred_ft,geoalign_ft}`)는 복사본이며 git 밖에 있습니다. 재현 확인은 `docs/stage1-port.md`(본 저장소)를 보십시오.
> 아래 본문의 `../SCSegmentation/...` 경로는 이제 `../...`(본 저장소 루트)를 뜻합니다.


MVSeg 17장면의 **기준 프레임** 한 장면당 N장(9~46 카메라)에서, 기준 카메라(c_ini)의 정답 마스크 1장을
고정하고 나머지 카메라의 객체별 마스크를 추출하는 과제에 대해 여러 방법론을 같은 채점기로 비교한다.
설계는 `docs/superpowers/specs/2026-10-06-mvseg2d-benchmark-design.md`, 구현 계획은
`docs/superpowers/plans/2026-10-06-mvseg2d-benchmark.md`.

## 데이터 준비

`../SCSegmentation/Data/MVSeg/<scene>/`에서 기준 프레임의 이미지(전 카메라)와 정답 라벨맵(주석 3대)만
`Data/<scene>/{images,masks,meta.json,objects_labels.json}`으로 복사한다. 기준 프레임과 기준 카메라는 각
장면의 `MVSeed_gt/SEED_MANIFEST.json`의 `start_frame`·`c_ini`를 따른다.

    python3 prepare_data.py            # 전 장면; --datasets Blocks Fencing 로 일부만

## 호스트 환경

채점기와 보고서는 torch 없이 호스트 venv에서 돈다. 방법들은 기존 Docker 이미지에서 돈다.

    ~/.local/bin/uv venv .venv --python 3.12
    ~/.local/bin/uv pip install --python .venv/bin/python -r requirements-host.txt
    .venv/bin/pytest -q

## 방법 실행

모든 방법은 같은 인터페이스를 쓴다. 입력은 `images/*.jpg`와 `masks/<ref_cam>.png`뿐이고, 출력은
`Data/<scene>/Result/<method>/<cam>/<id>.png`(0/255 이진 PNG, 원 해상도)와 `RUN.json`이다.

    .venv/bin/python methods/oracle/run.py --scene all --out oracle     # 채점기 검증용 (J=F=1)
    .venv/bin/python methods/empty/run.py  --scene all --out empty      # 하한 (J=F=0)
    scripts/run_sam3_prop.sh all                                        # GPU 방법은 scripts/ 래퍼로

## 채점과 보고

    .venv/bin/python -m mvseg2d.evaluate --method <method> --out scores/<method>.json
    .venv/bin/python -m mvseg2d.report scores/*.json --out scores/report.md

채점은 기준 카메라를 뺀 주석 카메라 2대 × 17장면(664쌍)에서 객체별 J(IoU)와 DAVIS 경계 F를 낸다.
보고서는 전체·장면별·객체 크기 구간별·핀홀/어안·시점차별로 나누고 실행 시간과 GPU 메모리를 함께 적는다.

## 결과

1차(SAM 3 전파, π³ 재투영, π³+SAM 3 정제, DINOv2+SAM 3) 결과와 읽기는 `docs/results-stage1.md`에, GeoMem-star(SAM 3 기억을 3D 기하 기억으로 바꾼 어댑터, 학습 포함) 결과는 `docs/results-geomem.md`에, GeoMem-2(3D 거리 어텐션 바이어스, RoPE 제거 ablation, 점진적 기억 집합 2라운드) 결과는 `docs/results-geomem2.md`에, GeoQuery(B-1 객체 쿼리 허브; 허브가 학습되지 않은 음성 결과와 원인) 결과는 `docs/results-geoquery.md`에 있다.

GeoMem 코드는 `geomem/`(포인트맵 → 72×72 3D 격자, 임베딩, 래퍼, 캐시, 학습), 벤치마크 방법은 `methods/geomem/`, torch 테스트는 `scripts/test_torch.sh`(scsam3 컨테이너)로 돈다.
