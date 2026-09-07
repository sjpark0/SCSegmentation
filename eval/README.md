# eval/ — MVSeg J&F 채점·집계 도구

발표 수치를 만든 `Data/MVSeg/eval_jf.py`(2026-09-02)·`report_jf.py`(2026-09-03)를 저장소로 가져와 확장한 것입니다.
원본은 첫 커밋(`b54e7b6`)에 그대로 있고, `Data/MVSeg/`의 사본도 건드리지 않았습니다. 판독은 [docs/phase1-measurement.md](../docs/phase1-measurement.md).

| 스크립트 | 하는 일 | 어디서 | 의존성 |
|---|---|---|---|
| `eval_jf.py` | DAVIS J&F 채점 (v2: 프레임별 J·F, GT/예측 면적, 시점 인덱스·시드 ID·기준시점 메타) | 컨테이너 | numpy, cv2 |
| `report_jf.py` | 집계와 표. `--paired`, `--split ref\|nonref`, `--bin-by-nb`, `--area-weighted`, `--aggregation pooled\|sequence`, `--ceiling`, `--paper` | 호스트 | 표준 라이브러리 |
| `seed_census.py` | 퇴화 시드 집계 (REPORT P5 진단) | 컨테이너 | numpy, cv2 |
| `manifest.py` | 마스크 폴더 `MANIFEST.json` 쓰기·검증·일괄 생성 (`write` / `verify` / `show` / `sweep`) | 호스트 | 표준 라이브러리 |

```bash
cd /home/sjpark/Documents/SCSegmentation
IN_CONTAINER="docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host/$PWD scsam3"

# 1) 원시 점수 (CPU만, 약 3분) → Data/MVSeg/jf_v2.json
$IN_CONTAINER python eval/eval_jf.py --methods SegMaskNew1 SegMaskSam3OneStage SegMaskSam3MVOpt \
    --out /host$PWD/Data/MVSeg/jf_v2.json

# 2) 표
python3 eval/report_jf.py --methods SegMaskNew1 SegMaskSam3OneStage SegMaskSam3MVOpt          # 헤드라인 (--common 기본)
python3 eval/report_jf.py --paired SegMaskSam3OneStage SegMaskSam3MVOpt --split nonref --bin-by-nb
python3 eval/report_jf.py --paper docs/raw/paper_tables.md                                     # 논문 표 전부

# 3) 퇴화 시드, 매니페스트
$IN_CONTAINER python eval/seed_census.py
python3 eval/manifest.py verify Data/MVSeg/Fencing/SegMaskSam3MVOpt
```

주의: 컨테이너 안에서 절대 경로를 쓸 때는 `/host` 접두사를 붙입니다. `--user`를 빼면 결과 파일이 root 소유가 됩니다.
`report_jf.py`의 v2 전용 기능(`--split`, `--bin-by-nb`, `--area-weighted`, `--ceiling`, `--paper`)은 v1 파일(`jf_raw.json` 등)에서는 메시지와 함께 종료합니다.
