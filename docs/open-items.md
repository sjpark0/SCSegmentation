> **2026-09-04 상태 변경.** `demoSCSam3OneStageNew`는 이제 `demoSCSam3MVOpt`의 바이트 동일
> 동결 스냅샷입니다. 아래에서 "OneStageNew"가 *무패치 원본 코드*를 뜻하는 서술은
> **git 태그 `baseline-onestagenew`** 기준으로 읽으십시오. 배경: [README.md](README.md)

# 미해결 항목

## 1. ~~HuggingFace 토큰 폐기~~ — 해소 (2026-09-04)

**폐기가 필요하지 않은 것으로 확인됐습니다.** 두 토큰 모두 이 머신을 벗어난 적이 없습니다:

- git 이력에 없음 — 푸시된 `99c05c7`의 README·Dockerfile 모두 0개. README의 토큰은 커밋되지 않은 작업트리에만 있었습니다.
- 이미지는 레지스트리에 푸시된 적 없음 (`RepoDigests: []`).
- 유일한 노출이던 로컬 이미지의 빌드 이력 레이어 3개는 토큰 없는 재빌드로 사라졌고, 옛 이미지(`pre-secret-backup`)는 2026-09-07 Phase 0에서 삭제했습니다.

- **2026-09-07 추가**: 로드맵 워크플로 원본 출력(`docs/raw/roadmap_raw.json`)에 옛 토큰 전문이 들어간 로컬 커밋이 있었으나 GitHub 푸시 보호가 막았고, 푸시 전에 커밋을 다시 써서 가렸습니다. 원격에는 간 적 없습니다 — [operations.md](operations.md) 함정 8.

빌드는 이제 토큰이 필요 없습니다 (`SCSam3/hf_cache/` COPY, README 참조). 폐기는 위생 차원의 선택 사항입니다.

## 2. `demoSCSam3ForSam2New`의 메모리 절감

메모리 수정을 넣었지만 무거운 3개 데이터셋은 여전히 OOM입니다.
SAM 2 스타일 API라 S1·S2의 앵커가 없어 절감이 0.1 GiB 수준에 그쳤습니다.

이 폴더의 inference 경로를 따로 분석해 S1·S2에 대응하는 자리를 찾아야 합니다.
근거와 실패 지점은 [investigations-closed.md](investigations-closed.md) 10번.

## 3. `demoSCSam3TwoStage` / `demoSCSam3TwoStageNew`

메모리 수정을 적용했지만(TwoStage 6곳, TwoStageNew 8곳) **검증하지 못했습니다.**
두 폴더가 `runMVSeg.py`의 `ALGOS`에 등록되어 있지 않아 MVSeg에서 한 번도 돌린 적이 없습니다.
현재는 구문 검사만 통과한 상태입니다.

**필요한 판단.** 이 두 알고리즘을 러너에 등록해서 돌릴 것인지. 등록하지 않을 거라면 패치를 되돌리는 편이 안전합니다
(검증되지 않은 변경이 남아 있는 것보다는).

원본 백업: `/tmp/sibling_backup/` — **`/tmp`이므로 재부팅 시 사라집니다.** 필요하면 옮기십시오.

## 4. ~~MVOpt를 OneStageNew에 반영할 것인가~~ — 완료 (2026-09-04)

**반영했습니다.** `.py` 6개를 복사했고, 숫자 폴더·`.dockerignore`·백업 파일은 보존했습니다.
OneStageNew는 동결 스냅샷이 되었고 개발은 MVOpt에서 진행합니다.
반영 전 상태는 git 태그 `baseline-onestagenew`.

검증: Welder 941장 완전 일치(반영 전에는 OOM), 독립 검증 5개 축 통과.

아래는 당시의 판단 근거입니다.

MVOpt는 OneStageNew + 메모리 수정입니다. 알고리즘 변화가 없으므로 되돌려 넣을 수 있습니다.

**다만 두 가지가 순수 수정이 아닙니다.**
- S6 (`RetireSpatialPredictor`) — 넣는 순간 `runMVSeg.py`가 OneStageNew에 대해 다르게 동작합니다
- S7 (`TRIM_CACHED_OUTPUTS`) — 새 기능, 기본 꺼짐

**추가 고려.** OneStageNew는 git 추적 중이고 발표된 기준선을 만든 코드입니다. MVOpt는 아직 커밋되지 않았습니다.
덮어쓰기 전에 MVOpt를 먼저 커밋해 두면 두 상태가 모두 이력에 남습니다.

## 5. 개선 제안 (Phase 2 이후)

[analysis/SCHEDULE.md](analysis/SCHEDULE.md)에 보류 항목 11건과 로드맵이 있습니다. 요약:

| 항목 | 내용 |
|---|---|
| P2 | 교차시점 윈도 파라미터화 + C1/C2 위생 수정 → 패키지 내 W=0 대조군 가능 |
| P4 | 양방향 윈도 (현재는 `range(-4, 0)`, 한쪽만) |
| P5 | 시드 복구 |

감사에서 나온 검증된 36건 전체는 [analysis/REPORT.md](analysis/REPORT.md)에 있습니다.
