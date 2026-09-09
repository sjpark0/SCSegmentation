# SCSegmentation 연구 문서

다중시점 비디오 객체분리 — SAM 3 기반. 이 폴더는 **분석 결과와 판단 근거**를 담습니다.
코드 사용법이 아니라 "왜 이렇게 되어 있는가"와 "무엇을 이미 조사했는가"가 목적입니다.

---

## ★ 2026-09-04 상태 변경 — 먼저 읽으십시오

`SCSam3/demoSCSam3OneStageNew`는 이제 **`demoSCSam3MVOpt`의 바이트 동일 동결 스냅샷**입니다.
개발은 `demoSCSam3MVOpt`에서 진행합니다.

**폴더 이름이 오해를 부릅니다.** 발표된 J&F 수치를 만든 원본 OneStageNew 코드는 이제
git 태그에만 있습니다:

```bash
git show baseline-onestagenew:SCSam3/demoSCSam3OneStageNew/SCSam3Video.py
git checkout baseline-onestagenew -- SCSam3/demoSCSam3OneStageNew   # 원본 복원
```

`Data/MVSeg/*/SegMaskSam3OneStageNew/`의 마스크는 **원본 코드**가 만든 것입니다.
현재 코드로 재생성하면 출력은 같지만(실측 확인) 출처가 달라집니다.

이 변경 이전을 기술하는 문장이 아래 문서들에 남아 있을 수 있습니다.
각 문서에 시점을 표시해 두었으니, `docs/raw/`의 자료는 **조사 당시 상태**로 읽으십시오.

## 먼저 읽을 것

**[investigations-closed.md](investigations-closed.md) — 종결된 조사 13건.**
같은 조사를 반복하지 않기 위한 문서입니다. 이상한 코드를 발견했거나 이해되지 않는 결과를 만났다면 여기부터 보십시오.
특히 다음 두 개는 **고치면 안 되는 것처럼 보이는 것을 고쳐서 회귀를 낸 이력**이 있습니다:

- 4번 — 시점 전파에 시작 인덱스를 명시하면 J&F가 무너집니다 (Blocks 0.7484 → 0.2827)
- 5번 — `SCSam3/sam3`에 `__init__.py`가 없어 생기는 네임스페이스 섀도잉

## 문서 지도

| 문서 | 내용 |
|---|---|
| **[mvvos-algorithm-analysis.md](mvvos-algorithm-analysis.md)** | **알고리즘 분석 보고서 (2026-09-09)** — 뷰=프레임 가정의 세 위반(격자 정렬 위치 부호·시간 행·자기 확신 기억, 파일:줄), 실패 해부의 검산(면적 단조, SAM 2가 반구 실패 13쌍 중 10쌍 처리, 쌍별 오라클 +0.78, 이득 봉투 +2.07), 문헌 지형 7가족, 설계 3안과 반박, 권고 설계 **SeedVote**(동결 전파기 여러 개의 시드 후보를 위치 기준으로 검증·선택)와 실험 순서 E0~E9 |
| [investigations-closed.md](investigations-closed.md) | 종결된 조사 13건 — 재조사 금지 목록 |
| [memory-optimization.md](memory-optimization.md) | 메모리 수정 7종의 내용·근거·동등성 증거 |
| [mvopt-audit.md](mvopt-audit.md) | 그 7종의 반증 검증 결과와 잔여 위험 |
| **[phase3-direction-cini.md](phase3-direction-cini.md)** | **E4 결과 (2026-09-09) — 채택.** 중앙 기준 카메라에서는 양방향 읽기(모드 C)가 이웃 없던 카메라 9대를 +0.0138 올리고(CI 0 제외) 17장면 +0.22점. 헤드라인 90.0, 기준선 대비 +10.7 |
| [phase3-direction-cini-prereg.md](phase3-direction-cini-prereg.md) | E4 사전 등록 — P4가 조건부였던 이유, 판정 (a)(b)와 감시값 |
| **[xview-recovery.md](xview-recovery.md)** | **E3 결과 (2026-09-09)** — 이웃 읽기는 가려졌거나 뒤늦게 들어온 객체를 되찾는다: 갈림 구간 3/3 회복, 새 유실 0, 시드에 무관하게 재현. 사건이 셋뿐이라 "지지·검정력 부족". D1 마무리 |
| [xview-recovery-prereg.md](xview-recovery-prereg.md) | E3 사전 등록 — 사건·되찾음·판정 정의, 검증 데이터 읽기 전 커밋 |
| **[phase5-seed-repair.md](phase5-seed-repair.md)** | **Phase 3 / P5 결과 (2026-09-09) — 기각.** 17장면 Δ +0.000, Blocks −1.01; 잘못 놓인 복구 시드가 옆 시점 메모리와 같은 카메라 겹침 경쟁을 타고 다른 객체를 지움. Welder는 +2.06이나 규칙상 채택 안 함 |
| [phase5-seed-repair-prereg.md](phase5-seed-repair-prereg.md) | P5 사전 등록 — 규칙·엔드포인트·판정·감시값, 스모크 뒤 수정 이력 §8 |
| **[phase5-seed-repair-prep.md](phase5-seed-repair-prep.md)** | **P5 준비 (2026-09-09)** — c_ini 기준 시드 재집계(퇴화 15개), REPORT 감지 규칙의 오탐 90% 확인과 정정, 남은 결정 |
| **[muvod-comparison.md](muvod-comparison.md)** | **MUVOD 벤치마크 비교 결과 (2026-09-09)** — 공개 기준선 대비 basic +10.5 / complete +9.6, 17개 중 16개 우세, 공정성 대조와 Welder 진단 |
| **[muvod-protocol.md](muvod-protocol.md)** | **MUVOD 평가 프로토콜 채택 (2026-09-09)** — 대외 수치의 채점 규약, 장면별 c_ini를 확정한 근거, 우리 기존 기준 카메라와의 차이 |
| [experiments.md](experiments.md) | J&F 평가 방법과 전체 결과 |
| [phase1-measurement.md](phase1-measurement.md) | **Phase 1 결과** — 재현 확인, 짝지은 통계, ref/nonref, nb 구간, 면적 가중, ceiling, 퇴화 시드, 엔드포인트 제안 |
| [phase3-conditioning.md](phase3-conditioning.md) | **Phase 3 / P12 결과** — 공간축 메모리 공정화(게이트·이웃 포인터·행 이동): 네 변형이 CI 통과하나 동률·미소, 정보 없는 S4가 포인터와 동등 |
| [phase3-neighbourhood.md](phase3-neighbourhood.md) | **Phase 3 / P4 결과** — 이웃 메모리 방향·프레임 절제(A~E): 사전 등록, 감시값, 1차 미달·2차 통과, C 채택 결정 입력 |
| [phase2-control.md](phase2-control.md) | **Phase 2 결과** — XW 계열, XW0 ≡ OneStage, closure == all, E1 판정(미달), C2 효과, W 곡선, 계열 결정 입력 |
| [sam2-port-spec.md](sam2-port-spec.md) | **SAM 2 실험 사양서 (보류)** — A 설정 매칭 재실행 / B XW 이식. MUVOD 기준선 확보로 착수 보류, 담긴 측정 결과는 유효 |
| [sam2-baseline.md](sam2-baseline.md) | **SAM 2 기준선 조사** — 인용 열은 재현 불가, `New3`가 재현 가능; `fill_hole_area`는 무효였음; E6 정정 |
| [operations.md](operations.md) | 실행 방법, 러너 옵션, 자원 한계, Docker |
| [open-items.md](open-items.md) | 미해결 항목 |
| **[ROADMAP.md](ROADMAP.md)** | **구현 계획 (살아있는 문서) — P1~P14 상태, 단계, 실험 행 상태. 노션과 함께 갱신** |

## 분석 원본

| 경로 | 내용 |
|---|---|
| [analysis/REPORT.md](analysis/REPORT.md) | 알고리즘 감사 보고서 — 검증된 36건 (에이전트 131개, 반박 18건 제외) |
| [analysis/SCHEDULE.md](analysis/SCHEDULE.md) | 메모리 최적화 구현 스케줄 + 보류 항목 11건 + Phase 2 로드맵 |
| [analysis/BRIEF.md](analysis/BRIEF.md) | 실측으로 확립한 기준선 — 동작에 관한 근거 자료 |
| [analysis/report.html](analysis/report.html) | 감사 보고서 웹페이지 (`build_report_page.py`로 REPORT.md에서 생성) |
| [diffs/](diffs/) | upstream SAM 3 대비 알고리즘 diff 9종 + MVOpt vs OneStageNew diff (**복사 이전 기준**, 지금은 두 폴더가 동일하므로 재생성 불가) |
| [tools/](tools/) | 평가 타당성 검증·GT 커버리지·프레임 단위 분석 스크립트와 그 출력 |
| [raw/](raw/) | 사실 조사 원본 — J&F 표, 실행 이력, 출력 목록, 러너 분석, 패치 현황, 환경. **2026-09-04 복사 이전 상태** |
| `raw/copy_verify.json` | 복사 직후 독립 검증 5개 축 + 비평 (차단급 5건 포함) |
| `raw/sweep_*.md` | 복사 전 위험 분석 — 현재 상태를 가장 정확히 기술한 자료 |
| `raw/audit_full.json` | 감사 7종 × (분석 1 + 반증 3)의 전체 근거와 file:line 인용 |

## 노션 대응

같은 내용이 노션에도 있습니다. 이중 보존입니다.

| 노션 경로 | 대응 문서 |
|---|---|
| 연구 / 알고리즘 / 다중시점 비디오 객체분리 / **SAM 3 다중시점 알고리즘 감사** | `analysis/REPORT.md`, `analysis/report.html` |
| 연구 / 실험결과 / **다중시점 동영상 객체분리** | `experiments.md` |

## 코드 위치

| 경로 | 역할 |
|---|---|
| `SCSam3/demoSCSam3MVOpt/` | **개발 대상.** 교차시점 메모리 추적 + 메모리 수정 7종 |
| `SCSam3/demoSCSam3OneStageNew/` | **동결 스냅샷** — 위와 바이트 동일. 개발 금지 (`FROZEN.md` 참조) |
| git 태그 `baseline-onestagenew` | 발표 결과를 만든 원본 OneStageNew 코드 |
| `SCSam3/demoSCSam3OneStage/` | 교차시점 메모리 없는 판 |
| `SCSam3/demoSCSam3ForSam2*/` | SAM 2 스타일 API 판 |
| `SCSam3/demoSCSam3TwoStage*/` | 2단계 판 — 러너 미등록, 미검증 |
| `SCSam3/runMVSeg.py` | MVSeg 실행기 |
| `SCSam3/sam3/` | upstream SAM 3 체크아웃 — **수정 금지** |
| `Data/MVSeg/eval_jf.py`, `report_jf.py` | DAVIS J&F 평가·집계 |
| `figures/` | 논문 그림 생성 |
