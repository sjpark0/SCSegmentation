# TCSVT 원고

목표: **IEEE TCSVT 특집호 "Towards Spatial Intelligence: 3D Understanding, Generation,
and Reasoning from Real-World Videos", 투고 마감 2026-10-31.**  집필 시작 2026-09-16.
CFP 원문은 `cfp/TCSVT-SI-SpatialIntelligence-CFP.pdf`.

---

## 빌드

호스트에 TeX을 깔지 않습니다. 도커로만 돌립니다.

```bash
make image     # 최초 1회 (약 2.1 GB 내려받음)
make           # main.pdf  — 초안. \todo 표시가 빨간 글씨로 보입니다
make pages     # 쪽수
make final     # final.pdf — \todo 를 숨긴 제출본. 쪽수 확인용
make clean
```

현재 상태: `main.pdf` **5쪽**(TODO 포함), `final.pdf` **2쪽**(표 7개 + 초록만).
즉 표와 골격이 2쪽을 쓰고 본문에 쓸 수 있는 여유가 남습니다.

---

## 무엇이 고정돼 있는가

| 파일 | 내용 |
|---|---|
| `vendor/IEEEtran/IEEEtran.cls` | **V1.8b (2015/08/26)**, CTAN 원본 그대로. md5 `5b2e4fa15b0f7eabb840ebf67df4c0f7` |
| `vendor/IEEEtran/IEEEtran.bst` | v1.14 (2015/08/26). md5 `7c8250ecf02814ce6ddc0cdbb63df1dd` |
| `vendor/IEEEtran/IEEEtran_HOWTO.pdf` | 공식 사용 설명서 — 표·수식·저자 블록 규칙은 여기를 볼 것 |
| `vendor/IEEEtran/bare_jrnl.tex` | IEEE가 배포하는 저널 논문 빈 양식(참고용) |
| `docker/Dockerfile` | `texlive/texlive:latest-medium` + `multirow`·`orcidlink`·`preprint` |

클래스 파일을 이미지가 아니라 저장소에 둔 이유: 5년 뒤에도 같은 PDF가 나와야 합니다.
IEEEtran은 2015년 V1.8b 이후 갱신되지 않았고, IEEE Author Center가 지금도 이 판본을 배포합니다.

---

## 집필 규약

1. **숫자를 본문에 직접 타이핑하지 마십시오.** 전부 `numbers.tex`의 매크로를 씁니다.
   `numbers.tex`는 `docs/paper-numbers.md`와 1:1로 대응하고, 그 문서가 유일한 출처입니다.
   수치가 바뀌면 두 파일을 함께 고칩니다.
2. `\todo{...}`는 아직 안 쓴 곳, `\note{...}`는 쓸 때 지켜야 할 주의사항입니다.
   제출 직전 `\drafttrue` → `\draftfalse` (또는 `make final`)로 전부 사라집니다.
3. 참고문헌 중 `VERIFY` 주석이 붙은 9건은 **원문을 열어 확인하기 전에는 인용하지 마십시오.**
   `docs/paper-numbers.md` §9-6에 적은 약점 그대로입니다 — 조사 문서의 서지는 검색 결과에서
   전사한 것이 섞여 있습니다.

---

## 특집호 CFP (원문 확보: `cfp/TCSVT-SI-SpatialIntelligence-CFP.pdf`)

**Towards Spatial Intelligence: 3D Understanding, Generation, and Reasoning from Real-World Videos**

| | |
|---|---|
| 투고 마감 | **2026-10-31** |
| 1차 심사 통보 | 2026-12-15 (CFP에는 "December 15, **2027**"로 찍혀 있으나 앞뒤 일정상 2026 오타로 보입니다) |
| 수정본 마감 | 2027-01-15 |
| 2차 심사 | 2027-02-15 |
| 최종본 | 2027-03-15 |
| 게재 | 2027-03-31 |

객원편집자: Wengang Zhou (USTC) · Hezhen Hu (UT Austin) · Jose M. Alvarez (NVIDIA) ·
Jianfei Cai (Monash) · Nicu Sebe (Trento).

**우리 논문이 걸리는 주제 항목** (CFP의 topics of interest 그대로):
- *New benchmarks for 3D understanding/generation/reasoning* ← MUVOD 프로토콜 복원 + 오차 예산
- *Foundation models for 3D reconstruction/generation/reasoning* ← SAM 3 2단계 파이프라인
- *Advanced 3D compression (e.g., 3DGS)* / *Large-scale scene reconstruction* ← E4a·E4b·E4c 음성 결과

> **적합성 주의.** 이 특집호의 축은 "3D 이해·생성·추론"입니다. 우리 방법은 보정도 3D 재구성도
> 하지 않는 다중시점 2D 마스크 전파라서, 제목과 서론에서 **왜 이것이 spatial intelligence 문제인가**를
> 명시적으로 이어 주어야 합니다. 역설적으로 **E4 계열(3DGS·포인트맵·SfM 리프팅이 리그에서 실패한다는
> 음성 결과)이 이 특집호에 가장 직접적으로 말을 거는 부분**입니다.

---

## TCSVT 규정 — 확인 결과

### 쪽수 제한: **TCSVT는 공개적으로 쪽수 제한을 명시하지 않습니다**

이번에 확인할 수 있는 공식 경로를 전부 뒤졌고, 어디에도 숫자가 없습니다.

| 확인한 곳 | 결과 |
|---|---|
| 특집호 CFP 원문 | 쪽수 언급 없음 (일정·주제·객원편집자만) |
| CAS 저자 안내 `guidelines-authors` | 없음 — "IEEE 안내를 보라"고 넘김 |
| CAS 투고 안내 `tcsvt-manuscript-submission-guide` | 없음 |
| IEEE Xplore 저널 메타데이터(`rest/publication/home/metadata?pubid=76`) | `authorResources`·`submissionsGuidelinesLink`가 **위 두 CAS 페이지를 가리킴** — 즉 IEEE 본사도 이 둘이 전부라고 봄 |
| 다른 TCSVT 특집호 CFP 3건(LLM·SAM·데이터압축) | 전부 쪽수 언급 없음 |

**실제로 적용되는 규칙**은 IEEE Policy 6.10입니다: "각 간행물이 정한 쪽수 한도를 넘는 쪽마다
의무 초과쪽 요금을 물릴 수 있다." 한도 자체는 교정쇄 때 오는 *IEEE Page Charges and Reprint
Order Form*에 적혀 옵니다. CASS가 링크하는 IEEE 저자 안내서(`cfp/IEEE-info-for-authors-kit.pdf`)의
일반 지침은 **Transactions 정규 논문 = 인쇄 8~10쪽**입니다(2007년판이라 오래됐습니다).

**실무 결론:** 하드 리밋이 아니라 요금 문턱입니다. 10쪽 안팎을 겨냥하되, 넘겨야 할 이유가 있으면
넘기고 초과쪽 요금을 부담하면 됩니다. 정확한 문턱이 필요하면 객원편집자나 EiC(Shan Liu, Tencent)에게
메일 한 통이 가장 빠릅니다.

### 그 밖에 확인된 것

- 제출 형식: **TeX / LaTeX / Word / RTF**.
- 원고 종류 둘: **Transactions Papers**(정규)와 **Transactions Letters**(단문, 인쇄 3~4쪽).
- 초록 **50~200 낱말** (IEEE 일반 지침).
- 유사도: 원저 **30%**, 학회논문 확장본 **50%** 초과 시 즉시 반려.
- 게재료(2026) **$2,800**. IEEE 회원 5% · 학회 회원 20% 할인.
- 제출처: `https://ieee.atyponrex.com/journal/TCSVT` (ScholarOne 아님 — **Atypon Rex**).
- 저널 지표(2026-09 Xplore): Impact Factor **10.8**, CiteScore **18.1**, 월간, 현재 vol. 36 no. 9.

---

## 현재 뼈대

| 절 | 상태 | 근거 문서 |
|---|---|---|
| I. Introduction | 문단 5개 지시문만 | — |
| II. Related Work | 4개 소절 지시문 | `docs/mvimage-seg-survey.md` |
| III. Problem Setting and Protocol | 지시문 | `docs/muvod-protocol.md` |
| IV. Method | 4개 구성요소 지시문 | `docs/stage2-R2.md`, `docs/phase3-*.md` |
| V. Experiments | **표 4개 완성** (헤드라인·사다리·LOSO·비용) | `docs/paper-numbers.md` §1~§4 |
| VI. Error Budget | **표 2개 완성** (오라클·격자 천장) | 〃 §5~§6 |
| VII. Negative Results | **표 1개 완성** (다섯 생성기) | 〃 §7 |
| VIII. Limitations | 6개 항목 지시문 | 〃 §9 |
| IX. Conclusion | 지시문 | — |

표 7개는 전부 실제 측정값으로 채워져 있고 컴파일됩니다. 남은 것은 산문입니다.
