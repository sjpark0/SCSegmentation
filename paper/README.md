# TCSVT 원고

목표: **IEEE TCSVT 특집호 "Spatial Intelligence", 마감 2026-10-31.**
집필 시작 2026-09-16.

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

## TCSVT 규정 — 확인된 것과 못 한 것

### 확인됨 (2026-09-16, ieee-cas.org)

- 제출 형식: **TeX / LaTeX / Word / RTF** 모두 허용.
- 원고 종류 둘: **Transactions Papers**(정규 논문)와 **Transactions Letters**(단문).
- 유사도 검사: 원저 **30%**, 학회논문 확장본 **50%** 초과 시 즉시 반려.
  → 우리 `docs/`는 공개돼 있지 않으므로 문제 없지만, arXiv 선공개를 한다면 이 수치를 염두에 둘 것.
- 게재료(2026): **$2,800**, IEEE 회원 5% · 학회 회원 20% 할인.
- 제출처: `https://ieee.atyponrex.com/journal/TCSVT` (ScholarOne이 아니라 **Atypon Rex**로 바뀌었습니다).

### 확인 못 함 — 투고 전에 반드시 확인할 것

| 항목 | 왜 못 했나 | 어떻게 확인 |
|---|---|---|
| **정규 논문 쪽수 제한**과 초과 쪽 요금 | ieee-cas.org의 저자 안내 페이지가 이 값을 싣지 않음 | 특집호 CFP PDF, 또는 Atypon Rex 제출 화면의 안내 |
| 특집호 "Spatial Intelligence" 고유 규정 | CFP 페이지를 열지 못함(403) | **선생님이 가진 CFP 원문** — 여기에 쪽수·마감·심사 일정이 있습니다 |
| 초록 낱말 수, 색인어 개수 | 일반 IEEE 규정을 따르는 것으로 보이나 미확인 | IEEE Author Center |
| 저자 약력(biography)·사진 필요 여부 | 미확인 | 〃 |

> **쪽수 제한을 모른 채 쓰면 나중에 줄이는 데 며칠이 듭니다.** CFP PDF를 주시면 바로 반영하겠습니다.
> 그 전까지는 IEEE Transactions 통례대로 두 단 조판으로 진행합니다.

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
