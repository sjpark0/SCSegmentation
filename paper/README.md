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

현재 상태(2026-09-17): `final.pdf` **14쪽**(표 10개, 그림 10개, §VIII 32대 스튜디오 링 실험 포함).
외부 리뷰 5차까지 반영. `make bundle` → `submission/`(제출 PDF, 소스 zip, 번호 매긴 그림 파일 + captions.txt).
저자 정보 완료(2026-09-22): 저자 4인·IITP 사사·약력 4건(사진 `figures/bio/`)·우편번호 34129 확인. 공개 URL 없음(비공개, 요청 시 제공 문장). AI 사용 고지는 일단 제거(주석으로 보존, 사용자 재검토 예정). ORCID는 투고 시스템에 입력. 원고 TODO 0건.

---

## 무엇이 고정돼 있는가

| 파일 | 내용 |
|---|---|
| `vendor/IEEEtran/IEEEtran.cls` | **V1.8b (2015/08/26), IEEE Author Center 배포본** (2026-09-16 교체; 사용자가 준 `IEEE-Transactions-LaTeX2e-templates-and-instructions.zip`). CTAN 원본과의 차이는 IEEE가 2023-02-27에 제목 글꼴(`\LARGE`/`\huge`/`\Huge`)을 **18pt로 고정**한 것뿐. CTAN 원본은 `IEEEtran-ctan-v1.8b.cls.txt`로 보존. |
| `vendor/IEEEtran/bare_jrnl_new_sample4.tex`, `New_IEEEtran_how-to.{tex,pdf}` | 같은 zip의 새 샘플·설명서. 샘플의 `[lettersize,journal]` 옵션 중 `lettersize`는 클래스에 없는 옵션(기본이 letterpaper)이라 쓰지 않음 |
| `vendor/IEEEtran/IEEEtran.bst` | v1.14 (2015/08/26). md5 `7c8250ecf02814ce6ddc0cdbb63df1dd` |
| `vendor/IEEEtran/IEEEtran_HOWTO.pdf` | 공식 사용 설명서 — 표·수식·저자 블록 규칙은 여기를 볼 것 |
| `vendor/IEEEtran/bare_jrnl.tex` | IEEE가 배포하는 저널 논문 빈 양식(참고용) |
| `docker/Dockerfile` | `texlive/texlive:latest-medium` + `multirow`·`orcidlink`·`preprint` |

클래스 파일을 이미지가 아니라 저장소에 둔 이유: 5년 뒤에도 같은 PDF가 나와야 합니다.
IEEEtran은 2015년 V1.8b 이후 갱신되지 않았고, IEEE Author Center가 지금도 이 판본을 배포합니다.

---

## IEEE 스타일 가이드 대조 (2026-09-16)

사용자가 준 **IEEE Magazine Style Guide v.2026**은 잡지 편집 지침이고, TCSVT는 Transactions라
**IEEE Editorial Style Manual for Authors**(저널용, `scratchpad`에서 받아 대조)가 기준입니다. 둘이 다른 곳은
저널 규칙을 따랐습니다.

| 규칙 | 잡지 가이드 | 저널 매뉴얼 / IEEEtran | 원고 |
|---|---|---|---|
| 철자 | 미국식 | 미국식 ("Change all British spellings") | **영국식 36곳 → 미국식**으로 바꿈 (참고문헌 제목은 원문 유지) |
| 접두어 붙임표 | non/pre/multi/re 붙여 씀 | 같은 관행 (nonlinear 등) | non-reference→nonreference, pre-registered→preregistered, multi-view→**multiview**, re-run→rerun 등 57곳. `re-registered`는 붙이면 뜻이 흐려져 유지 |
| 아홉 이하 숫자 | 말로 | (명시 없음, 관행 동일) | 산문의 3·4·9 → three·four·nine (표는 숫자 유지) |
| 천 단위 | 쉼표 (1,000) | **얇은 공백** (62 000), 네 자리는 구분 없음 (4000) | 다섯 자리 이상 `\,` 유지, **네 자리(1066·1139·1051·1090·5254)는 구분 제거** |
| 숫자–단위 | — | **보통 공백** (얇은 공백 금지) | `\,GiB` 등 → `~GiB` |
| 퍼센트 범위 | 20%–30% (기호 반복) | 〃 | 91%–100% |
| 그림 참조 | "Figure 3" | **"Fig. 3"**, 표는 로마숫자 | Fig./TABLE I 유지 |
| 그림 설명 | "FIGURE 1." 문장형; 위/아래 대신 (a)(b) | "Fig. 1." ; 첫 단어 대문자, 관사로 시작 금지 | (a)(b) 라벨을 그림 안에 넣고 설명도 (a)(b)로; "The pipeline." → "Pipeline." |
| 약어 | 첫 사용에 정의; 표가 먼저 나오면 표에서도 정의 | 〃 | Table VIII 설명에 SfM·MVS 정의 추가; 초록의 CI는 풀어 씀 |
| 초록 | 한 문단, 수식·인용·각주 없음 | **150~250 낱말** | 242낱말 |
| 소속 | 국가 필수, 우편번호 | 〃 | "Daejeon 34129, Republic of Korea" (우편번호 확인 TODO) |
| 사사 | 단수 "Acknowledgment"; **AI 생성 내용 공개 의무** | 같은 조항 (p.12–13) | `\section*{Acknowledgment}` 추가, Claude 사용 공개 문장 초안 (문구 확정 TODO) |
| 약력 | GDPR상 요구 불가 | 〃 | 주석 틀만 (`IEEEbiographynophoto`) |

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

## 현재 상태 (2026-09-16, 커밋 168ff40 이후)

**초고 완성 — 10개 절 + 그림 9개 + 표 10개, 13쪽(참고문헌 포함).** §VIII에 32대 플레놉틱 리그 시연(텍스트 프롬프트 시딩, 링 거리 곡선, 가림 사건)을 추가했습니다(`docs/plenoptic-P.md`, 사전 등록 `docs/plenoptic-P-prereg.md`, 코드 `plenoptic/`). 절별 검증(515개 주장)의 지적 56건과 **외부 리뷰(ChatGPT, 2026-09-16) 1차 7개·2차 6개·3차 4개 항목**을 반영했습니다(`docs/paper-numbers.md` §10~§14a). 3차에서는 집단별 기여를 네 집단(기준 398 / F0 41 / N0 623 / 기타 4)으로 완결한 Table VII 추가, 초록·§VII 첫 문단·결론의 표현 축소, §VI-B·§VII 소제목 완화. 4차(최종 편집)에서는 §VII-C 소제목 "Candidate Availability and the Limits of SeedVote", 결론 "does not demonstrate a net gain", 오타 centered, 캡션·Fig. 5 높이 축소. 이어서 §V-A에 평가 조건 대조(채점 카메라·프레임·객체 수·정답 마스크·c_ini 패턴, 프로토콜 재량 0.06점)를 넣어 XMem 재실행 없이 조건 동일성의 근거를 제시(§14b) — 12쪽. 2차에서는 §VI-A의 대조군 기저(89.519→89.549) 정정, "초기화·추적 절반씩"을 "개입이 격차의 47.4 % 회복"으로, 식 (1)의 가정 명시, 벤치마크 가중 집단별 기여(`MVSeed/e7_contrib.py`, stage1-E7.md §7) 추가, Fig. 5 범례를 그래프 밖으로.
리뷰 지적 1(XMem·SAM 2 동일 파이프라인 비교)은 사용자 결정으로 미실행 — 한계 1·2에 그대로 둠. 지적 4에 답하는
**S1-E7**(정답 시드·투표 시드·오라클 시드를 2단계에 실제 통과)을 사전 등록·실행해 §VI Table VI를 오라클 표로
재구성했고 §VII-C의 환산값을 실측으로 바꿨습니다([../docs/stage1-E7.md](../docs/stage1-E7.md)).

| 절 | 상태 | 근거 문서 |
|---|---|---|
| Abstract | 씀 — 특집호 적합성 문장 포함, 약 245 낱말 (IEEE 일반 지침 50~200; TCSVT 게재 논문은 250 안팎이 흔함) | `docs/paper-numbers.md` |
| I. Introduction | 씀 — 5문단 + 기여 4개 | 〃, `mvimage-seg-survey.md` §0 |
| II. Related Work | 씀 — 4개 소절 | `mvimage-seg-survey.md` |
| III. Problem Setting and Protocol | 씀 — 과제·지표·c_ini 복원(증거 A/B/C, Dog 불확실성) | `muvod-protocol.md` |
| IV. Method | 씀 — 2단계, 이웃 읽기(등록 이력 포함), 프레임 0 시드, 크롭 재추적, 사전 등록 | `phase2/3-*.md`, `stage2-R1/R2.md` |
| V. Experiments | 씀 — 표 I·II(장면별)·III(사다리)·IV(LOSO)·V(비용) | `paper-numbers.md` §1~4 |
| VI. Error Budget | 씀 — 표 VI(오라클)·VII(격자 천장), 상수 결손 | `error-budget.md`, `stage1-E0c.md` |
| VII. Negative Results | 씀 — 표 VIII, 생성기별 실패 양상, 합집합 대 환각, 투표, 1:15 | `stage1-E*.md` |
| VIII. Limitations | 씀 — 7개 항목 | `paper-numbers.md` §9 |
| IX. Conclusion | 씀 | — |
| Fig. 1 | TikZ 파이프라인 도식 (외부 파일 없음) | — |

**그림 (2026-09-16 추가, `figures/make_figs.py`가 생성)**

| 그림 | 파일 | 내용 | 자리 |
|---|---|---|---|
| Fig. 1 | (TikZ) | 파이프라인 도식 | §IV |
| Fig. 2 | `fig_crop.pdf` | 크롭 재추적 — 회복 예(Breakfast 181 px 그릇)와 창 이탈 예(CBA 공) | §IV-D |
| Fig. 3 | `fig_propagation.pdf` | 세 리그(반구 어안·평면·호)에서 기준→두 카메라, t=0/20, 정답 윤곽 | §V-B (두 단) |
| Fig. 4 | `fig_scenes.pdf` | 장면별 Δ 막대 | §V-B |
| Fig. 5 | `fig_perframe.pdf` | 프레임별 J&F, 기준 398쌍 대 비기준 668쌍 — 본문 수치와 일치 확인 | §VI-B |
| Fig. 6 | `fig_seed_failures.pdf` | 시드 실패 둘: Welder 상자(자리 틀림), CBA 관중석(빈 시드) | §VII |

생성은 `scsam3` 컨테이너에서(호스트에 cv2·matplotlib 없음):
```bash
docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
    scsam3 python paper/figures/make_figs.py
```
읽기만 합니다(`Data/MVSeg/<scene>/{Video,Mask}`, 공개 예측 폴더, `jf_r4.json`); 쓰는 곳은 `paper/figures/fig_*.pdf`뿐.
Fig. 4의 장면별 값만 스크립트에 옮겨 적은 것(`docs/muvod-comparison.md`)이라 그 표가 바뀌면 같이 고칠 것.

**특집호 적합성 (2026-09-16 명시)**
- 원고: 초록의 "spatial intelligence" 문장, 서론 4번째 문단(리그 영상 = 볼류메트릭 캡처·객체별 3D 재구성의 입력이고 그 파이프라인이 다중시점 마스크를 소비함; 2D 기초 모델의 시점 간 일관성 + 명시적 3D가 보태는가), 기여 C3(3DGS 발견), 결론 마지막 문장, 색인어 `spatial intelligence`.
- **`cover-letter.md`**: CFP 주제 항목에 대응시킨 영문 커버레터 초안 (Atypon Rex 투고 칸용). 이해충돌 문장은 확인 후 유지.

**아직 없는 것**
- (완료 2026-09-22) 사사·우편번호·약력. AI 고지는 제거 상태에서 재검토. ORCID는 투고 시스템에서 입력.
- 참고문헌 31건 전부 arXiv 원문과 대조 완료, `VERIFY` 0건 (WildSeg3D ICCV 2025 DOI 확인, SA3D는 NeurIPS 판 저자 9인).
- 관련연구 문헌 수치는 arXiv PDF 원문과 대조 완료 — 결과표 `docs/paper-numbers.md` §11. 원문에 없던 두 문장은 뺐음.

**집필 규약 보충**: 장면별 표(Table II)만은 매크로가 아니라 `docs/muvod-comparison.md`의 표를 옮긴 것입니다.
그 표가 바뀌면 Table II를 다시 옮겨야 합니다.
