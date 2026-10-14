# 인계 메모 — 2026-10-11

새 세션은 이 문서부터 읽습니다. 앞 세션(2026-09-04 ~ 10-11)의 결과와 남은 일을 정리했습니다.

## 1. 제출한 논문은 태그로 고정돼 있습니다

| 항목 | 값 |
|---|---|
| 학술지 | IEEE TCSVT 특집호 "Towards Spatial Intelligence" (마감 2026-10-31) |
| 원고 번호 | TCSVT-32023-2026 (ScholarOne: https://mc.manuscriptcentral.com/tcsvt) |
| 제목 | Propagating a Single-View Mask Across a Static Camera Rig: Attribution and Error Budget of a Foundation-Model Pipeline for Multiview Video Object Segmentation |
| 저자 | Seongjin Park(교신), Dohyung Kim, Jiwan Kim, Yugu Jung |
| 제출본 커밋 | `772fb35` — **태그 `tcsvt-submitted`** |
| 분량 | 14쪽(상한 14쪽, 편집실 메일로 명시). 표 10, 그림 10, 약력 4건 |

- 제출 파일은 `cd paper && make bundle`로 다시 만듭니다: `paper/submission/tcsvt-manuscript.pdf`, `tcsvt-source.zip`, `figures-numbered/`. 커버레터는 `paper/cover-letter.tex` → `.pdf`.
- 원고의 모든 수치는 `paper/numbers.tex` 매크로이고 `docs/paper-numbers.md`와 짝입니다.
- 제출 경과: 첫 제출이 "결론 절에 그림"으로 반려 → 그림 8·9·10 위치를 옮겨 재제출(`51b92e1`) → 그때 절 라벨이 빠져 2쪽 "Section ??" → `772fb35`에서 복구. **저자 센터의 파일이 `772fb35` 판인지 사용자 확인이 필요합니다.**

## 2. 리뷰가 오면

`main`에서 고치지 말고 태그를 별도 작업 폴더로 꺼내 씁니다.

```
git worktree add ../SCSegmentation-review tcsvt-submitted
cd ../SCSegmentation-review && git switch -c tcsvt-revision
```

- 원고 빌드는 그 폴더의 `paper/`에서 `make bundle`. 수치를 바꾸면 `numbers.tex`와 `docs/paper-numbers.md`를 함께 고칩니다.
- 새 실험이 필요하면 데이터·출력(`Data/`, `PlenopticServer1/`)은 원래 폴더의 것을 씁니다(대용량, git 밖).
- 빌드 뒤 꼭 확인할 것: 미정의 참조 0(`grep -c "Reference.*undefined" final.log`), PDF에 "??" 없음, 14쪽 이하, 결론 쪽에 그림 없음. 그림 위치를 옮길 때 `\label{}` 줄이 함께 잘리지 않았는지 라벨 목록을 이전 판과 대조합니다(2026-09-22 실수).
- 리뷰 대응 결정사항(외부 ChatGPT 리뷰 1~5차에서 나온 것): 동일 조건 XMem·SAM 2 재실행은 **하지 않기로 결정**(MUVOD가 기준선 코드를 공개하지 않음, §V-A 평가 조건 대조로 대체). 심사에서 다시 나올 수 있는 가장 큰 지적입니다.

## 3. 사용자 결정으로 확정된 것 (원고)

- 코드·데이터 공개: 비공개, "교신저자에게 요청 시 제공".
- ORCID: 원고에 넣지 않고 투고 시스템에 입력.
- AI 사용 고지: 일단 뺀 상태. 사용자가 재검토 중(IEEE 정책상 본문·그림 스크립트 초안을 AI로 작성했으면 사사에 공개가 원칙).
- 저자 소속: 네 명 모두 ETRI 공간미디어플랫폼연구실, 대전 34129. 사사: IITP(MSIT) RS-2022-II220022.
- 원고 소스에 작업 주석 없음(제출 zip에 그대로 들어가므로).

## 4. 연구 상태 요약

- 헤드라인 구성(제출본) `SegMaskSam3XW1CGPS4MFsCs10kG2`: MUVOD basic 91.015 / complete 86.399 (XMem 79.35).
- **제출 이후(2026-10-13) S1-E8**: 1단계를 `SCSegmentation2D`(브랜치 `geomem`)의 GeoSet-2 `geoset_pred_ft_rep`로 바꾼 새 헤드라인 `SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2` = basic **92.173** / complete **87.507**(+1.16 / +1.11, 17장면 전부 XMem 우위). 1단계가 MUVOD 다른 장면으로 학습(장면 단위 5-fold)했다는 단서를 붙여야 합니다. `docs/stage1-E8.md`.
- **S2-R6(2026-10-13)**: 기준 카메라 정지 물체에 프레임 0 프롬프트 마스크 유지 후처리 → 헤드라인 `SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs` = basic **92.538** / complete **87.869**. 같은 정답으로 설계했으므로 **LOSO +0.387을 반드시 함께 표기**. `docs/stage2-R6.md`.
- **1단계 코드 이전(2026-10-13)**: GeoSet-2 1단계는 이제 본 저장소 `stage1/`에 있습니다(SCSegmentation2D에서 이전, 재실행 바이트 동일). 입력·체크포인트는 git 밖 복사본(`stage1/Data`, `stage1/geomem/runs`). `docs/stage1-port.md`.
- **한 프로세스 실행(2026-10-14)**: 헤드라인은 `runMVSeg.py … --stage1 geoset_rep --frame0-seed --crop-small 10000 --ref-snap`(스윕: `STAGE1=geoset_rep … REFSNAP=1 ./runMVOptThree.sh`) 한 번으로 1단계부터 S2-R6까지 돕니다. 중간 결과는 `--save-stages`일 때만 `<out>_stages/`에. 파일 경로와 17장면 바이트 동일. `docs/pipeline-inprocess.md`.
- **S1-E11(2026-10-15) 채택 — 현 헤드라인 `SegMaskSam3XW1CGPS4MSdgeoset_rep_xFsCs10kG2Rs` = basic 92.626 / complete 87.954**: 1단계에 뷰 안 객체 배타(라운드 2 로짓을 원본 해상도에서 화소별 argmax, ≥ 50 % 잃을 마스크는 통째로 보호; `run.py --xcl out`, 프리셋 `geoset_rep_x`)를 더한 것. 키프레임 +0.0007 [+0.0001, +0.0013](관문 통과), 헤드라인 +0.088 [+0.019, +0.185](E8 규칙 통과; 본페로니 97.5 % 하한 +0.015). 한 프로세스 실행은 `--stage1 geoset_rep_x`(스윕 `STAGE1=geoset_rep_x`); S2-R6의 "시험 정답으로 설계, LOSO +0.387 병기"와 E8 학습 데이터 단서는 그대로 따라갑니다. 실측으로 덜 덮음 22쌍(0.66)은 기준 카메라가 라벨하지 않은 범위(놓친 화소 91 % 비가시, 이웃 시드 19 %만 덮음)라 **디코딩만으로는 못 닿음(18쌍 0.607, 새 정보 필요)**으로 확정했습니다(세 팔 모두 ±0.003). X2(기억 배타 `geoset_rep_xm`)는 미달, D(보호 없음)는 손해. 등록 `docs/stage1-E11-prereg.md`, 결과 `docs/stage1-E11.md`, 분석 `MVSeed/refloss/e11/`, 체인 `MVSeed/run_e11.sh`, 판정 `MVSeed/e11_judge.py` → `MVSeed/runs/e11/e11_judge.json`. 출력은 남겨 두었습니다: `stage1/Data/<scene>/Result/geoset_pred_ft_rep_{x,xm,xu}_e11`(+ `_r1_`), `Data/MVSeg/<scene>/SegMaskSam3XW1CGPS4MSdgeoset_rep_xFsCs10kG2Rs`, `…Rs_ip2`(Fencing Z4 실측).
- **S1-E10(2026-10-14) 채택 안 함 — 1단계 통과, 2단계 미달**: 1단계 출력 격자 288 → 576을 학습 없이 2× 크롭 확대 재디코딩(`stage1/geomem/zoom.py`, `run.py --zoom 2 --zoom-rope grid|remap`, 프리셋 `geoset_rep_z2`·`geoset_rep_z2r`; 라운드 2 마스크의 경계 띠 안에서만 교체, 삭제 없음). 키프레임 ΔJ&F: 격자 RoPE 그대로 +0.0023(관문 미달), 쿼리 RoPE 재매핑 **+0.0025 [+0.0001, +0.0051] 통과** → 재매핑 팔 헤드라인 basic 92.538 → 92.626(+0.088, CI [−0.034, +0.212]), complete +0.074 → CI 하한 < 0이라 채택 기준 미달. 헤드라인·`geoset_rep`는 그대로입니다. 결과 `docs/stage1-E10.md`, 등록 `docs/stage1-E10-prereg.md`, 체인 `MVSeed/run_e10.sh`, 판정 `MVSeed/e10_judge.py` → `MVSeed/runs/e10/e10_judge.json`. 출력은 남겨 두었습니다: `stage1/Data/<scene>/Result/geoset_pred_ft_rep_z2{,r}_e10`(+ `_r1_`, `timing_z2`, `z4_check`, `geoset_pred_ft_rep_gf_e10`)와 `Data/MVSeg/<scene>/SegMaskSam3XW1CGPS4MSdgeoset_rep_z2rFsCs10kG2Rs`. 읽을 점: 격자 몫은 실재하지만 작고(천장 +0.0078의 약 40 %), 2단계로는 약 0.5배만 전달됩니다.
- **S1-E9(2026-10-14) 효과 없음**: 1단계 마스크 디코더 미세조정 두 팔(`geoset_dec_ft`·`geoset_decb_ft`)이 키프레임 관문(ΔJ&F CI 하한 > 0, 장면 −0.01 안전)에 모두 미달(D1 −0.0004, D2 −0.0023; 최악 장면 −0.014) → 등록 규칙대로 2단계에 가지 않았고 헤드라인·채택 1단계(`geoset_pred_ft_rep`)는 그대로입니다. 결과 `docs/stage1-E9.md`, 등록 `docs/stage1-E9-prereg.md`. 체인 `MVSeed/run_e9.sh`(STAGES로 단계 선택), 판정 `MVSeed/e9_judge.py --stage 1|2` → `MVSeed/runs/e9/e9_judge.json`; 체크포인트 `stage1/geomem/runs/geoset_dec_ft`·`geoset_decb_ft`(git 밖)와 디코딩 `stage1/Data/<scene>/Result/*_e9`는 남겨 두었습니다. 읽을 점: 디코더가 약 1 % 움직여도 좋은 시드의 ±2 px 띠 빈틈(0.039)은 그대로이고, 띠 BCE는 등록한 W 1.0에서 기본 손실의 10~20배라 F를 깎았습니다(균형 가중치는 새 등록이 있어야 시험).
- **나머지 카메라·1단계 시드 진단(2026-10-14)**: 2단계 시간 안정화는 닫힘(규칙 36개 최선 +0.0004). 시드 비용 3.11 = 실패 59쌍 1.35(범위 0.66·혼동 0.45) + 좋은 시드 경계 1.76. SAM 3 다시 그리기·경계 띠 융합·팽창/수축은 전부 손해 → 1단계 개선은 모델 수준. `docs/stage1-seedfail.md`, `docs/stage2-D-nonref.md` §6.
- 1단계 시드 교체 노선은 S1-E6(SeedVote 기각, 09-12)에서 닫힘. S1-E7: 정답 시드를 넣으면 +4.38(대조군) — 남은 9점 중 약 4.4점이 1단계.
- S3-P: ETRI 32대 플레놉틱 링에서 텍스트 프롬프트 "person" 시딩이 정답 시딩과 같은 1단계(J 0.937 대 0.939). 데이터·출력은 `PlenopticServer1/CROOM/SegResult/`.
- 상세: `docs/ROADMAP.md`(↔ 노션 "구현 계획"), `docs/stage1-plan.md`(↔ 노션 "1단계 구현 계획"), `docs/stage1-E7.md`, `docs/plenoptic-P.md`.

## 5. 다음 세션에서 바로 할 수 있는 일

- 새 발전 방향은 `main`에서 시작합니다. 사전 등록 규율(등록 → 구현 → 실행 → 판정, 데이터를 본 뒤 규칙 변경 금지)과 수정 금지 목록(`SCSam3/demoSCSam3*/`, `SCSam3/sam3/`, `SCSam2/`, `eval/` 채점 코드, `Data/MVSeg/<장면>/{Video,Mask}`, 발표 `SegMask*` 폴더)은 그대로입니다.
- 커밋되지 않은 `MVSeed/runs/e4c_project.json`은 메타데이터만 바뀐 것으로, 내용 확인 뒤 커밋하거나 되돌리면 됩니다.
