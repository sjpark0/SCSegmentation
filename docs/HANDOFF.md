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
- **1단계 개발 집합 표준(2026-10-15)**: 키프레임(664쌍)은 사전 등록한 관문 전용이고, 설계·탐색은 DEV-A(비기준 학습 프레임 ref+4..+20, 3 300쌍), 확인은 DEV-B(나머지 15프레임, 등록당 한 번)에서 합니다. `docs/stage1-dev-protocol.md`; 디코딩 `scripts/docker_run.sh scsam3 python -m methods.geomem.dev_decode --preset <tag> --split A|B`(→ `Data/_dev/<preset>_<split>`), 채점 `-m methods.geomem.dev_score`, 비교 `stage1/.venv/bin/python -m mvseg2d.dev_compare BASE CAND`. DEV 기준선: `stage1/Data/_dev/geoset_rep_x_{A,B}`(E11 프리셋; DEV-A J&F 0.9078)와 `geoset_rep_xs_{A,B}`(현 채택 프리셋, 각 분할에 `dev_pairs.json`). DEV-B는 등록마다 한 번씩 쓰는 확인 집합이고(규약 §1), S1-E14는 DEV-B를 본 뒤 규칙을 바꾸지 않았으므로 소진되지 않았습니다. S1-E14가 이 규약의 첫 등록·첫 완주입니다(아래).
- **목록 3·4번(2026-10-16)**: **4a 결과 `docs/stage2-R6H.md` — REPLICATES**(basic +0.605 [+0.337, +0.880], 시점당 +0.91; S2-R6 보고에는 등록대로 "같은 17장면의 다른 주석 시점 34개에 그대로 적용해 시점당 +0.91 [+0.51, +1.32]" 한 줄만 더함, 일반화·독립 데이터로 쓰지 않음. basic 이득의 77 %가 정답이 프레임 0과 같은 프레임이라는 주석 절차 사실을 공개 문구에 더할지는 사용자 결정). 등록 `docs/stage2-R6H-prereg.md` — 고정 S2-R6 규칙을 MUVOD의 다른 주석 카메라 34개 시점(`SegMaskSam3XW1CGPS4MSdgtFsCs10kG2`에 `…Rh`)에 그대로 적용, E8 상수로 REPLICATES / HARMS / DOES NOT REPLICATE(독립 데이터 아님). 3번 `docs/stage2-R7-prereg.md` — `--ref-snap-mem`(접미사 `Rm`): c_ini PNG를 쓴 직후 같은 `ref_snap.decide`로 교체하면 프레임 0 마스크를 그 프레임 기억에 제자리 인코딩(크롭 대상 제외), 한 팔·E8 규칙·멈춤 규칙. 4b 독립 데이터는 다운로드·라이선스 사용자 결정 대기(MOSE 22 GB CC BY-NC-SA, MOVi-D, CROOM 직접 주석). 근거 `MVSeed/refloss/s2r6x/`.
- **S1-E14(2026-10-16) 채택 — 현 헤드라인 `SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs` = basic 92.766 / complete 88.086**: 1단계 최종 라운드 2 출력에 덮음 보정(`run.py --xcl outs`, `run.XCL_SHIFT` 0.5: pos = L > −0.5를 비교로, argmax는 이동 전 L, E11 배타·보호 그대로; 프리셋 `geoset_rep_xs`)을 더한 것. 개발 집합 규약의 첫 완주: DEV-A에서만 설계(+0.0028) → DEV-B 확인 +0.0032 [+0.0016, +0.0054] → 키프레임 +0.0026 [+0.0010, +0.0045] → 헤드라인 +0.140 [+0.056, +0.250](E8 규칙 통과, 최악 Dog −0.06). 한 프로세스 실행은 `--stage1 geoset_rep_xs`(스윕 `STAGE1=geoset_rep_xs`); S2-R6의 "LOSO +0.387 병기"와 E8 학습 데이터 단서는 그대로. 좋은 시드 경계 블록은 디코딩만으로는 닫혔습니다(S는 ±2 px 띠 오라클의 약 4–5 %만; 업샘플러·멀티마스크 선택·확대 뒤 배타·띠 BCE 균형 학습은 DEV-A에서 0 이하). 남은 길은 모델 수준 변경(마스크 경로만 미세 조정)이나 정답 없는 틈 주인 신호로, 각각 새 등록과 DEV-A 천장부터. 등록 `docs/stage1-E14-prereg.md`(정오표 §9), 결과 `docs/stage1-E14.md`, 근거 `MVSeed/refloss/e14/`, 체인 `MVSeed/run_e14.sh`, 판정 `MVSeed/e14_judge.py` → `MVSeed/runs/e14/`, DEV 출력 git 밖 `stage1/Data/_dev/geoset_rep_x{,s}_{A,B}`.
- **S1-E13(2026-10-15, 개발 집합만) 등록 안 함**: 덜 덮음용 새 정보원을 키프레임 없이 개발 집합(비기준 프레임 85 × 2카메라 = 3 300쌍)에서 쟀습니다. 규약 안 후보(SAM 3 프롬프트 범위 −0.0012, π³ 다중 시점 성장 −0.0048) 모두 손해, 규약 밖(객체 이름 텍스트 +0.0003, 두 번째 주석 카메라는 범위 부족의 약 10 %만)도 거의 돕지 않아 덜 덮음 블록을 1단계에서 닫았습니다. 결과 `docs/stage1-E13-dev.md`, 근거 `MVSeed/refloss/e13/`, 개발 디코딩 경로 `MVSeed/refloss/e13/dev-route/`(출력은 git 밖 `stage1/geomem/runs/_e13_dev`). 사용자 결정(10-15): 별도 트랙과 규약 안 마지막 시도는 하지 않고, 경계 오차로 가며, 개발 집합을 표준화.
- **S1-E12(2026-10-15) 효과 없음 — 반대 방향**: 매 단계 객체별로 기준 기억 토큰의 반평면을 지우고 전체 정답으로 학습한 M+(`geoset_cutall_ft`)가 키프레임에서 C0 대비 −0.0014(최악 CBABasketball −0.024), 같은 코드 0 용량 재학습 C0'(`geoset_cut0_ft`) 대비 −0.0028 [−0.0059, −0.0006]로 관문 미달. 기전 판정은 "효과 없음"이며 방향이 반대입니다(라운드 1 이미지 밖 재현율 −0.013; 모델이 "기억 증거 없음 → 배경"을 학습). 헤드라인과 `geoset_rep_x`는 그대로입니다. 덜 덮음 약 0.6점은 디코딩(E11)도 학습 신호 재배치(E12)도 닿지 않아 **새 정보가 필요**합니다. 재학습 한 번만으로 키프레임이 +0.0013 [+0.0001, +0.0030] 움직였으므로, 앞으로 학습 팔에는 0 용량 재학습 대조를 둡니다. 결과 `docs/stage1-E12.md`, 등록 `docs/stage1-E12-prereg.md`, 근거 `MVSeed/refloss/e12/`, 체크포인트 `stage1/geomem/runs/geoset_cut{all,0}_ft`(git 밖), 디코딩 `stage1/Data/<scene>/Result/geoset_cut{all,0}_ft_{rep,r1}_x_e12`.
- **S1-E11(2026-10-15) 채택 — 당시 헤드라인 `SegMaskSam3XW1CGPS4MSdgeoset_rep_xFsCs10kG2Rs` = basic 92.626 / complete 87.954(S1-E14가 대체)**: 1단계에 뷰 안 객체 배타(라운드 2 로짓을 원본 해상도에서 화소별 argmax, ≥ 50 % 잃을 마스크는 통째로 보호; `run.py --xcl out`, 프리셋 `geoset_rep_x`)를 더한 것. 키프레임 +0.0007 [+0.0001, +0.0013](관문 통과), 헤드라인 +0.088 [+0.019, +0.185](E8 규칙 통과; 본페로니 97.5 % 하한 +0.015). 한 프로세스 실행은 `--stage1 geoset_rep_x`(스윕 `STAGE1=geoset_rep_x`); S2-R6의 "시험 정답으로 설계, LOSO +0.387 병기"와 E8 학습 데이터 단서는 그대로 따라갑니다. 실측으로 덜 덮음 22쌍(0.66)은 기준 카메라가 라벨하지 않은 범위(놓친 화소 91 % 비가시, 이웃 시드 19 %만 덮음)라 **디코딩만으로는 못 닿음(18쌍 0.607, 새 정보 필요)**으로 확정했습니다(세 팔 모두 ±0.003). X2(기억 배타 `geoset_rep_xm`)는 미달, D(보호 없음)는 손해. 등록 `docs/stage1-E11-prereg.md`, 결과 `docs/stage1-E11.md`, 분석 `MVSeed/refloss/e11/`, 체인 `MVSeed/run_e11.sh`, 판정 `MVSeed/e11_judge.py` → `MVSeed/runs/e11/e11_judge.json`. 출력은 남겨 두었습니다: `stage1/Data/<scene>/Result/geoset_pred_ft_rep_{x,xm,xu}_e11`(+ `_r1_`), `Data/MVSeg/<scene>/SegMaskSam3XW1CGPS4MSdgeoset_rep_xFsCs10kG2Rs`, `…Rs_ip2`(Fencing Z4 실측).
- **S1-E10(2026-10-14) 채택 안 함 — 1단계 통과, 2단계 미달**: 1단계 출력 격자 288 → 576을 학습 없이 2× 크롭 확대 재디코딩(`stage1/geomem/zoom.py`, `run.py --zoom 2 --zoom-rope grid|remap`, 프리셋 `geoset_rep_z2`·`geoset_rep_z2r`; 라운드 2 마스크의 경계 띠 안에서만 교체, 삭제 없음). 키프레임 ΔJ&F: 격자 RoPE 그대로 +0.0023(관문 미달), 쿼리 RoPE 재매핑 **+0.0025 [+0.0001, +0.0051] 통과** → 재매핑 팔 헤드라인 basic 92.538 → 92.626(+0.088, CI [−0.034, +0.212]), complete +0.074 → CI 하한 < 0이라 채택 기준 미달. 헤드라인·`geoset_rep`는 그대로입니다. 결과 `docs/stage1-E10.md`, 등록 `docs/stage1-E10-prereg.md`, 체인 `MVSeed/run_e10.sh`, 판정 `MVSeed/e10_judge.py` → `MVSeed/runs/e10/e10_judge.json`. 출력은 남겨 두었습니다: `stage1/Data/<scene>/Result/geoset_pred_ft_rep_z2{,r}_e10`(+ `_r1_`, `timing_z2`, `z4_check`, `geoset_pred_ft_rep_gf_e10`)와 `Data/MVSeg/<scene>/SegMaskSam3XW1CGPS4MSdgeoset_rep_z2rFsCs10kG2Rs`. 읽을 점: 격자 몫은 실재하지만 작고(천장 +0.0078의 약 40 %), 2단계로는 약 0.5배만 전달됩니다.
- **S1-E9(2026-10-14) 효과 없음**: 1단계 마스크 디코더 미세조정 두 팔(`geoset_dec_ft`·`geoset_decb_ft`)이 키프레임 관문(ΔJ&F CI 하한 > 0, 장면 −0.01 안전)에 모두 미달(D1 −0.0004, D2 −0.0023; 최악 장면 −0.014) → 등록 규칙대로 2단계에 가지 않았고 헤드라인·채택 1단계(`geoset_pred_ft_rep`)는 그대로입니다. 결과 `docs/stage1-E9.md`, 등록 `docs/stage1-E9-prereg.md`. 체인 `MVSeed/run_e9.sh`(STAGES로 단계 선택), 판정 `MVSeed/e9_judge.py --stage 1|2` → `MVSeed/runs/e9/e9_judge.json`; 체크포인트 `stage1/geomem/runs/geoset_dec_ft`·`geoset_decb_ft`(git 밖)와 디코딩 `stage1/Data/<scene>/Result/*_e9`는 남겨 두었습니다. 읽을 점: 디코더가 약 1 % 움직여도 좋은 시드의 ±2 px 띠 빈틈(0.039)은 그대로이고, 띠 BCE는 등록한 W 1.0에서 기본 손실의 10~20배라 F를 깎았습니다(균형 가중치는 새 등록이 있어야 시험).
- **나머지 카메라·1단계 시드 진단(2026-10-14)**: 2단계 시간 안정화는 닫힘(규칙 36개 최선 +0.0004). 시드 비용 3.11 = 실패 59쌍 1.35(범위 0.66·혼동 0.45) + 좋은 시드 경계 1.76. SAM 3 다시 그리기·경계 띠 융합·팽창/수축은 전부 손해 → 1단계 개선은 모델 수준. `docs/stage1-seedfail.md`, `docs/stage2-D-nonref.md` §6.
- 1단계 시드 교체 노선은 S1-E6(SeedVote 기각, 09-12)에서 닫힘. S1-E7: 정답 시드를 넣으면 +4.38(대조군) — 남은 9점 중 약 4.4점이 1단계.
- S3-P: ETRI 32대 플레놉틱 링에서 텍스트 프롬프트 "person" 시딩이 정답 시딩과 같은 1단계(J 0.937 대 0.939). 데이터·출력은 `PlenopticServer1/CROOM/SegResult/`.
- 상세: `docs/ROADMAP.md`(↔ 노션 "구현 계획"), `docs/stage1-plan.md`(↔ 노션 "1단계 구현 계획"), `docs/stage1-E7.md`, `docs/plenoptic-P.md`.

## 5. 다음 세션에서 바로 할 수 있는 일

- 새 발전 방향은 `main`에서 시작합니다. 사전 등록 규율(등록 → 구현 → 실행 → 판정, 데이터를 본 뒤 규칙 변경 금지)과 수정 금지 목록(`SCSam3/demoSCSam3*/`, `SCSam3/sam3/`, `SCSam2/`, `eval/` 채점 코드, `Data/MVSeg/<장면>/{Video,Mask}`, 발표 `SegMask*` 폴더)은 그대로입니다.
- 커밋되지 않은 `MVSeed/runs/e4c_project.json`은 메타데이터만 바뀐 것으로, 내용 확인 뒤 커밋하거나 되돌리면 됩니다.
