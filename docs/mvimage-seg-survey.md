# 다중시점 이미지 객체분리(한 순간 · 3D 객체분리) 최신동향 조사 (2023–2026)

> 2026-09-10. 사용자 요청: "딱 한 프레임에서 객체분리 잘하는 게 목적이니 다중시점영상에서의 객체분리 알고리즘을
> 최신동향으로 조사." 문제 정의만 주고(한 순간의 N장 사진, 한 뷰의 객체 마스크 → 다른 모든 뷰의 마스크; 보정 없음,
> 카테고리 없음, 환경 객체 포함, 반구 어안 리그) 우리 방법은 빼고 훑었습니다. 시간축(동영상)과 2023년 이전은 다루지
> 않습니다.
>
> **만든 방법.** 여섯 갈래(NeRF/3DGS 위 3D 객체 선택 / 자세 있는 3D 인스턴스 병합·대응 기반 전이 / 뷰=비디오 전파와
> SAM 프롬프트·기억 / 피드포워드 다중시점 모델·비보정 기하 사전 / 벤치마크·지표 / 2025~2026 arXiv 최신)를 병렬로
> 웹 검색 → 세 관점(연도별·인용 사슬·인접 분야)으로 보충 2라운드 → 모든 출처 URL의 제목·연도를 열어 검증(353편 중
> 1편 제외: 제목 불일치) → 종합. 검증된 출처 **352편**(가족별: NeRF/3DGS 151, 3D 인스턴스 73, 피드포워드 50,
> 대응·매칭 28, 벤치마크 15, 기하 사전 14, 프롬프트·기억 14, 뷰=비디오 7). 본문은 조사자(에이전트)의 종합이며 저는
> 구성·표기 규약을 검토했고 개별 수치는 원문 재확인 없이 실었습니다 — 논문에 인용하기 전에 원문을 볼 것.
> 노션 사본: 연구 › 연구메모 › 2026년 연구메모.


**표기 규약.** 본문 문장은 기본적으로 조사 목록에서 확인한 사실이다. 조사자의 해석·외삽은 문장 앞에 "추정:"을 붙여 구분한다. 수치는 각 논문이 보고했거나 다른 논문의 비교표에 재인용된 것만 쓴다. 벤치마크·프롬프트·평가 뷰 수가 논문마다 달라 표 간 직접 비교는 불가하며, 같은 표 안이라도 출처가 다른 재인용 수치는 같은 조건이 아닐 수 있다.

**약어 정리.** SAM(Segment Anything Model, 프롬프트형 2D 분할 기초 모델; SAM2는 비디오 기억 확장판, SAM3는 개념·예시 프롬프트 확장판), NeRF(Neural Radiance Field, 신경 방사장), 3DGS/2DGS(3D/2D Gaussian Splatting, 가우시안 스플래팅), SfM(Structure from Motion, 운동 기반 구조·자세 복원; COLMAP은 그 대표 도구), RGB-D(색+깊이 영상), IoU(Intersection over Union), mIoU(평균 IoU), mAcc(평균 픽셀 정확도), BIoU(경계 IoU), J/F(영역 IoU인 J와 경계 정확도 F), PQ(Panoptic Quality, 팬옵틱 품질; PQ^scene은 장면 단위 인스턴스 ID 일관성까지 재는 변형), AP(Average Precision), AUPRC(정밀도–재현율 곡선 아래 면적), mAA(누적 정확도 곡선 아래 면적), FPS(초당 프레임), FoV(시야각), GT(정답), ID(인스턴스 정체성 번호), DINOv2/DINOv3(자기지도 비전 특징 모델), CLIP(언어–이미지 대조 모델), VLM(비전–언어 모델), DUSt3R/MASt3R/MUSt3R(포인트맵 회귀형 기하 기초 모델 계열), VGGT(Visual Geometry Grounded Transformer), π³(순열 등변 기하 모델), DA3(Depth Anything 3), NVOS(Neural Volumetric Object Selection 벤치마크), LLFF(Local Light Field Fusion 데이터셋), LERF(Language Embedded Radiance Fields; LERF-Mask·LERF-OVS는 그 파생 벤치마크), OVS(개방어휘 분할), VOS(비디오 객체 분할), MVS(다중뷰 스테레오), LP(선형 계획), NBV(다음 최적 시점), TTT(테스트 시 학습), HDBSCAN/DBSCAN(밀도 기반 군집 알고리즘), XMem/DEVA/Cutie/SAM-Track/DeAOT(비디오 객체 분할 추적기 이름), LoFTR/LightGlue/RoMa(이미지 매처 이름).

---

## 0. 한눈에 보기

2023년 SAM과 NeRF가 만나면서 "한 뷰의 프롬프트 → 모든 뷰의 마스크"라는 과제가 NVOS·SPIn-NeRF 벤치마크 위에서 정식화됐다. 2024년에는 표현이 3DGS로 옮겨가 장면별 특징 필드 학습(20~60분)이 표준이 됐고, 2024~2026년에는 사전학습 3DGS 위에 마스크를 학습 없이 올리는 방법(FlashSplat 26초 → ArtisanGS 1~5초)이 NVOS mIoU를 94 근처까지 밀어 올려 포화시켰다. 같은 기간 두 축이 새로 생겼다. 첫째, DUSt3R·VGGT·π³ 같은 피드포워드 기하 기초 모델이 보정 없이 포인트맵·자세를 내주면서, 이를 SAM 계열에 주입하는 무보정 프롬프트형(MV-SAM, 3AM, G²TAM)과 세그먼트 단위 매칭형(SegMASt3R, MuViSeg)이 등장했다. 둘째, 뷰를 비디오로 보고 SAM2로 전파하는 방식이 넓은 시점차에서 어떻게 무너지는지가 정량화됐다. 그러나 이 문제의 조건(마스크 1뷰 · 보정 없음 · 학습 없음 · 벽·바닥 조각 같은 환경 객체 · 반구 어안 정적 리그)을 동시에 만족하는 방법도, 그것을 채점하는 공개 프로토콜도 아직 없다.

- **3D 표현 계열은 포화됐고 전부 포즈를 전제한다.** NVOS mIoU는 SA3D 90.3 → FlashSplat 91.8 → GaussianCut 92.5 → SAGOnline 92.7 → VCAR 93.5 → ArtisanGS 94.1로 올라왔고, SAGOnline 저자는 "최근 방법 간 격차가 2D 기초 모델 상한에 수렴"한다고 명시했다. 예외 없이 COLMAP 포즈로 만든 사전학습 3DGS/NeRF가 필요하다.
- **2025~2026년에 무보정 프롬프트형이 나타났다.** MV-SAM은 π³ 포인트맵을 SAM2.1 디코더에 주입해 NVOS 92.1(SAM2 비디오 전파 88.7), 3AM은 MUSt3R 특징을 SAM2 기억에 주입해 ScanNet++ IoU 90.6(SAM2Long 74.7), G²TAM은 무순서 정적 다중뷰 과제(PIST)를 명시적으로 정의해 InsTrack 74.3(SAM2 47.6)을 보고했다.
- **"뷰=비디오" 전파의 실패가 수치로 드러났다.** SegMASt3R 비교표에서 SAM2 비디오 전파의 세그먼트 대응 AUPRC는 시점차 0–45°에서 61.9, 135–180°에서 17.0으로 붕괴한다. Clutt3R-Seg는 희소 8뷰에서 MaskClustering 47.2, GraphSeg 30.5를 보고했고, SAGOnline은 광기선 전환 불가를 저자가 명시했다.
- **환경 객체는 어느 계열도 평가하지 않았다.** VoteSplat이 "FoV보다 큰 인스턴스(벽·바닥류)에서 2D 투표 부정확"을 명시한 것이 유일한 언급이고, 학습형 매칭·인스턴스 모델은 벽·바닥을 stuff 또는 dustbin으로 처리한다. 3D-OVS(배경어 wall·desktop 포함)와 ScanNet++(환경 클래스 인스턴스 래스터화)만 간접 검증 재료다.
- **리그 벤치마크는 MUVOD 하나이고 후속이 없다.** 17장면·카메라 9~46대·459 인스턴스/73 카테고리이지만, 공식 3D 벤치마크는 기준뷰 점/스트로크 프롬프트에 카메라 자세를 쓰고 어안 반구 4장면을 제외했으며, 조사 시점 Semantic Scholar·Google Scholar 기준 MUVOD를 인용한 논문은 0건이다. 어안은 Fisheye3R·RayTun3R가 기하만 다루고, 어안에 적응된 분할 모델은 없다.

---

## 1. 최근 3년의 흐름

| 시기 | 무엇이 가능해졌나 | 대표 | 뷰 간 일치의 근거 | 남은 병목 |
|---|---|---|---|---|
| 2023 | 한 뷰 스크리블/점을 장면별 NeRF에 리프팅해 전 뷰 렌더(SPIn-NeRF, ISRF, SA3D, SANeRF-HQ); COLMAP 희소 점군으로 SAM 점 프롬프트를 다른 뷰에 재투영(OR-NeRF, Obj-NeRF, NTO3D, 학습 없음); 점군+뷰별 SAM 마스크 병합으로 자동 3D 인스턴스(SAM3D, SAI3D, OpenMask3D, SAMPro3D); 참조 마스크 1장의 특징 매칭 분할(Matcher, PerSAM); 비디오 추적기를 다중시점에 적용(OSTRA) | SA3D, OR-NeRF, SAI3D, Matcher, DUSt3R | 밀도 필드 역렌더링, 희소 3D점 재투영, superpoint 공존 빈도, DINOv2/SAM 특징 최근접 | 장면별 NeRF 수십 분, COLMAP 포즈 필수, SAM 자기 프롬프팅이 뷰 순서 의존, OSTRA가 "정렬 안 된 뷰에서 추적 실패" 보고 |
| 2024 | 3DGS 특징 필드(SAGA, Gaussian Grouping, OmniSeg3D, GARField, Click-Gaussian, 20~60분)와 학습 없는 리프팅(FlashSplat 26초, SAGD, GaussianCut); 참조 뷰 마스크의 깊이 워핑 전이(View-consistent Object Removal, SPIn-NeRF IoU 94.27); 뷰 합의율 마스크 그래프(MaskClustering); 뷰=비디오 전처리(Gaussian Grouping의 DEVA); 시점 변화에서 특징 대응 붕괴 정량화(Probing 3D Awareness); 밀집 매처(RoMa, MASt3R) | FlashSplat, GaussianCut, MaskClustering, View-consistent Object Removal, MASt3R | 가우시안 기여도 역스플래팅과 대조 리프팅, 깊이 투영, 뷰 합의율, 밀집 매칭 | 여전히 포즈+3DGS; 넓은 기선에서 DEVA·의미 대응 실패; 환경 객체 미평가 |
| 2025 | 사전학습 3DGS 위 학습 없는 분할 표준화(LUDVIG, iSegMan, SAGOnline, COB-GS, Trace3D); 무보정 입력 처리(WildSeg3D의 MASt3R 정렬, MASt3R-SfM, VGGT, π³, MUSt3R, DA3, MapAnything); 세그먼트 매칭 학습(SegMASt3R, 135–180° AUPRC 83.6); 비자세 피드포워드 파놉틱(PanSt3R); 교차뷰 마스크 대응(O-MaMa, DOMR, V²-SAM); MUVOD 리그 벤치마크 공개; SAM 3 개념·예시 프롬프트 | LUDVIG, SAGOnline, WildSeg3D, SegMASt3R, PanSt3R, VGGT, MUVOD, SAM 3 | 렌더 궤적 SAM2 추적, 에피폴라 전파, 포인트맵 정렬, 최적수송 세그먼트 매칭 | 무보정 재구성 품질이 병목(Z3D 이미지만 설정 Acc@0.5 54.2→12.9); 학습형은 실내·Ego-Exo 편향; 리그 미평가 |
| 2026 | 포인트맵을 SAM/SAM2에 주입한 무보정 프롬프트형(MV-SAM, 3AM, G²TAM); N뷰 동시 세그먼트 매칭(MuViSeg); VGGT 위 인스턴스 헤드(SegVGGT, EPS3D, InstanceSplat, GroupForward, Scenes-as-Objects); 어안 적응 기하(Fisheye3R, RayTun3R); 경계 지표 도입(NG-GS B-mIoU, BEA-GS BIoU); 3DGS 가상 궤도 SAM2 추적(Seed2GS, SAGO); 구면 보조 뷰 투표(VCAR); 1~5초 대화형(ArtisanGS) | MV-SAM, 3AM, G²TAM, MuViSeg, SegVGGT, Fisheye3R, NG-GS, Seed2GS, VCAR, ArtisanGS | 3D 점 임베딩 교차주의, 기하 기억, 순열 등변 토큰, 다중뷰 어텐션 | 마스크 1뷰·무순서 리그·환경 객체·어안을 동시에 만족하는 방법 없음; MUVOD 후속 인용 없음 |

---

## 2. 접근 축 분류

뷰 간 일치를 만드는 "근거"를 기준으로 가족을 재배열하면 다음과 같다. 원래의 가족 코드(B: NVOS·필드, C: 자세 있는 3D 인스턴스, D: 뷰=비디오, E: 대응·매칭, F: 피드포워드, G: 기하 사전, H: 프롬프트·기억)와 대응시켜 표기했다.

| 가족(근거) | 대표 | 입력 가정(보정 · 장면별 최적화 · 프롬프트 형식) | 학습 | 비용 | 주된 실패 |
|---|---|---|---|---|---|
| 3D 표현 최적화(B): 장면별 NeRF/3DGS에 라벨을 올려 렌더 | SA3D, SAGA, Gaussian Grouping, FlashSplat, LUDVIG, GaussianCut, SAGOnline, VCAR, ArtisanGS, Seed2GS | COLMAP 포즈 + 사전학습 3DGS 필수; 특징 필드형은 장면별 20~60분 최적화, 리프팅형은 최적화 없음; 프롬프트는 점/스크리블/마스크(대부분 1뷰) | 장면별 최적화(특징 필드형) 또는 없음(리프팅형) | 3DGS 학습 별도(GaussianCut 기준 ~8.5분) + 분할 0.4초~수 분 | 3DGS 품질이 상한; SAM 자기 프롬프팅의 뷰 순서 의존; 볼록 객체 가정(VCAR); 환경 객체 미검증 |
| 기하 투영(C+G): 추정·측정 깊이로 마스크를 3D에 올려 재투영·병합 | OR-NeRF, View-consistent Object Removal, MaskClustering, SAMPro3D, CCGS, GraphSeg, MV3DIS, Z3D, DivAS, CDSeg | 대부분 RGB-D·점군·자세 필요; CCGS·GraphSeg·Z3D·WildSeg3D는 DUSt3R/MASt3R로 자세 대체; 프롬프트는 자동(전체 분할)이 대부분, OR-NeRF·CDSeg·DivAS는 1뷰 점/마스크 | 없음(대부분) | OR-NeRF 뷰당 ~0.5초, MaskClustering 56초/장면(Z3D 보고), CDSeg 수 초 | 깊이·자세 오차가 프롬프트를 어긋나게 함; 희소 뷰에서 합의 불안정(Clutt3R-Seg); 가림 구멍 |
| 외형 연속성(D+H 일부): 뷰 열을 비디오로 보고 SAM2 기억 전파 | WildSeg3D, OSTRA, SAP, Seed2GS·SAGOnline의 렌더 궤적, Re-Prompting SAM 3, Correspondence as Video | 자세 불필요(WildSeg3D는 MASt3R 정렬); 카메라를 인접 순서로 정렬해야 함; 프롬프트는 첫 프레임 마스크 | 없음 | SAM2 추론(실시간급) | 샷 전환·넓은 기선에서 붕괴(SegMASt3R 표 SAM2 17.0, 3AM, SAAS); 가림 후 ID 오류 누적(SAM2-Splat) |
| 대응·매칭(E): 학습 없는 특징 대응으로 프롬프트 재생성 | Matcher, PerSAM, RoMa, LightGlue, HOMER, INSID3, REBASE, FoRIS, MESA | 보정 불필요; 참조 이미지+마스크 1장; 쌍 단위 | 없음(매처 자체는 사전학습) | 이미지당 매칭 1회 + SAM 1회 | 큰 시점차에서 SAM·확산 특징 급락(Probing 3D Awareness); 동일 카테고리 다중 인스턴스 혼동; 무질감 벽·바닥에 키포인트 없음; 좌우 대칭 혼동(GeoAware-SC) |
| 학습된 대응(F): 기하 기초 모델 위에 대응·분할 헤드를 학습 | SegMASt3R, MuViSeg, MV-SAM, 3AM, G²TAM, PanSt3R, SegVGGT, EPS3D, V²-SAM, O-MaMa, DOMR, VGGT-Segmentor | 비자세·비보정 입력; 장면별 최적화 없음; 프롬프트는 마스크(3AM, V²-SAM, O-MaMa)·클릭(MV-SAM, G²TAM)·없음(PanSt3R, SegVGGT, EPS3D) | 있음(추론만 사용 가능; ScanNet++·Ego-Exo4D 편향) | 단일 순전파(MV-SAM 20프레임 1.1초+전처리 5.1초, PanSt3R ~4분, G²TAM 21.6 FPS) | 실내·소품 도메인 편향; 환경 객체는 stuff/dustbin; 어안 미평가; 절대 척도 없음 |
| 검출–재식별(H): 참조 마스크를 예시로 각 뷰에서 검출 후 연결 | SAM 3, InsDet 기초선, T-Rex2, FoundYou, Explicit Memory(3DGS 기억) | 보정 불필요; 예시 박스/마스크 | 없음(SAM 3 추론) 또는 경량 학습 | 뷰당 검출 1회 | 같은 개념의 다른 인스턴스를 모두 잡음(인스턴스 구분 별도); 벽·바닥 조각은 개념 프롬프트가 모호 |

추정: 이 문제에서 "바로 사용" 후보는 학습된 대응(F)의 추론 전용 모델과 대응·매칭(E)뿐이고, 나머지는 기하 기초 모델(G)로 포즈·깊이·3DGS를 먼저 만들어 주는 브리지가 있어야 동작한다.

---

## 3. 가족별 절

### 3.1 3D 표현 최적화 계열(B: NVOS·필드)

**문제의식.** 뷰별 2D 분할은 뷰마다 다르게 나오므로, 뷰 간 일관성을 3D 표현에 강제하고 그 표현을 렌더링해 마스크를 얻자는 것이다. SPIn-NeRF(CVPR 2023)가 "한 뷰 희소 스크리블 → 전 뷰 마스크" 벤치마크를 정의했고, SA3D(NeurIPS 2023)가 SAM을 여기에 결합했다.

**핵심 아이디어.** 세 세대로 나뉜다. (i) NeRF 세대(2023): SA3D는 2D 마스크를 밀도 가중 3D 격자로 역렌더링하고, 렌더 마스크에서 새 뷰 점 프롬프트를 뽑아 SAM을 다시 호출하는 "교차뷰 자기 프롬프팅"을 교대로 수행한다. ISRF는 DINO 특징 증류 후 영역 성장, SANeRF-HQ는 SAM 특징 필드 증류(~666초)와 밀도·RGB 유사도로 객체 필드를 학습한다. (ii) 3DGS 특징 필드 세대(2024): SAGA·OmniSeg3D·Click-Gaussian·Contrastive Gaussian Clustering은 뷰별 SAM 마스크로 가우시안 특징을 대조 학습하고, Gaussian Grouping은 뷰를 비디오로 보고 DEVA로 ID를 연결한 뒤 identity 인코딩을 학습하며, GARField는 SAM 마스크의 3D 스케일을 조건으로 한 친화 필드를 학습한다. (iii) 학습 없는 리프팅 세대(2024~2026): 사전학습 3DGS 위에서 마스크를 가우시안 기여도로 역스플래팅해 라벨을 결정한다 — FlashSplat은 LP로 전역 최적 할당(26초, 8 GB), GaussianCut은 가우시안 그래프컷(~89초), LUDVIG는 참조 마스크 역스플래팅 후 DINOv2 유사도 그래프 확산, SAGD는 경계 가우시안 분해, iSegMan은 클릭을 에피폴라 선으로 타 뷰에 전파하고 가시성 투표, SAGOnline은 연속 렌더 궤적을 SAM2로 추적해 래스터화 합의(27 ms/프레임), DivAS는 렌더 깊이 가중 복셀 융합(<70 ms/뷰), ArtisanGS는 자동 생성 ~50개 렌더 뷰에 Cutie로 마스크 전파(1~5초), VCAR는 객체 중심 구면 나선으로 보조 뷰를 만들어 가시성 가중 투표와 이방성 경계 정제, SAGO(가상 드론)는 분할을 온라인 NBV 계획으로 재정식화(0.4~0.9초), Seed2GS는 참조 카메라에서 출발하는 가상 궤도 두 클립을 SAM2로 추적(9.3초), CrashSplat은 단일 뷰 2D 마스크 안 가우시안을 깊이 순 Z-버퍼 통계로 선택한다(인스턴스당 <0.31초). 경계 정제 후처리로 GaussianTrimmer(가상 카메라 SAM2 역알파블렌딩)와 NG-GS(마스크 분산으로 경계 가우시안을 찾아 연속 특징 필드로 보정, ~9분)가 있다.

**대표 논문.** SPIn-NeRF(CVPR 2023, https://arxiv.org/abs/2211.12254), SA3D(NeurIPS 2023, https://arxiv.org/abs/2304.12308), SANeRF-HQ(CVPR 2024, https://arxiv.org/abs/2312.01531), Gaussian Grouping(ECCV 2024, https://arxiv.org/abs/2312.00732), SAGA(AAAI 2025, https://arxiv.org/abs/2312.00860), OmniSeg3D(CVPR 2024, https://arxiv.org/abs/2311.11666), FlashSplat(ECCV 2024, https://arxiv.org/abs/2409.08270), GaussianCut(arXiv 2024, https://arxiv.org/abs/2411.07555), LUDVIG(ICCV 2025, https://arxiv.org/abs/2410.14462), iSegMan(CVPR 2025, https://arxiv.org/abs/2505.11934), SAGOnline(arXiv 2025, https://arxiv.org/abs/2508.08219), VCAR(ACM MM 2026, https://arxiv.org/abs/2608.30870), ArtisanGS(arXiv 2026, https://arxiv.org/abs/2602.10173), Seed2GS(arXiv 2026, https://arxiv.org/abs/2608.11928), NG-GS(CVPR 2026 Highlight, https://arxiv.org/abs/2604.14706).

**대표 수치(같은 벤치마크 이름 안에서만 비교; 재인용 출처가 달라 프로토콜 차이 있음).**

| 벤치마크 | 수치(mIoU 또는 IoU / mAcc) |
|---|---|
| NVOS(LLFF 8장면, 스크리블 1뷰→목표뷰 1장) | ISRF 83.8/96.4; SA3D 90.3/98.2; SAGD 90.4/98.2(GaussianCut 표에서는 72.1); Gaussian Grouping 90.6; SAGA 90.9~92.6; OmniSeg3D 91.7/98.4; FlashSplat 91.8/98.6; iSegMan 92.0/98.4; COB-GS 92.1/98.6; LUDVIG 91.3(SAM2)~92.4; GaussianCut 92.5/98.4; Trace3D 92.5/98.6; NG-GS 92.6/99.2(B-mIoU 84.7); SAGOnline 92.7/98.7; SAGO 92.7/98.7; VCAR 93.5/98.6; ArtisanGS 94.1/98.8; Gradient conflict(SIVP 2026) 95.6 |
| SPIn-NeRF(10장면, 1뷰 프롬프트→전 뷰) | ISRF 71.5; SPIn-NeRF MVSeg 90.4~91.0(IoU 91.66/Acc 98.91로도 인용); Gaussian Grouping 88.4; SAGD 90.0/98.7; SA3D 91.9~92.4; iSegMan 92.4/99.1; Point'n Move IoU 92.71/Acc 99.51; GaussianCut 92.9/99.2; LUDVIG 93.8; Click-Gaussian 94.0; OmniSeg3D 94.3~95.2; SAGOnline 95.2/99.3; SAGO 92.5/99.3; CrashSplat(5장면) 79.9/96.9 |
| LERF-Mask(3장면, 텍스트→마스크, mIoU/mBIoU) | Gaussian Grouping 72.8/67.6; Gradient-Weighted Back-Projection 73.4; Gaga 74.7/72.2; OmniSeg3D 74.7/71.8; FlashSplat 76.5; Unified-Lift 80.9/77.1; InstaScene 85.6; Binary-Gaussian 87.1; ObjectGS 88.4; OMEGAS 88.86; Click-Gaussian 89.1; SAGO 91.0; Seed2GS 92.1 |
| 경계 지표 | NG-GS NVOS B-mIoU 84.7(COB-GS 79.1); BEA-GS Mip-NeRF360 BIoU 85.8/IoU 92.0, LERF BIoU 83.6; GaussianTrimmer NVOS SAGA 90.9→92.1, COB-GS 92.1→92.5 |
| 비용 | SAGA 20~40분·15 GB; Gaussian Grouping ~37분·34 GB; GARField 45분; FlashSplat 26초·8 GB; GaussianCut ~89초; VCAR ~30초; ArtisanGS 1~5초; SAGO 0.4~0.9초; Seed2GS 9.3초; NG-GS ~9분; COB-GS 0.46분 |

**실패 양상.** 3DGS 재구성 품질이 곧 분할 상한이라는 점을 LUDVIG·SANeRF-HQ가 명시한다. SA3D식 자기 프롬프팅은 뷰 순서대로 진행돼 광기선 전환에서 프롬프트가 엉뚱한 객체로 튀고, SAGOnline은 Stage I이 연속 카메라 궤적 전제라 광기선 전환이 불가함을 저자가 명시했다. VCAR의 구면 샘플링은 볼록 객체 전제라 케이블·기둥 같은 비볼록 객체에서 실패한다. VoteSplat은 FoV보다 큰 인스턴스에서 2D 투표가 부정확하다고 명시했고, SAGD는 표마다 수치 편차(72.1 vs 90.4)가 커 재현이 불안정하다. Gaussian Grouping의 DEVA 추적은 광기선·희소 뷰에서 ID가 뒤섞이며(Gaga의 동기), ArtisanGS 저자는 NVOS 벤치마크 자체가 클릭 샘플링 차이로 잡음이 크다고 지적했다.

**이 문제 조건에서의 적용 가능성.** 그대로는 전부 불가다 — 포즈와 사전학습 3DGS가 예외 없이 필요하고(Seed2GS의 "camera-free"도 원본 촬영 카메라가 필요 없다는 뜻이지 보정 불요가 아니다), 벤치마크는 수십~수백 뷰 연속 궤적이다. 추정: 개조 경로는 VGGT/MASt3R-SfM → 3DGS 초기화 → 학습 없는 리프팅이며, 주어진 마스크를 그대로 가우시안에 역스플래팅하는 FlashSplat·LUDVIG·GaussianCut(단항항)·CrashSplat·ArtisanGS가 환경 객체 보존에 유리하고, SAM 재프롬프팅에 의존하는 SA3D·iSegMan·SAGO·VCAR는 벽 조각 같은 비객체 영역을 재해석할 위험이 있다. 추정: 3DGS가 있으면 광기선 카메라 사이를 가상 뷰로 메워 SAM2 추적을 가능하게 하는 SAGOnline·Seed2GS·ArtisanGS의 "렌더 궤적" 구조가 리그 문제의 핵심 우회로가 될 수 있다. 어안 반구 리그에서 3DGS 재구성 자체는 미검증이며, MUVOD도 어안 4장면을 3D 벤치마크에서 제외했다.

### 3.2 기하 투영 계열(C: 자세 있는 3D 인스턴스 · 마스크 병합)

**문제의식.** 점군·깊이·자세가 있을 때 뷰별 SAM 마스크를 어떻게 하나의 3D 인스턴스로 병합하고, 그것을 다시 각 뷰로 투영할 것인가.

**핵심 아이디어.** 2023~2024년에는 superpoint(기하 프리미티브)에 여러 뷰 SAM 마스크가 함께 덮는 빈도(SAI3D의 affinity), 인접 프레임 양방향 병합(SAM3D), 3D 마스크를 뷰로 투영해 상위 k 가시 뷰에서 정제(OpenMask3D), 3D 점 프롬프트를 모든 뷰에 투영해 SAM을 호출하는 프롬프트 정렬(SAMPro3D), 뷰 합의율(두 마스크를 함께 포함하는 다른 뷰 마스크 비율)을 간선으로 한 마스크 그래프 군집(MaskClustering)이 확립됐다. 2025~2026년에는 SAM2 추적으로 뷰 일관 마스크를 먼저 만들고 저품질 마스크를 걸러 병합(Any3DIS, SAM2Object, CDIS, OpenTrack3D, Details Matter), 3D 앵커 투영 → 뷰 간 친화도 그래프 → 연결 성분 + 다수결(SAM-Zero3D), 복셀 해싱 겹침 조회(OnlineAnySeg), 거친 3D 세그먼트를 공통 참조로 뷰별 마스크를 매칭(MV3DIS), 리프팅 후 split-then-grow 정제(SGS-3D)로 병합 규칙이 명시화됐다. 무보정 쪽에서는 CCGS가 DUSt3R 포인트맵의 유클리드 거리로 픽셀 대응을 정의해 부분 매칭 허용 헝가리안으로 마스크 ID를 연관하고, GraphSeg가 DUSt3R 점군 위 2D/3D 이중 그래프 수축으로 3~5뷰를 처리하며, Z3D가 DUSt3R+MaskClustering으로 이미지만 받는다. "1뷰 프롬프트 → 전 뷰"에 가장 가까운 것은 OR-NeRF(마스크 안 점을 COLMAP 희소 3D점으로 올려 각 뷰에 재투영해 SAM 점 프롬프트, 뷰당 ~0.5초), View-consistent Object Removal(참조 마스크를 단안 깊이로 3D에 올려 각 뷰로 워핑, SAM 재실행 없음), CDSeg(카메라 근접 순 SAM2 전파 + 렌더링 중 픽셀–프리미티브 연관 기록 + 가시성 투표), DivAS(깊이 가중 복셀 융합)다.

**대표 논문.** SAI3D(CVPR 2024, https://arxiv.org/abs/2312.11557), SAMPro3D(3DV 2025, https://arxiv.org/abs/2311.17707), MaskClustering(CVPR 2024, https://arxiv.org/abs/2401.07745), OR-NeRF(arXiv 2023, https://arxiv.org/abs/2305.10503), View-consistent Object Removal(ACM MM 2024, https://arxiv.org/abs/2408.02100), CCGS(IJCV 2025, https://arxiv.org/abs/2502.16303), GraphSeg(IROS 2026, https://arxiv.org/abs/2504.03129), Clutt3R-Seg(ICRA 2026, https://arxiv.org/abs/2602.11660), MV3DIS(arXiv 2026, https://arxiv.org/abs/2604.08916), Z3D(ACL 2026, https://arxiv.org/abs/2602.03361), CDSeg(arXiv 2026, https://arxiv.org/abs/2608.05482), SAM-Zero3D(IEEE TCSVT 2026, https://doi.org/10.1109/tcsvt.2026.3666899).

**대표 수치.** OR-NeRF는 SPIn-NeRF 데이터셋 마스크 IoU 95.42 / Acc 99.71(SPIn-NeRF MVSeg 91.66 / 98.91), View-consistent Object Removal은 같은 셋 IoU 94.27 / Acc 99.48 — 자세 있는 조건에서 "1뷰 마스크 → 전 뷰 워핑"이 장면별 NeRF 학습 방법보다 높다. MaskClustering은 ScanNet200 제로샷 AP 12.0 / AP50 23.3, MV3DIS는 ScanNet200 클래스 무관 AP50 54.7 / AP25 69.7, Stream3Dv2는 ScanNet200 클래스 무관 AP 27.1(MaskClustering 19.7). CCGS는 Replica mIoU_3D 65.46(Gaussian Grouping 54.12). GraphSeg는 GraspNet-1B IoU 0.5945/0.6522/0.7308. CDSeg는 DesktopObjects-360 mIoU 92.35, NeRDS-360 95.89, ScanNet-v2 65.77. Z3D는 ScanRefer Acc@0.5 46.0(제로샷 SOTA)이지만 이미지만 입력하는 설정에서 54.2→12.9로 급락하고 MaskClustering이 장면당 56초 걸린다. Clutt3R-Seg의 희소 8뷰 표에서 Clutt3R-Seg 74.3 vs MaskClustering 47.2 vs GraphSeg 30.5. Joint 2D-3D(ICPR 2026)는 거리 건물 30개에서 adjusted coverage 0.841 vs SAM2+MOTRv2 0.606. SpaCeFormer는 다중뷰 군집 마스크 recall@IoU0.5 54.3% / precision 33.6% vs 단일뷰 리프팅 2.5% / 4.8%.

**실패 양상.** 깊이·자세 오차가 프롬프트 투영을 어긋나게 하고(SAMPro3D), 희소 뷰에서 관측자 집합이 작아 뷰 합의가 불안정하며(Clutt3R-Seg의 MaskClustering 분석), 고정 임계로 과·저분할(GraphSeg)한다. 3DIML은 극단적 시점 변화에서 실패를 명시했고, CCGS는 강한 가림·급격한 시점 변화에서 저하한다. SpaCeFormer의 합의율 필터(>0.9)는 소수 뷰에만 보이는 객체를 버린다. SAI3D 계열은 SAM 부품 과분할이 3D로 전이되고 큰 평면은 프리미티브 경계가 어긋난다(SA3DIP 지적). Z3D의 급락은 무보정 재구성 품질이 병목임을 보여준다.

**적용 가능성.** 대부분 RGB-D·점군·자세를 가정하고 자동 전체 분할이라 1뷰 마스크 인터페이스가 없다. 추정: VGGT/π³ 포인트맵으로 깊이·자세를 대체하면 OR-NeRF의 점 재투영, View-consistent Object Removal의 깊이 워핑, SAM-Zero3D의 연결 성분+다수결, MaskClustering의 뷰 합의율을 "주어진 마스크와 겹치는 3D 영역 선택 → 재투영"으로 재구성할 수 있으며, 이것이 보정·학습 없이 환경 객체까지 그대로 옮기는 가장 직접적인 경로다. GraphSeg는 무보정·학습 없음·희소 뷰를 모두 만족하는 드문 사례지만 탁상 3~5뷰만 검증됐고, Aerial Lifting의 첫 단계(투영 + intersection-over-minimum ≥0.5 병합)도 NeRF 없이 분리 사용 가능하다. 어안은 이 계열 어디에도 없다.

### 3.3 외형 연속성 계열(D: 뷰=비디오 전파)

**문제의식.** 카메라들을 인접 순서로 늘어놓으면 비디오처럼 보이니, 첫 프레임 마스크를 VOS 추적기로 전파하면 되지 않는가.

**핵심 아이디어.** OSTRA(2023)는 SfM 정렬 시퀀스에 XMem/DeAOT를 적용했고, Gaussian Grouping은 DEVA를 전처리로 썼다. WildSeg3D(ICCV 2025)는 MASt3R 포인트맵을 Dynamic Global Aligning으로 정렬하고 한 뷰 SAM2 마스크를 프롬프트로 다른 뷰까지 추적해 마스크 캐시에 저장한다. SAGOnline·Seed2GS·ArtisanGS는 3DGS 렌더 궤적 위에서 SAM2/Cutie를 돌려 광기선 문제를 우회한다. SAP는 4K 파노라마를 구면 경로의 원근 패치 열로 잘라 "고정 궤적 비디오"로 SAM2에 넣는다. Correspondence as Video는 참조→목표 쌍 사이를 확산 기반 의미 전이 의사 비디오로 채운다.

**대표 논문.** OSTRA(arXiv 2023, https://arxiv.org/abs/2308.06974), WildSeg3D(ICCV 2025, https://arxiv.org/abs/2503.08407), SAM2Object(CVPR 2025, https://doi.org/10.1109/cvpr52734.2025.01800), SplatXtRact(IEEE RA-L 2026, https://doi.org/10.1109/lra.2026.3699257), SAP(arXiv 2026, https://arxiv.org/abs/2603.12759).

**대표 수치.** WildSeg3D는 기존 SOTA 대비 40배 속도로 SOTA 정확도 유지를 주장하며, 조사 메모는 NVOS 94.1을 기록했다(원문 초록에는 수치가 없어 재확인 필요). SAP는 4K 파노라마에서 SAM2 대비 제로샷 mIoU +17.2. SplatXtRact는 TUM RGB-D에서 COLMAP 기반 대비 약 15배 빠름(93초 vs 1433초). 실패 크기는 다른 가족이 실측했다: SegMASt3R 표에서 SAM2 비디오 전파의 세그먼트 대응 AUPRC는 시점차 0–45/45–90/90–135/135–180°에서 61.9/46.6/27.9/17.0; MV-SAM은 SAM2-Video가 ScanNet++ 46.1, DL3DV 67.3 mIoU(MV-SAM 자체 대비 NVOS 88.7 vs 92.1, SPIn-NeRF 86.6 vs 92.9); 3AM 표에서 SAM2Long ScanNet++ IoU 74.7 / 추적 재현율 41.3.

**실패 양상.** OSTRA는 조명·가림·블러로 추적 오류 시 대화형 수정이 필요하고 카메라 순서 없는 리그에 부적합하다고 명시했다. SAAS(AAAI 2026)는 기존 VOS가 샷 불연속에서 붕괴함을 정량화했으며, 카메라 간 점프는 샷 전환과 동형이다. SAM2-Splat(RA-L 2025)은 가림 후 재등장 시 ID 오류 누적을 3D 기억으로 교정하는 것이 목적이다. SAMURAI식 칼만 운동 사전은 카메라 간에 운동 연속성이 없어 오히려 해롭다(추정).

**적용 가능성.** WildSeg3D가 무보정·무학습·마스크 캐시라는 점에서 조건에 가장 가깝지만, 환경 객체와 46대 넓은 기선에서 SAM2 추적·MASt3R 정렬은 검증되지 않았다. 추정: 리그 카메라를 공간 인접 순(카메라 그래프 최단 경로나 뱀형 순회)으로 정렬하고, 반구 리그는 SAP처럼 매끄러운 궤적으로 재구성하는 순서 설계가 전제이며, 3AM·G²TAM처럼 외형 기억을 기하 기억으로 바꾸거나 Re-Prompting SAM 3처럼 앵커를 재주입하지 않으면 시점차 90° 이상에서 실패할 가능성이 크다.

### 3.4 대응·매칭 계열(E: 학습 없는 특징 대응)

**문제의식.** 참조 이미지 1장과 마스크만 주고, 기초 모델 특징 대응으로 다른 이미지의 같은 것을 분할하자. 이 문제의 입력 형식(마스크 1뷰·보정 없음·학습 없음·카테고리 없음)과 정확히 같다.

**핵심 아이디어.** Matcher(ICLR 2024)는 참조 마스크 패치와 질의 패치를 DINOv2로 양방향 매칭해 점/박스 프롬프트로 SAM을 호출하고 제안을 강건 선택한다. PerSAM은 SAM 특징 코사인 유사도로 양·음 점을 만든다. 후속으로 Bridge the Points(점–마스크 그래프로 프롬프트 선택), No time to train(기억 은행 + 의미 대응), FoRIS(전경 정화 → 국소화 → 통합), INSID3(DINOv3 특징만으로 SAM 없이 분할), REBASE(참조 배경 부분공간을 제거해 같은 장면 배경 공유 문제를 직접 다룸)가 이어졌다. 기하 쪽 부품으로 밀집 매처 RoMa(참조 마스크를 워프+신뢰도로 직접 옮김), 희소 매처 LightGlue, 크로스뷰 완성 모델의 어텐션을 대응으로 쓰는 ZeroCo, SAM 영역 그래프 매칭 MESA, 스테레오 다각형 매칭이 있다. HOMER는 LoFTR 매칭 → 호모그래피로 마스크 워핑 → 질량중심 앵커로 SAM 재프롬프트를 자세·학습 없이 수행한다.

**대표 논문.** Matcher(ICLR 2024, https://arxiv.org/abs/2305.13310), PerSAM(arXiv 2023, https://arxiv.org/abs/2305.03048), Probing the 3D Awareness of Visual Foundation Models(CVPR 2024, https://arxiv.org/abs/2404.08636), RoMa(CVPR 2024, https://arxiv.org/abs/2305.15404), Telling Left from Right(CVPR 2024, https://arxiv.org/abs/2311.17034), HOMER(arXiv 2025, https://arxiv.org/abs/2501.17636), INSID3(CVPR 2026, https://arxiv.org/abs/2603.28480), REBASE(arXiv 2026, https://arxiv.org/abs/2607.09082), FoRIS(arXiv 2026, https://arxiv.org/abs/2609.03384).

**대표 수치(모두 단일 시점 few-shot 벤치마크이며 다중시점 마스크 전이 수치가 아님).** Matcher COCO-20i 52.7 mIoU / LVIS-92i 33.0; SEGIC COCO-20i one-shot 65.3; INSID3 이전 대비 +7.5 mIoU; FoRIS 1-shot +4.5 / 5-shot +4.8 mIoU; GeoAware-SC SPair-71k PCK@0.10 제로샷 65.4; DIFT SPair-71k에서 DINO 대비 +19; RoMa WxBS(극단 넓은 기선) +36%; ZeroCo HPatches-240 AEPE 9.41. 시점 변화 관련 유일한 정량 보고는 Probing 3D Awareness로, 뷰 쌍의 회전 각도가 커질수록 대응 재현율이 급락하고 Stable Diffusion·SAM 특징은 소각도 상위 → 대각도 최하위, DINOv2·DeiT가 상대적으로 강건하다. SegMASt3R 표에서 DINOv2 세그먼트 대응 AUPRC는 0–45°에서 57.9, 135–180°에서 32.4, RoMa는 61.6 → 30.0, MASt3R 키포인트는 59.5 → 45.4다.

**실패 양상.** 동일 카테고리 인스턴스가 여러 개면 의미 특징이 구분하지 못한다(Matcher, No time to train). 좌우 대칭 혼동(GeoAware-SC). 희소 매처는 무질감 벽·바닥에 키포인트가 없어 환경 객체 전이에 취약하고 넓은 기선에서 매칭 수가 급감한다(LightGlue). HOMER는 평면 호모그래피 가정과 연속 워핑 오차 누적이 있으며 마스크 IoU 직접 비교가 없다. DINOv3를 쓸 때 층 선택이 성능을 좌우한다(Revealing the Semantic Selection Gap). 어안 원본·환경 객체 전이는 어느 논문도 평가하지 않았다.

**적용 가능성.** Matcher·PerSAM·INSID3·FoRIS·REBASE는 형식상 바로 사용 가능하고, REBASE는 같은 장면 다중 카메라의 "배경 공유" 조건을 정면으로 다룬다. 추정: 그러나 리그처럼 시점이 90° 이상 도는 경우 의미 대응만으로는 인스턴스 동일성이 보장되지 않으므로, 기하 기초 모델의 포인트맵 대응으로 후보를 제한하고(SegMASt3R 표에서 MASt3R 키포인트가 대각도에서 DINOv2·RoMa보다 강함) 의미 대응은 재순위에만 쓰는 조합이 필요하다. "키포인트 매칭 + SAM 프롬프트로 다중뷰 마스크 전이"를 주제로 한 독립 논문은 확인되지 않았다.

### 3.5 학습된 대응 계열(F: 피드포워드 다중뷰 모델)

**문제의식.** 장면별 최적화 없이, 동결된 기하 기초 모델 위에 경량 헤드를 학습해 비자세 다중뷰에서 대응·인스턴스·마스크를 한 번에 내자.

**핵심 아이디어.** 세 줄기다. (1) 세그먼트 매칭: SegMASt3R(NeurIPS 2025 Spotlight)는 동결 MASt3R 쌍 디코더 특징 위 세그먼트–특징 헤드를 ScanNet++로 학습(22시간)하고 dustbin 포함 최적수송으로 두 뷰 세그먼트를 일대일 대응시킨다(0.58초/쌍). MuViSeg(2026)는 MASt3R 또는 VGGT 특징에서 세그먼트 기술자를 풀링해 LightGlue식 어텐션 헤드로 N뷰 세그먼트를 동시에 매칭한다. (2) 프롬프트형: MV-SAM은 동결 π³ 포인트맵으로 이미지·프롬프트를 3D로 올리고 동결 SAM2.1 인코더 특징을 3D 점 임베딩으로 바꿔 전 뷰 마스크를 한 번에 디코딩한다(학습 파라미터 4.1M, SA-1B 단일 뷰로만 훈련). 3AM은 MUSt3R 3D 특징을 경량 Feature Merger로 SAM2 메모리에 주입해 외형 기억 대신 기하 기억으로 동일성을 유지한다. G²TAM(ICML 2026)은 π³ 기반 공간 인코더를 암묵 기억으로 삼아 무순서 다중뷰 PIST(Promptable Instance Spatial Tracking) 과제를 정의한다. (3) 자동 인스턴스/파놉틱: PanSt3R(MUSt3R 위 쿼리 디코더, ~4분), SegVGGT(VGGT 토큰과 객체 쿼리, 닫힌 집합 실내), EPS3D(VGGT 위 의미·인스턴스 동시, 열린 어휘), IGGT·UNITE·Scenes-as-Objects·FAST3DIS·InstanceSplat·GroupForward. 별도로 Ego-Exo 교차뷰 줄기(ObjectRelator → O-MaMa → DOMR → V²-SAM → VGGT-Segmentor → Cycle-Consistent Mask Prediction)는 "질의 뷰 마스크 → 타깃 뷰 마스크"라는 입출력이 이 문제와 같으며, V²-SAM의 Anchor 전문가(DINOv3 대응 → SAM2 점 프롬프트)는 무학습이다.

**대표 논문.** SegMASt3R(NeurIPS 2025 Spotlight, https://arxiv.org/abs/2510.05051), MuViSeg(arXiv 2026, https://arxiv.org/abs/2607.17938), MV-SAM(arXiv 2026, https://arxiv.org/abs/2601.17866), 3AM(arXiv 2026, https://arxiv.org/abs/2601.08831), G²TAM(ICML 2026, https://arxiv.org/abs/2607.03789), PanSt3R(ICCV 2025, https://arxiv.org/abs/2506.21348), SegVGGT(arXiv 2026, https://arxiv.org/abs/2603.19926), EPS3D(ICML 2026, https://arxiv.org/abs/2606.08980), O-MaMa(ICCV 2025, https://arxiv.org/abs/2506.06026), V²-SAM(arXiv 2025, https://arxiv.org/abs/2511.20886), VGGT-Segmentor(arXiv 2026, https://arxiv.org/abs/2604.13596), TrianguLang(arXiv 2026, https://arxiv.org/abs/2603.08096).

**대표 수치.**

| 벤치마크 | 수치 |
|---|---|
| ScanNet++ 세그먼트 대응 AUPRC, 시점차 0–45/45–90/90–135/135–180° | SegMASt3R 92.8/91.1/88.0/83.6; SAM2 비디오 전파 61.9/46.6/27.9/17.0; RoMa 61.6/58.9/47.4/30.0; MASt3R 키포인트 59.5/57.3/52.9/45.4; DINOv2 57.9/43.0/33.5/32.4 |
| N뷰 세그먼트 대응 AUPRC | MuViSeg Replica 84.48(SegMASt3R+LGv2), VKITTI2 82.78(+25.9); VGGT 특징 N=4 변형 81.4 |
| NVOS / SPIn-NeRF(클릭 다중 뷰 프로토콜) | MV-SAM 92.1 / 92.9; SAM2-Video 88.7 / 86.6; TrianguLang(텍스트) NVOS 93.5 |
| ScanNet++ 마스크 프롬프트 | 3AM IoU 90.6 / 추적 재현율 71.7(SAM2Long 74.7/41.3, DAM4SAM 76.5/43.6); Replica 81.2; MV-SAM 클릭 51.0 vs TrianguLang 텍스트 62.4 |
| 무순서 다중뷰 | G²TAM InsTrack S-mIoU 74.3 vs SAM2 47.6; 21.6 FPS |
| 자동 파놉틱/인스턴스 | PanSt3R PQ ScanNet 65.7·Hypersim 62.0·ScanNet++ 54.7; SegVGGT ScanNetv2 mAP 62.9(PanSt3R 26.9), ScanNet200 53.7(IGGT 28.2); EPS3D Replica 8뷰 의미 50.0·인스턴스 33.8; InstanceSplat ScanNet 16뷰 mIoU 64.60; Scenes-as-Objects ScanNet 8뷰 AP 0.235(IGGT 0.122); FAST3DIS ScanNet AP50 9.6 |
| Ego-Exo4D IoU(Ego2Exo/Exo2Ego) | V²-SAM 46.3/49.6(O-MaMa 42.6/44.1, ObjectRelator 35.3/40.3); DOMR 49.7/55.2; LM-EEC 54.98/65.77; VGGT-Segmentor 67.7 |

**실패 양상.** 학습 도메인 편향(ScanNet++ 실내, Ego-Exo4D 손 조작 소품 ~30 카테고리)이 공통이며, SegMASt3R는 실외 MapFree에서 저하하고 작은·반복 구조 세그먼트가 모호하다. MV-SAM은 π³ 포인트맵에 종속돼 무텍스처 영역·잡동사니 실내에서 실패하고 동적·합성 도메인에 취약하다. 3AM은 정렬된 카메라 궤적 전제이고 동적 객체는 원리상 불가하다. PanSt3R는 매우 희소한 입력(10장)에서 저하하고 벽·바닥은 stuff로 뭉쳐 조각 인스턴스가 불가하다. SegVGGT는 닫힌 집합 실내 카테고리라 사람·동물·환경 객체가 밖이다. V²-SAM은 DINOv3 의미 대응 의존으로 유사 객체 다수·비의미 영역에서 모호하고 쌍 단위라 N뷰 일관성이 없다. VGGT-Segmentor는 VGGT 내부 객체 어텐션은 일관되나 픽셀 투영 드리프트로 밀집 예측이 실패한다는 관찰을 보고했다. 어안은 어느 모델도 평가하지 않았고, SegVGGT는 절대 척도를 복원하지 못한다.

**적용 가능성.** 3AM(마스크 1뷰 프롬프트·비보정·추론만)과 G²TAM(무순서 정적 다중뷰 명시, 클릭/박스)이 "바로 사용 후보"이며, MV-SAM은 기준 뷰 마스크를 클릭으로 샘플링하면 비보정·무학습 추론이 되지만 논문 프로토콜이 다중 뷰 클릭이라 1뷰 조건 재검증이 필요하다. SegMASt3R·MuViSeg는 타 뷰를 SAM으로 자동 분할하고 기준 마스크와 매칭하는 "마스크 단위 전이"로 바로 구성 가능하나, 벽·바닥 조각은 학습 라벨상 인스턴스가 아니라 dustbin 처리 위험이 있다. 자동 인스턴스 모델(PanSt3R, EPS3D 등)은 예측 인스턴스를 기준 마스크와 IoU로 대응시키는 후처리가 필요하다. G²TAM은 마스크 프롬프트 지원·코드 공개가 미확인이고, MuViSeg는 코드·가중치 공개가 미확인이다. 어안 원본은 MASt3R·π³가 원근 가정이라 Fisheye3R 결합이 전제다.

### 3.6 기하 기초 모델(G: 입력 제공자)

**문제의식.** 보정이 없을 때 포즈·깊이·포인트맵을 누가 주는가. 이 문제에서 다른 모든 가족의 "개조"는 이 가족에 의존한다.

**핵심 아이디어와 흐름.** DUSt3R(CVPR 2024)는 이미지 쌍 → 공통 좌표 포인트맵 회귀, N뷰는 전역 정렬 최적화. MASt3R는 밀집 국소 특징 헤드로 극단 시점차 매칭(Map-free VCRE AUC +30%p), MASt3R-SfM은 무순서 집합을 검색 그래프로 선형 규모 정렬. 2025년 MUSt3R·Fast3R·VGGT·π³·MapAnything·DA3가 단일 순전파로 이동했다 — VGGT는 1~수백 뷰에서 카메라·깊이·포인트맵·트랙을 1초 내 동시 예측, π³는 기준 뷰 없는 순열 등변 아키텍처(무순서 리그에 유리), MapAnything은 선택적 보정 입력 융합으로 1~2000뷰, DA3는 자체 벤치에서 VGGT 대비 자세 44.3%·기하 25.1% 개선. Rig3R는 동기화 다중카메라 리그를 명시적으로 조건화해 Waymo 자세 mAA@30° 82.1(보정 메타)/74.6(비구조) vs DUSt3R-GA 37.5, Fast3R 20.6, MV-DUSt3R 15.8을 보고했다 — 즉 범용 순전파 모델은 리그·저중첩 입력에서 무너진다. 어안은 Fisheye3R(ECCV 2026)가 VGGT·π³·MapAnything에 보정 토큰(약 29.5만 파라미터)만 학습해 ScanNet++ 어안에서 π³ 자세 이동 오차 0.164→0.072, 챔퍼 8.29→2.94, FoV AUC 0.55→0.87을 얻었고, RayTun3R는 10,752 파라미터 어댑터로 110–200° 어안에서 회전 오차 2–12배 감소를 보고했다.

**대표 논문.** DUSt3R(https://arxiv.org/abs/2312.14132), MASt3R(https://arxiv.org/abs/2406.09756), MASt3R-SfM(https://arxiv.org/abs/2409.19152), MUSt3R(CVPR 2025, https://arxiv.org/abs/2503.01661), VGGT(CVPR 2025, https://arxiv.org/abs/2503.11651), π³(https://arxiv.org/abs/2507.13347), MapAnything(3DV 2026, https://arxiv.org/abs/2509.13414), DA3(https://arxiv.org/abs/2511.10647), Rig3R(https://arxiv.org/abs/2506.02265), Fisheye3R(ECCV 2026, https://arxiv.org/abs/2603.28896), RayTun3R(https://arxiv.org/abs/2607.02711).

**실패 양상과 라이선스.** 원근 학습이라 어안에서 저하(VGGT, π³, MapAnything — Fisheye3R 동기), 무텍스처·잡동사니에서 포인트맵 열화(π³, MV-SAM 보고), 절대 척도 없음(VGGT), 리그 입력 붕괴(Fast3R·DUSt3R-GA, Rig3R 보고). Rig3R는 코드·가중치 공개가 미기재이고 주행 5카메라 편향이다. 라이선스는 NAVER 계열(DUSt3R/MASt3R/MUSt3R/PanSt3R) 전부 비상업, VGGT 원본·Fast3R·π³ 가중치·DA3 Large/Giant 비상업이며, 상업 가능한 것은 MapAnything-apache, DA3 Small/Base·Mono/Metric-Large, VGGT-1B-Commercial(신청제), π³ 코드(BSD-3)다. 반사면·투명체에서의 실패를 정량화한 검증 가능한 문헌은 이번 조사에서 찾지 못했다.

**적용 가능성.** 바로 사용(기하 제공자). 추정: 직선·호·평면 리그는 π³(무순서)·VGGT·MapAnything, 반구 어안은 Fisheye3R 적응판이 유일한 검증 경로이며, Rig3R가 보인 리그 붕괴가 9~46대 정적 리그에서 재현되는지는 아무도 측정하지 않았다.

### 3.7 프롬프트·기억·예시 계열(H)

**문제의식.** 참조 마스크를 SAM 계열의 기억이나 예시로 주고, 다른 카메라에서 같은 객체를 "찾아" 프롬프트를 재생성하자.

**핵심 아이디어.** 세 계열로 수렴한다. (1) 기하로 프롬프트 재생성: 라이트필드 논문은 에피폴라 기하로 마스크를 뷰 간 전파하고 SAM2 잠재공간을 탐침해 가림을 추정한 뒤 제약 프롬프트로 정제한다(SAM2 비디오 추적 대비 7배 빠름). Explicit Memory(RA-L 2025)는 온라인 3DGS에 세그먼트 ID를 저장해 SAM2가 놓친 객체를 재프롬프트한다. (2) 의미 대응으로 프롬프트 생성: SANSA는 SAM2의 prompt-and-propagate를 few-shot 분할로 재목적화하며 SAM2 특징이 추적 단서에 얽혀 의미 매칭이 약하다고 진단한다. VRP-SAM·DC-SAM은 참조 프로토타입 교차 어텐션으로 프롬프트 임베딩을 만든다(학습). (3) 예시 검출 → 매칭 → 앵커 재주입: SAM 3는 이미지 예시(박스/마스크)로 이미지마다 모든 일치 인스턴스를 검출·분할하고(SA-Co에서 기존 대비 정확도 2배), Re-Prompting SAM 3는 SAM3 검출 후보를 DINOv3 객체 수준 매칭으로 걸러 첫 프레임 마스크와 함께 트래커에 재주입한다(MOSEv2 J&F 51.17, 3위). InsDet 벤치마크는 SAM 제안 + DINOv2 매칭 비학습 기초선이 종단학습 검출기보다 AP >10 우세함을 보였다. LM-EEC(NeurIPS 2025)는 SAM2 기억 융합을 뷰 특성으로 가중해 Ego-Exo4D Ego→Exo 54.98 / Exo→Ego 65.77을 얻었다.

**대표 논문.** SAM 3(https://arxiv.org/abs/2511.16719), Re-Prompting SAM 3(https://arxiv.org/abs/2603.23788), Segment Anything in Light Fields(https://arxiv.org/abs/2411.13840), Explicit Memory through Online 3DGS(IEEE RA-L 2025, https://arxiv.org/abs/2510.23521), SANSA(NeurIPS 2025 Spotlight, https://arxiv.org/abs/2505.21795), Segment Anything Across Shots(AAAI 2026, https://arxiv.org/abs/2511.13715), InsDet(NeurIPS 2023 D&B, https://arxiv.org/abs/2310.19257), Robust Ego-Exo Correspondence with Long-Term Memory(NeurIPS 2025, https://arxiv.org/abs/2510.11417).

**실패 양상.** 예시 검출은 같은 개념의 다른 인스턴스(사람 여럿)를 모두 잡아 인스턴스 연결이 별도로 필요하고, 벽·바닥 조각 같은 비정형 환경 객체는 개념 프롬프트가 모호하다. 라이트필드 방식은 조밀·보정 배열 전제다. SAMURAI식 운동 사전은 시간 연속성 전제라 카메라 간에는 부적합하다. LM-EEC는 목표 객체가 사라진 뷰에서 유사 배경 객체를 잘못 분할한다.

**적용 가능성.** 추정: "참조 마스크 → SAM 3 예시 검출 → DINOv3/포인트맵 매칭으로 후보 선택 → 앵커 재주입" 구조는 학습 없이 카메라 점프에 그대로 옮길 수 있으며, D 계열 순서 전파의 실패를 보완하는 부품으로 가장 유망하다. 라이트필드의 "에피폴라 제약 프롬프트 + 가림 추정"은 VGGT 포즈로 에피폴라를 추정하면 광역 리그에도 적용 가능하다. SAM2Long·DAM4SAM식 기억 선택을 시간 없는 카메라 집합에 맞춘 연구는 없다.

---

## 4. 벤치마크·지표

| 이름 | 장면·뷰 | 정답·프롬프트 형식 | 지표 | 보정 제공? | 뷰 간 일관성을 재는가 |
|---|---|---|---|---|---|
| MUVOD 3D 벤치마크(arXiv 2025) | 12장면·50객체(지배적 20·가림 12·소형 9·복잡구조 9); 리그 9~30대(어안 반구 4장면과 MATF 제외) | 사용자가 기준뷰에 점 또는 스트로크 → 목표뷰 예측을 GT와 IoU 비교(객체당 목표뷰 수는 본문에 명시 없음); 각 방법의 원래 프롬프트 방식 사용 | mIoU(전체/유형별) | 3D 방법들이 필요로 하므로 자세 사용(출처 미명시); 다운로드 페이지에 내·외부 파라미터 파일 기재 | 기준뷰→목표뷰 IoU만; ID 일관성 별도 채점 없음 |
| MUVOD VOS 벤치마크 | 17장면·카메라 9~46대·비디오당 30프레임·7,830장; 장면당 주석 카메라 3대(중앙 근처 c_ini 포함) | c_ini 한 키프레임 GT 마스크 1장 → 프레임·카메라 전파; 주석은 XMem+LightGlue 전파 후 뷰별 수동 교정 | J&F^N(주석 카메라 N=3 평균); XMem 기준선 79.4 | 위와 동일 | 카메라 평균 J&F로 간접 |
| NVOS(LLFF) | 8장면, 기준뷰 1·목표뷰 1 | 전경/배경 스크리블 | Acc/IoU(mIoU/mAcc) | COLMAP | 아니오(목표뷰 1장) |
| SPIn-NeRF | 10장면(360 inward), 60+40장 | 희소 점(후속은 마스크에서 양성 8+/음성 2) → 전 뷰 | Acc/IoU | COLMAP | 전 뷰 IoU 평균(ID는 단일 객체) |
| LERF-Mask | 3장면(figurines, ramen, teatime)·테스트뷰 2~4 | 텍스트 질의 → 마스크 | mIoU/mBIoU | COLMAP | 아니오 |
| LERF-OVS / 3D-OVS / DL3DV-OVS | 4장면 / 10장면·20~30장·5 GT뷰(배경어 wall·desktop 포함) / 4장면·22질의(CVPR 2026) | 텍스트 | localization Acc/mIoU; mIoU/Acc; mIoU/mAcc@0.25 | COLMAP; COLMAP; 3DGS 30k iter | 아니오 |
| Replica / ScanNet NeRF-split, Messy Rooms | 8장면 ≈180/180장; 7장면 ≈300/100장; 8장면·N≤500객체·M≤1200뷰 | 기계 라벨 panoptic 또는 class-agnostic 인스턴스 | PQ^scene(장면 단위 인스턴스 부분집합 IoU>0.5 매칭), mIoU, 장면 헝가리안 IoU/P/R | 제공/렌더 | 예 — PQ^scene은 뷰 간 ID 보존을 직접 잼 |
| ScanNet++ | 460→1000+ 장면, DSLR 28만 장 + iPhone RGB-D | 3D 인스턴스(벽·바닥 등 환경 클래스 포함)를 프레임에 래스터화; MV-SAM 프로토콜: 장면당 100프레임·객체 5개·여러 뷰 양성 10/음성 2 클릭 | 3D mAP; MV-SAM식 mIoU/mAcc; SegMASt3R식 세그먼트 대응 AUPRC(시점차 구간별) | 제공 | 래스터화 GT는 정의상 ID 일관; AUPRC 프로토콜은 직접 잼 |
| Ego-Exo4D 대응 트랙 | 1,286시간·740명; ego 1 + exo 다수 | 한 뷰 객체 마스크 → 다른 뷰 같은 객체 마스크 | IoU(Ego2Exo/Exo2Ego) | 없음(비보정 학습) | 2뷰 쌍만 |
| uCO3D(MV-SAM 사용) | 50 시퀀스 × 50 프레임 | langSAM+XMem 생성 마스크(모델 생성 GT) | mIoU/mAcc | VGGSfM | 단일 객체 |
| MVRefer(CVPR 2026) | ScanNet 8뷰 균일 샘플 | 텍스트 | 3D mIoU + 뷰별 mIoU_pos(가시)/mIoU_neg(비가시) | 제공 | 가시/비가시 뷰 분리 |
| MOVi-MC-AC(합성) | 다중카메라 비디오, ~580만 인스턴스 | 카메라 간 동일 ID 모달/아모달 마스크 | (기초선 미기재) | 합성(감출 수 있음) | 예(ID GT) |
| Charge(합성 NVS) | 밀집 다중뷰 동적 장면(카메라 수·ID 일관성 미확인) | 객체 분할 GT | NVS 지표 | 렌더 | 미확인 |

**MUVOD 3D 벤치마크 규약의 요점.** 12장면·50객체를 지배적/가림/소형/복잡구조로 나누고, 각 방법이 자기 프롬프트 방식(점·스트로크)으로 기준뷰에서 시작해 목표뷰 마스크를 내면 GT와 IoU를 잰다. 공식 수치(mIoU 전체/지배/가림/소형/복잡): Gaussian Grouping 78.8(84.5/69.4/86.4/71.1), SA3D 65.8(70.9/58.7/78.3/50.4), SAGA 47.2(63.7/44.3/23.8/37.0), ISRF 28.0(38.8/24.0/15.2/22.1). 반구(어안) 4장면은 "fisheye undistortion으로 인한 심한 화질 저하·객체 잘림" 때문에, MATF는 NeRF/3DGS 복원 품질 불량으로 제외됐다. 주석은 장면당 3대 카메라뿐이라 나머지 뷰는 GT가 없다. 따라서 "마스크 1뷰·보정 없음·한 순간"이라는 이 문제의 프로토콜은 MUVOD 논문 안에 없으며 새로 정의해야 한다. 추정: 정의할 때 (a) Panoptic Lifting의 PQ^scene에서 클래스 항을 제거한 장면 단위 ID 매칭, (b) MVRefer식 가시/비가시 뷰 분리 IoU, (c) 유형별(환경 객체 별도) mIoU를 결합하면 기존 지표와 호환되면서 ID 일관성까지 잴 수 있다. "Mip-NeRF360 분할 세트"는 공식 GT가 확인되지 않아(정성 위주) 제외했다.

---

## 5. 적용 가능성 총표

조건: 마스크 1뷰 · 보정 없음 · 학습 없음(추론만) · 환경 객체(벽·바닥 조각) 포함 · 반구 어안 리그.

| 방법 | 판정 | 이유 | 개조하려면 무엇이 필요한가 |
|---|---|---|---|
| Matcher / PerSAM / FoRIS / REBASE / INSID3 (E) | 바로 사용 | 참조 마스크 1장·보정·학습 불필요·카테고리 없음 | 큰 시점차·동일 카테고리 다중 인스턴스·환경 객체 혼동을 포인트맵 대응으로 검증; DINOv3 층 선택; 어안은 미평가 |
| 3AM (F) | 바로 사용 후보 | 마스크 1뷰 프롬프트·비보정·추론만 | 카메라를 인접 순으로 정렬해 비디오로 입력; stuff 인스턴스는 학습 라벨에 없어 검증; 어안 미평가 |
| G²TAM (F) | 바로 사용 후보 | 무순서 정적 다중뷰 과제를 명시, 비보정, 추론만 | 마스크→점/박스 변환; 마스크 프롬프트·가중치 공개 미확인; 환경 객체 미확인 |
| MV-SAM (F) | 개조하면 사용 | π³+SAM2.1 디코더, 비보정·최적화 없음 | 기준 뷰 마스크를 클릭으로 샘플링(논문은 다중 뷰 클릭); 무텍스처 취약; 어안은 Fisheye3R 결합 전제 |
| SegMASt3R / MuViSeg (F) | 개조하면 사용 | 비보정 세그먼트 매칭, 추론만, 135–180°에서도 유지 | 타 뷰 SAM 자동 분할 + 기준 마스크 매칭 구성; 벽·바닥 조각 dustbin 위험; MuViSeg 코드 미확인; 어안은 MASt3R 원근 가정 |
| V²-SAM Anchor 전문가 (F) | 바로 사용(부품) | DINOv3 대응→SAM2 점 프롬프트, 무학습 | 유사 객체 다수·비의미 영역 오매칭 보완; 쌍 단위라 N뷰 일관성 별도 |
| WildSeg3D (D) | 개조하면 사용 | 무보정(MASt3R 정렬)·무학습·마스크 캐시 | 46대 넓은 기선 순서 설계; SAM2 추적 실패 보완; 환경 객체·어안 미검증 |
| OR-NeRF / View-consistent Object Removal (C) | 개조하면 사용 | 1뷰 마스크→전 뷰 재투영/워핑, 학습 없음, SPIn-NeRF 95.42/94.27 | COLMAP 점군·단안 깊이를 VGGT/π³ 포인트맵으로 대체; 가림 구멍 처리 |
| MaskClustering / SAM-Zero3D / MV3DIS / Aerial Lifting 1단계 (C) | 개조하면 사용 | 학습 없는 병합 규칙(뷰 합의율·연결 성분+다수결·3D 참조 매칭·IoMin) | RGB-D·자세를 포인트맵으로 대체; 자동 분할을 "참조 마스크와 겹치는 영역 선택"으로 바꿈; 희소 뷰 합의 불안정 |
| GraphSeg / CCGS / Z3D (C) | 개조하면 사용 | DUSt3R 기반 무보정 마스크 연관 | 1뷰 마스크를 시드로; 46뷰 규모·환경 객체 미검증; Z3D 이미지만 설정 급락 |
| CDSeg / DivAS (C) | 개조하면 사용 | 1뷰 프롬프트→전 뷰, 학습 없음 | 보정 점군·렌더 깊이를 포인트맵으로 대체; CDSeg의 SAM2 전파는 넓은 기선 위험; DivAS 앵커를 실제 카메라로 |
| FlashSplat / LUDVIG / GaussianCut / CrashSplat / ArtisanGS (B) | 개조하면 사용 | 사전학습 3DGS 위 학습 없는 리프팅, 마스크 입력 그대로 | VGGT/MASt3R-SfM→3DGS 브리지 필요; 어안 3DGS 미검증; 희소 뷰 3DGS 품질이 상한 |
| SAGOnline / Seed2GS / SAGO / VCAR / iSegMan (B) | 개조하면 사용 | 학습 없음, 3DGS 렌더 궤적으로 광기선 우회 | 3DGS·포즈 선행; SAM 재프롬프팅이 환경 객체 경계를 바꿀 위험; VCAR 볼록 가정 |
| SAGA / Gaussian Grouping / OmniSeg3D / Click-Gaussian / NG-GS / COB-GS (B) | 불가 | 장면별 최적화(수십 분)·포즈 필수 | 조건 위반(학습 없음) |
| PanSt3R / SegVGGT / EPS3D / InstanceSplat / GroupForward (F) | 개조하면 사용(부분) | 비자세 입력이지만 자동 인스턴스, 학습됨 | 예측 인스턴스↔기준 마스크 IoU 후처리; 벽·바닥 stuff; SegVGGT 닫힌 집합 |
| O-MaMa / DOMR / VGGT-Segmentor / CCMP (F) | 개조하면 사용(학습 감수) | 입출력 동형이나 Ego-Exo4D 학습·쌍 단위 | 무학습 매칭으로 교체하거나 TTT(CCMP); 환경 객체 미학습 |
| SAM 3 예시 검출 + Re-Prompting SAM 3 (H) | 개조하면 사용 | 참조 마스크를 예시로 각 카메라 검출, 학습 없음 | 인스턴스 연결(DINOv3/포인트맵 매칭); 환경 객체 개념 모호; 시간 대신 카메라 순서 앵커 풀 |
| Segment Anything in Light Fields (H) | 개조하면 사용 | 에피폴라 제약 프롬프트 + 가림 추정, SAM2 수정 없음 | 에피폴라를 VGGT 포즈로 추정; 조밀 배열 가정 제거 |
| HOMER (E) | 바로 사용(한계) | 자세·학습 없음, LoFTR 호모그래피 워핑 | 평면 가정·오차 누적; 마스크 IoU 정량 없음 |
| VGGT / π³ / MapAnything / DA3 / MASt3R-SfM (G) | 바로 사용(기하 제공자) | 비보정 포즈·깊이·포인트맵 | 어안은 Fisheye3R/RayTun3R 적응판; 라이선스(비상업) 확인; 리그 붕괴(Rig3R 보고) 재검증 |
| Fisheye3R / RayTun3R (G) | 바로 사용(어안 기하) | 어안 원본에서 자세·깊이·FoV 추정 | 분할 모델과의 결합은 미검증 |
| Rig3R (G) | 불가(현재) | 가중치 미공개, 주행 리그 편향 | 리그 조건화 개념만 참고 |
| 텍스트 프롬프트 계열(LangSplat, ReferSplat, TrianguLang, ZeroSplat, OpenVoxel 등) | 불가 | 텍스트 질의 전제, 카테고리 없는 환경 객체와 불일치 | — |
| SAMURAI 류 운동 기억 (H) | 불가 | 카메라 간 운동 연속성 없음 | — |

---

## 6. 빈칸

**사실(조사 목록 기준으로 확인된 부재).**
- "보정 없음 + 학습 없음 + 마스크 1뷰 + 정적 리그 한 순간"을 동시에 만족하며 마스크 IoU를 보고한 논문은 HOMER 하나뿐이고, 그마저 하류 인페인팅 품질로만 평가해 마스크 IoU 직접 비교가 없다. WildSeg3D는 무보정·무학습이지만 프롬프트가 점·클릭이고 리그 평가가 없다.
- 벽·바닥 조각 같은 환경 객체에 대한 뷰 간 마스크 전이 평가는 어느 가족에도 없다. VoteSplat이 FoV보다 큰 인스턴스 취약을 명시했고, 학습형 매칭(SegMASt3R, MuViSeg, PanSt3R, SegVGGT)은 벽·바닥을 stuff 또는 dustbin으로 처리한다. 3D-OVS의 배경어(wall, desktop)와 ScanNet++의 환경 클래스 래스터화가 유일한 간접 재료다.
- 어안·반구 리그에서 동작하는 분할 모델은 없다. Fisheye3R·RayTun3R는 기하만 적응했고, MUVOD도 어안 4장면을 3D 벤치마크에서 제외했다. MV-SAM·3AM·G²TAM·SegMASt3R 모두 어안 미평가다.
- MUVOD 벤치마크를 재채점한 후속 논문은 조사 시점 0건이며, MV-SAM을 직접 비교 대상으로 삼은 논문은 TrianguLang뿐이다(텍스트 62.4 vs 클릭 51.0).
- 정적 리그(직선·호·평면·반구)에 대해 카메라 순서(카메라 그래프·순회 순서)를 체계적으로 비교한 논문은 없고, SAM2Long·DAM4SAM식 기억 선택을 시간 없는 카메라 집합에 맞춘 연구도 없다.
- "키포인트/밀집 매칭 + SAM 프롬프트로 다중뷰 마스크 전이"를 주제로 한 독립 논문은 확인되지 않았다(관련 기법은 부품으로만 존재).
- 피드포워드 기하 모델의 반사면·투명체 실패를 정량화한 검증 가능한 문헌은 찾지 못했고, 9~46대 정적 리그에서 VGGT·π³가 Rig3R가 보인 리그 붕괴(Fast3R mAA 20.6)를 재현하는지도 아무도 측정하지 않았다.
- 다중 객체 마스크 1뷰를 한 번에 전이하는 프로토콜은 DOMR(학습, 쌍 단위)과 FlashSplat 장면 모드(비디오 추적기 ID) 외에 없으며, 리프팅 계열 대부분은 단일 객체 이진 파이프라인이다.
- 다중시점 매팅(뷰 일관 알파), 자율주행 서라운드뷰의 겹침 카메라 간 인스턴스 일관, 로봇 다중카메라의 카메라 간 마스크 전이 자체를 평가한 2023년 이후 논문은 확인되지 않았다.

**추정(조사자의 해석).**
- 가장 직접적인 무학습 경로는 "π³/VGGT(어안은 Fisheye3R) 포인트맵 → 참조 마스크 깊이 워핑(View-consistent Object Removal 방식) 또는 3D점 재투영(OR-NeRF 방식) → SAM3 예시 검출/REBASE 프롬프트로 후보 생성 → 뷰 합의율 또는 연결 성분+다수결(MaskClustering, SAM-Zero3D)로 ID 확정 → 가림 뷰는 Obj-NeRF식 IoU 임계로 폐기"이며, 여기에 환경 객체를 위해 SAM 재프롬프팅 대신 마스크 그대로의 투영을 우선하는 규칙이 필요하다.
- 3DGS를 만들 수 있다면 SAGOnline·Seed2GS의 "가상 궤도 렌더링 + SAM2"가 광기선을 우회하는 강력한 대안이지만, 9~46뷰 희소 리그와 어안에서 3DGS 품질이 상한이 되므로(LUDVIG·SANeRF-HQ의 지적을 외삽) 반구 리그에서는 성립하지 않을 가능성이 크다.
- 학습형 중에서는 G²TAM(무순서 명시)과 3AM(마스크 프롬프트)이 리그 조건에 가장 가깝고, 이들의 ScanNet++ 실내 편향이 MUVOD의 야외·사람·동물 장면에서 어떻게 나타나는지가 첫 실험 질문이 된다. SegMASt3R 표의 시점차 구간별 AUPRC 프로토콜을 MUVOD 리그 기하(직선·호·반구)에 옮기면 "어느 카메라 쌍이 전이 가능한가"를 처음으로 정량화할 수 있다.
- 새 평가 프로토콜은 (a) 주석 카메라 3대 중 하나를 기준뷰로 두고 나머지 둘을 목표뷰로, (b) 카메라 자세 파일을 감추고, (c) 다중 객체 ID 매칭을 PQ^scene(클래스 항 제거)으로, (d) 환경 객체·가림·소형·복잡구조를 유형별로 분리해 보고하는 형태가 기존 문헌과 호환되면서 빈칸을 정확히 겨냥한다. MOVi-MC-AC(카메라 간 ID GT 합성)는 사전 검증장으로 바로 쓸 수 있다.

**조사 한계.** 웹 검색 예산 소진으로 arXiv API·abs/HTML 페이지·Semantic Scholar(후반 429 차단)·Crossref·OpenAlex로만 확인했다. IEEE·Springer·Elsevier·SPIE 게재물 9편(Gaussian Prompter TPAMI 2025, SCGS, SA3D-L, OV-BIS, 최적 뷰 선택 EAAI 등)은 서지만 확인되고 초록은 미확인이다. 2026년 8월 이후 arXiv 제목 스윕은 미완이다. Gaussian Prompter(TPAMI)와 "3D-aware mask alignment and identity expansion"(IEEE Access 2026)은 제목이 이 문제와 가장 직접적이므로 원문 우선 열람을 권한다.

---

## 7. 출처 목록(가족별, 연도순)

**B. 3D 표현 최적화(NVOS·필드)**
- 2023 SPIn-NeRF (CVPR 2023) https://arxiv.org/abs/2211.12254
- 2023 Interactive Segmentation of Radiance Fields, ISRF (CVPR 2023) https://arxiv.org/abs/2212.13545
- 2023 Segment Anything in 3D with Radiance Fields, SA3D (NeurIPS 2023) https://arxiv.org/abs/2304.12308
- 2023 Point'n Move (arXiv) https://arxiv.org/abs/2311.16737
- 2023 Obj-NeRF (arXiv) https://arxiv.org/abs/2311.15291
- 2023 NTO3D (CVPR 2024) https://arxiv.org/abs/2309.12790
- 2024 SANeRF-HQ (CVPR 2024) https://arxiv.org/abs/2312.01531
- 2024 OmniSeg3D (CVPR 2024) https://arxiv.org/abs/2311.11666
- 2024 Gaussian Grouping (ECCV 2024) https://arxiv.org/abs/2312.00732
- 2024 GARField (arXiv) https://arxiv.org/abs/2401.09419
- 2024 Gaga (TMLR) https://arxiv.org/abs/2404.07977
- 2024 SAGD (arXiv) https://arxiv.org/abs/2401.17857
- 2024 FlashSplat (ECCV 2024) https://arxiv.org/abs/2409.08270
- 2024 Click-Gaussian (ECCV 2024) https://arxiv.org/abs/2407.11793
- 2024 GaussianCut (arXiv) https://arxiv.org/abs/2411.07555
- 2024 Gradient-Weighted Feature Back-Projection (SIGGRAPH Asia 2024) https://arxiv.org/abs/2411.15193
- 2024 Aerial Lifting (CVPR 2024) https://arxiv.org/abs/2403.11812
- 2025 Segment Any 3D Gaussians, SAGA (AAAI 2025) https://arxiv.org/abs/2312.00860
- 2025 LUDVIG (ICCV 2025) https://arxiv.org/abs/2410.14462
- 2025 Unified-Lift / Rethinking End-to-End 2D to 3D Scene Segmentation (CVPR 2025) https://arxiv.org/abs/2503.14029
- 2025 COB-GS (CVPR 2025) https://arxiv.org/abs/2503.19443
- 2025 iSegMan (CVPR 2025) https://arxiv.org/abs/2505.11934
- 2025 SAGOnline (arXiv) https://arxiv.org/abs/2508.08219
- 2025 CCGS (IJCV 2025) https://arxiv.org/abs/2502.16303
- 2025 Trace3D (ICCV 2025) https://arxiv.org/abs/2508.03227
- 2025 VoteSplat (ICCV 2025) https://arxiv.org/abs/2506.22799
- 2025 CrashSplat (SYNASC 2025) https://arxiv.org/abs/2509.23947
- 2025 ObjectGS (ICCV 2025) https://arxiv.org/abs/2507.15454
- 2025 InstaScene (ICCV 2025) https://arxiv.org/abs/2507.08416
- 2025 Binary-Gaussian (AAAI 2026) https://arxiv.org/abs/2512.00944
- 2025 Gaussian Prompter (IEEE TPAMI 2025, 서지만 확인) https://doi.org/10.1109/tpami.2025.3576839
- 2026 DivAS (arXiv) https://arxiv.org/abs/2601.04860
- 2026 ArtisanGS (arXiv) https://arxiv.org/abs/2602.10173
- 2026 NG-GS (CVPR 2026 Highlight) https://arxiv.org/abs/2604.14706
- 2026 Online Segment 3D Gaussians via Launching Virtual Drones, SAGO (arXiv) https://arxiv.org/abs/2607.01628
- 2026 VCAR (ACM MM 2026) https://arxiv.org/abs/2608.30870
- 2026 Seed2GS (arXiv) https://arxiv.org/abs/2608.11928
- 2026 GaussianTrimmer (arXiv) https://arxiv.org/abs/2601.12683
- 2026 BEA-GS (CVPR 2026 Highlight) https://arxiv.org/abs/2605.09662
- 2026 Gradient conflict for boundary-aware segmentation in 3DGS (SIVP 2026) https://doi.org/10.1007/s11760-026-05653-3
- 2026 Consistent Scene Understanding via Multi-Cue Mask Refinement (ICPR 2026) https://arxiv.org/abs/2607.01708

**C. 기하 투영·3D 인스턴스 병합**
- 2023 SAM3D (arXiv) https://arxiv.org/abs/2306.03908
- 2023 SAI3D (CVPR 2024) https://arxiv.org/abs/2312.11557
- 2023 OpenMask3D (NeurIPS 2023) https://arxiv.org/abs/2306.13631
- 2023 SAMPro3D (3DV 2025) https://arxiv.org/abs/2311.17707
- 2023 OR-NeRF (arXiv) https://arxiv.org/abs/2305.10503
- 2024 MaskClustering (CVPR 2024) https://arxiv.org/abs/2401.07745
- 2024 SA3DIP (arXiv) https://arxiv.org/abs/2411.03819
- 2024 Any3DIS (CVPR 2025) https://arxiv.org/abs/2411.16183
- 2024 PointSeg (ICCV 2025 Workshop) https://arxiv.org/abs/2403.06403
- 2024 3DIML (ICRA 2024) https://arxiv.org/abs/2403.19797
- 2024 View-Consistent Object Removal in Radiance Fields (ACM MM 2024) https://arxiv.org/abs/2408.02100
- 2024 OpenSU3D (arXiv) https://arxiv.org/abs/2407.14279
- 2025 SAM2Object (CVPR 2025) https://doi.org/10.1109/cvpr52734.2025.01800
- 2025 OnlineAnySeg (CVPR 2025) https://arxiv.org/abs/2503.01309
- 2025 SGS-3D (AAAI 2026) https://arxiv.org/abs/2509.05144
- 2025 GraphSeg (IROS 2026) https://arxiv.org/abs/2504.03129
- 2026 SAM-Zero3D (IEEE TCSVT 2026) https://doi.org/10.1109/tcsvt.2026.3666899
- 2026 Clutt3R-Seg (ICRA 2026) https://arxiv.org/abs/2602.11660
- 2026 SpaCeFormer (arXiv) https://arxiv.org/abs/2604.20395
- 2026 MV3DIS (arXiv) https://arxiv.org/abs/2604.08916
- 2026 Z3D (ACL 2026) https://arxiv.org/abs/2602.03361
- 2026 CDSeg (arXiv) https://arxiv.org/abs/2608.05482
- 2026 Joint 2D-3D Segmentation and Association in Street-level Imaging (ICPR 2026) https://arxiv.org/abs/2605.26725
- 2026 Stream3Dv2 (arXiv) https://arxiv.org/abs/2608.21136
- 2026 Split&Splat (arXiv) https://arxiv.org/abs/2602.03809
- 2026 Lift, Associate, and Fuse (arXiv, 서베이) https://arxiv.org/abs/2608.20659

**D. 뷰=비디오 전파**
- 2023 OSTRA: A One Stop 3D Target Reconstruction and multilevel Segmentation Method (arXiv) https://arxiv.org/abs/2308.06974
- 2025 WildSeg3D (ICCV 2025) https://arxiv.org/abs/2503.08407
- 2026 SplatXtRact (IEEE RA-L 2026) https://doi.org/10.1109/lra.2026.3699257
- 2026 SAP: Segment Any 4K Panorama (arXiv) https://arxiv.org/abs/2603.12759

**E. 대응·매칭**
- 2023 Matcher (ICLR 2024) https://arxiv.org/abs/2305.13310
- 2023 PerSAM (arXiv) https://arxiv.org/abs/2305.03048
- 2023 DIFT: Emergent Correspondence from Image Diffusion (NeurIPS 2023) https://arxiv.org/abs/2306.03881
- 2023 Telling Left from Right, GeoAware-SC (CVPR 2024) https://arxiv.org/abs/2311.17034
- 2023 RoMa (CVPR 2024) https://arxiv.org/abs/2305.15404
- 2023 LightGlue (arXiv) https://arxiv.org/abs/2306.13643
- 2023 SEGIC (ECCV 2024) https://arxiv.org/abs/2311.14671
- 2024 Probing the 3D Awareness of Visual Foundation Models (CVPR 2024) https://arxiv.org/abs/2404.08636
- 2024 MESA (CVPR 2024) https://arxiv.org/abs/2401.16741
- 2024 ZeroCo (arXiv) https://arxiv.org/abs/2412.09072
- 2025 No time to train! (arXiv) https://arxiv.org/abs/2507.02798
- 2025 HOMER (arXiv) https://arxiv.org/abs/2501.17636
- 2025 Zero-Shot Polygon Matching (arXiv) https://arxiv.org/abs/2511.05949
- 2026 Revealing the Semantic Selection Gap in DINOv3 (arXiv) https://arxiv.org/abs/2602.07550
- 2026 INSID3 (CVPR 2026) https://arxiv.org/abs/2603.28480
- 2026 REBASE (arXiv) https://arxiv.org/abs/2607.09082
- 2026 FoRIS (arXiv) https://arxiv.org/abs/2609.03384

**F. 학습된 대응·피드포워드**
- 2024 ObjectRelator (ICCV 2025 Highlight) https://arxiv.org/abs/2411.19083
- 2025 O-MaMa (ICCV 2025) https://arxiv.org/abs/2506.06026
- 2025 DOMR (ACM MM 2025) https://arxiv.org/abs/2508.04050
- 2025 SegMASt3R (NeurIPS 2025 Spotlight) https://arxiv.org/abs/2510.05051
- 2025 PanSt3R (ICCV 2025) https://arxiv.org/abs/2506.21348
- 2025 V²-SAM (arXiv) https://arxiv.org/abs/2511.20886
- 2025 IGGT (arXiv) https://arxiv.org/abs/2510.22706
- 2026 MV-SAM (arXiv) https://arxiv.org/abs/2601.17866
- 2026 3AM (arXiv) https://arxiv.org/abs/2601.08831
- 2026 Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction (CVPR 2026) https://arxiv.org/abs/2602.18996
- 2026 TrianguLang (arXiv) https://arxiv.org/abs/2603.08096
- 2026 SegVGGT (arXiv) https://arxiv.org/abs/2603.19926
- 2026 FAST3DIS (arXiv) https://arxiv.org/abs/2603.25993
- 2026 VGGT-Segmentor (arXiv) https://arxiv.org/abs/2604.13596
- 2026 EPS3D (ICML 2026) https://arxiv.org/abs/2606.08980
- 2026 Scenes as Objects, Not Primitives (arXiv) https://arxiv.org/abs/2606.29513
- 2026 G²TAM (ICML 2026) https://arxiv.org/abs/2607.03789
- 2026 MuViSeg (arXiv) https://arxiv.org/abs/2607.17938
- 2026 Extending a Large View Synthesis Model for Multi-view Panoptic Segmentation (ECCV 2026) https://arxiv.org/abs/2607.19765
- 2026 InstanceSplat (arXiv) https://arxiv.org/abs/2608.07144
- 2026 GroupForward (arXiv) https://arxiv.org/abs/2608.17535
- 2026 PanORama (arXiv) https://arxiv.org/abs/2603.19920

**G. 기하 기초 모델**
- 2023 DUSt3R (CVPR 2024) https://arxiv.org/abs/2312.14132
- 2024 MASt3R (ECCV 2024) https://arxiv.org/abs/2406.09756
- 2024 MASt3R-SfM (arXiv) https://arxiv.org/abs/2409.19152
- 2025 Fast3R (CVPR 2025) https://arxiv.org/abs/2501.13928
- 2025 MUSt3R (CVPR 2025) https://arxiv.org/abs/2503.01661
- 2025 VGGT (CVPR 2025) https://arxiv.org/abs/2503.11651
- 2025 Rig3R (arXiv) https://arxiv.org/abs/2506.02265
- 2025 π³ (arXiv) https://arxiv.org/abs/2507.13347
- 2025 MapAnything (3DV 2026) https://arxiv.org/abs/2509.13414
- 2025 Depth Anything 3 (arXiv) https://arxiv.org/abs/2511.10647
- 2026 Fisheye3R (ECCV 2026) https://arxiv.org/abs/2603.28896
- 2026 RayTun3R (arXiv) https://arxiv.org/abs/2607.02711

**H. 프롬프트·기억·예시**
- 2023 InsDet (NeurIPS 2023 Datasets and Benchmarks) https://arxiv.org/abs/2310.19257
- 2024 SAMURAI (arXiv) https://arxiv.org/abs/2411.11922
- 2024 VRP-SAM (CVPR 2024) https://arxiv.org/abs/2402.17726
- 2024 Segment Anything in Light Fields via Constrained Prompting (arXiv) https://arxiv.org/abs/2411.13840
- 2025 Correspondence as Video (ICCV 2025) https://arxiv.org/abs/2508.07759
- 2025 Explicit Memory through Online 3D Gaussian Splatting (IEEE RA-L 2025) https://arxiv.org/abs/2510.23521
- 2025 SAM 3: Segment Anything with Concepts (arXiv) https://arxiv.org/abs/2511.16719
- 2025 SANSA (NeurIPS 2025 Spotlight) https://arxiv.org/abs/2505.21795
- 2025 Segment Anything Across Shots (AAAI 2026) https://arxiv.org/abs/2511.13715
- 2025 Robust Ego-Exo Correspondence with Long-Term Memory (NeurIPS 2025) https://arxiv.org/abs/2510.11417
- 2026 FoundYou (ECCV 2026) https://arxiv.org/abs/2608.29917
- 2026 Re-Prompting SAM 3 via Object Retrieval (arXiv) https://arxiv.org/abs/2603.23788

**K. 벤치마크·데이터셋**
- 2023 Panoptic Lifting (CVPR 2023) https://arxiv.org/abs/2212.09802
- 2023 Weakly Supervised 3D Open-vocabulary Segmentation, 3D-OVS (NeurIPS 2023) https://arxiv.org/abs/2305.14093
- 2023 ScanNet++ (ICCV 2023) https://arxiv.org/abs/2308.11417
- 2024 LangSplat, LERF-Mask/LERF-OVS 주석 (CVPR 2024) https://arxiv.org/abs/2312.16084
- 2024 Ego-Exo4D (CVPR 2024) https://arxiv.org/abs/2311.18259
- 2025 MUVOD (arXiv) https://arxiv.org/abs/2507.07519
- 2025 UnCommon Objects in 3D (arXiv) https://arxiv.org/abs/2501.07574
- 2025 MOVi-MC-AC (arXiv) https://arxiv.org/abs/2507.00339
- 2025 Charge (arXiv) https://arxiv.org/abs/2512.13639
- 2026 LightSplat, DL3DV-OVS (CVPR 2026) https://arxiv.org/abs/2603.24146
- 2026 MVGGT / MVRefer (CVPR 2026) https://arxiv.org/abs/2601.06874