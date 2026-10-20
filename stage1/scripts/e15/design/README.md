# S1-E15 설계 탐침 보관 (읽기 전용, 2026-10-20)

[S1-E15 사전 등록](../../../../docs/stage1-E15-prereg.md) §2의 탐침 수치를 낸 스크립트입니다. 스크래치 `.../scratchpad/e15/`에서 고치지 않고 그대로 옮겼습니다. 검토 1(M1)의 기준점 확인 2개는 스크래치 루트에서 옮겼습니다.

- **근거가 아닙니다.** 이 수치는 Step 0(`e15_select.py` + `stage1/geomem/id3d.py`)이 추적 코드로 다시 낸 뒤에만 근거로 씁니다(prereg §6.1).
- 경로는 스크래치 기준입니다(`ring_d15/` 출력, 행 JSON 등). 그대로는 다시 돌지 않을 수 있습니다.
- 읽은 것: DEV-A 정답, `Data/_dev/geoset_rep_xs_A`(라운드 1·최종 마스크, round1_conf), T06 900번 정답·D1 출력(스크래치 `ring_d15`)·NAS 마스크, π³ 캐시, c_ini 입력 마스크.
- 읽지 않은 것: DEV-B·키프레임의 정답(평가 카메라)과 예측, CROOM 새 정답, ScanNet++.
- 선언(prereg L3): `id3d_eval.py`(72점 + 시뮬레이션 규칙)와 `id3d_eval2.py`(108점, κ 1.25 고정)가 A-c\* 상수를 Step 1과 같은 DEV-A·T06 자료에서 이미 훑었습니다.

| 파일 | 하는 일 | 출력 | prereg | 상태 |
|---|---|---|---|---|
| `angles_devA.py` | DEV-A 프레임마다 모든 카메라의 a_v(장면·객체 기준점), 광축 각 | `angles_devA.json` | N1 | 쓰임 |
| `t06_angles.py` | T06 카메라별 a_v(두 기준점)와 C0·라운드 1·사슬 뷰 J | 화면 출력 | N1 T06 표 | 쓰임 |
| `fixa_sim.py` | T06 라운드 1·최종 마스크에 A-c0·A-v | `fixa_sim.json` | N5 | 쓰임 |
| `fixa_devA.py` | DEV-A 라운드 1 기억 후보에서 Fix A 판정과 reproj 비교(예측과 입력만) | `fixa_devA.json` | N6 | 쓰임 |
| `id3d_calib.py` | 정답·합성 바꿔치기·예측 마스크의 3D 통계 행(첫 판) | `id3d_rows.json`(보관 안 함) | — | 대체됨(`id3d_calib2.py`) |
| `id3d_eval.py` | 첫 판 행에 A-c 규칙 72점 + 시뮬레이션 규칙 | 화면 출력 | — | 대체됨(`id3d_eval2.py`) |
| `id3d_ff.py` | 첫 판 행에서 거짓 FLAG 장면·객체 분해, N_min 20–200 | 화면 출력 | — | 대체됨 |
| `id3d_calib2.py` | 첫 판 + 테두리 몫, 질의 반경(γ용) | `id3d_rows2.json`(보관 안 함) | N3, N4 | 쓰임 |
| `id3d_eval2.py` | 둘째 판 행에 A-c\* 부분 격자 108점 | 화면 출력 | N3, N4 | 쓰임 |
| `id3d_t06r1.py` | T06 라운드 1 먼 절반에 후보 규칙 4개 | 화면 출력 | N4, N5 | 쓰임 |
| `anchor_cov.py` | MUVOD 5장면 c_ini 입력 마스크(basic)가 유효 화소를 덮는 몫(모두 1.0) | 화면 출력 | N1, M1 | 검토 1 확인 |
| `anchor_chk.py` | 장면·객체 기준점 차이(0.0)와 최대 각(CBABasketball v00 58.04°, Blocks cam9 47.12°) | 화면 출력 | N1, M1 | 검토 1 확인 |

**보관하지 않은 출력** (다시 낼 수 있음, 크기 때문에 뺌)

| 파일 | 크기 | sha256 |
|---|---|---|
| `id3d_rows.json` | 21.6 MB | `7fb28bdec99ba040dcfb88ba6d36cd520599492cd09c6bdc9a48f9086d9d0567` |
| `id3d_rows2.json` | 28.2 MB | `883f6e51381b049c82383f743eaef4e6a00bbda5de496711a7cd9e6aebdd33fe` |

**남지 않은 탐침.** 설계 1이 DEV-B·키프레임의 π³ 자세와 입력만 읽어 "평가 카메라 최대 47.1°"를 낸 탐침은 인라인으로 실행돼 파일이 없습니다. 키프레임 모든 카메라 최대 58.12°(CBABasketball)는 [`../../ring_diag/baselines.json`](../../ring_diag/baselines.json)(`obj_max`, 장면 기준점 = MUVOD에서는 객체 기준점)에 있습니다. DEV-B 모든 카메라 각도는 잰 적이 없습니다. Step 0이 DEV-A·DEV-B·키프레임 각도를 추적 코드로 다시 잽니다(prereg L1).

- `d1_t06_c0.sha256`: sha256 list of the ring-diagnosis D1 decode (preset/, preset_r1/) persisted untracked at `stage1/geomem/runs/_e15/d1_ref/` — the T06 C0 byte reference and Step 0 input (prereg §6.1).
