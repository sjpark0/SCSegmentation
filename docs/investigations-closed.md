# 종결된 조사 — 다시 하지 말 것

이 문서의 목적은 하나입니다: **이미 답이 나온 질문을 다시 조사하는 데 시간을 쓰지 않게 하는 것.**

각 항목은 `질문 → 답 → 근거 → 재조사 금지 이유` 형식입니다. 근거가 재현 가능한 것은 명령어를 적었습니다.
날짜는 조사가 끝난 시점입니다.

---

## 1. Painter 데이터셋의 14개 파일 차이 (2026-09-04 종결)

**질문.** OneStage에 메모리 수정을 넣고 재실행했더니 15개 중 Painter만 14개 PNG가 달랐다. 패치가 원인인가?

**답. 아니다. 패치와 무관하며, 점수에도 영향이 없다.**

**근거 (배제 실험 순서대로).**

| 시도 | 결과 |
|---|---|
| 동일 조건 2회 연속 실행 | 두 번 모두 같은 출력 → 비결정성 아님 |
| `io_utils.py` fp32 캐스트만 되돌림 | 차이 그대로 |
| `torch.clear_autocast_cache()` 호출 제거 | 차이 그대로 |
| **패치 이전 원본 폴더(`/tmp/sibling_backup`)로 실행** | **같은 차이 재현** |

마지막 줄이 결정적입니다. 손대지 않은 코드가 같은 차이를 내므로 원인은 코드 밖에 있습니다.
2026-09-03 기준선 생성과 09-04 재실행 사이에 호스트 RAM 고갈로 인한 재부팅이 여러 번 있었고, 그 사이 환경이 달라진 것으로 판단했습니다.

**점수 영향 없음.** 차이나는 14개는 전부 면적 0px 빈 마스크이고 해당 GT도 0px입니다.
DAVIS 규약상 `빈 GT + 빈 예측 = 1.0`이므로 양쪽 모두:

```
J = 0.845425   F = 0.891022   J&F = 0.868223
```

객체 단위 값이 다른 것은 0건이었습니다.

**재조사 금지 이유.** 이분 탐색을 이미 끝까지 했습니다. 다시 하면 같은 네 줄을 반복하게 됩니다.
Painter에서 또 차이가 보이면 원인 탐색이 아니라 **점수 동일 여부만** 확인하십시오
(`Data/MVSeg/eval_jf.py`로 두 출력의 J·F를 뽑아 비교).

---

## 2. SAM 2는 되는데 SAM 3는 왜 죽는가 (2026-09-04 종결)

**질문.** 무거운 3개 데이터셋(AlexaMeadeExhibit, CoffeeMartini, FlameSteak)이 SAM 3에서만 OOM으로 죽는다.
코드 결함인가, 모델 차이인가?

**답. 모델 규모 차이다. 코드 결함이 아니다.**

**근거.** 같은 알고리즘의 SAM 2판 `SCSam2/demo/sam2_demoVideoNew_maskSingleInputMVSeg.py`를 돌려
2025-10-15 당시 수치를 **정확히 재현**했습니다. 최대 약 40 GB에서 완주. 같은 데이터에서 SAM 3는 약 46 GB에서 OOM.
체크포인트: SAM 2 `sam2.1_hiera_large.pt` 857 MB vs SAM 3 약 3.3 GB.

**재조사 금지 이유.** 사용자 제안으로 수행한 대조 실험이고 결론이 명확합니다.
"코드가 잘못됐나?"로 되돌아가지 마십시오. 남은 길은 메모리 절감이지 버그 수정이 아닙니다.

---

## 3. `expandable_segments:True`가 원인인가 (2026-09-03 종결)

**질문.** `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` 때문에 OOM이 나는가?

**답. 아니다.** 이 옵션은 단편화를 줄이는 방향이라 오히려 도움이 됩니다. OOM은 실제 상주 텐서 총량 때문입니다.

**설정 위치** (참고용):

| 파일 | 방식 |
|---|---|
| `SCSam3/runMVSeg.py:33` | `setdefault` (외부에서 덮어쓸 수 있음) |
| `SCSam3/runMVSegForSam2.py:23` | `setdefault` |
| `SCSam2/demo/sam2_demoVideoNew_maskSingleInputMVSeg.py:13` | 무조건 대입 |
| `SCSam2/demo/sam2_MVSeg_recheck.py:13` | 무조건 대입 |
| `SCSam3/demo/sam2_demoVideoNew_maskSingleInputMVSeg.py:13` | 무조건 대입 |

러너 쪽은 `setdefault`라 `PYTORCH_CUDA_ALLOC_CONF=` 를 넣어 실행하면 끌 수 있습니다.

---

## 4. 시점 전파에 시작 인덱스를 명시하면 안 된다 (2026-09-03 종결) ★ 함정

**질문.** 시점 전파 요청의 키 이름이 맞지 않아 보인다. 시작 인덱스를 명시적으로 넘겨 "고쳐야" 하는가?

**답. 절대 아니다. 넘기면 결과가 망가진다.**

**근거.** 기준 시점이 마지막 인덱스인 경우 전파가 깨집니다. Blocks 기준 **J&F 0.7484 → 0.2827**.
Painter도 같은 방향으로 하락했습니다.

**현재 방어 장치.** `SCSam3/runMVSeg.py:37`

```python
SPATIAL_START_IMPLICIT = os.environ.get("SPATIAL_START_IMPLICIT", "1") == "1"
```

기본값이 켜짐(암묵적)이고, `:208`에서 이 값이 거짓일 때만 시작 인덱스를 넣습니다.

**재조사 금지 이유.** 이건 제가 "키 이름 불일치를 고친다"며 한 번 집어넣었다가 되돌린 회귀입니다.
코드를 읽다가 같은 불일치를 발견하더라도 고치지 마십시오. 의도된 것입니다.

---

## 5. `sam3` 네임스페이스 섀도잉 (2026-09-02 종결) ★ 함정

**질문.** 러너에서 모델 빌더가 `sam3.__file__ is None`으로 죽는다.

**답.** `SCSam3/sam3`는 upstream 체크아웃 디렉터리이고 **`__init__.py`가 없습니다**.
`SCSam3`가 `sys.path`에 있으면 이 디렉터리가 네임스페이스 패키지로 잡혀 진짜 `sam3` 패키지를 가립니다.

**해결.** `runMVSeg.py`의 `build_runner`에서 자기 디렉터리를 경로에서 제거:

```python
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != HERE]
sys.path.insert(0, algo_dir)
os.chdir(algo_dir)
```

**재조사 금지 이유.** 새 러너를 만들 때 이 세 줄을 빠뜨리면 같은 증상이 재발합니다. 증상만 보면 원인을 찾기 어렵습니다.

---

## 6. OneStage의 잘못된 평균값 0.7753 — 그리고 파일 이름이 거짓말을 한다 (2026-09-04 확정)

**질문.** OneStage 평균이 0.7753으로 낮게 나왔다.

**답. 항목 4의 시점 전파 시작 인덱스 회귀가 들어간 출력으로 계산한 stale 결과입니다.**

**★ 파일 이름을 믿지 마십시오.** `Data/MVSeg/jf_raw_sam3_onestage.STALE-prefix-bug.json` —
이름의 "prefix-bug"는 당시 제가 잘못 붙인 것입니다. 카메라 폴더명 자리수(`prefix1`)와 무관합니다.

이 오해는 실제로 재발했습니다. 2026-09-04 사실 확인에서 한 에이전트가
"Blocks와 Painter는 `prefix1: 0`인 유일한 두 데이터셋"이라며 이름이 맞다고 결론지었습니다. **틀렸습니다.**

| 가설 | 검증 | 판정 |
|---|---|---|
| `prefix1: 0`이 원인 | Barn·Breakfast·Carpark·Fencing·Frog·MATF·PoznanStreet도 전부 `prefix1: 0` (15개 중 9개) | **반증** |
| 기준시점이 마지막 인덱스일 때 깨짐 | Blocks(9/9), Painter(15/15) — **이 둘만** 마지막 | **확정** |

확인 명령:
```bash
python3 -c "
import json; c=json.load(open('SCSam3/demo/MVSeg.json'))
for n in ['Barn','Blocks','Painter','Frog','MATF']:
    print(n, 'prefix1=', c[n]['prefix1'], 'num_cam=', c[n].get('num_cam'))"
```

**증상.** 망가진 카메라는 `missing_files`가 (객체수 × 프레임수)와 **정확히 일치**했습니다 —
즉 마스크가 나쁜 게 아니라 **아예 생성되지 않았습니다.** 기준시점이 아닌 시점으로 전파가 되지 않은 것입니다.
기준 카메라(Blocks `cam9`, Painter `v15`)만 stale과 현재가 동일합니다.

| 데이터셋/카메라 | stale J | 현재 J | stale missing | 객체×프레임 |
|---|---|---|---|---|
| Blocks/cam0 | 0.0095 | 0.5753 | 420 | 20 × 21 = 420 |
| Blocks/cam4 | 0.0030 | 0.7716 | 336 | 16 × 21 = 336 |
| Painter/v0 | 0.0267 | 0.8303 | 525 | 25 × 21 = 525 |
| Painter/v6 | 0.0311 | 0.8440 | 162 | 26 × 21 = 546 |

45개 항목 중 41개는 stale과 현재가 바이트 동일합니다.

**현재 값.** `Data/MVSeg/jf_sam3_onestage.json` (OneStage 15개 평균 J&F **0.8427**).

**재조사 금지 이유.** 0.7753을 어디선가 보더라도 폐기된 값입니다.
파일을 지우지 않고 이름만 바꿔둔 이유가 이것인데, 그 이름이 오히려 오해를 낳았습니다.
이름은 그대로 두되 **이 문서가 정정본**입니다.

---

## 7. 45대 카메라 데이터셋의 GPU OOM (2026-09-02 종결)

**질문.** 카메라가 45대인 데이터셋에서 OneStage가 GPU OOM.

**답.** MVSeg는 3대만 채점합니다. 45대 전부에 시간 추적 세션을 만들 필요가 없습니다.

**해결.** `--track-cams written` — 출력이 기록되는 카메라만 추적.
`runMVSeg.py`에서 알고리즘별 기본값이 다릅니다: `NEEDS_ALL_VIEWS = ("OneStageNew", "MVOpt")`는 `all`, 나머지는 `written`.

---

## 8. 호스트 RAM 고갈로 인한 머신 재부팅 (2026-09-03 종결)

**질문.** CoffeeMartini, FlameSteak 실행 중 머신이 통째로 다운됐다.

**답. GPU가 아니라 호스트 RAM 고갈입니다.** 컨테이너에 상한이 없어 OOM killer가 아니라 시스템이 죽었습니다.

**해결.** `docker run --memory=90g`. 이후 재부팅 없음.

**부수 효과.** 항목 1의 환경 변화가 이 재부팅들과 같은 기간에 일어났습니다.

---

## 9. MVOpt = OneStageNew + 메모리 수정 (2026-09-04 종결)

**질문.** MVOpt가 정말 출력을 바꾸지 않는가?

**답. 바꾸지 않습니다. 실측으로 확인했습니다.**

| 대조 | 규모 | 차이 |
|---|---|---|
| MVOpt vs OneStageNew, 12개 데이터셋 (저장본끼리) | PNG 12,692장 | **0** |
| 카메라 단위 J·F 값 집합 | 36개 | **0** |
| 트림 명시 OFF vs 저장본(트림 ON), Blocks | PNG 882장 | **0** |
| **OneStageNew 신규 재실행 vs MVOpt 저장본** (2026-09-04) | PNG 12,428장 | **0** |

마지막 줄은 저장본 비교가 아니라 **OneStageNew를 처음부터 다시 돌려** 대조한 것입니다.
11개는 완전 일치, Welder는 아래 12번 사유로 중간에 죽었으나 기록된 677장이 전부 바이트 동일했습니다.

★ MVOpt 결과는 **전부 트림 ON**으로 만들어졌습니다(`runMVOptThree.sh:14`).
로그 디렉토리 접두어가 스크립트를 특정합니다 — `mvopt-*`는 `runMVOptThree.sh`(트림 ON),
`mvseg-*`는 `runMVSegAll.sh`(설정 안 함). 이걸 혼동해 "트림 증거 없음"으로 오판한 사례가 있습니다.

그리고 OneStageNew로는 불가능했던 3개 데이터셋이 MVOpt로 완주합니다.

---

## 10. ForSam2New에는 메모리 수정이 충분하지 않다 (2026-09-04 종결)

**질문.** 같은 메모리 수정을 `demoSCSam3ForSam2New`에 넣으면 무거운 3개가 도는가?

**답. 안 됩니다. 어제와 같은 지점에서 같은 이유로 죽습니다.**

| 데이터셋 | 실패 단계 | 사용량 |
|---|---|---|
| AlexaMeadeExhibit | 시점 전파 중 (33객체 프롬프트 직후) | 45.84 GiB |
| CoffeeMartini | 시점 전파 완료 후 시간 추적 진입 | 45.81 GiB |
| FlameSteak | 동일 | — |

**이유.** 이 폴더는 SAM 2 스타일 API라 파일 구조가 다릅니다.
GPU를 가장 많이 회수하는 두 수정(S1·S2)이 **구조적으로 적용 불가**입니다 —
`def add_tracker_new_mask`가 아예 없고, `add_tracker_new_points`도 `previous_stages_out`을 건드리지 않으며,
S2가 패치하는 객체 등록 블록도 없습니다. 즉 부분 적용이 아니라 **앵커 자체가 존재하지 않습니다.**

실제로 들어간 것은 S3·S4·S5뿐이고, 이들은 주로 일시적 피크나 소량 상주분을 줄이는 것들이라
절감이 0.1 GiB 수준에 그쳤습니다. 폴더별 현황은 [memory-optimization.md](memory-optimization.md) 참조.

**안전성은 확인됨.** Frog 데이터셋에서 패치 전/후 315장 완전 일치.

**다음에 할 일** (아직 안 함). 이 폴더의 inference 경로를 따로 분석해 S1·S2에 대응하는 자리를 찾아야 합니다.
"수정을 복사하면 될 것"이라는 가정은 이미 반증됐으니 다시 시도하지 마십시오.

---

## 11-A. OneStageNew는 Welder를 90 GiB 안에서 완주하지 못합니다 (2026-09-04 발견)

**상황.** MVOpt 동등성 검증을 위해 OneStageNew를 12개 데이터셋에 다시 돌렸습니다.
11개는 통과했고 **Welder만 exit 137(cgroup OOM kill)** 로 죽었습니다.

| 항목 | 값 |
|---|---|
| 규모 | 카메라 46대, 2560×1920, 객체 16개 |
| 죽은 지점 | 시점 전파 완료 후 **시간 추적 중** (21프레임 중 14) |
| 상한 | `--memory=90g` (`runMVOptThree.sh`) |
| MVOpt | 같은 상한에서 **완주** |

**중요한 점.** Welder는 "완주 불가 3개"에 속하지 않습니다 — 기존 OneStageNew 출력이 존재합니다.
그 출력은 `logs/mvseg-20260903-114432/OneStageNew-Welder.log`, 즉 **메모리 상한이 없는 `runMVSegAll.sh`** 로 만들어졌습니다.

즉 OneStageNew는 Welder에서 90 GiB를 넘게 씁니다. 예전에 "성공"한 것은 상한이 없어서였고,
그것이 머신을 두 번 재부팅시킨 바로 그 조건입니다.

**이유는 알고 있습니다.** OneStageNew에는 `RetireSpatialPredictor`가 없어
약 3.2 GiB 교차시점 모델과 46시점 유사 비디오, 특징 캐시가 시간 추적 내내 상주합니다.
[operations.md](operations.md) 함정 5번 참조.

**결론.** 메모리 수정의 이득은 "불가능했던 3개"에 그치지 않습니다.
Welder처럼 이미 돌던 데이터셋도 **안전 한도 안으로** 들어옵니다.

---

## 11. 마스크 출력 폴더 구조 (2026-09-02 종결)

**답.** `<method>/<camera>/<frame>/<objectid>.png` — 객체 ID 폴더가 **프레임 아래**입니다.
GT만 `Mask/<camera>/<frame>.png` 형태의 그레이스케일(픽셀값 = 객체 ID)입니다.

기억과 다를 수 있어 적어둡니다. 실제 데이터로 확인한 것입니다.

---

## 12. `demoSCSam3ForSam2New` 러너 작성 시 걸린 것 (2026-09-03 종결)

- `LoadVideo_Folder_MVSeg`는 **죽은 코드**입니다. `init_state(video_path=...)`가 upstream에서 사라졌습니다.
  `LoadVideo_File`의 폴더 버전으로 대체했습니다 (`runMVSegForSam2.py`).
- `add_new_mask`는 **2차원 torch 텐서**를 단언합니다. numpy나 3차원을 넘기면 죽습니다:

```python
torch.from_numpy(((ref_gt == (i + 1)) * 255).astype(np.uint8))
```

---

## 13. 워크플로 스크립트의 함정 (2026-09-03 종결)

검증자 에이전트가 세션 한도로 죽으면 `null`이 됩니다. 아래처럼 쓰면 **죽은 검증자가 반박으로 집계**되어 진짜 발견이 조용히 사라집니다:

```javascript
const survives = ok.length > 0 && ok.every(v => !v.refuted)   // 틀림
```

살아 있는 표만 세십시오:

```javascript
const votes = all.filter(Boolean)
const survives = votes.filter(v => !v.refuted).length >= 2    // 맞음
```

---

## 아직 열려 있는 것은 여기 없음

미해결 항목은 [open-items.md](open-items.md)를 보십시오.
