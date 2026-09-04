# 이 폴더는 동결된 스냅샷입니다 (2026-09-04부터)

## 무엇인가

`demoSCSam3MVOpt`의 **바이트 동일 사본**입니다. `.py` 12개 전부 같습니다.

```bash
diff -rq --exclude=__pycache__ . ../demoSCSam3MVOpt   # 차이 없음
```

## 하지 말 것

**여기서 개발하지 마십시오. 개발은 `../demoSCSam3MVOpt`에서 합니다.**

이 폴더는 백업 역할입니다. 여기에 변경을 넣으면 두 폴더가 조용히 어긋납니다.

## 이름이 오해를 부릅니다 ★

폴더 이름이 `OneStageNew`이지만, **여기 있는 코드는 발표 결과를 만든 코드가 아닙니다.**

발표된 J&F 수치(15개 0.8451, 12개 공통 0.8497)를 만든 원본 코드는 git 태그에만 있습니다:

```bash
git show baseline-onestagenew:SCSam3/demoSCSam3OneStageNew/SCSam3Video.py
git checkout baseline-onestagenew -- SCSam3/demoSCSam3OneStageNew   # 원본 복원
```

`Data/MVSeg/*/SegMaskSam3OneStageNew/`의 마스크는 **원본 코드**가 만든 것입니다.
이 폴더의 현재 코드로 재생성하면 이름은 같지만 다른 코드의 출력이 됩니다
(출력은 실측상 동일하지만, 출처가 달라집니다).

## 왜 덮어썼나

메모리 수정 7종을 반영했습니다. 알고리즘 변화는 없고, 실측으로 확인했습니다:

| 대조 | 규모 | 차이 |
|---|---|---|
| 신규 재실행 vs MVOpt | PNG 12,428장 | 0 |
| Welder (복사 전 OOM → 복사 후 완주) | PNG 941장 | 0 |

효과: Welder가 90 GiB 상한 안에서 완주하고, 이전에 불가능하던 3개 데이터셋도 돌아갑니다.

## 삭제하면 안 되는 것

숫자 폴더 `0`~`31` (PNG 1,600장, 13 GB)은 `figures/make_figure.py`와
`make_baseline_figure.py`가 읽습니다. **`.gitignore`의 `*.png` 때문에 git 복구가 불가능합니다.**

폴더 단위 동기화(`rsync --delete`, `rm -rf` 후 `cp -r`)를 하면 이것들이 사라집니다.
`.py` 파일만 복사하십시오.

---

배경: `docs/README.md`, `docs/memory-optimization.md`, `docs/investigations-closed.md`
