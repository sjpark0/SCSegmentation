#!/usr/bin/env python3
"""Read the E4 pre-registered decision rule off the finished run (host, stdlib only).

docs/phase3-direction-cini-prereg.md sections 2-3, verbatim:

  (a) E4-1   the 9 cameras that had NO neighbour under mode A: paired delta > 0 and the
             cluster bootstrap 95% CI excludes 0                       [mechanism]
  (b) E4-0   17-scene paired mean delta >= 0 AND no scene delta < -0.005
  S4         the reference camera's J&F changes by no more than 0.005

Both must pass to adopt.  Prints every reading, then ADOPT / DO NOT ADOPT.

    python3 eval/e4_judge.py --raw Data/MVSeg/jf_e4.json
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from eval.report_jf import (Store, load, Cfg, MUVOD_CFG, MUVOD_ORDER, score_all,  # noqa: E402
                            mean, boot_ci)

CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CONTROL, TREAT = "SegMaskSam3XW1GPS4M", "SegMaskSam3XW1CGPS4M"


def view_index_map(scene):
    """camera name -> view index in the loaded camera ring, plus the c_ini index."""
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    d = cfg.get(scene) or next(v for v in cfg.values() if v.get("folder") == scene)
    perms = d.get("perms") or list(range(d["start_cam"], d["num_cam"] + d["start_cam"]))
    names = {f"{d['prefix']}{x:0{d['prefix1']}d}": perms.index(x) for x in perms}
    return names, perms.index(d["c_ini"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(REPO, "Data", "MVSeg", "jf_e4.json"))
    ap.add_argument("--control", default=CONTROL)
    ap.add_argument("--treat", default=TREAT)
    args = ap.parse_args()
    store = Store(load([args.raw]))
    scenes = [d for d in MUVOD_ORDER if d in store.datasets]
    cfg = Cfg(**MUVOD_CFG).replace(objects="basic")
    ds, cs = score_all(store, [args.control, args.treat], scenes, cfg)

    # ---------------------------------------------------------------- camera groups
    groups = {"no neighbour (view 0)": [], "below c_ini": [], "toward c_ini": []}
    ref_deltas = []
    for d in scenes:
        vmap, ci = view_index_map(d)
        for cam in store.cams[d]:
            if (d, cam, args.control) not in cs or (d, cam, args.treat) not in cs:
                continue
            v = vmap[cam]
            delta = cs[(d, cam, args.treat)][0] - cs[(d, cam, args.control)][0]
            if v == ci:
                ref_deltas.append((d, cam, delta))
            if v == 0:
                groups["no neighbour (view 0)"].append((d, cam, delta))
            elif v < ci:
                groups["below c_ini"].append((d, cam, delta))
            else:
                groups["toward c_ini"].append((d, cam, delta))

    print("== 카메라 묶음별 Δ (MUVOD basic, C − A) ==")
    for name, rows in groups.items():
        if not rows:
            print(f"  {name:24} (없음)")
            continue
        by_scene = {}
        for d, cam, dl in rows:
            by_scene.setdefault(d, []).append(dl)
        lo, hi = boot_ci(list(by_scene.values()))
        m = mean([dl for _, _, dl in rows])
        print(f"  {name:24} n={len(rows):>2}  평균 {m:+.4f}  클러스터 CI [{lo:+.4f}, {hi:+.4f}]"
              f"  상승 {sum(1 for _,_,x in rows if x>1e-6)} 하락 {sum(1 for _,_,x in rows if x<-1e-6)}")

    # ---------------------------------------------------------------- (a) E4-1
    nb0 = groups["no neighbour (view 0)"]
    by_scene = {}
    for d, cam, dl in nb0:
        by_scene.setdefault(d, []).append(dl)
    lo, hi = boot_ci(list(by_scene.values())) if nb0 else (float("nan"),) * 2
    m = mean([dl for _, _, dl in nb0]) if nb0 else float("nan")
    a_ok = bool(nb0) and m > 0 and lo > 0
    print(f"\n== (a) E4-1: A에서 이웃이 없던 카메라 {len(nb0)}대 ==")
    for d, cam, dl in sorted(nb0, key=lambda r: r[2]):
        print(f"  {d:20}{cam:14}{dl:+.4f}")
    print(f"  평균 {m:+.4f}, 클러스터 95% CI [{lo:+.4f}, {hi:+.4f}]  -> {'PASS' if a_ok else 'FAIL'}")

    # ---------------------------------------------------------------- (b) E4-0
    deltas = [(d, ds[(d, args.control)][0], ds[(d, args.treat)][0]) for d in scenes
              if (d, args.control) in ds and (d, args.treat) in ds]
    print(f"\n== (b) E4-0: MUVOD basic 17장면 ==")
    worst = None
    for d, c, t in sorted(deltas, key=lambda r: r[2] - r[1]):
        dl = t - c
        worst = worst or (d, dl)
        print(f"  {d:20}{100*c:6.1f} -> {100*t:6.1f}  {100*dl:+6.2f}"
              + ("   <-- < -0.5" if dl < -0.005 else ""))
    md = mean([t - c for _, c, t in deltas])
    b_ok = bool(deltas) and md >= 0 and all(t - c >= -0.005 for _, c, t in deltas)
    print(f"  평균 Δ {100*md:+.3f} pts over {len(deltas)} scenes -> {'PASS' if b_ok else 'FAIL'}")

    # ---------------------------------------------------------------- S4
    s4 = all(abs(dl) <= 0.005 for _, _, dl in ref_deltas)
    print(f"\n== S4: 기준 카메라(c_ini)의 변화 ==")
    for d, cam, dl in sorted(ref_deltas, key=lambda r: r[2])[:5]:
        print(f"  {d:20}{cam:14}{dl:+.4f}")
    print(f"  최대 |Δ| {max((abs(x) for _,_,x in ref_deltas), default=0):.4f} -> "
          f"{'PASS' if s4 else 'FAIL'}")

    print("\n== 판정 (사전 등록 §2) ==")
    print(f"  {'PASS' if a_ok else 'FAIL'}  (a) E4-1 메커니즘")
    print(f"  {'PASS' if b_ok else 'FAIL'}  (b) E4-0 전체")
    print(f"  {'PASS' if s4 else 'FAIL'}  S4 기준 카메라")
    adopt = a_ok and b_ok
    print("\n  ==> " + ("ADOPT: 헤드라인을 모드 C로" if adopt else "DO NOT ADOPT"))
    if not a_ok:
        print("      (a) 실패 -> P4의 'A 유지'가 c_ini에서도 유효. 이 줄기를 닫습니다.")
    elif not b_ok:
        print("      (a) 통과·(b) 실패 -> 방향은 그 카메라들을 고치지만 다른 곳을 망가뜨립니다.")
    return 0 if adopt else 1


if __name__ == "__main__":
    sys.exit(main())
