#!/usr/bin/env python3
"""Read the P5 pre-registered decision rule off the finished run (host, stdlib only).

docs/phase5-seed-repair-prereg.md sections 3-4, verbatim:

  (a) E-P5-1   Welder camera_0004 obj 8 J_all >= 0.8 AND obj 14 J_all >= 0.8
  (b) E-P5-0   17-scene paired mean delta >= 0 AND no scene delta < -0.005 (MUVOD basic)
  (c) S1       scenes with zero flags are byte-identical to the control
  S2           the 14 in-scope true degenerates are all flagged (from the manifests)
  S4           the reference view's seeds never change

Prints every reading, then ADOPT / DO NOT ADOPT with the clause that failed.  It does
not compute J&F itself: point --raw at an eval_jf.py file that scores both methods.

    python3 eval/p5_judge.py --raw Data/MVSeg/jf_p5.json
"""
import argparse
import filecmp
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from eval.report_jf import Store, load, Cfg, MUVOD_CFG, MUVOD_ORDER, score_all, mean  # noqa: E402

DATA = os.path.join(REPO, "Data", "MVSeg")
CONTROL, TREAT = "SegMaskSam3XW1GPS4M", "SegMaskSam3XW1GPS4MRp"
# the 15 true degenerates of the c_ini census (phase5-seed-repair-prep.md section 2);
# MATF S1_CAM_1 obj 26 is misaligned and out of the detector's scope by design
TRUE_DEGENERATE = [
    ("AlexaMeadeExhibit", "camera_0004", 17), ("AlexaMeadeExhibit", "camera_0004", 21),
    ("AlexaMeadeExhibit", "camera_0004", 24), ("Blocks", "cam0", 4), ("Blocks", "cam0", 14),
    ("Blocks", "cam9", 3), ("Breakfast", "v5", 19), ("Breakfast", "v5", 24),
    ("Breakfast", "v9", 24), ("CBABasketball", "v06", 20), ("MATF", "S1_CAM_10", 22),
    ("Painter", "v0", 9), ("Welder", "camera_0004", 8), ("Welder", "camera_0004", 14),
]
OUT_OF_SCOPE = [("MATF", "S1_CAM_1", 26), ("Welder", "camera_0004", 12)]


CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")


def loaded_cameras(scene):
    """The runner's view index -> camera name, from the config (perms + prefix).  The
    manifest's `cameras` field lists only the written cameras, so it cannot map the
    view indices that seed_repair records."""
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    d = cfg.get(scene) or next(v for v in cfg.values() if v.get("folder") == scene)
    perms = d.get("perms") or list(range(d["start_cam"], d["num_cam"] + d["start_cam"]))
    return [f"{d['prefix']}{x:0{d['prefix1']}d}" for x in perms]


def manifest(scene, method):
    p = os.path.join(DATA, scene, method, "MANIFEST.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def tree_identical(a, b):
    """diff -rq -x MANIFEST.json, in Python."""
    for root, _, files in os.walk(a):
        rel = os.path.relpath(root, a)
        for f in files:
            if f == "MANIFEST.json":
                continue
            pa, pb = os.path.join(root, f), os.path.join(b, rel, f)
            if not os.path.isfile(pb) or not filecmp.cmp(pa, pb, shallow=False):
                return False
    for root, _, files in os.walk(b):
        rel = os.path.relpath(root, b)
        for f in files:
            if f != "MANIFEST.json" and not os.path.isfile(os.path.join(a, rel, f)):
                return False
    return True


def j0_of(store, scene, cam, method, obj):
    """J on the seed frame alone: did the seed repair itself land, regardless of what
    the temporal tracker does afterwards (prereg section 8, E-P5-1b)."""
    e = store.entry(scene, cam, method)
    if e is None or obj not in e["objects"]:
        return None
    return e["result"]["per_frame"]["J"][e["objects"].index(obj)][0]


def j_of(store, scene, cam, method, obj):
    e = store.entry(scene, cam, method)
    if e is None or obj not in e["objects"]:
        return None
    return e["result"]["J_all"][e["objects"].index(obj)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(DATA, "jf_p5.json"))
    ap.add_argument("--control", default=CONTROL)
    ap.add_argument("--treat", default=TREAT)
    args = ap.parse_args()
    store = Store(load([args.raw]))
    scenes = [d for d in MUVOD_ORDER if d in store.datasets]
    verdict = {}

    # ---------------------------------------------------------------- manifests
    print("== 매니페스트: 플래그와 복구 ==")
    flags, cam_of_view, ref_changed = {}, {}, []
    for d in scenes:
        m = manifest(d, args.treat)
        rep = (m or {}).get("provenance", {}).get("seed_repair")
        if not m or rep is None:
            print(f"  {d:20} 매니페스트에 seed_repair 없음"); flags[d] = None; continue
        cams = m.get("cameras") or []
        n_flag = sum(r["flagged"] for r in rep["rounds"][:1])         # round-1 flags
        n_rep = sum(len(r["repairs"]) for r in rep["rounds"])
        n_unrep = sum(len(r["unrepairable"]) for r in rep["rounds"][:1])
        flags[d] = n_flag
        print(f"  {d:20} flagged {n_flag:>3}  repaired {n_rep:>3}  unrepairable {n_unrep:>2}  "
              f"rounds {len(rep['rounds'])}")
        for r in rep["rounds"]:
            for x in r["repairs"]:
                if x["view"] == rep["ref_view"]:
                    ref_changed.append((d, x))
    verdict["S4 reference seeds untouched"] = not ref_changed

    # ---------------------------------------------------------------- (a)
    print("\n== (a) E-P5-1: Welder camera_0004 obj 8 / 14 (J_all) ==")
    a_ok = True
    for obj in (8, 14):
        j0 = j_of(store, "Welder", "camera_0004", args.control, obj)
        j1 = j_of(store, "Welder", "camera_0004", args.treat, obj)
        ok = j1 is not None and j1 >= 0.8
        a_ok &= ok
        print(f"  obj {obj:>2}: {j0!s:>7} -> {j1!s:>7}   {'ok' if ok else 'FAIL (< 0.8)'}")
    verdict["(a) E-P5-1"] = a_ok
    print("  E-P5-1b (보조, 시드 프레임의 J만 — 시드 복구 자체가 됐는가):")
    for obj in (8, 14):
        print(f"    obj {obj:>2}: {j0_of(store, 'Welder', 'camera_0004', args.control, obj)!s:>7} -> "
              f"{j0_of(store, 'Welder', 'camera_0004', args.treat, obj)!s:>7}")
    print("  감지 범위 밖 (보고만):")
    for d, cam, obj in OUT_OF_SCOPE:
        print(f"    {d} {cam} obj {obj}: {j_of(store, d, cam, args.control, obj)!s:>7} -> "
              f"{j_of(store, d, cam, args.treat, obj)!s:>7}")

    # ---------------------------------------------------------------- (b)
    print("\n== (b) E-P5-0: MUVOD basic, 장면 짝지음 ==")
    cfg = Cfg(**MUVOD_CFG).replace(objects="basic")
    ds, _ = score_all(store, [args.control, args.treat], scenes, cfg)
    deltas = []
    for d in scenes:
        if (d, args.control) in ds and (d, args.treat) in ds:
            c, t = ds[(d, args.control)][0], ds[(d, args.treat)][0]
            deltas.append((d, c, t, t - c))
    worst = min(deltas, key=lambda x: x[3]) if deltas else None
    md = mean([x[3] for x in deltas]) if deltas else float("nan")
    for d, c, t, dl in deltas:
        print(f"  {d:20} {100*c:6.1f} -> {100*t:6.1f}  {100*dl:+6.2f}"
              + ("   <-- < -0.5" if dl < -0.005 else ""))
    print(f"  mean delta {100*md:+.3f} pts over {len(deltas)} scenes; worst {worst[0]} {100*worst[3]:+.2f}")
    verdict["(b) E-P5-0"] = bool(deltas) and md >= 0 and worst[3] >= -0.005

    # ---------------------------------------------------------------- (c) S1
    print("\n== (c) S1: 플래그 0인 장면은 바이트 동일 ==")
    c_ok = True
    zero = [d for d in scenes if flags.get(d) == 0]
    for d in zero:
        same = tree_identical(os.path.join(DATA, d, args.control), os.path.join(DATA, d, args.treat))
        c_ok &= same
        print(f"  {d:20} {'identical' if same else 'DIFFERS'}")
    if not zero:
        print("  (플래그 0인 장면 없음)")
    verdict["(c) S1"] = c_ok

    # ---------------------------------------------------------------- S2
    print("\n== S2: 진짜 퇴화 14개가 전부 플래그됐는가 ==")
    hit = 0
    for d, cam, obj in TRUE_DEGENERATE:
        m = manifest(d, args.treat)
        rep = (m or {}).get("provenance", {}).get("seed_repair") or {}
        names = loaded_cameras(d)
        found, before, after = False, None, None
        for r in rep.get("rounds", []):
            for x, a in zip(r["repairs"], r["after"]):
                if x["obj"] == obj and names[x["view"]] == cam:
                    found, before, after = True, x["area"], a["area"]
            for x in r["unrepairable"]:
                if x["obj"] == obj and names[x["view"]] == cam:
                    found, before = True, x["area"]
        hit += found
        j0 = j_of(store, d, cam, args.control, obj)
        j1 = j_of(store, d, cam, args.treat, obj)
        print(f"  {d:20} {cam:14} obj {obj:>3}  {'flagged    ' if found else 'NOT flagged'}"
              f"  seed {before!s:>6} -> {after!s:>6} px   J {j0!s:>6} -> {j1!s:>6}")
    print(f"  {hit}/14")
    verdict["S2 14/14 flagged"] = hit == 14

    # ---------------------------------------------------------------- verdict
    print("\n== 판정 (사전 등록 §3) ==")
    for k, v in verdict.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    adopt = verdict["(a) E-P5-1"] and verdict["(b) E-P5-0"] and verdict["(c) S1"]
    print("\n  ==> " + ("ADOPT: headline becomes " + args.treat if adopt else "DO NOT ADOPT"))
    if not verdict["(a) E-P5-1"] and verdict["(b) E-P5-0"] and verdict["(c) S1"]:
        print("      (복구는 다른 곳에서 작동하지만 표적은 못 고쳤다 — Welder 서술은 쓰지 않음)")
    return 0 if adopt else 1


if __name__ == "__main__":
    sys.exit(main())
