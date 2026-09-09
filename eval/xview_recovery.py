#!/usr/bin/env python3
"""E3: does reading the neighbour view during tracking recover objects a single view lost?

Implements docs/xview-recovery-prereg.md section 1 verbatim.  Host, stdlib only; reads an
eval_jf.py raw file with per_frame scores.

    python3 eval/xview_recovery.py --raw Data/MVSeg/jf_paper.json \
        --control SegMaskSam3XW0 --treat SegMaskSam3XW1GPS4 --ref-rule maxid

Definitions (per (camera, object), frames with gt_area > 0 only):
  HELD   J > 0.5          LOST  J < 0.1
  A "disagreement stretch" is a maximal run of >= MIN_RUN consecutive GT-present frames
  on which one method is LOST, with the other method HELD on >= MIN_HELD of those frames.
  Direction = the method that holds.  treat holds -> RECOVERED; control holds -> INDUCED.
  Kinds (reporting only): ENTRY (no GT at frame 0 and the stretch starts at the first
  GT-present frame), OCCLUSION (the losing method held the object before the stretch),
  NEVER (GT from frame 0, never held before the stretch).
"""
import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from eval.report_jf import Store, load, MUVOD_ORDER  # noqa: E402

HELD, LOST = 0.5, 0.1
MIN_RUN, MIN_HELD = 3, 2


def runs(mask):
    """Maximal runs of True in a boolean list -> [(start, end_exclusive)]."""
    out, start = [], None
    for i, m in enumerate(mask + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


def stretches(J_lose, J_hold, gt):
    """Disagreement stretches where `lose` is LOST and `hold` is HELD (prereg 1)."""
    present = [g > 0 for g in gt]
    lost = [present[t] and J_lose[t] < LOST for t in range(len(gt))]
    out = []
    for s, e in runs(lost):
        if e - s < MIN_RUN:
            continue
        held = sum(1 for t in range(s, e) if J_hold[t] > HELD)
        if held < MIN_HELD:
            continue
        first_gt = next((t for t in range(len(gt)) if present[t]), None)
        if not present[0] and s == first_gt:
            kind = "entry"
        elif any(present[t] and J_lose[t] > HELD for t in range(0, s)):
            kind = "occlusion"
        else:
            kind = "never"
        gain = sum(J_hold[t] - J_lose[t] for t in range(s, e)) / (e - s)
        out.append(dict(start=s, end=e, held=held, kind=kind, gain=gain))
    return out


def binom_two_sided(k, n):
    """Exact two-sided binomial p for k successes of n at p=0.5."""
    if n == 0:
        return float("nan")
    p_k = lambda x: math.comb(n, x) / 2 ** n
    obs = p_k(k)
    return min(1.0, sum(p_k(x) for x in range(n + 1) if p_k(x) <= obs + 1e-15))


def analyse(store, control, treat, ref_rule, scenes=None):
    events = []
    zero_nb_disagreements = 0
    for d in [x for x in (scenes or MUVOD_ORDER) if x in store.datasets]:
        basic = set(store.ref_entry(d, ref_rule)["seed_ids"])
        for cam in store.cams[d]:
            e0, e1 = store.entry(d, cam, control), store.entry(d, cam, treat)
            if e0 is None or e1 is None:
                continue
            nb0 = store.view_index(d, cam) == 0
            for i, obj in enumerate(e0["objects"]):
                if obj not in basic or obj not in e1["objects"]:
                    continue
                J0 = e0["result"]["per_frame"]["J"][i]
                J1 = e1["result"]["per_frame"]["J"][e1["objects"].index(obj)]
                gt = e0["result"]["per_frame"]["gt_area"][i]
                for st in stretches(J0, J1, gt):
                    events.append(dict(scene=d, cam=cam, obj=obj, direction="recovered", **st))
                    zero_nb_disagreements += nb0
                for st in stretches(J1, J0, gt):
                    events.append(dict(scene=d, cam=cam, obj=obj, direction="induced", **st))
                    zero_nb_disagreements += nb0
    return events, zero_nb_disagreements


def summarise(label, events, zero_nb):
    rec = [e for e in events if e["direction"] == "recovered"]
    ind = [e for e in events if e["direction"] == "induced"]
    n, k = len(events), len(rec)
    p = binom_two_sided(k, n)
    print(f"== {label} ==")
    print(f"  갈림 구간 {n}개: 되찾음 {k}, 새 유실 {len(ind)}   정확 이항검정(양측) p = {p:.4f}")
    if n >= 5 and p < 0.05 and k > len(ind):
        verdict = "확인"
    elif n < 5 and len(ind) == 0:
        verdict = "지지·검정력 부족"
    elif len(ind) >= k:
        verdict = "기각"
    else:
        verdict = "미확인"
    print(f"  판정: {verdict}")
    kinds = {}
    for e in rec:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"  되찾음 종류: " + ", ".join(f"{k_}={v}" for k_, v in sorted(kinds.items())) +
          (f"   구간당 평균 J 이득 {sum(e['gain'] for e in rec)/len(rec):+.3f}" if rec else ""))
    print(f"  감시값: 이웃 없는 카메라(뷰 0)의 갈림 구간 = {zero_nb} (0이어야 함)")
    for e in sorted(events, key=lambda x: (x["direction"], -x["gain"])):
        print(f"    {e['direction']:9} {e['scene']:18}{e['cam']:13}obj {e['obj']:>3}  "
              f"frames {e['start']:>2}-{e['end']-1:<2} held {e['held']}/{e['end']-e['start']}  "
              f"{e['kind']:9} gain {e['gain']:+.3f}")
    return dict(n=n, recovered=k, induced=len(ind), p=p, verdict=verdict,
                events=events, zero_nb=zero_nb)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--treat", required=True)
    ap.add_argument("--ref-rule", choices=("maxid", "count", "muvod"), required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--dump", default=None, help="write the events as JSON")
    args = ap.parse_args()
    store = Store(load(args.raw))
    events, zero_nb = analyse(store, args.control, args.treat, args.ref_rule, args.datasets)
    res = summarise(args.label or f"{args.treat} vs {args.control} ({args.ref_rule})", events, zero_nb)
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
