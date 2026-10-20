"""D1 transitions (round 1 -> round 2 raw -> round 2 + exclusivity), swap-set overlap with the preset, round-2 memory
gate on swapped round-1 masks; writes metrics.json (per-arm summary) and pairs.json (per-pair rows).

    analyze.py   (reads scores_raw.json written by score.py)
"""
import json
import os
from collections import Counter

import numpy as np

D15 = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(D15, "data", "T06_flag_ganggang_F", "Result")
S = json.load(open(os.path.join(D15, "scores_raw.json")))
rows = {a: {(r["cam"], r["p"]): r for r in R} for a, R in S["rows"].items()}


def status(r):
    return "ok" if r["covers"] == r["p"] else ("empty" if r["covers"] == 0 else "wrong")


def transitions(a, b, far_only=True):
    t = Counter()
    for k, ra in rows[a].items():
        if far_only and ra["d"] < 7:
            continue
        t[f"{status(ra)}->{status(rows[b][k])}"] += 1
    return dict(sorted(t.items()))


def swaps(a, far_only=False):
    return {(k, r["covers"]) for k, r in rows[a].items() if status(r) == "wrong" and (not far_only or r["d"] >= 7)}


out = {"arms": S["arms"], "d1": {}, "swap_overlap_with_preset": {}, "per_pair_vs_chain": {}}
# ---- D1: where do the swaps come from?
if "preset_r1" in rows and "xnone" in rows:
    out["d1"]["far_transitions_r1_to_final"] = transitions("preset_r1", "preset")
    out["d1"]["far_transitions_r1_to_r2raw"] = transitions("preset_r1", "xnone")
    out["d1"]["far_transitions_r2raw_to_final"] = transitions("xnone", "preset")
    s1, s2 = swaps("preset_r1"), swaps("preset")
    out["d1"]["swaps_r1"] = len(s1)
    out["d1"]["swaps_final"] = len(s2)
    out["d1"]["swaps_same_pair_and_same_wrong_person"] = len(s1 & s2)
    out["d1"]["swaps_final_new_in_round2"] = sorted(f"{c}/p{p}->p{q}" for (c, p), q in s2 - s1)
    out["d1"]["swaps_r1_fixed_by_round2"] = sorted(f"{c}/p{p}->p{q}" for (c, p), q in s1 - s2)
    # round-2 memory gate on the round-1 masks (round1_conf.json of the preset run; reproj rule as in run.py)
    conf = json.load(open(os.path.join(RES, "preset", "round1_conf.json")))

    def passes(c, p):
        e = conf[c][str(p)]
        if e["presence"] <= 0:
            return False
        return e["iou"] >= 0.7 if e["overlap"] is None else e["overlap"] >= 0.5
    w = [(k, r) for k, r in rows["preset_r1"].items() if status(r) == "wrong"]
    ok = [(k, r) for k, r in rows["preset_r1"].items() if status(r) == "ok"]
    out["d1"]["r1_wrong_masks_passing_reproj_gate"] = f"{sum(passes(*k) for k, _ in w)}/{len(w)}"
    out["d1"]["r1_correct_masks_passing_reproj_gate_near"] = \
        f"{sum(passes(*k) for k, r in ok if r['d'] <= 6)}/{sum(r['d'] <= 6 for _, r in ok)}"
    out["d1"]["r1_correct_masks_passing_reproj_gate_far"] = \
        f"{sum(passes(*k) for k, r in ok if r['d'] >= 7)}/{sum(r['d'] >= 7 for _, r in ok)}"
# ---- swap-set overlap and per-pair comparison
base = swaps("preset")
for a in rows:
    sa = swaps(a)
    out["swap_overlap_with_preset"][a] = dict(n=len(sa), shared=len(sa & base), only_here=len(sa - base),
                                              preset_only=len(base - sa))
    if "nas_chain" in rows and a != "nas_chain":
        dJ = [rows[a][k]["J"] - rows["nas_chain"][k]["J"] for k in rows[a]]
        out["per_pair_vs_chain"][a] = dict(better=int(sum(x > 0.01 for x in dJ)), worse=int(sum(x < -0.01 for x in dJ)))
json.dump(out, open(os.path.join(D15, "metrics.json"), "w"), indent=1)
json.dump({a: list(r.values()) for a, r in rows.items()}, open(os.path.join(D15, "pairs.json"), "w"), indent=0)
print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
