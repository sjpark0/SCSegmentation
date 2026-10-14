"""Upper bounds / risk grid for an ADD-ONLY composition in the reference-invisible region (E12 design), from the
visibility-signal analyst's keyframe_vis.json (per pair: GT, hit, FP split by vis/outside/occluded/nodepth) for the
adopted seeds geoset_pred_ft_rep_x_e11.  J only (F needs pixels).  union = gt + fp (adding missed GT pixels keeps it)."""
import json, sys, statistics as st
p = sys.argv[1]
s = open(p).read(); s = s[s.index('{'):]
rows = json.loads(s)["rows"]
SEED = "geoset_pred_ft_rep_x_e11"
def grp(r):
    g = []
    if r["JF_e8"] >= 0.8: g.append("good605")
    else:
        g.append("fail59")
        if r["kind"].startswith("under"): g.append("under22")
    g.append("all664")
    return g
def dJ(r, rec_out, rec_occ, rec_nod, fp_growth):
    """recover a fraction of the missed GT per invisible class; invisible FP grows by fp_growth x (current invisible FP)."""
    e = r["seeds"][SEED]
    gt, fp = r["gt_px"], e["fp_px"]
    hit = e["hit_px"]
    miss_out = r["gt_outside"] - e["hit_outside"]; miss_occ = r["gt_occluded"] - e["hit_occluded"]; miss_nod = r["gt_nodepth"] - e["hit_nodepth"]
    fp_inv = e["fp_outside"] + e["fp_occluded"] + e["fp_nodepth"]
    add_tp = rec_out * miss_out + rec_occ * miss_occ + rec_nod * miss_nod
    add_fp = fp_growth * fp_inv
    J0 = hit / max(gt + fp, 1); J1 = (hit + add_tp) / max(gt + fp + add_fp, 1)
    return J1 - J0
groups = {}
for r in rows:
    for g in grp(r): groups.setdefault(g, []).append(r)
print({g: len(v) for g, v in groups.items()})
# sanity: J reproduces
mx = max(abs(r["seeds"][SEED]["hit_px"] / max(r["gt_px"] + r["seeds"][SEED]["fp_px"], 1) - r["seeds"][SEED]["J"]) for r in rows)
print("max |J recomputed - J| =", f"{mx:.2e}")
print("\n== oracle: add ALL missed invisible GT (dJ mean over group), by class ==")
for g in ("all664", "good605", "fail59", "under22"):
    v = groups[g]
    full = st.mean(dJ(r, 1, 1, 1, 0) for r in v)
    out = st.mean(dJ(r, 1, 0, 0, 0) for r in v); occ = st.mean(dJ(r, 0, 1, 0, 0) for r in v); nod = st.mean(dJ(r, 0, 0, 1, 0) for r in v)
    print(f"  {g:8s} n={len(v):3d}  all-invisible {full:+.4f}   outside {out:+.4f}  occluded {occ:+.4f}  nodepth {nod:+.4f}")
print("\n== risk grid: dJ mean when a fraction REC of every invisible missed class is recovered and invisible FP grows by G x ==")
for g in ("all664", "good605", "under22"):
    v = groups[g]
    print(f"  {g}")
    print("     REC\\G " + " ".join(f"{G:>8.2f}" for G in (0, 0.25, 0.5, 1.0, 2.0)))
    for rec in (0.0, 0.1, 0.2, 0.3, 0.5):
        print(f"     {rec:4.1f}  " + " ".join(f"{st.mean(dJ(r, rec, rec, rec, G) for r in v):+8.4f}" for G in (0, 0.25, 0.5, 1.0, 2.0)))
print("\n== pixel totals (e11 seeds) ==")
for g in ("all664", "good605", "under22"):
    v = groups[g]
    e = lambda k: sum(r["seeds"][SEED][k] for r in v)
    gt = sum(r["gt_px"] for r in v); gi = sum(r["gt_outside"] + r["gt_occluded"] + r["gt_nodepth"] for r in v)
    hi = e("hit_outside") + e("hit_occluded") + e("hit_nodepth"); fi = e("fp_outside") + e("fp_occluded") + e("fp_nodepth")
    print(f"  {g}: GT {gt:,} invisible {gi:,} ({gi/gt:.3f}); missed invisible {gi-hi:,}; FP {e('fp_px'):,} of which invisible {fi:,} ({fi/max(e('fp_px'),1):.3f}); FP_vis {e('fp_vis'):,}")
# break-even: for each group, the FP growth G at which REC=0.2 gives dJ = 0
print("\n== per-pair: pairs whose dJ < -0.01 when G = 1.0 and REC = 0 (pure FP doubling), by group ==")
for g in ("all664", "good605", "under22"):
    v = groups[g]
    n = sum(dJ(r, 0, 0, 0, 1.0) < -0.01 for r in v)
    print(f"  {g}: {n} pairs")
