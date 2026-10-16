"""(1) Where the S2-R6 +0.3645 came from (old headline E8 base vs Rs; GT used: J&F of frames t, copy-GT0 J&F)."""
from common import *
ref = {k: f for k, f in S.items() if f["ref"]}
objs_ref = sorted({k[:3] for k in ref})
def cat(k):
    c = jf(A[k], "Jc", "Fc")
    return "static" if c >= 0.98 else "near" if c >= 0.9 else "moving"
gain = {k: W[k[:3]] * (jf(A[k], "Jc", "Fc") - jf(A[k])) for k in ref}    # gain if snapped (copy - tracker)
sn = {k for k, f in ref.items() if snap(f)}
tot = sum(gain[k] for k in sn); print(f"total gain {tot:+.4f}  snaps {len(sn)}/{len(ref)}")
# per scene
print("\n== per scene: objframes snaps snap% gain(x1 = contribution to 17-scene mean) per-scene-delta(x17) neg-snaps loss_before(ref) loss_after")
for sc in SCENES:
    ks = [k for k in ref if k[0] == sc]; s = [k for k in ks if k in sn]
    g = sum(gain[k] for k in s); neg = sum(1 for k in s if gain[k] < 0)
    lb = sum(W[k[:3]] * (1 - jf(A[k])) for k in A if k[0] == sc and A[k]["ref"])
    print(f"{sc:20s} {len(ks):5d} {len(s):5d} {100*len(s)/len(ks):5.1f}% {g:+.4f} {17*g:+.3f} neg {neg:4d} lossref {lb:.3f} -> {lb-g:.3f}")
# by GT category
print("\n== by GT copy category (copy J&F of GT0 vs GT_t): frames, snapped, gain from snaps, sum loss tracker, sum copy-loss")
for c in ("static", "near", "moving"):
    ks = [k for k in ref if cat(k) == c]; s = [k for k in ks if k in sn]
    print(f"{c:7s} n {len(ks):5d} snapped {len(s):5d} ({100*len(s)/max(1,len(ks)):.1f}%)  gain {sum(gain[k] for k in s):+.4f}  "
          f"neg-gain snaps {sum(1 for k in s if gain[k]<0)} sum-neg {sum(gain[k] for k in s if gain[k]<0):+.4f}  "
          f"unsnapped-loss {sum(W[k[:3]]*(1-jf(A[k])) for k in ks if k not in sn):.4f}  recoverable-if-snapped {sum(gain[k] for k in ks if k not in sn):+.4f}")
# per snapped frame gain distribution in J&F units
d = sorted(jf(A[k], "Jc", "Fc") - jf(A[k]) for k in sn)
q = lambda p: d[int(p * (len(d) - 1))]
print(f"\nper-snap ΔJ&F: mean {statistics.mean(d):+.4f} median {q(.5):+.4f} p10 {q(.1):+.4f} p90 {q(.9):+.4f}; negative {sum(x<0 for x in d)} ({100*sum(x<0 for x in d)/len(d):.1f}%), < -0.05: {sum(x<-0.05 for x in d)}")
# GT pixel-identical copies (Jc==1 and Fc==1)
ident = [k for k in sn if A[k]["Jc"] == 1.0 and A[k]["Fc"] == 1.0]
print(f"snapped frames where GT_t == GT0 exactly (J=F=1): {len(ident)} ({100*len(ident)/len(sn):.1f}%), gain {sum(gain[k] for k in ident):+.4f}")
allid = [k for k in ref if A[k]["Jc"] == 1.0 and A[k]["Fc"] == 1.0]
print(f"all ref obj-frames with GT_t == GT0 (J=F=1): {len(allid)} of {len(ref)}; of these snapped {len(set(allid)&sn)}")
# per object patterns
print("\n== per object snap patterns (ref, t=1..20)")
pat = {}
for o in objs_ref:
    seq = [(o + (t,)) in sn for t in range(1, NT[o])]
    pat[o] = seq
cnt = collections.Counter(sum(s) for s in pat.values())
full = [o for o, s in pat.items() if all(s)]; none = [o for o, s in pat.items() if not any(s)]; part = [o for o, s in pat.items() if any(s) and not all(s)]
print(f"objects {len(pat)}: all-20 {len(full)}, none {len(none)}, partial {len(part)}")
print("gain by object type: all", round(sum(gain[o+(t,)] for o in full for t in range(1,NT[o])),4),
      "partial", round(sum(gain[o+(t,)] for o in part for t in range(1,NT[o]) if o+(t,) in sn),4))
# transitions
s2s = s2n = n2s = n2n = 0
stops = []; resnap = 0; runs = []; first = []; after_last = 0; tail_unsnapped_static = 0
for o, s in pat.items():
    if not any(s): continue
    for a, b in zip(s, s[1:]):
        if a and b: s2s += 1
        elif a and not b: s2n += 1; stops.append(o)
        elif not a and b: n2s += 1
        else: n2n += 1
    # runs
    r = 0
    for i, x in enumerate(s):
        if x: r += 1
        if (not x or i == len(s) - 1) and r: runs.append(r); r = 0
    fi = s.index(True); first.append(fi + 1)
    li = len(s) - 1 - s[::-1].index(True)
    after_last += len(s) - 1 - li
    # re-snap after a break
    seen_break = False
    for i in range(1, len(s)):
        if s[i - 1] and not s[i]: seen_break = True
        if seen_break and s[i] and not s[i - 1]: resnap += 1
print(f"P(snap_t | snap_t-1) = {s2s/(s2s+s2n):.3f} ({s2s}/{s2s+s2n});  P(snap_t | not snap_t-1) = {n2s/(n2s+n2n):.3f} ({n2s}/{n2s+n2n}) [objects with >=1 snap]")
print(f"stops (snap -> no snap): {s2n} in {len(set(stops))} objects; re-snaps after a stop: {resnap}")
print(f"runs: n {len(runs)}, length dist {sorted(collections.Counter(runs).items())}")
print(f"first snap frame dist: {sorted(collections.Counter(first).items())}")
print(f"unsnapped frames after the last snap (objects with >=1 snap): {after_last}")
# objects that stop: why
why = collections.Counter()
for o, s in pat.items():
    for i in range(1, len(s)):
        if s[i - 1] and not s[i]:
            f = ref[o + (i + 1,)]
            why[("tau" if f["iou_p_g0"] < 0.8 else "") + ("delta" if f["imgdiff"] > 4 else "")] += 1
print("criterion failing at a stop:", dict(why))
# unsnapped frames of objects with >=1 snap, by failing criterion and category
uf = collections.Counter(); ul = collections.Counter(); ug = collections.Counter()
for o, s in pat.items():
    if not any(s): continue
    for i, x in enumerate(s):
        if x: continue
        k = o + (i + 1,); f = ref[k]
        key = (("tau" if f["iou_p_g0"] < 0.8 else "") + ("delta" if f["imgdiff"] > 4 else ""), cat(k), "after1st" if i + 1 > first[0] or any(s[:i]) else "before1st")
        uf[key] += 1; ul[key] += W[o] * (1 - jf(A[k])); ug[key] += gain[k]
print("\nunsnapped frames of objects with >=1 snap: (failing, GTcat, before/after first snap): n, tracker loss, copy-minus-tracker")
for key in sorted(uf): print(f"  {key}: n {uf[key]}, loss {ul[key]:.4f}, snapgain {ug[key]:+.4f}")
