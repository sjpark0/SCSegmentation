"""Sensitivity of a camera's output to its neighbours' memory, from existing runs (GT used: J&F of existing outputs)."""
from common import *
def cmp(fa, ma, fb, mb, label, frames=range(1, 21)):
    a, b = jfrows(fa, ma), jfrows(fb, mb); w = pair_w(a)
    assert set(a) == set(b), len(set(a) ^ set(b))
    res = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    per_view = collections.defaultdict(float)
    for k in a:
        sc, cam, o = k; d = dist(sc, cam)
        b_ = "ref" if d == 0 else "|d|=1" if abs(d) == 1 else "|d|=2" if abs(d) == 2 else "|d|>=3"
        va = [x for t in frames if (x := pjf(a[k][0], a[k][1], t)) is not None]
        vb = [x for t in frames if (x := pjf(b[k][0], b[k][1], t)) is not None]
        if not va: continue
        dd = statistics.mean(vb) - statistics.mean(va)
        r = res[b_]; r[0] += 1; r[1] += w[k] * dd * len(va) / 21; r[2] += abs(dd); r[3] = max(r[3], abs(dd))
        per_view[(sc, cam, b_)] += w[k] * dd * len(va) / 21
    print(f"\n== {label}: {mb} minus {ma} (frames {frames.start}-{frames.stop-1}; contribution in basic points, mean|ΔJ&F| per pair, max|ΔJ&F|)")
    for b_ in ("ref", "|d|=1", "|d|=2", "|d|>=3"):
        r = res[b_]; print(f"   {b_:7s} pairs {r[0]:4d}  contrib {r[1]:+.4f}  mean|Δ| {r[2]/max(1,r[0]):.4f}  max|Δ| {r[3]:.4f}")
    return per_view
pv = cmp("jf_fs.json", "SegMaskSam3XW0MFs", "jf_fs.json", "SegMaskSam3XW1CGPS4MFs", "whole cross-view memory on/off (same G0 seeds)")
for k, v in sorted(pv.items()):
    if k[2] in ("ref", "|d|=1"): print(f"     {k}: {v:+.4f}")
pv = cmp("jf_e8.json", "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2", "jf_e7.json", "SegMaskSam3XW1CGPS4MSdgtFsCs10kG2", "non-ref seeds stage-1 -> GT (c_ini prompt unchanged)")
for k, v in sorted(pv.items()):
    if k[2] in ("ref",): print(f"     {k}: {v:+.4f}")
pv = cmp("jf_s2r6.json", "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs", "jf_e14.json", "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs", "E8 -> E14 stage-1 seeds (both Rs)")
