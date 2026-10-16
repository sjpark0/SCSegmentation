"""(2a) Item 3, reference camera: which unsnapped c_ini frames could memory feedback reach, and bounds.
GT used for: tracker J&F at t (anat2 J,F), copy-GT0 J&F (Jc,Fc).  Snap decisions: GT-free."""
from common import *
ref = {k: f for k, f in S.items() if f["ref"]}
sn = {k for k, f in ref.items() if snap(f)}
def cat(k):
    c = jf(A[k], "Jc", "Fc"); return "static" if c >= 0.98 else "near" if c >= 0.9 else "moving"
objs_ref = sorted({k[:3] for k in ref})
# memory perturbation on snapped frames: 1 - IoU(pred_t, mask_0) and area change
iu = sorted(ref[k]["iou_p_g0"] for k in sn)
q = lambda d, p: d[int(p * (len(d) - 1))]
print(f"snapped frames: IoU(pred_t, mask_0) median {q(iu,.5):.4f} p10 {q(iu,.1):.4f} p1 {q(iu,.01):.4f} min {iu[0]:.4f}; share >= 0.95: {sum(x>=0.95 for x in iu)/len(iu):.3f}, >=0.98: {sum(x>=0.98 for x in iu)/len(iu):.3f}")
# reachable unsnapped frames: after >=1 snapped frame of the same object
rows = []
for o in objs_ref:
    seq = [(o + (t,)) in sn for t in range(1, NT[o])]
    last = None
    for t in range(1, NT[o]):
        k = o + (t,)
        if seq[t - 1]: last = t; continue
        if last is None: continue
        rows.append((k, t - last))
L = lambda ks: sum(W[k[:3]] * (1 - jf(A[k])) for k in ks)
print(f"\nunsnapped c_ini object-frames: {len(ref)-len(sn)}, loss {L([k for k in ref if k not in sn]):.4f}")
print(f"  of which before any snap of that object or object never snapped (unreachable): "
      f"{len(ref)-len(sn)-len(rows)}, loss {L([k for k in ref if k not in sn and k not in {r[0] for r in rows}]):.4f}")
print(f"  reachable (after >=1 snap): {len(rows)}, loss {L([r[0] for r in rows]):.4f}")
for lo, hi in ((1, 6), (7, 99)):
    ks = [k for k, d in rows if lo <= d <= hi]
    print(f"    distance to last snap {lo}-{hi}: n {len(ks)}, loss {L(ks):.4f}")
# per category and failing criterion
print("  reachable by GT cat x failing criterion: n, loss, copy-tracker if it had snapped, frame1-anchored drift max(0, JF_1 - JF_t)")
tab = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
for k, d in rows:
    f = ref[k]; crit = ("tau" if f["iou_p_g0"] < 0.8 else "") + ("delta" if f["imgdiff"] > 4 else "")
    key = (cat(k), crit); e = tab[key]
    e[0] += 1; e[1] += W[k[:3]] * (1 - jf(A[k])); e[2] += W[k[:3]] * (jf(A[k], "Jc", "Fc") - jf(A[k]))
    j1 = jf(A[k[:3] + (1,)])
    e[3] += W[k[:3]] * max(0.0, j1 - jf(A[k])); e[4] += W[k[:3]] * (j1 - jf(A[k]))
for key in sorted(tab): e = tab[key]; print(f"    {key}: n {e[0]}, loss {e[1]:.4f}, snapgain {e[2]:+.4f}, drift+ {e[3]:.4f} (signed {e[4]:+.4f})")
tot = [sum(e[i] for e in tab.values()) for i in range(5)]
print(f"    total: n {tot[0]}, loss {tot[1]:.4f}, drift+ {tot[3]:.4f} (signed {tot[4]:+.4f})")
# extra-snap channel: tau-only failures after a snap (delta passes)
ks = [k for k, d in rows if ref[k]["iou_p_g0"] < 0.8 and ref[k]["imgdiff"] <= 4]
print(f"\nextra-snap channel (delta passes, tau fails, after a snap): n {len(ks)}, gain if snapped {sum(W[k[:3]]*(jf(A[k],'Jc','Fc')-jf(A[k])) for k in ks):+.4f}")
ks_all = [k for k in ref if k not in sn and ref[k]["iou_p_g0"] < 0.8 and ref[k]["imgdiff"] <= 4]
print(f"  same over all unsnapped frames (incl. never-snapped objects): n {len(ks_all)}, gain {sum(W[k[:3]]*(jf(A[k],'Jc','Fc')-jf(A[k])) for k in ks_all):+.4f}")
# snapped frames: the drift the base tracker had there (memory of a snapped run is already mask_0-like?)
d1 = [jf(A[k[:3] + (1,)]) - jf(A[k]) for k in sn]
print(f"\nsnapped frames: base tracker JF_1 - JF_t mean {statistics.mean(d1):+.4f}")
# per-frame ref loss curve of base tracker (frame 1 = memory holds only the GT prompt)
for t in (1, 2, 5, 10, 20):
    ks = [k for k in ref if k[3] == t]
    print(f"  t={t}: ref loss/frame {L(ks):.4f}  static-only {L([k for k in ks if cat(k)=='static']):.4f}")
# crop pass on c_ini: objects < 10k px prompt (their frames come from the crop tracker when kept)
small = [o for o in objs_ref if A[o + (0,)]["gt"] < 10000 and A[o + (0,)]["gt"] > 0]
print(f"\nc_ini objects with prompt < 10k px (crop-pass candidates): {len(small)} of {len(objs_ref)}; their snaps {sum(1 for k in sn if k[:3] in set(small))}; reachable unsnapped frames {sum(1 for k,d in rows if k[:3] in set(small))}, loss {L([k for k,d in rows if k[:3] in set(small)]):.4f}")
