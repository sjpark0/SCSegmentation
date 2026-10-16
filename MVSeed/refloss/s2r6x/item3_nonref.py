"""(2b) Item 3, other cameras.  Old headline (E8 base == E8 Rs on non-ref) from anat2/snap2.
GT used for tracker J&F at t and seed-copy J&F (Js,Fs = frame-0 output copied, vs GT_t)."""
from common import *
ref = {k: f for k, f in S.items() if f["ref"]}
sn = {k for k, f in ref.items() if snap(f)}
cref = {sc: OBJ[sc]["c_ini"] for sc in SCENES}
non = {k: f for k, f in S.items() if not f["ref"]}
def cat(k):
    c = jf(A[k], "Jc", "Fc"); return "static" if c >= 0.98 else "near" if c >= 0.9 else "moving"
def band(d): return "1" if abs(d) == 1 else "2" if abs(d) == 2 else ">=3"
tab = collections.defaultdict(lambda: collections.Counter())
views = collections.defaultdict(set)
for k, f in non.items():
    sc, cam, o, t = k; d = dist(sc, cam); b = band(d); views[b].add((sc, cam))
    # frame of c_ini whose output reaches this camera first: upward chain same t, downward chain t-|d|
    tr = t if d > 0 else t - abs(d)
    reach = tr >= 1 and (sc, cref[sc], o, tr) in sn
    ever = any((sc, cref[sc], o, u) in sn for u in range(1, t + 1))
    a = A[k]; w = W[k[:3]]
    loss = w * (1 - jf(a)); above = w * max(0.0, 0.5 * (f["Js"] + f["Fs"]) - jf(a)); signed = w * (0.5 * (f["Js"] + f["Fs"]) - jf(a))
    for key in (("all",), ("snapped_at_reach_frame",) if reach else ("not_reach",), ("snapped_ever",) if ever else ("never",)):
        c = tab[(b,) + key]; c["n"] += 1; c["loss"] += loss; c["above_seed"] += above; c["signed"] += signed
        if cat(k) == "static": c["n_static"] += 1; c["loss_static"] += loss; c["above_static"] += above
print("non-ref views by |distance| to c_ini:", {b: len(v) for b, v in views.items()}, {b: sorted(v) for b, v in views.items() if b != '>=3'})
print("band | subset | n frames | loss | above-seed loss (sum max(0, seedcopy - tracker)) | signed | static: n, loss, above")
for key in sorted(tab):
    c = tab[key]
    print(f"  {key}: n {c['n']}, loss {c['loss']:.4f}, above-seed {c['above_seed']:.4f}, signed {c['signed']:+.4f} | static n {c['n_static']}, loss {c['loss_static']:.4f}, above {c['above_static']:.4f}")
# per direct-neighbour view details
for sc, cam in sorted(views["1"]):
    ks = [k for k in non if k[0] == sc and k[1] == cam]
    objs = sorted({k[2] for k in ks})
    print(f"\n{sc} {cam}: objects {len(objs)}")
    for o in objs:
        kk = [k for k in ks if k[2] == o]
        nsn = sum(1 for k in kk if (sc, cref[sc], o, k[3]) in sn)
        seed = 0.5 * (A[(sc, cam, o, 0)]["J"] + A[(sc, cam, o, 0)]["F"])
        tr = statistics.mean(jf(A[k]) for k in kk)
        print(f"   obj {o}: c_ini snaps {nsn}/20, seed JF {seed:.3f}, track JF(1-20) {tr:.3f}, loss {sum(W[k[:3]]*(1-jf(A[k])) for k in kk):.4f}, above-seed {sum(W[k[:3]]*max(0,0.5*(non[k]['Js']+non[k]['Fs'])-jf(A[k])) for k in kk):.4f}")
