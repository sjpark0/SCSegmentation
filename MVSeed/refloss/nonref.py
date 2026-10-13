"""Non-reference camera loss map: E8 headline vs the GT-seed headline (E7) vs control, per pair/frame."""
import json, os, sys, statistics, collections
sys.path.insert(0, "/home/sjpark/Documents/SCSegmentation/MVSeed")
from e7_judge import load, rows, jf
H = "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2"; G = "SegMaskSam3XW1CGPS4MSdgtFsCs10kG2"; C = "SegMaskSam3XW0MSdgeoset_repFs"; GC = "SegMaskSam3XW0MSdgtFs"
d = load(["jf_e8.json", "jf_e7.json"])
F0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open("/home/sjpark/Documents/SCSegmentation/MVSeed/runs/F0.json"))["pairs"]}
N0 = {(p["scene"], p["cam"], p["obj"]) for p in json.load(open("/home/sjpark/Documents/SCSegmentation/MVSeed/runs/N0.json"))["pairs"]}
def tab(m):
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows(d, m): per[r["scene"]][r["cam"]][r["obj"]] = r
    out = {}
    for sc, cams in per.items():
        for cam, objs in cams.items():
            for o, r in objs.items():
                out[(sc, cam, o)] = dict(r=r, w=100 / (17 * len(cams) * len(objs)) / len(r["J"]))
    return out
T = {m: tab(m) for m in (H, G, C, GC)}
keys = [k for k, x in T[H].items() if not x["r"]["ref"]]
def L(m, k, frames=None):
    x = T[m][k]; r = x["r"]
    return sum(x["w"] * (1 - jf(r, t)) for t in range(len(r["J"])) if (frames is None or t in frames) and jf(r, t) is not None)
grp = lambda k: "F0" if k in F0 else "N0" if k in N0 else "other"
print("nonref loss: H %.3f  GTseed %.3f  C %.3f  GTseedC %.3f" % tuple(sum(L(m, k) for k in keys) for m in (H, G, C, GC)))
for g in ("F0", "N0", "other"):
    ks = [k for k in keys if grp(k) == g]
    print(f" {g:5s} n={len(ks):4d}  H {sum(L(H,k) for k in ks):.3f} (f0 {sum(L(H,k,{0}) for k in ks):.3f})  GTseed {sum(L(G,k) for k in ks):.3f} (f0 {sum(L(G,k,{0}) for k in ks):.3f})")
# frame-0 vs later per pair: does tracking amplify or repair seed error?
amp = collections.Counter(); n = collections.Counter()
for k in keys:
    r = T[H][k]["r"]; j0 = jf(r, 0); later = [jf(r, t) for t in range(1, len(r["J"])) if jf(r, t) is not None]
    if j0 is None or not later: continue
    m = statistics.mean(later); b = "seed<0.5" if j0 < 0.5 else "seed0.5-0.8" if j0 < 0.8 else "seed0.8-0.9" if j0 < 0.9 else "seed>=0.9"
    n[b] += 1; amp[b + (" better" if m > j0 + 0.02 else " worse" if m < j0 - 0.02 else " same")] += 1
print("seed bins (frame-0 J&F) -> frames 1-20 mean vs frame 0:")
for b in ("seed<0.5", "seed0.5-0.8", "seed0.8-0.9", "seed>=0.9"):
    print(f"  {b:12s} n={n[b]:4d}  better {amp[b+' better']:4d}  same {amp[b+' same']:4d}  worse {amp[b+' worse']:4d}")
# seed-attributable loss (H - GTseed) by seed bin and by scene
S = collections.Counter(); B = collections.Counter(); Lb = collections.Counter()
for k in keys:
    r = T[H][k]["r"]; j0 = jf(r, 0) or 0
    b = "seed<0.5" if j0 < 0.5 else "seed0.5-0.8" if j0 < 0.8 else "seed0.8-0.9" if j0 < 0.9 else "seed>=0.9"
    dl = L(H, k) - L(G, k); S[k[0]] += dl; B[b] += dl; Lb[b] += L(H, k)
print("H loss and seed-attributable (H - GTseed) by seed bin:")
for b in ("seed<0.5", "seed0.5-0.8", "seed0.8-0.9", "seed>=0.9"): print(f"  {b:12s} H loss {Lb[b]:.3f}  seed-attrib {B[b]:+.3f}")
print("seed-attributable by scene:", {s: round(v, 3) for s, v in S.most_common()})
# per-frame curve nonref
for m in (H, G):
    print(m[-22:], "per-frame loss:", " ".join(f"{t}:{sum(L(m,k,{t}) for k in keys):.3f}" for t in (0, 1, 2, 5, 10, 15, 20)))
