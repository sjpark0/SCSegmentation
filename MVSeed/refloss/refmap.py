#!/usr/bin/env python3
"""Reference-camera loss map (descriptive, no rule).  Loss of a pair in benchmark points =
w_i * (1 - S_i), S_i = mean J&F over 21 frames, w_i = MUVOD basic weight.  Ref frame 0 is GT."""
import json, os, statistics, sys, collections
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "MVSeed"))
from e7_judge import load, rows, jf

H = "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2"
C = "SegMaskSam3XW0MSdgeoset_repFs"
GTH = "SegMaskSam3XW1CGPS4MSdgtFsCs10kG2"
GTC = "SegMaskSam3XW0MSdgtFs"
OLDH = "SegMaskSam3XW1CGPS4MFsCs10kG2"
SAM2 = "SegMaskNew3"
METHODS = [H, C, GTH, GTC, OLDH, SAM2]
d = load(["jf_e8.json", "jf_e7.json", "jf_r4.json", "jf_paper.json"])


def table(m):
    R = list(rows(d, m))
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in R: per[r["scene"]][r["cam"]][r["obj"]] = r
    out = {}
    n_sc = len(per)
    for sc, cams in per.items():
        for cam, objs in cams.items():
            for o, r in objs.items():
                w = 100.0 / (n_sc * len(cams) * len(objs))
                v = [jf(r, t) for t in range(len(r["J"]))]
                out[(sc, cam, o)] = dict(r=r, w=w, v=v)
    return out


T = {m: table(m) for m in METHODS}
out = {}
for m in METHODS:
    tot = collections.Counter(); ref_loss = collections.Counter()
    for k, x in T[m].items():
        vv = [a for a in x["v"] if a is not None]
        loss = x["w"] * (1 - statistics.mean(vv)) if vv else 0
        tot["all"] += loss
        if x["r"]["ref"]: ref_loss[k[0]] += loss; tot["ref"] += loss
    out[m] = dict(total_loss=round(tot["all"], 3), ref_loss=round(tot["ref"], 3),
                  ref_by_scene={s: round(v, 3) for s, v in ref_loss.most_common()})
    print(f"{m:45s} total loss {tot['all']:.3f}  ref loss {tot['ref']:.3f}")

# per-object detail on the headline's reference camera
H_ = T[H]
objs = []
for k, x in H_.items():
    r = x["r"]
    if not r["ref"]: continue
    v = x["v"]; vv = [a for a in v if a is not None]
    loss = x["w"] * (1 - statistics.mean(vv)) if vv else 0
    gt = r["gt"]; pa = r["r"]["pred_area"] if False else None
    row = dict(scene=k[0], cam=k[1], obj=k[2], loss=round(loss, 4), w=round(x["w"], 4),
               jf=[None if a is None else round(a, 3) for a in v], gt=gt)
    for m in (C, GTH, GTC, OLDH, SAM2):
        y = T[m].get(k)
        if y:
            yy = [a for a in y["v"] if a is not None]
            row["loss_" + m] = round(y["w"] * (1 - statistics.mean(yy)), 4) if yy else None
    objs.append(row)
objs.sort(key=lambda r: -r["loss"])
cum = 0; tot = sum(r["loss"] for r in objs)
print(f"\nheadline ref loss {tot:.3f} over {len(objs)} objects; top 25:")
for i, r in enumerate(objs[:25]):
    cum += r["loss"]
    onset = next((t for t, a in enumerate(r["jf"]) if a is not None and a < 0.8), None)
    gtz = sum(1 for a in r["gt"] if not a)
    print(f"{i:2d} {r['scene']:18s} {r['cam']} o{r['obj']:<3} loss {r['loss']:.3f} cum {cum/tot:5.1%}  minJF {min(a for a in r['jf'] if a is not None):.2f} onset {onset} gt0frames {gtz} gtpx f0 {r['gt'][0]} f20 {r['gt'][-1]}  C {r.get('loss_'+C)} GTH {r.get('loss_'+GTH)} SAM2 {r.get('loss_'+SAM2)}")
# concentration
ls = sorted((r["loss"] for r in objs), reverse=True)
for q in (5, 10, 20, 50):
    print(f"top {q} objects: {sum(ls[:q])/tot:.1%}")
json.dump(dict(methods=out, ref_objects=objs), open(os.path.join(os.path.dirname(__file__), "refmap.json"), "w"), indent=1)
