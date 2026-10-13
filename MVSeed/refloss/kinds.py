import json, sys, collections
sys.path.insert(0, "/home/sjpark/Documents/SCSegmentation/MVSeed")
from e7_judge import load, CEN
d = load(["jf_e8.json", "jf_e7.json"])
G = {(r["scene"], r["cam"], r["obj"]): r for r in json.load(open("/home/sjpark/Documents/SCSegmentation/MVSeed/runs/grid_ceiling.json"))["rows"]}
def run(m, refonly):
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in d:
        if r["method"] != m or not r.get("result"): continue
        sc = r["dataset"]; basic = set(CEN[sc]["seed_ids_c_ini"]); isref = r["camera"] == CEN[sc]["c_ini"]
        pf = r["result"]["per_frame"]
        for i, o in enumerate(r["objects"]):
            if o in basic: per[sc][r["camera"]][o] = (isref, pf["J"][i], pf["F"][i], pf["gt_area"][i], pf["pred_area"][i])
    n = len(per); K = collections.Counter(); byframe = collections.Counter(); ceil = 0
    for sc, cams in per.items():
        for cam, objs in cams.items():
            for o, (isref, J, F, ga, pa) in objs.items():
                if isref != refonly: continue
                w = 100 / (n * len(cams) * len(objs)) / len(J)
                g = G.get((sc, cam, o)); ceil += w * len(J) * (1 - g["JF_288"]) * (20/21 if isref else 1)
                for t in range(len(J)):
                    if J[t] is None or F[t] is None: continue
                    dfc = w * (1 - 0.5 * (J[t] + F[t]))
                    if ga[t] == 0 and pa[t] > 0: k = "FP_gt_empty"
                    elif ga[t] == 0: k = "both_empty"
                    elif pa[t] == 0: k = "lost_pred_empty"
                    elif 0.5 * (J[t] + F[t]) < 0.5: k = "major_jf<0.5"
                    elif 0.5 * (J[t] + F[t]) < 0.8: k = "mid_0.5-0.8"
                    else: k = "minor_>=0.8"
                    K[k] += dfc; byframe[t] += dfc
    tot = sum(K.values())
    print(f"{m} {'ref' if refonly else 'nonref'}: loss {tot:.3f}  grid-ceiling est {ceil:.3f}")
    for k, v in K.most_common(): print(f"   {k:18s} {v:.3f} ({v/tot:.0%})")
    print("   by frame:", " ".join(f"{t}:{byframe[t]:.3f}" for t in (0,1,2,5,10,15,20)))
for m in ["SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2", "SegMaskSam3XW0MSdgeoset_repFs"]:
    run(m, True)
run("SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2", False)
run("SegMaskSam3XW1CGPS4MSdgtFsCs10kG2", False)
