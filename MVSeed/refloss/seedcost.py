"""Host: per non-reference pair, stage-2 loss of the headline (LH) and of the GT-seed headline (LG) -> cost.json (docs/stage1-seedfail.md)."""
import json, sys, collections
sys.path.insert(0, "/home/sjpark/Documents/SCSegmentation/MVSeed")
from e7_judge import load, rows, jf
H = "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2"; G = "SegMaskSam3XW1CGPS4MSdgtFsCs10kG2"
d = load(["jf_e8.json", "jf_e7.json"])
def tab(m):
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows(d, m): per[r["scene"]][r["cam"]][r["obj"]] = r
    out = {}
    for sc, cams in per.items():
        for cam, objs in cams.items():
            for o, r in objs.items(): out[(sc, cam, o)] = (r, 100 / (17 * len(cams) * len(objs)) / len(r["J"]))
    return out
T = {m: tab(m) for m in (H, G)}
L = lambda m, k: sum(T[m][k][1] * (1 - jf(T[m][k][0], t)) for t in range(len(T[m][k][0]["J"])) if jf(T[m][k][0], t) is not None)
out = {"|".join(map(str, k)): dict(LH=L(H, k), LG=L(G, k), jf0=jf(T[H][k][0], 0)) for k in T[H] if not T[H][k][0]["ref"]}
json.dump(out, open("cost.json", "w")); print(len(out), sum(v["LH"] for v in out.values()), sum(v["LH"]-v["LG"] for v in out.values()))
