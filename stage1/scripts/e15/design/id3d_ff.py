import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
rows = json.load(open(os.path.join(os.path.dirname(__file__), "id3d_rows.json")))
beta, kappa, sigma, renv = 1.0, 1.25, 0.05, 1.0
def st(r, nmin=20):
    if not r["ok"] or r["n"] < nmin: return "abstain"
    o = str(r["o"]); d, rho = r["d"], r["rho"]
    if rho[o] > renv: return "abstain"
    comp = [j for j in d if j != o and rho[j] <= renv]
    own_in = d[o] <= beta * rho[o] + sigma
    jn = min(comp, key=lambda j: d[j]) if comp else None
    nearer = jn is not None and kappa * d[jn] < d[o]
    if own_in and not nearer: return "pass"
    if (not own_in) and nearer: return "flag"
    return "abstain"
gt = [r for r in rows if r["src"] == "muvod" and r["kind"] == "gt"]
ff = [r for r in gt if st(r) == "flag"]
from collections import Counter
print("by scene", Counter(r["scene"] for r in ff))
print("by scene/obj", Counter((r["scene"], r["o"]) for r in ff).most_common(12))
for r in ff[:0]: pass
# size of flagged vs all
print("n lifted pts flagged median", np.median([r["n"] for r in ff]), "all", np.median([r["n"] for r in gt if r["ok"]]))
for s in ("CBABasketball", "Blocks", "Fencing", "CoffeeMartini"):
    x = [r for r in ff if r["scene"] == s][:6]
    for r in x:
        o = str(r["o"]); jn = min((j for j in r["d"] if j != o), key=lambda j: r["d"][j])
        print(s, r["t"], r["cam"], "o", o, "n", r["n"], "d_own %.3f rho %.3f | nearest %s d %.3f rho %.3f" % (r["d"][o], r["rho"][o], jn, r["d"][jn], r["rho"][jn]))
for nmin in (20, 50, 100, 200):
    f = sum(st(r, nmin) == "flag" for r in gt); p = [r for r in rows if r["src"] == "t06" and r["kind"] == "pred_final" and r.get("covers", 0) not in (0,)]
    tf = sum(st(r, nmin) == "flag" for r in p if r["covers"] != r["o"]); fc = sum(st(r, nmin) == "flag" for r in p if r["covers"] == r["o"])
    ab = sum(st(r, nmin) == "abstain" for r in gt)
    print("nmin", nmin, "muvod gt ff", f, "abstain", ab, "t06 tf", tf, "ff", fc)
