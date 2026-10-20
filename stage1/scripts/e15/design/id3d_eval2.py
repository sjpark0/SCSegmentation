import json, os, itertools
import numpy as np
rows = json.load(open(os.path.join(os.path.dirname(__file__), "id3d_rows2.json")))
FISH = {"AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder"}
def st(r, beta, kappa, sigma, renv, gam, bmax):
    if not r["ok"]: return "abstain"
    o = str(r["o"]); d, rho = r["d"], r["rho"]
    if rho[o] > renv or r["border"][o] > bmax or r["rho_q"] > gam * rho[o] + sigma: return "abstain"
    comp = [j for j in d if j != o and rho[j] <= renv]
    own_in = d[o] <= beta * rho[o] + sigma
    nearer = any(kappa * d[j] < d[o] for j in comp)
    if own_in and not nearer: return "pass"
    if (not own_in) and nearer: return "flag"
    return "abstain"
mu = [r for r in rows if r["src"] == "muvod"]; t6 = [r for r in rows if r["src"] == "t06"]
gt = [r for r in mu if r["kind"] == "gt"]
pr = {s: [r for r in mu if r["kind"] == "pred_" + s and r.get("covers", 0) != 0] for s in ("r1", "final")}
t6p = [r for r in t6 if r["kind"] == "pred_final" and r.get("covers", 0) != 0]
scenes = sorted({r["scene"] for r in gt})
res = []
for beta, kappa, sigma, renv, gam, bmax in itertools.product((1.0, 1.25, 1.5), (1.25,), (0.03, 0.05), (0.5, 1.0), (1e9, 3.0, 2.0), (1e9, 5.0, 1.0)):
    P = (beta, kappa, sigma, renv, gam, bmax)
    S = lambda r: st(r, *P)
    ff = sum(S(r) == "flag" for r in gt); ps = sum(S(r) == "pass" for r in gt)
    per = {s: np.mean([S(r) == "flag" for r in gt if r["scene"] == s]) for s in scenes}
    worst = max(per.items(), key=lambda x: x[1])
    fish = sum(S(r) == "flag" for r in gt if r["scene"] in FISH)
    r1c = [r for r in pr["r1"] if r["covers"] == r["o"]]; r1w = [r for r in pr["r1"] if r["covers"] != r["o"]]
    tcor = [r for r in t6p if r["covers"] == r["o"]]; twr = [r for r in t6p if r["covers"] != r["o"]]
    res.append(dict(P=P, mu_gt_ff=ff, mu_gt_pass=ps, worst=(worst[0], round(worst[1], 3)), n_scenes_over2=sum(v > 0.02 for v in per.values()), fish=fish,
                    mu_r1_ff=sum(S(r) == "flag" for r in r1c), mu_r1_tf=f"{sum(S(r) == 'flag' for r in r1w)}/{len(r1w)}",
                    t06_tf=sum(S(r) == "flag" for r in twr), t06_ff=sum(S(r) == "flag" for r in tcor), t06_pass=sum(S(r) == "pass" for r in tcor),
                    t06_gt_ff=sum(S(r) == "flag" for r in t6 if r["kind"] == "gt"), t06_swap=sum(S(r) == "flag" for r in t6 if r["kind"] == "swap")))
print(f"MUVOD gt n={len(gt)}, r1 correct n={sum(r['covers']==r['o'] for r in pr['r1'])}, T06 final correct {sum(r['covers']==r['o'] for r in t6p)} wrong {sum(r['covers']!=r['o'] for r in t6p)}")
for x in sorted(res, key=lambda x: (x["n_scenes_over2"], x["mu_gt_ff"] - 3 * x["t06_tf"])):
    if x["t06_tf"] >= 18:
        print(x)
