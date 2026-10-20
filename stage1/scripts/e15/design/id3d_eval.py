import json, os, itertools
import numpy as np
rows = json.load(open(os.path.join(os.path.dirname(__file__), "id3d_rows.json")))
FISH = {"AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder"}

def status(r, rule):
    """-> 'flag' | 'pass' | 'abstain'"""
    if not r["ok"]: return "abstain"
    o = str(r["o"]); d, rho = r["d"], r["rho"]
    if rule == "sim":
        return "flag" if min(d, key=d.get) != o else "pass"
    beta, kappa, sigma, renv = rule
    if rho[o] > renv: return "abstain"
    comp = [j for j in d if j != o and rho[j] <= renv]
    own_in = d[o] <= beta * rho[o] + sigma
    nearer = any(kappa * d[j] < d[o] for j in comp)
    if own_in and not nearer: return "pass"
    if (not own_in) and nearer: return "flag"
    return "abstain"

def summarize(rule):
    out = {}
    def rate(sel, st):
        n = len(sel); return (sum(status(r, rule) == st for r in sel), n)
    mu = [r for r in rows if r["src"] == "muvod"]; t6 = [r for r in rows if r["src"] == "t06"]
    gt_mu = [r for r in mu if r["kind"] == "gt"]
    out["muvod_gt_falseflag"] = rate(gt_mu, "flag"); out["muvod_gt_pass"] = rate(gt_mu, "pass")
    per = {}
    for s in sorted({r["scene"] for r in gt_mu}):
        f, n = rate([r for r in gt_mu if r["scene"] == s], "flag"); per[s] = round(f / n, 3)
    out["muvod_gt_ff_worst"] = max(per.items(), key=lambda x: x[1])
    out["muvod_gt_ff_fish"] = rate([r for r in gt_mu if r["scene"] in FISH], "flag")
    out["muvod_swap_detect"] = rate([r for r in mu if r["kind"] == "swap"], "flag")
    for st in ("r1", "final"):
        pr = [r for r in mu if r["kind"] == "pred_" + st and r.get("covers", 0) != 0]
        cor = [r for r in pr if r["covers"] == r["o"]]; wr = [r for r in pr if r["covers"] != r["o"]]
        out[f"muvod_{st}_correct_ff"] = rate(cor, "flag"); out[f"muvod_{st}_wrong_tf"] = rate(wr, "flag")
    gt6 = [r for r in t6 if r["kind"] == "gt"]
    out["t06_gt_ff"] = rate(gt6, "flag"); out["t06_gt_pass"] = rate(gt6, "pass")
    out["t06_swap_detect"] = rate([r for r in t6 if r["kind"] == "swap"], "flag")
    pr = [r for r in t6 if r["kind"] == "pred_final" and r.get("covers", 0) != 0]
    out["t06_final_correct_ff"] = rate([r for r in pr if r["covers"] == r["o"]], "flag")
    out["t06_final_correct_pass"] = rate([r for r in pr if r["covers"] == r["o"]], "pass")
    out["t06_final_wrong_tf"] = rate([r for r in pr if r["covers"] != r["o"]], "flag")
    return out

def fmt(x):
    return f"{x[0]}/{x[1]}" if isinstance(x, tuple) and isinstance(x[0], int) else str(x)
print("SIM", {k: fmt(v) for k, v in summarize("sim").items()})
for beta, kappa, sigma, renv in itertools.product((1.0, 1.5, 2.0), (1.0, 1.25, 1.5), (0.02, 0.05), (0.5, 1.0)):
    s = summarize((beta, kappa, sigma, renv))
    print((beta, kappa, sigma, renv), {k: fmt(v) for k, v in s.items()})
