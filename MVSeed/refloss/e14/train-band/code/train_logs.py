#!/usr/bin/env python3
"""S1-E14 train-band: loss-term magnitudes of the E9 / E12 training logs (host, numpy only).

For each arm/fold: check total = 20 focal + dice + iou + presence + W_band band (W_band 1 in geoset_decb_ft, 0 else);
base = total - band term; band/base ratios over phases (first 100 steps, epoch means, last 100 steps); the W at which
W * band == base in each phase; flow identity of D1/D2/C0' (same samples per step?).  -> train_logs.json
"""
import csv, json, os, sys
import numpy as np

RUNS = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs"
ARMS = ["geoset_pred_ft", "geoset_cut0_ft", "geoset_dec_ft", "geoset_decb_ft"]


def read(arm, k):
    with open(os.path.join(RUNS, arm, f"fold{k}", "log.csv")) as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for key in ("step", "epoch", "loss", "focal", "dice", "iou", "presence", "seconds", "n_mem", "n_targets"):
        cols[key] = np.array([float(r[key]) if r.get(key) not in (None, "") else np.nan for r in rows])
    cols["band"] = np.array([float(r["band"]) if r.get("band") not in (None, "") else 0.0 for r in rows])
    cols["objs"] = [r.get("objs") for r in rows]
    return cols


def stats(x):
    return dict(mean=float(np.mean(x)), median=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))


out = {}
for arm in ARMS:
    W = 1.0 if arm == "geoset_decb_ft" else 0.0
    out[arm] = {}
    for k in range(5):
        c = read(arm, k)
        base = 20 * c["focal"] + c["dice"] + c["iou"] + c["presence"]
        recon = base + W * c["band"]
        n = len(base)
        ep = c["epoch"].astype(int)
        rec = dict(steps=n, max_abs_recon_err=float(np.max(np.abs(recon - c["loss"]))),
                   seconds_last=float(c["seconds"][-1]))
        phases = {"first100": slice(0, 100), "last100": slice(n - 100, n), "all": slice(0, n)}
        for e in sorted(set(ep)):
            phases[f"epoch{e}"] = np.where(ep == e)[0]
        for name, sl in phases.items():
            b, bd = base[sl], c["band"][sl]
            r = dict(base=float(b.mean()), focal20=float(20 * c["focal"][sl].mean()), dice=float(c["dice"][sl].mean()),
                     iou=float(c["iou"][sl].mean()), presence=float(c["presence"][sl].mean()))
            if W:
                r.update(band=float(bd.mean()), band_over_base=float(bd.mean() / b.mean()),
                         W_eq=float(b.mean() / bd.mean()),                       # W at which W*band == base (means)
                         W_eq_median_step=float(np.median(b / np.maximum(bd, 1e-9))),
                         band_share_of_total=float((bd / (b + bd)).mean()))
            rec[name] = r
        out[arm][f"fold{k}"] = rec
        out[arm][f"fold{k}"]["_cols"] = c
# flow identity: is the per-step sample stream the same (n_targets, n_mem) between arms?
flow = {}
for a, b in [("geoset_dec_ft", "geoset_decb_ft"), ("geoset_cut0_ft", "geoset_dec_ft"), ("geoset_cut0_ft", "geoset_decb_ft"),
             ("geoset_pred_ft", "geoset_cut0_ft")]:
    flow[f"{a}~{b}"] = {}
    for k in range(5):
        ca, cb = out[a][f"fold{k}"]["_cols"], out[b][f"fold{k}"]["_cols"]
        same_nt = float(np.mean(ca["n_targets"] == cb["n_targets"]))
        same_nm = float(np.mean(ca["n_mem"] == cb["n_mem"]))
        base_a = 20 * ca["focal"] + ca["dice"] + ca["iou"] + ca["presence"]
        base_b = 20 * cb["focal"] + cb["dice"] + cb["iou"] + cb["presence"]
        flow[f"{a}~{b}"][f"fold{k}"] = dict(same_n_targets=same_nt, same_n_mem=same_nm,
                                            step1_base_equal=bool(abs(base_a[0] - base_b[0]) < 1e-6),
                                            base_last100_a=float(base_a[-100:].mean()), base_last100_b=float(base_b[-100:].mean()))
spikes = {}
for k in range(5):                                       # per-step distribution of the D2 band term: rare large steps
    c = out["geoset_decb_ft"][f"fold{k}"]["_cols"]
    base = 20 * c["focal"] + c["dice"] + c["iou"] + c["presence"]
    b = c["band"]
    n = len(b)
    spikes[f"fold{k}"] = {}
    for nm, sl in (("first300", slice(0, 300)), ("last300", slice(n - 300, n)), ("all", slice(0, n))):
        bb, ba = b[sl], base[sl]
        spikes[f"fold{k}"][nm] = dict(band_pct=dict(zip(("p10", "p50", "p90", "p99", "max"), [float(x) for x in np.percentile(bb, [10, 50, 90, 99, 100])])),
                                      band_mean=float(bb.mean()), base_median=float(np.median(ba)), base_mean=float(ba.mean()),
                                      frac_steps_band_gt10=float((bb > 10).mean()), share_band_sum_from_gt10=float(bb[bb > 10].sum() / bb.sum()),
                                      band_median_over_base_median=float(np.median(bb) / np.median(ba)),
                                      W_eq_median_step=float(np.median(ba / np.maximum(bb, 1e-9))),
                                      band_mean_without_gt10=float(bb[bb <= 10].mean()), base_mean_without_band_gt10=float(ba[bb <= 10].mean()))
for arm in ARMS:
    for k in range(5):
        out[arm][f"fold{k}"].pop("_cols")
res = dict(arms=out, flow=flow, decb_spikes=spikes)
# summary across folds for the band arm
d2 = out["geoset_decb_ft"]
summ = {}
for ph in ("first100", "epoch0", "epoch1", "epoch2", "last100", "all"):
    summ[ph] = dict(band=[round(d2[f"fold{k}"][ph]["band"], 3) for k in range(5)],
                    base=[round(d2[f"fold{k}"][ph]["base"], 4) for k in range(5)],
                    band_over_base=[round(d2[f"fold{k}"][ph]["band_over_base"], 2) for k in range(5)],
                    W_eq=[round(d2[f"fold{k}"][ph]["W_eq"], 4) for k in range(5)],
                    band_share=[round(d2[f"fold{k}"][ph]["band_share_of_total"], 3) for k in range(5)])
res["decb_summary"] = summ
json.dump(res, open(sys.argv[1], "w"), indent=1)
for ph, v in summ.items():
    print(ph, json.dumps(v))
for arm in ARMS:
    print(arm, "recon err", max(out[arm][f"fold{k}"]["max_abs_recon_err"] for k in range(5)),
          "first100 base", [round(out[arm][f"fold{k}"]["first100"]["base"], 3) for k in range(5)],
          "last100 base", [round(out[arm][f"fold{k}"]["last100"]["base"], 4) for k in range(5)],
          "last100 focal20/dice/iou/pres f0", {kk: round(out[arm]["fold0"]["last100"][kk], 4) for kk in ("focal20", "dice", "iou", "presence")})
for k in range(5):
    for nm in ("first300", "last300", "all"):
        v = spikes[f"fold{k}"][nm]
        print(k, nm, "band p50 %.3f mean %.3f | base p50 %.3f mean %.3f | steps>10 %.3f carry %.3f of band | median ratio %.2f | W_eq(step median) %.3f | without spikes band %.3f base %.3f" % (
            v["band_pct"]["p50"], v["band_mean"], v["base_median"], v["base_mean"], v["frac_steps_band_gt10"], v["share_band_sum_from_gt10"],
            v["band_median_over_base_median"], v["W_eq_median_step"], v["band_mean_without_gt10"], v["base_mean_without_band_gt10"]))
print({k: {f: (v[f]["same_n_targets"], v[f]["step1_base_equal"]) for f in v} for k, v in flow.items()})
