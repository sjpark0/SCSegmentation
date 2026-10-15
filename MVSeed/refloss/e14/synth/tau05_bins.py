import json, glob
import numpy as np
R = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs"
base = json.load(open(f"{R}/_e13_dev/dev_pairs.json"))["rows"]
kb = {(r["scene"], int(r["t"]), r["cam"], int(r["obj"])): r for r in base}
V = {}
for p in glob.glob(f"{R}/_e14_threshold-upsample/parts/*.json"):
    v = json.load(open(p))
    for i, o in enumerate(v["objs"]):
        k = (v["scene"], int(v["t"]), v["cam"], int(o))
        if k in kb:
            V[k] = {t: {m: v["res"][f"bil2|{t}"][m][i] for m in ("J", "F", "tp", "pp")} for t in ("0", "0.5")}
bins = [(0, 2e3), (2e3, 1e4), (1e4, 1e5), (1e5, 1e12)]
for lab, sel in (("good", lambda r: r["JF"] >= 0.8), ("fail", lambda r: r["JF"] < 0.8)):
    ks = [k for k in kb if sel(kb[k])]
    d = np.array([(V[k]["0.5"]["J"] + V[k]["0.5"]["F"] - V[k]["0"]["J"] - V[k]["0"]["F"]) / 2 for k in ks])
    rec = lambda t: np.mean([V[k][t]["tp"] / kb[k]["gt_px"] for k in ks])
    prec = lambda t: np.mean([V[k][t]["tp"] / V[k][t]["pp"] if V[k][t]["pp"] else 0 for k in ks])
    print(lab, len(ks), "pairmean dJF %+.5f" % d.mean(), "1-rec %.4f -> %.4f" % (1 - rec("0"), 1 - rec("0.5")),
          "1-prec %.4f -> %.4f" % (1 - prec("0"), 1 - prec("0.5")))
    for lo, hi in bins:
        m = np.array([lo <= kb[k]["gt_px"] < hi for k in ks])
        print("   gt_px [%g,%g) n=%d dJF %+.5f" % (lo, hi, m.sum(), d[m].mean()))
