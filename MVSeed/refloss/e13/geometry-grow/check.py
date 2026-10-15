import json, glob, sys
dp = json.load(open("/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_dev/dev_pairs.json"))
base = {(r["scene"], r["t"], r["cam"], r["obj"]): r for r in dp["rows"]}
maxd, bad, n = 0, 0, 0
for f in sorted(glob.glob(sys.argv[1] + "/frames/*.json")):
    F = json.load(open(f))
    for v in F["views"]:
        for o, (J, Fm) in v["base"].items():
            k = (v["scene"], v["t"], v["cam"], int(o)); b = base[k]; n += 1
            maxd = max(maxd, abs(J - b["J"]), abs(Fm - b["F"]))
            g = v["gtcls"][o]
            if g != [b["gt_vis"], b["gt_outside"], b["gt_occluded"], b["gt_nodepth"]]:
                bad += 1; print("cls mismatch", k, g, [b["gt_vis"], b["gt_outside"], b["gt_occluded"], b["gt_nodepth"]])
print(f"pairs {n} max|dJF base| {maxd:.2e} class mismatches {bad}")
