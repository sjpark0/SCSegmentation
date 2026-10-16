import os, sys, cv2, numpy as np, json
a, b = sys.argv[1], sys.argv[2]
snapj = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
snapped = set()
if snapj:
    for fr in snapj["frames"]:
        for o, r in fr["objs"].items():
            if r.get("snap"): snapped.add((fr["t"], int(o)))
res = {}
for cam in sorted(os.listdir(a)):
    if not os.path.isdir(os.path.join(a, cam)): continue
    for t in sorted(os.listdir(os.path.join(a, cam)), key=int):
        for f in os.listdir(os.path.join(a, cam, t)):
            pa, pb = os.path.join(a, cam, t, f), os.path.join(b, cam, t, f)
            ma = cv2.imread(pa, 0) > 127; mb = cv2.imread(pb, 0) > 127
            x = int(np.count_nonzero(ma ^ mb))
            if x == 0: continue
            u = int(np.count_nonzero(ma | mb)); o = int(f.split(".")[0])
            r = res.setdefault(cam, {"n": 0, "xor": 0, "union": 0, "first_t": 99, "objs": {}, "iou_ab": []})
            r["n"] += 1; r["xor"] += x; r["union"] += u; r["first_t"] = min(r["first_t"], int(t))
            r["iou_ab"].append(1 - x / u if u else 1.0)
            ro = r["objs"].setdefault(o, [0, 0]); ro[0] += 1; ro[1] += x
for cam, r in res.items():
    ious = sorted(r["iou_ab"])
    print(f"{cam}: {r['n']} PNG differ, first frame {r['first_t']}, xor px {r['xor']} (/{r['union']} union = {r['xor']/max(1,r['union']):.4f}), IoU(a,b) median {ious[len(ious)//2]:.4f} min {ious[0]:.4f}")
    print("   per obj (n PNG, xor px):", dict(sorted(r["objs"].items())))
