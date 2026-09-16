#!/usr/bin/env python3
"""Per-frame J&F by object type and per-pair start-vs-end change (reviewer point 6).

Reads Data/MVSeg/jf_<tag>.json (eval_jf.py v2 with per_frame) and each scene's
Mask/objects_labels.json (type: dynamic / static / env).  Basic set = objects in the
c_ini frame-0 ground truth (SCSam3/demo/MVSeg.json c_ini).  CPU only, host python.

    python3 eval/drift_by_type.py --raw jf_r4.json --method SegMaskSam3XW1CGPS4MFsCs10kG2

Prints (1) mean J&F per frame split by reference/non-reference x type, (2) the
distribution of J&F(t=20) - J&F(t=1) per pair, and (3) the share of pairs that fall
below 0.5 at t=20 having been above it at t=1 (and the reverse).  Writes
MVSeed/runs/drift_by_type.json.
"""
import argparse, json, os, re, statistics
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data", "MVSeg")
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json"), encoding="utf-8"))

def cam_by_num(scene, n):
    for c in sorted(os.listdir(os.path.join(DATA, CFG[scene]["folder"], "Mask"))):
        if not c.endswith(".json") and int(re.search(r"(\d+)$", c).group(1)) == int(n):
            return c
    raise KeyError((scene, n))

def basic_ids(scene):
    from PIL import Image
    import array
    c = CFG[scene]; ref = cam_by_num(scene, c["c_ini"])
    im = Image.open(os.path.join(DATA, c["folder"], "Mask", ref, f"{c['start_frame']:06d}.png"))
    return sorted(set(im.getdata()) - {0})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="jf_r4.json")
    ap.add_argument("--method", default="SegMaskSam3XW1CGPS4MFsCs10kG2")
    ap.add_argument("--out", default=os.path.join(REPO, "MVSeed", "runs", "drift_by_type.json"))
    a = ap.parse_args()
    J = json.load(open(os.path.join(DATA, a.raw)))
    rows = []
    for e in J:
        if e["method"] != a.method or not e.get("result"): continue
        scene = e["dataset"]; basic = set(basic_ids(scene)); ref = cam_by_num(scene, CFG[scene]["c_ini"])
        labels = json.load(open(os.path.join(DATA, CFG[scene]["folder"], "Mask", "objects_labels.json")))
        pf = e["result"]["per_frame"]
        for i, o in enumerate(e["objects"]):
            if o not in basic: continue
            jf = [None if (pf["J"][i][t] is None or pf["F"][i][t] is None) else 0.5 * (pf["J"][i][t] + pf["F"][i][t]) for t in range(21)]
            rows.append(dict(scene=scene, cam=e["camera"], obj=o, ref=e["camera"] == ref,
                             type=labels[str(o)]["type"], jf=jf))
    def curve(rs): return [round(statistics.mean(v), 4) if (v := [r["jf"][t] for r in rs if r["jf"][t] is not None]) else None for t in range(21)]
    out = {"method": a.method, "n_pairs": len(rows), "curves": {}, "delta": {}, "transitions": {}}
    print(f"{a.method}: {len(rows)} basic pairs")
    for ref in (True, False):
        for typ in ("dynamic", "static", "env", "all"):
            rs = [r for r in rows if r["ref"] == ref and (typ == "all" or r["type"] == typ)]
            if not rs: continue
            c = curve(rs); key = f"{'ref' if ref else 'nonref'}/{typ}"
            out["curves"][key] = {"n": len(rs), "jf": c}
            print(f"  {key:16s} n={len(rs):4d}  t0 {c[0]:.3f}  t1 {c[1]:.3f}  t10 {c[10]:.3f}  t20 {c[20]:.3f}")
    # per-pair change from the first tracked frame to the last
    for ref in (True, False):
        for typ in ("dynamic", "static", "env", "all"):
            rs = [r for r in rows if r["ref"] == ref and (typ == "all" or r["type"] == typ) and r["jf"][1] is not None and r["jf"][20] is not None]
            if not rs: continue
            d = sorted(r["jf"][20] - r["jf"][1] for r in rs)
            n = len(d); q = lambda p: d[min(n - 1, int(p * n))]
            down = sum(1 for x in d if x < -0.1); up = sum(1 for x in d if x > 0.1)
            fell = sum(1 for r in rs if r["jf"][1] >= 0.5 and r["jf"][20] < 0.5); rose = sum(1 for r in rs if r["jf"][1] < 0.5 and r["jf"][20] >= 0.5)
            key = f"{'ref' if ref else 'nonref'}/{typ}"
            out["delta"][key] = {"n": n, "mean": round(statistics.mean(d), 4), "median": round(q(0.5), 4), "p10": round(q(0.1), 4), "p90": round(q(0.9), 4), "down_gt_0.1": down, "up_gt_0.1": up}
            out["transitions"][key] = {"fell_below_0.5": fell, "rose_above_0.5": rose}
            print(f"  Δ(t20−t1) {key:16s} n={n:4d} mean {statistics.mean(d):+.4f} median {q(0.5):+.4f} p10 {q(0.1):+.3f} p90 {q(0.9):+.3f} | drop>0.1: {down} rise>0.1: {up} | fell<0.5: {fell} rose≥0.5: {rose}")
    json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)

if __name__ == "__main__":
    main()
