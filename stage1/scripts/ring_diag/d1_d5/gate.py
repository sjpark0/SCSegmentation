"""Round-1 reproj gate statistics recomputed from round1_conf.json + Pi3 centres + verify_rows.json (read-only)."""
import json, os, sys
import numpy as np
D15 = "/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_d15"
arm, r1arm = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("preset", "preset_r1")
conf = json.load(open(f"{D15}/data/T06_flag_ganggang_F/Result/{arm}/round1_conf.json"))
rows = json.load(open(f"{D15}/verify/verify_rows.json"))
idx = {(r["cam"], r["p"]): r for r in rows if r["arm"] == r1arm}
CAMS = [f"cam_{i:02d}" for i in range(32)]
cache = f"{D15}/data/T06_flag_ganggang_F/Cache/pi3"
C = {c: np.load(os.path.join(cache, f"{c}.npz"))["c2w"][:3, 3] for c in CAMS}
def passes(c, p):
    r = conf[c][str(p)]
    if not r["presence"] > 0:
        return False
    ov = r["overlap"]
    return (r["iou"] >= 0.7) if ov is None else (ov >= 0.5)
def st(r):
    return "empty" if r["cov_q"] == 0 else ("ok" if r["cov_q"] == r["p"] else "wrong")
P = [(c, p) for c in CAMS if c != "cam_06" for p in range(1, 7) if idx[(c, p)]["gt_px"] > 0]
out = {}
for nm, lo, hi in (("near", 1, 6), ("far", 7, 16)):
    S = [k for k in P if lo <= idx[k]["d"] <= hi]
    for s in ("ok", "wrong", "empty"):
        T = [k for k in S if st(idx[k]) == s]
        out[f"{nm}_{s}_pass"] = f"{sum(passes(*k) for k in T)}/{len(T)}"
    nomem = 0
    for (c, p) in S:
        cand = [o for _, o in sorted((np.linalg.norm(C[o] - C[c]), o) for o in CAMS if o != c)[:3] if o != "cam_06"]
        nomem += not any(passes(o, p) for o in cand)
    out[f"{nm}_no_neighbour_mem_for_p"] = f"{nomem}/{len(S)} = {nomem/len(S):.2f}"
wp = [(k, round(conf[k[0]][str(k[1])]["overlap"] or -1, 2), idx[k]["d"]) for k in P if st(idx[k]) == "wrong" and passes(*k)]
out["wrong_passing"] = wp
# ring order check: are the 3 nearest Pi3 centres the ring neighbours?
nn = {c: [o for _, o in sorted((np.linalg.norm(C[o] - C[c]), o) for o in CAMS if o != c)[:3]] for c in CAMS}
out["nn_sample"] = {c: nn[c] for c in ("cam_00", "cam_12", "cam_22", "cam_24")}
print(json.dumps(out, indent=1))
