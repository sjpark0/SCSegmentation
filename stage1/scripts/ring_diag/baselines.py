"""Read-only: viewpoint change between the reference camera and every other camera, per stage-1 scene (Pi3 cache).
angle_axis = angle between optical axes; angle_obj = angle at the reference view's median 3D point between the two
camera centres (how far around the content the target camera sits)."""
import json, os, sys
import numpy as np
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points
D = os.path.join(REPO, "stage1/Data")
res = {}
for s in sorted(os.listdir(D)):
    m = os.path.join(D, s, "meta.json")
    if s.startswith("_") or not os.path.isfile(m) or not os.path.isdir(os.path.join(D, s, "Cache/pi3")):
        continue
    meta = json.load(open(m)); ref = meta["ref_cam"]
    z = np.load(os.path.join(D, s, "Cache/pi3", f"{ref}.npz"))
    T_r = z["c2w"]; valid = (z["conf"] > CONF_MIN) & (z["depth"] > 0)
    P = local_points(z["depth"], z["K"])[valid]
    X = T_r[:3, :3] @ np.median(P, 0) + T_r[:3, 3]           # world coords of the ref view's median point
    a_ax, a_obj = [], []
    for c in meta["cams"]:
        if c == ref: continue
        T = np.load(os.path.join(D, s, "Cache/pi3", f"{c}.npz"))["c2w"]
        a_ax.append(np.degrees(np.arccos(np.clip(T_r[:3, 2] @ T[:3, 2], -1, 1))))
        u, v = T_r[:3, 3] - X, T[:3, 3] - X
        a_obj.append(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
    a_ax, a_obj = np.array(a_ax), np.array(a_obj)
    res[s] = dict(n=len(a_ax), axis_max=float(a_ax.max()), axis_p90=float(np.percentile(a_ax, 90)),
                  obj_max=float(a_obj.max()), obj_p90=float(np.percentile(a_obj, 90)), frac_obj_gt60=float((a_obj > 60).mean()),
                  frac_obj_gt90=float((a_obj > 90).mean()))
    r = res[s]
    print(f"{s:22s} n={r['n']:2d} axis max {r['axis_max']:6.1f} p90 {r['axis_p90']:6.1f} | around-content max {r['obj_max']:6.1f} p90 {r['obj_p90']:6.1f} >60deg {r['frac_obj_gt60']:.2f} >90deg {r['frac_obj_gt90']:.2f}")
json.dump(res, open(os.path.join(os.path.dirname(__file__), "baselines.json"), "w"), indent=1)
