"""For a pi3_multi_<tag> cache whose anchor is another instant of the SAME rig: per camera, how far the target camera's
centre is from the anchor's centre of the same physical camera (units: median neighbour spacing of the anchor ring), and
the anchor ring's own azimuth structure.  Input only.

    python stage1/scripts/croom/anchor_agreement.py --scene CROOM_T04_001000_r00 --cache pi3_multi_t06
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Data"))
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    cd = os.path.join(a.data, a.scene, "Cache", a.cache)
    m = json.load(open(os.path.join(cd, "MANIFEST.json")))
    cams = m["cams"]
    T = {c: np.load(os.path.join(cd, f"{c}.npz"))["c2w"][:3, 3] for c in cams}
    A = {n.split("/")[-1]: np.asarray(v)[:3, 3] for n, v in m["anchors_c2w"].items()}
    order = sorted(cams)
    Ac = np.stack([A[c] for c in order])
    spacing = float(np.median(np.linalg.norm(np.diff(Ac, axis=0), axis=1)))
    ctr = Ac.mean(0); X = Ac - ctr; _, _, vt = np.linalg.svd(X, full_matrices=False)
    az = np.degrees(np.arctan2(X @ vt[1], X @ vt[0]))
    print("anchor azimuths:", [round(float(x), 1) for x in az])
    dist = {c: round(float(np.linalg.norm(T[c] - A[c]) / spacing), 2) for c in order}
    print("target-anchor centre distance / anchor spacing:")
    for c in order:
        print(f"  {c}: {dist[c]}")
    print("median", round(float(np.median(list(dist.values()))), 2), "cams > 1.0:", [c for c in order if dist[c] > 1.0])


if __name__ == "__main__":
    main()
