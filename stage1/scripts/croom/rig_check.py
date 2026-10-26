"""Input-only rig diagnostics of a Pi3 cache: E15 rig check phi (geomem.id3d.rig_phi, threshold phi_rig 0.12) and the
shape of the recovered camera ring (neighbour spacing of the camera centres, breaks where a gap exceeds k x median,
azimuth of each camera about the centroid in the best-fit plane).

    python stage1/scripts/croom/rig_check.py --scene CROOM_T03_004181_r00 [--cache pi3] [--json out.json]
"""
import argparse
import json
import os
import sys

import numpy as np

S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, S1)
from geomem import id3d  # noqa: E402

PHI_RIG = 0.12
BREAK_K = 4.0


def ring_shape(cache, cams):
    C = np.stack([np.load(os.path.join(cache, f"{c}.npz"))["c2w"][:3, 3] for c in cams])
    ctr = C.mean(0)
    X = C - ctr
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    az = np.degrees(np.arctan2(X @ e2, X @ e1))
    # the CROOM rig is an open ring: cam_31 -> cam_00 is a physical opening (T06: ~94 deg), so steps are i -> i+1, i < n-1
    gaps = np.linalg.norm(C[1:] - C[:-1], axis=1)
    med = float(np.median(gaps))
    breaks = [int(i) for i in np.nonzero(gaps > BREAK_K * med)[0]]
    # monotonic ring: the azimuth should advance in one direction with no sign change
    # angular steps i -> i+1 (cyclic, wrapped to (-180, 180]); a true ring has ~360/n per step, one sign, no jump
    d = (az[1:] - az[:-1] + 180.0) % 360.0 - 180.0
    step_med = float(np.median(np.abs(d)))
    jumps = [int(i) for i in np.nonzero(np.abs(d) > BREAK_K * step_med)[0]]
    sign = np.sign(d[np.abs(d) <= BREAK_K * step_med])
    return dict(azimuth_deg=[round(float(a), 1) for a in az], gap_over_median=[round(float(g / med), 2) for g in gaps],
                breaks_after=breaks, az_step_deg=[round(float(x), 1) for x in d], az_step_median=round(step_med, 2),
                az_jumps_after=jumps, one_direction=bool((sign > 0).all() or (sign < 0).all()),
                ring_ok=bool(not breaks and not jumps and ((sign > 0).all() or (sign < 0).all())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default=os.path.join(S1, "Data"))
    ap.add_argument("--cache", default="pi3")
    ap.add_argument("--json", default=None)
    ap.add_argument("--order", default=None, help="comma list: the cameras in physical ring order (default: index order)")
    a = ap.parse_args()
    sd = os.path.join(a.data, a.scene)
    meta = json.load(open(os.path.join(sd, "meta.json")))
    cache = os.path.join(sd, "Cache", a.cache)
    cams = list(meta["cams"])
    order = a.order.split(",") if a.order else cams
    assert sorted(order) == sorted(cams), "order must list every camera once"
    med, per = id3d.rig_phi(cache, meta["ref_cam"], cams)
    out = dict(scene=a.scene, cache=a.cache, ref_cam=meta["ref_cam"], rig_phi_median=round(med, 4),
               rig_ok=bool(med <= PHI_RIG), phi_per_cam={c: round(float(v), 3) for c, v in per.items()},
               ring_order=order, ring=ring_shape(cache, order))
    print(f"{a.scene} [{a.cache}] ref {meta['ref_cam']}: phi {med:.3f} -> rig_ok {out['rig_ok']}; "
          f"centre breaks after {out['ring']['breaks_after']}, azimuth jumps after {out['ring']['az_jumps_after']}, "
          f"one direction {out['ring']['one_direction']} -> ring_ok {out['ring']['ring_ok']}")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
