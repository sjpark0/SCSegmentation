"""GT-free: on the 4,577 snapped c_ini frames of the old headline, how different is the tracker mask from mask_0
at full resolution, at the tracker's 1008x1008 mask resolution, and as 72x72 cell occupancy (memory grid)."""
import os, json, collections
from multiprocessing import Pool
import numpy as np, cv2
from common import REPO, CFG, SCENES, manifest_snaps
BASE = "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2"
def job(k):
    sc, cam, o, t = k; f = CFG[sc]["folder"]; s0 = CFG[sc]["start_frame"]
    a = cv2.imread(f"{REPO}/Data/MVSeg/{f}/{BASE}/{cam}/{s0+t}/{o}.png", 0) > 127
    b = cv2.imread(f"{REPO}/Data/MVSeg/{f}/{BASE}/{cam}/{s0}/{o}.png", 0) > 127
    def iou(x, y):
        u = np.count_nonzero(x | y); return 1.0 if u == 0 else np.count_nonzero(x & y) / u
    r = dict(full=iou(a, b), px=int(np.count_nonzero(a ^ b)), area=int(b.sum()))
    A = cv2.resize(a.astype(np.float32), (1008, 1008), interpolation=cv2.INTER_AREA); B = cv2.resize(b.astype(np.float32), (1008, 1008), interpolation=cv2.INTER_AREA)
    r["r1008"] = iou(A > 0.5, B > 0.5)
    a72 = cv2.resize(A, (72, 72), interpolation=cv2.INTER_AREA); b72 = cv2.resize(B, (72, 72), interpolation=cv2.INTER_AREA)
    r["occ72_l1_rel"] = float(np.abs(a72 - b72).sum() / max(b72.sum(), 1e-9))
    r["cells_changed_gt_0.25"] = int((np.abs(a72 - b72) > 0.25).sum()); r["cells_obj"] = int((b72 > 0.5).sum())
    return r
if __name__ == "__main__":
    ks = sorted(manifest_snaps("SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs"))
    with Pool(14) as p: R = p.map(job, ks, chunksize=20)
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    for key in ("full", "r1008", "occ72_l1_rel"):
        v = [r[key] for r in R]; print(f"{key}: median {q(v,.5):.4f} p10 {q(v,.1):.4f} p90 {q(v,.9):.4f}")
    v = [r["cells_changed_gt_0.25"] for r in R]; print(f"72x72 cells with |occupancy change| > 0.25: median {q(v,.5)}, p90 {q(v,.9)}, zero in {sum(x==0 for x in v)} of {len(v)}; object cells median {q([r['cells_obj'] for r in R],.5)}")
