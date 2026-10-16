"""GT-free: how many output PNGs change between two existing runs, and the pixel IoU of the changed ones."""
import sys, hashlib, cv2, numpy as np
from common import *
def run(ma, mb, which):
    tot = collections.Counter(); ious = []
    for sc in SCENES:
        f = CFG[sc]["folder"]; s0 = CFG[sc]["start_frame"]
        for cam in VI[sc]:
            d = dist(sc, cam)
            if not which(d): continue
            da, db = f"{REPO}/Data/MVSeg/{f}/{ma}/{cam}", f"{REPO}/Data/MVSeg/{f}/{mb}/{cam}"
            for t in sorted(os.listdir(da), key=int):
                for o in os.listdir(os.path.join(da, t)):
                    if int(o.split(".")[0]) not in OBJ[sc]["seed_ids_c_ini"]: continue
                    pa, pb = os.path.join(da, t, o), os.path.join(db, t, o)
                    key = "f0" if int(t) == s0 else "t>=1"
                    if not os.path.exists(pb): tot[key + "_missing"] += 1; continue
                    if open(pa, "rb").read() == open(pb, "rb").read(): tot[key + "_same"] += 1; continue
                    tot[key + "_diff"] += 1
                    if key == "t>=1":
                        a = cv2.imread(pa, 0) > 127; b = cv2.imread(pb, 0) > 127
                        u = np.count_nonzero(a | b); ious.append(1.0 if u == 0 else np.count_nonzero(a & b) / u)
    ious.sort()
    q = (lambda p: ious[int(p * (len(ious) - 1))]) if ious else (lambda p: None)
    print(f"{ma} vs {mb}: {dict(tot)}; changed t>=1 IoU median {q(.5)}, p10 {q(.1)}, min {q(0)}")
mode = sys.argv[1]
if mode == "xw":
    run("SegMaskSam3XW0MFs", "SegMaskSam3XW1CGPS4MFs", lambda d: abs(d) == 1)
    run("SegMaskSam3XW0MFs", "SegMaskSam3XW1CGPS4MFs", lambda d: d == 0)
    run("SegMaskSam3XW0MFs", "SegMaskSam3XW1CGPS4MFs", lambda d: abs(d) >= 2)
elif mode == "e7":
    run("SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2", "SegMaskSam3XW1CGPS4MSdgtFsCs10kG2", lambda d: d == 0)
    run("SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2", "SegMaskSam3XW1CGPS4MSdgtFsCs10kG2", lambda d: abs(d) == 1)
if mode == "xw0":
    run("SegMaskSam3XW0MSdgeoset_repFs", "SegMaskSam3XW0MSdgtFs", lambda d: d == 0)
    run("SegMaskSam3XW0MFs", "SegMaskSam3XW0MSdgtFs", lambda d: d == 0)
