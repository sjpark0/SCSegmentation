"""GT-free: decisions of the frozen S2-R6 rule (0.8, 4, 31) on the annotated cameras of existing GT-seed runs
(E7: frame-0 PNG of every scored camera == its frame-0 GT seed).  Reads output PNGs and video only; no GT at t>=1.
Writes nothing but a JSON of decision features (iou, imgdiff) for counting."""
import sys, os, json, collections
from multiprocessing import Pool
import numpy as np, cv2
sys.path.insert(0, "/home/sjpark/Documents/SCSegmentation/SCSam3")
import ref_snap as RS
REPO = "/home/sjpark/Documents/SCSegmentation"; DATA = f"{REPO}/Data/MVSeg"
CFG = json.load(open(f"{REPO}/SCSam3/demo/MVSeg.json")); OBJ = json.load(open(f"{REPO}/docs/raw/muvod_object_sets.json"))
def job(a):
    scene, method, cam = a
    d = CFG[scene]; src = os.path.join(DATA, d["folder"], method, cam); vdir = os.path.join(DATA, d["folder"], "Video", cam)
    frames = sorted((x for x in os.listdir(src) if x.isdigit()), key=int); assert frames[0] == str(d["start_frame"])
    objs = sorted(os.listdir(os.path.join(src, frames[0])))
    m0 = {o: cv2.imread(os.path.join(src, frames[0], o), cv2.IMREAD_GRAYSCALE) > 127 for o in objs}
    shape = next(iter(m0.values())).shape
    def img(t):
        im = cv2.imread(os.path.join(vdir, "%06d.jpg" % int(t)), cv2.IMREAD_GRAYSCALE)
        if im.shape != shape: im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return im.astype(np.int16)
    i0 = img(frames[0]); rows = []
    for k, t in enumerate(frames[1:], 1):
        it = img(t)
        for o in sorted(os.listdir(os.path.join(src, t))):
            if o not in m0: continue
            mt = cv2.imread(os.path.join(src, t, o), cv2.IMREAD_GRAYSCALE) > 127
            s, iou, diff = RS.decide(m0[o], mt, i0, it, np, cv2)
            rows.append(dict(scene=scene, method=method, cam=cam, obj=int(o.split(".")[0]), t=k, snap=bool(s), iou=iou, diff=diff, m0px=int(m0[o].sum())))
    return rows
if __name__ == "__main__":
    jobs = []
    for m in sys.argv[2:]:
        for sc in sorted(OBJ):
            for cam in OBJ[sc]["cameras"]:
                jobs.append((sc, m, cam))
    with Pool(14) as p: out = [r for rs in p.imap_unordered(job, jobs) for r in rs]
    json.dump(out, open(sys.argv[1], "w")); print("rows", len(out))
