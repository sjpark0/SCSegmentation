"""Host, no GT: for the headline folder of each scene, on c_ini frames t>start, count objects with
delta <= 4 (image-only half of S2-R6) and how many of those the tau half rejects (IoU(PNG_t, PNG_0) < 0.8).
Snapped PNGs are byte copies of frame 0 (IoU 1); a delta-pass non-snapped PNG is a tau rejection."""
import json, os, sys
import cv2, numpy as np
sys.dont_write_bytecode = True
ROOT = "/home/sjpark/Documents/SCSegmentation"
import importlib.util
sp = importlib.util.spec_from_file_location("ref_snap", os.path.join(ROOT, "SCSam3/ref_snap.py"))
rs = importlib.util.module_from_spec(sp); sp.loader.exec_module(rs)
cfg = json.load(open(os.path.join(ROOT, "SCSam3/demo/MVSeg.json")))
H = "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs"
out = {}
for name, d in cfg.items():
    if d.get("c_ini") is None: continue
    cam = f'{d["prefix"]}{d["c_ini"]:0{d["prefix1"]}d}'
    base = os.path.join(ROOT, "Data/MVSeg", d["folder"], H, cam)
    vid = os.path.join(ROOT, "Data/MVSeg", d["folder"], "Video", cam)
    if not os.path.isdir(base): continue
    frames = sorted((x for x in os.listdir(base) if x.isdigit()), key=int)
    m0 = {o: cv2.imread(os.path.join(base, frames[0], o), 0) > 127 for o in os.listdir(os.path.join(base, frames[0]))}
    shape = next(iter(m0.values())).shape
    def img(t):
        im = cv2.imread(os.path.join(vid, "%06d.jpg" % int(t)), 0)
        if im.shape != shape: im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return im.astype(np.int16)
    i0 = img(frames[0]); n = dpass = snap = tfail = 0; near = 0; fails = []
    for t in frames[1:]:
        it = img(t)
        for o in os.listdir(os.path.join(base, t)):
            if o not in m0: continue
            mt = cv2.imread(os.path.join(base, t, o), 0) > 127
            s, iou, diff = rs.decide(m0[o], mt, i0, it, np, cv2)
            if diff is None: continue
            n += 1
            if diff <= rs.DELTA:
                dpass += 1
                if s: snap += 1
                else: tfail += 1; fails.append([int(t), o, round(iou, 3)])
            elif diff <= rs.DELTA + 1: near += 1
    out[name] = dict(cam=cam, objects_frames=n, delta_pass=dpass, snapped=snap, tau_fail=tfail,
                     delta_4_to_5=near, tau_fail_examples=fails[:8])
    print(name, out[name], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
