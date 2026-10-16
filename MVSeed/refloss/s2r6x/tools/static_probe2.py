import sys, json, io, numpy as np, cv2
from PIL import Image
sys.path.insert(0, sys.argv[1]); import rzip2
url, out = sys.argv[2], sys.argv[3]
videos = json.load(open(sys.argv[4])); fmt_img, fmt_ann = sys.argv[5], sys.argv[6]
T = (0, 10, 20)
rz = rzip2.RemoteZip(url)
names = []
for v in videos:
    names += [fmt_img.format(v=v, t=t) for t in T] + [fmt_ann.format(v=v, t=0)]
names = [n for n in names if n in rz.info]
blobs = rz.get_many(names)
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
res = {}
for v in videos:
    try:
        ims = {t: cv2.imdecode(np.frombuffer(blobs[fmt_img.format(v=v, t=t)], np.uint8), cv2.IMREAD_GRAYSCALE) for t in T}
        ann = np.array(Image.open(io.BytesIO(blobs[fmt_ann.format(v=v, t=0)])))
    except KeyError as e:
        res[v] = dict(err=str(e)); continue
    if ann.ndim == 3: ann = ann[..., 0]
    M = ann > 0; B = cv2.dilate(M.astype(np.uint8), K) == 0
    I0 = ims[0].astype(np.int16)
    r = dict(shape=list(I0.shape), b={}, d={}, gate={}, area={})
    win = cv2.createHanningWindow(I0.shape[::-1], cv2.CV_32F)
    for o in np.unique(ann):
        if o: r['area'][int(o)] = int((ann == o).sum())
    for t in T[1:]:
        It = ims[t].astype(np.int16); D = np.abs(It - I0)
        r['b'][t] = float(np.median(D[B])) if B.any() else None
        (dx, dy), _ = cv2.phaseCorrelate(ims[0].astype(np.float32), ims[t].astype(np.float32), win)
        r['d'][t] = float(np.hypot(dx, dy))
        for o in r['area']:
            reg = cv2.dilate((ann == o).astype(np.uint8), K) > 0
            r['gate'].setdefault(o, {})[t] = float(D[reg].mean())
    r['static'] = bool(max(r['b'].values()) <= 4 and max(r['d'].values()) <= 1.0)
    res[v] = r
json.dump(dict(fetched=rzip2.FETCHED[0], res=res), open(out, 'w'))
print('fetched MB', round(rzip2.FETCHED[0] / 1e6, 2), 'videos', len(res))
