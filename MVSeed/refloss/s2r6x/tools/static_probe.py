import sys, json, io, numpy as np, cv2
sys.path.insert(0, sys.argv[1]); import rzip
url, root, out = sys.argv[2], sys.argv[3], sys.argv[4]
videos = json.load(open(sys.argv[5]))
fmt_img, fmt_ann = sys.argv[6], sys.argv[7]   # e.g. 'DAVIS/JPEGImages/480p/{v}/{t:05d}.jpg'
z = rzip.open_zip(url)
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
res = {}
for v in videos:
    try:
        ims = {t: cv2.imdecode(np.frombuffer(z.read(fmt_img.format(v=v, t=t)), np.uint8), cv2.IMREAD_GRAYSCALE) for t in (0, 10, 20)}
        from PIL import Image
        ann = np.array(Image.open(io.BytesIO(z.read(fmt_ann.format(v=v, t=0)))))
    except KeyError as e:
        res[v] = dict(err=str(e)); continue
    if ann.ndim == 3: ann = ann[..., 0]
    M = ann > 0
    B = cv2.dilate(M.astype(np.uint8), K) == 0
    I0 = ims[0].astype(np.int16)
    r = dict(shape=list(I0.shape), b={}, d={}, gate={})
    win = cv2.createHanningWindow(I0.shape[::-1], cv2.CV_32F)
    for t in (10, 20):
        It = ims[t].astype(np.int16)
        D = np.abs(It - I0)
        r['b'][t] = float(np.median(D[B])) if B.any() else None
        (dx, dy), _ = cv2.phaseCorrelate(ims[0].astype(np.float32), ims[t].astype(np.float32), win)
        r['d'][t] = float(np.hypot(dx, dy))
        for o in np.unique(ann):
            if o == 0: continue
            reg = cv2.dilate((ann == o).astype(np.uint8), K) > 0
            r['gate'].setdefault(int(o), {})[t] = float(D[reg].mean())
    r['static'] = bool(max(r['b'].values()) <= 4 and max(r['d'].values()) <= 1.0)
    res[v] = r
    print(v, r['static'], {t: round(r['b'][t], 1) for t in r['b']}, {t: round(r['d'][t], 2) for t in r['d']}, 'MB', round(rzip.FETCHED[0] / 1e6, 1), flush=True)
json.dump(dict(fetched=rzip.FETCHED[0], res=res), open(out, 'w'))
