"""How many temporal frames does one view step correspond to?

For every dataset and every tracked view v (the closure/all set), compare
  spatial : d(image[v][t], image[v-1][t])          - one view step, same instant
  temporal: d(image[v][t], image[v][t-k]) for k=1..8 - k frame steps, same view
on the frames the benchmark scores (start_frame .. start_frame+num_frame-1).

d is the mean absolute difference of the grayscale image downsampled to 128 px
on the short side - crude, but it is exactly the quantity the question is about:
how much does the picture change per view step versus per frame step.

k* = the k whose temporal distance is closest to the spatial distance, i.e.
"one view step looks like k* frame steps here".  Reported per dataset as the
median over (view, frame) pairs, with the quartiles.
"""
import json, os, sys, statistics as st
from multiprocessing import Pool
import cv2, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("SCSEG_REPO", "/home/sjpark/Documents/SCSegmentation")
CFG = json.load(open(f"{REPO}/SCSam3/demo/MVSeg.json"))
DS = ["AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast",
      "Carpark", "CoffeeMartini", "Dog", "Fencing", "FlameSteak", "Frog",
      "MATF", "Painter", "PoznanStreet", "Welder"]
KMAX = 8
SHORT = 128


def cam_name(c, p, p1):
    return f"{p}{c:0{p1}d}"


def load(path, cache):
    if path in cache:
        return cache[path]
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        cache[path] = None
        return None
    h, w = im.shape[:2]
    s = SHORT / min(h, w)
    im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32)
    if len(cache) > 400:
        cache.clear()
    cache[path] = im
    return im


def one_dataset(ds):
    d = CFG[ds]
    perms = d.get("perms") or list(range(d["start_cam"], d["num_cam"] + d["start_cam"]))
    names = [cam_name(c, d["prefix"], d["prefix1"]) for c in perms]
    scored = [names.index(cam_name(c, d["prefix"], d["prefix1"])) for c in d["cam_list"]]
    vroot = f"{REPO}/Data/MVSeg/{ds}/Video"
    start, n = d["start_frame"], d["num_frame"]
    # the sessions a closure run opens: 0..max(scored)
    views = range(1, max(scored) + 1)
    cache, ks, ratios = {}, [], []
    for v in views:
        for t in range(start + KMAX, start + n):      # need t-KMAX to exist
            cur = load(f"{vroot}/{names[v]}/{t:06d}.jpg", cache)
            nb = load(f"{vroot}/{names[v-1]}/{t:06d}.jpg", cache)
            if cur is None or nb is None:
                continue
            sp = float(np.abs(cur - nb).mean())
            temps = []
            for k in range(1, KMAX + 1):
                prev = load(f"{vroot}/{names[v]}/{t-k:06d}.jpg", cache)
                temps.append(float(np.abs(cur - prev).mean()) if prev is not None else float("nan"))
            if not temps or np.isnan(temps[0]) or temps[0] <= 0:
                continue
            kstar = min(range(1, KMAX + 1), key=lambda k: abs(temps[k-1] - sp))
            ks.append(kstar)
            ratios.append(sp / temps[0])              # view step / one frame step
    if not ks:
        return ds, None
    q = statistics_quartiles(ks)
    return ds, dict(n=len(ks), k_median=st.median(ks), k_q1=q[0], k_q3=q[2],
                    k_frac_at_max=sum(1 for k in ks if k == KMAX) / len(ks),
                    ratio_median=st.median(ratios), views=len(list(views)))


def statistics_quartiles(v):
    s = sorted(v)
    return [s[len(s)//4], s[len(s)//2], s[(3*len(s))//4]]


if __name__ == "__main__":
    out = {}
    with Pool(6) as pool:
        for ds, r in pool.imap_unordered(one_dataset, DS):
            out[ds] = r
            if r:
                print(f"{ds:22s} views {r['views']:2d}  k* median {r['k_median']:.1f} "
                      f"[q1 {r['k_q1']}, q3 {r['k_q3']}]  at-max {100*r['k_frac_at_max']:.0f}%  "
                      f"view/frame ratio {r['ratio_median']:.2f}  (n={r['n']})", flush=True)
            else:
                print(f"{ds:22s} no data", flush=True)
    json.dump(out, open(f"{ROOT}/view_vs_frame.json", "w"), indent=1)
    print("wrote view_vs_frame.json")
