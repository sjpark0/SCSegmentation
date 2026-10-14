"""Host measurement of the registered half-plane cut on the training reference label maps (no torch).
Object cells = nearest resize to 1008 (torch 'nearest' index floor(i*h/1008)) -> 14x14 mean >= 0.5 on 72x72.
Cut rule (registered): side ~ U{L,R,T,B}, f ~ U(0.3,0.7) = target kept fraction; n = |C|; k = floor((1-f) n);
if k == 0 no cut; coordinate a sorted ascending (L: x, R: -x, T: y, B: -y); a_c = a_sorted[k]; drop every grid
cell with a < a_c.  Realised dropped fraction of object cells and of all grid cells."""
import json, os, sys
import numpy as np
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.data import enumerate_samples, _labelmap, memory_rng  # noqa
from mvseg2d.io import load_meta, load_labelmap, labelmap_objects  # noqa
from methods.common import frame_dir  # noqa
DATA = os.path.join(REPO, "stage1", "Data")
SCENES = sorted(d for d in os.listdir(DATA) if os.path.isfile(os.path.join(DATA, d, "meta.json")))
MIN_CELLS, KEEP = 20, (0.3, 0.7)

def cells_of(lab):
    h, w = lab.shape
    ri = (np.arange(1008) * h // 1008); ci = (np.arange(1008) * w // 1008)
    l = lab[ri][:, ci]
    ids = labelmap_objects(lab)
    out = {}
    for o in ids:
        m = (l == o).reshape(72, 14, 72, 14).mean(axis=(1, 3)) >= 0.5
        out[o] = (m, int((lab == o).sum()))
    return out

def cut(m, g):
    ys, xs = np.nonzero(m); n = len(xs)
    side = int(g.integers(4)); f = float(g.uniform(*KEEP))
    k = int(np.floor((1 - f) * n))
    yy, xx = np.mgrid[0:72, 0:72]
    a_cells = [xs, -xs, ys, -ys][side]; a_grid = [xx, -xx, yy, -yy][side]
    if k == 0:
        return f, 0.0, 0.0, False
    a_c = np.sort(a_cells)[k]
    drop = a_grid < a_c
    dropped_obj = int((drop & m).sum())
    return f, dropped_obj / n, float(drop.mean()), dropped_obj > 0

samples = enumerate_samples(DATA, SCENES, frame_stride=2)
rows = []
cache = {}
for s in samples:
    key = (s.scene, s.t, s.ref_cam)
    if key not in cache:
        cache[key] = cells_of(_labelmap(DATA, s, s.ref_cam))
    rows_s = []
    for o, (m, px) in cache[key].items():
        rows_s.append((s.scene, int(m.sum()), px))
    rows.append(rows_s)
g = np.random.default_rng(0)
n_rows = sum(len(r) for r in rows)
elig = [(sc, c, px) for r in rows for (sc, c, px) in r if c >= MIN_CELLS]
px_all = sum(px for r in rows for (_, _, px) in r)
px_el = sum(px for (_, _, px) in elig)
print(f"samples {len(samples)}  object rows {n_rows}  eligible (>= {MIN_CELLS} cells) {len(elig)} = {len(elig)/n_rows:.3f} of rows, "
      f"{px_el/px_all:.3f} of reference GT px")
print("samples with >= 1 eligible object:", sum(any(c >= MIN_CELLS for (_, c, _) in r) for r in rows) / len(rows))
# realised cut statistics over unique (scene,t,ref,obj) masks, 5 draws each
fr, dobj, dgrid, nocut = [], [], [], 0
for key, objs in cache.items():
    for o, (m, px) in objs.items():
        if m.sum() < MIN_CELLS:
            continue
        for _ in range(5):
            f, d, dg, ok = cut(m, g)
            if not ok:
                nocut += 1; continue
            fr.append(f); dobj.append(d); dgrid.append(dg)
dobj, dgrid, fr = map(np.array, (dobj, dgrid, fr))
print(f"cut draws {len(dobj) + nocut}, no-cut (k == 0 or nothing dropped) {nocut}")
print(f"realised dropped fraction of object cells: mean {dobj.mean():.3f}  p10/50/90 {np.percentile(dobj,[10,50,90]).round(3)}  "
      f"nominal 1-f mean {(1-fr).mean():.3f}; realised - nominal mean {(dobj-(1-fr)).mean():+.4f}, min {(dobj-(1-fr)).min():+.3f}")
print(f"realised dropped fraction of the 72x72 grid: mean {dgrid.mean():.3f}  p10/50/90 {np.percentile(dgrid,[10,50,90]).round(3)}")
kept = 1 - dobj
print(f"realised kept fraction >= f always: {bool((kept + 1e-9 >= fr).all())}")
# per-scene eligible share
per = {}
for r in rows:
    for (sc, c, px) in r:
        a = per.setdefault(sc, [0, 0, 0, 0]); a[0] += 1; a[1] += c >= MIN_CELLS; a[2] += px; a[3] += px * (c >= MIN_CELLS)
print("per scene eligible rows / px share:", {k: (round(v[1]/v[0], 2), round(v[3]/v[2], 2)) for k, v in sorted(per.items())})
# keyframe: reference-view cell counts of the 22 under-cover objects
sf = json.load(open(os.path.join(REPO, "MVSeed", "runs", "seedfail.json")))["rows"]
under = [r for r in sf if r["JF"] < 0.8 and r.get("kind", "").startswith("under")]
good = [r for r in sf if r["JF"] >= 0.8]
kc = {}
def kf_cells(scene):
    if scene not in kc:
        meta = load_meta(DATA, scene)
        lab = load_labelmap(os.path.join(frame_dir(DATA, scene, int(meta["ref_frame"])), "masks", f"{meta['ref_cam']}.png"))
        kc[scene] = cells_of(lab)
    return kc[scene]
print("under-cover 22 (scene cam obj label kind: reference cells):")
ue = 0
for r in under:
    c = kf_cells(r["scene"]).get(r["obj"], (np.zeros((72, 72), bool), 0))[0].sum()
    ue += c >= MIN_CELLS
    print(f"  {r['scene']} {r['cam']} {r['obj']} {r.get('label')} {r['kind']}: {int(c)}")
print(f"under-cover with >= {MIN_CELLS} reference cells: {ue}/{len(under)}")
ge = sum(kf_cells(r["scene"]).get(r["obj"], (np.zeros((72, 72), bool), 0))[0].sum() >= MIN_CELLS for r in good)
print(f"good seeds whose object has >= {MIN_CELLS} reference cells: {ge}/{len(good)}  (rows in seedfail: {len(sf)})")
