#!/usr/bin/env python3
"""Qualitative and plot figures for the TCSVT manuscript.

Reads only: Data/MVSeg/<scene>/{Video,Mask}, the published prediction folders,
Data/MVSeg/jf_r4.json, SCSam3/demo/MVSeg.json, MVSeed/runs/F0.json.
Writes only: paper/figures/fig_*.{pdf,jpg}.  Run inside the scsam3 container
(cv2, numpy, matplotlib); the host has none of them:

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD \
        -e HOME=/tmp -e MPLCONFIGDIR=/tmp scsam3 python paper/figures/make_figs.py
"""
import json, os, re, sys, statistics
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Data", "MVSeg")
OUT = os.path.join(REPO, "paper", "figures")
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json"), encoding="utf-8"))
H = "SegMaskSam3XW1CGPS4MFsCs10kG2"      # headline
B = "SegMaskSam3XW0MFs"                 # crop-tracking base (seed written as frame 0)
C = "SegMaskSam3XW0MFsCr10k"            # crop-tracked, no gate
PANEL_W = 560

# ---------- palette: one colour per object id, stable across cameras ----------
_tab = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors
def colour(obj):
    r, g, b = _tab[(obj * 7) % len(_tab)]
    return np.array([b, g, r]) * 255  # BGR for cv2

# ---------- data access ----------
def cams(scene):
    return sorted(d for d in os.listdir(os.path.join(DATA, CFG[scene]["folder"], "Mask")) if not d.endswith(".json"))
def cam_by_num(scene, n):
    for c in cams(scene):
        if int(re.search(r"(\d+)$", c).group(1)) == int(n):
            return c
    raise KeyError((scene, n))
def frame(scene, cam, t):
    c = CFG[scene]; p = os.path.join(DATA, c["folder"], "Video", cam, f"{c['start_frame']+t:06d}.jpg")
    im = cv2.imread(p); assert im is not None, p; return im
def gt(scene, cam, t):
    c = CFG[scene]; p = os.path.join(DATA, c["folder"], "Mask", cam, f"{c['start_frame']+t:06d}.png")
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE); assert m is not None, p; return m
def pred(scene, cam, t, obj, method=H):
    c = CFG[scene]; p = os.path.join(DATA, c["folder"], method, cam, str(c["start_frame"]+t), f"{obj}.png")
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m > 127)
def basic_objects(scene):
    ref = cam_by_num(scene, CFG[scene]["c_ini"])
    return sorted(int(v) for v in np.unique(gt(scene, ref, 0)) if v != 0)

# ---------- drawing ----------
def overlay(im, fills, contours, alpha=0.55, thick=None):
    """fills: [(mask, colour)], contours: [(mask, colour)] drawn on a copy of im."""
    out = im.astype(np.float32).copy()
    for m, col in fills:
        if m is None or not m.any(): continue
        out[m] = (1 - alpha) * out[m] + alpha * col
    out = out.astype(np.uint8)
    t = thick or max(2, im.shape[1] // 640)
    for m, col in contours:
        if m is None or not m.any(): continue
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, tuple(int(x) for x in col), t)
    return out
def shrink(im, w=PANEL_W):
    h = int(round(im.shape[0] * w / im.shape[1]))
    return cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
def bbox(m):
    ys, xs = np.where(m); return xs.min(), ys.min(), xs.max(), ys.max()
def crop_around(im, masks, scale=2.5, min_side=256):
    """square crop around the union bbox of the given masks, expanded by `scale`."""
    ms = [m for m in masks if m is not None and m.any()]
    Hh, Ww = im.shape[:2]
    if not ms: return im, (0, 0, Ww, Hh)
    x0, y0, x1, y1 = bbox(np.any(np.stack(ms), 0))
    side = int(min(max(scale * max(x1 - x0 + 1, y1 - y0 + 1), min_side), min(Hh, Ww)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    xa = int(np.clip(cx - side // 2, 0, Ww - side)); ya = int(np.clip(cy - side // 2, 0, Hh - side))
    return im[ya:ya+side, xa:xa+side], (xa, ya, side, side)
def sub(m, box):
    if m is None: return None
    xa, ya, w, h = box; return m[ya:ya+h, xa:xa+w]
WHITE = np.array([255, 255, 255]); BLACK = np.array([0, 0, 0])

# =====================================================================
# Fig. 2: cross-view propagation, three rigs
# =====================================================================
def fig_propagation():
    scenes = [("AlexaMeadeFacePaint", "FacePaint\nhemisphere, 46 fisheye"),
              ("Breakfast", "Breakfast\n5$\\times$3 planar, 15"),
              ("CBABasketball", "CBABasketball\narc, 30")]
    T = 20
    fig, axes = plt.subplots(len(scenes), 5, figsize=(7.16, 1.05 * len(scenes) + 0.35))
    for r, (scene, rig) in enumerate(scenes):
        ref = cam_by_num(scene, CFG[scene]["c_ini"]); others = [c for c in cams(scene) if c != ref]
        objs = basic_objects(scene)
        panels = []
        g0 = gt(scene, ref, 0)
        panels.append(("Reference $c_{\\mathrm{ini}}$, $t=0$ (input)", overlay(frame(scene, ref, 0), [(g0 == o, colour(o)) for o in objs], [])))
        for k, cam in enumerate(others[:2]):
            for t in (0, T):
                g = gt(scene, cam, t)
                fills = [(pred(scene, cam, t, o), colour(o)) for o in objs]
                conts = [(g == o, WHITE) for o in objs if (g == o).any()]
                panels.append((f"Camera {'AB'[k]}, $t={t}$", overlay(frame(scene, cam, t), fills, conts)))
        for c, (title, im) in enumerate(panels):
            ax = axes[r, c]; ax.imshow(shrink(im)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if r == 0: ax.set_title(title, fontsize=6.5, pad=2)
            if c == 0: ax.set_ylabel(f"({'abc'[r]}) " + rig, fontsize=5.5, labelpad=2)
    plt.subplots_adjust(wspace=0.02, hspace=0.04, left=0.045, right=0.995, top=0.93, bottom=0.01)
    fig.savefig(os.path.join(OUT, "fig_propagation.pdf"), dpi=200)
    plt.close(fig); print("fig_propagation.pdf")

# =====================================================================
# Fig. 3: seed failures (misplaced / empty)
# =====================================================================
def fig_seed_failures():
    F0 = json.load(open(os.path.join(REPO, "MVSeed", "runs", "F0.json")))["pairs"]
    mis = next(p for p in F0 if p["scene"] == "Welder" and p["cam"] == "camera_0004" and p["obj"] == 8)
    emp = max((p for p in F0 if p["kind"] == "empty" and p["scene"] != "Welder"), key=lambda p: p["gt_px"])
    rows = [mis, emp]
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.55))
    for r, p in enumerate(rows):
        scene, cam, obj = p["scene"], p["cam"], p["obj"]
        ref = cam_by_num(scene, CFG[scene]["c_ini"])
        lab = json.load(open(os.path.join(DATA, CFG[scene]["folder"], "Mask", "objects_labels.json")))[str(obj)]["label"]
        # (a) reference camera, GT fill, cropped around the object
        im, box = crop_around(frame(scene, ref, 0), [gt(scene, ref, 0) == obj], scale=3.0)
        a = overlay(im, [(sub(gt(scene, ref, 0) == obj, box), colour(obj))], [])
        # (b,c) target camera at t=0 and t=20: prediction fill + GT contour, same crop box
        g0 = gt(scene, cam, 0) == obj; p0 = pred(scene, cam, 0, obj)
        im0, box2 = crop_around(frame(scene, cam, 0), [g0, p0], scale=2.2)
        b = overlay(im0, [(sub(p0, box2), colour(obj))], [(sub(g0, box2), WHITE)])
        g20 = gt(scene, cam, 20) == obj; p20 = pred(scene, cam, 20, obj)
        c = overlay(sub(frame(scene, cam, 20), box2), [(sub(p20, box2), colour(obj))], [(sub(g20, box2), WHITE)])
        for k, (title, im) in enumerate([(f"$c_{{\\mathrm{{ini}}}}$, $t=0$ (input)", a), ("target, $t=0$ (seed)", b), ("target, $t=20$", c)]):
            ax = axes[r, k]; ax.imshow(shrink(im, 400)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if r == 0: ax.set_title(title, fontsize=6.5, pad=2)
            if k == 0: ax.set_ylabel(f"({'ab'[r]}) {scene}\n{lab} (obj {obj})\n{p['kind']}, $J_{{\\mathrm{{seed}}}}$={p['raw_J']:.2f}", fontsize=5.5, labelpad=2)
        print(f"  seed failure row {r}: {scene} {cam} obj {obj} {lab} kind={p['kind']} rawJ={p['raw_J']:.3f} gt_px={p['gt_px']} view_distance={p['view_distance']}")
    plt.subplots_adjust(wspace=0.03, hspace=0.05, left=0.14, right=0.995, top=0.92, bottom=0.01)
    fig.savefig(os.path.join(OUT, "fig_seed_failures.pdf"), dpi=200)
    plt.close(fig); print("fig_seed_failures.pdf")

# =====================================================================
# Fig. 4: crop re-tracking, gain and jump
# =====================================================================
def crop_window(seed, Hh, Ww, scale=4, min_side=256):
    x0, y0, x1, y1 = bbox(seed)
    side = int(min(max(scale * max(x1 - x0 + 1, y1 - y0 + 1), min_side), min(Hh, Ww)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    xa = int(np.clip(cx - side // 2, 0, Ww - side)); ya = int(np.clip(cy - side // 2, 0, Hh - side))
    return (xa, ya, side, side)
def fig_crop():
    rows = [("Breakfast", "v7", 19, 10), ("CBABasketball", "v06", 1, 14)]
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.5))
    for r, (scene, cam, obj, t) in enumerate(rows):
        ref = cam_by_num(scene, CFG[scene]["c_ini"]); isref = cam == ref
        seed = (gt(scene, cam, 0) == obj) if isref else pred(scene, cam, 0, obj, B)
        im0 = frame(scene, cam, 0); box = crop_window(seed, *im0.shape[:2])
        lab = json.load(open(os.path.join(DATA, CFG[scene]["folder"], "Mask", "objects_labels.json")))[str(obj)]["label"]
        g = gt(scene, cam, t) == obj
        pb = pred(scene, cam, t, obj, B); pc = pred(scene, cam, t, obj, C)
        # display region: the window enlarged 1.6x so that an object leaving the window stays visible
        Hh, Ww = im0.shape[:2]; xa, ya, w, h = box; side = int(min(1.6 * w, min(Hh, Ww)))
        cx, cy = xa + w // 2, ya + h // 2
        vx = int(np.clip(cx - side // 2, 0, Ww - side)); vy = int(np.clip(cy - side // 2, 0, Hh - side)); view = (vx, vy, side, side)
        def with_window(im):
            im = im.copy(); cv2.rectangle(im, (xa - vx, ya - vy), (xa - vx + w, ya - vy + h), (255, 255, 255), max(2, side // 300), lineType=cv2.LINE_AA)
            return im
        panels = [("seed, $t=0$ (window)", with_window(overlay(sub(im0, view), [(sub(seed, view), colour(obj))], []))),
                  (f"full-frame, $t={t}$", with_window(overlay(sub(frame(scene, cam, t), view), [(sub(pb, view), colour(obj))], [(sub(g, view), WHITE)]))),
                  (f"crop, $t={t}$", with_window(overlay(sub(frame(scene, cam, t), view), [(sub(pc, view), colour(obj))], [(sub(g, view), WHITE)])))]
        for k, (title, im) in enumerate(panels):
            ax = axes[r, k]; ax.imshow(shrink(im, 400)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_title(title, fontsize=6.5, pad=2)
            if k == 0: ax.set_ylabel(f"({'ab'[r]}) {scene}\n{lab.replace('_or_', '/')} (obj {obj})\nseed {int(seed.sum())} px, zoom {1008/box[2]:.1f}$\\times$", fontsize=5.5, labelpad=2)
        print(f"  crop row {r}: {scene} {cam} obj {obj} {lab} ref={isref} seed_px={int(seed.sum())} window={box} zoom={1008/box[2]:.2f}")
    plt.subplots_adjust(wspace=0.03, hspace=0.18, left=0.15, right=0.995, top=0.93, bottom=0.01)
    fig.savefig(os.path.join(OUT, "fig_crop.pdf"), dpi=200)
    plt.close(fig); print("fig_crop.pdf")

# =====================================================================
# Fig. 5: per-scene delta (basic), values from docs/muvod-comparison.md
# =====================================================================
def fig_scenes():
    d = [("AlexaMeadeFacePaint", 27.66), ("CBABasketball", 22.16), ("Breakfast", 19.02), ("MATF", 16.30),
         ("Dog", 12.85), ("Barn", 12.44), ("Blocks", 10.78), ("Painter", 10.45), ("CoffeeMartini", 10.32),
         ("PoznanStreet", 10.16), ("MartialArts", 10.09), ("Carpark", 9.19), ("Fencing", 7.50),
         ("FlameSteak", 7.33), ("AlexaMeadeExhibit", 7.25), ("Frog", 5.48), ("Welder", -0.73)]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    names = [n for n, _ in d][::-1]; vals = [v for _, v in d][::-1]
    cols = ["#b03a2e" if v < 0 else "#4d4d4d" for v in vals]
    ax.barh(names, vals, color=cols, height=0.72)
    ax.axvline(0, color="k", lw=0.6); ax.axvline(11.662, color="#888", lw=0.6, ls="--")
    ax.text(11.9, 0.2, "mean +11.66", fontsize=6, color="#555", va="bottom")
    ax.set_xlabel("$\\Delta$ J&F$^3$ over XMem (points, basic)", fontsize=7)
    ax.tick_params(axis="both", labelsize=6); ax.set_xlim(-3, 30)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.subplots_adjust(left=0.36, right=0.98, top=0.98, bottom=0.14)
    fig.savefig(os.path.join(OUT, "fig_scenes.pdf")); plt.close(fig); print("fig_scenes.pdf")

# =====================================================================
# Fig. 6: per-frame J&F, reference vs non-reference, basic set (jf_r4.json, H)
# =====================================================================
def fig_perframe():
    J = json.load(open(os.path.join(DATA, "jf_r4.json")))
    ref_rows, non_rows = [], []
    for e in J:
        if e["method"] != H or not e.get("result"): continue
        scene = e["dataset"]; basic = set(basic_objects(scene)); ref = cam_by_num(scene, CFG[scene]["c_ini"])
        pf = e["result"]["per_frame"]
        for i, o in enumerate(e["objects"]):
            if o not in basic: continue
            jf = [None if (pf["J"][i][t] is None or pf["F"][i][t] is None) else 0.5 * (pf["J"][i][t] + pf["F"][i][t]) for t in range(21)]
            (ref_rows if e["camera"] == ref else non_rows).append(jf)
    def curve(rows): return [statistics.mean([r[t] for r in rows if r[t] is not None]) for t in range(21)]
    cr, cn = curve(ref_rows), curve(non_rows)
    print(f"  per-frame: ref n={len(ref_rows)} t0/10/20 = {cr[0]:.3f}/{cr[10]:.3f}/{cr[20]:.3f}; nonref n={len(non_rows)} t0/10/20 = {cn[0]:.3f}/{cn[10]:.3f}/{cn[20]:.3f}")
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    ax.plot(range(21), cr, "o-", ms=2.5, lw=1, color="#1f5f8b", label=f"reference camera ($n$={len(ref_rows)})")
    ax.plot(range(21), cn, "s-", ms=2.5, lw=1, color="#b03a2e", label=f"non-reference cameras ($n$={len(non_rows)})")
    ax.set_xlabel("frame $t$", fontsize=7); ax.set_ylabel("mean J&F per pair", fontsize=7); ax.set_xticks(range(0, 21, 5))
    ax.set_xlim(-0.5, 20.5); ax.set_ylim(0.85, 1.0); ax.tick_params(labelsize=6); ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.2)
    fig.savefig(os.path.join(OUT, "fig_perframe.pdf")); plt.close(fig); print("fig_perframe.pdf")
    return cr, cn

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_propagation(); fig_seed_failures(); fig_crop(); fig_scenes(); fig_perframe()
