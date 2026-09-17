#!/usr/bin/env python3
"""Figures for the plenoptic-rig section (docs/plenoptic-P.md, paper/figures/fig_plenoptic_*.pdf).

    python plenoptic/make_figs.py ring   [--method SegMaskSam3XW0MFs] [--t 0]
        reference camera + cameras at ring distance 2, 4, 8, 12, 16 (t = 0): prediction filled per person,
        ground truth outlined in white  -> fig_plenoptic_ring.pdf
    python plenoptic/make_figs.py event --cam cam_XX --obj K --frames a b [--neighbor cam_YY]
        rows: control / headline / neighbour camera, columns: frames a-1, a, mid, b, b+1 -> fig_plenoptic_event_<cam>_<obj>.pdf
    python plenoptic/make_figs.py curve
        Q2 ring-distance curve from plenoptic/score.json (GT / text / point seeds) -> fig_plenoptic_curve.pdf

Runs in the scsam3 container (cv2, matplotlib); reads only, writes PDFs under paper/figures/ and PNG previews under plenoptic/.
"""
import argparse, json, os
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600          # IEEE: photos >= 300 dpi at printed size; matplotlib resamples imshow rasters at savefig dpi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(REPO, "PlenopticServer1", "CROOM", "SegResult", os.environ.get("PLEN_SCENE", "T06_flag_ganggang"))   # masks: PLEN_SCENE=T06_flag_ganggang_L540 for P4
IMG_SCENE = os.path.join(REPO, "PlenopticServer1", "CROOM", "SegResult", os.environ.get("PLEN_IMG_SCENE", os.environ.get("PLEN_SCENE", "T06_flag_ganggang")))  # frames may come from the 1080p copy
FIG_SCALE = float(os.environ.get("PLEN_FIG_SCALE", 0.5))     # display scale of the frames (1.0 for crops)
OUT = os.path.join(REPO, "paper", "figures"); PREV = os.path.join(REPO, "plenoptic")
CTRL, HEAD, TXT = "SegMaskSam3XW0MFs", "SegMaskSam3XW1CGPS4MFsCs10kG2", "SegMaskSam3XW0MFs_txt"
REF, NCAM = 6, 32
COLS = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180]], dtype=float)
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})


_FSIZE = {}


def frame(cam, t, scale=None):
    scale = FIG_SCALE if scale is None else scale
    im = cv2.imread(os.path.join(IMG_SCENE, "Video", cam, f"{t:06d}.jpg"))[:, :, ::-1]
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else im
    _FSIZE[cam] = im.shape[:2]; return im


def _to_frame(m, cam):
    """masks (possibly at a lower run resolution) resized to the displayed frame size"""
    h, w = _FSIZE.get(cam) or frame(cam, 0).shape[:2]
    return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def pred(method, cam, t, obj, scale=None):
    p = os.path.join(SCENE, method, cam, str(t), f"{obj}.png")
    if not os.path.exists(p): return None
    return _to_frame(cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 0, cam).astype(bool)


def gt(cam, scale=None):
    return _to_frame(cv2.imread(os.path.join(SCENE, "Mask", cam, "000000.png"), cv2.IMREAD_GRAYSCALE), cam)


def overlay(img, method, cam, t, with_gt):
    out = img.astype(float)
    for o in range(1, 7):
        m = pred(method, cam, t, o)
        if m is not None and m.any(): out[m] = 0.45 * out[m] + 0.55 * COLS[o]
    out = out.astype(np.uint8)
    if with_gt:
        g = gt(cam)
        for o in range(1, 7):
            cs, _ = cv2.findContours((g == o).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, (255, 255, 255), 1)
    return out


def ring_cam(d, side=+1): return f"cam_{(REF + side * d) % NCAM:02d}"


def fig_ring(method, t):
    cams = [(f"cam_{REF:02d}", "reference (d = 0)")] + [(ring_cam(d), f"d = {d}") for d in (2, 4, 8, 12, 16)]
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 2.75))
    for ax, (cam, lab) in zip(axs.ravel(), cams):
        ax.imshow(overlay(frame(cam, t), method, cam, t, with_gt=(t == 0))); ax.set_title(f"{cam} ({lab})", fontsize=7, pad=2); ax.axis("off")
    plt.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01, wspace=0.02, hspace=0.12)
    name = f"fig_plenoptic_ring_{'txt' if method == TXT else 'gt'}"
    fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=600); fig.savefig(os.path.join(PREV, name + ".png"), dpi=110); plt.close(fig); print(name)


def bbox_union(cam, obj, a, b, margin=0.4):
    """crop box (x0, y0, x1, y1) at figure scale around every mask of `obj` in either configuration over frames a-1..b+1."""
    ys, xs = [], []
    for t in range(max(a - 1, 0), min(b + 1, int(os.environ.get("PLEN_LAST", 100))) + 1):
        for method in (CTRL, HEAD):
            m = pred(method, cam, t, obj)
            if m is not None and m.any(): yy, xx = np.nonzero(m); ys += [yy.min(), yy.max()]; xs += [xx.min(), xx.max()]
    if not ys: return None
    y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs); h, w = y1 - y0, x1 - x0
    H, W = frame(cam, 0).shape[:2]
    return (max(int(x0 - margin * w), 0), max(int(y0 - margin * h), 0), min(int(x1 + margin * w), W), min(int(y1 + margin * h), H))


def fig_event(cam, obj, a, b, neighbor, zoom=False, cols=5, ts=None):
    last = int(os.environ.get("PLEN_LAST", 100))
    if ts is None: ts = sorted(set([max(a - 1, 0), a, (a + b) // 2, b, min(b + 1, last)])) if cols == 5 else sorted(set([max(a - 1, 0), (a + b) // 2, min(b + 1, last)]))
    cols = len(ts)
    rows = [(CTRL, "single view (XW0)"), (HEAD, "cross-view memory (ours)")]
    if neighbor: rows.append((None, f"neighbour {neighbor}"))
    box = bbox_union(cam, obj, a, b) if zoom else None
    fw = 3.5 if cols == 3 else 7.16                              # 3 panels: column width; otherwise text width
    if box: asp = (box[3] - box[1]) / max(box[2] - box[0], 1)
    else: asp = 9 / 16
    ph = (fw / len(ts)) * asp                                    # panel height from the crop's aspect ratio
    fig, axs = plt.subplots(len(rows), len(ts), figsize=(fw, ph * len(rows) * 1.10 + 0.2))
    for r, (method, lab) in enumerate(rows):
        for c, t in enumerate(ts):
            ax = axs[r, c]
            if method is None: ax.imshow(overlay(frame(neighbor, t), HEAD, neighbor, t, False))
            else:
                img = frame(cam, t).astype(float); m = pred(method, cam, t, obj)
                if m is not None and m.any(): img[m] = 0.4 * img[m] + 0.6 * COLS[obj]
                img = img.astype(np.uint8)
                if box: img = img[box[1]:box[3], box[0]:box[2]]
                ax.imshow(img)
            ax.axis("off")
            if r == 0: ax.set_title(f"t = {t}", fontsize=7, pad=2)
            if c == 0: ax.text(0.02, 0.97, lab, transform=ax.transAxes, va="top", ha="left", fontsize=7, bbox=dict(facecolor="white", alpha=0.85, pad=1.5, edgecolor="none"))
    plt.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01, wspace=0.02, hspace=0.08)
    name = f"fig_plenoptic_event_{cam}_{obj}" + ("_zoom" if zoom else "")
    fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=600); fig.savefig(os.path.join(PREV, name + ".png"), dpi=110); plt.close(fig); print(name)


def fig_curve():
    S = json.load(open(os.path.join(PREV, "score.json")))
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for key, lab, st in ((CTRL, "ground-truth mask on the reference camera", "o-"), (TXT, "text prompt \"person\"", "s--"), ("SegMaskSam3XW0MFs_pt", "one click per person", "^:")):
        if key not in S: continue
        q = S[key]["Q2_by_distance"]; ds = sorted(int(d) for d in q)
        ax.plot(ds, [q[str(d)]["mean_J"] for d in ds], st, ms=3, lw=1, label=lab)
    ax.set_xlabel("ring distance from the reference camera"); ax.set_ylabel("mean $\\mathcal{J}$ at $t=0$"); ax.set_xticks(range(0, 17, 2))
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=6, frameon=False, loc="lower left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.22)
    fig.savefig(os.path.join(OUT, "fig_plenoptic_curve.pdf"), dpi=600); fig.savefig(os.path.join(PREV, "fig_plenoptic_curve.png"), dpi=110); plt.close(fig); print("fig_plenoptic_curve")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["ring", "event", "curve"])
    ap.add_argument("--method", default=CTRL); ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--cam"); ap.add_argument("--obj", type=int); ap.add_argument("--frames", type=int, nargs=2); ap.add_argument("--neighbor"); ap.add_argument("--zoom", action="store_true"); ap.add_argument("--cols", type=int, default=5); ap.add_argument("--ts", type=int, nargs="+")
    a = ap.parse_args()
    if a.what == "ring": fig_ring(a.method, a.t)
    elif a.what == "event": fig_event(a.cam, a.obj, a.frames[0], a.frames[1], a.neighbor, a.zoom, a.cols, a.ts)
    else: fig_curve()


if __name__ == "__main__":
    main()
