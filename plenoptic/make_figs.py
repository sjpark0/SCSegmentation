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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(REPO, "PlenopticServer1", "CROOM", "SegResult", "T06_flag_ganggang")
OUT = os.path.join(REPO, "paper", "figures"); PREV = os.path.join(REPO, "plenoptic")
CTRL, HEAD, TXT = "SegMaskSam3XW0MFs", "SegMaskSam3XW1CGPS4MFsCs10kG2", "SegMaskSam3XW0MFs_txt"
REF, NCAM = 6, 32
COLS = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180]], dtype=float)
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})


def frame(cam, t, scale=0.5):
    im = cv2.imread(os.path.join(SCENE, "Video", cam, f"{t:06d}.jpg"))[:, :, ::-1]
    return cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def pred(method, cam, t, obj, scale=0.5):
    p = os.path.join(SCENE, method, cam, str(t), f"{obj}.png")
    if not os.path.exists(p): return None
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 0
    return cv2.resize(m.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(bool)


def gt(cam, scale=0.5):
    g = cv2.imread(os.path.join(SCENE, "Mask", cam, "000000.png"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


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
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(PREV, name + ".png"), dpi=110); plt.close(fig); print(name)


def fig_event(cam, obj, a, b, neighbor):
    ts = sorted(set([max(a - 1, 0), a, (a + b) // 2, b, min(b + 1, 100)]))
    rows = [(CTRL, "single-view tracker (XW0)"), (HEAD, "with cross-view memory (ours)")]
    if neighbor: rows.append((None, f"neighbour {neighbor}"))
    fig, axs = plt.subplots(len(rows), len(ts), figsize=(7.16, 1.45 * len(rows)))
    for r, (method, lab) in enumerate(rows):
        for c, t in enumerate(ts):
            ax = axs[r, c]
            if method is None: ax.imshow(overlay(frame(neighbor, t), HEAD, neighbor, t, False))
            else:
                img = frame(cam, t).astype(float); m = pred(method, cam, t, obj)
                if m is not None and m.any(): img[m] = 0.4 * img[m] + 0.6 * COLS[obj]
                ax.imshow(img.astype(np.uint8))
            ax.axis("off")
            if r == 0: ax.set_title(f"t = {t}", fontsize=7, pad=2)
            if c == 0: ax.text(-0.02, 0.5, lab, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=6.5)
    plt.subplots_adjust(left=0.035, right=0.995, top=0.9, bottom=0.01, wspace=0.02, hspace=0.05)
    name = f"fig_plenoptic_event_{cam}_{obj}"
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(PREV, name + ".png"), dpi=110); plt.close(fig); print(name)


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
    fig.savefig(os.path.join(OUT, "fig_plenoptic_curve.pdf")); fig.savefig(os.path.join(PREV, "fig_plenoptic_curve.png"), dpi=110); plt.close(fig); print("fig_plenoptic_curve")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["ring", "event", "curve"])
    ap.add_argument("--method", default=CTRL); ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--cam"); ap.add_argument("--obj", type=int); ap.add_argument("--frames", type=int, nargs=2); ap.add_argument("--neighbor")
    a = ap.parse_args()
    if a.what == "ring": fig_ring(a.method, a.t)
    elif a.what == "event": fig_event(a.cam, a.obj, a.frames[0], a.frames[1], a.neighbor)
    else: fig_curve()


if __name__ == "__main__":
    main()
