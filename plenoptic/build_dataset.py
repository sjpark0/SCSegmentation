#!/usr/bin/env python3
"""Build the plenoptic-rig scene T06_flag_ganggang in the MUVOD layout the runner and scorer expect.

    <root>/T06_flag_ganggang/Video/cam_XX/<t:06d>.jpg   32 cameras in RING order, frames 900 + 3t (t = 0..100)
    <root>/T06_flag_ganggang/Mask/cam_XX/000000.png     ground truth at t = 0 only, pixel value = person id 1..6
    <root>/T06_flag_ganggang/Mask/objects_labels.json
    <root>/T06_flag_ganggang/SOURCE.json                 provenance: source paths, frame list, camera mapping, digests

Sources: frames  PlenopticServer1/CROOM/JPG/T06_flag_ganggang/<jpg>/<f>.jpg (3840x2160, 59.94 fps)
         labels  Data/Set9/images/<NN>.json (labelme polygons in 3840x2160 coordinates, labels p1..p6),
                 drawn on source frame 900 (== Data/VideoSample2/<jpg>/0.jpg).
Camera order: the ring order of the SAM 2 demos (perms = [0..7, 16..31, 8..15] over JPG folders, i.e.
P01..P08, P21..P36, P09..P16), so that index neighbours are physical neighbours.
Runs inside the scsam3 container (numpy, PIL); copies ~7 GB, writes nothing under the repo.

    python plenoptic/build_dataset.py --masks           # ground truth + labels + QA figure + SOURCE.json
    python plenoptic/build_dataset.py --frames          # copy the 101 x 32 frames
"""
import argparse, hashlib, json, os, shutil, sys
import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_JPG = os.path.join(REPO, "PlenopticServer1", "CROOM", "JPG", "T06_flag_ganggang")
SRC_LAB = os.path.join(REPO, "Data", "Set9", "images")
ROOT = os.path.join(REPO, "PlenopticServer1", "CROOM", "SegResult")
SCENE = "T06_flag_ganggang"
FRAME0, STEP, NFRAMES = 900, 3, 101
PERMS = list(range(0, 8)) + list(range(16, 32)) + list(range(8, 16))       # ring order over JPG folders
PNUM = [i + 1 if i < 16 else i + 5 for i in range(32)]                       # JPG folder -> P number (01..16, 21..36)
PEOPLE = ["p1", "p2", "p3", "p4", "p5", "p6"]
SRC_W, SRC_H = 3840, 2160
SCALE = 0.5                                     # 2026-09-16: 4K x 101 frames x 32 cameras = 80 GB in host RAM -> SIGKILL; 1080p = 20 GB
W, H = int(SRC_W * SCALE), int(SRC_H * SCALE)


def cam_name(k): return f"cam_{k:02d}"


def rasterize(json_path):
    """labelme shape_to_mask equivalent: polygon outline+fill at full resolution, one plane per person."""
    d = json.load(open(json_path, encoding="utf-8"))
    assert d["imageWidth"] == SRC_W and d["imageHeight"] == SRC_H, json_path
    planes = {}
    for p in PEOPLE:
        m = Image.new("1", (W, H), 0); dr = ImageDraw.Draw(m)
        for s in d["shapes"]:
            if s["label"] != p: continue
            xy = [(pt[0] * SCALE, pt[1] * SCALE) for pt in s["points"]]      # polygon coordinates scaled, then rasterized
            if len(xy) >= 3: dr.polygon(xy, outline=1, fill=1)
        planes[p] = np.array(m, dtype=bool)
    return planes


def build_masks(scene_dir):
    mask_dir = os.path.join(scene_dir, "Mask"); os.makedirs(mask_dir, exist_ok=True)
    stats, digests = {}, {}
    for k, jpg in enumerate(PERMS):
        nn = f"{PNUM[jpg]:02d}"
        planes = rasterize(os.path.join(SRC_LAB, f"{nn}.json"))
        gt = np.zeros((H, W), dtype=np.uint8); overlap = 0
        for i, p in enumerate(PEOPLE, start=1):
            overlap += int((planes[p] & (gt > 0)).sum())
            gt[planes[p]] = i                                   # later ids overwrite earlier on overlap
        d = os.path.join(mask_dir, cam_name(k)); os.makedirs(d, exist_ok=True)
        out = os.path.join(d, f"{0:06d}.png"); Image.fromarray(gt, mode="L").save(out)
        digests[cam_name(k)] = hashlib.sha256(open(out, "rb").read()).hexdigest()
        stats[cam_name(k)] = dict(jpg_folder=jpg, P=nn, areas={p: int(planes[p].sum()) for p in PEOPLE}, overlap_px=overlap,
                                  n_present=sum(int(planes[p].sum() > 0) for p in PEOPLE))
        print(f"{cam_name(k)} <- JPG {jpg:2d} / P{nn}: present {stats[cam_name(k)]['n_present']}/6, overlap {overlap} px")
    json.dump({str(i): {"label": "person", "type": "dynamic"} for i in range(1, 7)},
              open(os.path.join(mask_dir, "objects_labels.json"), "w"), indent=1)
    return stats, digests


def qa_figure(scene_dir, cams=(0, 8, 16, 24)):
    """Four cameras with the six ground-truth persons colour-coded, to check identity consistency by eye."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180]], dtype=float)
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    for ax, k in zip(axs.ravel(), cams):
        img = np.array(Image.open(os.path.join(scene_dir, "Video", cam_name(k), "000000.jpg")).convert("RGB"), dtype=float)
        gt = np.array(Image.open(os.path.join(scene_dir, "Mask", cam_name(k), "000000.png")))
        over = img.copy(); m = gt > 0; over[m] = 0.45 * img[m] + 0.55 * cols[gt[m]]
        ax.imshow(over.astype(np.uint8)); ax.set_title(f"{cam_name(k)} (P{PNUM[PERMS[k]]:02d})"); ax.axis("off")
        for i in range(1, 7):
            ys, xs = np.nonzero(gt == i)
            if len(xs): ax.text(xs.mean(), ys.mean(), f"p{i}", color="white", fontsize=12, weight="bold", ha="center",
                                bbox=dict(facecolor=cols[i] / 255, alpha=0.8, pad=1))
    plt.tight_layout(); out = os.path.join(REPO, "plenoptic", "qa_gt.png"); fig.savefig(out, dpi=80); print("QA figure:", out)


def copy_frames(scene_dir):
    vdir = os.path.join(scene_dir, "Video"); n = 0
    for k, jpg in enumerate(PERMS):
        d = os.path.join(vdir, cam_name(k)); os.makedirs(d, exist_ok=True)
        for t in range(NFRAMES):
            src = os.path.join(SRC_JPG, str(jpg), f"{FRAME0 + STEP * t}.jpg"); dst = os.path.join(d, f"{t:06d}.jpg")
            im = Image.open(src); im.draft("RGB", (W, H))                      # fast JPEG downscale, then exact resize
            im.convert("RGB").resize((W, H), Image.LANCZOS).save(dst, quality=95); n += 1
        print(f"{cam_name(k)}: {NFRAMES} frames", flush=True)
    print("copied", n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--masks", action="store_true"); ap.add_argument("--frames", action="store_true")
    ap.add_argument("--nframes", type=int, default=NFRAMES); ap.add_argument("--suffix", default="")   # P3: 201 frames -> T06_flag_ganggang_L
    a = ap.parse_args(); globals()["NFRAMES"] = a.nframes; scene_dir = os.path.join(ROOT, SCENE + a.suffix); os.makedirs(scene_dir, exist_ok=True)
    if a.frames: copy_frames(scene_dir)
    if a.masks:
        stats, digests = build_masks(scene_dir); qa_figure(scene_dir)
        src = {"frames": SRC_JPG, "labels": SRC_LAB, "frame0": FRAME0, "step": STEP, "n_frames": NFRAMES,
               "frames_used": [FRAME0 + STEP * t for t in range(NFRAMES)], "ring_order_jpg_folders": PERMS,
               "resolution": [W, H], "source_resolution": [SRC_W, SRC_H], "scale": SCALE,
               "cameras": stats, "gt_sha256": digests, "note": "GT only at t=0 (source frame 900); labels p1..p6 -> ids 1..6"}
        json.dump(src, open(os.path.join(scene_dir, "SOURCE.json"), "w"), indent=1)
        # candidate reference camera by the MUVOD-style rule: most persons present, tie -> largest total area
        best = max(stats, key=lambda c: (stats[c]["n_present"], sum(stats[c]["areas"].values())))
        print("c_ini candidate (most present, then largest area):", best, stats[best]["n_present"], sum(stats[best]["areas"].values()))


if __name__ == "__main__":
    main()
