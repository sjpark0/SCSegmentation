#!/usr/bin/env python3
"""S2-R1: track small objects inside a crop and paste the result back (docs/stage2-R1-prereg.md).

Stage 2 (temporal tracking) loses small objects to the 288 grid on every frame
(docs/stage1-E0.md sections 3 and 8).  This runner takes the (camera, object) pairs whose
frame-0 prompt is smaller than --max-area pixels, cuts a square window around the prompt
out of the 21 frames, runs the SAM 3 temporal tracker on that window alone -- so the object
fills 4..10x more of the 1008 input -- and pastes every mask back into a full-size canvas
that is 0 outside the window.  A copy of the --base folder then gets the frames
start+1 .. start+20 of exactly those pairs replaced (frame 0 stays the base's seed PNG),
so eval/eval_jf.py scores the treatment against the base pair for pair.

    python crop_track.py Breakfast --base SegMaskSam3XW0MFs --seeds MVSeed_e0_index \\
                         --max-area 2000 --suffix Cr2k [--min-side 256 --scale 4 --overwrite --dry-run]

Rules (prereg section 2, all decided from data available at inference time):
  cameras   the scene's cam_list (the three annotated cameras); c_ini is among them
  objects   docs/raw/muvod_object_sets.json[scene]["seed_ids_c_ini"] (the MUVOD basic set)
  prompt    c_ini: Data/MVSeg/<folder>/Mask/<cam>/<start_frame:06d>.png == obj (ground truth,
            what the mainline prompts); other cameras: <folder>/<seeds>/<cam>/<start_frame>/<obj>.png
            > 127 (the stage-1 seed the base run tracked from)
  target    0 < prompt area < max_area
  window    square of side s = clamp(ceil(scale * max(box w, box h)), min_side, min(H, W)),
            centred on the prompt's bounding box and pushed inside the image (no negative
            coordinates); the same window for all 21 frames
  tracking  one session per (camera, object): the 21 cropped frames through io_utils'
            AsyncVideoFrameCPUToGPU (same resize/normalise as the mainline), frame-0 mask
            prompt, forward propagation, fill_hole_area=0 (run_seed.py) -- no neighbour
            reading, i.e. the XW0 condition
  output    <folder>/<base><suffix>/ = copy of <base> with the target pairs' frames
            start+1..start+20 replaced; MANIFEST.json carries the usual provenance block plus
            "derived": {from, seeds, rule, targets [...]}, one entry per target with
            seed_px, box [x0, y0, x1, y1] (x1/y1 exclusive), crop [x0, y0], side,
            zoom = 1008 / side, frames_empty and frames_touching_border over the 20 written
            frames (touching = mask on a window edge that is not an image edge: the object
            is at the limit of the window), and the per-frame areas

The geometry, the target rule and the paste-back are plain functions at the top of the
file and import nothing heavy, so tests/test_crop_track.py runs them on the host; torch,
cv2 and the model package are imported only when tracking starts.  --dry-run prints the
targets and the windows and loads no model; without cv2 or numpy (the host) it reads the
prompt PNGs with the stdlib decoder below.

Run from SCSam3/ in the container, as the other runners:

    docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \\
      --user $(id -u):$(id -g) -v /:/host -w /host$PWD/SCSam3 \\
      -e HF_HOME=/host$PWD/SCSam3/hf_cache -e HF_HUB_OFFLINE=1 scsam3 \\
      python crop_track.py Breakfast --base SegMaskSam3XW0MFs --seeds MVSeed_e0_index \\
                           --max-area 2000 --suffix Cr2k
"""
import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import time
import zlib

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PKG = os.path.join(HERE, "demoSCSam3MVOpt")
CONFIG = os.path.join(HERE, "demo", "MVSeg.json")
DATA = os.path.join(REPO, "Data", "MVSeg")
OBJECT_SETS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
SEED_PREFIX = "MVSeed_"              # MVSeed/README.md R2
IMAGE_SIZE = 1008                    # model input side; zoom = IMAGE_SIZE / window side
THRESHOLD = 127                      # seed PNG value > 127 is foreground (seeds_from.THRESHOLD)
DEFAULT_MIN_SIDE = 256
DEFAULT_SCALE = 4.0
SUFFIX_RE = re.compile(r"[A-Za-z0-9]+")
PNG_SIG = b"\x89PNG\r\n\x1a\n"
PROMPT_GT, PROMPT_SEED = "gt", "seed"


# ------------------------------------------------------------------ config
def cam_name(c, prefix, prefix1):
    return f"{prefix}{c:0{prefix1}d}"


def load_config(path, name):
    """The MVSeg.json entry as runMVSeg.load_config reads it."""
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if name not in cfg:
        raise ValueError(f"{name!r} is not in {path}. Available: {', '.join(sorted(cfg))}")
    d = cfg[name]
    perms = d.get("perms")
    if perms is None:
        perms = list(range(d["start_cam"], d["num_cam"] + d["start_cam"]))
    return {"folder": d["folder"], "start_frame": d["start_frame"], "num_frame": d["num_frame"],
            "cam_list": d["cam_list"], "perms": perms, "prefix": d["prefix"],
            "prefix1": d["prefix1"], "c_ini": d.get("c_ini")}


def basic_objects(path, scene):
    """MUVOD basic object ids of the scene: seed_ids_c_ini of docs/raw/muvod_object_sets.json."""
    with open(path, encoding="utf-8") as fh:
        sets = json.load(fh)
    if scene not in sets:
        raise ValueError(f"{scene!r} is not in {path}")
    return [int(o) for o in sets[scene]["seed_ids_c_ini"]]


# ------------------------------------------------------------------ geometry (pure)
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crop_side(box_w, box_h, H, W, scale=DEFAULT_SCALE, min_side=DEFAULT_MIN_SIDE):
    """Window side: clamp(ceil(scale * max(w, h)), min_side, min(H, W)).  min_side is itself
    capped at min(H, W), so the window always fits the image."""
    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"box must be non-empty, got {box_w}x{box_h}")
    hi = min(H, W)
    lo = min(min_side, hi)
    want = int(math.ceil(scale * max(box_w, box_h)))
    return int(clamp(want, lo, hi))


def crop_window(box, H, W, scale=DEFAULT_SCALE, min_side=DEFAULT_MIN_SIDE):
    """box = (x0, y0, x1, y1), x1/y1 exclusive -> (cx0, cy0, side): the square window
    centred on the box, pushed inside the image so 0 <= cx0 <= W - side (same for y)."""
    x0, y0, x1, y1 = box
    s = crop_side(x1 - x0, y1 - y0, H, W, scale, min_side)
    cx0 = int(math.floor((x0 + x1) / 2.0 - s / 2.0 + 0.5))
    cy0 = int(math.floor((y0 + y1) / 2.0 - s / 2.0 + 0.5))
    return int(clamp(cx0, 0, W - s)), int(clamp(cy0, 0, H - s)), s


def is_target(area, max_area):
    return 0 < area < max_area


def prompt_source(cam, obj, c_ini_cam, ds_dir, seeds, start_frame):
    """(path, lo, hi, kind): the PNG holding the prompt of (cam, obj) and the value range
    that is foreground -- the ground truth (== obj) on c_ini, the stage-1 seed (> 127)
    elsewhere."""
    if cam == c_ini_cam:
        return (os.path.join(ds_dir, "Mask", cam, f"{start_frame:06d}.png"), obj, obj, PROMPT_GT)
    return (os.path.join(ds_dir, seeds, cam, f"{start_frame:d}", f"{obj:d}.png"),
            THRESHOLD + 1, 255, PROMPT_SEED)


# ------------------------------------------------------------------ images without numpy
class GrayRows:
    """An 8-bit grayscale image as one bytes object per row (the stdlib PNG reader's
    output when numpy is absent).  `shape` = (H, W) like a numpy array."""
    __slots__ = ("rows", "shape")

    def __init__(self, rows, width):
        self.rows = list(rows)
        self.shape = (len(self.rows), int(width))


def decode_gray_png(path):
    """Stdlib decoder for the mask PNGs (8-bit grayscale, non-interlaced): (rows, width)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != PNG_SIG:
        raise ValueError(f"{path}: not a PNG")
    pos, idat, hdr = 8, [], None
    while pos + 8 <= len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if typ == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif typ == b"IDAT":
            idat.append(body)
        elif typ == b"IEND":
            break
    if hdr is None:
        raise ValueError(f"{path}: no IHDR")
    W, H, depth, ctype, _, _, interlace = hdr
    if depth != 8 or ctype != 0 or interlace != 0:
        raise ValueError(f"{path}: the stdlib reader handles 8-bit non-interlaced grayscale "
                         f"only (depth {depth}, colour type {ctype}, interlace {interlace}); "
                         "run inside the container (cv2)")
    raw = zlib.decompress(b"".join(idat))
    if len(raw) != H * (W + 1):
        raise ValueError(f"{path}: {len(raw)} bytes of image data, expected {H * (W + 1)}")
    rows, prev = [], bytes(W)
    for y in range(H):
        off = y * (W + 1)
        f = raw[off]
        cur = bytearray(raw[off + 1:off + 1 + W])
        if f == 1:                                          # Sub
            for i in range(1, W):
                cur[i] = (cur[i] + cur[i - 1]) & 0xFF
        elif f == 2:                                        # Up
            cur = bytearray((a + b) & 0xFF for a, b in zip(cur, prev))
        elif f == 3:                                        # Average
            for i in range(W):
                left = cur[i - 1] if i else 0
                cur[i] = (cur[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif f == 4:                                        # Paeth
            for i in range(W):
                a = cur[i - 1] if i else 0
                b = prev[i]
                c = prev[i - 1] if i else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (cur[i] + pred) & 0xFF
        elif f != 0:
            raise ValueError(f"{path}: unknown PNG filter {f} on row {y}")
        row = bytes(cur)
        rows.append(row)
        prev = row
    return rows, W


def read_gray_png(path):
    """Grayscale PNG -> 2-D uint8 numpy array when cv2 or numpy is importable, else a
    GrayRows.  Lazy imports: the module itself needs neither."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"cannot read {path}")
        return img
    rows, width = decode_gray_png(path)
    try:
        import numpy as np
    except ImportError:
        return GrayRows(rows, width)
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), width)


def area_and_box(img, lo, hi):
    """(pixel count, (x0, y0, x1, y1)) of lo <= value <= hi; (0, None) when there is none.
    x1/y1 exclusive.  `img` is a 2-D array or a GrayRows."""
    if isinstance(img, GrayRows):
        table = bytes(1 if lo <= v <= hi else 0 for v in range(256))
        area, x0, y0, x1, y1 = 0, None, None, None, None
        for y, row in enumerate(img.rows):
            t = row.translate(table)
            n = t.count(b"\x01")
            if not n:
                continue
            area += n
            left, right = t.find(b"\x01"), t.rfind(b"\x01")
            if y0 is None:
                y0, x0, x1 = y, left, right
            else:
                x0, x1 = min(x0, left), max(x1, right)
            y1 = y
        return (0, None) if area == 0 else (area, (x0, y0, x1 + 1, y1 + 1))
    import numpy as np
    m = (img >= lo) & (img <= hi)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return 0, None
    return int(ys.size), (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def foreground(img, lo, hi):
    """bool array of lo <= value <= hi (numpy path only: the tracker needs an array)."""
    import numpy as np
    return (np.asarray(img) >= lo) & (np.asarray(img) <= hi)


# ------------------------------------------------------------------ targets (pure, reader injected)
def select_targets(cams, objs, c_ini_cam, ds_dir, seeds, start_frame, max_area,
                   scale=DEFAULT_SCALE, min_side=DEFAULT_MIN_SIDE, read=read_gray_png):
    """Every (cam, obj) with 0 < prompt area < max_area, with its window.

    Returns (targets, skipped, (H, W)).  A target: {cam, obj, prompt ("gt"/"seed"),
    seed_px, box, crop [x0, y0], side, zoom}.  skipped lists the other pairs with a reason
    (missing: no prompt PNG; empty: area 0; large: area >= max_area).  `read` maps a path
    to a 2-D array or GrayRows; a PNG is read once even when many objects share it (the
    ground truth of c_ini).  H, W come from the first PNG read; a PNG of another size is
    refused."""
    cache = {}

    def get(path):
        if path not in cache:
            cache[path] = read(path) if os.path.isfile(path) else None
        return cache[path]

    H = W = None
    targets, skipped = [], []
    for cam in cams:
        for obj in objs:
            path, lo, hi, kind = prompt_source(cam, obj, c_ini_cam, ds_dir, seeds, start_frame)
            img = get(path)
            if img is None:
                skipped.append({"cam": cam, "obj": obj, "reason": "missing", "seed_px": 0})
                continue
            h, w = int(img.shape[0]), int(img.shape[1])
            if H is None:
                H, W = h, w
            elif (h, w) != (H, W):
                raise ValueError(f"{path}: {w}x{h} but the scene is {W}x{H}")
            area, box = area_and_box(img, lo, hi)
            if not is_target(area, max_area):
                skipped.append({"cam": cam, "obj": obj, "seed_px": area,
                                "reason": "empty" if area == 0 else "large"})
                continue
            cx0, cy0, s = crop_window(box, H, W, scale, min_side)
            targets.append({"cam": cam, "obj": obj, "prompt": kind, "seed_px": area,
                            "box": list(box), "crop": [cx0, cy0], "side": s,
                            "zoom": round(IMAGE_SIZE / s, 4)})
    return targets, skipped, (H, W)


# ------------------------------------------------------------------ paste back (numpy, lazy)
def paste_back(crop_mask, cx0, cy0, H, W):
    """(side, side) mask -> (H, W) bool canvas, 0 outside the window."""
    import numpy as np
    m = np.asarray(crop_mask).astype(bool)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"crop mask must be square 2-D, got {m.shape}")
    s = m.shape[0]
    if cx0 < 0 or cy0 < 0 or cx0 + s > W or cy0 + s > H:
        raise ValueError(f"window ({cx0}, {cy0}) side {s} is outside {W}x{H}")
    canvas = np.zeros((H, W), dtype=bool)
    canvas[cy0:cy0 + s, cx0:cx0 + s] = m
    return canvas


def touches_window_edge(crop_mask, cx0, cy0, H, W):
    """True when the mask lies on a window edge that is not an image edge -- the object
    is at the limit of the window (prereg section 2: count these)."""
    import numpy as np
    m = np.asarray(crop_mask).astype(bool)
    s = m.shape[0]
    return bool((cy0 > 0 and m[0].any()) or (cy0 + s < H and m[-1].any())
                or (cx0 > 0 and m[:, 0].any()) or (cx0 + s < W and m[:, -1].any()))


# ------------------------------------------------------------------ manifest (pure)
def build_derived(base, seeds, max_area, min_side, scale, start_frame, num_frame,
                  targets, skipped, argv, seconds=None, model_seconds=None):
    """The "derived" block of the output MANIFEST.json (plain ints, floats, strs, lists)."""
    return {
        "from": base, "seeds": seeds,
        "generator": "SCSam3/crop_track.py", "prereg": "docs/stage2-R1-prereg.md",
        "rule": {"max_area": int(max_area), "min_side": int(min_side), "scale": float(scale),
                 "image_size": IMAGE_SIZE,
                 "target": "0 < seed_px < max_area",
                 "prompt": "c_ini: ground truth == obj; other cameras: "
                           "<seeds>/<cam>/<start_frame>/<obj>.png > 127",
                 "side": "clamp(ceil(scale * max(box w, box h)), min_side, min(H, W)), "
                         "square centred on the box, pushed inside the image",
                 "frames_replaced": [int(start_frame) + 1, int(start_frame) + int(num_frame) - 1],
                 "box": "[x0, y0, x1, y1], x1/y1 exclusive; crop = [x0, y0] of the window",
                 "frames_touching_border": "written frames whose mask lies on a window edge "
                                           "that is not an image edge",
                 "frames_no_base_png": "written frames for which the base folder had no PNG "
                                       "(its tracker dropped the object; scored as empty)"},
        "n_targets": len(targets), "targets": list(targets), "skipped": list(skipped),
        "argv": list(argv), "seconds": seconds, "model_load_seconds": model_seconds,
    }


def write_derived_manifest(out_dir, folder, method, derived, base_manifest=None):
    """<out_dir>/MANIFEST.json: eval/manifest.py's block (content digest, provenance) plus
    the "derived" block.  eval/ is imported by path for this one call (as runMVSeg does)."""
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, REPO)
    try:
        from eval.manifest import build_manifest, live_provenance, manifest_path
    finally:
        sys.path.remove(REPO)
        sys.dont_write_bytecode = dont_write
    prov = live_provenance(repo=REPO, package_dir=PKG, algo="CropTrack", argv=sys.argv,
                           track_cams="written",
                           extra={"out_name": method, "derived_from": derived["from"],
                                  "base_content_digest": (base_manifest or {}).get("content_digest"),
                                  "seeds_from": derived["seeds"]})
    m = build_manifest(out_dir, dataset=folder, method=method, provenance=prov, jobs=1)
    m["derived"] = derived
    path = manifest_path(out_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(m, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------ model (torch, lazy)
def load_package():
    """Put the MVOpt package first on sys.path (and SCSam3/ itself off it: it shadows the
    installed sam3, see runMVSeg.build_runner), then import what tracking needs."""
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != HERE]
    if PKG not in sys.path:
        sys.path.insert(0, PKG)
    import cv2
    import numpy as np
    import torch
    from build_scsam3 import build_scsam3_video_predictor
    from io_utils import AsyncVideoFrameCPUToGPU
    predictor = build_scsam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
    predictor.model.fill_hole_area = 0
    return cv2, np, torch, predictor, AsyncVideoFrameCPUToGPU


def to_bool_mask(m, np):
    m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
    return np.squeeze(m) > 0


def track_crop(predictor, torch, np, AsyncVideoFrameCPUToGPU, crops, prompt_crop, obj, num_frame):
    """One session over the cropped frames: frame-0 mask prompt, forward propagation.
    Returns {frame index: (side, side) bool} for every yielded frame; an object the
    tracker dropped (no output for it) gives an all-False mask."""
    s = int(prompt_crop.shape[0])
    images = AsyncVideoFrameCPUToGPU(crops, offload_video_to_cpu=True)
    sid = predictor.handle_request(request=dict(type="start_session", images=images,
                                                orig_height=s, orig_width=s))["session_id"]
    try:
        predictor.handle_request(request=dict(type="reset_session", session_id=sid))
        predictor.handle_request(request=dict(
            type="add_prompt", session_id=sid, frame_index=0,
            mask=torch.tensor(np.asarray(prompt_crop).astype("float32"), dtype=torch.float32),
            obj_id=obj))
        out = {}
        for r in predictor.handle_stream_request(request=dict(
                type="propagate_in_video", session_id=sid, propagation_direction="forward",
                start_frame_index=0, max_frame_num_to_track=num_frame)):
            o = r["outputs"]
            mask = None
            for i, oid in enumerate(o["out_obj_ids"].tolist()):
                if int(oid) == obj:
                    mask = to_bool_mask(o["out_binary_masks"][i], np)
            out[int(r["frame_index"])] = mask if mask is not None else np.zeros((s, s), dtype=bool)
        return out
    finally:
        predictor.handle_request(request=dict(type="close_session", session_id=sid))


def read_frames(cv2, video_dir, start_frame, num_frame):
    frames = []
    for k in range(num_frame):
        p = os.path.join(video_dir, f"{start_frame + k:06d}.jpg")
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        frames.append(img)
    return frames


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="dataset name as it appears in MVSeg.json")
    ap.add_argument("--base", required=True, help="mask folder to copy and patch (SegMaskSam3XW0MFs)")
    ap.add_argument("--seeds", required=True, help=f"{SEED_PREFIX}<tag> folder with the non-c_ini prompts")
    ap.add_argument("--max-area", type=int, required=True, help="target: 0 < prompt area < this")
    ap.add_argument("--suffix", required=True, help="output folder = <base><suffix> (Cr2k)")
    ap.add_argument("--min-side", type=int, default=DEFAULT_MIN_SIDE)
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--object-sets", default=OBJECT_SETS)
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="targets and windows only, no model")
    return ap.parse_args(argv)


def main(argv=None):
    t_start = time.time()
    args = parse_args(argv)
    if not args.seeds.startswith(SEED_PREFIX) or "/" in args.seeds:
        sys.exit(f"--seeds must be one {SEED_PREFIX}<tag> folder name, got {args.seeds!r}")
    if not SUFFIX_RE.fullmatch(args.suffix or ""):
        sys.exit(f"--suffix must be [A-Za-z0-9]+, got {args.suffix!r}")
    if args.max_area <= 0 or args.min_side <= 0 or args.scale <= 0:
        sys.exit("--max-area, --min-side and --scale must be positive")
    try:
        c = load_config(args.config, args.scene)
        objs = basic_objects(args.object_sets, args.scene)
    except ValueError as exc:
        sys.exit(str(exc))
    if c["c_ini"] is None:
        sys.exit(f"{args.scene}: no c_ini in {args.config}")
    if c["c_ini"] not in c["cam_list"]:
        sys.exit(f"{args.scene}: c_ini {c['c_ini']} is not in cam_list {c['cam_list']}")
    cams = [cam_name(x, c["prefix"], c["prefix1"]) for x in c["cam_list"]]
    c_ini_cam = cam_name(c["c_ini"], c["prefix"], c["prefix1"])
    ds_dir = os.path.join(os.path.abspath(args.data_root), c["folder"])
    base_dir = os.path.join(ds_dir, args.base)
    seed_dir = os.path.join(ds_dir, args.seeds)
    out_name = args.base + args.suffix
    out_dir = os.path.join(ds_dir, out_name)
    start, num_frame = c["start_frame"], c["num_frame"]
    for d in (base_dir, seed_dir):
        if not os.path.isdir(d):
            sys.exit(f"missing folder: {d}")
    missing = [cam for cam in cams if not os.path.isdir(os.path.join(base_dir, cam))]
    if missing:
        sys.exit(f"{base_dir} has no camera folder for {missing}")

    try:
        targets, skipped, (H, W) = select_targets(
            cams, objs, c_ini_cam, ds_dir, args.seeds, start, args.max_area,
            scale=args.scale, min_side=args.min_side)
    except ValueError as exc:
        sys.exit(str(exc))
    n_pairs = len(cams) * len(objs)
    reasons = {}
    for s in skipped:
        reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1

    print(f"scene          {args.scene}  ({ds_dir})")
    print(f"cameras        {' '.join(cams)}  (c_ini {c_ini_cam}: ground-truth prompt; "
          f"others: {args.seeds})")
    print(f"objects        {len(objs)} basic ids {objs}")
    print(f"frames         {start} .. {start + num_frame - 1}  ({num_frame}); "
          f"{start + 1}..{start + num_frame - 1} replaced, {start} kept from base")
    print(f"image          {W}x{H}" if H is not None else "image          (no prompt PNG read)")
    print(f"rule           0 < seed_px < {args.max_area}; side = clamp(ceil({args.scale:g} x "
          f"max(w, h)), {args.min_side}, {min(H, W) if H else '?'})")
    print(f"base           {base_dir}")
    print(f"output         {out_dir}")
    print(f"targets        {len(targets)} of {n_pairs} (cam, obj) pairs; skipped "
          + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())), flush=True)
    for t in targets:
        print(f"  {t['cam']:<12} obj {t['obj']:>3}  {t['prompt']:<4} {t['seed_px']:>6} px  "
              f"box {t['box']}  crop {t['crop']}  side {t['side']:>4}  zoom {t['zoom']:.2f}")
    if args.dry_run:
        print(f"dry run, no model loaded ({time.time() - t_start:.1f} s)")
        return

    if os.path.isdir(out_dir):
        if not args.overwrite:
            sys.exit(f"{out_dir} exists; pass --overwrite")
        shutil.rmtree(out_dir)
    shutil.copytree(base_dir, out_dir, ignore=shutil.ignore_patterns("MANIFEST.json"))
    base_manifest = None
    bm = os.path.join(base_dir, "MANIFEST.json")
    if os.path.isfile(bm):
        with open(bm) as fh:
            base_manifest = json.load(fh)
    print(f"copied         {base_dir} -> {out_dir}", flush=True)

    t_model = time.time()
    cv2, np, torch, predictor, AsyncVideoFrameCPUToGPU = load_package()
    model_seconds = time.time() - t_model
    print(f"model          loaded in {model_seconds:.1f} s "
          f"({torch.cuda.device_count()} GPU)", flush=True)

    n_written = 0
    frames = None
    frames_cam = None
    for t in targets:
        t0 = time.time()
        cam, obj = t["cam"], t["obj"]
        cx0, cy0, s = t["crop"][0], t["crop"][1], t["side"]
        if frames_cam != cam:
            frames = read_frames(cv2, os.path.join(ds_dir, "Video", cam), start, num_frame)
            frames_cam = cam
            if any(f.shape[:2] != (H, W) for f in frames):
                sys.exit(f"{cam}: video frames are not {W}x{H}")
        path, lo, hi, _ = prompt_source(cam, obj, c_ini_cam, ds_dir, args.seeds, start)
        prompt = foreground(read_gray_png(path), lo, hi)
        prompt_crop = prompt[cy0:cy0 + s, cx0:cx0 + s]
        crops = [np.ascontiguousarray(f[cy0:cy0 + s, cx0:cx0 + s]) for f in frames]
        masks = track_crop(predictor, torch, np, AsyncVideoFrameCPUToGPU, crops, prompt_crop,
                           obj, num_frame)
        areas, empty, touching = [], 0, 0
        for k in range(num_frame):
            m = masks.get(k)
            if m is None:
                m = np.zeros((s, s), dtype=bool)
            a = int(m.sum())
            areas.append(a)
            if k == 0:
                continue                   # frame 0 stays the base's seed PNG
            empty += int(a == 0)
            touching += int(touches_window_edge(m, cx0, cy0, H, W))
            canvas = paste_back(m, cx0, cy0, H, W)
            folder = os.path.join(out_dir, cam, f"{start + k:d}")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, f"{obj:d}.png"), canvas.astype(np.uint8) * 255)
            n_written += 1
        # the base has no PNG where its tracker dropped the object (eval reads that as
        # empty); the derived folder writes every target frame, so png_count grows by this
        no_base = sum(1 for k in range(1, num_frame)
                      if not os.path.isfile(os.path.join(base_dir, cam, f"{start + k:d}", f"{obj:d}.png")))
        t.update({"frames_empty": empty, "frames_touching_border": touching,
                  "frames_no_base_png": no_base, "frames_yielded": len(masks), "areas": areas,
                  "seconds": round(time.time() - t0, 2)})
        print(f"  {cam:<12} obj {obj:>3}  {t['seed_px']:>6} px  side {s:>4}  zoom {t['zoom']:.2f}  "
              f"empty {empty}/{num_frame - 1}  border {touching}/{num_frame - 1}  "
              f"area f1 {areas[1] if len(areas) > 1 else '-'}  {t['seconds']:.1f} s", flush=True)
    frames = None

    seconds = round(time.time() - t_start, 1)
    derived = build_derived(args.base, args.seeds, args.max_area, args.min_side, args.scale,
                            start, num_frame, targets, skipped, sys.argv,
                            seconds=seconds, model_seconds=round(model_seconds, 1))
    print(f"done -> {out_dir}  ({len(targets)} targets, {n_written} PNG replaced, "
          f"{sum(t['frames_empty'] for t in targets)} empty frames, "
          f"{sum(t['frames_touching_border'] for t in targets)} border frames, {seconds} s)",
          flush=True)
    try:
        path = write_derived_manifest(out_dir, c["folder"], out_name, derived, base_manifest)
        print(f"manifest -> {path}", flush=True)
    except Exception as exc:          # bookkeeping only: never let it fail the run
        print(f"manifest not written ({type(exc).__name__}: {exc})", flush=True)


if __name__ == "__main__":
    main()
