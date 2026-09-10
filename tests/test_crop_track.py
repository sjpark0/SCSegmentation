"""S2-R1: SCSam3/crop_track.py -- the crop rule, the target rule, the paste-back and the
derived-manifest schema (docs/stage2-R1-prereg.md section 2), torch-free.

crop_track.py imports torch/cv2/numpy only inside the functions that need them, so the
module loads on the host; the geometry and the target selection are exercised as plain
functions with an injected PNG reader, the stdlib PNG decoder against PNGs encoded here
with every filter type, and the numpy parts (paste-back, window edge) under importorskip.
"""
import json
import os
import struct
import zlib

import pytest

from conftest import REPO, _load

ct = _load("crop_track", os.path.join(REPO, "SCSam3", "crop_track.py"))

H, W = 1080, 1920


def test_module_loads_without_torch_cv2_numpy():
    for name in ("torch", "cv2", "np", "numpy"):
        assert not hasattr(ct, name)


# ------------------------------------------------------------------ crop rule
def test_side_scales_the_larger_box_side():
    assert ct.crop_side(80, 60, H, W, scale=4, min_side=256) == 320
    assert ct.crop_side(60, 80, H, W, scale=4, min_side=256) == 320
    assert ct.crop_side(70, 10, H, W, scale=4.0, min_side=256) == 280


def test_side_is_clamped_below_by_min_side_and_above_by_the_short_image_side():
    assert ct.crop_side(5, 30, H, W, scale=4, min_side=256) == 256        # 120 -> 256
    assert ct.crop_side(64, 64, H, W, scale=4, min_side=256) == 256       # exactly 256
    assert ct.crop_side(65, 65, H, W, scale=4, min_side=256) == 260
    assert ct.crop_side(300, 100, H, W, scale=4, min_side=256) == 1080    # 1200 -> min(H, W)
    assert ct.crop_side(10, 10, 100, 200, scale=4, min_side=256) == 100   # min_side > image
    assert ct.crop_side(3, 3, H, W, scale=2.5, min_side=1) == 8            # ceil(7.5)
    with pytest.raises(ValueError):
        ct.crop_side(0, 5, H, W)


def test_window_is_square_centred_and_inside_the_image():
    # a 40x20 box in the middle: side 256 centred on (1000, 550)
    x0, y0, s = ct.crop_window((980, 540, 1020, 560), H, W, scale=4, min_side=256)
    assert s == 256 and (x0, y0) == (1000 - 128, 550 - 128)
    # top-left corner: pushed to (0, 0), never negative
    x0, y0, s = ct.crop_window((0, 0, 10, 10), H, W)
    assert (x0, y0, s) == (0, 0, 256)
    # bottom-right corner: pushed so the window ends at the image edge
    x0, y0, s = ct.crop_window((1900, 1070, 1920, 1080), H, W)
    assert (x0, y0, s) == (W - 256, H - 256, 256)
    # a box wider than the window: side = min(H, W), x pushed inside, y = 0
    x0, y0, s = ct.crop_window((100, 100, 1500, 200), H, W)
    assert s == 1080 and 0 <= x0 <= W - s and y0 == 0
    for box in [(0, 0, 1, 1), (5, 1000, 400, 1079), (1919, 0, 1920, 1), (700, 300, 760, 340)]:
        x0, y0, s = ct.crop_window(box, H, W)
        assert x0 >= 0 and y0 >= 0 and x0 + s <= W and y0 + s <= H
        assert isinstance(x0, int) and isinstance(y0, int) and isinstance(s, int)


# ------------------------------------------------------------------ target rule
def test_is_target_is_strict_on_both_ends():
    assert not ct.is_target(0, 2000)
    assert ct.is_target(1, 2000)
    assert ct.is_target(1999, 2000)
    assert not ct.is_target(2000, 2000)
    assert not ct.is_target(50000, 2000)


def test_prompt_source_reads_ground_truth_on_c_ini_and_seeds_elsewhere():
    p, lo, hi, kind = ct.prompt_source("v7", 12, "v7", "/ds", "MVSeed_x", 0)
    assert p == os.path.join("/ds", "Mask", "v7", "000000.png") and (lo, hi) == (12, 12)
    assert kind == ct.PROMPT_GT
    p, lo, hi, kind = ct.prompt_source("v5", 12, "v7", "/ds", "MVSeed_x", 40)
    assert p == os.path.join("/ds", "MVSeed_x", "v5", "40", "12.png") and (lo, hi) == (128, 255)
    assert kind == ct.PROMPT_SEED


def _blob(img, x0, y0, x1, y1, value):
    for y in range(y0, y1):
        for x in range(x0, x1):
            img[y][x] = value


def _fake_scene(tmp_path):
    """A 100x120 scene: c_ini v1 ground truth with objects 1 (large), 2 (small), 3 (absent),
    4 (small); seeds for v0 (obj 1 small, obj 2 empty, obj 4 large) and v2 (obj 2 only)."""
    h, w = 100, 120
    ds = tmp_path / "Scene"
    gt = [[0] * w for _ in range(h)]
    _blob(gt, 10, 10, 60, 60, 1)          # 2500 px
    _blob(gt, 70, 20, 80, 30, 2)          # 100 px
    _blob(gt, 100, 90, 110, 95, 4)        # 50 px
    seeds = {("v0", 1): (5, 5, 15, 15, 255),            # 100 px
             ("v0", 2): None,                            # empty PNG
             ("v0", 4): (0, 0, 60, 60, 200),             # 3600 px, value 200 > 127
             ("v2", 2): (50, 50, 52, 52, 128)}           # 4 px, 128 > 127 counts
    images = {}
    gt_path = str(ds / "Mask" / "v1" / "000000.png")
    images[gt_path] = gt
    for (cam, obj), blob in seeds.items():
        img = [[0] * w for _ in range(h)]
        if blob is not None:
            _blob(img, *blob)
        images[str(ds / "MVSeed_t" / cam / "0" / f"{obj}.png")] = img
    for p in images:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b"png")
    return str(ds), images, (h, w)


def _reader(images):
    def read(path):
        return ct.GrayRows([bytes(row) for row in images[path]], len(images[path][0]))
    return read


def test_select_targets_applies_the_area_rule_per_prompt_kind(tmp_path):
    ds, images, (h, w) = _fake_scene(tmp_path)
    targets, skipped, shape = ct.select_targets(
        ["v0", "v1", "v2"], [1, 2, 3, 4], "v1", ds, "MVSeed_t", 0, max_area=2000,
        scale=4, min_side=16, read=_reader(images))
    assert shape == (h, w)
    got = {(t["cam"], t["obj"]): t for t in targets}
    assert sorted(got) == [("v0", 1), ("v1", 2), ("v1", 4), ("v2", 2)]
    assert got[("v0", 1)]["prompt"] == "seed" and got[("v0", 1)]["seed_px"] == 100
    assert got[("v0", 1)]["box"] == [5, 5, 15, 15]
    assert got[("v1", 2)]["prompt"] == "gt" and got[("v1", 2)]["seed_px"] == 100
    assert got[("v1", 2)]["box"] == [70, 20, 80, 30]
    assert got[("v1", 4)]["seed_px"] == 50 and got[("v1", 4)]["box"] == [100, 90, 110, 95]
    assert got[("v2", 2)]["seed_px"] == 4
    for t in targets:                                     # window inside the image, zoom = 1008/side
        x0, y0 = t["crop"]
        assert 0 <= x0 <= w - t["side"] and 0 <= y0 <= h - t["side"]
        assert t["zoom"] == round(ct.IMAGE_SIZE / t["side"], 4)
    assert got[("v1", 2)]["side"] == 40 and got[("v1", 4)]["side"] == 40      # 4 x 10
    assert got[("v2", 2)]["side"] == 16                                       # clamp to min_side
    why = {(s["cam"], s["obj"]): s["reason"] for s in skipped}
    assert why == {("v0", 2): "empty", ("v0", 3): "missing", ("v0", 4): "large",
                   ("v1", 1): "large", ("v1", 3): "empty",
                   ("v2", 1): "missing", ("v2", 3): "missing", ("v2", 4): "missing"}
    assert {s["seed_px"] for s in skipped if s["reason"] == "large"} == {3600, 2500}


def test_select_targets_reads_each_png_once_and_refuses_a_foreign_size(tmp_path):
    ds, images, _ = _fake_scene(tmp_path)
    calls = []
    base = _reader(images)

    def read(path):
        calls.append(path)
        return base(path)
    ct.select_targets(["v1"], [1, 2, 3, 4], "v1", ds, "MVSeed_t", 0, 2000, read=read)
    assert len(calls) == 1                        # the ground truth serves every object
    small = [[0] * 10 for _ in range(10)]
    images[str(os.path.join(ds, "MVSeed_t", "v0", "0", "1.png"))] = small
    with pytest.raises(ValueError):
        ct.select_targets(["v1", "v0"], [1], "v1", ds, "MVSeed_t", 0, 2000, read=_reader(images))


# ------------------------------------------------------------------ PNG reading
def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)


def _encode_png(rows, filters):
    """8-bit grayscale PNG with the given per-row filter types applied."""
    w = len(rows[0])
    out = bytearray()
    prev = bytes(w)
    for row, f in zip(rows, filters):
        cur = bytes(row)
        if f == 0:
            filt = cur
        elif f == 1:
            filt = bytes((cur[i] - (cur[i - 1] if i else 0)) & 0xFF for i in range(w))
        elif f == 2:
            filt = bytes((cur[i] - prev[i]) & 0xFF for i in range(w))
        elif f == 3:
            filt = bytes((cur[i] - (((cur[i - 1] if i else 0) + prev[i]) >> 1)) & 0xFF
                         for i in range(w))
        else:
            filt = bytes((cur[i] - _paeth(cur[i - 1] if i else 0, prev[i],
                                          prev[i - 1] if i else 0)) & 0xFF for i in range(w))
        out += bytes([f]) + filt
        prev = cur

    def chunk(typ, body):
        return (struct.pack(">I", len(body)) + typ + body
                + struct.pack(">I", zlib.crc32(typ + body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, len(rows), 8, 0, 0, 0, 0)
    return (ct.PNG_SIG + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(out)))
            + chunk(b"IEND", b""))


def test_stdlib_png_decoder_undoes_every_filter(tmp_path):
    import random
    rng = random.Random(3)
    h, w = 7, 11
    rows = [bytes(rng.choice([0, 7, 128, 200, 255]) for _ in range(w)) for _ in range(h)]
    filters = [0, 1, 2, 3, 4, 4, 2]
    p = tmp_path / "m.png"
    p.write_bytes(_encode_png(rows, filters))
    got, width = ct.decode_gray_png(str(p))
    assert width == w and got == rows
    img = ct.read_gray_png(str(p))                        # cv2 / numpy / GrayRows, whichever
    assert tuple(img.shape) == (h, w)
    area, box = ct.area_and_box(ct.GrayRows(rows, w), 128, 255)
    assert area == sum(1 for r in rows for v in r if v >= 128)
    ys = [y for y, r in enumerate(rows) if any(v >= 128 for v in r)]
    xs = [x for r in rows for x, v in enumerate(r) if v >= 128]
    assert box == (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
    assert ct.area_and_box(ct.GrayRows(rows, w), 1, 1) == (0, None)


def test_stdlib_png_decoder_refuses_what_it_cannot_read(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"not a png")
    with pytest.raises(ValueError):
        ct.decode_gray_png(str(p))
    rgb = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0)
    body = (ct.PNG_SIG + struct.pack(">I", len(rgb)) + b"IHDR" + rgb + b"\0\0\0\0"
            + struct.pack(">I", 0) + b"IEND" + b"\0\0\0\0")
    p.write_bytes(body)
    with pytest.raises(ValueError, match="colour type 2"):
        ct.decode_gray_png(str(p))


def test_area_and_box_agree_between_rows_and_numpy():
    np = pytest.importorskip("numpy")
    rng = np.random.RandomState(5)
    img = (rng.rand(30, 40) * 255).astype(np.uint8)
    rows = ct.GrayRows([img[y].tobytes() for y in range(30)], 40)
    for lo, hi in [(128, 255), (7, 7), (0, 255), (250, 255)]:
        assert ct.area_and_box(img, lo, hi) == ct.area_and_box(rows, lo, hi)
    assert ct.area_and_box(np.zeros((4, 4), np.uint8), 1, 255) == (0, None)


# ------------------------------------------------------------------ paste back
def test_paste_back_fills_only_the_window():
    np = pytest.importorskip("numpy")
    crop = np.zeros((4, 4), dtype=bool)
    crop[1:3, 2:4] = True
    canvas = ct.paste_back(crop, 10, 20, 30, 50)
    assert canvas.shape == (30, 50) and canvas.dtype == bool
    assert canvas.sum() == 4
    assert canvas[21:23, 12:14].all()
    outside = canvas.copy()
    outside[20:24, 10:14] = False
    assert not outside.any()
    assert ct.paste_back(np.zeros((3, 3), bool), 47, 27, 30, 50).sum() == 0     # touches the corner
    with pytest.raises(ValueError):
        ct.paste_back(crop, 47, 27, 30, 50)                                     # overhangs
    with pytest.raises(ValueError):
        ct.paste_back(crop, -1, 0, 30, 50)
    with pytest.raises(ValueError):
        ct.paste_back(np.zeros((3, 4), bool), 0, 0, 30, 50)                     # not square


def test_touches_window_edge_ignores_image_edges():
    np = pytest.importorskip("numpy")
    m = np.zeros((5, 5), bool)
    assert not ct.touches_window_edge(m, 10, 10, 100, 100)
    m[0, 2] = True                                    # top row
    assert ct.touches_window_edge(m, 10, 10, 100, 100)
    assert not ct.touches_window_edge(m, 10, 0, 100, 100)   # window at the image top
    m[:] = False
    m[2, 4] = True                                    # right column
    assert ct.touches_window_edge(m, 10, 10, 100, 100)
    assert not ct.touches_window_edge(m, 95, 10, 100, 100)  # window at the image right
    m[:] = False
    m[2, 2] = True                                    # interior
    assert not ct.touches_window_edge(m, 10, 10, 100, 100)


# ------------------------------------------------------------------ manifest
def test_derived_manifest_schema():
    targets = [{"cam": "v5", "obj": 12, "prompt": "seed", "seed_px": 1623,
                "box": [771, 720, 826, 772], "crop": [671, 618], "side": 256, "zoom": 3.9375,
                "frames_empty": 0, "frames_touching_border": 2, "frames_no_base_png": 1,
                "areas": [1600] * 21}]
    skipped = [{"cam": "v5", "obj": 1, "reason": "large", "seed_px": 50000}]
    d = ct.build_derived("SegMaskSam3XW0MFs", "MVSeed_e0_index", 2000, 256, 4.0, 0, 21,
                         targets, skipped, ["crop_track.py", "Breakfast"], seconds=12.5,
                         model_seconds=30.0)
    assert d["from"] == "SegMaskSam3XW0MFs" and d["seeds"] == "MVSeed_e0_index"
    assert d["rule"]["max_area"] == 2000 and d["rule"]["min_side"] == 256
    assert d["rule"]["scale"] == 4.0 and d["rule"]["image_size"] == 1008
    assert d["rule"]["frames_replaced"] == [1, 20]
    assert d["n_targets"] == 1 and d["targets"] == targets and d["skipped"] == skipped
    t = d["targets"][0]
    for key in ("cam", "obj", "seed_px", "box", "side", "zoom", "frames_empty",
                "frames_touching_border", "frames_no_base_png"):
        assert key in t
    assert t["zoom"] == 1008 / t["side"]
    assert d["argv"] == ["crop_track.py", "Breakfast"] and d["seconds"] == 12.5
    json.dumps(d)                                     # plain JSON


# ------------------------------------------------------------------ config / CLI
def test_config_and_object_sets_of_the_real_scene():
    c = ct.load_config(ct.CONFIG, "Breakfast")
    assert c["folder"] == "Breakfast" and c["c_ini"] == 7 and c["cam_list"] == [5, 7, 9]
    assert c["num_frame"] == 21 and c["start_frame"] == 0
    objs = ct.basic_objects(ct.OBJECT_SETS, "Breakfast")
    assert len(objs) == 27 and 15 not in objs and 30 in objs
    with pytest.raises(ValueError):
        ct.load_config(ct.CONFIG, "NoSuchScene")
    with pytest.raises(ValueError):
        ct.basic_objects(ct.OBJECT_SETS, "NoSuchScene")


@pytest.mark.parametrize("argv", [
    ["Breakfast", "--base", "B", "--seeds", "SegMaskSam3XW0", "--max-area", "5", "--suffix", "Cr"],
    ["Breakfast", "--base", "B", "--seeds", "MVSeed_x", "--max-area", "5", "--suffix", "Cr 2k"],
    ["Breakfast", "--base", "B", "--seeds", "MVSeed_x", "--max-area", "0", "--suffix", "Cr"],
    ["Breakfast", "--base", "B", "--seeds", "MVSeed_x", "--max-area", "5", "--suffix", "Cr",
     "--scale", "0"],
])
def test_cli_refusals(argv):
    with pytest.raises(SystemExit):
        ct.main(argv + ["--dry-run"])


def test_cli_dry_run_on_a_fake_scene(tmp_path, capsys):
    """End to end without a model: config + object set + PNGs on disk -> the target list.
    The PNGs are written by the encoder above, so the stdlib decoder path is what runs
    when cv2 is absent."""
    h, w = 100, 120
    root = tmp_path / "MVSeg"
    ds = root / "Scene"
    cfg = {"Scene": {"start_frame": 0, "cam_list": [0, 1, 2], "c_ini": 1, "num_frame": 21,
                     "folder": "Scene", "num_cam": 3, "start_cam": 0, "prefix": "v",
                     "prefix1": 0}}
    sets = {"Scene": {"seed_ids_c_ini": [1, 2]}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    (tmp_path / "sets.json").write_text(json.dumps(sets))
    gt = [[0] * w for _ in range(h)]
    _blob(gt, 10, 10, 60, 60, 1)
    _blob(gt, 70, 20, 80, 30, 2)
    (ds / "Mask" / "v1").mkdir(parents=True)
    (ds / "Mask" / "v1" / "000000.png").write_bytes(_encode_png([bytes(r) for r in gt], [2] * h))
    for cam in ("v0", "v2"):
        d = ds / "MVSeed_t" / cam / "0"
        d.mkdir(parents=True)
        img = [[0] * w for _ in range(h)]
        _blob(img, 5, 5, 15, 15, 255)
        d.joinpath("2.png").write_bytes(_encode_png([bytes(r) for r in img], [1] * h))
    for cam in ("v0", "v1", "v2"):
        (ds / "Base" / cam).mkdir(parents=True)
    ct.main(["Scene", "--base", "Base", "--seeds", "MVSeed_t", "--max-area", "2000",
             "--suffix", "Cr2k", "--min-side", "16", "--config", str(tmp_path / "cfg.json"),
             "--object-sets", str(tmp_path / "sets.json"), "--data-root", str(root), "--dry-run"])
    out = capsys.readouterr().out
    assert "targets        3 of 6" in out
    assert "dry run" in out
    assert not (ds / "BaseCr2k").exists()
