"""S2-R2: the pure parts of runMVSeg --crop-small (docs/stage2-R2-prereg.md section 1) --
the gate verdict, the IoU, target selection from injected prompts, the crop session's
request shape and frame bookkeeping, and the paste-back golden -- torch-free.

The rules live in SCSam3/crop_track.py (R1) and are reused by runMVSeg.CropSmall; the
gate is crop_guard.py's (R1'/R1") as crop_track.gate_verdict.  The model-bound method
itself (CropSmall) is exercised by the GPU smoke, not here.
"""
import pytest

# one module instance: test_crop_track's GrayRows must be the class area_and_box tests for
from test_crop_track import _fake_scene, _reader, ct


# ------------------------------------------------------------------ gate (pure)
def test_gate_constants_are_the_registered_rule():
    assert ct.GATES == ("none", "v1", "v2")
    assert ct.GATE_TAU == 0.05 and ct.GATE_ZOOM_MIN == 1.2 and ct.GATE_BORDER_MAX == 10


def test_gate_verdict_table():
    # none keeps everything, whatever the signals say
    assert ct.gate_verdict("none", 0.0, 0.5, 20) == (True, None)
    # v1: the IoU rule only
    assert ct.gate_verdict("v1", 0.0, 0.5, 20) == (False, "iou")
    assert ct.gate_verdict("v1", 0.049, 3.0, 0) == (False, "iou")
    assert ct.gate_verdict("v1", 0.05, 0.5, 20) == (True, None)          # boundary: kept
    assert ct.gate_verdict("v1", 0.9, 0.9, 19) == (True, None)
    # v2: iou, then zoom, then border (crop_guard's order of `why`)
    assert ct.gate_verdict("v2", 0.0, 0.5, 20) == (False, "iou")
    assert ct.gate_verdict("v2", 0.5, 1.1999, 0) == (False, "zoom")
    assert ct.gate_verdict("v2", 0.5, 1.2, 0) == (True, None)             # boundary: kept
    assert ct.gate_verdict("v2", 0.5, 3.9375, 10) == (False, "border")   # boundary: reverted
    assert ct.gate_verdict("v2", 0.5, 3.9375, 9) == (True, None)
    assert ct.gate_verdict("v2", 0.5, 0.9, 12) == (False, "zoom")         # zoom fires first
    # the thresholds are parameters
    assert ct.gate_verdict("v2", 0.5, 1.2, 0, zoom_min=1.5) == (False, "zoom")
    assert ct.gate_verdict("v1", 0.5, 1.2, 0, tau=0.6) == (False, "iou")
    assert ct.gate_verdict("v2", 0.5, 3.0, 5, border_max=5) == (False, "border")
    for bad in ("G2", "V2", "", None, 2):
        with pytest.raises(ValueError):
            ct.gate_verdict(bad, 0.5, 2.0, 0)


# What crop_guard.py decided for the six Cr10k targets of Fencing (R1", the assembled
# folder SegMaskSam3XW0MFsCr10kG2/MANIFEST.json "guard"): the integrated gate must agree
# pair for pair, since S2-R2 prediction b says the reverted set is the same.
FENCING_CR10K_GUARD = [
    # cam, obj, iou_f1_seed, zoom, border, why (None = kept)
    ("v0", 3, 0.021197252208047104, 0.9655, 0, "iou"),
    ("v0", 4, None, 1.5849, 0, None),                       # the kept pair (iou not recorded)
    ("v4", 3, 0.06963845521774856, 0.9333, 0, "zoom"),
    ("v4", 4, 0.29118773946360155, 0.9333, 0, "zoom"),
    ("v9", 3, 0.2381427858212681, 1.0161, 0, "zoom"),
    ("v9", 4, 0.5294348508634223, 0.9333, 0, "zoom"),
]


def test_gate_verdict_replays_the_fencing_guard_records():
    for cam, obj, iou, zoom, border, why in FENCING_CR10K_GUARD:
        kept, got = ct.gate_verdict("v2", 1.0 if iou is None else iou, zoom, border)
        assert (kept, got) == (why is None, why), (cam, obj)
    # under v1 only the IoU pair goes back
    v1 = {(cam, obj): ct.gate_verdict("v1", 1.0 if iou is None else iou, zoom, border)[1]
          for cam, obj, iou, zoom, border, _ in FENCING_CR10K_GUARD}
    assert v1 == {("v0", 3): "iou", ("v0", 4): None, ("v4", 3): None, ("v4", 4): None,
                  ("v9", 3): None, ("v9", 4): None}


def test_mask_iou():
    np = pytest.importorskip("numpy")
    a = np.zeros((6, 8), bool)
    b = np.zeros((6, 8), bool)
    assert ct.mask_iou(a, b) == 1.0                       # two empty masks: nothing to disagree on
    a[1:3, 1:3] = True
    assert ct.mask_iou(a, b) == 0.0 and ct.mask_iou(b, a) == 0.0
    assert ct.mask_iou(a, a) == 1.0
    b[2:4, 2:4] = True                                    # overlap 1 px, union 7 px
    assert ct.mask_iou(a, b) == pytest.approx(1 / 7)
    assert ct.mask_iou(a.astype(np.uint8), b.astype(np.uint8) * 255) == pytest.approx(1 / 7)


# ------------------------------------------------------------------ targets from prompts
def _rows(img):
    return ct.GrayRows([bytes(row) for row in img], len(img[0]))


def test_select_targets_from_applies_the_area_rule_per_prompt():
    h, w = 40, 50
    gt = [[0] * w for _ in range(h)]
    for y in range(5, 30):                                # obj 1: 25x25 = 625 (large)
        for x in range(5, 30):
            gt[y][x] = 1
    for y in range(32, 36):                               # obj 2: 4x6 = 24 (small)
        for x in range(40, 46):
            gt[y][x] = 2
    seed_small = [[0] * w for _ in range(h)]
    seed_small[10][10] = 1                                # 1 px, value 1 with lo = hi = 1
    seed_empty = [[0] * w for _ in range(h)]
    prompts = {("c1", 1): (_rows(gt), 1, 1, ct.PROMPT_GT),
               ("c1", 2): (_rows(gt), 2, 2, ct.PROMPT_GT),
               ("c1", 3): (_rows(gt), 3, 3, ct.PROMPT_GT),          # absent from the gt: empty
               ("c0", 1): (_rows(seed_small), 1, 1, ct.PROMPT_SEED),
               ("c0", 2): (_rows(seed_empty), 1, 1, ct.PROMPT_SEED),
               ("c0", 3): None}                                     # no seed for the pair
    calls = []

    def prompt(cam, obj):
        calls.append((cam, obj))
        return prompts[(cam, obj)]

    targets, skipped, shape = ct.select_targets_from(["c0", "c1"], [1, 2, 3], prompt, 100,
                                                     scale=4, min_side=8)
    assert shape == (h, w)
    assert calls == [("c0", 1), ("c0", 2), ("c0", 3), ("c1", 1), ("c1", 2), ("c1", 3)]
    got = {(t["cam"], t["obj"]): t for t in targets}
    assert sorted(got) == [("c0", 1), ("c1", 2)]
    assert got[("c0", 1)] == {"cam": "c0", "obj": 1, "prompt": "seed", "seed_px": 1,
                              "box": [10, 10, 11, 11], "crop": [7, 7], "side": 8,
                              "zoom": round(1008 / 8, 4)}
    t = got[("c1", 2)]
    assert t["prompt"] == "gt" and t["seed_px"] == 24 and t["box"] == [40, 32, 46, 36]
    assert t["side"] == 24 and t["crop"] == [w - 24, h - 24] and t["zoom"] == 42.0
    why = {(s["cam"], s["obj"]): (s["reason"], s["seed_px"]) for s in skipped}
    assert why == {("c0", 2): ("empty", 0), ("c0", 3): ("missing", 0),
                   ("c1", 1): ("large", 625), ("c1", 3): ("empty", 0)}
    # the rule is strict on both ends
    targets, _, _ = ct.select_targets_from(["c1"], [1], prompt, 625, scale=4, min_side=8)
    assert targets == []
    targets, _, _ = ct.select_targets_from(["c1"], [1], prompt, 626, scale=4, min_side=8)
    assert [(t["cam"], t["obj"]) for t in targets] == [("c1", 1)]
    # a prompt of another size is refused
    prompts[("c0", 3)] = (_rows([[0] * 10 for _ in range(10)]), 1, 1, ct.PROMPT_SEED)
    with pytest.raises(ValueError):
        ct.select_targets_from(["c0"], [1, 3], prompt, 100, scale=4, min_side=8)


def test_select_targets_is_select_targets_from_over_prompt_source(tmp_path):
    """The file-reading entry point of crop_track.py (R1) is the injected one: same list."""
    ds, images, _ = _fake_scene(tmp_path)
    read = _reader(images)
    via_files = ct.select_targets(["v0", "v1", "v2"], [1, 2, 3, 4], "v1", ds, "MVSeed_t", 0,
                                  max_area=2000, scale=4, min_side=16, read=read)

    def prompt(cam, obj):
        path, lo, hi, kind = ct.prompt_source(cam, obj, "v1", ds, "MVSeed_t", 0)
        return (read(path), lo, hi, kind) if path in images else None
    via_prompts = ct.select_targets_from(["v0", "v1", "v2"], [1, 2, 3, 4], prompt, 2000,
                                         scale=4, min_side=16)
    assert via_prompts == via_files


def test_select_targets_from_numpy_bool_seed_as_uint8():
    """runMVSeg.CropSmall hands masks_spatial seeds in as uint8 with lo = hi = 1."""
    np = pytest.importorskip("numpy")
    seed = np.zeros((30, 40), bool)
    seed[3:7, 10:15] = True
    targets, skipped, shape = ct.select_targets_from(
        ["c"], [7], lambda cam, obj: (seed.astype(np.uint8), 1, 1, ct.PROMPT_SEED), 100,
        scale=4, min_side=8)
    assert shape == (30, 40) and skipped == []
    assert targets == [{"cam": "c", "obj": 7, "prompt": "seed", "seed_px": 20,
                        "box": [10, 3, 15, 7], "crop": [3, 0], "side": 20, "zoom": 50.4}]


# ------------------------------------------------------------------ crop session (fakes)
class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def tensor(x, dtype=None):
        return ("tensor", x.shape, dtype)


class _FakeFrames:
    def __init__(self, origin, offload_video_to_cpu=False):
        self.origin = list(origin)
        self.offload = offload_video_to_cpu


class _FakePredictor:
    """Replies like the predictors: a session id, then one output per frame with
    `yield_objs` present.  Records every request."""
    def __init__(self, np, side, num_frame, yield_objs, drop_frames=()):
        self.np, self.side, self.num_frame = np, side, num_frame
        self.yield_objs, self.drop_frames = yield_objs, set(drop_frames)
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            assert isinstance(request["images"], _FakeFrames)
            assert request["orig_height"] == request["orig_width"] == self.side
            return {"session_id": "sid-1"}
        return {}

    def handle_stream_request(self, request):
        self.requests.append(request)
        np = self.np
        for k in range(request["max_frame_num_to_track"]):
            objs = [o for o in self.yield_objs if k not in self.drop_frames]
            masks = []
            for o in objs:
                m = np.zeros((1, self.side, self.side), np.float32)
                m[0, k % self.side, o % self.side] = 1.0        # one pixel per (frame, obj)
                masks.append(m)
            yield {"frame_index": k,
                   "outputs": {"out_obj_ids": np.array(objs), "out_binary_masks": masks}}


@pytest.mark.parametrize("multi", [False, True])
def test_track_crop_request_shape_and_bookkeeping(multi):
    np = pytest.importorskip("numpy")
    side, num_frame, obj = 8, 5, 3
    pred = _FakePredictor(np, side, num_frame, yield_objs=[obj, 9], drop_frames=[2])
    crops = [np.zeros((side, side, 3), np.uint8) for _ in range(num_frame)]
    prompt = np.zeros((side, side), bool)
    prompt[2:4, 2:4] = True
    out = ct.track_crop(pred, _FakeTorch, np, _FakeFrames, crops, prompt, obj, num_frame,
                        multi_session=multi)
    types = [r["type"] for r in pred.requests]
    assert types == ["start_session", "reset_session", "add_prompt", "propagate_in_video",
                     "close_session"]
    start, reset, add, prop, close = pred.requests
    assert start["images"].offload is True and len(start["images"].origin) == num_frame
    assert reset["session_id"] == add["session_id"] == close["session_id"] == "sid-1"
    assert add["frame_index"] == 0 and add["obj_id"] == obj
    assert add["mask"] == ("tensor", (side, side), "float32")
    assert (prop["propagation_direction"], prop["start_frame_index"],
            prop["max_frame_num_to_track"]) == ("forward", 0, num_frame)
    if multi:                                   # the NewMem (MVOpt) predictor's shape
        assert prop["session_ids"] == ["sid-1"] and prop["spatial_idx"] == 0
        assert "session_id" not in prop
    else:                                       # crop_track.py's own single-session predictor
        assert prop["session_id"] == "sid-1" and "session_ids" not in prop
    assert sorted(out) == list(range(num_frame))
    for k in range(num_frame):
        assert out[k].shape == (side, side) and out[k].dtype == bool
        if k == 2:
            assert not out[k].any()             # dropped by the tracker: all-False
        else:
            assert out[k].sum() == 1 and out[k][k % side, obj % side]


def test_track_crop_closes_the_session_when_propagation_fails():
    np = pytest.importorskip("numpy")

    class Broken(_FakePredictor):
        def handle_stream_request(self, request):
            self.requests.append(request)
            raise RuntimeError("boom")
            yield

    pred = Broken(np, 4, 3, yield_objs=[1])
    with pytest.raises(RuntimeError):
        ct.track_crop(pred, _FakeTorch, np, _FakeFrames, [np.zeros((4, 4, 3), np.uint8)] * 3,
                      np.ones((4, 4), bool), 1, 3, multi_session=True)
    assert pred.requests[-1]["type"] == "close_session"


# ------------------------------------------------------------------ paste back golden
def test_paste_back_golden_and_border_count():
    """The written canvas of one crop frame: the crop mask at its window, 0 elsewhere,
    then the window-edge test that feeds the v2 border rule."""
    np = pytest.importorskip("numpy")
    H, W, s = 12, 16, 6
    crop = np.zeros((s, s), bool)
    crop[1:4, 2:5] = True
    crop[5, 0] = True                                     # bottom-left corner of the window
    cx0, cy0 = 7, 3
    canvas = ct.paste_back(crop, cx0, cy0, H, W)
    expected = np.zeros((H, W), bool)
    expected[4:7, 9:12] = True
    expected[8, 7] = True
    assert canvas.shape == (H, W) and canvas.dtype == bool
    assert np.array_equal(canvas, expected)
    assert np.array_equal(canvas[cy0:cy0 + s, cx0:cx0 + s], crop)
    png_like = canvas.astype(np.uint8) * 255                 # what cv2.imwrite receives
    assert png_like.dtype == np.uint8 and set(np.unique(png_like).tolist()) == {0, 255}
    # the bottom row (window edge y = 8 < H - 1) and left column (x = 7 > 0) are touched
    assert ct.touches_window_edge(crop, cx0, cy0, H, W)
    # the same crop at the image's bottom-left corner touches no window edge
    assert not ct.touches_window_edge(crop, 0, H - s, H, W)
    # frames_touching_border over written frames, as CropSmall counts it
    masks = {0: crop, 1: crop, 2: np.zeros((s, s), bool), 3: crop}
    touching = sum(int(ct.touches_window_edge(masks[k], cx0, cy0, H, W)) for k in range(1, 4))
    empty = sum(int(masks[k].sum() == 0) for k in range(1, 4))
    assert (touching, empty) == (2, 1)
