"""In-process pipeline (docs/pipeline-inprocess.md): runMVSeg.py --stage1 / --ref-snap / --save-stages.

The contract: --stage1 <tag> is the file route (stage1 CLI -> build_geoset_seeds.py -> --seeds-from
MVSeed_<tag>) without the files, so the same folder name and the same seeds; --ref-snap is
MVSeed/s2r6_snap.py applied in place, same rule and same bytes.  Torch-free; cv2 only where the snap
itself runs.
"""
import os
import sys

import pytest

from conftest import REPO, _load
from test_runner_cli import cams, mod, ns

import seeds_from as sf

np = pytest.importorskip("numpy")

supply = _load("stage1_supply", os.path.join(REPO, "SCSam3", "stage1_supply.py"))
H, W = 6, 7


# ------------------------------------------------------------------ resolve_run
HEADLINE = dict(xview_window=1, xview_mode="C", xview_gate=True, xview_ptr=True, xview_tpos_shift=4,
                ref_cam="muvod", frame0_seed=True, crop_small=10000, crop_gate="v2")


@pytest.mark.parametrize("tag", supply.TAGS)
def test_stage1_names_the_folder_as_seeds_from(tag):
    cam_names, written = cams("Fencing")
    a = mod.resolve_run(ns(stage1=tag, **HEADLINE), cam_names, written, num_frame=21, ref_suffix="M")
    b = mod.resolve_run(ns(seeds_from=f"MVSeed_{tag}", **HEADLINE), cam_names, written, num_frame=21, ref_suffix="M")
    assert a["out_name"] == b["out_name"] == f"SegMaskSam3XW1CGPS4MSd{tag}FsCs10kG2"
    assert (a["stage1"], a["seeds_from"], b["stage1"]) == (tag, None, None)
    assert a["track_idx"] == b["track_idx"]


def test_headline_in_one_process_folder_name():
    cam_names, written = cams("Fencing")
    r = mod.resolve_run(ns(stage1="geoset_rep", ref_snap=True, **HEADLINE), cam_names, written,
                        num_frame=21, ref_suffix="M")
    assert r["out_name"] == "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs"
    assert r["ref_snap"] is True and r["save_stages"] is False


def test_stage1_refusals():
    cam_names, written = cams("Fencing")
    for flags in (dict(stage1="geoset_rep", seeds_from="MVSeed_geoset_rep"),
                  dict(stage1="geoset_rep", repair_seeds=True),
                  dict(stage1="nope"),
                  dict(stage1="geoset_rep", algo="OneStageNew", xview_window=None),
                  dict(stage1="geoset_rep", out="SegMaskSam3XW0Mx")):
        kw = dict(xview_window=0, ref_cam="muvod")
        kw.update(flags)
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(**kw), cam_names, written, num_frame=21, ref_suffix="M")
    r = mod.resolve_run(ns(xview_window=0, ref_cam="muvod", stage1="geoset_rep", out="SegMaskSam3XW0MSdgeoset_rep_x"),
                        cam_names, written, num_frame=21, ref_suffix="M")
    assert r["out_name"] == "SegMaskSam3XW0MSdgeoset_rep_x"


def test_ref_snap_refusals():
    cam_names, written = cams("Fencing")
    with pytest.raises(SystemExit):           # the rule is defined on c_ini
        mod.resolve_run(ns(xview_window=0, ref_snap=True), cam_names, written, num_frame=21)
    with pytest.raises(SystemExit):           # an explicit --out carries Rs
        mod.resolve_run(ns(xview_window=0, ref_cam="muvod", ref_snap=True, out="SegMaskSam3XW0Mx"),
                        cam_names, written, num_frame=21, ref_suffix="M")
    r = mod.resolve_run(ns(xview_window=0, ref_cam="muvod", ref_snap=True), cam_names, written,
                        num_frame=21, ref_suffix="M")
    assert r["out_name"] == "SegMaskSam3XW0MRs"


def test_parse_args_new_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Fencing"])
    a = mod.parse_args()
    assert (a.stage1, a.ref_snap, a.save_stages) == (None, False, False)
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Fencing", "--stage1", "geoset_rep", "--ref-snap", "--save-stages"])
    a = mod.parse_args()
    assert (a.stage1, a.ref_snap, a.save_stages) == ("geoset_rep", True, True)


# ------------------------------------------------------------------ seeds: memory == folder
def test_apply_seeds_equals_apply_seed_folder(tmp_path):
    rng = np.random.RandomState(3)
    cam_names = ["v0", "v1", "v2", "v3"]
    seeds = {c: {o: rng.rand(H, W) > 0.6 for o in (1, 2)} for c in cam_names}
    seeds["v2"][2][:] = False                                       # an empty seed
    folder = tmp_path / "MVSeed_t"
    images = {}
    for c, objs in seeds.items():
        for o, m in objs.items():
            d = folder / c / "7"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{o}.png").write_bytes(b"x")
            images[(c, o)] = m.astype(np.uint8) * 255

    def read(path):
        parts = path.replace(os.sep, "/").split("/")
        return images[(parts[-3], int(parts[-1][:-4]))]

    def g0():
        r = np.random.RandomState(0)
        return {v: {o: r.rand(H, W) > 0.5 for o in (1, 2)} for v in range(4)}
    a, b = g0(), g0()
    ra = sf.apply_seed_folder(a, str(folder), 7, cam_names, [0, 1, 2], [1, 2], (H, W), read)
    rb = sf.apply_seeds(b, seeds, cam_names, [0, 1, 2], [1, 2], (H, W))
    assert ra == rb
    assert list(a) == list(b) and all(list(a[v]) == list(b[v]) for v in a)
    assert all(np.array_equal(a[v][o], b[v][o]) and b[v][o].dtype == bool for v in a for o in a[v])
    with pytest.raises(ValueError):
        sf.apply_seeds(g0(), {"v9": {1: seeds["v0"][1]}}, cam_names, [0], [1, 2], (H, W))
    with pytest.raises(ValueError):
        sf.apply_seeds(g0(), {"v0": {1: np.zeros((2, 2), bool)}}, cam_names, [0], [1, 2], (H, W))
    with pytest.raises(ValueError):
        sf.apply_seeds(g0(), {"v0": {5: seeds["v0"][1]}}, cam_names, [0], [1, 2], (H, W))


# ------------------------------------------------------------------ to_seeds == build_geoset_seeds rules
def test_to_seeds_rules():
    m = np.ones((H, W), bool)
    final = {"c1": {1: m, 3: m}, "c0": {1: m, 2: np.zeros((H, W), bool)}}
    seeds, counts = supply.to_seeds(final, ["c0", "c1", "c2"], [1, 2, 3], (H, W), "c0", np)
    assert sorted(seeds) == ["c0", "c1", "c2"] and all(sorted(seeds[c]) == [1, 2, 3] for c in seeds)
    assert seeds["c1"][1].all() and not seeds["c1"][2].any() and not seeds["c2"][3].any()
    assert counts == dict(n_seeds=9, n_from_stage1=4, n_empty=5, n_from_stage1_zero=1, cams_without_masks=["c2"])
    for bad in (dict(final={"c0": {1: m}, "c9": {1: m}}),                       # camera outside Video/
                dict(final={"c1": {1: m}}),                                     # no reference camera
                dict(final={"c0": {4: m}}),                                     # object outside the basic set
                dict(final={"c0": {1: np.ones((2, 2), bool)}})):                # wrong size
        with pytest.raises(ValueError):
            supply.to_seeds(bad["final"], ["c0", "c1", "c2"], [1, 2, 3], (H, W), "c0", np)


def test_mask_digest_order_free_and_content_sensitive():
    a = {"c0": {1: np.eye(H, W, dtype=bool), 2: np.zeros((H, W), bool)}, "c1": {1: np.ones((H, W), bool)}}
    b = {"c1": {1: np.ones((H, W), bool)}, "c0": {2: np.zeros((H, W), bool), 1: np.eye(H, W, dtype=bool)}}
    assert supply.mask_digest(a, np) == supply.mask_digest(b, np)
    b["c0"][2][0, 0] = True
    assert supply.mask_digest(a, np) != supply.mask_digest(b, np)


def test_tags_are_the_stage1_presets():
    pytest.importorskip("cv2")
    S = supply.stage1_module()
    assert tuple(S.PRESETS) == supply.TAGS
    assert S.PRESETS["geoset_rep"] == dict(arm="geoset_pred_ft", gate="reproj")


# ------------------------------------------------------------------ ref snap == s2r6_snap
def test_ref_snap_rule_is_s2r6():
    cv2 = pytest.importorskip("cv2")
    snap = mod.ref_snap_module()
    old = _load("s2r6_snap", os.path.join(REPO, "MVSeed", "s2r6_snap.py"))
    assert (snap.TAU, snap.DELTA, snap.DILATE, snap.SUFFIX) == (old.TAU, old.DELTA, old.DILATE, old.SUFFIX)
    rng = np.random.RandomState(1)
    for k in range(200):
        m0 = np.zeros((40, 50), bool)
        y, x = rng.randint(0, 30), rng.randint(0, 40)
        m0[y:y + rng.randint(1, 10), x:x + rng.randint(1, 10)] = k % 7 != 0
        mt = m0.copy()
        mt[rng.rand(40, 50) < rng.rand() * 0.2] ^= True
        i0 = rng.randint(0, 255, (40, 50)).astype(np.int16)
        it = (i0 + rng.randint(-rng.randint(1, 12), 12, (40, 50))).astype(np.int16)
        assert snap.decide(m0, mt, i0, it, np, cv2) == old.decide(m0, mt, i0, it, np, cv2)


def test_snap_in_place(tmp_path):
    cv2 = pytest.importorskip("cv2")
    snap = mod.ref_snap_module()
    out, video = tmp_path / "out", tmp_path / "Video" / "cam0"
    video.mkdir(parents=True)
    img = (np.arange(40 * 50).reshape(40, 50) % 200).astype(np.uint8)
    m0 = np.zeros((40, 50), np.uint8); m0[10:20, 10:20] = 255
    moved = np.zeros((40, 50), np.uint8); moved[25:35, 30:40] = 255
    near = m0.copy(); near[10, 10] = 0
    for t in (5, 6, 7):
        cv2.imwrite(str(video / f"{t:06d}.jpg"), img)
        (out / "cam0" / str(t)).mkdir(parents=True)
        (out / "cam1" / str(t)).mkdir(parents=True)
        cv2.imwrite(str(out / "cam1" / str(t) / "1.png"), near)
    cv2.imwrite(str(out / "cam0" / "5" / "1.png"), m0)
    cv2.imwrite(str(out / "cam0" / "6" / "1.png"), near)       # IoU 0.99, same image -> snapped
    cv2.imwrite(str(out / "cam0" / "7" / "1.png"), moved)      # IoU 0 -> kept
    before = {p: p.read_bytes() for p in out.rglob("*.png")}
    st = snap.snap_in_place(str(out), str(video), "cam0", 5, np, cv2, keep_dir=str(tmp_path / "keep"))
    assert st["snapped_files"] == ["cam0/6/1.png"] and st["objects_frames"] == 2
    assert (out / "cam0" / "6" / "1.png").read_bytes() == before[out / "cam0" / "5" / "1.png"]
    for p, b in before.items():
        if p != out / "cam0" / "6" / "1.png":
            assert p.read_bytes() == b
        assert (tmp_path / "keep" / p.relative_to(out)).read_bytes() == b      # the pre-snap copy kept its bytes
