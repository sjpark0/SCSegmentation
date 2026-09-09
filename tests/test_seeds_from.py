"""Stage-1 supply: seeds_from.py (torch-free) behind the runner's --seeds-from.

docs/stage1-plan.md section 3 ("2단계 공급") and the runMVSeg.py row of section 6 fix the
contract: only (view, obj) pairs with a PNG change, the rest keep G0; the folder must be
MVSeed_<tag> and names the output suffix Sd<tag>; a SEED_MANIFEST.json must cover every
scored camera; provenance is the manifest's sha256 and a PNG digest built like
MANIFEST.json's content_digest.  The PNG reader is injected, so no cv2 here.
"""
import hashlib
import os

import pytest

from conftest import REPO, _load

import seeds_from as sf

np = pytest.importorskip("numpy")

H, W = 4, 5
CAMS = ["v0", "v1", "v2", "v3"]
FRAME = 7
OBJS = (1, 2, 3)


# ------------------------------------------------------------------ fakes
def touch(folder, cam, obj, frame=FRAME, data=b"png"):
    d = folder / cam / str(frame)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{obj}.png").write_bytes(data)


def loader_for(images):
    """path -> array from {(cam, obj): array}; None (cv2's failure value) otherwise."""
    def read(path):
        parts = path.replace(os.sep, "/").split("/")
        return images.get((parts[-3], int(parts[-1][:-4])))
    return read


def g0(seed=0):
    rng = np.random.RandomState(seed)
    return {v: {o: rng.rand(H, W) > 0.5 for o in OBJS} for v in range(len(CAMS))}


def img(value):
    return np.full((H, W), value, dtype=np.uint8)


def seed_folder(tmp_path, name="MVSeed_t"):
    folder = tmp_path / name
    folder.mkdir()
    return folder


# ------------------------------------------------------------------ folder name
def test_parse_folder_and_suffix():
    assert sf.parse_folder("MVSeed_control") == "control"
    assert sf.folder_suffix("MVSeed_control") == "Sdcontrol"
    assert sf.folder_suffix("MVSeed_vote_v1.2-b") == "Sdvote_v1.2-b"
    assert sf.PREFIX == "MVSeed_" and sf.SUFFIX == "Sd"


@pytest.mark.parametrize("bad", ["SegMaskSam3XW0", "mvseed_x", "MVSeed", "MVSeed_",
                                 "MVSeed_a/b", "MVSeed_a b", "../MVSeed_x", "MVSeed_x/",
                                 "", None, 3])
def test_parse_folder_refusals(bad):
    with pytest.raises(ValueError):
        sf.parse_folder(bad)


# ------------------------------------------------------------------ manifest coverage
def test_written_cams_accepts_names_indices_and_the_older_key():
    assert sf.written_cams({"written_views": ["v0", "v3"]}, CAMS) == ["v0", "v3"]
    assert sf.written_cams({"written_views": [0, 3]}, CAMS) == ["v0", "v3"]
    assert sf.written_cams({"written_cameras": ["v1"]}, CAMS) == ["v1"]
    # written_views wins when both are present
    assert sf.written_cams({"written_views": ["v2"], "written_cameras": ["v1"]}, CAMS) == ["v2"]
    for m in ({}, {"areas": {}}, {"written_views": "v0"}, {"written_views": [4]},
              {"written_views": [-1]}, {"written_views": [True]}, {"written_views": [1.5]}):
        with pytest.raises(ValueError):
            sf.written_cams(m, CAMS)


def test_check_coverage():
    m = {"written_views": ["v0", "v1", "v3"]}
    assert sf.check_coverage(m, ["v0", "v3"], CAMS) == ["v0", "v1", "v3"]
    assert sf.check_coverage(m, [], CAMS) == ["v0", "v1", "v3"]
    with pytest.raises(ValueError, match=r"\['v2'\]"):
        sf.check_coverage(m, ["v0", "v2", "v3"], CAMS)
    with pytest.raises(ValueError):
        sf.check_coverage({}, ["v0"], CAMS)


def test_read_manifest(tmp_path):
    folder = seed_folder(tmp_path)
    assert sf.read_manifest(str(folder)) == (None, None)
    data = b'{"written_views": ["v0"], "start_frame": 7}\n'
    (folder / sf.MANIFEST).write_bytes(data)
    m, sha = sf.read_manifest(str(folder))
    assert m == {"written_views": ["v0"], "start_frame": 7}
    assert sha == hashlib.sha256(data).hexdigest()
    (folder / sf.MANIFEST).write_bytes(b"{not json")
    with pytest.raises(ValueError):
        sf.read_manifest(str(folder))


# ------------------------------------------------------------------ preflight
def test_preflight_without_a_manifest_warns_and_lists(tmp_path):
    folder = seed_folder(tmp_path)
    touch(folder, "v1", 2)
    touch(folder, "v3", 1)
    pre = sf.preflight(str(folder), FRAME, CAMS, ["v0", "v3"])
    assert pre["manifest"] is None and pre["manifest_sha256"] is None and pre["written"] is None
    assert "no SEED_MANIFEST.json" in pre["warning"] and "'v0', 'v3'" in pre["warning"]
    assert {c: sorted(p) for c, p in pre["pngs"].items()} == {"v1": [2], "v3": [1]}


def test_preflight_with_a_manifest(tmp_path):
    folder = seed_folder(tmp_path)
    touch(folder, "v0", 1)
    (folder / sf.MANIFEST).write_text('{"written_views": ["v0", "v3"], "dataset": "Fencing", '
                                      '"start_frame": 7}')
    pre = sf.preflight(str(folder), FRAME, CAMS, ["v0", "v3"],
                       expect={"dataset": "Fencing", "start_frame": 7})
    assert pre["warning"] is None and pre["written"] == ["v0", "v3"]
    assert pre["manifest_sha256"] == hashlib.sha256((folder / sf.MANIFEST).read_bytes()).hexdigest()
    with pytest.raises(ValueError):                                   # coverage
        sf.preflight(str(folder), FRAME, CAMS, ["v0", "v2", "v3"])
    with pytest.raises(ValueError):                                   # another scene
        sf.preflight(str(folder), FRAME, CAMS, ["v0"], expect={"dataset": "Welder"})
    with pytest.raises(ValueError):                                   # another frame
        sf.preflight(str(folder), FRAME, CAMS, ["v0"], expect={"start_frame": 0})
    # a key the manifest lacks is not checked
    assert sf.preflight(str(folder), FRAME, CAMS, ["v0"], expect={"order": "index"})["written"]


def test_preflight_refuses_missing_or_misnamed_folders(tmp_path):
    with pytest.raises(ValueError):
        sf.preflight(str(tmp_path / "MVSeed_nope"), FRAME, CAMS, [])
    other = tmp_path / "SegMaskSam3XW0"
    other.mkdir()
    with pytest.raises(ValueError):
        sf.preflight(str(other), FRAME, CAMS, [])
    foreign = seed_folder(tmp_path)
    touch(foreign, "camera_0001", 1)
    with pytest.raises(ValueError, match="camera_0001"):
        sf.preflight(str(foreign), FRAME, CAMS, [])


# ------------------------------------------------------------------ layout
def test_list_seed_pngs(tmp_path):
    folder = seed_folder(tmp_path)
    touch(folder, "v2", 3)
    touch(folder, "v2", 1)
    touch(folder, "v1", 1, frame=FRAME + 1)                # another frame: not this run's
    (folder / "v2" / str(FRAME) / "notes.txt").write_text("x")
    (folder / sf.MANIFEST).write_text("{}")
    (folder / "v0").mkdir()                                # camera dir without the frame
    got = sf.list_seed_pngs(str(folder), FRAME, CAMS)
    assert sorted(got) == ["v2"]                           # v0/v1 hold nothing for this frame
    assert sorted(got["v2"]) == [1, 3]
    assert got["v2"][3].endswith(os.path.join("v2", str(FRAME), "3.png"))
    (folder / "v2" / str(FRAME) / "obj3.png").write_bytes(b"")
    with pytest.raises(ValueError, match="obj3.png"):
        sf.list_seed_pngs(str(folder), FRAME, CAMS)


# ------------------------------------------------------------------ apply
def test_apply_replaces_only_the_pairs_with_a_png(tmp_path):
    folder = seed_folder(tmp_path)
    touch(folder, "v1", 2)
    touch(folder, "v3", 1)
    touch(folder, "v3", 3)
    masks = g0()
    before = {v: dict(d) for v, d in masks.items()}
    new = {("v1", 2): img(255), ("v3", 1): img(0), ("v3", 3): img(200)}
    new[("v3", 3)][0, 0] = 0
    rep = sf.apply_seed_folder(masks, str(folder), FRAME, CAMS, track_idx=range(4),
                               obj_ids=list(OBJS), shape=(H, W), read_png=loader_for(new))
    assert rep["replaced"] == 3 and rep["replaced_views"] == {"1": [2], "3": [1, 3]}
    assert rep["untracked_views"] == [] and rep["added"] == 0
    for v in range(4):
        for o in OBJS:
            if (CAMS[v], o) in new:
                assert masks[v][o].dtype == bool and masks[v][o].shape == (H, W)
                assert np.array_equal(masks[v][o], new[(CAMS[v], o)] > 127)
            else:
                assert masks[v][o] is before[v][o]                  # G0 kept, same object
    assert sorted(masks) == [0, 1, 2, 3]


def test_apply_threshold_is_the_eval_reading():
    assert not sf.to_seed(img(127)).any() and sf.to_seed(img(128)).all()
    assert sf.THRESHOLD == 127


def test_apply_reports_added_emptied_and_changed(tmp_path):
    folder = seed_folder(tmp_path)
    for o in OBJS:
        touch(folder, "v2", o)
    touch(folder, "v1", 1)
    masks = g0()
    del masks[2][3]                                        # G0 empty for (2, 3)
    masks[2][2] = np.zeros((H, W), bool)                   # G0 empty but present
    same = (masks[1][1].astype(np.uint8) * 255)            # identical to G0
    new = {("v2", 1): img(0), ("v2", 2): img(0), ("v2", 3): img(255), ("v1", 1): same}
    rep = sf.apply_seed_folder(masks, str(folder), FRAME, CAMS, range(4), list(OBJS), (H, W),
                               loader_for(new))
    assert rep["replaced"] == 4 and rep["replaced_views"] == {"1": [1], "2": [1, 2, 3]}
    assert rep["added"] == 1                               # (2, 3) had no G0 entry
    assert rep["emptied"] == 1                             # (2, 1): non-empty G0 -> blank PNG
    assert rep["changed_views"] == {"2": [1, 3]}            # (2, 2) empty->empty, (1, 1) same
    assert masks[2][3].all() and not masks[2][1].any()
    # a blank PNG for an object G0 never produced is not "added" in the changed sense
    del masks[3][2]
    touch(folder, "v3", 2)
    new[("v3", 2)] = img(0)
    rep = sf.apply_seed_folder(masks, str(folder), FRAME, CAMS, range(4), list(OBJS), (H, W),
                               loader_for(new))
    # (2, 3) exists now, so only (3, 2) is added; a blank seed for a blank G0 is no change
    assert rep["added"] == 1 and "3" not in rep["changed_views"] and not masks[3][2].any()
    assert rep["changed_views"] == {} and rep["emptied"] == 0


def test_apply_leaves_untracked_views_alone(tmp_path):
    folder = seed_folder(tmp_path)
    touch(folder, "v0", 1)
    touch(folder, "v2", 1)
    touch(folder, "v3", 2)
    (folder / "v1" / str(FRAME)).mkdir(parents=True)       # no PNG: not listed as untracked
    masks = g0()
    before = {v: dict(d) for v, d in masks.items()}
    new = {(c, o): img(255) for c in CAMS for o in OBJS}
    rep = sf.apply_seed_folder(masks, str(folder), FRAME, CAMS, track_idx=[0, 1],
                               obj_ids=list(OBJS), shape=(H, W), read_png=loader_for(new))
    assert rep["replaced"] == 1 and rep["replaced_views"] == {"0": [1]}
    assert rep["untracked_views"] == [2, 3]
    assert masks[2][1] is before[2][1] and masks[3][2] is before[3][2]
    assert masks[0][1].all()


def test_apply_creates_the_view_entry_when_the_pass_yielded_none(tmp_path):
    folder = seed_folder(tmp_path)
    touch(folder, "v1", 1)
    masks = {0: {1: img(255) > 127}}
    rep = sf.apply_seed_folder(masks, str(folder), FRAME, CAMS, [0, 1], [1], (H, W),
                               loader_for({("v1", 1): img(255)}))
    assert rep["added"] == 1 and masks[1][1].all()


@pytest.mark.parametrize("case", ["foreign_cam", "bad_stem", "foreign_obj", "shape", "unreadable"])
def test_apply_refusals(tmp_path, case):
    folder = seed_folder(tmp_path)
    new = {("v1", 1): img(255)}
    if case == "foreign_cam":
        touch(folder, "camera_0001", 1)
    elif case == "bad_stem":
        touch(folder, "v1", 1)
        (folder / "v1" / str(FRAME) / "mask.png").write_bytes(b"")
    elif case == "foreign_obj":
        touch(folder, "v1", 9)
        new[("v1", 9)] = img(255)
    elif case == "shape":
        touch(folder, "v1", 1)
        new[("v1", 1)] = np.zeros((H + 1, W), np.uint8)
    else:
        touch(folder, "v1", 1)
        new = {}
    masks = g0()
    before = {v: dict(d) for v, d in masks.items()}
    with pytest.raises(ValueError):
        sf.apply_seed_folder(masks, str(folder), FRAME, CAMS, range(4), list(OBJS), (H, W),
                             loader_for(new))
    # a foreign object or camera is refused before anything is touched
    if case in ("foreign_cam", "bad_stem", "foreign_obj"):
        assert all(masks[v][o] is before[v][o] for v in masks for o in masks[v])


# ------------------------------------------------------------------ digest / provenance
def test_png_digest_is_deterministic_and_content_only(tmp_path):
    a = seed_folder(tmp_path, "MVSeed_a")
    b = seed_folder(tmp_path, "MVSeed_b")
    for folder, order in ((a, (1, 2)), (b, (2, 1))):    # creation order must not matter
        for o in order:
            touch(folder, "v3", o, data=bytes([o]) * 10)
        touch(folder, "v0", 1, data=b"zero")
    (a / "notes.txt").write_text("ignored")
    (a / sf.MANIFEST).write_text("{}")                       # manifest is not a PNG
    da, na = sf.png_digest(str(a))
    db, nb = sf.png_digest(str(b))
    assert da == db and na == nb == 3 and len(da) == 64
    assert sf.png_digest(str(a)) == (da, 3)                   # a second call agrees
    touch(b, "v3", 2, data=b"\x02" * 10 + b"\x00")            # one byte more
    assert sf.png_digest(str(b))[0] != da
    (b / "v3" / str(FRAME) / "2.png").unlink()
    assert sf.png_digest(str(b)) != (da, 3)                   # a missing file changes it
    empty = seed_folder(tmp_path, "MVSeed_empty")
    assert sf.png_digest(str(empty)) == (hashlib.sha256().hexdigest(), 0)


def test_png_digest_matches_manifest_content_digest(tmp_path):
    """Same construction as eval/manifest.py content_digest, so provenance.seeds_from
    .png_digest can be compared with a MANIFEST.json written over the same folder."""
    manifest = _load("eval_manifest", os.path.join(REPO, "eval", "manifest.py"))
    folder = seed_folder(tmp_path)
    touch(folder, "v0", 1, data=b"one")
    touch(folder, "v3", 2, data=b"two")
    touch(folder, "v3", 1, data=b"three")
    (folder / sf.MANIFEST).write_text("{}")
    digest, n = sf.png_digest(str(folder))
    ref, pairs = manifest.content_digest(str(folder), jobs=1)
    assert digest == ref and n == len(pairs) == 3


def test_folder_provenance(tmp_path):
    folder = seed_folder(tmp_path, "MVSeed_vote")
    touch(folder, "v0", 1, data=b"one")
    p = sf.folder_provenance(str(folder))
    assert p["folder"] == "MVSeed_vote" and p["path"] == str(folder)
    assert p["manifest_sha256"] is None and p["n_png"] == 1
    assert p["png_digest"] == sf.png_digest(str(folder))[0]
    data = b'{"written_views": ["v0"]}'
    (folder / sf.MANIFEST).write_bytes(data)
    p = sf.folder_provenance(str(folder) + os.sep)            # trailing separator tolerated
    assert p["folder"] == "MVSeed_vote"
    assert p["manifest_sha256"] == hashlib.sha256(data).hexdigest()
    assert p["png_digest"] == sf.png_digest(str(folder))[0]   # the manifest is not hashed in
