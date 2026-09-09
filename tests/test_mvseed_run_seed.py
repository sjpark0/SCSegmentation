"""T10: MVSeed/run_seed.py view orders and the SEED_MANIFEST.json schema (torch-free).

run_seed.py imports torch/cv2 only inside main(), so the module loads on the host and
view_order() / build_manifest() are exercised as plain functions.  The manifest schema
is the contract the mainline `--seeds-from` hook and score_seeds.py read
(docs/stage1-plan.md section 6): every view written, per-view coverage consistent with
areas, the scored (annotated) cameras named separately.
"""
import importlib.util
import json
import os

import pytest

from conftest import REPO

CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")


def _load_run_seed():
    # same shape as conftest's loaders; MVSeed/ is not a package and not on sys.path
    path = os.path.join(REPO, "MVSeed", "run_seed.py")
    spec = importlib.util.spec_from_file_location("run_seed", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load_run_seed()
SIZES = (1, 2, 3, 4, 9, 10, 25, 46)          # 1..2 edge cases, then real view counts


def test_module_loads_without_torch():
    assert not hasattr(rs, "torch") and not hasattr(rs, "cv2") and not hasattr(rs, "np")


# ------------------------------------------------------------------ view_order
@pytest.mark.parametrize("order", rs.ORDERS)
@pytest.mark.parametrize("n", SIZES)
def test_every_order_is_a_permutation(order, n):
    for ref in range(n):
        assert sorted(rs.view_order(order, n, ref)) == list(range(n))


def test_index_is_the_control():
    for n in SIZES:
        assert rs.view_order("index", n, 0) == list(range(n))
    assert rs.view_order("index", 10, 4) == rs.view_order("index", 10, 9)   # ref-independent


def test_reverse_is_the_mirror_of_index():
    for n in SIZES:
        assert rs.view_order("reverse", n, 0) == rs.view_order("index", n, 0)[::-1]
    assert rs.view_order("reverse", 10, 4) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


def test_ref_outward_starts_at_the_reference_and_walks_nearest_first():
    for n in SIZES:
        for ref in range(n):
            seq = rs.view_order("ref_outward", n, ref)
            assert seq[0] == ref
            dist = [abs(v - ref) for v in seq]
            assert dist == sorted(dist)                    # never jumps back inward
            for a, b in zip(seq, seq[1:]):                 # at equal distance, lower first
                if abs(a - ref) == abs(b - ref):
                    assert a < b


def test_ref_outward_examples():
    assert rs.view_order("ref_outward", 10, 4) == [4, 3, 5, 2, 6, 1, 7, 0, 8, 9]
    assert rs.view_order("ref_outward", 4, 0) == [0, 1, 2, 3]      # reference at an end
    assert rs.view_order("ref_outward", 4, 3) == [3, 2, 1, 0]
    assert rs.view_order("ref_outward", 1, 0) == [0]


def test_prompt_position_per_order():
    # where the reference lands in the pseudo-video: index keeps it, reverse mirrors it,
    # ref_outward puts it first -- the number run_seed prompts on and writes to the manifest
    n, ref = 10, 4
    assert rs.view_order("index", n, ref).index(ref) == ref
    assert rs.view_order("reverse", n, ref).index(ref) == n - 1 - ref
    assert rs.view_order("ref_outward", n, ref).index(ref) == 0


def test_unknown_order_is_refused():
    with pytest.raises(ValueError):
        rs.view_order("random", 10, 4)


def test_orders_on_every_configured_scene():
    # the same resolution main() does: perms (or start_cam..), ref = position of c_ini
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    scenes = [(k, d) for k, d in cfg.items() if "c_ini" in d]
    assert len(scenes) == 17
    for _, d in scenes:
        perms = d.get("perms") or list(range(d["start_cam"], d["num_cam"] + d["start_cam"]))
        ref = perms.index(d["c_ini"])
        for order in rs.ORDERS:
            seq = rs.view_order(order, len(perms), ref)
            assert sorted(seq) == list(range(len(perms)))
        assert rs.view_order("ref_outward", len(perms), ref)[0] == ref


# ------------------------------------------------------------------ build_manifest
# Fencing's shape: 10 views v0..v9, c_ini v4, annotated v0/v4/v9, three objects.
CAMS = [f"v{i}" for i in range(10)]
SCORED = ["v0", "v4", "v9"]
REF, START, N_OBJ = 4, 0, 3
ARGV = ["MVSeed/run_seed.py", "Fencing", "--out", "MVSeed_x"]
REQUIRED = {"dataset", "generator", "order", "sequence", "reference", "reference_view_index",
            "prompt_position", "start_frame", "n_views", "n_objects",
            "written_views", "scored_views", "coverage", "areas", "argv"}


def areas_for(cams, n_obj=N_OBJ, empty=()):
    """{cam: {obj: px}} the way main() builds it -- int keys, view-index order."""
    return {c: {o: 0 if (c, o) in empty else 1000 * o + i for o in range(1, n_obj + 1)}
            for i, c in enumerate(cams)}


def manifest(order="index", areas=None, cams=CAMS):
    seq = rs.view_order(order, len(cams), REF)
    areas = areas_for(cams) if areas is None else areas
    return rs.build_manifest("Fencing", order, seq, cams, REF, START, N_OBJ, SCORED, areas, ARGV)


def test_manifest_has_every_required_key():
    m = manifest()
    assert REQUIRED <= set(m)
    assert m["generator"] == "video" == rs.GENERATOR
    assert (m["dataset"], m["order"], m["start_frame"]) == ("Fencing", "index", START)
    assert (m["n_views"], m["n_objects"]) == (len(CAMS), N_OBJ)
    assert m["argv"] == ARGV


def test_written_views_are_all_views_and_scored_is_the_annotated_subset():
    m = manifest()
    assert m["written_views"] == CAMS
    assert m["scored_views"] == SCORED
    assert set(m["scored_views"]) <= set(m["written_views"])


def test_reference_and_prompt_position_follow_the_order():
    for order, pos in (("index", 4), ("reverse", 5), ("ref_outward", 0)):
        m = manifest(order)
        assert (m["reference"], m["reference_view_index"]) == ("v4", REF)
        assert m["sequence"] == rs.view_order(order, len(CAMS), REF)
        assert m["prompt_position"] == pos
        assert m["sequence"][m["prompt_position"]] == m["reference_view_index"]


def test_coverage_matches_areas():
    m = manifest()
    assert list(m["coverage"]) == list(m["areas"]) == m["written_views"]
    for cam in CAMS:
        assert m["coverage"][cam] == sorted(m["areas"][cam]) == [1, 2, 3]


def test_an_empty_mask_is_still_covered():
    # an all-zero PNG is the generator saying "not in this view"; coverage keeps it and
    # areas (0) tells it apart from a real seed
    m = manifest(areas=areas_for(CAMS, empty={("v7", 2)}))
    assert 2 in m["coverage"]["v7"] and m["areas"]["v7"][2] == 0


def test_an_object_without_a_png_is_not_covered():
    # the shape a per-view generator can produce: no file for (v7, 2)
    areas = areas_for(CAMS)
    del areas["v7"][2]
    m = manifest(areas=areas)
    assert m["coverage"]["v7"] == [1, 3]
    assert m["coverage"]["v6"] == [1, 2, 3]


def test_written_views_follow_what_was_written():
    # a view with no masks is not claimed, while n_views still counts the pseudo-video
    areas = areas_for(CAMS)
    del areas["v9"]
    m = manifest(areas=areas)
    assert m["written_views"] == CAMS[:-1] and "v9" not in m["coverage"]
    assert m["n_views"] == len(CAMS)


def test_manifest_survives_json():
    # int object keys become strings on disk; coverage stays a list of ints
    r = json.loads(json.dumps(manifest()))
    assert r["sequence"] == list(range(10)) and r["prompt_position"] == REF
    for cam in r["written_views"]:
        assert sorted(int(o) for o in r["areas"][cam]) == r["coverage"][cam]
        assert all(isinstance(o, int) for o in r["coverage"][cam])
