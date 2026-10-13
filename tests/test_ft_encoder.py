"""S2-R5 (docs/stage2-R5-prereg.md): --ft-encoder folder rule, refusals, the checkpoint lookup and
leak check of SCSam3/ft_encoder.py, the swap / restore digests, and the registered checkpoint
table against stage1/'s files (skipped when stage1/geomem is missing).
"""
import argparse
import hashlib
import json
import os
import sys

import pytest

from conftest import REPO, _load, load_runmvseg

mod = load_runmvseg()
fm = _load("ft_encoder", os.path.join(REPO, "SCSam3", "ft_encoder.py"))
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
SCENES = ["AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "CBABasketball",
          "Carpark", "CoffeeMartini", "Dog", "Fencing", "FlameSteak", "Frog", "MATF", "MartialArts",
          "Painter", "PoznanStreet", "Welder"]
HEAD = dict(xview_window=1, xview_mode="C", xview_gate=True, xview_ptr=True, xview_tpos_shift=4)
# docs/stage2-R5-prereg.md section 1
REGISTERED_SHA256 = {
    0: "03f5d407b8bed936617395d1730db266c15a16f27f7fe7cdbd8cae5e153ea498",
    1: "8e043c4d48327ec2bbf69414bb06ad3e70e84094ccb9f96085e27323542a2518",
    2: "5b9ef4885d9e1f09caa88b5622c7b4442b57a11352447a5a26d8e7a8a0b7fc2f",
    3: "9b5e2b3010c4440ca8dda6621abd4231a457cdad3b6c1504f88b3167b663649d",
    4: "05402ec462fb0d90a0801c1b440e420639bdedf72832c2bfc6186b2db60b77af",
}
REGISTERED_FOLDS_SHA256 = "ab12356d9aaeef823c5100e90b6692ae6b486ca8de665ca461484210dfb0c0d7"
SIBLING = os.path.isfile(fm.DEFAULT_FOLDS)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in mod.XVIEW_ENV:
        monkeypatch.delenv(k, raising=False)


def ns(**kw):
    d = dict(algo="MVOpt", xview_window=None, xview_hygiene=False, track_cams=None, out=None,
             xview_mode=None, xview_gate=False, xview_ptr=False, xview_tpos_shift=None,
             ref_cam=None, repair_seeds=False, seeds_from=None, frame0_seed=False,
             crop_small=0, crop_gate=None, crop_scale=None, crop_min_side=None,
             ft_encoder=None, ft_runs=None, ft_folds=None)
    d.update(kw)
    return argparse.Namespace(**d)


def cams(name):
    c = mod.load_config(CONFIG, name)
    cam_names = [mod.cam_name(x, c["prefix"], c["prefix1"]) for x in c["perms"]]
    written = [mod.cam_name(x, c["prefix"], c["prefix1"]) for x in c["cam_list"]]
    return cam_names, written


# ------------------------------------------------------------------ folder rule
def test_folder_suffix():
    assert fm.folder_suffix("geoset_pred_ft") == "Ftgp"
    with pytest.raises(ValueError):
        fm.folder_suffix("geoset_ft")


@pytest.mark.parametrize("ds", SCENES)
def test_registered_folder_names(ds):
    """The two S2-R5 arms land on the registered folders, next to (never on) their E8 bases."""
    cam_names, written = cams(ds)
    seeds = dict(seeds_from="MVSeed_geoset_rep", frame0_seed=True, ref_cam="muvod")
    for flags, base_name in ((dict(HEAD, crop_small=10000, crop_gate="v2", **seeds),
                              "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2"),
                             (dict(xview_window=0, **seeds), "SegMaskSam3XW0MSdgeoset_repFs")):
        base = mod.resolve_run(ns(**flags), cam_names, written, num_frame=21, ref_suffix="M")
        assert base["out_name"] == base_name and base["ft_encoder"] is None
        r = mod.resolve_run(ns(ft_encoder="geoset_pred_ft", **flags), cam_names, written,
                            num_frame=21, ref_suffix="M")
        assert r["out_name"] == {"SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2":
                                 "SegMaskSam3XW1CGPS4MSdgeoset_repFsFtgpCs10kG2",
                                 "SegMaskSam3XW0MSdgeoset_repFs":
                                 "SegMaskSam3XW0MSdgeoset_repFsFtgp"}[base_name]
        assert r["ft_encoder"] == "geoset_pred_ft"
        # the flag changes nothing else about the run (sessions, kwargs, seeds, crop)
        assert {k: v for k, v in r.items() if k not in ("out_name", "ft_encoder")} == \
               {k: v for k, v in base.items() if k not in ("out_name", "ft_encoder")}


def test_explicit_out_and_refusals():
    cam_names, written = cams("Fencing")
    r = mod.resolve_run(ns(xview_window=0, out="SegMaskSam3XWmineFtgp", ft_encoder="geoset_pred_ft"),
                        cam_names, written, num_frame=21)
    assert r["out_name"] == "SegMaskSam3XWmineFtgp"
    # an explicit --out without the marker could be the base folder: refused
    for out in ("SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2", "SegMaskSam3XW0MSdgeoset_repFs", "Whatever"):
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(xview_window=0, out=out, ft_encoder="geoset_pred_ft"), cam_names,
                            written, num_frame=21)
    with pytest.raises(SystemExit):                                  # unknown arm
        mod.resolve_run(ns(xview_window=0, ft_encoder="geoset_ft"), cam_names, written, num_frame=21)
    for algo in ("OneStage", "OneStageNew"):                         # MVOpt only
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(algo=algo, ft_encoder="geoset_pred_ft"), cam_names, written)
    with pytest.raises(SystemExit):                                  # the two-pass loop
        mod.resolve_run(ns(xview_window=1, xview_mode="E", ft_encoder="geoset_pred_ft"), cam_names,
                        written, num_frame=21)
    for kw in (dict(ft_runs="/x"), dict(ft_folds="/x.json")):        # stray settings
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(xview_window=0, **kw), cam_names, written, num_frame=21)


def test_parse_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Fencing"])
    a = mod.parse_args()
    assert (a.ft_encoder, a.ft_runs, a.ft_folds) == (None, None, None)
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Fencing", "--ft-encoder", "geoset_pred_ft"])
    assert mod.parse_args().ft_encoder == "geoset_pred_ft"


def test_module_is_loaded_once():
    assert mod.ft_encoder_module() is mod.ft_encoder_module()
    assert mod.ft_encoder_module().ARMS == fm.ARMS


# ------------------------------------------------------------------ lookup and leak check
def _fake_runs(tmp_path, folds):
    (tmp_path / "folds.json").write_text(json.dumps(folds))
    for k in set(folds.values()):
        d = tmp_path / "runs" / "geoset_pred_ft" / f"fold{k}"
        d.mkdir(parents=True)
        (d / "ckpt.pt").write_bytes(f"ckpt{k}".encode())
    return str(tmp_path / "runs"), str(tmp_path / "folds.json")


def test_resolve(tmp_path):
    runs, folds = _fake_runs(tmp_path, {"Fencing": 0, "Dog": 2})
    info = fm.resolve("geoset_pred_ft", "Dog", runs, folds)
    assert info["fold"] == 2 and info["path"].endswith(os.path.join("geoset_pred_ft", "fold2", "ckpt.pt"))
    assert info["sha256"] == hashlib.sha256(b"ckpt2").hexdigest()
    assert info["folds_sha256"] == hashlib.sha256(open(folds, "rb").read()).hexdigest()
    with pytest.raises(ValueError):                                  # scene not in the table
        fm.resolve("geoset_pred_ft", "Welder", runs, folds)
    with pytest.raises(ValueError):                                  # no table
        fm.resolve("geoset_pred_ft", "Dog", runs, str(tmp_path / "none.json"))
    os.remove(info["path"])
    with pytest.raises(ValueError):                                  # no checkpoint
        fm.resolve("geoset_pred_ft", "Dog", runs, folds)


def test_check_checkpoint():
    info = dict(arm="geoset_pred_ft", scene="Dog", fold=2, path="x")
    good = dict(arm="geoset_pred_ft", fold=2, scenes=["Fencing", "Welder"], memory_attention={})
    assert fm.check_checkpoint(good, info) == ["Fencing", "Welder"]
    for bad in (dict(good, arm="geoset_ft"), dict(good, fold=3), dict(good, scenes=["Dog", "Welder"]),
                dict(good, scenes=[]), {k: v for k, v in good.items() if k != "memory_attention"}):
        with pytest.raises(ValueError):
            fm.check_checkpoint(bad, info)


# ------------------------------------------------------------------ swap / restore (torch)
def test_swap_restore_digests():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    enc = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
    other = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
    with torch.no_grad():
        other[1].weight.mul_(2.0)
    d_enc, d_other = fm.state_digest(enc.state_dict(), torch), fm.state_digest(other.state_dict(), torch)
    assert d_enc != d_other
    original = fm.swap_in(enc, other.state_dict(), torch)
    assert fm.state_digest(enc.state_dict(), torch) == d_other
    assert fm.state_digest(original, torch) == d_enc
    assert all(v.device.type == "cpu" for v in original.values())
    x = torch.randn(2, 4)
    assert torch.equal(enc(x), other(x))
    fm.restore(enc, original, torch)
    assert fm.state_digest(enc.state_dict(), torch) == d_enc
    # a foreign state is refused before anything is copied (a failing strict load_state_dict
    # has already copied the matching keys, so it alone would leave the module half-swapped)
    partial = {"0.weight": torch.zeros(3, 4)}
    with pytest.raises(ValueError):
        fm.swap_in(enc, partial, torch)
    bad_shape = dict(other.state_dict(), **{"0.weight": torch.zeros(4, 4)})
    with pytest.raises(ValueError):
        fm.swap_in(enc, bad_shape, torch)
    assert fm.state_digest(enc.state_dict(), torch) == d_enc


def test_probe_follows_the_weights():
    """The functional probe changes with the swap and comes back with the restore; it is
    deterministic and leaves the module untouched."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)
    enc = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    other = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    d0 = fm.state_digest(enc.state_dict(), torch)
    p0, dtypes = fm.probe_digest(enc, torch)
    assert dtypes == ["float32"] and fm.probe_digest(enc, torch)[0] == p0
    assert fm.state_digest(enc.state_dict(), torch) == d0
    original = fm.swap_in(enc, other.state_dict(), torch)
    p1, _ = fm.probe_digest(enc, torch)
    assert p1 != p0 and p1 == fm.probe_digest(other, torch)[0]
    fm.restore(enc, original, torch)
    assert fm.probe_digest(enc, torch)[0] == p0


# ------------------------------------------------------------------ the registered checkpoints
@pytest.mark.skipif(not SIBLING, reason="stage1/geomem/folds.json is missing")
def test_registered_checkpoint_table():
    assert fm.sha256_file(fm.DEFAULT_FOLDS) == REGISTERED_FOLDS_SHA256
    folds = json.load(open(fm.DEFAULT_FOLDS))
    assert sorted(folds) == sorted(SCENES)
    for sc in SCENES:
        info = fm.resolve("geoset_pred_ft", sc)
        assert info["sha256"] == REGISTERED_SHA256[info["fold"]], sc


@pytest.mark.skipif(not SIBLING, reason="stage1/geomem/folds.json is missing")
def test_registered_checkpoints_pass_the_leak_check_and_load():
    torch = pytest.importorskip("torch")
    build = pytest.importorskip("build_scsam3")
    enc = build._create_tracker_transformer().encoder
    by_fold = {}
    for sc in SCENES:
        info = fm.resolve("geoset_pred_ft", sc)
        if info["fold"] not in by_fold:
            by_fold[info["fold"]] = torch.load(info["path"], map_location="cpu")
        ck = by_fold[info["fold"]]
        scenes = fm.check_checkpoint(ck, info)
        assert sc not in scenes and len(scenes) in (13, 14)
    for k, ck in by_fold.items():
        original = fm.swap_in(enc, ck["memory_attention"], torch)
        assert fm.state_digest(enc.state_dict(), torch) == fm.state_digest(ck["memory_attention"], torch)
        fm.restore(enc, original, torch)
