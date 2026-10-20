"""S1-E15 (docs/stage1-E15-prereg.md §3.4-§3.6, §8): the relay round inside the REAL decode_scene, driven on CPU by a fake
SAM 3 wrapper over a synthetic ring (tests/e15_synth.py), plus the GPU byte checks Z4 / Z-R (Fencing keyframe, the §7.2
byte-only exception) and Z-I2 (a DEV-A frame), skipped without a GPU.

The fake decoder mimics the ring collapse: an object's mask copies the identity of the first non-reference memory view
that trusts it (right or swapped: errors propagate); with no such view it is swapped in a `far` view and right elsewhere
(`worse` views: any trusted memory swaps it; `always` views: swapped whatever the memory).  Each output's object pointer carries [camera, memory views, object,
swapped] so the memory encoded from it carries its identity.  Masks round-trip exactly through the decoder's
1008 x 1008 logits (tests/e15_synth.py resolutions), so every PNG the real code writes is predictable.

    scripts/docker_run.sh scsam3 python -m pytest -q tests_torch/test_e15_torch.py       (GPU tests run only with a GPU)
"""
import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import e15_synth as SY  # noqa: E402
from geomem import id3d  # noqa: E402
from geomem.id3d import ABSTAIN, FLAG, PASS  # noqa: E402
from geomem.wrapper import MemorySet, MemoryView  # noqa: E402
from methods import common  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem.stage1 import MemorySink  # noqa: E402

BASE = "f28ea32"          # the last commit before S1-E15: its run.py is the flags-off reference
E15_CONFIG_KEYS = {"relay_theta", "id3d", "id3d_scope", "relay_mode", "relay_order", "relay_mem", "relay_trust", "flagged",
                   "n_F", "n_relay", "n_unreachable", "n_orphan_views", "n_orphan_objects", "n_flag_final", "n_sub_r1",
                   "n_fallback"}


# ------------------------------------------------------------------ pure helpers
def test_mix_takes_s1_per_object():
    X = dict(logits_1008=torch.zeros(3, 1, 4, 4), presence=torch.tensor([1.0, 2, 3]), ious=torch.tensor([.1, .2, .3]),
             obj_ptr=torch.zeros(3, 2))
    S1 = dict(logits_1008=torch.ones(3, 1, 4, 4), presence=torch.tensor([-1.0, -2, -3]), ious=torch.tensor([.7, .8, .9]),
              obj_ptr=torch.ones(3, 2))
    m = R.e15_mix(["X", "S1", "keep"], X, S1)
    assert m["logits_1008"][:, 0, 0, 0].tolist() == [0, 1, 0] and m["presence"].tolist() == [1, -2, 3]
    assert m["ious"].tolist() == pytest.approx([.1, .8, .3]) and m["obj_ptr"][:, 0].tolist() == [0, 1, 0]
    same = R.e15_mix(["X", "chain", "empty"], X, S1)
    assert all(same[k] is X[k] for k in same)


def test_output_logits_fallbacks():
    L = torch.full((3, 4, 4), 5.0)
    chain = torch.zeros(4, 4, dtype=torch.bool)
    chain[:2] = True
    out = R.e15_output_logits(L, ["X", "chain", "empty"], {1: chain})
    assert torch.equal(out[0], L[0]) and torch.all(L == 5.0)                       # input untouched
    assert torch.equal(out[1] > -1, chain) and float(out[1].max()) == R.CHAIN_IN and float(out[1].min()) == R.NO_LOGIT
    assert torch.all(out[2] == R.NO_LOGIT)
    assert R.e15_output_logits(L, ["X", "S1", "keep"]) is L
    keep, _, _ = R.exclusive_masks(out, R.XCL_TAU_KEEP, R.XCL_SHIFT)              # under `outs`: chain in, empty out
    assert torch.equal(keep[1], chain & ~keep[0]) or torch.equal(keep[1], chain)
    assert not keep[2].any()


# ------------------------------------------------------------------ the fake decoder
class FakeWrapper:
    def __init__(self, labels, cams):
        self.labels, self.cams = labels, cams
        self.geo_embed = None
        self.set()

    def set(self, far=(), worse=(), helped_by="any", always=()):
        self.far, self.worse, self.helped_by, self.always = set(far), set(worse), helped_by, set(always)
        self.calls, self.encoded = [], []

    def eval(self):
        return self

    def close(self):
        pass

    def _view(self, name, B, conf=None, ptr=None, presence=None):
        z = torch.zeros(1, B, 1)
        return MemoryView(feats=z, pos=z, obj_ptr_tokens=(torch.zeros(B, 4) if ptr is None else ptr.float())[None],
                          obj_ptr_pos=z, object_score_logits=torch.zeros(B, 1) if presence is None else presence[:, None],
                          p=torch.zeros(1, 3), valid=torch.ones(1, dtype=torch.bool), name=name, conf=conf)

    def encode_reference(self, img, bb, masks, p, v, name="ref"):
        return MemorySet([self._view(name, masks.shape[0])])

    def encode_memory_from_prediction(self, bb, B, logits, presence, obj_ptr, p, v, name, ious=None, tau=None, conf=None):
        self.encoded.append((name, None if conf is None else conf.tolist(), obj_ptr.tolist()))
        return self._view(name, B, conf=conf, ptr=obj_ptr.clone(), presence=presence)

    def _verdict(self, views, b):
        """Identity of the first non-reference memory view trusting object b: True = right, False = swapped, None = none."""
        for v in views[1:]:
            if v.conf is not None and not bool(v.conf[b]):
                continue
            if self.helped_by == "later" and float(v.obj_ptr_tokens[0, b, 1]) <= 1:   # encoded from a round-1 output
                continue
            return float(v.obj_ptr_tokens[0, b, 3]) == 0.0
        return None

    def decode_view(self, bb, B, memory, p, valid, chunk=None, aligned=None, **kw):
        cam, views = bb["cam"], memory.views
        assert views[0].name == "ref" and views[0].conf is None
        self.calls.append((cam, [(v.name, None if v.conf is None else v.conf.tolist()) for v in views]))
        L = self.labels[cam]
        logits, presence, ptr = [], [], []
        for b in range(B):
            o = b + 1
            verdict = self._verdict(views, b)
            if cam in self.always:
                swap = True
            elif cam in self.worse:
                swap = verdict is not None
            else:
                swap = (cam in self.far) if verdict is None else (not verdict)
            m = torch.from_numpy(L == (3 - o if swap else o))
            up = torch.nn.functional.interpolate(m[None, None].float(), size=(1008, 1008), mode="nearest")[0]
            logits.append(torch.where(up > 0, 8.0, -8.0))
            presence.append(5.0 if bool(m.any()) else -5.0)
            ptr.append([float(self.cams.index(cam)), float(len(views)), float(b), float(swap)])
        return dict(logits_1008=torch.stack(logits), presence=torch.tensor(presence), ious=torch.full((B,), 0.9),
                    obj_ptr=torch.tensor(ptr), low_res=None)


@pytest.fixture(scope="module")
def ring(tmp_path_factory):
    return SY.write_scene(str(tmp_path_factory.mktemp("e15dec")))


@pytest.fixture
def env(ring, monkeypatch):
    from geomem import embed, sam3_io, train
    fake = FakeWrapper(ring["labels"], ring["cams"])
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(sam3_io, "load_image", lambda path: (path, SY.H, SY.W))
    monkeypatch.setattr(sam3_io, "backbone_features", lambda model, img: {"cam": os.path.splitext(os.path.basename(img))[0]})
    grid0 = embed.grid_tensors
    monkeypatch.setattr(embed, "grid_tensors", lambda vg, device: grid0(vg, "cpu"))
    monkeypatch.setattr(train, "build_wrapper", lambda model, arm, device="cuda": fake)
    rin0 = R.reproj_inputs
    monkeypatch.setattr(R, "reproj_inputs", lambda cache, cam, ref, hw, device: rin0(cache, cam, ref, hw, "cpu"))
    monkeypatch.setattr(R.decode_scene, "model", object())
    return dict(ring=ring, fake=fake, rin0=rin0)


def _scene(ring):
    return common.Scene(name="Syn", cams=ring["cams"], ref_cam=ring["ref"], mask_cams=ring["cams"], hw=(SY.H, SY.W),
                        image_paths={c: f"/fake/{c}.jpg" for c in ring["cams"]}, ref_labels=ring["labels"][ring["ref"]],
                        objects=[1, 2])


def _decode(env, mod=R, far=(), worse=(), helped_by="any", always=(), **e15):
    env["fake"].set(far, worse, helped_by, always)
    args = argparse.Namespace(data=env["ring"]["data"], out="final", r1_out="round1", arm="geoset_pred_ft", gate="reproj",
                              untrained=True, runs="unused", tau=0.7, k=3, round2="pred", xcl="outs", **e15)
    sink = MemorySink()
    cfg = mod.decode_scene(_scene(env["ring"]), args, sink=sink)
    return sink, cfg


def _masks(sink, key, cams=None):
    return {(c, o): m for c, objs in sink.masks.get(key, {}).items() if cams is None or c in cams for o, m in objs.items()}


def _same(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


def _gt(ring, cam, o):
    return ring["labels"][cam] == o


FAR = [f"cam{i:02d}" for i in range(2, 11)]          # a_v > 60 deg: c_ini-only decodes swap the two objects


def _check_trust(calls, states_of):
    """Every non-reference memory view's conf is the Fix A trust: True only for PASS / ABSTAIN, always for PASS."""
    n = 0
    for cam, views in calls:
        for m, conf in views[1:]:
            st = states_of(m)
            for b, o in enumerate(("1", "2")):
                assert (not conf[b]) or st[o] in (PASS, ABSTAIN), (cam, m, o, st[o])
                assert conf[b] or st[o] != PASS, (cam, m, o, st[o])
                n += 1
    assert n > 0


# ------------------------------------------------------------------ Z4 / Z-R on the fake: flags off and theta beyond every view
def test_flags_off_equals_the_base_commit(env, tmp_path, monkeypatch):
    try:
        src = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASE}:stage1/methods/geomem/run.py"], text=True,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {BASE} unavailable")
    p = tmp_path / "run_base_e15.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("run_base_e15", str(p))
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    monkeypatch.setattr(base, "reproj_inputs", lambda cache, cam, ref, hw, device: env["rin0"](cache, cam, ref, hw, "cpu"))
    monkeypatch.setattr(base.decode_scene, "model", object())
    s0, c0 = _decode(env, base, far=FAR)
    s1, c1 = _decode(env, R, far=FAR)
    assert c1 == c0 and not (set(c1) & E15_CONFIG_KEYS)
    assert _same(_masks(s1, "final"), _masks(s0, "final")) and _same(_masks(s1, "round1"), _masks(s0, "round1"))
    files = [json.dumps({"/".join(k): v for k, v in x.files.items()}, sort_keys=True) for x in (s1, s0)]
    assert files[0] == files[1]
    assert set(k[1] for k in s1.files) == {"round1_conf.json", "xcl_conf.json"}
    # the harness reproduces the collapse: some far view's final mask covers the other object
    assert any(np.array_equal(m, _gt(env["ring"], c, 3 - o)) and m.any() for (c, o), m in _masks(s0, "final").items() if c in FAR)


def test_unregistered_theta_is_refused_by_the_decode(env):
    with pytest.raises(ValueError, match="registered threshold"):
        _decode(env, far=FAR, relay=181.0, id3d="vote")


@pytest.mark.parametrize("theta", [math.inf])
def test_theta_beyond_every_view_is_c0_byte_for_byte(env, theta):
    s0, c0 = _decode(env, far=FAR)
    s1, c1 = _decode(env, far=FAR, relay=theta, id3d="vote")
    assert _same(_masks(s1, "final"), _masks(s0, "final")) and _same(_masks(s1, "round1"), _masks(s0, "round1"))
    for f in ("round1_conf.json", "xcl_conf.json"):
        assert json.dumps(s1.files[("final", f)]) == json.dumps(s0.files[("final", f)]), f
    assert ("final", "id3d_conf.json") not in s1.files                       # F empty in scope R: no Fix A computed
    rel = s1.files[("final", "relay.json")]
    assert rel["F"] == [] and rel["active"] is False and rel["theta"] == ("inf" if math.isinf(theta) else theta)
    assert set(c1) - set(c0) == E15_CONFIG_KEYS and c1["n_F"] == 0 and c1["n_relay"] == 0
    assert {k: c1[k] for k in c0} == c0


# ------------------------------------------------------------------ AB: relay with Fix A (scope R)
def test_relay_invariants_and_the_ring_is_repaired(env, tmp_path):
    ring = env["ring"]
    s0, _ = _decode(env, far=FAR)
    s1, cfg = _decode(env, far=FAR, relay=45.0, id3d="vote")
    rel, rec = s1.files[("final", "relay.json")], s1.files[("final", "id3d_conf.json")]
    S, F = rel["S"], rel["F"]
    assert S == ["cam01", "cam11"] and F == FAR
    # the relay order is the geometric one
    g = R.relay_geometry(ring["cache"], ring["cams"], ring["ref"], np.stack([_gt(ring, ring["ref"], o) for o in (1, 2)]), 45.0)
    seq, unr = id3d.relay_order(S, F, g["angles"], lambda m, v: id3d.hop_angle(g["centres"], g["X"], m, v), 45.0, ring["cams"])
    assert rel["order"] == seq and rel["unreachable"] == unr == []
    # I1: round 1 untouched; I2: the star views' outputs are C0's
    assert _same(_masks(s1, "round1"), _masks(s0, "round1"))
    assert json.dumps(s1.files[("final", "round1_conf.json")]) == json.dumps(s0.files[("final", "round1_conf.json")])
    assert _same(_masks(s1, "final", S), _masks(s0, "final", S))
    # I3: every relay decode reads c_ini + <= 3 already-decoded views within theta, with the registered trust as conf
    decoded = set(S)
    relay_calls = {c: views for c, views in env["fake"].calls if c in F and len(views) > 0}
    for v in rel["order"]:
        info = rel["views"][v]
        neigh = info["neighbours"]
        assert len(neigh) <= 3 and set(neigh) <= decoded and all(h <= 45.0 for m, h, _ in info["candidates"])
        if neigh:
            views = relay_calls[v]
            assert [n for n, _ in views] == ["ref"] + neigh
            for (m, conf) in views[1:]:
                states = rec["round2"][m] if m in S else {o: x["state"] for o, x in rec["final"][m].items()}
                for b, o in enumerate(("1", "2")):
                    assert (not conf[b]) or states[o] in (PASS, ABSTAIN)
                    assert conf[b] or states[o] != PASS
        decoded.add(v)
    # the repaired ring: every far object is right (the C0 decode swapped them)
    fin = _masks(s1, "final", F)
    for c in F:
        for o in (1, 2):
            assert np.array_equal(fin.get((c, o), np.zeros((SY.H, SY.W), bool)), _gt(ring, c, o)), (c, o)
    assert cfg["n_F"] == 9 and cfg["n_relay"] == 9 and cfg["n_unreachable"] == 0 and cfg["n_orphan_views"] == 0
    assert set(cfg) >= E15_CONFIG_KEYS and cfg["relay_theta"] == 45.0 and cfg["id3d"] == "vote"
    # Z-sim: the decode's round-1 states = the host re-judgement of the written round-1 PNGs
    for (c, o), m in _masks(s1, "round1").items():
        os.makedirs(tmp_path / c, exist_ok=True)
        cv2.imwrite(str(tmp_path / c / f"{o}.png"), m.astype(np.uint8) * 255)
    conf = json.loads(json.dumps(s1.files[("final", "round1_conf.json")]))
    host = id3d.rejudge(ring["cache"], ring["ref"], np.stack([_gt(ring, ring["ref"], o) for o in (1, 2)]), [1, 2], "vote",
                        str(tmp_path), conf, [c for c in ring["cams"] if c != ring["ref"]], (SY.H, SY.W))
    assert host == rec["round1"]
    assert any(s == FLAG for c in FAR for s in rec["round1"][c].values())        # the swaps are flagged in round 1


def test_unreachable_far_views_keep_c0(env):
    s0, _ = _decode(env, far=FAR)
    s1, cfg = _decode(env, far=FAR, relay=20.0, id3d="vote")         # S empty: nothing reachable
    rel = s1.files[("final", "relay.json")]
    assert rel["S"] == [] and rel["order"] == [] and len(rel["unreachable"]) == 11 and cfg["n_unreachable"] == 11
    assert _same(_masks(s1, "final"), _masks(s0, "final"))
    assert s1.files[("final", "id3d_conf.json")]["final"] == {}                 # no §3.4 choice on unreachable views


def test_angle_order_relays_every_view_and_orphans_take_round1(env):
    """ABnh at theta 20: S is empty and the angle order has no reachability rule, so every view is relayed from an empty
    pool: an orphan view whose output is its round-1 output (§3.5 step 6)."""
    allv = [c for c in env["ring"]["cams"] if c != env["ring"]["ref"]]
    s1, cfg = _decode(env, far=allv, relay=20.0, id3d="vote", relay_order="angle")
    rel, rec = s1.files[("final", "relay.json")], s1.files[("final", "id3d_conf.json")]
    assert rel["order"][0] in ("cam01", "cam11") and rel["unreachable"] == [] and rel["options"]["row"] == "ABnh"
    assert cfg["n_orphan_views"] == 11 and cfg["n_relay"] == 11
    assert all(v["orphan"] and v["neighbours"] == [] for v in rel["views"].values())
    assert _same(_masks(s1, "final", allv), _masks(s1, "round1", allv))          # every view kept its round-1 output
    assert all(x["choice"] == "keep" and x["x_state"] == FLAG for c in rec["final"].values() for x in c.values())
    assert rec["summary"]["keep_flagged"] == 22 and rec["summary"]["keep_absent"] == 0


def test_empty_fallback_is_never_memory(env):
    """ABe at theta 45 with every far view swapped whatever its memory: X and S1 are both FLAG, so each far object falls
    back to an empty mask that is never trusted as memory (I4)."""
    s1, cfg = _decode(env, always=FAR, relay=45.0, id3d="vote", flagged="empty")
    rel, rec = s1.files[("final", "relay.json")], s1.files[("final", "id3d_conf.json")]
    assert rel["options"]["row"] == "ABe"
    fin = [x for c in FAR for x in rec["final"][c].values()]
    assert fin and all(x["choice"] == "empty" and not x["trust"] and x["state"] is None for x in fin)
    assert not _masks(s1, "final", FAR)                                         # every flagged mask emptied
    assert cfg["n_fallback"] == len(fin) and rec["summary"]["by_choice"] == {"empty": len(fin)}
    assert rec["summary"]["by_choice_x_state"] == {"empty/FLAG": len(fin)}
    enc = [e for e in env["fake"].encoded if e[0] in FAR]       # round-1 memories (reproj gate) first, then the relayed outputs'
    assert len(enc) == 2 * len(FAR) and all(not any(conf) for _, conf, _ in enc[len(FAR):])
    for v, info in rel["views"].items():                                        # untrusted views are never neighbours
        assert not (set(info["neighbours"]) & set(FAR)) and set(info["untrusted"]) <= set(FAR)


def _write_chain(ring, d, skip=()):
    allv = [c for c in ring["cams"] if c != ring["ref"]]
    for c in allv:
        for o in (1, 2):
            if _gt(ring, c, o).any() and c not in skip:
                os.makedirs(d / c, exist_ok=True)
                cv2.imwrite(str(d / c / f"{o}.png"), _gt(ring, c, o).astype(np.uint8) * 255)


def test_chain_fallback_takes_the_chain_mask_and_records_missing_files(env, tmp_path):
    ring = env["ring"]
    _write_chain(ring, tmp_path, skip=("cam05",))                               # cam05's chain masks are missing
    s1, cfg = _decode(env, always=FAR, relay=45.0, id3d="vote", flagged="chain", chain_dir=str(tmp_path))
    rec = s1.files[("final", "id3d_conf.json")]
    assert s1.files[("final", "relay.json")]["options"]["row"] == "ABc"
    fin = _masks(s1, "final", FAR)
    n = n_missing = 0
    for c, objs in rec["final"].items():
        for o, x in objs.items():
            if x["choice"] == "chain":
                expect = np.zeros((SY.H, SY.W), bool) if c == "cam05" else _gt(ring, c, int(o))
                assert np.array_equal(fin.get((c, int(o)), np.zeros((SY.H, SY.W), bool)), expect)
                assert not x["trust"] and x["chain_missing"] == (c == "cam05")
                n += 1
                n_missing += x["chain_missing"]
            else:
                assert "chain_missing" not in x
    assert n > 0 and cfg["n_fallback"] == n and n_missing == 2
    assert rec["chain"]["n_chosen"] == n and rec["chain"]["n_missing"] == 2 and rec["chain"]["n_resolvable"] > 0


def test_chain_dir_that_resolves_nothing_is_an_error(env, tmp_path):
    with pytest.raises(FileNotFoundError, match="no chain PNG resolves"):
        _decode(env, always=FAR, relay=45.0, id3d="vote", flagged="chain", chain_dir=str(tmp_path / "{cam}" / "x{obj}.png"))


# ------------------------------------------------------------------ scopes G / GN, arm A, B-rep, oracle, ABml
def test_global_scope_takes_round1_where_round2_is_flagged(env):
    s0, _ = _decode(env, worse=["cam02"])
    s1, cfg = _decode(env, worse=["cam02"], relay=math.inf, id3d="vote", id3d_scope="global")
    rec = s1.files[("final", "id3d_conf.json")]
    assert all(rec["final"]["cam02"][o]["choice"] == "S1" for o in ("1", "2")) and cfg["n_sub_r1"] == 2
    for o in (1, 2):
        assert np.array_equal(_masks(s1, "final")[("cam02", o)], _gt(env["ring"], "cam02", o))
        assert not np.array_equal(_masks(s0, "final")[("cam02", o)], _gt(env["ring"], "cam02", o))   # C0 kept the swap
    # the round-2 memories carry the Fix A trust of the round-1 outputs (replaces reproj)
    _check_trust(env["fake"].calls, lambda m: rec["round1"][m])


def test_global_nf_redecodes_a_flagged_star_view(env):
    """GN: cam01 swaps with c_ini alone and is only helped by round-2 / relay memories (not round-1 ones), so G keeps
    its FLAG; GN re-decodes it from the other star views' final outputs."""
    sg, cg = _decode(env, far=["cam01"], helped_by="later", relay=math.inf, id3d="vote", id3d_scope="global")
    sn, cn = _decode(env, far=["cam01"], helped_by="later", relay=math.inf, id3d="vote", id3d_scope="global_nf")
    assert cg["n_flag_final"] == 2 and cn["n_flag_final"] == 0
    gn = sn.files[("final", "relay.json")]["gn"]
    assert list(gn) == ["cam01"] and len(gn["cam01"]["neighbours"]) == 3
    for o in (1, 2):
        assert np.array_equal(_masks(sn, "final")[("cam01", o)], _gt(env["ring"], "cam01", o))
        assert not np.array_equal(_masks(sg, "final")[("cam01", o)], _gt(env["ring"], "cam01", o))
    assert _same(_masks(sn, "final", [c for c in env["ring"]["cams"] if c != "cam01"]),
                 _masks(sg, "final", [c for c in env["ring"]["cams"] if c != "cam01"]))


def test_jacobi_arm_reads_fix_a_trust(env):
    s0, _ = _decode(env, far=FAR)
    s1, cfg = _decode(env, far=FAR, relay=45.0, id3d="vote", relay_mode="jacobi")
    rec, rel = s1.files[("final", "id3d_conf.json")], s1.files[("final", "relay.json")]
    assert cfg["n_relay"] == 0 and set(rel["jacobi"]) == set(FAR)
    assert _same(_masks(s1, "final", ["cam01", "cam11"]), _masks(s0, "final", ["cam01", "cam11"]))
    _check_trust([(c, v) for c, v in env["fake"].calls if c in FAR], lambda m: rec["round1"][m])
    # §5.3 record: per neighbour the trusted object ids, as relay views record them; the candidates and untrusted ones
    for v, info in rel["jacobi"].items():
        assert set(info["trusted"]) == set(info["neighbours"]) and set(info["untrusted"]) == set(info["candidates"]) - set(info["neighbours"])
        for m, objs in info["trusted"].items():
            assert objs == [o for o in (1, 2) if rec["round1"][m][str(o)] in (PASS, ABSTAIN)] and objs
    c = rel["counts"]["jacobi"]
    assert c["n_views"] == len(FAR) and c["n_candidates"] == c["n_untrusted"] + c["n_neighbours"]


def test_b_rep_relays_without_fix_a(env):
    s1, cfg = _decode(env, far=FAR, relay=45.0)
    assert ("final", "id3d_conf.json") not in s1.files and cfg["id3d"] == "off" and cfg["relay_trust"] == "reproj"
    assert cfg["n_sub_r1"] == cfg["n_fallback"] == cfg["n_flag_final"] == 0 and cfg["n_relay"] == 9


def test_oracle_uses_gt_identity(env, tmp_path):
    ring = env["ring"]
    for c, L in ring["labels"].items():
        cv2.imwrite(str(tmp_path / f"{c}.png"), L)
    s1, cfg = _decode(env, far=FAR, relay=45.0, relay_trust="gt", gt_dir=str(tmp_path))
    rec = s1.files[("final", "id3d_conf.json")]
    assert rec["status_source"] == "gt" and rec["round1"] == {} and "gt_round1" in rec
    assert all(rec["gt_round1"][c][o] == FLAG for c in FAR for o in ("1", "2") if _gt(ring, c, int(o)).any()
               and _gt(ring, c, 3 - int(o)).any())
    fin = _masks(s1, "final", FAR)
    assert all(np.array_equal(fin.get((c, o), np.zeros((SY.H, SY.W), bool)), _gt(ring, c, o)) for c in FAR for o in (1, 2))


def test_latest_memory_offers_passing_round1_outputs(env):
    """ABml (review 0 #2): only cam05 swaps in round 1.  A not-yet-relayed far view's round-1 output is a relay memory
    exactly where Fix A PASSes: the first relayed view gets its unrelayed far neighbour's round-1 output (PASS) as a
    trusted memory, which AB never offers; cam05's round-1 output (FLAG) is a candidate of the views relayed before it
    but untrusted, so it is never their neighbour."""
    sa, _ = _decode(env, far=["cam05"], relay=45.0, id3d="vote")
    sl, cl = _decode(env, far=["cam05"], relay=45.0, id3d="vote", relay_mem="latest")
    calls_ml = list(env["fake"].calls)
    rel_a, rel_l = sa.files[("final", "relay.json")], sl.files[("final", "relay.json")]
    rec_l = sl.files[("final", "id3d_conf.json")]
    assert cl["relay_mem"] == "latest" and rel_l["options"]["row"] == "ABml"
    assert rec_l["round1"]["cam05"] == {"1": FLAG, "2": FLAG}
    assert all(rec_l["round1"][c] == {"1": PASS, "2": PASS} for c in FAR if c != "cam05")
    order = rel_l["order"]
    assert order == rel_a["order"] and order.index("cam04") < order.index("cam05")
    S = rel_l["S"]
    # the first relayed view: AB's neighbours are star views only; ABml adds a far round-1 output (trusted, PASS)
    u = order[0]
    assert set(rel_a["views"][u]["neighbours"]) <= set(S)
    extra = [m for m in rel_l["views"][u]["neighbours"] if m in FAR]
    assert len(extra) == 1 and extra[0] not in order[:1] and rel_l["views"][u]["trusted"][extra[0]] == [1, 2]
    call = [views for cam, views in calls_ml if cam == u and len(views) > 1][-1]
    assert [n for n, _ in call] == ["ref"] + rel_l["views"][u]["neighbours"]
    assert dict(call[1:])[extra[0]] == [True, True]
    # before cam05 is relayed, its round-1 output is offered (a candidate within the hop) but never trusted
    seen = 0
    for v in order[:order.index("cam05")]:
        info = rel_l["views"][v]
        if "cam05" in [m for m, _, _ in info["candidates"]]:
            assert "cam05" in info["untrusted"] and "cam05" not in info["neighbours"]
            assert "cam05" not in [m for m, _, _ in rel_a["views"][v]["candidates"]]          # AB: not decoded yet
            seen += 1
    assert seen >= 1


def test_rig_distrust_turns_the_relay_off(env, monkeypatch):
    """A-c* with a finite phi_rig the scene exceeds (forced: phi_rig -1): every Fix A state is ABSTAIN, no view is
    relayed and the decode is C0's; the rig record says where its cameras came from (review 0 #7, #9)."""
    monkeypatch.setattr(id3d, "CSTAR_SELECTED", True)
    monkeypatch.setitem(id3d.CSTAR, "phi_rig", -1.0)
    s0, _ = _decode(env, far=FAR)
    s1, cfg = _decode(env, far=FAR, relay=45.0, id3d="cstar")
    rel, rec = s1.files[("final", "relay.json")], s1.files[("final", "id3d_conf.json")]
    assert rel["rig"]["ok"] is False and rel["rig"]["cams_source"] == "meta" and rel["rig"]["n_cams"] == 12
    assert rel["order"] == [] and rel["views"] == {} and cfg["n_relay"] == 0 and rec["rig_ok"] is False
    assert all(st in (ABSTAIN, id3d.EMPTY) for c in rec["round1"].values() for st in c.values())
    assert _same(_masks(s1, "final"), _masks(s0, "final"))
    assert json.dumps(s1.files[("final", "xcl_conf.json")]) == json.dumps(s0.files[("final", "xcl_conf.json")])
    # the meta file unreadable: the decoded cameras are used and the record says so
    monkeypatch.setattr(common, "load_meta", lambda data, name: (_ for _ in ()).throw(OSError("gone")))
    s2, _ = _decode(env, far=FAR, relay=45.0, id3d="cstar")
    rig = s2.files[("final", "relay.json")]["rig"]
    assert rig["cams_source"].startswith("decoded cams (meta unreadable: OSError") and rig["n_cams"] == 12


def test_hybrid_statistic_inside_the_decode(env):
    """A-h (review 0 #9): reads the reprojection record of each output; on the synthetic ring it repairs the far views."""
    ring = env["ring"]
    s1, cfg = _decode(env, far=FAR, relay=45.0, id3d="hybrid")
    rec = s1.files[("final", "id3d_conf.json")]
    assert rec["family"] == "hybrid" and all("vis" in x for c in rec["round1_info"].values() for x in c.values() if x.get("n"))
    fin = _masks(s1, "final", FAR)
    for c in FAR:
        for o in (1, 2):
            assert np.array_equal(fin.get((c, o), np.zeros((SY.H, SY.W), bool)), _gt(ring, c, o)), (c, o)
    assert any(s == FLAG for c in FAR for s in rec["round1"][c].values())


def test_global_nf_view_without_a_trusted_neighbour_is_not_redecoded(env):
    """GN (review 0 #9): every view swaps and no memory helps, so every star view keeps a FLAG and the pool trusts
    nothing: each flagged view is an orphan, nothing is re-decoded, the output is G's."""
    allv = [c for c in env["ring"]["cams"] if c != env["ring"]["ref"]]
    sg, cg = _decode(env, far=allv, helped_by="later", relay=math.inf, id3d="vote", id3d_scope="global")
    sn, cn = _decode(env, far=allv, helped_by="later", relay=math.inf, id3d="vote", id3d_scope="global_nf")
    gn = sn.files[("final", "relay.json")]["gn"]
    assert set(gn) == set(allv) and all(r.get("orphan") and r["neighbours"] == [] for r in gn.values())
    assert _same(_masks(sn, "final"), _masks(sg, "final")) and cn["n_flag_final"] == cg["n_flag_final"] > 0
    assert not any(x.get("gn") for c in sn.files[("final", "id3d_conf.json")]["final"].values() for x in c.values())
    assert sn.files[("final", "relay.json")]["counts"]["gn"]["n_neighbours"] == 0


def test_no_valid_cini_point_decodes_c0(env, tmp_path):
    """Review 0 #6: c_ini without a valid Pi3 point -> no anchor; every scope decodes C0 and records why (no crash)."""
    ring = SY.write_scene(str(tmp_path))
    p = os.path.join(ring["cache"], f"{ring['ref']}.npz")
    z = dict(np.load(p))
    z["conf"] = np.zeros_like(z["conf"])
    np.savez(p, **z)
    env2 = dict(env, ring=ring)
    env["fake"].labels = ring["labels"]
    s0, c0 = _decode(env2, far=FAR)
    for kw in (dict(relay=20.0, id3d="vote"), dict(relay=math.inf, id3d="vote", id3d_scope="global"),
               dict(relay=60.0, id3d="vote", id3d_scope="global_nf")):
        s1, c1 = _decode(env2, far=FAR, **kw)
        rel = s1.files[("final", "relay.json")]
        assert rel["active"] is False and rel["anchor"]["X"] is None and "no valid Pi3" in rel["fallback"]
        assert ("final", "id3d_conf.json") not in s1.files and c1["n_F"] == 0
        assert _same(_masks(s1, "final"), _masks(s0, "final"))
        json.dumps(rel, allow_nan=False)


STRICT = {
    "AB": dict(relay=45.0, id3d="vote"), "AB-c0": dict(relay=45.0, id3d="c0"), "AB-hybrid": dict(relay=45.0, id3d="hybrid"),
    "AB-vote_inf": dict(relay=60.0, id3d="vote_inf"), "A": dict(relay=45.0, id3d="vote", relay_mode="jacobi"),
    "ABml": dict(relay=45.0, id3d="vote", relay_mem="latest"), "ABnh": dict(relay=20.0, id3d="vote", relay_order="angle"),
    "ABe": dict(relay=45.0, id3d="vote", flagged="empty"), "G": dict(relay=math.inf, id3d="vote", id3d_scope="global"),
    "GN": dict(relay=60.0, id3d="vote", id3d_scope="global_nf"), "B-rep": dict(relay=45.0),
    "B-obj": dict(relay=45.0, relay_trust="obj"), "unreachable": dict(relay=20.0, id3d="vote"),
}


@pytest.mark.parametrize("arm", sorted(STRICT))
def test_e15_records_are_strict_json(env, arm, monkeypatch):
    """Review 0 #9: relay.json, id3d_conf.json and the config are strict JSON (no NaN / Infinity) for every arm."""
    s1, cfg = _decode(env, far=FAR, **STRICT[arm])
    for name in ("relay.json", "id3d_conf.json"):
        if ("final", name) in s1.files:
            json.loads(json.dumps(s1.files[("final", name)], allow_nan=False))
    json.dumps(cfg, allow_nan=False)
    assert s1.files[("final", "relay.json")]["options"]["row"] == ("AB" if arm in ("AB-c0", "AB-hybrid", "AB-vote_inf",
                                                                                     "unreachable") else arm)


def test_cstar_records_are_strict_json(env, monkeypatch):
    monkeypatch.setattr(id3d, "CSTAR_SELECTED", True)
    for k, v in dict(SY.P1_CSTAR, phi_rig=0.12).items():         # pinned: unchanged by the Step-1 commit
        monkeypatch.setitem(id3d.CSTAR, k, v)
    s1, cfg = _decode(env, far=FAR, relay=45.0, id3d="cstar")
    for name in ("relay.json", "id3d_conf.json"):
        json.dumps(s1.files[("final", name)], allow_nan=False)
    json.dumps(cfg, allow_nan=False)
    assert s1.files[("final", "id3d_conf.json")]["consts"]["gamma"] == "inf"


def test_t06_needs_the_fold2_checkpoint(env):
    """I9: an E15 decode of T06 is refused without its fold-2 checkpoint (here: --untrained)."""
    sc = replace(_scene(env["ring"]), name="T06_flag_ganggang_F")
    args = argparse.Namespace(data=env["ring"]["data"], out="final", r1_out="round1", arm="geoset_pred_ft", gate="reproj",
                              untrained=True, runs="unused", tau=0.7, k=3, round2="pred", xcl="outs", relay=60.0, id3d="vote")
    with pytest.raises(ValueError, match="fold-2"):
        R.decode_scene(sc, args, sink=MemorySink())


# ------------------------------------------------------------------ GPU: Z4 / Z-R on a real keyframe decode
DATA = os.path.join(ROOT, "Data")
RUNS = os.path.join(ROOT, "geomem", "runs")
E14 = os.path.join(DATA, "Fencing", "Result", "geoset_pred_ft_rep_xs_e14")
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir(E14) or not os.path.isdir(os.path.join(RUNS, "geoset_pred_ft")),
    reason="needs GPU, the geoset_pred_ft checkpoints and Data/Fencing/Result/geoset_pred_ft_rep_xs_e14")


def _pngs(d, skip=()):
    out = {}
    for cam in sorted(os.listdir(d)):
        if cam in skip or not os.path.isdir(os.path.join(d, cam)):
            continue
        for f in sorted(os.listdir(os.path.join(d, cam))):
            if f.endswith(".png"):
                out[(cam, f)] = open(os.path.join(d, cam, f), "rb").read()
    return out


@needs_gpu
@pytest.mark.parametrize("relay", [None, 60.0])
def test_z4_zr_fencing_keyframe_bytes(relay):
    """Z4 (flags off) and Z-R (relay 60: every MUVOD camera is <= 58 deg, F empty): final and round-1 PNG bytes,
    round1_conf.json and xcl_conf.json equal the S1-E14 keyframe decode (bytes only, not scored)."""
    from methods import sam3_single
    if R.decode_scene.model is None:
        R.decode_scene.model = sam3_single.load_model()
        R.decode_scene.init_state = R.snapshot_init(R.decode_scene.model)
    scene = common.load_scene(DATA, "Fencing")
    out = "dbg_e15_z4" if relay is None else "dbg_e15_zr"
    for d in (out, out + "_r1"):
        shutil.rmtree(os.path.join(DATA, "Fencing", "Result", d), ignore_errors=True)
    kw = {} if relay is None else dict(relay=relay, id3d="vote")
    args = argparse.Namespace(data=DATA, out=out, r1_out=out + "_r1", arm="geoset_pred_ft", gate="reproj", untrained=False,
                              runs=RUNS, tau=R.TAU_CONF, k=R.K_NEIGHBOURS, round2="pred", xcl="outs", **kw)
    cfg = R.decode_scene(scene, args)
    d = os.path.join(DATA, "Fencing", "Result", out)
    r1_e14 = os.path.join(DATA, "Fencing", "Result", "geoset_pred_ft_r1_xs_e14")
    assert _pngs(d, skip=(scene.ref_cam,)) == _pngs(E14, skip=(scene.ref_cam,))
    assert _pngs(d + "_r1", skip=(scene.ref_cam,)) == _pngs(r1_e14, skip=(scene.ref_cam,))
    for f in ("round1_conf.json", "xcl_conf.json"):
        assert open(os.path.join(d, f), "rb").read() == open(os.path.join(E14, f), "rb").read(), f
    assert (relay is None) == ("n_F" not in cfg) and cfg.get("n_F", 0) == 0


# ------------------------------------------------------------------ GPU: Z-I2 on a DEV-A frame (review 0 #3, prereg §8)
DEV_A = os.path.join(DATA, "_dev", "geoset_rep_xs_A")
SELECT = os.path.join(ROOT, "scripts", "e15", "runs", "select.json")
needs_gpu_dev = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir(DEV_A) or not os.path.isdir(os.path.join(RUNS, "geoset_pred_ft")),
    reason="needs GPU, the geoset_pred_ft checkpoints and Data/_dev/geoset_rep_xs_A")


def _zi2_frame():
    """The first DEV-A frame (scene order, frame order) whose full camera set at theta 20 has an eval cam among the star
    views S and a non-empty F (inputs only: c_ini masks + Pi3 poses)."""
    from geomem import devset
    from geomem.cache import pi3_frame_cache
    from methods.geomem import dev_decode as D
    for s in devset.dev_scenes():
        if not os.path.isdir(os.path.join(DEV_A, s)):
            continue
        t = devset.dev_frames(DATA, s, "A")[0]
        sc = D.frame_scene(s, t, devset.basic_set(s))
        g = R.relay_geometry(pi3_frame_cache(DATA, s, t), sc.cams, sc.ref_cam,
                             np.stack([sc.ref_labels == o for o in sc.objects]), 20.0)
        star = [c for c in devset.eval_cams(DATA, s) if c in g["S"]]
        if star and g["F"]:
            return sc, t, star, g
    pytest.skip("no DEV-A frame with an eval cam in S and F non-empty at theta 20")


@needs_gpu_dev
@pytest.mark.parametrize("arm", ["B-rep", "stress"])
def test_zi2_star_views_unchanged_on_a_dev_a_frame(arm, tmp_path, monkeypatch):
    """Z-I2 / I2 with the real model and far views present (scope R, theta 20, FULL camera set): the eval cams in S
    keep C0's final PNG bytes and every round-1 PNG / round1_conf entry of Data/_dev/geoset_rep_xs_A is unchanged, while
    the relay round (pool encodes between the round-2 decodes, relay decodes after them) runs.  Bytes only, never scored.
    The Fix-A arm (stress, A-v) runs only after the Step-1 commit (prereg §10: no Fix-A GPU decode before it)."""
    fam = None
    if arm == "stress":                                   # s*, and only after the Step-1 commit (run.e15_require_step1)
        try:
            fam = json.load(open(SELECT))["s_star"]["family"]
            R.e15_require_step1(fam)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            pytest.skip(f"Fix-A GPU decodes wait for the Step-1 commit (prereg §6.2 / §10): {exc}")
    from methods import sam3_single
    from methods.geomem import dev_decode as D
    monkeypatch.setattr(R, "cache_dir", R.cache_dir)                  # decode_frame re-points it to the frame cache
    if R.decode_scene.model is None:
        R.decode_scene.model = sam3_single.load_model()
        R.decode_scene.init_state = R.snapshot_init(R.decode_scene.model)
    sc, t, star, g = _zi2_frame()
    flags = dict(relay=20.0) if arm == "B-rep" else dict(relay=20.0, id3d=fam)
    preset = dict(D.PRESETS["geoset_rep_xs"], **flags)
    assert R.e15_options(SimpleNamespace(**preset))["row"] == ("B-rep" if arm == "B-rep" else "AB")
    sink, cfg = D.decode_frame(sc, t, restrict=False, preset=preset)
    assert cfg["n_relay"] > 0 and not cfg["restricted"]
    D.write_frame(str(tmp_path), sc, t, sink, cfg)
    ours, base = tmp_path / sc.name / f"{t:06d}", os.path.join(DEV_A, sc.name, f"{t:06d}")
    for cam in star:                                                  # I2: eval cams in S = C0 bytes
        assert _pngs(str(ours), skip=[c for c in os.listdir(ours) if c != cam]) == \
            _pngs(base, skip=[c for c in os.listdir(base) if c != cam]), cam
    r1_base = _pngs(os.path.join(base, "_r1"))
    r1_ours = _pngs(str(ours / "_r1"))
    assert r1_base and {k: r1_ours.get(k) for k in r1_base} == r1_base   # I1 on the C0 round-1 cams
    conf_base = json.load(open(os.path.join(base, "round1_conf.json")))
    conf_ours = json.load(open(ours / "round1_conf.json"))
    assert {c: conf_ours[c] for c in conf_base} == conf_base
    rel = json.load(open(ours / "relay.json"))
    assert set(star) <= set(rel["S"]) and rel["order"]
