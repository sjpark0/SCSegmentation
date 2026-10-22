#!/usr/bin/env python3
"""Decode a stage-1 preset on the standard DEV set (geomem/devset.py, docs/stage1-dev-protocol.md) -- never the keyframe.

From the S1-E13 scratch route (MVSeed/refloss/e13/dev-route/dev_decode.py), generalised to any preset of
methods/geomem/stage1.PRESETS and to the DEV-A / DEV-B splits.  decode_scene is used unchanged, given
    (1) common.load_scene(data, scene, frame=t)            -> images / reference GT of frame t (basic set only)
    (2) run.cache_dir patched to Cache/pi3_frames/<t>       -> the frame's Pi3 geometry (the only frame-0 hard-wire)
    (3) scene.cams restricted to {c_ini} + eval cams + the k=3 nearest candidates of each eval cam: round 2 of an eval
        cam reads only its own backbone, the reference memory and those candidates' round-1 memories, so its masks
        equal the full decode's (checked with --validate on each scene's first frame).
Reference rotation (protocol §4, DEV-A' / DEV-B'): --ref-rot r in {1, 2} makes the r-th original eval camera the input
camera (its GT at t, basic set, as c_ini) and the other two annotated cameras the eval cams; the frame's Pi3 cache is
used as is.  Rotated keyframes are TEST and never decoded (no keyframe is decoded here).
S1-E15 recorded override (docs/stage1-E15-prereg.md §4 M2, I8): --e15-arm <arm> with the run.py E15 flags on the base
preset geoset_rep_xs, DEV-A only; the arm name must be the flags' §4 whitelist row and (arm, set, theta, family) a §4 I11
bundle (e15_registered); config.json records e15_arm and every E15 option; default root
Data/_dev/e15/<arm>_<theta>_<family>_<split>[_rot<r>].  Any decode with a Fix A family (override or preset) needs the
committed Step-1 record scripts/e15/runs/select.json (prereg §6.2 L5, §10; run.e15_require_step1).
Scenes: the 17 MUVOD fold scenes (devset.dev_scenes), never the Data/ listing (external scenes such as T06 are not DEV).
Outputs (an absent PNG = empty prediction):
    <root>/<scene>/<t:06d>/<cam>/<obj>.png      eval cams = final masks; c_ini = its GT
    <root>/<scene>/<t:06d>/_r1/<cam>/<obj>.png  round-1 masks of every decoded non-reference cam
    <root>/<scene>/<t:06d>/{config,round1_conf,xcl_conf}.json, DONE;  <root>/index.json
Default root: Data/_dev/<preset>_<split>[_rot<r>].  Runs inside scsam3:
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_decode --preset geoset_rep_x --split A [--scenes A,B] [--validate]
    ... dev_decode --preset geoset_rep_xs --split A --ref-rot 1                                   (DEV-A' C0)
    ... dev_decode --preset geoset_rep_xs --split A --e15-arm AB --relay 35 --id3d vote --xcl outs  (E15 override)
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, S1)
import numpy as np  # noqa: E402

from geomem import devset  # noqa: E402
from geomem.cache import pi3_frame_cache  # noqa: E402
from methods import common  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem.stage1 import FINAL, PRESETS, ROUND1, MemorySink, check_preflight  # noqa: E402
from mvseg2d.io import labelmap_objects, load_labelmap, load_meta, write_mask  # noqa: E402

DATA = os.path.join(S1, "Data")
RUNS = os.path.join(S1, "geomem", "runs")
# S1-E15 recorded override (prereg §4 M2): the base preset, the arms registered on DEV-A / DEV-A' (-> their §4 whitelist
# row; AB(theta) and AB(s') are the AB arm at another theta / family, the stress arm is AB at theta 20), the main preset
# that the second registration commit appends (DEV-B' is decoded only after it exists, prereg §7.1 / §10)
E15_BASE, E15_MAIN = "geoset_rep_xs", "geoset_rep_xs_ab"
E15_DEV_ARMS = {"AB": "AB", "stress": "AB", "A": "A", "B-rep": "B-rep", "B-obj": "B-obj", "G": "G", "GN": "GN"}
E15_STRESS_THETA = R.E15_STRESS_THETA
# §4 I11 (prereg 2nd review): each override arm's registered set, theta and family -- e15_registered.  s*, theta_s*, s'
# and s_main (the main arm's statistic: s*, or s' after a Step-2b swap) are read from scripts/e15/runs/{select,step2,
# step2b}.json, A_E / A_all from angles.json; a bundle that needs one of them is refused until it exists.
E15_RUNS = os.path.join(S1, "scripts", "e15", "runs")
E15_GRID = (35.0, 40.0, 45.0, 50.0, 55.0, 60.0)       # the finite thresholds of Theta (§6.4)


def eval_cams(scene, rot=0):
    return devset.eval_cams(DATA, scene, rot)


def frame_scene(scene_name, t, basic, rot=0):
    """Scene of frame t with the reference GT restricted to the basic set (objects absent from c_ini at t drop out).
    rot > 0: the input camera is devset.ref_cam(rot) (its GT at t, basic set) -- protocol §4."""
    sc = common.load_scene(DATA, scene_name, frame=t)
    if rot:
        ref = devset.ref_cam(DATA, scene_name, rot)
        lab = load_labelmap(os.path.join(common.frame_dir(DATA, scene_name, t), "masks", f"{ref}.png"))
        assert lab.shape == tuple(sc.hw), (scene_name, t, ref, lab.shape, sc.hw)
        sc = replace(sc, ref_cam=ref, ref_labels=lab)
    lab = sc.ref_labels.copy()
    lab[~np.isin(lab, basic)] = 0
    return replace(sc, ref_labels=lab, objects=labelmap_objects(lab))


def e15_selection(runs=None):
    """The selection state the override bundles depend on -> dict(s_star, theta_s, s_prime, s_main, A_E, A_all); None =
    not known yet.  theta_s: from a finished Step 2 only (need_devA / need_zcheck = unfinished; a finished None = inf);
    s_prime: Step 2b's candidate s' (recorded even while its re-checks wait for a decode); s_main: s* when theta_s* = inf,
    else Step 2b's main_family once Step 2b finished."""
    runs = runs or E15_RUNS

    def rd(name):
        p = os.path.join(runs, name)
        if not os.path.isfile(p):
            return None
        with open(p) as f:
            return json.load(f)
    sel, s2, s2b, ang = rd("select.json"), rd("step2.json"), rd("step2b.json"), rd("angles.json")
    out = dict(s_star=((sel or {}).get("s_star") or {}).get("family"), theta_s=None, s_prime=None, s_main=None,
               A_E=None, A_all=None)
    if ang is not None:
        out.update(A_E=float(ang["summary"]["A_E"]["a"]), A_all=float(ang["summary"]["A_all"]["a"]))
    if s2 is not None and not (s2["rule"].get("need_devA") or s2["rule"].get("need_zcheck")):
        th = s2["rule"].get("theta_s")
        out["theta_s"] = math.inf if th in (None, "inf") else float(th)
    if out["theta_s"] is not None and math.isinf(out["theta_s"]):
        out["s_main"] = out["s_star"]
    if s2b is not None:
        to = (s2b.get("rule") or {}).get("to")
        if to:
            out["s_prime"] = s2b["alternatives"][to]["family"]
        if s2b.get("main_family") and not any(s2b.get(k) for k in ("need_devA", "need_devA2")):
            out["s_main"] = s2b["main_family"]
    return out


def e15_registered(arm, rot, theta, family, sel):
    """§4 I11: None when (arm, set, theta, family) is a registered DEV override bundle, else why not.  rot 0 = DEV-A,
    1 / 2 = DEV-A'."""
    s_star, th_s, s_p, s_main, A_E, A_all = (sel[k] for k in ("s_star", "theta_s", "s_prime", "s_main", "A_E", "A_all"))
    prime = rot in (1, 2)
    th_ok = th_s is not None and math.isfinite(th_s)
    if arm == "stress":
        return None if (rot == 0 and theta == E15_STRESS_THETA and family == s_star) else \
            f"stress = DEV-A, theta 20, s* ({s_star}) only"
    if arm in ("G", "GN"):
        return None if (rot == 0 and theta == 60.0 and family == s_star) else f"{arm} = DEV-A, theta 60, s* ({s_star}) only"
    if arm in ("B-rep", "B-obj"):
        if not prime:
            return f"{arm} = DEV-A' (--ref-rot 1, 2) only"
        if not th_ok:
            return f"{arm} runs at theta_s*: Step 2 is not finished or chose inf (runs/step2.json)"
        return None if theta == th_s else f"{arm} = DEV-A' at theta_s* = {th_s:g} only"
    if arm == "A":
        if not prime:
            return "A = DEV-A' (--ref-rot 1, 2) only"
        if s_main is None:
            return "A runs with s_main: Step 2 / 2b are not finished (runs/step2.json, step2b.json)"
        if not (theta == 60.0 or (th_ok and theta == th_s)):
            return f"A = DEV-A' at theta_s* ({th_s}) or 60 only"
        return None if family == s_main else f"A = s_main ({s_main}) only"
    if arm == "AB":
        if prime:
            if theta in E15_GRID and family == s_star:
                return None                                          # Step 2 grid (§6.4)
            if th_ok and theta == th_s and s_p is not None and family == s_p:
                return None                                          # Step 2b re-check of s' (§6.5)
            if theta == 60.0 and s_main is not None and s_main != s_star and family == s_main:
                return None                                          # after a swap: (a) condition / (b) proxy (§6.7)
            return (f"AB on DEV-A' = the finite grid with s* ({s_star}), theta_s* ({th_s}) with s' ({s_p}), or 60 with s' "
                    "after a swap")
        if A_E is None:
            return "AB on DEV-A needs A_E / A_all (runs/angles.json, Step 0)"
        if theta in E15_GRID and theta < A_E and family == s_star:
            return None                                              # Step 2 DEV-A safety (§6.4)
        if th_ok and theta == th_s and th_s < A_E and s_p is not None and family == s_p:
            return None                                              # Step 2b DEV-A safety of s' (§6.5)
        zr = theta == 60.0 or (theta in E15_GRID and A_E <= theta < A_all)
        if zr and (family == s_star or (s_main is not None and family == s_main)):
            return None                                              # Z-R (§6.3b, §6.4)
        return (f"AB on DEV-A = a grid theta < A_E ({A_E:g}) with s* ({s_star}) (theta_s* with s'), or Z-R at 60 / "
                f"[A_E, A_all) with s* or s_main")
    return f"{arm} is not a DEV override arm"


def e15_override(arm, flags, split, rot=0):
    """The recorded E15 override (prereg §4 M2, I8) -> (preset dict = geoset_rep_xs + the given E15 flags, resolved
    options).  Refuses DEV-B (DEV-B / DEV-B' take the main preset only, after the second commit), an arm not registered
    on DEV-A / DEV-A', flags whose §4 whitelist row is not the arm's, the stress threshold under another arm name, and
    a bundle (arm, set = reference rotation `rot`, theta, family) outside the §4 I11 table (e15_registered)."""
    if split != "A":
        raise SystemExit(f"--e15-arm on --split {split}: refused (prereg I8: DEV-B / DEV-B' decode the main preset "
                         f"{E15_MAIN} only, after the second registration commit)")
    if arm not in E15_DEV_ARMS:
        raise SystemExit(f"--e15-arm {arm}: not an arm registered on DEV-A / DEV-A' {sorted(E15_DEV_ARMS)} (prereg §4; "
                         "ABc / Ac / ABe / ABnh / ABml / oracle are T06 only)")
    given = {k: v for k, v in flags.items() if v is not None}
    preset = dict(PRESETS[E15_BASE], **given)
    try:
        o = R.e15_options(SimpleNamespace(**preset))
    except ValueError as exc:
        raise SystemExit(f"--e15-arm {arm}: {exc}")
    if o is None:
        raise SystemExit(f"--e15-arm {arm} needs --relay THETA")
    if o["row"] != E15_DEV_ARMS[arm]:
        raise SystemExit(f"--e15-arm {arm}: the flags are the {o['row']} row of the §4 whitelist, not {E15_DEV_ARMS[arm]}")
    if (arm == "stress") != (o["theta"] == E15_STRESS_THETA):
        raise SystemExit(f"--e15-arm {arm} at --relay {_theta_tag(o['theta'])}: theta {E15_STRESS_THETA:g} is the stress "
                         "arm and only the stress arm (prereg §4)")
    why = e15_registered(arm, rot, o["theta"], o["family"], e15_selection())
    if why:
        where = "DEV-A" if rot == 0 else f"DEV-A' (--ref-rot {rot})"
        raise SystemExit(f"--e15-arm {arm} on {where} at --relay {_theta_tag(o['theta'])} with --id3d {o['family'] or 'off'}: "
                         f"not a registered bundle -- {why} (prereg §4 I11)")
    return preset, o


# S1-E16 (docs/stage1-E16-prereg.md §3): the measurement arms as preset dicts over the committed presets; `_preset` /
# `_e16` are record keys (dropped by decode_scene, which reads known fields only), never flags.
E16_ARMS = {
    "NG-idx": dict(arm="nogeo_pred_ft", gate="obj", xcl="outs", nbr="index"),
    "NG-feat": dict(arm="nogeo_pred_ft", gate="obj", xcl="outs", nbr="feat"),
    "NG-pi3": dict(arm="nogeo_pred_ft", gate="reproj", xcl="outs"),
    "G": dict(PRESETS[E15_BASE]),
    "G-bias0": dict(PRESETS[E15_BASE], zero_bias=True),
    "AB": None,                                                    # PRESETS[E15_MAIN], present after the E15 second commit
}


def preset_runs(runs):
    return os.path.abspath(runs) if runs else RUNS


def seed_of_runs(runs, arm):
    """The training seed recorded in <runs>/<arm>/fold0/config.json (every fold of one runs dir shares it)."""
    try:
        with open(os.path.join(runs, arm, "fold0", "config.json")) as f:
            return int(json.load(f).get("seed", 0))
    except OSError:
        return 0


def e16_preset(name, runs=None):
    p = E16_ARMS[name]
    if name == "AB":
        if E15_MAIN not in PRESETS:
            raise SystemExit(f"--e16-arm AB needs the main preset {E15_MAIN} (S1-E15 second registration commit)")
        p = dict(PRESETS[E15_MAIN])
    p = dict(p, _preset=f"e16:{name}", _e16=name)
    if runs:
        p["runs"] = preset_runs(runs)
    return p


def _theta_tag(theta):
    return "inf" if math.isinf(theta) else f"{theta:g}"


def e15_root(arm, o, split, rot=0):
    """Data/_dev/e15/<arm>_<theta>_<family>_<split>[_rot<r>] (prereg §11.1)."""
    name = f"{arm}_{_theta_tag(o['theta'])}_{o['family'] or 'off'}_{split}" + (f"_rot{rot}" if rot else "")
    return os.path.join(DATA, "_dev", "e15", name)


def e15_record(arm, o):
    """config.json / index.json record of an override run (I8): the arm and every E15 option (strict JSON)."""
    return dict(e15_arm=arm, e15_row=o["row"], e15_options={k: ("inf" if isinstance(v, float) and math.isinf(v) else v)
                                                            for k, v in o.items()})


def restricted_cams(sc, cache, evals, k=R.K_NEIGHBOURS, coords=None):
    """c_ini + the eval cams + each eval cam's k nearest candidates (Pi3 centres, or the S1-E16 coordinates when given)."""
    centres = R.camera_centres(cache, sc.cams) if coords is None else coords
    neigh = {c: [n for n in R.select_candidates(c, centres, k) if n != sc.ref_cam] for c in evals}
    keep = {sc.ref_cam, *evals, *[n for v in neigh.values() for n in v]}
    return [c for c in sc.cams if c in keep], neigh


def e15_may_restrict(preset, sc, cache, evals):
    """S1-E15 I6 (docs/stage1-E15-prereg.md §8): a --relay preset decodes the restricted camera set only in scope R / G with
    no eval cam among the far views F (from the inputs alone); otherwise the full set.  -> (allowed, eval cams in F)."""
    if preset.get("relay") is None:
        return True, []
    masks = np.stack([sc.ref_labels == o for o in sc.objects])
    F = R.relay_geometry(cache, sc.cams, sc.ref_cam, masks, float(preset["relay"]))["F"]
    far = [c for c in evals if c in F]
    return (preset.get("id3d_scope") or "relay") in ("relay", "global") and not far, far


def decode_frame(sc, t, restrict, preset, rot=0):
    """preset: a PRESETS name, or a preset dict (the E15 override).  rot: the reference rotation of `sc` (protocol §4)."""
    p = PRESETS[preset] if isinstance(preset, str) else preset
    cache = pi3_frame_cache(DATA, sc.name, t)
    man = json.load(open(os.path.join(cache, "MANIFEST.json")))
    orig_ref = devset.ref_cam(DATA, sc.name, 0)                 # the cache is the frame's own (original c_ini) one
    assert man["ref_cam"] == orig_ref and (rot or sc.ref_cam == orig_ref), (sc.name, t, man["ref_cam"], orig_ref, sc.ref_cam)
    evals = eval_cams(sc.name, rot)
    assert sc.ref_cam == devset.ref_cam(DATA, sc.name, rot) and sc.ref_cam not in evals, (sc.name, rot, sc.ref_cam, evals)
    coords = None
    if (p.get("nbr") or "pi3") != "pi3":                           # S1-E16: the same neighbour rule as the decode itself
        coords = R.camera_coords(p["nbr"], cache, sc, R.decode_scene.model)
    cams, neigh = restricted_cams(sc, cache, evals, coords=coords)
    if restrict and p.get("relay") is not None:
        restrict, _ = e15_may_restrict(p, sc, cache, evals)
    sc_run = replace(sc, cams=cams) if restrict else sc
    R.cache_dir = lambda data, scene, provider: (pi3_frame_cache(data, scene, t) if provider == "pi3"
                                                 else os.path.join(data, scene, "Cache", provider))
    base = dict(data=DATA, out=FINAL, r1_out=ROUND1, untrained=False, runs=RUNS, tau=R.TAU_CONF, k=R.K_NEIGHBOURS, round2="pred")
    args = SimpleNamespace(**{**base, **p})
    R.check_gate_out(args.arm, args.gate, args.out, args.round2, getattr(args, "zoom", None), getattr(args, "xcl", "none"),
                     zero_bias=getattr(args, "zero_bias", False), relay=getattr(args, "relay", None), nbr=getattr(args, "nbr", "pi3"))
    sink = MemorySink()
    t0 = time.time()
    config = R.decode_scene(sc_run, args, sink=sink)
    check_restricted(sink, sc.name, t, evals, restrict)
    config.update(dict(seconds=time.time() - t0, frame=t, decoded_cams=sc_run.cams, eval_cams=evals, k3=neigh,
                       restricted=bool(restrict), preset=preset if isinstance(preset, str) else p.get("_preset", E15_BASE),
                       objects=sc.objects))
    if p.get("_e16"):                                              # S1-E16: the arm, the checkpoint dir and its training seed
        config.update(e16_arm=p["_e16"], runs=p.get("runs", RUNS))
        try:
            with open(os.path.join(os.path.dirname(config["ckpt"]) if "ckpt" in config else
                                   os.path.join(p.get("runs", RUNS), p["arm"], f"fold{config['fold']}"), "config.json")) as f:
                config["train_seed"] = json.load(f).get("seed")
        except (OSError, KeyError):
            config["train_seed"] = None
    if rot:
        config.update(ref_rot=int(rot), ref_cam=sc.ref_cam, orig_ref_cam=orig_ref)
    return sink, config


def check_restricted(sink, scene, t, evals, restrict):
    """S1-E15 I6: a restricted decode never has an eval cam in F, and only scopes R / G restrict."""
    relay = sink.files.get((FINAL, "relay.json"))
    if relay is not None and restrict:
        assert (relay["options"]["scope"] in ("relay", "global")) and not (set(evals) & set(relay["F"])), \
            (scene, t, relay["options"]["scope"], sorted(set(evals) & set(relay["F"])))


def write_frame(root, sc, t, sink, config):
    d = os.path.join(root, sc.name, f"{t:06d}")
    if os.path.isdir(d):
        shutil.rmtree(d)
    for o in sc.objects:                                        # the reference camera: its GT (copy_reference)
        write_mask(os.path.join(d, sc.ref_cam, f"{o}.png"), sc.ref_labels == o)
    n = 0
    for cam in config["eval_cams"]:
        for o, m in sink.masks.get(FINAL, {}).get(cam, {}).items():
            write_mask(os.path.join(d, cam, f"{int(o)}.png"), m)
            n += 1
    for cam, objs in sink.masks.get(ROUND1, {}).items():
        for o, m in objs.items():
            write_mask(os.path.join(d, "_r1", cam, f"{int(o)}.png"), m)
    for name, obj in (("round1_conf.json", sink.files.get((FINAL, "round1_conf.json"))),
                      ("xcl_conf.json", {c: v for c, v in (sink.files.get((FINAL, "xcl_conf.json")) or {}).items()
                                         if c in config["eval_cams"]}),
                      ("config.json", config)):
        with open(os.path.join(d, name), "w") as f:
            json.dump(obj, f, indent=1)
    for name in ("id3d_conf.json", "relay.json"):               # S1-E15 records (only a --relay preset writes them)
        if sink.files.get((FINAL, name)) is not None:
            with open(os.path.join(d, name), "w") as f:
                json.dump(sink.files[(FINAL, name)], f, indent=1)
    open(os.path.join(d, "DONE"), "w").close()
    return n


def compare(sink_a, sink_b, cams):
    """Byte-level equality of the eval cams' final masks of two decodes (same objects present, same pixels)."""
    bad = []
    for cam in cams:
        a, b = sink_a.masks.get(FINAL, {}).get(cam, {}), sink_b.masks.get(FINAL, {}).get(cam, {})
        if set(a) != set(b):
            bad.append((cam, "objects", sorted(set(a) ^ set(b))))
        for o in set(a) & set(b):
            if not np.array_equal(a[o], b[o]):
                bad.append((cam, o, int((a[o] != b[o]).sum())))
    return bad


def _parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="geoset_rep_x", choices=sorted(PRESETS))
    ap.add_argument("--split", default="A", choices=devset.SPLITS)
    ap.add_argument("--ref-rot", type=int, default=0, choices=devset.ROTS, dest="ref_rot",
                    help="reference rotation (protocol §4): 0 = the original c_ini; 1 / 2 = the first / second original eval "
                         "camera as the input camera (DEV-A' / DEV-B')")
    ap.add_argument("--e15-arm", default=None, dest="e15_arm", choices=sorted(E15_DEV_ARMS),
                    help="S1-E15 recorded override (prereg §4 M2, I8): this arm's run.py flags on --preset geoset_rep_xs, DEV-A only")
    ap.add_argument("--xcl", default=None, choices=R.XCL_MODES,
                    help="with --e15-arm only: must equal the base preset's (outs)")
    R.add_e15_args(ap)
    ap.add_argument("--e16-arm", default=None, dest="e16_arm", choices=sorted(E16_ARMS),
                    help="S1-E16 (docs/stage1-E16-prereg.md): a measurement arm -- NG-idx / NG-feat / NG-pi3 (nogeo_pred_ft), "
                         "G (= geoset_rep_xs), G-bias0 (geoset_rep_xs with the GeoBias zeroed), AB (= geoset_rep_xs_ab); "
                         "with --runs the checkpoints of one training seed; DEV-A / DEV-A' only; root Data/_dev/e16/...")
    ap.add_argument("--runs", default=None, help="S1-E16: the checkpoint dir of one seed (default geomem/runs)")
    ap.add_argument("--seed-tag", default=None, dest="seed_tag", help="S1-E16: the seed label in the root name (default s<seed of --runs>)")
    ap.add_argument("--scenes", default="all", help="'all' = the 17 DEV scenes (devset.dev_scenes), or a subset of them")
    ap.add_argument("--root", default=None, help="default Data/_dev/<preset>_<split>[_rot<r>] (override: Data/_dev/e15/...)")
    ap.add_argument("--validate", action="store_true", help="also decode the FULL camera set on each scene's first dev frame "
                                                            "and require byte-identical eval-cam masks (nothing written for it)")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def prepare(argv=None):
    """Everything before the model loads: the arguments, the preset or the recorded override (registered set / theta),
    the S1-E15 guards (theta 20, the Step-1 commit), the DEV scenes and <root>/index.json.
    -> (args, preset name or dict, override record or None, scenes to decode, index scenes)."""
    ap = _parser()
    a = ap.parse_args(argv)
    flags = {k: getattr(a, k) for k in R.E15_KEYS}
    if a.e16_arm is not None:                                       # S1-E16 measurement arms (no selection, DEV-A / DEV-A')
        if a.e15_arm is not None or any(v not in (None, "off") for v in flags.values()) or a.xcl is not None:
            ap.error("--e16-arm takes no E15 flags / --xcl / --e15-arm (its presets are fixed in E16_ARMS)")
        if a.split != "A":
            ap.error("--e16-arm is a measurement on DEV-A / DEV-A' only (prereg: DEV-B is kept for registrations' confirmations)")
        preset, rec = e16_preset(a.e16_arm, a.runs), dict(e16_arm=a.e16_arm, runs=preset_runs(a.runs))
        tag = a.seed_tag or f"s{seed_of_runs(preset_runs(a.runs), preset['arm'])}"
        p = preset
        a.root = a.root or os.path.join(DATA, "_dev", "e16", f"{a.e16_arm}_{tag}_{a.split}" + (f"_rot{a.ref_rot}" if a.ref_rot else ""))
    elif a.e15_arm is None:
        if any(v not in (None, "off") for v in flags.values()) or a.xcl is not None:
            ap.error("E15 flags / --xcl need --e15-arm (the recorded override); presets carry their own")
        preset, rec, p = a.preset, None, PRESETS[a.preset]
        a.root = a.root or os.path.join(DATA, "_dev", f"{a.preset}_{a.split}" + (f"_rot{a.ref_rot}" if a.ref_rot else ""))
    else:
        if a.preset != E15_BASE:
            ap.error(f"--e15-arm runs on --preset {E15_BASE} only (prereg §4 M2)")
        if a.xcl not in (None, PRESETS[E15_BASE]["xcl"]):
            ap.error(f"--e15-arm: --xcl {a.xcl} differs from the base preset's {PRESETS[E15_BASE]['xcl']}")
        preset, o = e15_override(a.e15_arm, flags, a.split, a.ref_rot)
        rec, p = e15_record(a.e15_arm, o), preset
        a.root = a.root or e15_root(a.e15_arm, o, a.split, a.ref_rot)
    try:                                         # S1-E15: theta 20 / the Step-1 commit for any Fix-A decode (§6.2 L5, §10)
        o = R.e15_options(SimpleNamespace(**p))
        R.e15_cli_check(o)
        if o is not None:
            R.e15_require_step1(o["family"])
    except ValueError as exc:
        ap.error(str(exc))
    if a.split == "B" and a.ref_rot and E15_MAIN not in PRESETS:
        ap.error(f"DEV-B' (--split B --ref-rot {a.ref_rot}) is decoded only after the S1-E15 second registration commit "
                 f"appends {E15_MAIN} (prereg §7.1, §10; protocol §4)")
    dev = devset.dev_scenes()
    scenes = dev if a.scenes == "all" else a.scenes.split(",")
    bad = [s for s in scenes if s not in dev]
    if bad:
        ap.error(f"--scenes {bad}: not DEV scenes (the 17 MUVOD fold scenes of geomem/folds.json)")
    os.makedirs(a.root, exist_ok=True)
    idx = devset.index(DATA, dev, a.split, a.ref_rot)
    with open(os.path.join(a.root, "index.json"), "w") as f:
        json.dump(dict(preset=a.preset, **(rec or {}), **idx), f, indent=1)
    return a, preset, rec, scenes, idx["scenes"]


def _load_model():
    from methods import sam3_single
    model = sam3_single.load_model()
    return model, R.snapshot_init(model)


def main(argv=None):
    a, preset, rec, scenes, index = prepare(argv)
    R.decode_scene.model, R.decode_scene.init_state = _load_model()
    val = {}
    for s in scenes:
        pf = check_preflight(a.preset, s)                       # own-fold checkpoint + leak guard
        basic = index[s]["basic"]
        for i, t in enumerate(index[s]["frames"]):
            assert t != index[s]["ref_frame"]
            done = os.path.join(a.root, s, f"{t:06d}", "DONE")
            if os.path.isfile(done) and not a.overwrite and not (a.validate and i == 0):
                print(f"{s:20s} t={t:4d} skip (DONE)", flush=True)
                continue
            sc = frame_scene(s, t, basic, a.ref_rot)
            sink, config = decode_frame(sc, t, restrict=True, preset=preset, rot=a.ref_rot)
            config.update(fold=pf["fold"], ckpt=pf["checkpoint"], **(rec or {}))
            n = write_frame(a.root, sc, t, sink, config)
            msg = f"{s:20s} t={t:4d} objs {len(sc.objects):2d} cams {len(config['decoded_cams']):2d}/{len(sc.cams):2d} " \
                  f"masks {n:3d} {config['seconds']:6.1f}s"
            if a.validate and i == 0:
                sink_full, cfg_full = decode_frame(sc, t, restrict=False, preset=preset, rot=a.ref_rot)
                bad = compare(sink, sink_full, config["eval_cams"])
                val[f"{s}/{t}"] = dict(equal=not bad, diffs=bad, restricted_s=config["seconds"], full_s=cfg_full["seconds"],
                                       cams_restricted=len(config["decoded_cams"]), cams_full=len(sc.cams))
                msg += f" | full {cfg_full['seconds']:6.1f}s  equal={not bad} {bad[:3]}"
            print(msg, flush=True)
    if val:
        p = os.path.join(a.root, "_validate.json")
        old = json.load(open(p)) if os.path.isfile(p) else {}
        old.update(val)
        with open(p, "w") as f:
            json.dump(old, f, indent=1)
        print("VALIDATE", all(v["equal"] for v in val.values()), flush=True)


if __name__ == "__main__":
    main()
