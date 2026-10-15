#!/usr/bin/env python3
"""Decode a stage-1 preset on the standard DEV set (geomem/devset.py, docs/stage1-dev-protocol.md) -- never the keyframe.

From the S1-E13 scratch route (MVSeed/refloss/e13/dev-route/dev_decode.py), generalised to any preset of
methods/geomem/stage1.PRESETS and to the DEV-A / DEV-B splits.  decode_scene is used unchanged, given
    (1) common.load_scene(data, scene, frame=t)            -> images / reference GT of frame t (basic set only)
    (2) run.cache_dir patched to Cache/pi3_frames/<t>       -> the frame's Pi3 geometry (the only frame-0 hard-wire)
    (3) scene.cams restricted to {c_ini} + eval cams + the k=3 nearest candidates of each eval cam: round 2 of an eval
        cam reads only its own backbone, the reference memory and those candidates' round-1 memories, so its masks
        equal the full decode's (checked with --validate on each scene's first frame).
Outputs (an absent PNG = empty prediction):
    <root>/<scene>/<t:06d>/<cam>/<obj>.png      eval cams = final masks; c_ini = its GT
    <root>/<scene>/<t:06d>/_r1/<cam>/<obj>.png  round-1 masks of every decoded non-reference cam
    <root>/<scene>/<t:06d>/{config,round1_conf,xcl_conf}.json, DONE;  <root>/index.json
Default root: Data/_dev/<preset>_<split>.  Runs inside scsam3:
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_decode --preset geoset_rep_x --split A [--scenes A,B] [--validate]
"""
import argparse
import json
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
from mvseg2d.io import labelmap_objects, load_meta, write_mask  # noqa: E402

DATA = os.path.join(S1, "Data")
RUNS = os.path.join(S1, "geomem", "runs")


def eval_cams(scene):
    return devset.eval_cams(DATA, scene)


def frame_scene(scene_name, t, basic):
    """Scene of frame t with the reference GT restricted to the basic set (objects absent from c_ini at t drop out)."""
    sc = common.load_scene(DATA, scene_name, frame=t)
    lab = sc.ref_labels.copy()
    lab[~np.isin(lab, basic)] = 0
    return replace(sc, ref_labels=lab, objects=labelmap_objects(lab))


def restricted_cams(sc, cache, evals, k=R.K_NEIGHBOURS):
    centres = R.camera_centres(cache, sc.cams)
    neigh = {c: [n for n in R.select_candidates(c, centres, k) if n != sc.ref_cam] for c in evals}
    keep = {sc.ref_cam, *evals, *[n for v in neigh.values() for n in v]}
    return [c for c in sc.cams if c in keep], neigh


def decode_frame(sc, t, restrict, preset):
    cache = pi3_frame_cache(DATA, sc.name, t)
    man = json.load(open(os.path.join(cache, "MANIFEST.json")))
    assert man["ref_cam"] == sc.ref_cam, (sc.name, t, man["ref_cam"], sc.ref_cam)
    evals = eval_cams(sc.name)
    cams, neigh = restricted_cams(sc, cache, evals)
    sc_run = replace(sc, cams=cams) if restrict else sc
    R.cache_dir = lambda data, scene, provider: (pi3_frame_cache(data, scene, t) if provider == "pi3"
                                                 else os.path.join(data, scene, "Cache", provider))
    base = dict(data=DATA, out=FINAL, r1_out=ROUND1, untrained=False, runs=RUNS, tau=R.TAU_CONF, k=R.K_NEIGHBOURS, round2="pred")
    args = SimpleNamespace(**{**base, **PRESETS[preset]})
    R.check_gate_out(args.arm, args.gate, args.out, args.round2, getattr(args, "zoom", None), getattr(args, "xcl", "none"))
    sink = MemorySink()
    t0 = time.time()
    config = R.decode_scene(sc_run, args, sink=sink)
    config.update(dict(seconds=time.time() - t0, frame=t, decoded_cams=sc_run.cams, eval_cams=evals, k3=neigh,
                       restricted=bool(restrict), preset=preset, objects=sc.objects))
    return sink, config


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="geoset_rep_x", choices=sorted(PRESETS))
    ap.add_argument("--split", default="A", choices=devset.SPLITS)
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--root", default=None, help="default Data/_dev/<preset>_<split>")
    ap.add_argument("--validate", action="store_true", help="also decode the FULL camera set on each scene's first dev frame "
                                                            "and require byte-identical eval-cam masks (nothing written for it)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    a.root = a.root or os.path.join(DATA, "_dev", f"{a.preset}_{a.split}")
    scenes = common.all_scenes(DATA) if a.scenes == "all" else a.scenes.split(",")
    os.makedirs(a.root, exist_ok=True)
    idx = devset.index(DATA, common.all_scenes(DATA), a.split)
    index = idx["scenes"]
    with open(os.path.join(a.root, "index.json"), "w") as f:
        json.dump(dict(preset=a.preset, **idx), f, indent=1)
    from methods import sam3_single
    R.decode_scene.model = sam3_single.load_model()
    R.decode_scene.init_state = R.snapshot_init(R.decode_scene.model)
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
            sc = frame_scene(s, t, basic)
            sink, config = decode_frame(sc, t, restrict=True, preset=a.preset)
            config.update(fold=pf["fold"], ckpt=pf["checkpoint"])
            n = write_frame(a.root, sc, t, sink, config)
            msg = f"{s:20s} t={t:4d} objs {len(sc.objects):2d} cams {len(config['decoded_cams']):2d}/{len(sc.cams):2d} " \
                  f"masks {n:3d} {config['seconds']:6.1f}s"
            if a.validate and i == 0:
                sink_full, cfg_full = decode_frame(sc, t, restrict=False, preset=a.preset)
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
