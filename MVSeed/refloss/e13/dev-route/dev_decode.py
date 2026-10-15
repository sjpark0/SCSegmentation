#!/usr/bin/env python3
"""S1-E13 DEV route: the adopted stage-1 preset (geoset_rep_x = arm geoset_pred_ft, --gate reproj --xcl out, held-out
fold checkpoint) decoded on NON-keyframe frames t of every scene (scratch wrapper, no tracked change).

DEV set (fixed before any decode; the keyframe = meta.json ref_frame is the TEST set and is never decoded here):
    frames   t in geomem.cache.train_frames(scene) minus the ref frame, every 4th: ref+4, ref+8, ..., ref+20 (5 / scene)
    cameras  the 2 annotated non-reference cameras (meta mask_cams minus ref_cam)
    input    the reference camera c_ini (= meta ref_cam) at frame t, reference mask = c_ini GT at t restricted to the
             scene's basic object set (docs/raw/muvod_object_sets.json seed_ids_c_ini; the keyframe protocol's objects)
    inputs   images  stage1/Data/<scene>/frames/<t>/images/<cam>.jpg, GT stage1/Data/<scene>/frames/<t>/masks/<cam>.png,
             Pi3     stage1/Data/<scene>/Cache/pi3_frames/<t>/   (same layout as Cache/pi3; byte-identical to it at t = ref)

How: methods.geomem.run.decode_scene unchanged, given
    (1) common.load_scene(data, scene, frame=t)            -> images / reference GT of frame t
    (2) run.cache_dir patched to Cache/pi3_frames/<t>       -> the frame's Pi3 geometry (the only frame-0 hard-wire)
    (3) scene.cams restricted to {c_ini} + eval cams + the k=3 nearest candidates of each eval cam (by the frame's Pi3
        centres, reference excluded): round 2 of an eval cam reads only its own backbone, the reference memory and the
        round-1 memories of those candidates, so its masks are those of the full decode (checked with --validate).
Outputs (Result/<out> layout per frame; an absent PNG = empty prediction):
    <root>/<scene>/<t:06d>/<cam>/<obj>.png      eval cams = round-2 final masks; c_ini = its GT (as copy_reference)
    <root>/<scene>/<t:06d>/_r1/<cam>/<obj>.png  round-1 masks of every decoded non-reference cam (= full-decode round 1)
    <root>/<scene>/<t:06d>/{config,round1_conf,xcl_conf}.json, DONE
    <root>/index.json                           the dev set definition

    flock <lock> stage1/scripts/docker_run.sh scsam3 python geomem/runs/_e13_dev-route/dev_decode.py \
        [--scenes A,B] [--root geomem/runs/_e13_dev] [--validate] [--overwrite]
"""
import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, S1)
import numpy as np  # noqa: E402

from geomem.cache import pi3_frame_cache, train_frames  # noqa: E402
from methods import common  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem.stage1 import FINAL, PRESETS, ROUND1, MemorySink, check_preflight  # noqa: E402
from mvseg2d.io import labelmap_objects, load_meta, write_mask  # noqa: E402

PRESET = "geoset_rep_x"
DATA = os.path.join(S1, "Data")
RUNS = os.path.join(S1, "geomem", "runs")
OBJSETS = os.path.join(os.path.dirname(S1), "docs", "raw", "muvod_object_sets.json")
STRIDE, OFFSET = 4, 3          # non-ref train frames [OFFSET::STRIDE] = ref+4, ref+8, ..., ref+20


def dev_frames(scene):
    ref = int(load_meta(DATA, scene)["ref_frame"])
    nonref = [t for t in train_frames(DATA, scene) if t != ref]
    return nonref[OFFSET::STRIDE]


def eval_cams(scene):
    m = load_meta(DATA, scene)
    return [c for c in m["mask_cams"] if c != m["ref_cam"]]


def basic_set(scene):
    return [int(o) for o in json.load(open(OBJSETS))[scene]["seed_ids_c_ini"]]


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


def decode_frame(sc, t, restrict):
    cache = pi3_frame_cache(DATA, sc.name, t)
    man = json.load(open(os.path.join(cache, "MANIFEST.json")))
    assert man["ref_cam"] == sc.ref_cam, (sc.name, t, man["ref_cam"], sc.ref_cam)
    evals = eval_cams(sc.name)
    cams, neigh = restricted_cams(sc, cache, evals)
    sc_run = replace(sc, cams=cams) if restrict else sc
    R.cache_dir = lambda data, scene, provider: (pi3_frame_cache(data, scene, t) if provider == "pi3"
                                                 else os.path.join(data, scene, "Cache", provider))
    base = dict(data=DATA, out=FINAL, r1_out=ROUND1, untrained=False, runs=RUNS, tau=R.TAU_CONF, k=R.K_NEIGHBOURS, round2="pred")
    args = SimpleNamespace(**{**base, **PRESETS[PRESET]})
    R.check_gate_out(args.arm, args.gate, args.out, args.round2, getattr(args, "zoom", None), getattr(args, "xcl", "none"))
    sink = MemorySink()
    t0 = time.time()
    config = R.decode_scene(sc_run, args, sink=sink)
    config.update(dict(seconds=time.time() - t0, frame=t, decoded_cams=sc_run.cams, eval_cams=evals, k3=neigh,
                       restricted=bool(restrict), preset=PRESET, objects=sc.objects))
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
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--root", default=os.path.join(RUNS, "_e13_dev"))
    ap.add_argument("--validate", action="store_true", help="also decode the FULL camera set on each scene's first dev frame "
                                                            "and require byte-identical eval-cam masks (nothing written for it)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    scenes = common.all_scenes(DATA) if a.scenes == "all" else a.scenes.split(",")
    os.makedirs(a.root, exist_ok=True)
    index = {s: dict(ref_frame=int(load_meta(DATA, s)["ref_frame"]), ref_cam=load_meta(DATA, s)["ref_cam"],
                     frames=dev_frames(s), eval_cams=eval_cams(s), basic=basic_set(s)) for s in common.all_scenes(DATA)}
    with open(os.path.join(a.root, "index.json"), "w") as f:
        json.dump(dict(preset=PRESET, stride=STRIDE, offset=OFFSET, rule="non-ref train frames [3::4]", scenes=index), f, indent=1)
    from methods import sam3_single
    R.decode_scene.model = sam3_single.load_model()
    R.decode_scene.init_state = R.snapshot_init(R.decode_scene.model)
    val = {}
    for s in scenes:
        pf = check_preflight(PRESET, s)                         # own-fold checkpoint + leak guard
        basic = basic_set(s)
        for i, t in enumerate(index[s]["frames"]):
            assert t != index[s]["ref_frame"]
            done = os.path.join(a.root, s, f"{t:06d}", "DONE")
            if os.path.isfile(done) and not a.overwrite and not (a.validate and i == 0):
                print(f"{s:20s} t={t:4d} skip (DONE)", flush=True)
                continue
            sc = frame_scene(s, t, basic)
            sink, config = decode_frame(sc, t, restrict=True)
            config.update(fold=pf["fold"], ckpt=pf["checkpoint"])
            n = write_frame(a.root, sc, t, sink, config)
            msg = f"{s:20s} t={t:4d} objs {len(sc.objects):2d} cams {len(config['decoded_cams']):2d}/{len(sc.cams):2d} " \
                  f"masks {n:3d} {config['seconds']:6.1f}s"
            if a.validate and i == 0:
                sink_full, cfg_full = decode_frame(sc, t, restrict=False)
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
