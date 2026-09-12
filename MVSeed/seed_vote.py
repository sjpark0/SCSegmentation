#!/usr/bin/env python3
"""S1-E6 SeedVote: pick a seed per (view, object) from seven candidates without ground truth.

Rules are docs/stage1-plan.md section 4 (selection) and section 5 (judgment), fixed before this file
existed; docs/stage1-E6-prereg.md fixes the stream definitions, the three signals, the tie-break and
the Delta-J -> J&F^3 conversion.  Nothing here may read a scored camera's ground truth: the only
ground truth used is c_ini's, which is the method's input.

    # 1) signals (GPU: the S4 pointer needs one SAM 3 session per camera)
    docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g \\
      --user $(id -u):$(id -g) -v /:/host -w /host$PWD \\
      -e HF_HOME=/host$PWD/SCSam3/hf_cache -e HF_HUB_OFFLINE=1 scsam3 \\
      python MVSeed/seed_vote.py signals
    # 2) vote (CPU) -> Data/MVSeg/<folder>/MVSeed_e6_vote/
    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \\
      python MVSeed/seed_vote.py vote
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "SCSam3", "demoSCSam3MVOpt")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load("MVSeed/gate_det.py", "gate_det")
DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
CACHE = os.path.join(REPO, "MVSeed", "cache")
SIGNALS = os.path.join(REPO, "MVSeed", "runs", "e6_signals.json")
SIG_PARTS = os.path.join(REPO, "MVSeed", "runs", "e6_signals_parts")
OUT_VOTE = os.path.join(REPO, "MVSeed", "runs", "e6_vote.json")
SEED_TAG = "MVSeed_e6_vote"          # R2: MVSeed writes only into MVSeed_-prefixed folders
BASE = "MVSeed_e0_index"             # G0

# registered stream order; it is also the tie-break order after S4 (prereg section 2)
STREAMS = [("G0", "seed", BASE), ("G1", "seed", "MVSeed_e1_reverse"),
           ("Gdet", "cache", "cand_det"), ("Gcorr", "cache", "cand_corr"),
           ("Ggeo_a", "cache", "proj_a"), ("Ggeo_b", "cache", "proj_b"),
           ("Ggeo_c", "cache", "proj_c")]
NAMES = [s[0] for s in STREAMS]
CAND = [n for n in NAMES if n != "G0"]
GEO_SELF = ("Ggeo_a", "Ggeo_b")      # these two ARE the S2 projection -> they may not use S2 (prereg)
TAU = 0.5                            # every threshold of the registered rule
OVERLAP_TAU = 0.5                    # a candidate overlapping another object's G0 by more than this
EMPTY_MIN_AGREE = {"pinhole": 2, "hemisphere": 3}


def mask_path(kind, name, scene, folder, cam, start, obj):
    return (os.path.join(DATA, folder, name, cam, str(start), f"{obj}.png") if kind == "seed"
            else os.path.join(CACHE, name, scene, cam, f"{obj}.png"))


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


# ------------------------------------------------------------------ stage 1: signals (GPU)
def run_scene_signals(scene, cfg, census, e4a_avail, log):
    import cv2
    import numpy as np
    import torch
    from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU
    d = cfg[scene]
    start, folder = d["start_frame"], d["folder"]
    ref = census[scene]["c_ini"]
    targets = [c for c in census[scene]["cameras"] if c != ref]
    objects = [int(o) for o in census[scene]["seed_ids_c_ini"]]
    ds = os.path.join(DATA, folder)
    model, tap = run_scene_signals.model, run_scene_signals.tap
    rig = "hemisphere" if scene in gate.HEMISPHERE else "pinhole"

    def frame_of(cam):
        cpu_image, h, w = load_video_frames(video_path=os.path.join(ds, "Video", cam))
        return AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)[start], h, w

    def pointers(frame, hw, masks):
        """{key: pointer or None} from one single-frame session, as S1-E5b."""
        state = model.init_state([frame], hw[0], hw[1])
        out, nid = {}, 1000
        for key, m in masks.items():
            if m is None or m.sum() == 0:
                out[key] = None
                continue
            tap.last = None
            nid += 1
            model.add_tracker_new_mask(inference_state=state, frame_idx=0, obj_id=nid,
                                       mask=torch.tensor(m, dtype=torch.float32))
            out[key] = tap.last
        del state
        torch.cuda.empty_cache()
        return out

    ref_frame, h, w = frame_of(ref)
    gt_ref = cv2.imread(os.path.join(ds, "Mask", ref, f"{start:06d}.png"), cv2.IMREAD_GRAYSCALE)
    assert gt_ref is not None and gt_ref.shape == (h, w)
    ref_ptr = pointers(ref_frame, (h, w), {o: gt_ref == o for o in objects})
    del ref_frame

    rows = []
    for cam in targets:
        frame, _, _ = frame_of(cam)
        masks = {}
        for obj in objects:
            for name, kind, folder_name in STREAMS:
                m = cv2.imread(mask_path(kind, folder_name, scene, folder, cam, start, obj),
                               cv2.IMREAD_GRAYSCALE)
                masks[(obj, name)] = (m > 127) if m is not None else np.zeros((h, w), bool)
        ptr = pointers(frame, (h, w), masks)
        del frame
        for obj in objects:
            g0 = masks[(obj, "G0")]
            avail_a = bool(e4a_avail.get((scene, cam, obj)))
            proj = masks[(obj, "Ggeo_a")] if avail_a else masks[(obj, "Ggeo_b")]
            # a candidate must not swallow another object's current seed (registered overlap rule)
            others = [masks[(o2, "G0")] for o2 in objects if o2 != obj]
            rec = dict(scene=scene, cam=cam, obj=obj, ref_cam=ref, rig=rig,
                       s2_source=("Ggeo_a" if avail_a else "Ggeo_b"), streams={})
            for name in NAMES:
                m = masks[(obj, name)]
                px = int(m.sum())
                agree = [round(iou(m, masks[(obj, o)]), 4) for o in NAMES if o != name]
                rec["streams"][name] = dict(
                    px=px,
                    s4=(None if ptr[(obj, name)] is None or ref_ptr[obj] is None
                        else round(cosine(ref_ptr[obj], ptr[(obj, name)]), 6)),
                    iou_g0=round(iou(m, g0), 4) if px else 0.0,
                    agree_max=max(agree) if agree else 0.0,
                    n_agree=sum(1 for v in agree if v >= TAU),
                    s2=(None if name in GEO_SELF else (round(iou(m, proj), 4) if px else 0.0)),
                    overlap_other=round(max([iou(m, o) for o in others], default=0.0), 4) if px else 0.0)
            rows.append(rec)
    log(f"  {scene:16s} {len(rows)} rows")
    return rows


def cosine(a, b):
    import torch
    if a is None or b is None:
        return None
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.nn.functional.cosine_similarity(a[None], b[None])[0])


# ------------------------------------------------------------------ stage 2: the registered rule (pure)
def choose(rec):
    """Apply docs/stage1-plan.md section 4 to one row.  Returns (stream or None, why)."""
    st = rec["streams"]
    g0 = st["G0"]
    passed = []
    for name in CAND:
        c = st[name]
        if c["px"] == 0:
            continue
        if c["overlap_other"] > OVERLAP_TAU:                      # registered overlap rejection
            continue
        if g0["px"] > 0:
            if not c["iou_g0"] < TAU:                             # (i) must differ from G0
                continue
            s2_ok = c["s2"] is not None and c["s2"] >= TAU
            if not (c["n_agree"] >= 1 or s2_ok):                  # (ii) agreement or geometry
                continue
            if c["s4"] is None or g0["s4"] is None or not c["s4"] > g0["s4"]:   # (iii) appearance
                continue
        else:                                                     # empty G0: the strengthened branch
            need = EMPTY_MIN_AGREE[rec["rig"]]
            if c["n_agree"] < need:
                continue
            if rec["rig"] == "pinhole" and not (c["s2"] is not None and c["s2"] >= TAU):
                continue
        passed.append(name)
    if not passed:
        return None, "no candidate passed"
    best = max(passed, key=lambda n: (st[n]["s4"] if st[n]["s4"] is not None else -2.0,
                                      -CAND.index(n)))
    return best, f"{len(passed)} passed"


def vote_rows(rows):
    out = []
    for r in rows:
        pick, why = choose(r)
        out.append(dict(scene=r["scene"], cam=r["cam"], obj=r["obj"], rig=r["rig"],
                        g0_px=r["streams"]["G0"]["px"], pick=pick, why=why,
                        pick_px=(r["streams"][pick]["px"] if pick else r["streams"]["G0"]["px"])))
    return out


def summarize(votes):
    by_stream, by_pop = {}, {"replaced": 0, "kept": 0, "empty_g0_filled": 0}
    for v in votes:
        if v["pick"]:
            by_pop["replaced"] += 1
            by_stream[v["pick"]] = by_stream.get(v["pick"], 0) + 1
            if v["g0_px"] == 0:
                by_pop["empty_g0_filled"] += 1
        else:
            by_pop["kept"] += 1
    return {"n_rows": len(votes), **by_pop, "by_stream": by_stream}


def write_seed_folder(scene, cfg, census, votes, sig, log):
    """Copy G0 whole, then overwrite the replaced (camera, object) pairs.  Never touches c_ini."""
    import cv2
    import numpy as np
    d = cfg[scene]
    start, folder = d["start_frame"], d["folder"]
    src = os.path.join(DATA, folder, BASE)
    dst = os.path.join(DATA, folder, SEED_TAG)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    kinds = {n: (k, f) for n, k, f in STREAMS}
    n = 0
    for v in votes:
        if v["scene"] != scene or not v["pick"]:
            continue
        k, f = kinds[v["pick"]]
        m = cv2.imread(mask_path(k, f, scene, folder, v["cam"], start, v["obj"]), cv2.IMREAD_GRAYSCALE)
        assert m is not None, (scene, v)
        p = os.path.join(dst, v["cam"], str(start), f"{v['obj']}.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        cv2.imwrite(p, ((m > 127).astype("uint8") * 255))
        n += 1
    man = dict(schema="MVSeed seed_vote", schema_version=1, base=BASE, prereg="docs/stage1-E6-prereg.md",
               rules=dict(tau=TAU, overlap_tau=OVERLAP_TAU, empty_min_agree=EMPTY_MIN_AGREE,
                          streams=NAMES, tie_break="highest S4, then stream order"),
               replaced=n, c_ini=census[scene]["c_ini"],
               votes=[v for v in votes if v["scene"] == scene])
    json.dump(man, open(os.path.join(dst, "SEED_VOTE.json"), "w"), indent=1)
    log(f"  {scene:16s} wrote {SEED_TAG}, replaced {n}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("signals", "vote"))
    ap.add_argument("--scenes", default="")
    ap.add_argument("--parts-dir", default=SIG_PARTS)
    ap.add_argument("--signals", default=SIGNALS)
    ap.add_argument("--out", default=OUT_VOTE)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    census = json.load(open(CENSUS))
    scenes = args.scenes.split(",") if args.scenes else sorted(census)
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    if args.stage == "signals":
        e4a = {(r["scene"], r["cam"], r["obj"]): r["available"]
               for r in json.load(open(os.path.join(REPO, "MVSeed", "runs", "e4a_project.json")))["rows"]}
        os.makedirs(args.parts_dir, exist_ok=True)
        sys.path.insert(0, PKG)
        import torch
        from build_scsam3 import build_scsam3_video_model
        e5b = _load("MVSeed/e5b_pointer.py", "e5b_pointer")
        run_scene_signals.model = build_scsam3_video_model()
        run_scene_signals.model.eval()
        run_scene_signals.model.fill_hole_area = 0
        run_scene_signals.tap = e5b.PointerTap(run_scene_signals.model.tracker)
        with torch.inference_mode():
            for s in scenes:
                part = os.path.join(args.parts_dir, f"{s}.json")
                if os.path.exists(part) and not args.overwrite:
                    log(f"  {s}: part exists, skipping")
                    continue
                res = run_scene_signals(s, cfg, census, e4a, log)
                json.dump(res, open(part + ".tmp", "w"), indent=1)
                os.replace(part + ".tmp", part)
        run_scene_signals.tap.remove()
        rows, missing = [], []
        for s in sorted(census):
            part = os.path.join(args.parts_dir, f"{s}.json")
            if os.path.exists(part):
                rows += json.load(open(part))
            else:
                missing.append(s)
        json.dump({"schema": "MVSeed e6_signals", "schema_version": 1, "streams": NAMES,
                   "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "missing_scenes": missing, "rows": rows},
                  open(args.signals, "w"), indent=1)
        print(f"wrote {args.signals} ({len(rows)} rows, missing {missing}, {time.time() - t0:.0f}s)")
        return
    sig = json.load(open(args.signals))
    assert not sig.get("missing_scenes"), f"signals incomplete: {sig['missing_scenes']}"
    rows = sig["rows"]
    votes = vote_rows(rows)
    total = sum(write_seed_folder(s, cfg, census, votes, rows, log) for s in scenes)
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    doc = {"schema": "MVSeed e6_vote", "schema_version": 1, "git_head": head,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": "docs/stage1-E6-prereg.md",
           "seed_folder": SEED_TAG, "base": BASE, "summary": summarize(votes), "votes": votes}
    json.dump(doc, open(args.out, "w"), indent=1)
    print(json.dumps(doc["summary"], indent=1))
    print(f"wrote {args.out}  ({total} masks replaced, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
