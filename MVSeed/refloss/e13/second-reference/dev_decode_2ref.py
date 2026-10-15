#!/usr/bin/env python3
"""S1-E13 candidate E2 — PROTOCOL CHANGE, diagnostic upper bound of new information (scratch, no tracked change).

A second ANNOTATED reference view.  MUVOD gives stage 1 one annotated view (c_ini's mask); here each DEV target X (one of
the 2 annotated non-reference cameras) is decoded with the OTHER annotated non-reference camera B's GT of the same frame t
as a second reference memory view, i.e. what a two-reference protocol would provide.  Under that protocol B would be an
input, not a scored camera, so J&F^3 itself changes; this is a diagnostic, not a candidate for the main track.

DEV set and inputs: exactly geomem/runs/_e13_dev-route/dev_decode.py (frames ref+4..ref+20, basic object set, c_ini GT of
frame t, Cache/pi3_frames/<t>); the keyframe (meta ref_frame) is never decoded.

Decode = methods.geomem.run.decode_scene's path for the adopted preset geoset_rep_x (arm geoset_pred_ft, own-fold checkpoint,
--gate reproj, --xcl out, round2 pred), re-assembled from the same functions for one target X:
    mode none  : memory_r1 = {c_ini GT};            memory_r2 = {c_ini GT} + gated round-1 views of X's k=3 candidates (minus c_ini)
                 -> must reproduce the adopted dev masks of X byte for byte (--validate checks it against _e13_dev)
    mode both  : the two-reference protocol: references {c_ini GT, B GT};
                 memory_r1 = {c_ini, B} for X and for each candidate, memory_r2 = {c_ini, B} + gated round-1 views of X's
                 k=3 candidates minus {c_ini, B} (a reference among the k nearest is already in the set, as c_ini is today)
    mode r2    : B GT only in round 2 (round 1 = {c_ini}); candidates as in `both`
B's view = wrapper.encode_reference(B image, B backbone, B GT masks of the c_ini objects, B's Pi3 grid in c_ini's frame)
(the GeoProj proj_bin / geoset_ft GT-memory path), conf = object present in B's GT (an absent object's B tokens are gated,
the GeoProj precedent).  The reproj gate of the candidates is unchanged (overlap with c_ini GT).  Fixed before any run.

Outputs  <root>/<mode>/<scene>/<t:06d>/<X>/<obj>.png  (round-2 final, absent = empty; the dev_score.py layout)
         <root>/<mode>/<scene>/<t:06d>/_r1/<X>/<obj>.png  (X's round-1),  <t>/rec_<X>.json, <t>/DONE_<X>

    flock <lock> stage1/scripts/docker_run.sh scsam3 python geomem/runs/_e13_second-reference/dev_decode_2ref.py \
        --mode both [--scenes A,B] [--first-only] [--validate]
"""
import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import replace

S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, S1)
sys.path.insert(0, os.path.join(S1, "geomem", "runs", "_e13_dev-route"))
import numpy as np  # noqa: E402

import dev_decode as DD  # noqa: E402  (frame_scene, dev_frames, eval_cams, basic_set, PRESET, DATA, RUNS)
from geomem.cache import pi3_frame_cache  # noqa: E402
from methods import common  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem.stage1 import PRESETS, check_preflight  # noqa: E402
from mvseg2d.io import labelmap_to_mask, load_labelmap, read_pred_mask, write_mask  # noqa: E402

MODES = ("none", "both", "r2")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(S1, "geomem", "runs", "_e13_dev")


def load_wrapper(model, scene_name, arm):
    """decode_scene's wrapper + own-fold checkpoint (leak guard), once per scene."""
    from geomem.folds import load_folds
    from geomem.train import build_wrapper
    wrapper = build_wrapper(model, arm)
    ckpt_path = R.checkpoint_for(scene_name, arm, DD.RUNS, load_folds())
    with open(os.path.join(os.path.dirname(ckpt_path), "config.json")) as f:
        cfg = json.load(f)
    R.assert_no_leak(cfg, scene_name)
    ck = R.load_checkpoint_into(wrapper, wrapper.geo_embed, ckpt_path)
    wrapper.eval()
    return wrapper, ck, cfg


def decode_target(model, wrapper, sc, t, X, Bcam, mode, gt_B):
    """One target X of frame t -> (final bool [N,h,w] after --xcl out, round-1 bool [N,h,w] of X, record)."""
    import torch
    from geomem import sam3_io
    from geomem.embed import grid_tensors
    from geomem.pointmap import view_grid
    from geomem.wrapper import MemorySet
    p = PRESETS[DD.PRESET]
    assert p == dict(arm="geoset_pred_ft", gate="reproj", xcl="out"), p
    cache = pi3_frame_cache(DD.DATA, sc.name, t)
    masks = torch.from_numpy(np.stack([labelmap_to_mask(sc.ref_labels, o) for o in sc.objects])).cuda()
    N = masks.shape[0]
    centres = R.camera_centres(cache, sc.cams)
    refs = {sc.ref_cam} | ({Bcam} if mode != "none" else set())
    cand = [c for c in R.select_candidates(X, centres, R.K_NEIGHBOURS) if c not in refs]
    rec = dict(scene=sc.name, t=t, target=X, second_ref=Bcam if mode != "none" else None, mode=mode, cand=cand)
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_img, h, w = sam3_io.load_image(sc.image_paths[sc.ref_cam])
        assert (h, w) == sc.hw
        ref_bb = sam3_io.backbone_features(model, ref_img)
        p_ref, v_ref = grid_tensors(view_grid(cache, sc.ref_cam, sc.ref_cam), "cuda")
        ref = wrapper.encode_reference(ref_img, ref_bb, masks, p_ref, v_ref)
        del ref_img, ref_bb
        mem_r1 = mem_r2 = ref
        if mode != "none":
            img_B, hB, wB = sam3_io.load_image(sc.image_paths[Bcam])
            assert (hB, wB) == sc.hw and gt_B.shape == sc.hw
            bb_B = sam3_io.backbone_features(model, img_B)
            p_B, v_B = grid_tensors(view_grid(cache, Bcam, sc.ref_cam), "cuda")
            mB = torch.from_numpy(np.stack([gt_B == o for o in sc.objects])).cuda()
            confB = mB.flatten(1).any(1)
            vB = replace(wrapper.encode_reference(img_B, bb_B, mB, p_B, v_B, name=Bcam).views[0], conf=confB)
            del img_B, bb_B
            two = MemorySet(ref.views + [vB])
            mem_r2 = two
            if mode == "both":
                mem_r1 = two
            rec["B_present"] = int(confB.sum())
            rec["B_valid_cells"] = float(v_B.float().mean())
        round1 = {}
        for cam in [X] + cand:
            img, _, _ = sam3_io.load_image(sc.image_paths[cam])
            bb = sam3_io.backbone_features(model, img)
            del img
            p_v, v_v = grid_tensors(view_grid(cache, cam, sc.ref_cam), "cuda")
            out = wrapper.decode_view(bb, N, mem_r1, p_v, v_v, chunk=R.OBJ_CHUNK)   # chunk only acts on a gated set
            round1[cam] = dict(bb=bb, p=p_v, valid=v_v, logits=out["logits_1008"], presence=out["presence"],
                               ious=out["ious"], obj_ptr=out["obj_ptr"])
            del out
        r1_X = (sam3_io.logits_to_original(round1[X]["logits"], sc.hw) > 0)[:, 0].cpu().numpy()
        mem_views, gate_rec = {}, {}
        for c in cand:
            r = round1[c]
            pred = (sam3_io.logits_to_original(r["logits"], sc.hw) > 0)[:, 0]
            ov, n_valid = R.reproj_overlap(pred, *R.reproj_inputs(cache, c, sc.ref_cam, sc.hw, "cuda"), masks)
            del pred
            conf = R.object_conf_mode("reproj", r["presence"].float().cpu().numpy(), r["ious"].float().cpu().numpy(),
                                      ov.cpu().numpy(), tau=R.TAU_CONF)
            mem_views[c] = wrapper.encode_memory_from_prediction(
                r["bb"], N, r["logits"], r["presence"], r["obj_ptr"], r["p"], r["valid"], c,
                conf=torch.from_numpy(conf).to(r["presence"].device))
            gate_rec[c] = int(conf.sum())
        neigh = [c for c in cand if bool(mem_views[c].conf.any())]
        mem = MemorySet(mem_r2.views + [mem_views[c] for c in neigh])
        rx = round1[X]
        out = wrapper.decode_view(rx["bb"], N, mem, rx["p"], rx["valid"], chunk=R.OBJ_CHUNK)
        L = sam3_io.logits_to_original(out["logits_1008"], sc.hw)[:, 0]
        del out
        keep, xc, _ = R.exclusive_masks(L, R.XCL_TAU_KEEP)
        del L
        final = keep.cpu().numpy()
        del keep, round1, mem_views, mem, mem_r1, mem_r2, ref
    torch.cuda.empty_cache()
    rec.update(neigh=neigh, gate_conf=gate_rec, n_mem_r2=len(neigh) + (1 if mode == "none" else 2),
               n_mem_r1=1 if mode in ("none", "r2") else 2, xcl=dict(xc), seconds=time.time() - t0, n_objects=N)
    return final, r1_X, rec


def write_target(root, sc, t, X, final, r1_X, rec):
    d = os.path.join(root, sc.name, f"{t:06d}")
    for sub in (os.path.join(d, X), os.path.join(d, "_r1", X)):
        if os.path.isdir(sub):
            shutil.rmtree(sub)
    n = 0
    for i, o in enumerate(sc.objects):
        if final[i].any():
            write_mask(os.path.join(d, X, f"{int(o)}.png"), final[i])
            n += 1
        if r1_X[i].any():
            write_mask(os.path.join(d, "_r1", X, f"{int(o)}.png"), r1_X[i])
    with open(os.path.join(d, f"rec_{X}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    open(os.path.join(d, f"DONE_{X}"), "w").close()
    return n


def compare_base(sc, t, X, final):
    """Byte-level equality with the adopted dev decode of X (geomem/runs/_e13_dev): same non-empty objects, same pixels."""
    d = os.path.join(BASE, sc.name, f"{t:06d}", X)
    bad = []
    for i, o in enumerate(sc.objects):
        pth = os.path.join(d, f"{int(o)}.png")
        b = read_pred_mask(pth) if os.path.isfile(pth) else np.zeros(sc.hw, bool)
        a = final[i] if final[i].any() else np.zeros(sc.hw, bool)
        if not np.array_equal(a, b):
            bad.append((int(o), int((a != b).sum())))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--root", default=HERE)
    ap.add_argument("--first-only", action="store_true", help="only each scene's first dev frame")
    ap.add_argument("--validate", action="store_true", help="compare X's final masks with geomem/runs/_e13_dev (mode none)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    root = os.path.join(a.root, a.mode)
    os.makedirs(root, exist_ok=True)
    idx = json.load(open(os.path.join(BASE, "index.json")))
    scenes = common.all_scenes(DD.DATA) if a.scenes == "all" else a.scenes.split(",")
    from methods import sam3_single
    model = sam3_single.load_model()
    init_state = R.snapshot_init(model)
    val = {}
    for s in scenes:
        S = idx["scenes"][s]
        pf = check_preflight(DD.PRESET, s)
        assert S["frames"] == DD.dev_frames(s) and S["eval_cams"] == DD.eval_cams(s), s
        basic = DD.basic_set(s)
        frames = S["frames"][:1] if a.first_only else S["frames"]
        wrapper, ck, cfg = load_wrapper(model, s, PRESETS[DD.PRESET]["arm"])
        try:
            for t in frames:
                assert t != S["ref_frame"], (s, t)
                sc = DD.frame_scene(s, t, basic)
                fdir = os.path.join(DD.DATA, s, "frames", f"{t:06d}", "masks")
                for X in S["eval_cams"]:
                    Bcam = [c for c in S["eval_cams"] if c != X]
                    assert len(Bcam) == 1
                    Bcam = Bcam[0]
                    done = os.path.join(root, s, f"{t:06d}", f"DONE_{X}")
                    if os.path.isfile(done) and not a.overwrite:
                        print(f"{s:20s} t={t:4d} {X:12s} skip (DONE)", flush=True)
                        continue
                    gt_B = load_labelmap(os.path.join(fdir, f"{Bcam}.png"))
                    final, r1_X, rec = decode_target(model, wrapper, sc, t, X, Bcam, a.mode, gt_B)
                    rec.update(fold=pf["fold"], ckpt=pf["checkpoint"], ckpt_sha256=cfg.get("ckpt_sha256"))
                    n = write_target(root, sc, t, X, final, r1_X, rec)
                    msg = (f"{s:20s} t={t:4d} {X:12s} B={Bcam:12s} objs {rec['n_objects']:2d} cand {len(rec['cand'])} "
                           f"neigh {len(rec['neigh'])} masks {n:3d} {rec['seconds']:5.1f}s")
                    if a.validate:
                        bad = compare_base(sc, t, X, final)
                        val[f"{s}/{t}/{X}"] = dict(equal=not bad, diffs=bad)
                        msg += f" | equal_to_adopted={not bad} {bad[:3]}"
                    print(msg, flush=True)
        finally:
            R.restore_init(model, init_state, ck)
            wrapper.close()
    if val:
        with open(os.path.join(root, "_validate.json"), "w") as f:
            json.dump(val, f, indent=1)
        print("VALIDATE", all(v["equal"] for v in val.values()), sum(v["equal"] for v in val.values()), "/", len(val), flush=True)


if __name__ == "__main__":
    main()
