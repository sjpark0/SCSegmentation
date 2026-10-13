#!/usr/bin/env python3
"""Train the GeoMem-star adapter for one fold (spec 4).

    scripts/docker_run.sh scsam3 python -m geomem.train --arm {geomem,geomem_ft,geomem_ssl} --fold K
        [--epochs 3] [--frame-stride 2] [--max-steps N] [--scenes A,B] [--data Data] [--out geomem/runs] [--seed 0]

Writes <out>/<arm>/fold<K>/{ckpt.pt, log.csv, config.json}.  Fixed schedule, no model selection.
The tracker stays in eval(); only autograd is on.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from geomem import losses  # noqa: E402
from geomem.cache import pi3_frame_cache  # noqa: E402
from geomem.data import (MAX_OBJECTS, SampleIterator, choose_objects, enumerate_pairs, enumerate_samples,  # noqa: E402
                         hub_views, load_sample, load_sample_pair, objects_for, objects_for_pair, pair_targets,
                         unannotated_views)
from geomem.arms import ALIGN_ARMS, CONTINUE_ARMS, FUSE_ARMS, NO_EMBED_ARMS, PTR_ARMS, QUERY_ARMS, SET2_ARMS, base_arm  # noqa: E402
from geomem.folds import load_folds, train_scenes  # noqa: E402
from geomem.pointmap import CONF_MIN, local_points, to_reference, view_grid  # noqa: E402
from methods import common  # noqa: E402
from mvseg2d.io import load_meta  # noqa: E402

ARMS = ("geomem", "geomem_ft", "geomem_ssl", "geobias_ft", "geobias_norope_ft", "geobias_only_ft", "geoset_ft") \
    + QUERY_ARMS + CONTINUE_ARMS + SET2_ARMS + FUSE_ARMS + ALIGN_ARMS
BIAS_ARMS = ("geobias_ft", "geobias_norope_ft", "geobias_only_ft", "geoset_ft") + SET2_ARMS + FUSE_ARMS + ALIGN_ARMS
SET_ARMS = ("geoset_ft",) + SET2_ARMS + FUSE_ARMS + ALIGN_ARMS   # arms trained with a second memory view
FT_ARMS = ("geomem_ft",) + BIAS_ARMS
PAIR_ARMS = QUERY_ARMS + CONTINUE_ARMS   # GeoQuery arms: (scene, t, ref) samples with both annotated targets
EXTRA_MEM_PROB = 0.5     # geoset arms: probability of adding a second memory view
PRED_MEM_TAU = 0.7       # geoset_pred_ft: object gate on the predicted memory (= methods/geomem/run.TAU_CONF, spec 2.1)
HUB_RANDOM_VIEWS = 5     # GeoQuery spec 3: hub subset K = ref + 2 targets + 5 random
LR_EMBED, LR_ATTN, WEIGHT_DECAY = 1e-3, 1e-5, 0.01
LOG_COLUMNS = ["step", "epoch", "loss", "focal", "dice", "iou", "presence", "ssl", "lr", "seconds", "n_mem", "n_views", "n_targets",
               "gated_frac", "g", "self_mass", "align_valid"]


def cosine_lr(step, total):
    """Multiplier in [0, 1]: 1 at step 0, cosine decay to 0 at `total`."""
    if total <= 1:
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * min(step, total) / total))


def arm_flags(arm):
    b = base_arm(arm)
    return {"use_bias": b in BIAS_ARMS, "no_rope": b == "geobias_norope_ft", "use_geometry": b not in NO_EMBED_ARMS,
            "extra_mem_prob": EXTRA_MEM_PROB if b in SET_ARMS else 0.0, "extra_mem_pred": b in ("geoset_pred_ft",) + FUSE_ARMS,
            "use_fusion": b in FUSE_ARMS, "extra_mem_aligned": b in ALIGN_ARMS,
            "fuse_two_sided": b == "geofuse2_ft", "fuse_exclude_self": b == "geofuse2_ft",
            "use_hub": arm in QUERY_ARMS, "use_objbias": arm in ("geoquery_ft",) + PTR_ARMS,
            "init_from_arm": b if arm in PAIR_ARMS else None,
            "inject": "ptr" if arm in PTR_ARMS else "prompt", "g_cross_init": 0.1 if arm in PTR_ARMS else 0.0,
            "objbias_from_geobias": arm in PTR_ARMS}


def build_wrapper(model, arm, device="cuda"):
    from geomem.bias import GeoBias
    from geomem.embed import GeoEmbed
    from geomem.wrapper import GeoMemStar, GeoQuery
    f = arm_flags(arm)
    kw = dict(use_geometry=f["use_geometry"], use_bias=f["use_bias"], no_rope=f["no_rope"])
    geo_bias = GeoBias().to(device) if f["use_bias"] else None
    if arm in PAIR_ARMS:
        from geomem.query_hub import QueryHub
        hub = QueryHub(use_bias=f["use_objbias"], ptr=f["inject"] == "ptr", g_cross_init=f["g_cross_init"]).to(device) \
            if f["use_hub"] else None
        return GeoQuery(model, GeoEmbed().to(device), geo_bias, hub, inject=f["inject"], **kw)
    fusion = None
    if f["use_fusion"]:
        from geomem.fusion import ViewFusion
        fusion = ViewFusion().to(device)
    return GeoMemStar(model, GeoEmbed().to(device), geo_bias=geo_bias, fusion=fusion, **kw)


def build_optimizer(wrapper, arm):
    import torch
    if arm in PAIR_ARMS:                     # GeoQuery spec 2.6: hub 1e-3; memory attention + GeoBias 1e-5 (continued)
        attn = [p for p in wrapper.tracker.transformer.encoder.parameters() if p.requires_grad]
        attn += list(wrapper.geo_bias.parameters())
        groups = [{"params": attn, "lr": LR_ATTN, "name": "memory_attention"}]
        if getattr(wrapper, "hub", None) is not None:
            groups.insert(0, {"params": list(wrapper.hub.parameters()), "lr": LR_EMBED, "name": "query_hub"})
        return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    embed_params = [p for p in wrapper.geo_embed.parameters() if p.requires_grad]
    if wrapper.geo_bias is not None and arm in BIAS_ARMS:
        embed_params += list(wrapper.geo_bias.parameters())
    if getattr(wrapper, "fusion", None) is not None and arm in FUSE_ARMS:       # GeoFuse spec 3: fusion at 1e-3
        embed_params += list(wrapper.fusion.parameters())
    groups = [{"params": embed_params, "lr": LR_EMBED, "name": "geo_embed"}]
    if arm in FT_ARMS:
        groups.append({"params": [p for p in wrapper.tracker.transformer.encoder.parameters() if p.requires_grad],
                       "lr": LR_ATTN, "name": "memory_attention"})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def load_init_from(wrapper, path, flags=None):
    """Warm start from a GeoMem-2 checkpoint: memory attention and GeoBias; with `objbias_from_geobias` the hub's
    ObjBias starts from the same RBF kernel.  Returns the file's SHA256."""
    import torch
    assert not torch.is_autocast_cache_enabled(), "autocast weight cache must be disabled before loading weights"
    ck = torch.load(path, map_location="cuda")
    wrapper.tracker.transformer.encoder.load_state_dict(ck["memory_attention"])
    wrapper.geo_bias.load_state_dict(ck["geo_bias"])
    if flags and flags.get("objbias_from_geobias") and getattr(wrapper, "hub", None) is not None and wrapper.hub.obj_bias is not None:
        wrapper.hub.obj_bias.load_state_dict(ck["geo_bias"])
    return _sha256(path)


def train_pair_step(wrapper, batch, flags):
    """GeoQuery step: reference memory, one hub pass over the batch's hub views (if any), both targets decoded with
    the query, loss summed over targets; backward done here.  Returns the loss terms."""
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = wrapper.encode_reference(batch["ref_image"], batch["ref_backbone"], batch["ref_masks"],
                                       batch["p_ref"], batch["valid_ref"])
        query, n_views = None, 0
        if flags["use_hub"]:
            hub = wrapper.run_hub(ref, batch["hub"]["backbones"], batch["hub"]["p"], batch["hub"]["valid"],
                                  batch["ref_masks_np"], batch["p_ref_np"], batch["valid_ref_np"])
            query, n_views = hub["q"], len(batch["hub"]["cams"])
        B = batch["ref_masks"].shape[0]
        outs = [wrapper.decode_view(t["backbone"], B, ref, t["p"], t["valid"], query=query) for t in batch["targets"]]
    total, terms = None, {}
    for out, t in zip(outs, batch["targets"]):
        loss, tm = losses.supervised_loss(out, t["masks"])
        total = loss if total is None else total + loss
        for k, v in tm.items():
            terms[k] = terms.get(k, 0.0) + float(v)
    total.backward()
    terms.update({"loss": float(total.detach()), "ssl": 0.0, "n_mem": 1, "n_views": n_views, "n_targets": len(outs)})
    return terms


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ssl_view(data, sample, cam, model, device):
    """An unannotated view u: backbone (on the fly), pooled grid, and full-resolution reprojection into the reference."""
    import cv2
    import torch
    from geomem import sam3_io
    from geomem.embed import grid_tensors
    scene = common.load_scene(data, sample.scene, frame=sample.t)
    image, h, w = sam3_io.load_image(scene.image_paths[cam])
    bb = sam3_io.backbone_features(model, image)
    cache = pi3_frame_cache(data, sample.scene, sample.t)
    p_u, v_u = grid_tensors(view_grid(cache, cam, sample.ref_cam), device)
    zu = np.load(os.path.join(cache, f"{cam}.npz"))
    zr = np.load(os.path.join(cache, f"{sample.ref_cam}.npz"))
    pts = to_reference(local_points(zu["depth"], zu["K"]), zu["c2w"], zr["c2w"]).astype(np.float32)
    valid_u = ((zu["conf"] > CONF_MIN) & (zu["depth"] > 0)).astype(np.uint8)
    pts_full = cv2.resize(pts, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_full = cv2.resize(valid_u, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    depth_ref_full = cv2.resize(zr["depth"], (w, h), interpolation=cv2.INTER_NEAREST)
    wh_work = zr["depth"].shape
    sx, sy = w / wh_work[1], h / wh_work[0]
    K = zr["K"].copy()
    K_full = np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5],
                       [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5], [0, 0, 1]], np.float32)
    return {"backbone": bb, "p": p_u, "valid": v_u,
            "pts_full": torch.from_numpy(pts_full).to(device), "valid_full": torch.from_numpy(valid_full).to(device),
            "depth_ref_full": torch.from_numpy(depth_ref_full).to(device), "K_ref_full": torch.from_numpy(K_full).to(device),
            "hw": (h, w)}


def train_one_step(wrapper, batch, arm, rng, data=None, sample=None, model=None, device="cuda"):
    """One optimisation step's forward/backward (no optimizer.step); returns the loss terms."""
    import torch
    from geomem.sam3_io import logits_to_original
    from geomem.warp_torch import reproject_coords
    B = batch["ref_masks"].shape[0]
    fuse = getattr(wrapper, "fusion", None) is not None and "views" in batch

    fflags = arm_flags(arm) if fuse else {}

    def aligned_for(p, valid, cam=None):             # GeoFuse spec 3: every camera's features aligned to this view's grid
        if not fuse:
            return None
        from geomem.align import align_views, fusion_views
        views = fusion_views(batch["views"], batch["view_cams"], exclude=cam) if fflags.get("fuse_exclude_self") else batch["views"]
        return align_views(p, valid, batch["scale"], batch["c2w_ref"], views, two_sided=fflags.get("fuse_two_sided", False))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = wrapper.encode_reference(batch["ref_image"], batch["ref_backbone"], batch["ref_masks"],
                                       batch["p_ref"], batch["valid_ref"])
        gated_frac, align_valid = None, None
        if "mem_view" in batch:                          # geoalign_ft (GeoAlign spec 3): a neighbour aligned to the reference grid
            from geomem.align import align_views
            W, valid = align_views(batch["p_ref"], batch["valid_ref"], batch["scale"], batch["c2w_ref"], [batch["mem_view"]],
                                   two_sided=True)
            view = wrapper.encode_aligned(batch["ref_image"], batch["ref_backbone"], batch["ref_masks"], batch["p_ref"],
                                          batch["valid_ref"], W[0], valid[0], batch["mem_cam"])
            ref = ref.with_view(view)
            align_valid = float(valid[0].float().mean())
        elif "mem_masks" in batch:                       # geoset_ft / geoset_only_ft: a GT memory view
            extra = wrapper.encode_reference(batch["mem_image"], batch["mem_backbone"], batch["mem_masks"],
                                             batch["p_mem"], batch["valid_mem"], name=batch["mem_cam"])
            ref = ref.with_view(extra.views[0])
        elif "mem_backbone" in batch:                    # geoset_pred_ft (GeoSet-2 spec 4.2): the model's own prediction
            with torch.no_grad():
                pred = wrapper.decode_view(batch["mem_backbone"], B, ref, batch["p_mem"], batch["valid_mem"],
                                           aligned=aligned_for(batch["p_mem"], batch["valid_mem"], batch["mem_cam"]))
            view = wrapper.encode_memory_from_prediction(
                batch["mem_backbone"], B, pred["logits_1008"].detach(), pred["presence"].detach(), pred["obj_ptr"].detach(),
                batch["p_mem"], batch["valid_mem"], batch["mem_cam"], ious=pred["ious"].detach(), tau=PRED_MEM_TAU)
            if bool(view.conf.any()):
                ref = ref.with_view(view)
                gated_frac = 1.0 - float(view.conf.float().mean())
        out = wrapper.decode_view(batch["tgt_backbone"], B, ref, batch["p_tgt"], batch["valid_tgt"],
                                  aligned=aligned_for(batch["p_tgt"], batch["valid_tgt"], batch["tgt_cam"]))
    loss, terms = losses.supervised_loss(out, batch["tgt_masks"])
    terms["ssl"] = 0.0
    terms["n_mem"] = len(ref.views)
    terms["gated_frac"] = gated_frac
    terms["align_valid"] = align_valid
    terms["g"] = float(wrapper.fusion.g.detach()) if fuse else None
    terms["self_mass"] = wrapper.last_fusion_stats["self_mass"] if fuse else None
    if arm == "geomem_ssl":
        views = unannotated_views(data, sample.scene, 2, 4, rng)
        ssl_terms = []
        for cam in views:
            u = ssl_view(data, sample, cam, model, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_u = wrapper.decode_view(u["backbone"], B, ref, u["p"], u["valid"])
            logits_u = logits_to_original(out_u["logits_1008"], u["hw"])
            grid, valid = reproject_coords(u["pts_full"], u["K_ref_full"], batch["hw"], depth_ref=u["depth_ref_full"])
            valid = valid & u["valid_full"]
            ssl_terms.append(losses.reprojection_consistency(logits_u, batch["ref_masks"], grid, valid))
        if ssl_terms:
            ssl = torch.stack(ssl_terms).mean()
            loss = loss + losses.W_SSL * ssl
            terms["ssl"] = float(ssl.detach())
    loss.backward()
    terms["loss"] = float(loss.detach())
    return terms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3,
                    help="fixed schedule, no model selection (spec said 10; cut to 3 for the GPU budget, see plan ledger)")
    ap.add_argument("--frame-stride", type=int, default=2, help="use every k-th labelled frame (2 -> 11 of 21)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--scenes", default=None, help="comma-separated override of the fold's training scenes (smoke tests)")
    ap.add_argument("--data", default=os.path.join(common.REPO, "Data"))
    ap.add_argument("--out", default=os.path.join(common.REPO, "geomem", "runs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from-runs", default=None,
                    help="runs dir holding the warm-start checkpoints of GeoQuery arms (default: --out)")
    a = ap.parse_args(argv)

    import torch
    from geomem.data import memory_rng, random_memory_camera, third_camera
    from methods import sam3_single
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda"
    flags = arm_flags(a.arm)

    folds = load_folds()
    scenes = a.scenes.split(",") if a.scenes else train_scenes(folds, a.fold)
    held_out = sorted(s for s, f in folds.items() if f == a.fold)
    pair_mode = a.arm in PAIR_ARMS
    samples = enumerate_pairs(a.data, scenes, a.frame_stride) if pair_mode else enumerate_samples(a.data, scenes, a.frame_stride)
    total_steps = a.epochs * len(samples) if a.max_steps is None else min(a.max_steps, a.epochs * len(samples))

    model = sam3_single.load_model()
    wrapper = build_wrapper(model, a.arm, device)
    trainable = wrapper.freeze(a.arm)
    init_from, init_from_sha256 = None, None
    if flags["init_from_arm"]:
        init_from = os.path.join(a.out if a.init_from_runs is None else a.init_from_runs, flags["init_from_arm"], f"fold{a.fold}", "ckpt.pt")
        init_from_sha256 = load_init_from(wrapper, init_from, flags)
    n_trainable = sum(p.numel() for p in wrapper.trainable_parameters())
    opt = build_optimizer(wrapper, a.arm)
    base_lrs = [g["lr"] for g in opt.param_groups]

    out_dir = os.path.join(a.out, a.arm, f"fold{a.fold}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.csv")
    print(f"arm {a.arm} fold {a.fold}: {len(scenes)} scenes, {len(samples)} samples, {total_steps} steps, "
          f"{n_trainable} trainable params; held out {held_out}", flush=True)

    it = SampleIterator(samples, seed=a.seed)
    step, t0 = 0, time.time()
    with open(log_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        wr.writeheader()
        done = False
        for epoch in range(a.epochs):
            for sample in it.epoch():
                if step >= total_steps:
                    done = True
                    break
                objects = choose_objects((objects_for_pair if pair_mode else objects_for)(sample, a.data), MAX_OBJECTS, rng)
                if not objects:
                    continue
                if pair_mode:
                    cams = hub_views(load_meta(a.data, sample.scene)["cams"], sample.ref_cam, pair_targets(a.data, sample),
                                     HUB_RANDOM_VIEWS, rng)
                    batch = load_sample_pair(a.data, sample, objects, cams, device)
                else:
                    extra = None
                    if flags["extra_mem_prob"] > 0 and rng.random() < flags["extra_mem_prob"]:
                        extra = random_memory_camera(a.data, sample, memory_rng(a.seed, step)) \
                            if (flags["extra_mem_pred"] or flags["extra_mem_aligned"]) else third_camera(a.data, sample)
                    batch = load_sample(a.data, sample, objects, device, extra_mem_cam=extra,
                                        extra_mem_gt=not (flags["extra_mem_pred"] or flags["extra_mem_aligned"]),
                                        aligned=flags["use_fusion"], aligned_mem=flags["extra_mem_aligned"])
                mult = cosine_lr(step, total_steps)
                for g, lr0 in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr0 * mult
                opt.zero_grad(set_to_none=True)
                terms = train_pair_step(wrapper, batch, flags) if pair_mode else \
                    train_one_step(wrapper, batch, a.arm, rng, a.data, sample, model, device)
                torch.nn.utils.clip_grad_norm_(wrapper.trainable_parameters(), 1.0)
                opt.step()
                step += 1
                wr.writerow({"step": step, "epoch": epoch, "loss": f"{terms['loss']:.5f}", "focal": f"{terms['focal']:.5f}",
                             "dice": f"{terms['dice']:.5f}", "iou": f"{terms['iou']:.5f}", "presence": f"{terms['presence']:.5f}",
                             "ssl": f"{terms['ssl']:.5f}", "lr": f"{base_lrs[0] * mult:.2e}", "seconds": f"{time.time() - t0:.1f}",
                             "n_mem": terms.get("n_mem", 1), "n_views": terms.get("n_views", 0),
                             "n_targets": terms.get("n_targets", 1),
                             "gated_frac": "" if terms.get("gated_frac") is None else f"{terms['gated_frac']:.4f}",
                             "g": "" if terms.get("g") is None else f"{terms['g']:.5f}",
                             "self_mass": "" if terms.get("self_mass") is None else f"{terms['self_mass']:.4f}",
                             "align_valid": "" if terms.get("align_valid") is None else f"{terms['align_valid']:.4f}"})
                if step % 50 == 0 or step == total_steps:
                    f.flush()
                    print(f"step {step}/{total_steps} epoch {epoch} loss {terms['loss']:.4f} "
                          f"({(time.time() - t0) / step:.2f} s/step)", flush=True)
                del batch
            if done:
                break

    ckpt = {"geo_embed": wrapper.geo_embed.state_dict(), "arm": a.arm, "fold": a.fold, "scenes": scenes}
    if wrapper.geo_bias is not None:
        ckpt["geo_bias"] = wrapper.geo_bias.state_dict()
    if a.arm in FT_ARMS or a.arm in PAIR_ARMS:
        ckpt["memory_attention"] = wrapper.tracker.transformer.encoder.state_dict()
    if getattr(wrapper, "hub", None) is not None:
        ckpt["query_hub"] = wrapper.hub.state_dict()
    if getattr(wrapper, "fusion", None) is not None:
        ckpt["fusion"] = wrapper.fusion.state_dict()
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    torch.save(ckpt, ckpt_path)
    try:
        git = subprocess.check_output(["git", "-C", common.REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git = None
    config = {"arm": a.arm, "fold": a.fold, "scenes": scenes, "held_out": held_out, "epochs": a.epochs,
              "frame_stride": a.frame_stride, "steps": step,
              "samples": len(samples), "seed": a.seed, "trainable_params": n_trainable, "trainable_names": trainable[:8],
              "lr": {g["name"]: lr0 for g, lr0 in zip(opt.param_groups, base_lrs)},
              "init_from": init_from, "init_from_sha256": init_from_sha256,
              "weight_decay": WEIGHT_DECAY, "seconds": time.time() - t0, "git": git, "ckpt_sha256": _sha256(ckpt_path),
              **flags}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"done: {step} steps in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
