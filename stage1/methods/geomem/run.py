#!/usr/bin/env python3
"""geomem: decode every camera of the reference frame from the reference camera's memory with the
GeoMem-star adapter of the fold that did NOT train on the scene (spec 5).

    scripts/run_geomem.sh <scene|all> --arm {geomem,geomem_ft,geomem_ssl} [--out NAME]
    scripts/run_geomem.sh <scene|all> --untrained --out star          # control: zero-initialised heads

A missing checkpoint is an error for that scene (never a silent fall-back); a checkpoint whose
training scenes include the scene is refused.  Runs inside `scsam3`.
"""
import json
import os
import sys
from collections import Counter
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from geomem.folds import load_folds  # noqa: E402
from geomem.pointmap import view_grid  # noqa: E402
from methods import common  # noqa: E402
from mvseg2d.io import cache_dir, labelmap_to_mask, result_dir  # noqa: E402

from geomem.arms import ALIGN_ARMS, CONTINUE_ARMS, FUSE_ARMS, QUERY_ARMS, SET2_ARMS  # noqa: E402

ARMS = ("geomem", "geomem_ft", "geomem_ssl", "geobias_ft", "geobias_norope_ft", "geobias_only_ft", "geoset_ft") \
    + QUERY_ARMS + CONTINUE_ARMS + SET2_ARMS + FUSE_ARMS + ALIGN_ARMS
SET_ARMS = ("geoset_ft",) + SET2_ARMS + FUSE_ARMS + ALIGN_ARMS
PAIR_ARMS = QUERY_ARMS + CONTINUE_ARMS   # GeoQuery: all views' features first, one hub pass, single-round decoding
TAU_CONF = 0.7     # spec 2.2: a view is a confident memory when mean predicted IoU over present objects >= tau
K_NEIGHBOURS = 3
OBJ_CHUNK = 8      # round-2 decoding with a per-object gate mask: at most this many objects per forward (OOM guard)
THETA_REPROJ = 0.5 # §8 reproj gate: fraction of a predicted mask's reprojected pixels that must land on the reference GT
MIN_VALID = 64     # §8: fewer validly reprojected mask pixels than this -> fall back to the IoU rule
OBJ_GATES = ("obj", "absent", "reproj")   # per-object gates (view-level candidates, per-object masking)


def with_external_fold(folds, scene, external_fold):
    """folds.json plus a scene OUTSIDE the benchmark at `external_fold` (its stage-1 meta.json; every fold is held out
    for it, the leak guard still runs).  A benchmark scene keeps its own fold: another one is refused."""
    if external_fold is None:
        return folds
    if scene in folds:
        if int(external_fold) != folds[scene]:
            raise KeyError(f"{scene} is a benchmark scene of fold{folds[scene]}: external_fold {external_fold} refused")
        return folds
    return dict(folds, **{scene: int(external_fold)})


def checkpoint_for(scene, arm, runs_dir, folds):
    k = folds[scene]
    p = os.path.join(runs_dir, arm, f"fold{k}", "ckpt.pt")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"no checkpoint for {scene}: expected {p} (arm {arm}, fold{k})")
    return p


def camera_centres(cache_dir, cams):
    """Pi3 camera centres (c2w translations) in the scene's common frame."""
    return {c: np.load(os.path.join(cache_dir, f"{c}.npz"))["c2w"][:3, 3].astype(np.float64) for c in cams}


def select_neighbours(view, centres, confident, k=K_NEIGHBOURS):
    """The k nearest other cameras by centre distance, kept only if confident; nearest first."""
    others = sorted((np.linalg.norm(centres[c] - centres[view]), c) for c in centres if c != view)
    return [c for _, c in others[:k] if confident.get(c, False)]


def select_candidates(view, centres, k=K_NEIGHBOURS):
    """The k nearest other cameras by centre distance, no confidence filter (object gate, GeoSet-2 spec 3.1)."""
    others = sorted((np.linalg.norm(centres[c] - centres[view]), c) for c in centres if c != view)
    return [c for _, c in others[:k]]


def aligned_neighbours(ref_cam, cams, centres, k=K_NEIGHBOURS, exclude=None):
    """GeoAlign spec 4: the k cameras nearest to the reference by centre distance (reference excluded, clamped).
    `exclude` (review follow-up): the target being decoded is never its own aligned memory source; the next nearest fills in."""
    others = sorted((np.linalg.norm(centres[c] - centres[ref_cam]), c) for c in cams if c != ref_cam and c != exclude)
    return [c for _, c in others[:k]]


def object_conf(presence, ious, tau=TAU_CONF):
    """Per-object confidence of a predicted memory view: present and predicted IoU >= tau (GeoSet-2 spec 2.1)."""
    return (presence > 0) & (ious >= tau)


def check_gate_out(arm, gate, out, round2="pred", zoom=None, xcl="none"):
    """`--gate obj` / `--round2 proj_*` / `--zoom` / `--xcl` must not write under the arm's own name: that is the
    view-gate, predicted-memory, full-view, non-exclusive result."""
    if (gate != "view" or round2 != "pred" or zoom is not None or xcl != "none") and out == arm:
        raise SystemExit(f"--gate {gate} --round2 {round2} --zoom {zoom} --xcl {xcl} with --out {out}: that name is the default result (pass another --out)")


# S1-E11 (docs/stage1-E11-prereg.md section 2): in-view object exclusivity by the decoder's own logits.
XCL_MODES = ("none", "out", "outmem", "out_nocap", "outs")
XCL_TAU_KEEP = 0.5          # nested-mask cap: a mask that would lose >= this fraction of its pixels is exempt (kept whole)
XCL_MEM_CLAMP = -10.0       # the clamp SAM 3 uses in _apply_non_overlapping_constraints (sigmoid(-10) = 4.5e-5)
# S1-E14 (docs/stage1-E14-prereg.md section 2.2): `outs` = `out` with the output threshold lowered to L > -XCL_SHIFT
# (a comparison, not an addition: the argmax stays on the unshifted logits); round-2 output only.
XCL_SHIFT = 0.5


def exclusive_masks(L, tau_keep=XCL_TAU_KEEP, shift=0.0):
    """X1 / D / S rule on output-resolution logits L [B,h,w] float -> (keep [B,h,w] bool, counts, per-object records).

    pos = L > 0 (S1-E14 S: L > -shift); w = per-pixel argmax over objects of the UNSHIFTED L (exact ties -> the
    lowest index; absent objects carry NO_OBJ_SCORE and never win); a mask that would lose >= tau_keep of its pos
    pixels is exempt and kept whole (tau_keep None = no cap, D); keep = pos & (win | exempt).  keep is a subset of
    pos; only pixels claimed by >= 2 masks can change; at most one non-exempt mask survives at a contested pixel."""
    import torch
    B = L.shape[0]
    pos = L > 0 if not shift else L > -float(shift)
    w = L.argmax(0)                                                            # [h,w]
    win = w[None] == torch.arange(B, device=L.device)[:, None, None]           # [B,h,w]
    del w
    npos = pos.flatten(1).sum(1)                                               # [B]
    lost = (pos & ~win).flatten(1).sum(1)                                      # [B]
    if tau_keep is None:
        exempt = torch.zeros(B, dtype=torch.bool, device=L.device)
    else:
        exempt = (npos > 0) & (lost.float() >= tau_keep * npos.float())
    keep = pos & (win | exempt[:, None, None])
    del win
    removed = pos & ~keep
    contested = pos.sum(0) >= 2
    nkeep = keep.flatten(1).sum(1)
    counts = dict(contested_px=int(contested.sum()), removed_entries=int(removed.sum()),
                  masks_changed=int(removed.flatten(1).any(1).sum()), masks_exempt=int(exempt.sum()),
                  masks_emptied=int(((npos > 0) & (nkeep == 0)).sum()), masks_present=int((npos > 0).sum()))
    recs = [dict(px=int(npos[b]), lost=int(lost[b]), exempt=bool(exempt[b]), kept=int(nkeep[b]),
                 removed=int(removed[b].sum())) for b in range(B)]
    del removed, contested
    return keep, counts, recs


def exclusive_logits_1008(logits, eligible, tau_keep=XCL_TAU_KEEP):
    """X2 rule on round-1 logits [B,1,1008,1008] before they are encoded as round-2 memory: the X1 rule among the
    ELIGIBLE objects only (the reproj-confident ones); a losing, non-exempt pixel is clamped to XCL_MEM_CLAMP
    (= background for the memory encoder); non-eligible objects' logits are untouched.  -> (logits_mem, counts)."""
    import torch
    from sam3.model.sam3_tracker_base import NO_OBJ_SCORE
    B = logits.shape[0]
    elig = torch.as_tensor(eligible, dtype=torch.bool, device=logits.device)
    zero = dict(contested_cells=0, clamped_cells=0, masks_exempt=0, eligible=int(elig.sum()))
    if B < 2 or int(elig.sum()) < 2:
        return logits, zero
    Lm = logits[:, 0]
    e3 = elig[:, None, None]
    pos = (Lm > 0) & e3
    w = torch.where(e3, Lm, torch.full_like(Lm, NO_OBJ_SCORE)).argmax(0)
    win = w[None] == torch.arange(B, device=logits.device)[:, None, None]
    del w
    npos = pos.flatten(1).sum(1)
    lost = (pos & ~win).flatten(1).sum(1)
    exempt = torch.zeros(B, dtype=torch.bool, device=logits.device) if tau_keep is None else \
        (npos > 0) & (lost.float() >= tau_keep * npos.float())
    clamp = pos & ~win & ~exempt[:, None, None]
    del win
    out = torch.where(clamp, Lm.clamp(max=XCL_MEM_CLAMP), Lm)[:, None]
    counts = dict(contested_cells=int((pos.sum(0) >= 2).sum()), clamped_cells=int(clamp.sum()),
                  masks_exempt=int(exempt.sum()), eligible=int(elig.sum()))
    del clamp, pos
    return out, counts


def object_conf_mode(mode, presence, ious, overlap, tau=TAU_CONF, theta=THETA_REPROJ):
    """Per-object confidence under one gate signal (GeoSet-2 §8), numpy in / numpy bool out.
    obj: present & iou >= tau; absent: present; reproj: present & overlap >= theta, IoU rule where overlap is nan."""
    presence, ious, overlap = (np.asarray(x, np.float64) for x in (presence, ious, overlap))
    present = presence > 0
    if mode == "absent":
        return present
    by_iou = ious >= tau
    if mode == "obj":
        return present & by_iou
    assert mode == "reproj", mode
    return present & np.where(np.isnan(overlap), by_iou, overlap >= theta)


def reproj_overlap(pred_bool, pts_full, valid_full, depth_ref_full, K_ref_full, ref_masks_bool, min_valid=MIN_VALID):
    """Fraction of each predicted mask's pixels that reproject (Pi3 points in the reference frame, occlusion-checked)
    onto the reference GT mask of the same object.  pred_bool [B,h,w], pts_full [h,w,3], valid_full [h,w],
    depth_ref_full [h_r,w_r], K_ref_full 3x3, ref_masks_bool [B,h_r,w_r] -> (overlap [B] float, nan below `min_valid`;
    n_valid [B] long)."""
    import torch
    from geomem.warp_torch import reproject_coords, sample_reference
    h_r, w_r = ref_masks_bool.shape[-2:]
    grid, valid = reproject_coords(pts_full, K_ref_full, (h_r, w_r), depth_ref=depth_ref_full)
    valid = valid & valid_full
    on_gt = sample_reference(ref_masks_bool.float()[:, None], grid)[:, 0] > 0.5         # [B,h,w]
    sel = pred_bool & valid[None]
    n = sel.sum(dim=(1, 2))
    ov = (sel & on_gt).sum(dim=(1, 2)).float() / n.clamp(min=1).float()
    ov = torch.where(n >= min_valid, ov, torch.full_like(ov, float("nan")))
    return ov, n


def project_reference_masks(ref_masks_bool, pts_full, valid_full, depth_ref_full, K_ref_full):
    """GeoProj: the reference GT masks seen from a neighbour view.  Every pixel of the view (Pi3 point in the reference
    frame) is reprojected into the reference (occlusion-checked) and samples the GT mask there.  Returns
    (masks [B,h,w] bool — object label where visible from the reference, False elsewhere; vis [h,w] bool)."""
    import torch
    from geomem.warp_torch import reproject_coords, sample_reference
    h_r, w_r = ref_masks_bool.shape[-2:]
    grid, valid = reproject_coords(pts_full, K_ref_full, (h_r, w_r), depth_ref=depth_ref_full)
    vis = valid & valid_full
    on_gt = sample_reference(ref_masks_bool.float()[:, None], grid)[:, 0] > 0.5
    return on_gt & vis[None], vis


def reproj_inputs(cache, cam, ref_cam, hw, device):
    """Full-resolution reprojection inputs of view `cam` into the reference (as geomem.train.ssl_view builds them)."""
    import cv2
    import torch
    from geomem.pointmap import CONF_MIN, local_points, to_reference
    h, w = hw
    zu = np.load(os.path.join(cache, f"{cam}.npz"))
    zr = np.load(os.path.join(cache, f"{ref_cam}.npz"))
    pts = to_reference(local_points(zu["depth"], zu["K"]), zu["c2w"], zr["c2w"]).astype(np.float32)
    valid_u = ((zu["conf"] > CONF_MIN) & (zu["depth"] > 0)).astype(np.uint8)
    pts_full = cv2.resize(pts, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_full = cv2.resize(valid_u, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    depth_ref_full = cv2.resize(zr["depth"], (w, h), interpolation=cv2.INTER_NEAREST)
    hw_work = zr["depth"].shape
    sx, sy = w / hw_work[1], h / hw_work[0]
    K = zr["K"]
    K_full = np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5],
                       [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5], [0, 0, 1]], np.float32)
    return (torch.from_numpy(pts_full).to(device), torch.from_numpy(valid_full).to(device),
            torch.from_numpy(depth_ref_full).to(device), torch.from_numpy(K_full).to(device))


def r1_aux_outs(out, r1_out):
    """Round-1 result folder(s) run_all must clear: `<out>_r1` by default, a given name, or none ("none")."""
    if r1_out == "none":
        return []
    return [r1_out if r1_out else out + "_r1"]


def view_confident(presence, ious, tau=TAU_CONF):
    present = presence > 0
    if not bool(present.any()):
        return False
    return bool(ious[present].float().mean().item() >= tau)


def hub_stats(mass, ref_index):
    """mass [L, M, N] attention mass per view -> entropy (normalised by log N), reference-view fraction and
    top-3-view fraction, each averaged over layers and objects."""
    m = np.asarray(mass, np.float64)
    p = m / np.clip(m.sum(-1, keepdims=True), 1e-12, None)
    n = p.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(p > 0, p * np.log(p), 0.0)).sum(-1) / (np.log(n) if n > 1 else 1.0)
    top3 = np.sort(p, axis=-1)[..., -3:].sum(-1)
    return {"entropy_mean": float(ent.mean()), "ref_frac_mean": float(p[..., ref_index].mean()),
            "top3_frac_mean": float(top3.mean())}


def assert_no_leak(ckpt_config, scene):
    if scene in ckpt_config.get("scenes", []):
        raise RuntimeError(f"checkpoint of fold{ckpt_config.get('fold')} was trained on {scene}; refusing to evaluate it")


def load_checkpoint_into(wrapper, geo, ckpt_path):
    """Load a GeoMem checkpoint: geo heads, and the memory attention for `geomem_ft`.

    Requires the autocast weight cache to be off (methods/sam3_single.load_model does this); with the
    tracker's never-exited autocast context the cache would otherwise keep the bf16 casts of whatever
    weights the first forward saw, and later loads would be ignored or, before any forward, poison it."""
    import torch
    assert not torch.is_autocast_cache_enabled(), "autocast weight cache must be disabled before loading weights"
    ck = torch.load(ckpt_path, map_location="cuda")
    geo.load_state_dict(ck["geo_embed"])
    if "geo_bias" in ck:
        assert wrapper.geo_bias is not None, "checkpoint has geo_bias but the wrapper has none"
        wrapper.geo_bias.load_state_dict(ck["geo_bias"])
    if "memory_attention" in ck:
        wrapper.tracker.transformer.encoder.load_state_dict(ck["memory_attention"])
    if "mask_decoder" in ck:                 # S1-E9 DEC_ARMS
        wrapper.tracker.sam_mask_decoder.load_state_dict(ck["mask_decoder"])
    if "fusion" in ck:
        assert getattr(wrapper, "fusion", None) is not None, "checkpoint has a fusion module but the wrapper has none"
        wrapper.fusion.load_state_dict(ck["fusion"])
    if "query_hub" in ck:
        assert getattr(wrapper, "hub", None) is not None, "checkpoint has a query hub but the wrapper has none"
        wrapper.hub.load_state_dict(ck["query_hub"])
    return ck


class FileSink:
    """Where decode_scene puts its outputs: the benchmark layout Data/<scene>/Result/<out>/ (the CLI).
    stage1.MemorySink is the in-process alternative (the masks stay in memory for stage 2)."""

    def __init__(self, data, scene_name):
        self.data, self.scene_name = data, scene_name

    def mask(self, out_name, cam, obj, mask_bool):
        common.write_result(self.data, self.scene_name, out_name, cam, obj, mask_bool)

    def json(self, out_name, filename, obj):
        os.makedirs(result_dir(self.data, self.scene_name, out_name), exist_ok=True)
        with open(os.path.join(result_dir(self.data, self.scene_name, out_name), filename), "w") as f:
            json.dump(obj, f, indent=1)

    def finish_aux(self, scene, out_name, config):
        """A round-1 folder is complete: the reference GT and its RUN.json (run_all only does both for --out)."""
        common.copy_reference(scene, self.data, out_name)
        common.write_run(self.data, scene.name, out_name, None, None, config, None)


def decode_scene(scene, args, sink=None):
    """Decode one scene; masks go to `sink` (default: FileSink, the Result/<out> folders)."""
    import torch
    from geomem import sam3_io
    from geomem.embed import grid_tensors
    from geomem.train import arm_flags, build_wrapper
    from geomem.wrapper import MemorySet
    from methods import sam3_single
    model = decode_scene.model
    if model is None:
        model = sam3_single.load_model()
        decode_scene.model = model
        decode_scene.init_state = snapshot_init(model)      # the lazy path must restore too (review 2026-10-14)
    arm = args.arm
    flags = arm_flags(arm)
    tau, k = getattr(args, "tau", TAU_CONF), getattr(args, "k", K_NEIGHBOURS)
    gate = getattr(args, "gate", "view")
    round2 = getattr(args, "round2", "pred")           # GeoProj: pred | proj_bin | proj_soft
    zoom, zoom_rope = getattr(args, "zoom", None), getattr(args, "zoom_rope", "grid")   # S1-E10 crop-zoom pass (None = off)
    r1_outs = r1_aux_outs(args.out, getattr(args, "r1_out", None))
    two_round = arm in SET_ARMS
    pair_mode = arm in PAIR_ARMS
    if zoom is not None:
        from geomem import zoom as zoom_mod
        assert zoom == zoom_mod.ZOOM and zoom_rope in ("grid", "remap"), (zoom, zoom_rope)
        assert two_round and round2 == "pred" and not pair_mode and arm not in FUSE_ARMS and arm not in ALIGN_ARMS, \
            f"--zoom is registered for the two-round predicted-memory arms only, not {arm}"
    xcl = getattr(args, "xcl", "none") or "none"           # S1-E11 in-view exclusivity: none | out | outmem | out_nocap; S1-E14: outs
    assert xcl in XCL_MODES, xcl
    if xcl != "none":
        assert zoom is None, "--xcl and --zoom are registered separately (no combination arm)"
        assert two_round and round2 == "pred" and not pair_mode and gate in OBJ_GATES and arm not in FUSE_ARMS and arm not in ALIGN_ARMS, \
            f"--xcl is registered for the two-round predicted-memory arms with an object gate only, not {arm}/{gate}"
    config = {"arm": arm, "untrained": bool(args.untrained), **flags, "two_round": two_round}
    wrapper = build_wrapper(model, arm)
    geo = wrapper.geo_embed
    ck = None
    if not args.untrained:
        folds = with_external_fold(load_folds(), scene.name, getattr(scene, "external_fold", None))
        ckpt_path = checkpoint_for(scene.name, arm, args.runs, folds)
        with open(os.path.join(os.path.dirname(ckpt_path), "config.json")) as f:
            ck_cfg = json.load(f)
        assert_no_leak(ck_cfg, scene.name)
        ck = load_checkpoint_into(wrapper, geo, ckpt_path)
        config.update({"fold": ck_cfg["fold"], "ckpt_sha256": ck_cfg["ckpt_sha256"], "train_git": ck_cfg.get("git"),
                       "train_steps": ck_cfg.get("steps"), "epochs": ck_cfg.get("epochs"), "frame_stride": ck_cfg.get("frame_stride")})
    wrapper.eval()

    cache = cache_dir(args.data, scene.name, "pi3")
    masks_np = np.stack([labelmap_to_mask(scene.ref_labels, o) for o in scene.objects])
    masks = torch.from_numpy(masks_np).cuda()
    B = masks.shape[0]
    targets = [c for c in scene.cams if c != scene.ref_cam]
    written = 0
    if sink is None:
        sink = FileSink(args.data, scene.name)

    def to_bool(logits_1008):
        return (sam3_io.logits_to_original(logits_1008, scene.hw) > 0)[:, 0].cpu().numpy()

    def write_bool(out_name, cam, pred):
        n = 0
        for i, o in enumerate(scene.objects):
            if pred[i].any():
                sink.mask(out_name, cam, o, pred[i])
                n += 1
        return n

    def write(out_name, cam, logits_1008):
        return write_bool(out_name, cam, to_bool(logits_1008))

    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref_img, h, w = sam3_io.load_image(scene.image_paths[scene.ref_cam])
            assert (h, w) == scene.hw
            ref_bb = sam3_io.backbone_features(model, ref_img)
            vg_ref = view_grid(cache, scene.ref_cam, scene.ref_cam)
            p_ref, v_ref = grid_tensors(vg_ref, "cuda")
            ref = wrapper.encode_reference(ref_img, ref_bb, masks, p_ref, v_ref)

            if pair_mode:
                # every view's features and grid first; one hub pass; then each target decoded with the query
                bbs, grids = {scene.ref_cam: ref_bb}, {scene.ref_cam: (p_ref, v_ref)}
                for cam in targets:
                    img, _, _ = sam3_io.load_image(scene.image_paths[cam])
                    bbs[cam] = sam3_io.backbone_features(model, img)
                    grids[cam] = grid_tensors(view_grid(cache, cam, scene.ref_cam), "cuda")
                    del img
                query = None
                if getattr(wrapper, "hub", None) is not None:
                    order = [scene.ref_cam] + targets
                    hub = wrapper.run_hub(ref, [bbs[c] for c in order], torch.stack([grids[c][0] for c in order]),
                                          torch.stack([grids[c][1] for c in order]), masks_np,
                                          np.asarray(vg_ref["p"], np.float32).reshape(-1, 3),
                                          np.asarray(vg_ref["valid"], bool).reshape(-1))
                    query = hub["q"]
                    config.update(hub_stats(hub["mass"].float().cpu().numpy(), 0))
                    config.update({"n_hub_views": len(order), "cloud_size_mean": float(np.mean(hub["cloud_sizes"])),
                                   "n_empty_clouds": int(sum(c == 0 for c in hub["cloud_sizes"]))})
                for cam in targets:
                    out = wrapper.decode_view(bbs[cam], B, ref, *grids[cam], query=query)
                    written += write(args.out, cam, out["logits_1008"])
                targets = []                                 # the generic loop below has nothing left to do

            # GeoFuse spec 4: every view's backbone first (kept on the GPU, ~0.5 GB for 46 views) and its Pi3 geometry,
            # so each target can fuse all views' level-72 features aligned to its grid
            fuse = arm in FUSE_ARMS
            align_mem = arm in ALIGN_ARMS
            fstats, align_for = [], None
            if fuse or align_mem:
                from geomem.align import align_views, reference_frame, view_geometry
                bbs = {scene.ref_cam: ref_bb}
                for cam in targets:
                    img, _, _ = sam3_io.load_image(scene.image_paths[cam])
                    bbs[cam] = sam3_io.backbone_features(model, img)
                    del img
                fuse_views = [dict(feat=bbs[c]["backbone_fpn"][2][0], **view_geometry(cache, c, "cuda")) for c in scene.cams]
                fscale, fc2w_ref = reference_frame(cache, scene.ref_cam, "cuda")

                from geomem.align import fusion_views

                def align_for(p_v, v_v, cam=None):
                    views = fusion_views(fuse_views, scene.cams, exclude=cam) if flags.get("fuse_exclude_self") else fuse_views
                    return align_views(p_v, v_v, fscale, fc2w_ref, views, two_sided=flags.get("fuse_two_sided", False))
            base_ref = ref
            align_pool = {}
            if align_mem:
                # GeoAlign spec 4: the k nearest cameras aligned onto the REFERENCE grid (two-sided validity, holes filled
                # with the reference) + the GT masks -> memory views; no prediction, no gate.  k+1 views are encoded once
                # per scene and each target uses the k nearest that are not itself (review follow-up: a target is never
                # its own memory source).
                acentres = camera_centres(cache, scene.cams)
                pool = aligned_neighbours(scene.ref_cam, scene.cams, acentres, k + 1)
                W_all, valid_all = align_views(p_ref, v_ref, fscale, fc2w_ref, [fuse_views[scene.cams.index(c)] for c in pool],
                                               two_sided=True)
                align_pool = {c: wrapper.encode_aligned(ref_img, ref_bb, masks, p_ref, v_ref, W_all[i], valid_all[i], c)
                              for i, c in enumerate(pool)}
                align_valid = {c: float(valid_all[i].float().mean()) for i, c in enumerate(pool)}
                config.update({"align_views": min(k, len(pool)), "align_pool": pool,
                               "align_valid_mean": float(np.mean([align_valid[c] for c in pool[:k]])),
                               "align_targets_substituted": int(sum(c in pool[:k] for c in targets))})
                del W_all

                def memory_for(cam):
                    """c_ini + the k nearest aligned views that are not the target itself."""
                    chosen = aligned_neighbours(scene.ref_cam, pool, acentres, k, exclude=cam)
                    return MemorySet(base_ref.views + [align_pool[c] for c in chosen])
            else:
                def memory_for(cam):
                    return base_ref

            round1 = {}
            for cam in targets:
                if fuse or align_mem:
                    bb = bbs[cam]
                else:
                    img, _, _ = sam3_io.load_image(scene.image_paths[cam])
                    bb = sam3_io.backbone_features(model, img)
                p_v, v_v = grid_tensors(view_grid(cache, cam, scene.ref_cam), "cuda")
                aligned = align_for(p_v, v_v, cam) if fuse else None
                out = wrapper.decode_view(bb, B, memory_for(cam), p_v, v_v, aligned=aligned)
                if fuse:
                    fstats.append(dict(wrapper.last_fusion_stats, valid_align_frac=float(aligned[1].float().mean())))
                    del aligned
                if two_round:
                    round1[cam] = dict(bb=bb, p=p_v, valid=v_v, logits=out["logits_1008"], presence=out["presence"],
                                       ious=out["ious"], obj_ptr=out["obj_ptr"])
                    for r1_out in r1_outs:
                        written += write(r1_out, cam, out["logits_1008"])
                else:
                    written += write(args.out, cam, out["logits_1008"])

            if two_round:
                # §8: reprojection overlap of every round-1 mask with the reference GT (recorded for every gate)
                for c, r in round1.items():
                    pred = (sam3_io.logits_to_original(r["logits"], scene.hw) > 0)[:, 0]
                    ov, n_valid = reproj_overlap(pred, *reproj_inputs(cache, c, scene.ref_cam, scene.hw, "cuda"), masks)
                    r["overlap"], r["n_valid"] = ov.cpu().numpy(), n_valid.cpu().numpy()
                    del pred
                # round-1 confidence record (spec 3.2), into the two-round folder so `--r1-out none` keeps it
                rec = {c: {str(o): {"iou": float(r["ious"][i]), "presence": float(r["presence"][i]),
                                    "overlap": None if np.isnan(r["overlap"][i]) else float(r["overlap"][i]),
                                    "n_valid": int(r["n_valid"][i])}
                           for i, o in enumerate(scene.objects)} for c, r in round1.items()}
                sink.json(args.out, "round1_conf.json", rec)
                centres = camera_centres(cache, scene.cams)
                n_mem, written2, n_gated, n_pairs, dropped = [], 0, 0, 0, 0
                proj_stats = []
                zoom_recs, zoom_counts = {}, Counter()
                xcl_recs, xcl_counts, xmem_counts = {}, Counter(), Counter()
                if round2 != "pred":
                    # GeoProj: prediction-free second-round memory — the neighbour's OWN features with the reference GT
                    # masks PROJECTED into it (pixels not visible from the reference: background in proj_bin, logit 0 in
                    # proj_soft); conf = projected mask non-empty (the "absent" rule); round-1 predictions unused.
                    mem_views = {}
                    for c, r in round1.items():
                        pm, vis = project_reference_masks(masks, *reproj_inputs(cache, c, scene.ref_cam, scene.hw, "cuda"))
                        present = pm.flatten(1).any(1)
                        proj_stats.append((float(vis.float().mean()), float(present.float().mean())))
                        conf = present
                        if round2 == "proj_bin":
                            img_c, _, _ = sam3_io.load_image(scene.image_paths[c])
                            view = wrapper.encode_reference(img_c, r["bb"], pm, r["p"], r["valid"], name=c).views[0]
                            del img_c
                            mem_views[c] = replace(view, conf=conf)
                        else:
                            lab = sam3_io.mask_inputs(pm, 1008)[:, 0] > 0.5
                            vis_1008 = sam3_io.mask_inputs(vis[None], 1008)[0, 0] > 0.5
                            logits = torch.where(vis_1008[None], torch.where(lab, 10.0, -10.0), 0.0)[:, None]
                            presence = torch.where(present, 10.0, -10.0)
                            obj_ptr = ref.views[0].obj_ptr_tokens.permute(1, 0, 2).reshape(B, -1).float()
                            mem_views[c] = wrapper.encode_memory_from_prediction(
                                r["bb"], B, logits, presence, obj_ptr, r["p"], r["valid"], c, conf=conf)
                    config.update({"round2": round2, "proj_vis_mean": float(np.mean([x[0] for x in proj_stats])),
                                   "proj_objects_visible_frac": float(np.mean([x[1] for x in proj_stats]))})
                elif gate == "view":
                    confident = {c: view_confident(r["presence"], r["ious"], tau) for c, r in round1.items()}
                    mem_views = {c: wrapper.encode_memory_from_prediction(r["bb"], B, r["logits"], r["presence"], r["obj_ptr"],
                                                                          r["p"], r["valid"], c)
                                 for c, r in round1.items() if confident[c]}
                    config["n_confident_views"] = int(sum(confident.values()))
                else:
                    mem_views = {}
                    for c, r in round1.items():
                        conf = object_conf_mode(gate, r["presence"].float().cpu().numpy(), r["ious"].float().cpu().numpy(),
                                                r["overlap"], tau=tau)
                        logits_mem = r["logits"]
                        if xcl == "outmem":
                            # S1-E11 X2: exclusivity among this view's confident objects before the memory encoding
                            logits_mem, xm = exclusive_logits_1008(r["logits"], conf, XCL_TAU_KEEP)
                            xmem_counts.update(xm)
                        mem_views[c] = wrapper.encode_memory_from_prediction(
                            r["bb"], B, logits_mem, r["presence"], r["obj_ptr"], r["p"], r["valid"], c,
                            conf=torch.from_numpy(conf).to(r["presence"].device))
                        del logits_mem
                for cam in targets:
                    if gate == "view" and round2 == "pred":
                        neigh = select_neighbours(cam, centres, confident, k)
                    else:
                        # as in `view` mode the reference may be among the k nearest; it is already in the set
                        cand = [c for c in select_candidates(cam, centres, k) if c != scene.ref_cam]
                        neigh = [c for c in cand if bool(mem_views[c].conf.any())]
                        dropped += len(cand) - len(neigh)
                        n_gated += (len(cand) - len(neigh)) * B          # a dropped view = all its objects gated
                        n_pairs += len(cand) * B
                    mem = MemorySet(memory_for(cam).views + [mem_views[c] for c in neigh])
                    n_mem.append(len(mem.views))
                    n_gated += mem.n_gated()
                    aligned = align_for(round1[cam]["p"], round1[cam]["valid"], cam) if fuse else None
                    out = wrapper.decode_view(round1[cam]["bb"], B, mem, round1[cam]["p"], round1[cam]["valid"], chunk=OBJ_CHUNK,
                                              aligned=aligned)
                    if zoom is None and xcl == "none":
                        written2 += write(args.out, cam, out["logits_1008"])
                    elif zoom is None:
                        # S1-E11 X1 / D: per-pixel logit argmax at the output resolution (the to_bool tensor), nested-mask cap;
                        # S1-E14 S (outs): the same with pos = L > -XCL_SHIFT
                        L = sam3_io.logits_to_original(out["logits_1008"], scene.hw)[:, 0]
                        del out
                        keep, xc, xr = exclusive_masks(L, None if xcl == "out_nocap" else XCL_TAU_KEEP,
                                                       XCL_SHIFT if xcl == "outs" else 0.0)
                        del L
                        xcl_counts.update(xc)
                        xcl_recs[cam] = {str(int(o)): xr[i] for i, o in enumerate(scene.objects)}
                        written2 += write_bool(args.out, cam, keep.cpu().numpy())
                        del keep
                    else:
                        # S1-E10: the crop-zoom pass on the round-2 masks (same memory set), band-clipped paste
                        pred = to_bool(out["logits_1008"])
                        pres_full, iou_full = out["presence"].float().cpu().numpy(), out["ious"].float().cpu().numpy()
                        del out
                        pred, zoom_recs[cam], zc = zoom_mod.refine_camera(
                            wrapper, model, mem, scene.image_paths[cam], cache, cam, scene.ref_cam, scene.hw, pred,
                            pres_full, iou_full, scene.objects, zoom=zoom, zoom_rope=zoom_rope)
                        zoom_counts.update(zc)
                        written2 += write_bool(args.out, cam, pred)
                if fuse:
                    config.update({f"fuse_{k2}" if k2 != "valid_align_frac" else k2: float(np.mean([f[k2] for f in fstats]))
                                   for k2 in ("self_mass", "view_entropy", "valid_views_mean", "valid_align_frac")})
                config.update({"gate": gate, "tau": tau, "k": k, "mem_per_view_mean": float(np.mean(n_mem)),
                               "views_with_ref_only_frac": float(np.mean([n == 1 for n in n_mem])),
                               "n_targets": len(targets), "masks_written_r1": written})
                if xcl != "none":
                    sink.json(args.out, "xcl_conf.json", xcl_recs)
                    config.update({"xcl": xcl, "xcl_tau_keep": None if xcl == "out_nocap" else XCL_TAU_KEEP})
                    if xcl == "outs":
                        config["xcl_shift"] = XCL_SHIFT
                    config.update({f"n_xcl_{key}": int(xcl_counts[key]) for key in
                                   ("contested_px", "removed_entries", "masks_changed", "masks_exempt", "masks_emptied", "masks_present")})
                    if xcl == "outmem":
                        config.update({f"n_xmem_{key}": int(xmem_counts[key]) for key in ("contested_cells", "clamped_cells", "masks_exempt", "eligible")})
                if zoom is not None:
                    sink.json(args.out, "zoom_conf.json", zoom_recs)
                    config.update({"zoom": zoom, "zoom_rope": zoom_rope, "zoom_margin": zoom_mod.MARGIN, "zoom_tau_iou": zoom_mod.TAU_IOU,
                                   "band_px": zoom_mod.band_px(scene.hw), "zoom_seconds": float(zoom_counts["seconds"])})
                    config.update({f"n_zoom_{key}": int(zoom_counts[key]) for key in
                                   ("candidates", "run", "skipped_large", "kept_absent", "kept_iou", "kept_empty", "accepted", "changed")})
                    config.update({"zoom_px_added": int(zoom_counts["px_added"]), "zoom_px_removed": int(zoom_counts["px_removed"])})
                if gate in OBJ_GATES or round2 != "pred":
                    config.update({"gated_pairs_frac": float(n_gated / n_pairs) if n_pairs else 0.0,
                                   "views_dropped": int(dropped)})
                written = written2
                for r1_out in r1_outs:
                    sink.finish_aux(scene, r1_out, dict(config, round=1))
    finally:
        if ck is not None:
            restore_init(model, decode_scene.init_state, ck)
        wrapper.close()
    torch.cuda.empty_cache()
    config["masks_written"] = written
    config["n_objects"] = B
    return config


decode_scene.model = None
decode_scene.init_state = None

# the tracker modules a checkpoint may overwrite (checkpoint key -> module); their initial weights are snapshotted once
# per process and put back after every scene (the next scene's fold, and stage 2 of the in-process pipeline, need them)
INIT_MODULES = {"memory_attention": lambda model: model.tracker.transformer.encoder,
                "mask_decoder": lambda model: model.tracker.sam_mask_decoder}


def snapshot_init(model):
    """{checkpoint key: deep-copied initial state_dict} of INIT_MODULES."""
    import copy
    return {k: copy.deepcopy(get(model).state_dict()) for k, get in INIT_MODULES.items()}


def restore_init(model, init_state, ck):
    """Put the initial weights back for every INIT_MODULES key the checkpoint `ck` carried."""
    assert init_state is not None, "decode_scene.init_state was never snapshotted (call snapshot_init after load_model)"
    for k, get in INIT_MODULES.items():
        if k in ck:
            get(model).load_state_dict(init_state[k])


def _extra(ap):
    ap.add_argument("--arm", choices=ARMS, default="geomem")
    ap.add_argument("--untrained", action="store_true", help="zero-initialised heads: the `star` control")
    ap.add_argument("--runs", default=os.path.join(common.REPO, "geomem", "runs"))
    ap.add_argument("--tau", type=float, default=TAU_CONF, help="memory-set confidence threshold (fixed; tests override)")
    ap.add_argument("--k", type=int, default=K_NEIGHBOURS, help="nearest neighbours considered for the memory set")
    ap.add_argument("--gate", choices=("view",) + OBJ_GATES, default="view",
                    help="round-2 memory gate: per view (mean IoU >= tau), or per object by IoU (obj), presence only "
                         "(absent) or reprojection overlap with the reference GT (reproj) — GeoSet-2 spec 3.1 / §8")
    ap.add_argument("--r1-out", default=None, help="round-1 result folder (default <out>_r1; 'none' writes no round-1 masks)")
    ap.add_argument("--zoom", type=int, choices=(2,), default=None,
                    help="S1-E10: 2x crop-zoom re-decode of every round-2 mask, band-clipped paste (docs/stage1-E10-prereg.md)")
    ap.add_argument("--zoom-rope", choices=("grid", "remap"), default="grid", dest="zoom_rope",
                    help="S1-E10: query RoPE of the crop decode: the crop lattice as is (C1) or remapped to full-view cells (C2)")
    ap.add_argument("--xcl", choices=XCL_MODES, default="none",
                    help="S1-E11: in-view object exclusivity by the decoder logits: out (X1, capped), outmem (X2, + round-1 memories), "
                         "out_nocap (D, diagnostic without the nested-mask cap); docs/stage1-E11-prereg.md.  S1-E14: outs (S, = out "
                         "with the output threshold L > -XCL_SHIFT = -0.5; docs/stage1-E14-prereg.md)")
    ap.add_argument("--round2", choices=("pred", "proj_bin", "proj_soft"), default="pred",
                    help="GeoProj: second-round memory from round-1 predictions (pred) or from the reference GT masks projected "
                         "into the neighbour (binary mask / soft logits with unknown = 0); no prediction, 'absent'-style conf")


def main():
    args = common.parse_args(__doc__, _extra)
    from methods import sam3_single
    decode_scene.model = sam3_single.load_model()
    decode_scene.init_state = snapshot_init(decode_scene.model)
    check_gate_out(args.arm, args.gate, args.out, args.round2, args.zoom, args.xcl)
    common.run_all(args, decode_scene, aux_outs=r1_aux_outs(args.out, args.r1_out) if args.arm in SET_ARMS else [])


if __name__ == "__main__":
    main()
