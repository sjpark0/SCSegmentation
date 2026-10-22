#!/usr/bin/env python3
"""geomem: decode every camera of the reference frame from the reference camera's memory with the
GeoMem-star adapter of the fold that did NOT train on the scene (spec 5).

    scripts/run_geomem.sh <scene|all> --arm {geomem,geomem_ft,geomem_ssl} [--out NAME]
    scripts/run_geomem.sh <scene|all> --untrained --out star          # control: zero-initialised heads

A missing checkpoint is an error for that scene (never a silent fall-back); a checkpoint whose
training scenes include the scene is refused.  Runs inside `scsam3`.
"""
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from geomem.folds import load_folds  # noqa: E402
from geomem.pointmap import view_grid  # noqa: E402
from methods import common  # noqa: E402
from mvseg2d.io import cache_dir, labelmap_to_mask, result_dir  # noqa: E402

from geomem.arms import ALIGN_ARMS, CONTINUE_ARMS, FUSE_ARMS, NOGEO_ARMS, QUERY_ARMS, SET2_ARMS  # noqa: E402

ARMS = ("geomem", "geomem_ft", "geomem_ssl", "geobias_ft", "geobias_norope_ft", "geobias_only_ft", "geoset_ft") \
    + QUERY_ARMS + CONTINUE_ARMS + SET2_ARMS + FUSE_ARMS + ALIGN_ARMS + NOGEO_ARMS
SET_ARMS = ("geoset_ft",) + SET2_ARMS + FUSE_ARMS + ALIGN_ARMS + NOGEO_ARMS   # S1-E16 NG: two-round, predicted memory
PAIR_ARMS = QUERY_ARMS + CONTINUE_ARMS   # GeoQuery: all views' features first, one hub pass, single-round decoding
TAU_CONF = 0.7     # spec 2.2: a view is a confident memory when mean predicted IoU over present objects >= tau
K_NEIGHBOURS = 3
NBR_MODES = ("pi3", "index", "feat")   # S1-E16: how the k nearest neighbours are measured -- Pi3 camera centres (the
                                       # unchanged path), the rig index (|number difference|), or pooled SAM 3 features
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


def camera_index_coords(cams):
    """S1-E16 NG-idx: one coordinate per camera = the number in its name (cam9 -> 9, camera_0012 -> 12); the k nearest
    = the k closest rig numbers.  A name without a digit is refused."""
    out = {}
    for c in cams:
        digits = re.sub(r"\D", "", c)
        if not digits:
            raise ValueError(f"--nbr index: no camera number in {c!r}")
        out[c] = np.array([float(int(digits))], dtype=np.float64)
    return out


def camera_feat_coords(model, image_paths, cams):
    """S1-E16 NG-feat: one unit vector per camera = the spatial mean of the SAM 3 backbone's coarsest map (level 2 of
    backbone_fpn, 256 channels) of that camera's image, L2-normalised, so the k nearest by Euclidean distance are the k
    most cosine-similar views.  Computed for every camera of the scene from the images alone (no geometry)."""
    import torch
    from geomem import sam3_io
    out = {}
    with torch.no_grad():
        for c in cams:
            img, _, _ = sam3_io.load_image(image_paths[c])
            f = sam3_io.backbone_features(model, img)["backbone_fpn"][2][0].float().mean(dim=(1, 2))
            out[c] = torch.nn.functional.normalize(f, dim=0).cpu().numpy().astype(np.float64)
    return out


def camera_coords(nbr, cache, scene, model=None):
    """The coordinates the neighbour rules (select_neighbours / select_candidates) measure distances in: Pi3 centres
    (`pi3`, the unchanged path) or the S1-E16 geometry-free alternatives."""
    if nbr == "pi3":
        return camera_centres(cache, scene.cams)
    if nbr == "index":
        return camera_index_coords(scene.cams)
    if nbr == "feat":
        return camera_feat_coords(model, scene.image_paths, scene.cams)
    raise ValueError(f"--nbr {nbr!r}: not one of {NBR_MODES}")


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


def check_gate_out(arm, gate, out, round2="pred", zoom=None, xcl="none", zero_bias=False, relay=None, nbr="pi3"):
    """`--gate obj` / `--round2 proj_*` / `--zoom` / `--xcl` / `--zero-bias` / `--relay` / `--nbr` must not write under the arm's own
    name: that is the view-gate, predicted-memory, full-view, non-exclusive, trained-bias, no-relay result."""
    if (gate != "view" or round2 != "pred" or zoom is not None or xcl != "none" or zero_bias or relay is not None
            or (nbr or "pi3") != "pi3") and out == arm:
        raise SystemExit(f"--gate {gate} --round2 {round2} --zoom {zoom} --xcl {xcl} --zero-bias {bool(zero_bias)} --nbr {nbr} "
                         f"--relay {relay} with --out {out}: that name is the default result (pass another --out)")


def zero_geo_bias(wrapper):
    """Ring diagnosis D3 (stage1/docs/ring-diagnosis-2026-10-20.md §2.6), diagnostic only: every GeoBias parameter (the RBF
    weights and b_none) set to 0 after the checkpoint is loaded, so the 3D bias adds exactly 0 to every memory
    cross-attention logit while the fine-tuned memory attention stays.  The attention mask is still passed (zeros +
    the gate), so the attention kernel is the same as with the trained bias.  -> max |parameter| before zeroing."""
    import torch
    assert wrapper.use_bias and wrapper.geo_bias is not None, "--zero-bias needs an arm with the 3D bias (use_bias)"
    with torch.no_grad():
        before = max(float(p.detach().abs().max()) for p in wrapper.geo_bias.parameters())
        for p in wrapper.geo_bias.parameters():
            p.zero_()
    return before


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


# ---------------------------------------------------------------------------------------------------------------------
# S1-E15 (docs/stage1-E15-prereg.md): ring fix 1 = the 3D identity check (Fix A, geomem/id3d.py) + relay decoding of the
# far views (Fix B), no training.  Off unless `--relay THETA` is given; then every other E15 flag has a default.
E15_SCOPES = ("relay", "global", "global_nf")       # R (Fix A on F only) / G (every view) / GN (G + S-view re-decodes)
E15_MODES = ("seq", "jacobi")                        # F views: relay in order / round-2 Jacobi with Fix A trust
E15_ORDERS = ("hop", "angle")                        # relay order with the reachability rule / pure a_v order (ABnh)
E15_MEMS = ("pool", "latest")                        # relay memory pool: decoded views / + round-1 outputs that PASS (ABml)
E15_TRUSTS = ("id3d", "reproj", "obj", "gt")         # relay memory confidence: Fix A / reproj (B-rep) / obj (B-obj) / GT
E15_FLAGGED = ("keep", "chain", "empty")             # §3.4 fallback: keep X / TCSVT chain mask / empty mask
E15_XCL = ("outs",)                                  # §3.7: the adopted exclusivity rule only (L6)
E15_THETAS = (20.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, math.inf)    # §3.7: the grid Theta + the stress 20 (L6, I7)
E15_KEYS = ("relay", "id3d", "id3d_scope", "relay_mode", "relay_order", "relay_mem", "relay_trust", "flagged",
            "chain_dir", "gt_dir")
CHAIN_IN, NO_LOGIT = 1.0, -1024.0                    # §3.4: chain fallback logits inside / outside; empty = NO_LOGIT
# §4 whitelist (L6, I7): the only flag combinations an E15 decode accepts.  Row -> (Fix A family given, scope, F mode,
# relay order, relay memory, memory trust, flagged fallback).  AB(theta), AB(s'), the stress arm and the main arm are
# the AB row at another theta / family.
E15_ROWS = {
    "AB": (True, "relay", "seq", "hop", "pool", "id3d", "keep"),
    "A": (True, "relay", "jacobi", "hop", "pool", "id3d", "keep"),
    "B-rep": (False, "relay", "seq", "hop", "pool", "reproj", "keep"),
    "B-obj": (False, "relay", "seq", "hop", "pool", "obj", "keep"),
    "ABc": (True, "relay", "seq", "hop", "pool", "id3d", "chain"),
    "Ac": (True, "relay", "jacobi", "hop", "pool", "id3d", "chain"),
    "ABe": (True, "relay", "seq", "hop", "pool", "id3d", "empty"),
    "ABnh": (True, "relay", "seq", "angle", "pool", "id3d", "keep"),
    "ABml": (True, "relay", "seq", "hop", "latest", "id3d", "keep"),
    "G": (True, "global", "seq", "hop", "pool", "id3d", "keep"),
    "GN": (True, "global_nf", "seq", "hop", "pool", "id3d", "keep"),
    "oracle": (False, "relay", "seq", "hop", "pool", "gt", "keep"),
}
E15_FOLD = {"T06_flag_ganggang_F": 2}                # I9 (M4): every E15 decode of T06 uses the fold-2 checkpoint


def e15_row(o):
    """The §4 whitelist row of resolved E15 options, or None."""
    key = (o["family"] is not None, o["scope"], o["mode"], o["order"], o["mem"], o["trust"], o["flagged"])
    rows = [name for name, row in E15_ROWS.items() if row == key]
    return rows[0] if rows else None


def e15_options(args):
    """The S1-E15 flags of `args` -> a validated dict, or None when `relay` is None (then every other E15 flag must be
    unset: the unchanged decode path).  Only the §4 whitelist rows, `--xcl outs` and the §3.7 thresholds pass."""
    from geomem import id3d
    vals = {k: getattr(args, k, None) for k in E15_KEYS}
    if vals["relay"] is None:
        stray = sorted(k for k, v in vals.items() if v not in (None, "off"))
        if stray:
            raise ValueError(f"S1-E15 flags {stray} need --relay THETA")
        return None
    theta = float(vals["relay"])
    if math.isnan(theta) or theta < 0:
        raise ValueError(f"--relay {vals['relay']}: a viewpoint threshold in degrees >= 0 (inf = no far view)")
    if theta not in E15_THETAS:
        raise ValueError(f"--relay {vals['relay']}: not a registered threshold {[_finite(x) for x in E15_THETAS]} "
                         "(stage1-E15-prereg §3.7)")
    xcl = getattr(args, "xcl", "none") or "none"
    if xcl not in E15_XCL:
        raise ValueError(f"--relay needs --xcl {' / '.join(E15_XCL)}, not {xcl} (stage1-E15-prereg §3.7)")
    family = None if vals["id3d"] in (None, "off") else vals["id3d"]
    o = dict(theta=theta, family=family, scope=vals["id3d_scope"] or "relay", mode=vals["relay_mode"] or "seq",
             order=vals["relay_order"] or "hop", mem=vals["relay_mem"] or "pool",
             trust=vals["relay_trust"] or ("id3d" if family else "reproj"), flagged=vals["flagged"] or "keep",
             chain_dir=vals["chain_dir"], gt_dir=vals["gt_dir"])
    for key, allowed in (("family", (None,) + id3d.FAMILIES), ("scope", E15_SCOPES), ("mode", E15_MODES),
                         ("order", E15_ORDERS), ("mem", E15_MEMS), ("trust", E15_TRUSTS), ("flagged", E15_FLAGGED)):
        if o[key] not in allowed:
            raise ValueError(f"S1-E15 {key} {o[key]!r} is not one of {allowed}")
    judged = family is not None or o["trust"] == "gt"          # a state source exists: the §3.4 choice runs
    if o["trust"] == "id3d" and family is None:
        raise ValueError("--relay-trust id3d needs --id3d <family>")
    if family is not None and o["trust"] in ("reproj", "obj"):
        raise ValueError("--relay-trust reproj / obj are the B arms (Fix A off): pass --id3d off")
    if o["trust"] == "gt" and not o["gt_dir"]:
        raise ValueError("--relay-trust gt (AB-oracle, diagnostic) needs --gt-dir")
    if o["gt_dir"] and o["trust"] != "gt":
        raise ValueError("--gt-dir is read by --relay-trust gt only")
    if o["scope"] != "relay" and (family is None or o["trust"] != "id3d"):
        raise ValueError(f"--id3d-scope {o['scope']} replaces the round-2 trust by Fix A: needs --id3d and --relay-trust id3d")
    if o["scope"] != "relay" and o["flagged"] != "keep":
        raise ValueError("--flagged chain / empty are registered for scope relay only")
    if o["mode"] == "jacobi" and not judged:
        raise ValueError("--relay-mode jacobi is Fix A alone (arm A): needs --id3d")
    if o["mode"] == "jacobi" and (o["order"] != "hop" or o["mem"] != "pool"):
        raise ValueError("--relay-order / --relay-mem are relay (seq) settings")
    if o["mem"] == "latest" and not judged:
        raise ValueError("--relay-mem latest trusts round-1 outputs that PASS Fix A: needs --id3d")
    if o["flagged"] != "keep" and not judged:
        raise ValueError("--flagged chain / empty act on Fix A flags: needs --id3d")
    if (o["flagged"] == "chain") != bool(o["chain_dir"]):
        raise ValueError("--flagged chain and --chain-dir go together")
    o["row"] = e15_row(o)
    if o["row"] is None:
        raise ValueError(f"S1-E15: id3d {family or 'off'} / scope {o['scope']} / mode {o['mode']} / order {o['order']} / "
                         f"mem {o['mem']} / trust {o['trust']} / flagged {o['flagged']} is not a registered arm "
                         f"(stage1-E15-prereg §4 whitelist: {sorted(E15_ROWS)})")
    if family == "cstar" and not id3d.CSTAR_SELECTED:
        raise ValueError("--id3d cstar: the A-c* constants are not selected yet (stage1-E15-prereg §6.2 Step 1 commit)")
    return o


# S1-E15 §6.2 (L5) / §10: no GPU decode with Fix A on (any family) before the Step-1 commit, which commits this record
E15_SELECT = os.path.join(common.REPO, "scripts", "e15", "runs", "select.json")
E15_STRESS_THETA = 20.0                              # §4: theta 20 = the stress arm = the AB row on DEV-A only


def e15_git_committed(path):
    """(ok, why): `path` is tracked by git and its working copy equals HEAD (git runs in scsam3 too: the repo is mounted
    with the owner's uid)."""
    d, f = os.path.dirname(os.path.abspath(path)), os.path.basename(path)
    try:
        r = subprocess.run(["git", "-C", d, "ls-files", "--error-unmatch", "--", f], capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"{path} is not tracked by git ({r.stderr.strip() or r.returncode})"
        r = subprocess.run(["git", "-C", d, "diff", "--quiet", "HEAD", "--", f], capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"{path} differs from HEAD (uncommitted change; git exit {r.returncode})"
    except OSError as exc:
        return False, f"git unavailable ({exc})"
    return True, "committed"


def e15_require_step1(family, select_path=None):
    """§6.2 (L5) / §10: a decode with a Fix A family (c0, cstar, vote, vote_inf, hybrid) runs only after the Step-1 commit:
    scripts/e15/runs/select.json committed and unmodified, Step 1 feasible, the family s* or a family's best feasible
    candidate (§6.5), and for cstar CSTAR_SELECTED with id3d.CSTAR = select.json c_best.  family None (B arms, oracle):
    nothing to check.  -> the select.json record; ValueError to refuse."""
    if family is None:
        return None
    from geomem import id3d
    path = select_path or E15_SELECT
    if not os.path.isfile(path):
        raise ValueError(f"--id3d {family}: no {path} -- Fix-A decodes wait for the Step-1 commit (stage1-E15-prereg "
                         "§6.2 L5, §10)")
    ok, why = e15_git_committed(path)
    if not ok:
        raise ValueError(f"--id3d {family}: {why} -- Fix-A decodes need the committed Step-1 record (stage1-E15-prereg "
                         "§6.2 L5, §10)")
    with open(path) as f:
        sel = json.load(f)
    if not sel.get("s_star"):
        raise ValueError(f"--id3d {family}: Step 1 found no feasible statistic, Fix A is closed (stage1-E15-prereg §6.2)")
    allowed = {sel["s_star"]["family"]} | {b["family"] for b in (sel.get("best_per_family") or {}).values() if b}
    if family not in allowed:
        raise ValueError(f"--id3d {family}: not s* or a family's best feasible candidate {sorted(allowed)} "
                         "(stage1-E15-prereg §6.2, §6.5)")
    if family == "cstar":
        cb = sel.get("c_best")
        if not (id3d.CSTAR_SELECTED and cb):
            raise ValueError("--id3d cstar: CSTAR_SELECTED is False or select.json has no c_best (stage1-E15-prereg §6.2)")
        want = {k: float(v) for k, v in cb["consts"].items()}
        if want != {k: float(v) for k, v in id3d.CSTAR.items()}:
            raise ValueError(f"--id3d cstar: geomem/id3d.py CSTAR {id3d.CSTAR} is not select.json c_best {cb['consts']}")
    return sel


def e15_cli_check(o):
    """dev_decode: theta 20 belongs to the stress arm only, i.e. the AB row on DEV-A (§4; dev_decode --e15-arm stress
    enforces the set).  run.py refuses theta 20 outright (no T06 arm has it, I7)."""
    if o is not None and o["theta"] == E15_STRESS_THETA and o["row"] != "AB":
        raise ValueError(f"--relay {E15_STRESS_THETA:g} is the stress arm only (the AB row), not {o['row']} "
                         "(stage1-E15-prereg §4)")


def e15_mix(choices, X, S1):
    """§3.4: per object the output whose raw logits / presence / IoU / pointer go on: S1 where the choice is "S1", X
    otherwise (keep / chain / empty keep X's tensors: the memory is X's and gated, the output is replaced later)."""
    import torch
    use = torch.tensor([c == "S1" for c in choices], dtype=torch.bool, device=X["presence"].device)
    if not bool(use.any()):
        return {k: X[k] for k in ("logits_1008", "presence", "ious", "obj_ptr")}
    return dict(logits_1008=torch.where(use[:, None, None, None], S1["logits_1008"], X["logits_1008"]),
                presence=torch.where(use, S1["presence"], X["presence"]), ious=torch.where(use, S1["ious"], X["ious"]),
                obj_ptr=torch.where(use[:, None], S1["obj_ptr"], X["obj_ptr"]))


def e15_output_logits(L, choices, chain_masks=None):
    """§3.4 fallbacks at the output resolution: chain -> CHAIN_IN inside the chain mask, NO_LOGIT outside; empty -> NO_LOGIT.
    L [B,h,w] (not modified); chain_masks: {object index: bool [h,w]}."""
    import torch
    idx = [i for i, c in enumerate(choices) if c in ("chain", "empty")]
    if not idx:
        return L
    L = L.clone()
    for i in idx:
        if choices[i] == "chain":
            m = chain_masks[i].to(L.device)
            L[i] = torch.where(m, torch.full_like(L[i], CHAIN_IN), torch.full_like(L[i], NO_LOGIT))
        else:
            L[i] = NO_LOGIT
    return L


def e15_final_trust(choices, trust_x, trust_s1):
    """Memory confidence of the chosen outputs: S1's where S1 was chosen, X's otherwise; never a chain / empty fallback (I4)."""
    c = np.asarray(choices, object)
    t = np.where(c == "S1", np.asarray(trust_s1, bool), np.asarray(trust_x, bool))
    return t & ~np.isin(c, ("chain", "empty"))


def relay_geometry(cache, cams, ref_cam, masks_np, theta):
    """§3.1 from the inputs alone: anchor, camera centres, a_v and the star / far split of `cams`.  When c_ini has no
    valid Pi3 point there is no anchor (review 0 #6): X None, no angle, every view a star view (F empty) and `fallback`
    says why; the E15 decode then equals C0 (RelayRound inactive)."""
    from geomem import id3d
    from geomem.pointmap import CONF_MIN
    z = np.load(os.path.join(cache, f"{ref_cam}.npz"))
    centres = camera_centres(cache, cams)
    if not bool(((z["conf"] > CONF_MIN) & (z["depth"] > 0)).any()):
        return dict(X=None, kind="none", centres=centres, angles={}, S=[c for c in cams if c != ref_cam], F=[],
                    fallback="c_ini has no valid Pi3 point: no anchor, decoded as C0")
    union = id3d.mask_to_work(np.asarray(masks_np, bool).any(0), z["depth"].shape)
    X, kind = id3d.object_anchor(z["depth"], z["conf"], z["K"], z["c2w"], union)
    angles = id3d.view_angles(centres, ref_cam, X)
    S, F = id3d.split_views(angles, cams, theta)
    return dict(X=X, kind=kind, centres=centres, angles=angles, S=S, F=F, fallback=None)


def _finite(x):
    """inf -> "inf" for the JSON records (strict JSON has no Infinity)."""
    return "inf" if isinstance(x, float) and math.isinf(x) else x


def _template_path(base, cam, obj=None):
    """`base` with {cam} / {obj} fields, else the stage-1 layout <base>/<cam>/<obj>.png (GT: <base>/<cam>.png)."""
    if "{cam}" in base:
        return base.format(cam=cam, obj=obj)
    return os.path.join(base, cam, f"{obj}.png") if obj is not None else os.path.join(base, f"{cam}.png")


class _Entry:
    """A decoded view in the relay pool: its memory (conf = trust), its final Fix A states and trust [B]."""

    def __init__(self, mem, states, trust):
        self.mem, self.states, self.trust = mem, states, np.asarray(trust, bool)


class RelayRound:
    """S1-E15 round 2 around decode_scene's round-2 loop (docs/stage1-E15-prereg.md §3.4-§3.6).

    decode_scene still decodes the views of the scope's Jacobi round 2 (`jacobi(cam)`: the star views and the unreachable
    far views; scope R = C0 exactly) and passes each output through `after_round2` (scope R: recorded as relay memory and
    returned unchanged; G / GN: the §3.4 choice).  `finish` decodes the rest -- GN re-decodes, then the far views (relay in
    order, or Jacobi with Fix A trust) -- each through Fix A, the §3.4 choice, the exclusivity rule and the sink.  The
    round-1 outputs are never changed."""

    def __init__(self, opts, scene, cache, data, wrapper, round1, mem_r1, base_ref, masks, masks_np, k, tau, emit_L):
        from geomem import id3d
        self.o, self.sc, self.cache, self.wrapper = opts, scene, cache, wrapper
        self.round1, self.mem_r1, self.base_ref, self.masks = round1, dict(mem_r1), base_ref, masks
        self.B, self.k, self.tau, self.emit_L = masks.shape[0], k, tau, emit_L
        self.objects = [int(o) for o in scene.objects]
        self.cams = list(scene.cams)
        self.targets = [c for c in self.cams if c != scene.ref_cam]
        theta = opts["theta"]
        g = relay_geometry(cache, self.cams, scene.ref_cam, masks_np, theta)
        self.X, self.anchor_kind, self.centres, self.angles, self.S, self.F = (g[x] for x in ("X", "kind", "centres", "angles", "S", "F"))
        self.fallback = g["fallback"]
        self._hops = {}
        fam = opts["family"]
        self.src = "gt" if opts["trust"] == "gt" else ("id3d" if fam else None)
        # §3.6: F empty in scope R -> no Fix A, C0; no anchor (c_ini without a valid Pi3 point) -> C0 in every scope
        self.active = self.X is not None and (opts["scope"] != "relay" or bool(self.F))
        self.rig = None
        rig_ok = True
        if self.active and fam == "cstar" and not math.isinf(id3d.CSTAR["phi_rig"]):
            try:
                rig_cams, rig_src = list(common.load_meta(data, scene.name)["cams"]), "meta"
            except (OSError, KeyError, ValueError) as exc:          # review 0 #7: recorded, never silent
                rig_cams, rig_src = self.cams, f"decoded cams (meta unreadable: {type(exc).__name__}: {exc})"
            med, per = id3d.rig_phi(cache, scene.ref_cam, rig_cams)
            rig_ok = med <= id3d.CSTAR["phi_rig"]
            self.rig = dict(phi_median=med, phi_rig=id3d.CSTAR["phi_rig"], ok=bool(rig_ok), phi=per, cams_source=rig_src,
                            n_cams=len(rig_cams))
        self.rig_ok = rig_ok
        self.order, self.unreachable = [], []
        if self.active and self.F and opts["mode"] == "seq" and rig_ok:
            self.order, self.unreachable = id3d.relay_order(self.S, self.F, self.angles, self.hop, theta, self.cams, opts["order"])
        # the views decode_scene's loop decodes: seq -> S, the unreachable F views and (rig distrust) every F view
        relayed_or_far = set(self.order) if opts["mode"] == "seq" else set(self.F)
        self._jacobi = [c for c in self.targets if c not in relayed_or_far] if self.active else list(self.targets)
        self.need_pool = (opts["mode"] == "seq" and bool(self.order)) or opts["scope"] == "global_nf"
        self.judge = id3d.Judge(cache, scene.ref_cam, masks_np, self.objects, fam, rig_ok=rig_ok) \
            if (self.active and fam is not None) else None
        self.chain = None
        if self.active and opts["flagged"] == "chain":                  # review 0 #5: a wrong template is an error
            n_res = sum(os.path.isfile(_template_path(opts["chain_dir"], c, o)) for c in self.targets for o in self.objects)
            if not n_res:
                raise FileNotFoundError(f"--chain-dir {opts['chain_dir']}: no chain PNG resolves for {scene.name} "
                                        "(stage1-E15-prereg §8: the run would be the empty-mask arm)")
            self.chain = dict(n_resolvable=int(n_res), n_chosen=0, n_missing=0)
        self._gt = {}
        self.pool, self.stash = {}, {}
        self.rec = dict(round1={}, round1_info={}, round2={}, relay={}, gn={}, final={})
        self.views, self.gn_views, self.jac_views = {}, {}, {}
        self.n_mem, self.n_gated, self.n_pairs = [], 0, 0
        self.n_orphan_views = self.n_orphan_objects = 0
        self.r1 = {}
        if self.active:
            for c in self.targets:
                r = self.round1[c]
                self.r1[c] = self._evaluate(c, self._r1out(c), "round1", overlap=r["overlap"], n_valid=r["n_valid"])

    # -------------------------------------------------------------- helpers
    def hop(self, m, v):
        from geomem import id3d
        key = (m, v)
        if key not in self._hops:
            self._hops[key] = id3d.hop_angle(self.centres, self.X, m, v)
        return self._hops[key]

    def jacobi(self, cam):
        return cam in self._jacobi

    def _r1out(self, c):
        r = self.round1[c]
        return dict(logits_1008=r["logits"], presence=r["presence"], ious=r["ious"], obj_ptr=r["obj_ptr"])

    def _gt_labels(self, cam):
        if cam not in self._gt:
            from PIL import Image
            self._gt[cam] = np.array(Image.open(_template_path(self.o["gt_dir"], cam)))
        return self._gt[cam]

    def _chain_masks(self, cam, choices):
        """{object index: chain mask [H,W] bool} for the chain choices of one view; a missing PNG is an empty mask and is
        recorded (`chain_missing` in id3d_conf.json final, review 0 #5)."""
        import cv2
        import torch
        H, W = self.sc.hw
        out = {}
        for i, c in enumerate(choices):
            if c != "chain":
                continue
            p = _template_path(self.o["chain_dir"], cam, self.objects[i])
            missing = not os.path.isfile(p)
            m = np.zeros((H, W), bool) if missing else cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            out[i] = torch.from_numpy(m)
            self.chain["n_chosen"] += 1
            self.chain["n_missing"] += int(missing)
            self.rec["final"][cam][str(self.objects[i])]["chain_missing"] = bool(missing)
        return out

    def _trust(self, states, presence, ious, overlap):
        from geomem import id3d
        t = self.o["trust"]
        if t == "id3d":
            return id3d.trust(states, presence, ious, overlap, self.tau)
        if t == "gt":
            return (np.asarray(presence, np.float64) > 0) & (np.asarray(states, object) == id3d.PASS)
        return object_conf_mode(t, presence, ious, overlap, tau=self.tau)

    def _evaluate(self, cam, out, stage, overlap=None, n_valid=None):
        """Raw L > 0 masks of `out` -> reproj overlap (unless given), Fix A states (family; GT for the oracle), trust."""
        from geomem import id3d, sam3_io
        pred = (sam3_io.logits_to_original(out["logits_1008"], self.sc.hw) > 0)[:, 0]
        if overlap is None:
            ov, nv = reproj_overlap(pred, *reproj_inputs(self.cache, cam, self.sc.ref_cam, self.sc.hw, "cuda"), self.masks)
            overlap, n_valid = ov.cpu().numpy(), nv.cpu().numpy()
        pres = out["presence"].float().cpu().numpy()
        ious = out["ious"].float().cpu().numpy()
        pred_np = pred.cpu().numpy()
        del pred
        fam_states = infos = None
        if self.judge is not None:
            fam_states, infos = self.judge.statuses(cam, pred_np, pres, n_valid, overlap)
            self.rec[stage][cam] = {str(o): s for o, s in zip(self.objects, fam_states)}
            if stage == "round1":
                self.rec["round1_info"][cam] = {str(o): x for o, x in zip(self.objects, infos)}
        states = fam_states
        if self.src == "gt":
            states = id3d.gt_states(pred_np, pres, self.objects, self._gt_labels(cam))
            self.rec.setdefault("gt_" + stage, {})[cam] = {str(o): s for o, s in zip(self.objects, states)}
        return dict(states=states, trust=self._trust(states, pres, ious, overlap), overlap=overlap, n_valid=n_valid)

    def _encode(self, cam, chosen, trust):
        import torch
        r = self.round1[cam]
        return self.wrapper.encode_memory_from_prediction(
            r["bb"], self.B, chosen["logits_1008"], chosen["presence"], chosen["obj_ptr"], r["p"], r["valid"], cam,
            conf=torch.from_numpy(np.asarray(trust, bool)).to(chosen["presence"].device))

    def _select(self, cam, X, ev):
        """§3.4 on one view -> (chosen outputs, choices, final states, final trust)."""
        from geomem import id3d
        r1 = self.r1[cam]
        if self.src is None:                        # B arms: the relay output as is
            choices = ["X"] * self.B
        else:
            choices = [id3d.select_final(ev["states"][i], r1["states"][i], self.o["flagged"]) for i in range(self.B)]
        chosen = e15_mix(choices, X, self._r1out(cam))
        trust = e15_final_trust(choices, ev["trust"], r1["trust"])
        if self.src is None:
            states = None
        else:
            states = [r1["states"][i] if c == "S1" else (None if c in ("chain", "empty") else ev["states"][i])
                      for i, c in enumerate(choices)]
            self.rec["final"][cam] = {str(o): dict(choice=c, state=states[i], trust=bool(trust[i]), x_state=ev["states"][i],
                                                   s1_state=r1["states"][i])
                                      for i, (o, c) in enumerate(zip(self.objects, choices))}
        return chosen, choices, states, trust

    def _emit(self, cam, chosen, choices):
        from geomem import sam3_io
        L = sam3_io.logits_to_original(chosen["logits_1008"], self.sc.hw)[:, 0]
        L = e15_output_logits(L, choices, self._chain_masks(cam, choices) if "chain" in choices else None)
        n = self.emit_L(cam, L)
        del L
        return n

    def _decode(self, cam, neigh_mems):
        from geomem.wrapper import MemorySet
        r = self.round1[cam]
        mem = MemorySet(self.base_ref.views + list(neigh_mems))
        self.n_mem.append(len(mem.views))
        self.n_gated += mem.n_gated()
        self.n_pairs += len(neigh_mems) * self.B
        return self.wrapper.decode_view(r["bb"], self.B, mem, r["p"], r["valid"], chunk=OBJ_CHUNK)

    def _candidates(self, v, pool):
        from geomem import id3d
        hop = None if self.o["order"] == "angle" else self.hop
        return id3d.relay_candidates(v, pool, self.centres, self.cams, hop=hop, theta=self.o["theta"])

    # -------------------------------------------------------------- the round-2 loop's hooks
    def round2_memories(self, mem_views):
        """G / GN: the round-2 memory confidence of every round-1 output = its Fix A trust (replaces reproj)."""
        import torch
        if self.o["scope"] == "relay" or not self.active:
            return mem_views
        return {c: replace(v, conf=torch.from_numpy(self.r1[c]["trust"]).to(v.conf.device)) for c, v in mem_views.items()}

    def after_round2(self, cam, out):
        """A Jacobi round-2 output -> what decode_scene writes (None: written later, by `finish`)."""
        if not self.active:
            return out
        if self.o["scope"] == "relay":
            if cam in self.S and self.need_pool:
                ev = self._evaluate(cam, out, "round2")
                self.pool[cam] = _Entry(self._encode(cam, out, ev["trust"]), ev["states"], ev["trust"])
            return out                                  # scope R: C0's output, unchanged (I2)
        ev = self._evaluate(cam, out, "round2")
        chosen, choices, states, trust = self._select(cam, out, ev)
        if cam in self.S and self.need_pool:
            self.pool[cam] = _Entry(self._encode(cam, chosen, trust), states, trust)
        if self.o["scope"] == "global_nf" and cam in self.S:
            self.stash[cam] = dict(chosen=chosen, choices=choices, states=states, trust=trust)
            return None
        return chosen

    # -------------------------------------------------------------- after the loop
    def finish(self):
        """GN re-decodes, the far views; -> masks written."""
        if not self.active:
            return 0
        n = 0
        if self.o["scope"] == "global_nf":
            n += self._gn()
        if self.o["mode"] == "seq":
            n += self._relay()
        else:
            n += self._jacobi_far()
        return n

    def _gn(self):
        """GN: star views with a FLAG left after the choice, by a_v, re-decoded with the relay neighbour rule (self
        excluded); only the FLAG objects take the new output; then every star view is written in camera order."""
        from geomem import id3d
        rank = {c: i for i, c in enumerate(self.cams)}
        todo = sorted((c for c in self.stash if any(s == id3d.FLAG for s in self.stash[c]["states"])),
                      key=lambda c: (self.angles[c], rank[c]))
        for v in todo:
            cands = self._candidates(v, list(self.pool))
            neigh = id3d.relay_neighbours(cands, {m: bool(self.pool[m].trust.any()) for m in cands})
            rec = dict(neighbours=neigh, flagged=[self.objects[i] for i, s in enumerate(self.stash[v]["states"]) if s == id3d.FLAG])
            self.gn_views[v] = rec
            if not neigh:
                rec["orphan"] = True
                continue
            out = self._decode(v, [self.pool[m].mem for m in neigh])
            ev = self._evaluate(v, out, "gn")
            st = self.stash[v]
            fl = [i for i, s in enumerate(st["states"]) if s == id3d.FLAG]
            choices = ["S1" if i in fl else "X" for i in range(self.B)]       # "S1" slot = the re-decode (mix only)
            st["chosen"] = e15_mix(choices, st["chosen"], out)
            for i in fl:
                st["states"][i] = ev["states"][i]
                st["trust"][i] = bool(ev["trust"][i])
                self.rec["final"][v][str(self.objects[i])].update(gn=True, state=ev["states"][i], trust=bool(ev["trust"][i]))
            self.pool[v] = _Entry(self._encode(v, st["chosen"], st["trust"]), st["states"], st["trust"])
        n = 0
        for c in self.targets:
            if c in self.stash:
                n += self._emit(c, self.stash[c]["chosen"], self.stash[c]["choices"])
        self.stash = {}
        return n

    def _r1_entry(self, c):
        """ABml: a not-yet-relayed far view's round-1 memory, trusted only where Fix A PASSes."""
        import torch
        from geomem import id3d
        r1 = self.r1[c]
        t = (self.round1[c]["presence"].float().cpu().numpy() > 0) & (np.asarray(r1["states"], object) == id3d.PASS)
        return _Entry(replace(self.mem_r1[c], conf=torch.from_numpy(t).to(self.mem_r1[c].conf.device)), r1["states"], t)

    def _relay(self):
        """§3.5: the far views in relay order, memory = c_ini + up to K_RELAY trusted decoded views."""
        from geomem import id3d
        n = 0
        for v in self.order:
            entries = dict(self.pool)
            if self.o["mem"] == "latest":
                entries.update({c: self._r1_entry(c) for c in self.F if c not in self.pool and c != v})
            cands = self._candidates(v, list(entries))
            neigh = id3d.relay_neighbours(cands, {m: bool(entries[m].trust.any()) for m in cands})
            assert len(neigh) <= id3d.K_RELAY and v not in neigh, (v, neigh)                                   # I3
            assert self.o["order"] == "angle" or all(self.hop(m, v) <= self.o["theta"] for m in neigh), (v, neigh)
            if neigh:
                out = self._decode(v, [entries[m].mem for m in neigh])
                ev = self._evaluate(v, out, "relay")
                orphans = id3d.orphan_objects(neigh, {m: entries[m].trust for m in neigh})
            else:                                       # orphan view: memory = {c_ini} = round 1's (§3.5 step 6)
                out, ev = self._r1out(v), self.r1[v]
                self.n_mem.append(len(self.base_ref.views))
                orphans = list(range(self.B))
                self.n_orphan_views += 1
            self.n_orphan_objects += len(orphans)
            chosen, choices, states, trust = self._select(v, out, ev)
            n += self._emit(v, chosen, choices)
            self.pool[v] = _Entry(self._encode(v, chosen, trust), states, trust)
            self.views[v] = dict(a=self.angles[v], orphan=not neigh, neighbours=neigh,
                                 candidates=[[m, self.hop(m, v), float(np.linalg.norm(self.centres[m] - self.centres[v]))]
                                             for m in cands],
                                 untrusted=[m for m in cands if not entries[m].trust.any()],
                                 trusted={m: [self.objects[i] for i in np.flatnonzero(entries[m].trust)] for m in neigh},
                                 orphan_objects=[self.objects[i] for i in orphans])
            del out
        return n

    def _jacobi_far(self):
        """Arm A: every far view by the round-2 Jacobi rule (k nearest round-1 outputs) with Fix A trust, then §3.4."""
        import torch
        n = 0
        for v in self.F:
            cand = [c for c in select_candidates(v, self.centres, self.k) if c != self.sc.ref_cam]
            mems = {c: replace(self.mem_r1[c], conf=torch.from_numpy(self.r1[c]["trust"]).to(self.mem_r1[c].conf.device))
                    for c in cand}
            neigh = [c for c in cand if bool(mems[c].conf.any())]
            out = self._decode(v, [mems[c] for c in neigh])
            ev = self._evaluate(v, out, "round2")
            chosen, choices, _, _ = self._select(v, out, ev)
            n += self._emit(v, chosen, choices)
            self.jac_views[v] = dict(a=self.angles[v], neighbours=neigh, candidates=cand,
                                     untrusted=[c for c in cand if c not in neigh],
                                     trusted={m: [self.objects[i] for i in np.flatnonzero(self.r1[m]["trust"])] for m in neigh})
            del out
        return n

    # -------------------------------------------------------------- records
    def config(self):
        from geomem import id3d
        fin = [x for cam in self.rec["final"].values() for x in cam.values()]
        return {"relay_theta": _finite(self.o["theta"]), "id3d": self.o["family"] or "off", "id3d_scope": self.o["scope"],
                "relay_mode": self.o["mode"], "relay_order": self.o["order"], "relay_mem": self.o["mem"],
                "relay_trust": self.o["trust"], "flagged": self.o["flagged"], "n_F": len(self.F), "n_relay": len(self.views),
                "n_unreachable": len(self.unreachable), "n_orphan_views": self.n_orphan_views,
                "n_orphan_objects": self.n_orphan_objects, "n_flag_final": sum(x["state"] == id3d.FLAG for x in fin),
                "n_sub_r1": sum(x["choice"] == "S1" for x in fin),
                "n_fallback": sum(x["choice"] in E15_FLAGGED for x in fin)}

    def counts(self):
        """Relay-pool candidate counts (review 0 #4): E15's n_pairs counts the chosen neighbours only, so its
        gated_pairs_frac / views_dropped are not C0's; these are the numbers to recount from."""
        out = {}
        for name, recs in (("relay", self.views), ("jacobi", self.jac_views), ("gn", self.gn_views)):
            out[name] = dict(n_views=len(recs), n_candidates=sum(len(r.get("candidates", [])) for r in recs.values()),
                             n_untrusted=sum(len(r.get("untrusted", [])) for r in recs.values()),
                             n_neighbours=sum(len(r.get("neighbours", [])) for r in recs.values()))
        return out

    def relay_record(self):
        return dict(theta=_finite(self.o["theta"]), options={k: _finite(v) for k, v in self.o.items()},
                    anchor=dict(kind=self.anchor_kind, X=None if self.X is None else [float(x) for x in self.X]),
                    fallback=self.fallback, angles=self.angles, S=self.S, F=self.F,
                    active=self.active, order=self.order, unreachable=self.unreachable,
                    rig=None if self.rig is None else dict(self.rig, phi_rig=_finite(self.rig["phi_rig"])),
                    jacobi_views=[c for c in self._jacobi if c in self.F], views=self.views, gn=self.gn_views,
                    jacobi=self.jac_views, counts=self.counts())

    def choice_summary(self):
        """§8 reporting rule (review 0 #4): the final choices split by choice and by X's state; `keep` with X EMPTY and
        round 1 not PASS is an absent object, not a fallback."""
        from geomem import id3d
        fin = [x for cam in self.rec["final"].values() for x in cam.values()]
        by = Counter(f"{x['choice']}/{x.get('x_state')}" for x in fin)
        return dict(n=len(fin), by_choice=dict(Counter(x["choice"] for x in fin)), by_choice_x_state=dict(sorted(by.items())),
                    keep_absent=sum(x["choice"] == "keep" and x.get("x_state") == id3d.EMPTY and x.get("s1_state") != id3d.PASS
                                    for x in fin),
                    keep_flagged=sum(x["choice"] == "keep" and x.get("x_state") == id3d.FLAG for x in fin))

    def id3d_record(self):
        """id3d_conf.json (None when Fix A was not computed)."""
        from geomem import id3d
        if not self.active or self.src is None:
            return None
        consts = {k: _finite(v) for k, v in (self.judge.consts.items() if self.judge is not None else [])}
        return dict(family=self.o["family"], status_source=self.src, consts=consts,
                    cstar_selected=bool(id3d.CSTAR_SELECTED) if self.o["family"] == "cstar" else None,
                    rig_ok=bool(self.rig_ok), models=None if self.judge is None else self.judge.model_record(),
                    summary=self.choice_summary(), chain=self.chain, **self.rec)


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
    e15 = e15_options(args)                                # S1-E15; None = off (the unchanged path, no config key)
    if e15 is not None:
        assert two_round and round2 == "pred" and gate == "reproj" and zoom is None and not pair_mode and xcl in E15_XCL \
            and arm not in FUSE_ARMS and arm not in ALIGN_ARMS, \
            f"--relay is registered on the two-round reproj-gate decode (C0 = geoset_rep_xs) only, not {arm}/{gate}/{xcl}"
    nbr = getattr(args, "nbr", "pi3") or "pi3"            # S1-E16; pi3 = the unchanged path (no config key)
    assert nbr in NBR_MODES, nbr
    if nbr != "pi3":
        assert two_round and round2 == "pred" and gate in ("obj", "absent") and zoom is None and e15 is None \
            and not pair_mode and arm not in FUSE_ARMS and arm not in ALIGN_ARMS, \
            f"--nbr {nbr} is registered for the two-round predicted-memory decode with the obj / absent gate only (S1-E16), not {arm}/{gate}"
        assert not flags["use_bias"] and not flags["use_geometry"], \
            f"--nbr {nbr}: the geometry-free neighbour rules are registered for geometry-free arms only (S1-E16), not {arm}"
    config = {"arm": arm, "untrained": bool(args.untrained), **flags, "two_round": two_round}
    if nbr != "pi3":
        config["nbr"] = nbr
    wrapper = build_wrapper(model, arm)
    geo = wrapper.geo_embed
    ck = None
    if e15 is not None and args.untrained and scene.name in E15_FOLD:
        raise ValueError(f"S1-E15 decodes {scene.name} with its fold-{E15_FOLD[scene.name]} checkpoint only (I9), not --untrained")
    if not args.untrained:
        folds = with_external_fold(load_folds(), scene.name, getattr(scene, "external_fold", None))
        ckpt_path = checkpoint_for(scene.name, arm, args.runs, folds)
        with open(os.path.join(os.path.dirname(ckpt_path), "config.json")) as f:
            ck_cfg = json.load(f)
        assert_no_leak(ck_cfg, scene.name)
        if e15 is not None and scene.name in E15_FOLD and int(ck_cfg["fold"]) != E15_FOLD[scene.name]:
            raise ValueError(f"S1-E15 decodes {scene.name} with the fold-{E15_FOLD[scene.name]} checkpoint only, not "
                             f"{ckpt_path} (stage1-E15-prereg I9)")
        ck = load_checkpoint_into(wrapper, geo, ckpt_path)
        config.update({"fold": ck_cfg["fold"], "ckpt_sha256": ck_cfg["ckpt_sha256"], "train_git": ck_cfg.get("git"),
                       "train_steps": ck_cfg.get("steps"), "epochs": ck_cfg.get("epochs"), "frame_stride": ck_cfg.get("frame_stride")})
        if e15 is not None:
            config["ckpt"] = ckpt_path                             # S1-E15 I9: the checkpoint path is recorded
    if getattr(args, "zero_bias", False):                # ring diagnosis D3; off = the unchanged path (no config key)
        config.update({"zero_bias": True, "zero_bias_max_abs_before": zero_geo_bias(wrapper)})
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
                centres = camera_coords(nbr, cache, scene, model)   # S1-E16: Pi3 centres unless --nbr index / feat
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
                rr = None
                if e15 is not None:
                    def emit_L(cam, L):
                        """S1-E15: output-resolution logits [B,h,w] of one target -> the round-2 rule -> the sink."""
                        if xcl == "none":
                            return write_bool(args.out, cam, (L > 0).cpu().numpy())
                        keep, xc, xr = exclusive_masks(L, None if xcl == "out_nocap" else XCL_TAU_KEEP,
                                                       XCL_SHIFT if xcl == "outs" else 0.0)
                        xcl_counts.update(xc)
                        xcl_recs[cam] = {str(int(o)): xr[i] for i, o in enumerate(scene.objects)}
                        return write_bool(args.out, cam, keep.cpu().numpy())
                    rr = RelayRound(e15, scene, cache, args.data, wrapper, round1, mem_views, base_ref, masks, masks_np,
                                    k, tau, emit_L)
                    mem_views = rr.round2_memories(mem_views)
                for cam in targets:
                    if rr is not None and not rr.jacobi(cam):
                        continue                                 # S1-E15: decoded by rr.finish() below
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
                    if rr is not None:
                        out = rr.after_round2(cam, out)
                        if out is None:
                            continue                             # S1-E15 GN: written by rr.finish()
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
                if rr is not None:
                    written2 += rr.finish()
                    n_mem += rr.n_mem
                    n_gated += rr.n_gated
                    n_pairs += rr.n_pairs
                    ordered = {c: xcl_recs[c] for c in targets if c in xcl_recs}      # camera order, as the plain loop
                    xcl_recs.clear()
                    xcl_recs.update(ordered)
                    sink.json(args.out, "relay.json", rr.relay_record())
                    rec = rr.id3d_record()
                    if rec is not None:
                        sink.json(args.out, "id3d_conf.json", rec)
                    config.update(rr.config())
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
    ap.add_argument("--nbr", choices=NBR_MODES, default="pi3",
                    help="S1-E16 (docs/stage1-E16-prereg.md): the k-nearest-neighbour rule of the round-2 memory candidates -- "
                         "Pi3 camera centres (default, unchanged), the rig index, or pooled SAM 3 features; geometry-free arms only")
    ap.add_argument("--zero-bias", action="store_true", dest="zero_bias",
                    help="diagnostic (ring diagnosis D3, stage1/docs/ring-diagnosis-2026-10-20.md): the checkpoint's GeoBias parameters "
                         "set to 0 at decode (fine-tuned memory attention kept); needs a bias arm and an --out other than the arm")
    add_e15_args(ap)


def add_e15_args(ap):
    """S1-E15 (docs/stage1-E15-prereg.md §11.1): every flag below is off / unset by default; all need --relay.  Shared by
    this CLI and the recorded DEV override (methods/geomem/dev_decode.py --e15-arm)."""
    ap.add_argument("--relay", type=float, default=None, metavar="THETA",
                    help="S1-E15: views whose viewpoint change from c_ini (object anchor) exceeds THETA degrees are far views, "
                         "decoded after the star views; THETA in {20, 35, 40, 45, 50, 55, 60, inf} (inf = no far view); "
                         "needs --xcl outs; off when absent")
    ap.add_argument("--id3d", choices=("off", "c0", "cstar", "vote", "vote_inf", "hybrid"), default=None,
                    help="S1-E15 Fix A statistic (A-c0 / A-c* / A-v r_max 0.2 / A-v r_max inf / A-h); default off")
    ap.add_argument("--id3d-scope", choices=E15_SCOPES, default=None, dest="id3d_scope",
                    help="S1-E15: Fix A on the far views only (relay = R, default), every view (global = G), G + star-view "
                         "re-decodes (global_nf = GN)")
    ap.add_argument("--relay-mode", choices=E15_MODES, default=None, dest="relay_mode",
                    help="S1-E15 far views: relay decoding in order (seq, default) or round-2 Jacobi with Fix A trust (jacobi)")
    ap.add_argument("--relay-order", choices=E15_ORDERS, default=None, dest="relay_order",
                    help="S1-E15 relay order: reachability rule with hops <= THETA (hop, default) or pure angle order (angle)")
    ap.add_argument("--relay-mem", choices=E15_MEMS, default=None, dest="relay_mem",
                    help="S1-E15 relay memory pool: decoded views (pool, default) or + round-1 outputs that PASS (latest)")
    ap.add_argument("--relay-trust", choices=E15_TRUSTS, default=None, dest="relay_trust",
                    help="S1-E15 relay memory confidence: Fix A (id3d; default with --id3d), reproj (default without), obj, "
                         "or gt (AB-oracle, diagnostic; needs --gt-dir)")
    ap.add_argument("--flagged", choices=E15_FLAGGED, default=None,
                    help="S1-E15 §3.4 fallback for an object flagged in both outputs: keep (default), chain (--chain-dir), empty")
    ap.add_argument("--chain-dir", default=None, dest="chain_dir",
                    help="S1-E15 --flagged chain: <dir>/<cam>/<obj>.png, or a path with {cam} and {obj} fields (absent = empty, "
                         "recorded as chain_missing; none resolving is an error)")
    ap.add_argument("--gt-dir", default=None, dest="gt_dir",
                    help="S1-E15 --relay-trust gt: label maps <dir>/<cam>.png, or a path with a {cam} field")


def main():
    args = common.parse_args(__doc__, _extra)
    check_gate_out(args.arm, args.gate, args.out, args.round2, args.zoom, args.xcl, args.zero_bias, args.relay, args.nbr)
    try:                                                    # S1-E15 whitelist / theta / xcl / Step-1 commit, before the model
        e15 = e15_options(args)
        if e15 is not None and e15["theta"] == E15_STRESS_THETA:   # I7: theta 20 = the stress arm, DEV-A (dev_decode) only
            raise ValueError(f"--relay {E15_STRESS_THETA:g} is the stress arm, registered on DEV-A only (dev_decode "
                             "--e15-arm stress); no run.py / T06 arm has theta 20 (stage1-E15-prereg §4, I7)")
        if e15 is not None:
            e15_require_step1(e15["family"])
    except ValueError as exc:
        raise SystemExit(f"run.py: {exc}")
    from methods import sam3_single
    decode_scene.model = sam3_single.load_model()
    decode_scene.init_state = snapshot_init(decode_scene.model)
    common.run_all(args, decode_scene, aux_outs=r1_aux_outs(args.out, args.r1_out) if args.arm in SET_ARMS else [])


if __name__ == "__main__":
    main()
