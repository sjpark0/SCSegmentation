"""Training samples for GeoMem (spec 4.2): (scene, t, reference camera r, target camera v), r != v
among the annotated cameras; objects are the ids in r's label map at t, at most 16 per step.

Pure functions here run on the host (numpy); `load_sample` imports torch lazily and reads the Task-5
caches (backbone features, shared positional encodings, per-frame Pi3 pointmaps).
"""
import os
from dataclasses import dataclass

import numpy as np

from geomem.cache import feat_cache_path, level72_path, pi3_frame_cache, pos_enc_path, train_frames
from geomem.pointmap import view_grid
from methods.common import frame_dir
from mvseg2d.io import labelmap_objects, labelmap_to_mask, load_labelmap, load_meta

MAX_OBJECTS = 16


@dataclass(frozen=True)
class Sample:
    scene: str
    t: int
    ref_cam: str
    tgt_cam: str


def enumerate_samples(data, scenes, frame_stride=1):
    """All (scene, t, r, v); with frame_stride k only every k-th labelled frame (first one included)."""
    out = []
    for scene in sorted(scenes):
        meta = load_meta(data, scene)
        cams = list(meta["mask_cams"])
        for t in train_frames(data, scene)[::max(1, int(frame_stride))]:
            for r in cams:
                for v in cams:
                    if r != v:
                        out.append(Sample(scene, int(t), r, v))
    return out


def _labelmap(data, sample, cam):
    return load_labelmap(os.path.join(frame_dir(data, sample.scene, sample.t), "masks", f"{cam}.png"))


def objects_for(sample, data):
    return labelmap_objects(_labelmap(data, sample, sample.ref_cam))


def choose_objects(ids, max_n=MAX_OBJECTS, rng=None):
    ids = list(ids)
    if len(ids) <= max_n:
        return ids
    rng = rng or np.random.default_rng()
    pick = rng.choice(len(ids), size=max_n, replace=False)
    return [ids[i] for i in sorted(pick)]


def unannotated_views(data, scene, k_min=2, k_max=4, rng=None):
    meta = load_meta(data, scene)
    pool = [c for c in meta["cams"] if c not in meta["mask_cams"]]
    if not pool:
        return []
    rng = rng or np.random.default_rng()
    k = min(len(pool), int(rng.integers(k_min, k_max + 1)))
    pick = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in sorted(pick)]


def third_camera(data, sample):
    """The annotated camera that is neither the reference nor the target (exactly one on MVSeg)."""
    cams = [c for c in load_meta(data, sample.scene)["mask_cams"] if c not in (sample.ref_cam, sample.tgt_cam)]
    assert len(cams) == 1, (sample, cams)
    return cams[0]


def memory_rng(seed, step):
    """Generator for the memory-camera draw of one step (GeoSet-2 review): separate from the shared training `rng`,
    so the object subsets and coin flips of `geoset_pred_ft` and `geoset_only_ft` stay on the same stream."""
    return np.random.default_rng([int(seed), int(step)])


def cut_rng(seed, step):
    """S1-E12: the generator of one step's memory-token cut (coin first, then (side, f) per eligible object); separate
    from the shared training `rng` and from memory_rng(seed, step), so object subsets, the memory coin and the memory
    camera of geoset_pred_ft are unchanged by the cut arms."""
    return np.random.default_rng([int(seed), int(step), 1])


GRID = 72
_YY, _XX = np.mgrid[0:GRID, 0:GRID]
_SIDE_COORD = (_XX, -_XX, _YY, -_YY)        # side 0 left, 1 right, 2 top, 3 bottom: drop the cells with coord < cut


def fov_cut(cells, g, keep=(0.3, 0.7), min_cells=20, min_kept=6):
    """S1-E12 M+ rule (docs/stage1-E12-prereg.md section 2, steps 3-5) on object cells [B,72,72] bool (the reference
    GT on the memory-token grid) -> (drop [B, 5184] bool, records).

    For each object b in order with n_b >= min_cells: side = g.integers(4), f = g.uniform(*keep), k = floor((1-f) n_b),
    a = the side's coordinate over the object cells, a_c = sorted(a)[k]; every grid cell (object or background) with
    coordinate < a_c is dropped.  Ties at the cut line are kept, so the kept fraction of the object is >= f; an object
    with nothing dropped counts as not cut.  Objects under min_cells draw nothing.  Token index = y * 72 + x."""
    cells = np.asarray(cells, bool)
    B = cells.shape[0]
    assert cells.shape[1:] == (GRID, GRID), cells.shape
    drop = np.zeros((B, GRID * GRID), bool)
    recs = []
    for b in range(B):
        n = int(cells[b].sum())
        if n < min_cells:
            recs.append(dict(eligible=False, n=n, side=None, f=None, dropped=0))
            continue
        side = int(g.integers(4))
        f = float(g.uniform(keep[0], keep[1]))
        k = int(np.floor((1.0 - f) * n))
        coord = _SIDE_COORD[side]
        a_c = np.sort(coord[cells[b]])[k]
        d = coord < a_c
        dropped = int((d & cells[b]).sum())
        if dropped:
            assert n - dropped >= min_kept, (n, dropped, f)
            drop[b] = d.reshape(-1)
        recs.append(dict(eligible=True, n=n, side=side, f=f, dropped=dropped))
    return drop, recs


def load_train_angles(path):
    """S1-E17: the input-only training angle cache (scripts/e17/train_angles.py) -> {(scene, t, ref): {cam: a(ref->cam) deg}}."""
    import json
    with open(path) as f:
        d = json.load(f)
    return {(e["scene"], int(e["t"]), e["ref"]): e["all"] for e in d["entries"]}


def sample_weights(samples, angles, lam, theta):
    """S1-E17 C-ov: weight 1 + lam [a(r->v) > theta] per sample; every sample needs an angle entry (the cache is complete)."""
    w = np.ones(len(samples), np.float64)
    for i, s in enumerate(samples):
        a = angles[(s.scene, s.t, s.ref_cam)][s.tgt_cam]
        if a > theta:
            w[i] += lam
    return w


def relay_memory_camera(pool, angles, tgt_cam, rng, prob):
    """S1-E17 C-ov second memory: coin (prob) from `rng`, then -- heads -- uniform over the cameras of `pool` with
    a(r->m) < a(r->v), falling back to the whole pool when none is nearer; tails -- uniform over `pool` (geoset_pred_ft).
    -> (camera, rule) with rule in {"relay", "relay_none", "uniform"}."""
    assert pool
    if rng.random() < prob:
        near = [c for c in pool if angles[c] < angles[tgt_cam]]
        if near:
            return near[int(rng.integers(len(near)))], "relay"
        return pool[int(rng.integers(len(pool)))], "relay_none"
    return pool[int(rng.integers(len(pool)))], "uniform"


def random_memory_camera(data, sample, rng):
    """A memory-source camera for predicted-memory training (GeoSet-2 spec 4.2): uniform over the scene's cameras
    other than the reference and the target, annotated or not."""
    pool = [c for c in load_meta(data, sample.scene)["cams"] if c not in (sample.ref_cam, sample.tgt_cam)]
    assert pool, sample
    return pool[int(rng.integers(len(pool)))]


def load_level72(data, scene, t, device):
    """GeoFuse spec 2.5: (cams, feats fp16 [N,256,72,72] on device) of frame t; the camera order must equal meta["cams"]."""
    import torch
    path = level72_path(data, scene, t)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no level-72 feature cache: {path} (run geomem.cache_feats --level72)")
    d = torch.load(path, map_location=device)
    cams = list(d["cams"])
    assert cams == list(load_meta(data, scene)["cams"]), f"{path}: camera order differs from meta.json"
    return cams, d["feats"]


def load_backbone(data, scene, t, cam, device):
    """Cached backbone features of one view -> (image [1,3,1008,1008] f32, backbone_out dict f32, hw)."""
    import torch
    d = torch.load(feat_cache_path(data, scene, t, cam), map_location=device)
    pos = torch.load(pos_enc_path(data, scene), map_location=device)
    bb = {"backbone_fpn": [f.float() for f in d["backbone_fpn"]], "vision_pos_enc": [p.float() for p in pos]}
    return d["image"].float().unsqueeze(0), bb, tuple(d["hw"])


def load_sample(data, sample, objects, device="cuda", extra_mem_cam=None, extra_mem_gt=True, aligned=False, aligned_mem=False):
    """Everything one training step needs, on `device` (see spec 3.5).  With `extra_mem_cam` (GeoMem-2 spec 2.2)
    the batch also carries that camera's image, backbone and grid as a second memory view, plus its GT masks when
    `extra_mem_gt` (GeoSet-2 spec 4.2: a predicted memory carries no masks; the camera may be unannotated).
    With `aligned` (GeoFuse spec 3) it carries every camera's level-72 features and Pi3 geometry at frame t as
    `views` (list of dict(feat, depth, K, c2w) in meta["cams"] order), plus `scale` and `c2w_ref`.
    With `aligned_mem` (GeoAlign spec 3) the extra memory camera is carried as `mem_view` (its level-72 features and
    geometry, no backbone/image: the memory is built on the REFERENCE image and grid) plus `scale` and `c2w_ref`.
    A missing feature cache raises FileNotFoundError naming it."""
    import torch
    from geomem.embed import grid_tensors
    if extra_mem_cam is not None and aligned_mem and not os.path.isfile(level72_path(data, sample.scene, sample.t)):
        raise FileNotFoundError(f"no level-72 feature cache: {level72_path(data, sample.scene, sample.t)} "
                                "(run geomem.cache_feats --level72)")
    if extra_mem_cam is not None and not aligned_mem:
        path = feat_cache_path(data, sample.scene, sample.t, extra_mem_cam)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no cached backbone features for memory view {extra_mem_cam}: {path} "
                                    "(run geomem.cache_feats --all-views)")
    ref_img, ref_bb, hw = load_backbone(data, sample.scene, sample.t, sample.ref_cam, device)
    _, tgt_bb, hw2 = load_backbone(data, sample.scene, sample.t, sample.tgt_cam, device)
    assert hw == hw2, (hw, hw2)
    ref_lab = _labelmap(data, sample, sample.ref_cam)
    tgt_lab = _labelmap(data, sample, sample.tgt_cam)
    ref_masks = torch.from_numpy(np.stack([labelmap_to_mask(ref_lab, o) for o in objects])).to(device)
    tgt_masks = torch.from_numpy(np.stack([labelmap_to_mask(tgt_lab, o) for o in objects])).to(device)
    cache = pi3_frame_cache(data, sample.scene, sample.t)
    p_ref, v_ref = grid_tensors(view_grid(cache, sample.ref_cam, sample.ref_cam), device)
    p_tgt, v_tgt = grid_tensors(view_grid(cache, sample.tgt_cam, sample.ref_cam), device)
    out = {"ref_image": ref_img, "ref_backbone": ref_bb, "ref_masks": ref_masks,
           "tgt_backbone": tgt_bb, "tgt_masks": tgt_masks, "hw": hw,
           "p_ref": p_ref, "valid_ref": v_ref, "p_tgt": p_tgt, "valid_tgt": v_tgt, "objects": list(objects)}
    if extra_mem_cam is not None and aligned_mem:
        from geomem.align import reference_frame, view_geometry
        cams, feats = load_level72(data, sample.scene, sample.t, device)
        out["mem_cam"] = extra_mem_cam
        out["mem_view"] = dict(feat=feats[cams.index(extra_mem_cam)], **view_geometry(cache, extra_mem_cam, device))
        out["scale"], out["c2w_ref"] = reference_frame(cache, sample.ref_cam, device)
    elif extra_mem_cam is not None:
        mem_img, mem_bb, _ = load_backbone(data, sample.scene, sample.t, extra_mem_cam, device)
        out.update({"mem_cam": extra_mem_cam, "mem_image": mem_img, "mem_backbone": mem_bb})
        if extra_mem_gt:
            mem_lab = _labelmap(data, sample, extra_mem_cam)
            out["mem_masks"] = torch.from_numpy(np.stack([labelmap_to_mask(mem_lab, o) for o in objects])).to(device)
        out["p_mem"], out["valid_mem"] = grid_tensors(view_grid(cache, extra_mem_cam, sample.ref_cam), device)
    if aligned:
        from geomem.align import reference_frame, view_geometry
        cams, feats = load_level72(data, sample.scene, sample.t, device)
        out["views"] = [dict(feat=feats[i], **view_geometry(cache, c, device)) for i, c in enumerate(cams)]
        out["view_cams"], out["tgt_cam"] = list(cams), sample.tgt_cam
        out["scale"], out["c2w_ref"] = reference_frame(cache, sample.ref_cam, device)
    return out


@dataclass(frozen=True)
class PairSample:
    """GeoQuery training sample: (scene, t, reference camera); the targets are the other annotated cameras."""
    scene: str
    t: int
    ref_cam: str


def enumerate_pairs(data, scenes, frame_stride=1):
    out = []
    for scene in sorted(scenes):
        cams = list(load_meta(data, scene)["mask_cams"])
        for t in train_frames(data, scene)[::max(1, int(frame_stride))]:
            for r in cams:
                out.append(PairSample(scene, int(t), r))
    return out


def pair_targets(data, sample):
    return sorted(c for c in load_meta(data, sample.scene)["mask_cams"] if c != sample.ref_cam)


def objects_for_pair(sample, data):
    return labelmap_objects(_labelmap(data, sample, sample.ref_cam))


def hub_views(cams, ref_cam, targets, k_random=5, rng=None):
    """[ref] + targets + k_random distinct other cameras (fewer if the scene is small), no duplicates."""
    head = [ref_cam] + list(targets)
    pool = [c for c in cams if c not in head]
    rng = rng or np.random.default_rng()
    k = min(int(k_random), len(pool))
    pick = rng.choice(len(pool), size=k, replace=False) if k else []
    return head + [pool[i] for i in sorted(pick)]


def load_sample_pair(data, sample, objects, hub_cams, device="cuda"):
    """One GeoQuery training step: reference (image, backbone, GT masks, grid), both annotated targets, and the hub's
    views (backbones shared with the targets, grids stacked).  A missing feature cache raises FileNotFoundError naming it."""
    for cam in hub_cams:
        path = feat_cache_path(data, sample.scene, sample.t, cam)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no cached backbone features for hub view {cam}: {path} (run geomem.cache_feats --all-views)")
    import torch
    from geomem.embed import grid_tensors
    targets = pair_targets(data, sample)
    backbones, images, hws = {}, {}, {}
    for cam in dict.fromkeys([sample.ref_cam, *targets, *hub_cams]):
        images[cam], backbones[cam], hws[cam] = load_backbone(data, sample.scene, sample.t, cam, device)
    assert len(set(hws.values())) == 1, hws
    cache = pi3_frame_cache(data, sample.scene, sample.t)
    grids = {cam: view_grid(cache, cam, sample.ref_cam) for cam in backbones}
    ref_lab = _labelmap(data, sample, sample.ref_cam)
    ref_masks_np = np.stack([labelmap_to_mask(ref_lab, o) for o in objects])
    p_ref, v_ref = grid_tensors(grids[sample.ref_cam], device)
    out = {"ref_image": images[sample.ref_cam], "ref_backbone": backbones[sample.ref_cam],
           "ref_masks": torch.from_numpy(ref_masks_np).to(device), "ref_masks_np": ref_masks_np, "hw": hws[sample.ref_cam],
           "p_ref": p_ref, "valid_ref": v_ref,
           "p_ref_np": np.asarray(grids[sample.ref_cam]["p"], np.float32).reshape(-1, 3),
           "valid_ref_np": np.asarray(grids[sample.ref_cam]["valid"], bool).reshape(-1), "objects": list(objects), "targets": []}
    for cam in targets:
        lab = _labelmap(data, sample, cam)
        p, v = grid_tensors(grids[cam], device)
        out["targets"].append({"cam": cam, "backbone": backbones[cam], "p": p, "valid": v,
                               "masks": torch.from_numpy(np.stack([labelmap_to_mask(lab, o) for o in objects])).to(device)})
    hub_grids = [grid_tensors(grids[cam], device) for cam in hub_cams]
    out["hub"] = {"cams": list(hub_cams), "backbones": [backbones[c] for c in hub_cams],
                  "p": torch.stack([g[0] for g in hub_grids]), "valid": torch.stack([g[1] for g in hub_grids])}
    return out


class SampleIterator:
    """Seeded reshuffle of the samples every epoch; with `weights` (S1-E17 C-ov) every epoch is instead len(samples) draws
    with replacement, probability proportional to the weights (the epoch size is unchanged)."""

    def __init__(self, samples, seed=0, weights=None):
        self.samples = list(samples)
        self.rng = np.random.default_rng(seed)
        self.p = None
        if weights is not None:
            w = np.asarray(weights, np.float64)
            assert w.shape == (len(self.samples),) and (w > 0).all(), w.shape
            self.p = w / w.sum()

    def epoch(self):
        n = len(self.samples)
        order = self.rng.permutation(n) if self.p is None else self.rng.choice(n, size=n, replace=True, p=self.p)
        return [self.samples[i] for i in order]
