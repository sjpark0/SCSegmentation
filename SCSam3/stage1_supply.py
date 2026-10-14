"""In-process stage 1 for runMVSeg.py --stage1 <tag>: run the stage-1 GeoSet-2 decoder (stage1/) on the
scene's reference frame inside the stage-2 process and hand its masks to the seed slot as memory, not files.

Equivalent of the file route  stage1/scripts/run_geomem.sh -> MVSeed/build_geoset_seeds.py ->
runMVSeg.py --seeds-from MVSeed_<tag>:  to_seeds() applies the seed builder's rules (every Video camera x
every basic object, a missing prediction = an empty seed) and seeds_from.apply_seeds() the --seeds-from
rules, so the run is the same as with the folder and keeps its folder suffix Sd<tag>.

Torch-free at import (resolve_run needs the tag rule); stage 1 itself is imported only by run_stage1().
Inputs stay where stage 1 keeps them: stage1/Data/<scene>/ (frame images, reference GT, the precomputed
Pi3 geometry cache) and the fold checkpoints stage1/geomem/runs/<arm>/fold<k>/.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STAGE1_ROOT = os.path.join(REPO, "stage1")
CENSUS_PATH = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
TAGS = ("geoset_rep", "geoset_abs", "geoset_dec_rep", "geoset_decb_rep", "geoalign_rep",
        "geoset_rep_z2", "geoset_rep_z2r")   # == stage1/methods/geomem/stage1.py PRESETS (ordered); S1-E10 crop-zoom tags last
SUFFIX = "Sd"                                           # the --seeds-from MVSeed_<tag> suffix: same seeds, same name


def folder_suffix(tag):
    if tag not in TAGS:
        raise ValueError(f"--stage1 takes one of {TAGS}, not {tag!r}")
    return SUFFIX + tag


def stage1_module():
    """stage1/methods/geomem/stage1.py with stage1/ importable (its packages: methods, geomem, mvseg2d)."""
    if STAGE1_ROOT not in sys.path:
        sys.path.append(STAGE1_ROOT)       # appended: never ahead of the SAM 3 package build_runner put first
    from methods.geomem import stage1 as S
    return S


def basic_objects(dataset):
    with open(CENSUS_PATH) as fh:
        census = json.load(fh)
    return sorted(set(int(v) for v in census[dataset]["seed_ids_c_ini"])), census[dataset]["c_ini"]


def preflight(tag, dataset, start_frame, c_ini_cam):
    """Before any model: the preset, the stage-1 inputs and checkpoint (S.check_preflight), and the seed
    builder's consistency checks -- stage 1's keyframe and reference camera are stage 2's start_frame and
    c_ini.  ValueError to refuse."""
    folder_suffix(tag)
    S = stage1_module()
    info = S.check_preflight(tag, dataset)
    with open(os.path.join(S.DEFAULT_DATA, dataset, "meta.json")) as fh:
        meta = json.load(fh)
    basic, census_c_ini = basic_objects(dataset)
    if meta["ref_frame"] != start_frame or meta["ref_cam"] != census_c_ini or census_c_ini != c_ini_cam:
        raise ValueError(f"stage-1 keyframe/reference {meta['ref_frame']}/{meta['ref_cam']} != stage-2 "
                         f"start_frame/c_ini {start_frame}/{c_ini_cam} (census {census_c_ini})")
    info.update(basic_objects=basic, stage1_cams=list(meta["cams"]))
    return info


def to_seeds(final, video_cams, basic, shape, c_ini_cam, np):
    """MVSeed/build_geoset_seeds.py's rules on in-memory masks: for every Video camera x every basic
    object, the stage-1 mask if there is one, else an empty seed.  Refuses what the builder refuses:
    cameras outside Video/, a reference camera without masks, objects outside the basic set, a mask of
    another size.  Returns (seeds {cam: {obj: bool}}, counts)."""
    extra_cams = sorted(set(final) - set(video_cams))
    if extra_cams:
        raise ValueError(f"stage-1 cameras {extra_cams} are not Video/ cameras")
    if c_ini_cam not in final:
        raise ValueError(f"stage 1 has no masks for the reference camera {c_ini_cam}")
    extra = [(cam, o) for cam, objs in final.items() for o in objs if o not in basic]
    if extra:
        raise ValueError(f"stage-1 masks outside the basic object set: {extra[:5]}")
    seeds, n_copy, n_empty, n_zero = {}, 0, 0, 0
    for cam in video_cams:
        per = {}
        for o in basic:
            m = final.get(cam, {}).get(o)
            if m is None:
                per[o] = np.zeros(shape, bool)
                n_empty += 1
            else:
                if tuple(m.shape) != tuple(shape):
                    raise ValueError(f"stage-1 mask {cam}/{o}: shape {tuple(m.shape)}, video {tuple(shape)}")
                per[o] = np.asarray(m, bool)
                n_copy += 1
                n_zero += not per[o].any()
        seeds[cam] = per
    return seeds, dict(n_seeds=n_copy + n_empty, n_from_stage1=n_copy, n_empty=n_empty, n_from_stage1_zero=n_zero,
                       cams_without_masks=sorted(set(video_cams) - set(final)))


def mask_digest(seeds, np):
    """sha256 over the seeds in (camera, object) order: name, shape and the packed bits of each mask."""
    h = hashlib.sha256()
    for cam in sorted(seeds):
        for o in sorted(seeds[cam]):
            m = np.ascontiguousarray(seeds[cam][o], bool)
            h.update(f"{cam}/{o}:{m.shape[0]}x{m.shape[1]}\n".encode())
            h.update(np.packbits(m).tobytes())
    return h.hexdigest()


def write_seed_folder(seeds, folder, start_frame, manifest, cv2, np):
    """--save-stages: the seeds as an MVSeed_ folder (<cam>/<start_frame>/<obj>.png, 0/255, plus
    SEED_MANIFEST.json), the layout --seeds-from reads."""
    for cam, objs in seeds.items():
        d = os.path.join(folder, cam, str(start_frame))
        os.makedirs(d, exist_ok=True)
        for o, m in objs.items():
            cv2.imwrite(os.path.join(d, f"{o}.png"), m.astype(np.uint8) * 255)
    with open(os.path.join(folder, "SEED_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


def run_stage1(tag, dataset, video_cams, shape, c_ini_cam, start_frame, np, save_dir=None, cv2=None, log=print):
    """Stage 1 on one scene -> (seeds, record).  The stage-1 model is freed before returning, so stage 2
    starts on an empty GPU.  save_dir: also write stage 1's outputs (final/, round1/, round1_conf.json,
    config.json in the CLI's layout) and the seeds (seeds/, an MVSeed_ folder)."""
    import time
    S = stage1_module()
    s1 = S.Stage1(tag)
    t0 = time.time()
    try:
        res = s1.run(dataset)
    finally:
        s1.close()
    seconds = round(time.time() - t0, 1)
    basic, _ = basic_objects(dataset)
    seeds, counts = to_seeds(res["final"], video_cams, basic, shape, c_ini_cam, np)
    cfg = res["config"]
    record = dict(mode="stage1-inprocess", tag=tag, arm=cfg.get("arm"), gate=cfg.get("gate"), fold=cfg.get("fold"),
                  ckpt_sha256=cfg.get("ckpt_sha256"), stage1_seconds=seconds, mask_digest=mask_digest(seeds, np),
                  stage1_config=cfg, **counts,
                  **{k: cfg[k] for k in ("zoom", "zoom_rope", "n_zoom_run", "n_zoom_changed") if k in cfg})   # S1-E10 presets only
    if save_dir is not None:
        s1.save(res, os.path.join(save_dir, "stage1"))
        man = dict(schema="MVSeed stage1_supply", schema_version=1, set=tag, dataset=dataset, start_frame=start_frame,
                   c_ini=c_ini_cam, written_views=list(video_cams), objects=basic,
                   source=dict(mode="stage1-inprocess", arm=record["arm"], gate=record["gate"], fold=record["fold"],
                               ckpt_sha256=record["ckpt_sha256"]), mask_digest=record["mask_digest"], **counts)
        write_seed_folder(seeds, os.path.join(save_dir, "seeds"), start_frame, man, cv2, np)
        record["saved"] = save_dir
    log(f"stage 1        {tag} ({record['arm']}, gate {record['gate']}, fold{record['fold']}) {seconds} s: "
        f"{counts['n_from_stage1']} masks + {counts['n_empty']} empty seeds, digest {record['mask_digest'][:12]}"
        + (f"; saved -> {save_dir}" if save_dir else ""))
    return seeds, record
