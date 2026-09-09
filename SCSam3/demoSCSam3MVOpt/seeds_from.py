"""Stage-1 supply hook: replace cross-view seeds from an MVSeed_ folder (--seeds-from).

Why this exists.  Stage 1 (the seed each view starts tracking from) decides the result
(docs/stage1-headroom.md: R^2 0.933), so new seed generators are developed in MVSeed/ and
the winner must be measured through the unchanged stage 2.  docs/stage1-plan.md section 3
("2단계 공급") fixes how the seeds get in: the runner still prompts the reference and runs
PropagateAcrossViews (that gives obj_ids and a G0 seed for every view), then, in the slot
P5's RepairSeeds occupies, swaps in whatever the folder holds.  A (view, obj) with no PNG
keeps its G0 seed -- a partial folder is a partial treatment, not a hole.

Torch-free so it is unit-testable on the host, like seed_repair.py.  The runner's
LoadSeedsFrom() hands over masks_spatial, the folder and a PNG reader; every decision
that needs no model lives here.

  folder name    <scene>/MVSeed_<tag>/ (MVSeed/README.md rule R2); tag names the output
                 folder suffix Sd<tag>, so a treatment run never lands on a control folder
  layout         <cam>/<start_frame>/<obj>.png, 0/255 as run_seed.py and the runner write
                 them; a value > 127 is foreground, the reading eval_jf.py uses
  SEED_MANIFEST  optional.  When present its written_views (or the older written_cameras)
                 must cover every scored camera -- a seed set that skips one of them
                 would be scored on G0 there and read as a treatment effect; absent, the
                 caller is told to warn and go on
  provenance     sha256 of the manifest bytes and a digest over the PNGs built exactly as
                 eval/manifest.py's content_digest (sorted (relative path, sha256) pairs),
                 so a MANIFEST.json can say which seed bytes a run consumed
"""
import hashlib
import json
import os
import re

PREFIX = "MVSeed_"
SUFFIX = "Sd"
MANIFEST = "SEED_MANIFEST.json"
THRESHOLD = 127                 # PNG value > 127 -> foreground (eval_jf.py reads masks so)
TAG_RE = re.compile(r"[A-Za-z0-9._-]+")


# ------------------------------------------------------------------ folder name
def parse_folder(name):
    """'MVSeed_<tag>' -> tag.  ValueError for anything that is not one folder name of
    that shape: another prefix, an empty tag, a path, odd characters."""
    if not isinstance(name, str) or not name.startswith(PREFIX):
        raise ValueError(f"seed folders start with {PREFIX!r} (MVSeed/README.md R2), got {name!r}")
    tag = name[len(PREFIX):]
    if not tag:
        raise ValueError(f"{name!r} has an empty tag after {PREFIX!r}")
    if not TAG_RE.fullmatch(tag):
        raise ValueError(f"{name!r} must be one folder name under the scene directory "
                         f"(tag of [A-Za-z0-9._-]), not a path")
    return tag


def folder_suffix(name):
    """The output-folder suffix a run seeded from `name` gets: Sd<tag>."""
    return SUFFIX + parse_folder(name)


# ------------------------------------------------------------------ manifest
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_manifest(folder):
    """(manifest dict, sha256 of its bytes), or (None, None) when the file is absent."""
    path = os.path.join(folder, MANIFEST)
    if not os.path.isfile(path):
        return None, None
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        return json.loads(data.decode("utf-8")), sha256_bytes(data)
    except ValueError as exc:
        raise ValueError(f"{path} is not JSON ({exc})")


def written_cams(manifest, cam_names):
    """The cameras the manifest says it wrote, as names.  `written_views` (stage1-plan
    section 6) or the older `written_cameras` (run_seed.py before that plan); entries are
    camera names or view indices into cam_names."""
    for key in ("written_views", "written_cameras"):
        if key in manifest:
            entries = manifest[key]
            break
    else:
        raise ValueError(f"{MANIFEST} has neither written_views nor written_cameras, "
                         "so its coverage cannot be checked")
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"{MANIFEST} {key} must be a list, got {type(entries).__name__}")
    out = []
    for e in entries:
        if isinstance(e, bool) or not isinstance(e, (int, str)):
            raise ValueError(f"{MANIFEST} {key} entry {e!r} is neither a camera name nor a view index")
        if isinstance(e, int):
            if not 0 <= e < len(cam_names):
                raise ValueError(f"{MANIFEST} {key} view index {e} is outside 0..{len(cam_names) - 1}")
            e = cam_names[e]
        out.append(e)
    return out


def check_coverage(manifest, required, cam_names):
    """Every scored camera in the manifest's written set -> the written names.  ValueError
    naming the uncovered cameras otherwise."""
    written = written_cams(manifest, cam_names)
    missing = [c for c in required if c not in written]
    if missing:
        raise ValueError(f"{MANIFEST} covers {sorted(written)} but the scored cameras "
                         f"{list(required)} need {missing} too")
    return written


# ------------------------------------------------------------------ PNG digest
def list_pngs(folder):
    """Every *.png below `folder`, as sorted posix-style relative paths."""
    out = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for f in filenames:
            if f.lower().endswith(".png"):
                rel = os.path.relpath(os.path.join(dirpath, f), folder)
                out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def png_digest(folder):
    """(digest, n_png): sha256 over the sorted (relative path, sha256 of bytes) pairs, each
    fed as '<path>\\0<hex>\\n' -- eval/manifest.py CONTENT_DIGEST_SPEC, so the value is
    comparable with a MANIFEST.json content_digest of the same folder."""
    rels = list_pngs(folder)
    h = hashlib.sha256()
    for rel in rels:                                # list_pngs is sorted already
        with open(os.path.join(folder, rel), "rb") as fh:
            h.update(f"{rel}\0{sha256_bytes(fh.read())}\n".encode("utf-8"))
    return h.hexdigest(), len(rels)


def folder_provenance(folder):
    """What MANIFEST.json records under provenance.seeds_from about the folder itself."""
    _, manifest_sha = read_manifest(folder)
    digest, n_png = png_digest(folder)
    return {"folder": os.path.basename(os.path.normpath(folder)),
            "path": os.path.abspath(folder),
            "manifest_sha256": manifest_sha, "png_digest": digest, "n_png": n_png}


# ------------------------------------------------------------------ layout
def list_seed_pngs(folder, start_frame, cam_names):
    """{cam: {obj: path}} for <cam>/<start_frame>/<obj>.png.  A camera directory the run
    does not know, or a stem that is not an object id, is a folder for another scene or
    another writer: ValueError.  Plain files (the manifest) are ignored."""
    known = set(cam_names)
    frame = f"{start_frame:d}"
    pngs = {}
    for entry in sorted(os.listdir(folder)):
        cam_dir = os.path.join(folder, entry)
        if not os.path.isdir(cam_dir):
            continue
        if entry not in known:
            raise ValueError(f"camera folder {entry!r} is not one of this run's cameras "
                             f"({len(cam_names)} loaded): another scene's seeds?")
        frame_dir = os.path.join(cam_dir, frame)
        if not os.path.isdir(frame_dir):
            continue
        per = {}
        for f in sorted(os.listdir(frame_dir)):
            if not f.lower().endswith(".png"):
                continue
            stem = f[:-4]
            if not stem.isdigit():
                raise ValueError(f"{entry}/{frame}/{f}: object PNGs are named <obj id>.png")
            per[int(stem)] = os.path.join(frame_dir, f)
        pngs[entry] = per
    return pngs


def preflight(folder, start_frame, cam_names, required, expect=None):
    """Everything checkable before a model is loaded.  ValueError to refuse; returns
    {"manifest", "manifest_sha256", "written", "warning", "pngs"} -- `warning` is the
    text to print when there is no manifest to check coverage against, else None.
    `expect` = {manifest key: value} the manifest must agree with when it has the key
    (dataset, start_frame): cheap insurance against seeding one scene from another."""
    if not os.path.isdir(folder):
        raise ValueError(f"{folder} is not a directory")
    parse_folder(os.path.basename(os.path.normpath(folder)))
    manifest, sha = read_manifest(folder)
    written, warning = None, None
    if manifest is None:
        warning = (f"no {MANIFEST} in {folder}: coverage of the scored cameras "
                   f"{list(required)} is not checked")
    else:
        for key, value in (expect or {}).items():
            if key in manifest and manifest[key] != value:
                raise ValueError(f"{MANIFEST} says {key}={manifest[key]!r}, this run has {value!r}")
        written = check_coverage(manifest, required, cam_names)
    pngs = list_seed_pngs(folder, start_frame, cam_names)
    return {"manifest": manifest, "manifest_sha256": sha, "written": written,
            "warning": warning, "pngs": pngs}


# ------------------------------------------------------------------ apply
def to_seed(img):
    """Grayscale PNG array -> bool mask, the eval_jf.py reading (> 127)."""
    return img > THRESHOLD


def apply_seed_folder(masks_spatial, folder, start_frame, cam_names, track_idx, obj_ids,
                      shape, read_png):
    """Replace masks_spatial[view][obj] for every (tracked view, obj) that has a PNG.

    masks_spatial  {view: {obj: bool (H, W) array}} as PropagateAcrossViews left it; an
                   object absent from a view's dict is a G0 seed of area 0 (the tracker
                   drops zero-area outputs)
    track_idx      the views TrackForward will consume; a PNG for any other loaded view
                   is listed under untracked_views and left alone, so view_areas.seed of
                   the unscored views stays comparable with the control run
    obj_ids        the prompted objects; a PNG for any other id is a mismatch with the
                   reference prompt (ValueError), not something to guess about
    shape          (H, W) of the video; a PNG of another size is refused before it can
                   be reshaped into nonsense
    read_png       path -> 2-D array (cv2.imread(..., IMREAD_GRAYSCALE)) or None

    Returns the record for MANIFEST.json (str keys, ints and lists only):
      replaced        number of (view, obj) pairs now holding a folder seed
      replaced_views  {view: [obj, ...]}
      changed_views   {view: [obj, ...]} the subset whose seed differs from G0 -- the
                      reference view should not appear here (stage1-plan section 2)
      added           pairs whose G0 seed was empty (object absent from the view)
      emptied         pairs whose G0 seed was non-empty and whose PNG is all background
      untracked_views loaded views with PNGs that no session tracks (not applied)
    """
    pngs = list_seed_pngs(folder, start_frame, cam_names)
    tracked = set(track_idx)
    allowed = set(obj_ids)
    index = {name: i for i, name in enumerate(cam_names)}
    replaced, changed, untracked = {}, {}, []
    added = emptied = 0
    for cam in sorted(pngs, key=index.get):
        view = index[cam]
        per = pngs[cam]
        bad = sorted(o for o in per if o not in allowed)
        if bad:
            raise ValueError(f"{cam}: objects {bad} are not among the prompted {sorted(allowed)}")
        if view not in tracked:
            if per:
                untracked.append(view)
            continue
        seeds = masks_spatial.setdefault(view, {})
        for obj in sorted(per):
            img = read_png(per[obj])
            if img is None:
                raise ValueError(f"cannot read {per[obj]}")
            if tuple(img.shape) != tuple(shape):
                raise ValueError(f"{per[obj]}: shape {tuple(img.shape)} is not the video's {tuple(shape)}")
            new = to_seed(img)
            old = seeds.get(obj)
            new_any = bool(new.any())
            if old is None:
                added += 1
                differs = new_any
            else:
                if not new_any and bool(old.any()):
                    emptied += 1
                differs = bool((old != new).any())
            seeds[obj] = new
            replaced.setdefault(view, []).append(obj)
            if differs:
                changed.setdefault(view, []).append(obj)
    return {"replaced": sum(len(v) for v in replaced.values()),
            "replaced_views": {str(v): objs for v, objs in sorted(replaced.items())},
            "changed_views": {str(v): objs for v, objs in sorted(changed.items())},
            "added": added, "emptied": emptied, "untracked_views": sorted(untracked)}
