"""S2-R5 (docs/stage2-R5-prereg.md): run the stage-2 temporal tracker with the memory attention
that stage 1 fine-tuned (SCSegmentation2D, branch geomem, arm geoset_pred_ft).

The module swapped is `sc.predictor.model.tracker.transformer.encoder` -- the 4-layer
TransformerEncoderCrossAttention built by build_scsam3._create_tracker_transformer, the same
constructor the 2D repo builds its model with, so the checkpoint's "memory_attention" state_dict
loads with strict=True.  Only that state_dict is used: no 3D bias, no 3D embedding.

Loaded by path from runMVSeg.py (as crop_track.py: SCSam3/ never goes on sys.path).  The
folder rule and the checkpoint lookup are torch-free; swap / restore / digest take `torch`
as an argument.  The caller clears the autocast cache after every swap or restore: the
tracker's never-exited bf16 autocast keys its weight cache on the parameter objects, which
an in-place load_state_dict keeps, so without the clear the old bf16 copies stay in use.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SIBLING = os.path.normpath(os.path.join(HERE, "..", "..", "SCSegmentation2D"))
DEFAULT_RUNS = os.path.join(SIBLING, "geomem", "runs")
DEFAULT_FOLDS = os.path.join(SIBLING, "geomem", "folds.json")
SUFFIX_PREFIX = "Ft"
ARMS = {"geoset_pred_ft": "gp"}            # 2D arm -> folder tag: --ft-encoder geoset_pred_ft -> Ftgp


def folder_suffix(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown fine-tuned encoder arm {arm!r} (known: {sorted(ARMS)})")
    return SUFFIX_PREFIX + ARMS[arm]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(arm, scene, runs_dir=DEFAULT_RUNS, folds_path=DEFAULT_FOLDS):
    """The checkpoint a scene is evaluated with: its own fold's (the fold that did not train on
    it).  Pure file checks, before any model is built.  Raises ValueError."""
    folder_suffix(arm)
    # absolute now: build_runner changes the working directory before the checkpoint is read
    runs_dir, folds_path = os.path.abspath(runs_dir), os.path.abspath(folds_path)
    if not os.path.isfile(folds_path):
        raise ValueError(f"no fold table at {folds_path}")
    with open(folds_path, encoding="utf-8") as f:
        folds = json.load(f)
    if scene not in folds:
        raise ValueError(f"{scene} is not in the fold table {folds_path}")
    fold = int(folds[scene])
    path = os.path.join(runs_dir, arm, f"fold{fold}", "ckpt.pt")
    if not os.path.isfile(path):
        raise ValueError(f"no checkpoint for {scene}: expected {path} (arm {arm}, fold{fold})")
    return dict(arm=arm, scene=scene, fold=fold, path=path, sha256=sha256_file(path),
                folds_path=folds_path, folds_sha256=sha256_file(folds_path))


def check_checkpoint(ck, info):
    """The leak check (prereg section 2): the checkpoint's own record must say it is this arm,
    this fold, and was not trained on this scene.  Returns the training scenes."""
    if ck.get("arm") != info["arm"]:
        raise ValueError(f"{info['path']}: arm {ck.get('arm')!r}, expected {info['arm']!r}")
    if int(ck.get("fold", -1)) != info["fold"]:
        raise ValueError(f"{info['path']}: fold {ck.get('fold')!r}, the fold table says {info['fold']}")
    scenes = list(ck.get("scenes") or [])
    if not scenes:
        raise ValueError(f"{info['path']}: no training-scene record, cannot check for a leak")
    if info["scene"] in scenes:
        raise ValueError(f"{info['path']}: fold{info['fold']} was trained on {info['scene']}; refusing")
    if "memory_attention" not in ck:
        raise ValueError(f"{info['path']}: no memory_attention state_dict")
    return scenes


def state_digest(state, torch):
    """sha256 over (name, dtype, shape, fp32 bytes) of a state_dict, names sorted.  Device- and
    storage-independent, so a module's digest equals its checkpoint's after a strict load."""
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().to("cpu")
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def check_compatible(encoder, new_state):
    """Same keys and shapes, checked before anything is copied: a strict load_state_dict that
    fails has already copied the keys that matched, so it would leave the module half-swapped."""
    have = encoder.state_dict()
    missing, unexpected = sorted(set(have) - set(new_state)), sorted(set(new_state) - set(have))
    shapes = sorted(k for k in set(have) & set(new_state) if tuple(have[k].shape) != tuple(new_state[k].shape))
    if missing or unexpected or shapes:
        raise ValueError(f"state_dict does not fit the encoder: missing {missing[:3]}, unexpected "
                         f"{unexpected[:3]}, shape mismatch {shapes[:3]}")


def probe_digest(encoder, torch, n=4):
    """sha256 of what every nn.Linear of `encoder` outputs on a fixed input, run where the tracker
    runs (the caller's thread, inside its never-exited autocast), plus the output dtypes.  The
    parameter digests above cannot see a stale autocast cache -- the fp32 weights change while
    the forwards keep using the old bf16 copies -- this probe does: it goes through the same
    cache.  Value-neutral for the run: the caller clears the cache after every swap / restore."""
    h, dtypes = hashlib.sha256(), set()
    g = torch.Generator().manual_seed(0)
    dev = next(encoder.parameters()).device
    with torch.inference_mode():
        for name, m in encoder.named_modules():
            if isinstance(m, torch.nn.Linear):
                y = m(torch.randn(n, m.in_features, generator=g).to(dev))
                dtypes.add(str(y.dtype).replace("torch.", ""))
                h.update(name.encode())
                h.update(y.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest(), sorted(dtypes)


def swap_in(encoder, new_state, torch):
    """Load `new_state` into `encoder` (strict) and return the original state_dict, cloned to the
    CPU, for restore().  The caller clears the autocast cache afterwards."""
    check_compatible(encoder, new_state)
    original = {k: v.detach().to("cpu").clone() for k, v in encoder.state_dict().items()}
    with torch.no_grad():
        encoder.load_state_dict(new_state, strict=True)
    return original


def restore(encoder, original, torch):
    with torch.no_grad():
        encoder.load_state_dict(original, strict=True)
