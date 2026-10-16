"""Stage 1 as an in-process step: GeoSet-2 decoding of one scene's reference frame, masks kept in memory.

The CLI (methods/geomem/run.py, scripts/run_geomem.sh) writes Data/<scene>/Result/<out>/ for every scene;
this class runs the same decode_scene on one scene and hands the masks back, so the stage-2 runner
(SCSam3/runMVSeg.py --stage1) can seed its tracker without a file in between.  Nothing is written unless
`save` is called (debugging / provenance).  Runs inside `scsam3`.

    s1 = Stage1("geoset_rep")            # a registered preset (S1-E8): arm + gate
    res = s1.run("Fencing")              # {"final": {cam: {obj: bool HxW}}, "round1": ..., "round1_conf": ..., "config": ...}
    s1.save(res, "/tmp/x")               # optional: final/ round1/ <cam>/<obj>.png, round1_conf.json, config.json
    s1.close()                           # frees the model, restores the autocast-cache setting

`final` and `round1` hold exactly what the CLI's <out> and <r1_out> folders hold: the reference camera's GT
masks plus every non-empty predicted mask (an absent (cam, obj) = empty prediction).
"""
import gc
import json
import os
from types import SimpleNamespace

from methods import common
from methods.geomem import run as R
from mvseg2d.io import labelmap_to_mask, write_mask

# the registered stage-1 configurations (docs/stage1-E8-prereg.md; MVSeed_<tag> seed sets)
PRESETS = {
    "geoset_rep": dict(arm="geoset_pred_ft", gate="reproj"),      # S1-E8 adopted (headline)
    "geoset_abs": dict(arm="geoset_pred_ft", gate="absent"),
    "geoset_dec_rep": dict(arm="geoset_dec_ft", gate="reproj"),    # S1-E9 D1: + mask decoder trained
    "geoset_decb_rep": dict(arm="geoset_decb_ft", gate="reproj"),  # S1-E9 D2: + decoder + boundary-band BCE
    "geoalign_rep": dict(arm="geoalign_ft", gate="reproj"),
    # S1-E10 (docs/stage1-E10-prereg.md): the adopted preset + the 2x crop-zoom pass; appended last (TAGS order is pinned)
    "geoset_rep_z2": dict(arm="geoset_pred_ft", gate="reproj", zoom=2, zoom_rope="grid"),     # C1
    "geoset_rep_z2r": dict(arm="geoset_pred_ft", gate="reproj", zoom=2, zoom_rope="remap"),   # C2
    # S1-E11 (docs/stage1-E11-prereg.md): in-view object exclusivity by the decoder logits; appended last
    "geoset_rep_x": dict(arm="geoset_pred_ft", gate="reproj", xcl="out"),        # X1: output, nested-mask cap
    "geoset_rep_xm": dict(arm="geoset_pred_ft", gate="reproj", xcl="outmem"),    # X2: + round-1 memories
    # S1-E12 (docs/stage1-E12-prereg.md): M+ = memory-token half-plane dropout training, adopted decode path; appended last
    "geoset_cutall_rep_x": dict(arm="geoset_cutall_ft", gate="reproj", xcl="out"),
    # S1-E14 (docs/stage1-E14-prereg.md): S = the adopted preset with the output threshold L > -0.5 (coverage calibration); last
    "geoset_rep_xs": dict(arm="geoset_pred_ft", gate="reproj", xcl="outs"),
}
DEFAULT_DATA = os.path.join(common.REPO, "Data")
DEFAULT_RUNS = os.path.join(common.REPO, "geomem", "runs")
FINAL, ROUND1 = "final", "round1"      # sink keys standing in for the CLI's --out / --r1-out folder names


class MemorySink:
    """decode_scene's sink that keeps everything: masks[out][cam][obj] = bool array, files[(out, name)] = obj."""

    def __init__(self):
        self.masks, self.files = {}, {}

    def mask(self, out_name, cam, obj, mask_bool):
        self.masks.setdefault(out_name, {}).setdefault(cam, {})[int(obj)] = mask_bool.copy()

    def json(self, out_name, filename, obj):
        self.files[(out_name, filename)] = obj

    def finish_aux(self, scene, out_name, config):
        pass                                   # the reference GT is added by Stage1.run for both outputs


def reference_masks(scene):
    """What run_all/copy_reference write for the reference camera: the GT mask of every object."""
    return {int(o): labelmap_to_mask(scene.ref_labels, o) for o in scene.objects}


def check_preflight(tag, scene_name, data=DEFAULT_DATA, runs=DEFAULT_RUNS):
    """Everything checkable without a model: the preset, the scene's inputs, its own-fold checkpoint and the
    leak guard.  ValueError to refuse; returns {"arm", "gate", "checkpoint", "fold", "ckpt_sha256"}."""
    if tag not in PRESETS:
        raise ValueError(f"stage-1 preset {tag!r} is not one of {sorted(PRESETS)}")
    p = PRESETS[tag]
    if not os.path.isfile(os.path.join(data, scene_name, "meta.json")):
        raise ValueError(f"no stage-1 inputs for {scene_name}: {os.path.join(data, scene_name, 'meta.json')} is missing")
    from geomem.folds import load_folds
    try:
        folds = R.with_external_fold(load_folds(), scene_name, common.load_meta(data, scene_name).get("external_fold"))
        ckpt = R.checkpoint_for(scene_name, p["arm"], runs, folds)
    except (FileNotFoundError, KeyError) as exc:
        raise ValueError(str(exc))
    with open(os.path.join(os.path.dirname(ckpt), "config.json")) as f:
        cfg = json.load(f)
    try:
        R.assert_no_leak(cfg, scene_name)
    except RuntimeError as exc:
        raise ValueError(str(exc))
    return dict(arm=p["arm"], gate=p["gate"], checkpoint=ckpt, fold=cfg.get("fold"), ckpt_sha256=cfg.get("ckpt_sha256"),
                zoom=p.get("zoom"), zoom_rope=p.get("zoom_rope"), xcl=p.get("xcl", "none"))


class Stage1:
    def __init__(self, tag, data=DEFAULT_DATA, runs=DEFAULT_RUNS):
        if tag not in PRESETS:
            raise ValueError(f"stage-1 preset {tag!r} is not one of {sorted(PRESETS)}")
        self.tag, self.data, self.runs = tag, data, runs
        self.preset = dict(PRESETS[tag])                      # every key reaches decode_scene's args (S1-E10 review)
        self.arm, self.gate = self.preset["arm"], self.preset["gate"]
        self._cache_was = None

    def load(self):
        """The model, as methods/geomem/run.py main() loads it.  sam3_single.load_model turns the autocast
        weight cache off (checkpoint loading needs that); the previous setting comes back in close()."""
        if R.decode_scene.model is not None:
            return
        import torch
        from methods import sam3_single
        self._cache_was = torch.is_autocast_cache_enabled()
        R.decode_scene.model = sam3_single.load_model()
        R.decode_scene.init_state = R.snapshot_init(R.decode_scene.model)

    def run(self, scene_name):
        self.load()
        scene = common.load_scene(self.data, scene_name)
        base = dict(data=self.data, out=FINAL, r1_out=ROUND1, untrained=False, runs=self.runs,
                    tau=R.TAU_CONF, k=R.K_NEIGHBOURS, round2="pred")
        args = SimpleNamespace(**{**base, **self.preset})      # the preset's keys (arm, gate, zoom, ...) win
        R.check_gate_out(args.arm, args.gate, args.out, args.round2, getattr(args, "zoom", None), getattr(args, "xcl", "none"))
        sink = MemorySink()
        config = R.decode_scene(scene, args, sink=sink)
        ref = reference_masks(scene)
        out = {}
        for key in (FINAL, ROUND1):
            per = {cam: dict(objs) for cam, objs in sink.masks.get(key, {}).items()}
            per[scene.ref_cam] = {o: m.copy() for o, m in ref.items()}   # copy_reference overwrites the GT in place
            out[key] = per
        return dict(final=out[FINAL], round1=out[ROUND1], round1_conf=sink.files.get((FINAL, "round1_conf.json")),
                    zoom_conf=sink.files.get((FINAL, "zoom_conf.json")), xcl_conf=sink.files.get((FINAL, "xcl_conf.json")),
                    config=config, scene=scene, tag=self.tag)

    @staticmethod
    def save(result, folder):
        """Debug copy in the CLI's layout: <folder>/final/<cam>/<obj>.png (= Result/<out>), round1/ (= <r1_out>),
        round1_conf.json, config.json.  PNGs through the same writer as the CLI, so the bytes compare."""
        for key in (FINAL, ROUND1):
            for cam, objs in result[key].items():
                for o, m in objs.items():
                    write_mask(os.path.join(folder, key, cam, f"{int(o)}.png"), m)
        os.makedirs(folder, exist_ok=True)
        if result["round1_conf"] is not None:
            with open(os.path.join(folder, "round1_conf.json"), "w") as f:
                json.dump(result["round1_conf"], f, indent=1)
        if result.get("zoom_conf") is not None:
            with open(os.path.join(folder, "zoom_conf.json"), "w") as f:
                json.dump(result["zoom_conf"], f, indent=1)
        if result.get("xcl_conf") is not None:
            with open(os.path.join(folder, "xcl_conf.json"), "w") as f:
                json.dump(result["xcl_conf"], f, indent=1)
        with open(os.path.join(folder, "config.json"), "w") as f:
            json.dump(dict(result["config"], tag=result["tag"], scene=result["scene"].name), f, indent=1)

    def close(self):
        """Free the model (stage 2 needs the GPU) and put the autocast-cache setting back."""
        import torch
        R.decode_scene.model = None
        R.decode_scene.init_state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._cache_was is not None:
            torch.set_autocast_cache_enabled(self._cache_was)
            self._cache_was = None
