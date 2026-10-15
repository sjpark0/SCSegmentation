#!/usr/bin/env python3
"""S1-E14 logit dump (scratch, git-ignored): methods/geomem/dev_decode.py with the decode path unchanged, plus the
round-2 decoder outputs of every eval camera (layout and fields: e14dump.py).

Capture (no numerical change; the masks are checked against the adopted DEV-A PNGs):
  * GeoMemStar.decode_view is wrapped; a top-level call WITH `chunk` is round 2 (decode_scene passes chunk=OBJ_CHUNK
    only there), its returned dict gives low_res / ious / presence; the camera is the one whose round-1 call used the
    same backbone dict (round-1 calls are attributed by the path of the preceding sam3_io.load_image).
  * the tracker's MaskDecoder.predict_masks is wrapped (inside decode_view only) to keep all 4 token masks + IoUs; the
    stability score of token 0 is recomputed with the decoder's own _get_stability_scores on the same tensor.
Per frame, after writing: the npz is read back from disk and re-thresholded through the adopted path (e14dump.masks_torch)
and compared with (a) this decode's in-memory final masks, (b) the reference PNG root (default _e13_dev).
    cd stage1 && scripts/docker_run.sh scsam3 python geomem/runs/_e14_logitdump/code/dump_logits.py --scenes A,B
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, S1)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import e14dump as E  # noqa: E402
from geomem import devset, sam3_io  # noqa: E402
from geomem.wrapper import GeoMemStar  # noqa: E402
from methods import common  # noqa: E402
from methods.geomem import dev_decode as D  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem.stage1 import FINAL, check_preflight  # noqa: E402

CAP = dict(depth=0, dec=[], path=None, bb_path={}, r1=[], r2=[])


def install_capture(model):
    orig_load = sam3_io.load_image

    def load_image(path):
        CAP["path"] = path
        return orig_load(path)
    sam3_io.load_image = load_image

    orig_dv = GeoMemStar.decode_view

    def decode_view(self, backbone_out, B, memory, p_v, valid_v, chunk=None, **kw):
        top = CAP["depth"] == 0
        if top:
            CAP["dec"] = []
        CAP["depth"] += 1
        try:
            out = orig_dv(self, backbone_out, B, memory, p_v, valid_v, chunk=chunk, **kw)
        finally:
            CAP["depth"] -= 1
        if top:
            dec, CAP["dec"] = CAP["dec"], []
            if chunk is None:                                   # round 1
                CAP["bb_path"][id(backbone_out)] = CAP["path"]
                CAP["r1"].append(CAP["path"])
            else:                                               # round 2
                assert sum(d["masks"].shape[0] for d in dec) == B, ([d["masks"].shape for d in dec], B)
                CAP["r2"].append(dict(
                    path=CAP["bb_path"][id(backbone_out)], B=B,
                    low_res=out["low_res"].detach()[:, 0].float().cpu(),
                    ious=out["ious"].detach().float().cpu(), presence=out["presence"].detach().float().cpu(),
                    low_res_dtype=str(out["low_res"].dtype),
                    mm=torch.cat([d["masks"] for d in dec]).float(), mm_iou=torch.cat([d["iou"] for d in dec]),
                    obj=torch.cat([d["obj"] for d in dec]), st=torch.cat([d["st"] for d in dec]),
                    mm_dtype=dec[0]["dtype"], dyn=dec[0]["dyn"], thresh=dec[0]["thresh"]))
        return out
    GeoMemStar.decode_view = decode_view

    dec_mod = model.tracker.sam_mask_decoder
    cls = type(dec_mod)
    orig_pm = cls.predict_masks

    def predict_masks(self, *a, **kw):
        res = orig_pm(self, *a, **kw)
        if CAP["depth"] > 0:
            masks, iou_pred, _, obj = res
            st = self._get_stability_scores(masks[:, 0:1])           # the tensor _dynamic_multimask_via_stability sees
            CAP["dec"].append(dict(masks=masks.detach().cpu(), iou=iou_pred.detach().float().cpu(),
                                   obj=obj.detach().float().cpu(), st=st.detach().float().cpu(), dtype=str(masks.dtype),
                                   dyn=bool(self.dynamic_multimask_via_stability and not self.training),
                                   thresh=float(self.dynamic_multimask_stability_thresh)))
        return res
    cls.predict_masks = predict_masks
    return dict(decoder_class=f"{cls.__module__}.{cls.__name__}",
                dynamic_multimask_via_stability=bool(dec_mod.dynamic_multimask_via_stability),
                stability_delta=float(dec_mod.dynamic_multimask_stability_delta),
                stability_thresh=float(dec_mod.dynamic_multimask_stability_thresh),
                decoder_training=bool(dec_mod.training), image_size=int(model.tracker.image_size),
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))


def fp16_inexact(x32):
    return int((x32.half().float() != x32).sum())


def save_view(d, rec, objects, hw):
    """rec: one CAP['r2'] entry -> <cam>.npz, <cam>_mm.npz; returns the per-view record of the checks."""
    B = rec["B"]
    st = rec["st"][:, 0]
    if rec["dyn"]:
        stable = st >= rec["thresh"]
        token = torch.where(stable, 0, 1 + rec["mm_iou"][:, 1:].argmax(1))
    else:
        token = torch.zeros(B, dtype=torch.long)
    bi = torch.arange(B)
    recon = torch.where(rec["obj"][:, 0][:, None, None] > 0, rec["mm"][bi, token], E.NO_OBJ_SCORE)
    chk = dict(B=B, low_res_dtype=rec["low_res_dtype"], mm_dtype=rec["mm_dtype"],
               recon_low_res_equal=bool(torch.equal(recon, rec["low_res"])),
               recon_ious_equal=bool(torch.equal(rec["mm_iou"][bi, token], rec["ious"])),
               recon_presence_equal=bool(torch.equal(rec["obj"][:, 0], rec["presence"])),
               n_fallback=int((token > 0).sum()), n_absent=int((rec["presence"] <= 0).sum()),
               fp16_inexact_low_res=fp16_inexact(rec["low_res"]), fp16_inexact_mm=fp16_inexact(rec["mm"]),
               bf16_valued_low_res=bool(torch.equal(rec["low_res"].bfloat16().float(), rec["low_res"])))
    os.makedirs(d, exist_ok=True)
    cam = rec["cam"]
    np.savez_compressed(os.path.join(d, f"{cam}.npz"), objects=np.asarray(objects, np.int32), hw=np.asarray(hw, np.int32),
                        low_res=rec["low_res"].half().numpy(), ious=rec["ious"].numpy().astype(np.float32),
                        presence=rec["presence"].numpy().astype(np.float32), token=token.numpy().astype(np.int8),
                        stability=st.numpy().astype(np.float32))
    np.savez_compressed(os.path.join(d, f"{cam}_mm.npz"), mm_low_res=rec["mm"].half().numpy(),
                        mm_ious=rec["mm_iou"].numpy().astype(np.float32))
    return chk


def check_view(root, ref_root, scene, t, cam, sink_masks, objects, hw):
    """Re-threshold the npz READ BACK from disk -> compare with this decode's masks and the reference PNGs."""
    z = E.load(root, scene, t, cam)
    assert [int(o) for o in z["objects"]] == [int(o) for o in objects]
    keep = E.masks_torch(z["low_res"], z["hw"])
    mem = np.zeros_like(keep)
    for i, o in enumerate(objects):
        if o in sink_masks:
            mem[i] = sink_masks[o]
    ref = E.read_png_masks(ref_root, scene, t, cam, objects, hw)
    own = E.read_png_masks(root, scene, t, cam, objects, hw)
    per_obj = {}
    for i, o in enumerate(objects):
        pa = os.path.join(root, scene, f"{t:06d}", cam, f"{int(o)}.png")
        pb = os.path.join(ref_root, scene, f"{t:06d}", cam, f"{int(o)}.png")
        fa, fb = os.path.isfile(pa), os.path.isfile(pb)
        file_eq = (fa == fb) and (not fa or open(pa, "rb").read() == open(pb, "rb").read())
        per_obj[str(int(o))] = dict(
            rethr_vs_ref_px=int((keep[i] != ref[i]).sum()), rethr_vs_mem_px=int((keep[i] != mem[i]).sum()),
            mem_vs_ref_px=int((mem[i] != ref[i]).sum()), own_png_vs_ref_px=int((own[i] != ref[i]).sum()),
            png_bytes_equal=bool(file_eq), png_present=bool(fb), fg_px=int(ref[i].sum()))
    return per_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="geoset_rep_x")
    ap.add_argument("--split", default="A", choices=("A",))     # DEV-B is reserved; TEST never decoded
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--root", default=os.path.join(S1, "geomem", "runs", "_e14_logitdump", "geoset_rep_x_A"))
    ap.add_argument("--ref-root", default=os.path.join(S1, "geomem", "runs", "_e13_dev"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    scenes = common.all_scenes(D.DATA) if a.scenes == "all" else a.scenes.split(",")
    os.makedirs(a.root, exist_ok=True)
    idx = devset.index(D.DATA, common.all_scenes(D.DATA), a.split)
    with open(os.path.join(a.root, "index.json"), "w") as f:
        json.dump(dict(preset=a.preset, **idx), f, indent=1)
    from methods import sam3_single
    R.decode_scene.model = sam3_single.load_model()
    R.decode_scene.init_state = R.snapshot_init(R.decode_scene.model)
    env = install_capture(R.decode_scene.model)
    print("ENV", json.dumps(env), flush=True)
    for s in scenes:
        pf = check_preflight(a.preset, s)
        basic = idx["scenes"][s]["basic"]
        ref_frame = idx["scenes"][s]["ref_frame"]
        for t in idx["scenes"][s]["frames"]:
            assert t != ref_frame
            fd = os.path.join(a.root, s, f"{t:06d}")
            if os.path.isfile(os.path.join(fd, "_logits", "DONE")) and not a.overwrite:
                print(f"{s:20s} t={t:4d} skip (DONE)", flush=True)
                continue
            CAP.update(dec=[], path=None, bb_path={}, r1=[], r2=[])
            sc = D.frame_scene(s, t, basic)
            sink, config = D.decode_frame(sc, t, restrict=True, preset=a.preset)
            config.update(fold=pf["fold"], ckpt=pf["checkpoint"])
            n = D.write_frame(a.root, sc, t, sink, config)
            t1 = time.time()
            cam_of = {os.path.realpath(p): c for c, p in sc.image_paths.items()}
            targets = [c for c in config["decoded_cams"] if c != sc.ref_cam]
            r2_cams = [cam_of[os.path.realpath(r["path"])] for r in CAP["r2"]]
            assert r2_cams == targets, (r2_cams, targets)
            assert [cam_of[os.path.realpath(p)] for p in CAP["r1"]] == targets
            ld = os.path.join(fd, "_logits")
            meta = dict(scene=s, t=t, objects=[int(o) for o in sc.objects], hw=list(sc.hw), eval_cams=config["eval_cams"],
                        env=env, views={})
            for rec, cam in zip(CAP["r2"], r2_cams):
                if cam not in config["eval_cams"]:
                    continue
                rec["cam"] = cam
                chk = save_view(ld, rec, sc.objects, sc.hw)
                chk["objects"] = check_view(a.root, a.ref_root, s, t, cam, sink.masks.get(FINAL, {}).get(cam, {}),
                                            sc.objects, sc.hw)
                chk["npz_bytes"] = os.path.getsize(os.path.join(ld, f"{cam}.npz"))
                chk["mm_npz_bytes"] = os.path.getsize(os.path.join(ld, f"{cam}_mm.npz"))
                meta["views"][cam] = chk
            assert set(meta["views"]) == set(config["eval_cams"])
            meta["dump_seconds"] = time.time() - t1
            with open(os.path.join(ld, "meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
            open(os.path.join(ld, "DONE"), "w").close()
            bad = {c: {o: r for o, r in v["objects"].items()
                       if r["rethr_vs_ref_px"] or r["rethr_vs_mem_px"] or r["mem_vs_ref_px"] or not r["png_bytes_equal"]}
                   for c, v in meta["views"].items()}
            flags = {c: (v["recon_low_res_equal"], v["recon_ious_equal"], v["n_fallback"], v["fp16_inexact_low_res"])
                     for c, v in meta["views"].items()}
            print(f"{s:20s} t={t:4d} objs {len(sc.objects):2d} masks {n:3d} dec {config['seconds']:5.1f}s dump {meta['dump_seconds']:5.1f}s "
                  f"bad={ {c: len(b) for c, b in bad.items()} } (recon_lr, recon_iou, n_fb, f16_inexact)={flags}", flush=True)
            if any(bad.values()):
                print("  MISMATCH", json.dumps(bad)[:2000], flush=True)


if __name__ == "__main__":
    main()
