#!/usr/bin/env python3
"""S1-E14 zoom-xcl (scratch, git-ignored): the S1-E10 crop-zoom (--zoom 2 --zoom-rope remap, band-clipped paste)
COMPOSED with the adopted S1-E11 exclusivity (--xcl out), on DEV-A only (never the keyframe, never DEV-B).

run.decode_scene refuses --zoom with --xcl, so this decodes the registered zoom preset `geoset_rep_z2r` through
methods.geomem.dev_decode.decode_frame UNCHANGED (restricted cameras, own-fold checkpoint) and hooks two module
attributes (no tracked file is touched):
  * GeoMemStar.decode_view: the top-level round-2 call (the only one with `chunk`) leaves its logits_1008 in CAP; the
    top-level calls made inside the zoom pass (B = 1 crops) are appended in call order.
  * geomem.zoom.refine_camera (decode_scene looks it up as zoom_mod.refine_camera at call time): for an EVAL camera it
    calls the original refine_camera on the round-2 masks and then builds the zoomed output's logits:
        L   = sam3_io.logits_to_original(round-2 logits_1008, hw)[:, 0]          (the adopted xcl path's tensor)
        L_z = L, except for every ACCEPTED crop inside band_r(round-2 mask) & window, where it is the crop's logits
              F.interpolate(crop logits_1008, window size, bilinear)               (the tensor refine_camera thresholds)
    so (L_z > 0) == the zoomed masks exactly (asserted), and applies run.exclusive_masks(L_z, 0.5): zoom first, then the
    adopted exclusivity on the zoomed output.  Non-eval cameras (decoded only as k=3 memory candidates, never written)
    skip the crop pass; round 2 of a camera never reads another camera's round-2 output, so eval cams are unaffected.
Arms written (one GPU decode -> four dev_decode-layout roots under --out):
    r2   geoset_rep       L > 0                      (no zoom, no xcl)
    z    geoset_rep_z2r   zoomed masks = L_z > 0     (E10 C2 alone)
    zx   ZOOM -> XCL      exclusive_masks(L_z, 0.5)  (the composition; primary)
    zxb  diagnostic       pos = L_z > 0, winner = argmax of the ROUND-2 L among the claimants, same 0.5 cap
and   x  = exclusive_masks(L, 0.5) is compared in-job with the adopted DEV-A PNGs (_e13_dev; must be 0 px different),
round-1 masks / round1_conf with _e13_dev too.
    cd stage1 && scripts/docker_run.sh scsam3 python geomem/runs/_e14_zoom-xcl/code/zx_decode.py --scenes Dog
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, S1)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from geomem import devset, sam3_io  # noqa: E402
from geomem import zoom as Z  # noqa: E402
from geomem.wrapper import GeoMemStar  # noqa: E402
from methods import common  # noqa: E402
from methods.geomem import dev_decode as D  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem.stage1 import FINAL, ROUND1, check_preflight  # noqa: E402
from mvseg2d.io import read_pred_mask, write_mask  # noqa: E402

PRESET = "geoset_rep_z2r"          # arm geoset_pred_ft, gate reproj, zoom 2, zoom_rope remap (xcl none: decode_scene's rule)
ARMS = ("r2", "z", "zx", "zxb")
CAP = dict(depth=0, r2=None, in_refine=False, crops=[], eval=(), stash={})


def claimant_exclusive(L, pos, tau_keep=R.XCL_TAU_KEEP):
    """exclusive_masks with an external claim set: pos [B,h,w] bool; winner = argmax of L among the claimants of a pixel;
    same nested-mask cap.  claimant_exclusive(L, L > 0) == run.exclusive_masks(L) (asserted in-job)."""
    B = L.shape[0]
    w = torch.where(pos, L, torch.full_like(L, -float("inf"))).argmax(0)
    win = w[None] == torch.arange(B, device=L.device)[:, None, None]
    del w
    npos = pos.flatten(1).sum(1)
    lost = (pos & ~win).flatten(1).sum(1)
    exempt = (npos > 0) & (lost.float() >= tau_keep * npos.float())
    keep = pos & (win | exempt[:, None, None])
    removed = pos & ~keep
    counts = dict(contested_px=int((pos.sum(0) >= 2).sum()), removed_entries=int(removed.sum()),
                  masks_changed=int(removed.flatten(1).any(1).sum()), masks_exempt=int(exempt.sum()),
                  masks_emptied=int(((npos > 0) & (keep.flatten(1).sum(1) == 0)).sum()), masks_present=int((npos > 0).sum()))
    return keep, counts


def install():
    orig_dv = GeoMemStar.decode_view

    def decode_view(self, backbone_out, B, memory, p_v, valid_v, chunk=None, **kw):
        top = CAP["depth"] == 0
        CAP["depth"] += 1
        try:
            out = orig_dv(self, backbone_out, B, memory, p_v, valid_v, chunk=chunk, **kw)
        finally:
            CAP["depth"] -= 1
        if top:
            if chunk is not None:                                   # round 2 (decode_scene passes chunk only there)
                assert not CAP["in_refine"]
                CAP["r2"] = out["logits_1008"]
            elif CAP["in_refine"]:                                  # a crop decode of the zoom pass
                assert B == 1, B
                CAP["crops"].append(out["logits_1008"])
        return out
    GeoMemStar.decode_view = decode_view

    orig_refine = Z.refine_camera

    def refine_camera(wrapper, model, mem, image_path, cache, cam, ref_cam, hw, pred, pres_full, iou_full, objects, **kw):
        lg, CAP["r2"] = CAP["r2"], None
        assert lg is not None and lg.shape[0] == len(objects), (cam, None if lg is None else lg.shape)
        L = sam3_io.logits_to_original(lg, hw)[:, 0]
        del lg
        assert np.array_equal((L > 0).cpu().numpy(), pred), f"captured round-2 logits != decode_scene's masks ({cam})"
        if cam not in CAP["eval"]:
            del L
            return pred, {}, Counter()
        pred0 = pred.copy()
        CAP["crops"], CAP["in_refine"] = [], True
        try:
            pz, recs, n = orig_refine(wrapper, model, mem, image_path, cache, cam, ref_cam, hw, pred, pres_full, iou_full,
                                      objects, **kw)
        finally:
            CAP["in_refine"] = False
        crops, CAP["crops"] = CAP["crops"], []
        t1 = time.perf_counter()
        ran = [i for i, o in enumerate(objects) if str(int(o)) in recs and recs[str(int(o))]["window"] is not None]
        assert len(ran) == len(crops) == n["run"], (len(ran), len(crops), n["run"])
        r = Z.band_px(hw)
        Lz = L.clone()
        for i, lc8 in zip(ran, crops):
            rec = recs[str(int(objects[i]))]
            if rec["verdict"] != "accepted":
                continue
            y0, y1, x0, x1 = rec["window"]
            lc = F.interpolate(lc8.float(), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0, 0]
            band = torch.from_numpy(Z.band_mask(pred0[i], r)[y0:y1, x0:x1]).to(L.device)
            Lz[i, y0:y1, x0:x1] = torch.where(band, lc, Lz[i, y0:y1, x0:x1])
            del lc, band
        del crops
        posz = Lz > 0
        assert np.array_equal(posz.cpu().numpy(), pz), f"(L_z > 0) != zoomed masks ({cam})"
        out_band = 0
        for i in range(len(objects)):                                # E10 Z5: zoom changes only inside band_r(round-2)
            ch = pz[i] != pred0[i]
            if ch.any():
                out_band += int((ch & ~Z.band_mask(pred0[i], r)).sum())
        keep_x, cx, rx = R.exclusive_masks(L, R.XCL_TAU_KEEP)
        keep_c, _ = claimant_exclusive(L, L > 0)
        assert torch.equal(keep_c, keep_x), f"claimant_exclusive(L, L > 0) != exclusive_masks(L) ({cam})"
        del keep_c
        keep_zx, czx, rzx = R.exclusive_masks(Lz, R.XCL_TAU_KEEP)
        keep_zxb, czxb = claimant_exclusive(L, posz)
        del posz, Lz, L
        st = dict(r2=pred0, z=pz.copy(), x=keep_x.cpu().numpy(), zx=keep_zx.cpu().numpy(), zxb=keep_zxb.cpu().numpy(),
                  zoom_recs=recs, zoom_n=dict(n), xcl_x=cx, xcl_zx=czx, xcl_zxb=czxb,
                  xcl_recs_zx={str(int(o)): rzx[i] for i, o in enumerate(objects)},
                  xcl_recs_x={str(int(o)): rx[i] for i, o in enumerate(objects)},
                  zoom_out_of_band_px=out_band, post_seconds=time.perf_counter() - t1)
        del keep_x, keep_zx, keep_zxb
        assert not (st["zx"] & ~st["z"]).any() and not (st["zxb"] & ~st["z"]).any()
        CAP["stash"][cam] = st
        return pz, recs, n
    Z.refine_camera = refine_camera


def read_png(root, scene, t, cam, o, hw):
    p = os.path.join(root, scene, f"{t:06d}", cam, f"{int(o)}.png")
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    return read_pred_mask(p)


def write_arm(root, sc, t, cams, masks_of, config, extra_files):
    d = os.path.join(root, sc.name, f"{t:06d}")
    if os.path.isdir(d):
        import shutil
        shutil.rmtree(d)
    for o in sc.objects:                                            # the reference camera: its GT (as dev_decode)
        write_mask(os.path.join(d, sc.ref_cam, f"{o}.png"), sc.ref_labels == o)
    n = 0
    for cam in cams:
        m = masks_of(cam)
        for i, o in enumerate(sc.objects):
            if m[i].any():
                write_mask(os.path.join(d, cam, f"{int(o)}.png"), m[i])
                n += 1
    for name, obj in list(extra_files.items()) + [("config.json", config)]:
        with open(os.path.join(d, name), "w") as f:
            json.dump(obj, f, indent=1)
    open(os.path.join(d, "DONE"), "w").close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--out", default=os.path.join(S1, "geomem", "runs", "_e14_zoom-xcl"))
    ap.add_argument("--ref-root", default=os.path.join(S1, "geomem", "runs", "_e13_dev"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    split = "A"                                                      # DEV-B reserved; TEST never decoded
    all_sc = common.all_scenes(D.DATA)
    scenes = all_sc if a.scenes == "all" else a.scenes.split(",")
    idx = devset.index(D.DATA, all_sc, split)
    roots = {arm: os.path.join(a.out, f"{arm}_A") for arm in ARMS}
    for arm, rt in roots.items():
        os.makedirs(rt, exist_ok=True)
        with open(os.path.join(rt, "index.json"), "w") as f:
            json.dump(dict(preset=f"e14_zoom-xcl:{arm}", **idx), f, indent=1)
    from methods import sam3_single
    R.decode_scene.model = sam3_single.load_model()
    R.decode_scene.init_state = R.snapshot_init(R.decode_scene.model)
    install()
    print("ENV", json.dumps(dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), preset=PRESET)), flush=True)
    for s in scenes:
        pf = check_preflight(PRESET, s)
        S = idx["scenes"][s]
        for t in S["frames"]:
            assert t != S["ref_frame"]
            if os.path.isfile(os.path.join(roots["zx"], s, f"{t:06d}", "E14DONE")) and not a.overwrite:
                print(f"{s:20s} t={t:4d} skip (DONE)", flush=True)
                continue
            CAP.update(depth=0, r2=None, in_refine=False, crops=[], stash={}, eval=tuple(S["eval_cams"]))
            torch.cuda.reset_peak_memory_stats()
            sc = D.frame_scene(s, t, S["basic"])
            sink, config = D.decode_frame(sc, t, restrict=True, preset=PRESET)
            config.update(fold=pf["fold"], ckpt=pf["checkpoint"], peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            evals = config["eval_cams"]
            assert set(CAP["stash"]) == set(evals), (sorted(CAP["stash"]), evals)
            st = CAP["stash"]
            # ---- checks against the adopted DEV-A masks (_e13_dev) and the decode itself
            chk = dict(x_vs_adopted_px=0, x_vs_adopted_objs=0, z_vs_sink_px=0, r1_vs_adopted_px=0, r1_missing=0, r1_extra=0,
                       zoom_out_of_band_px=0)
            for cam in evals:
                fin = sink.masks.get(FINAL, {}).get(cam, {})
                for i, o in enumerate(sc.objects):
                    ref_m = read_png(a.ref_root, s, t, cam, o, sc.hw)
                    dpx = int((st[cam]["x"][i] != ref_m).sum())
                    chk["x_vs_adopted_px"] += dpx
                    chk["x_vs_adopted_objs"] += int(dpx > 0)
                    zm = fin.get(int(o), np.zeros(sc.hw, bool))
                    chk["z_vs_sink_px"] += int((zm != st[cam]["z"][i]).sum())
                chk["zoom_out_of_band_px"] += st[cam]["zoom_out_of_band_px"]
            r1 = sink.masks.get(ROUND1, {})
            for cam, objs in r1.items():
                d_ref = os.path.join(a.ref_root, s, f"{t:06d}", "_r1", cam)
                on_disk = {int(f[:-4]) for f in os.listdir(d_ref)} if os.path.isdir(d_ref) else set()
                chk["r1_missing"] += len(on_disk - set(objs))
                chk["r1_extra"] += len(set(objs) - on_disk)
                for o, m in objs.items():
                    chk["r1_vs_adopted_px"] += int((m != _r1(a.ref_root, s, t, cam, o, sc.hw)).sum())
            rc_ref = json.load(open(os.path.join(a.ref_root, s, f"{t:06d}", "round1_conf.json")))
            chk["round1_conf_equal"] = rc_ref == json.loads(json.dumps(sink.files.get((FINAL, "round1_conf.json"))))
            # ---- write the four arms
            tw = time.time()
            zoom_n = Counter()
            for cam in evals:
                zoom_n.update(st[cam]["zoom_n"])
            cfg = dict(config, e14_arm=None, zoom_counts_eval={k: float(v) for k, v in zoom_n.items()},
                       post_seconds=sum(st[c]["post_seconds"] for c in evals), checks=chk,
                       xcl_counts={arm: {c: st[c][f"xcl_{arm}"] for c in evals} for arm in ("x", "zx", "zxb")})
            nw = {}
            for arm in ARMS:
                extra = {}
                if arm in ("zx", "z"):
                    extra["zoom_conf.json"] = {c: st[c]["zoom_recs"] for c in evals}
                if arm == "zx":
                    extra["xcl_conf.json"] = {c: st[c]["xcl_recs_zx"] for c in evals}
                    extra["round1_conf.json"] = sink.files.get((FINAL, "round1_conf.json"))
                nw[arm] = write_arm(roots[arm], sc, t, evals, lambda c, arm=arm: st[c][arm], dict(cfg, e14_arm=arm), extra)
            for cam, objs in r1.items():                             # round-1 masks once, in the primary root
                for o, m in objs.items():
                    write_mask(os.path.join(roots["zx"], s, f"{t:06d}", "_r1", cam, f"{int(o)}.png"), m)
            open(os.path.join(roots["zx"], s, f"{t:06d}", "E14DONE"), "w").close()     # all four arms + _r1 written
            ok = (chk["x_vs_adopted_px"] == 0 and chk["z_vs_sink_px"] == 0 and chk["r1_vs_adopted_px"] == 0
                  and chk["r1_missing"] == 0 and chk["r1_extra"] == 0 and chk["round1_conf_equal"] and chk["zoom_out_of_band_px"] == 0)
            print(f"{s:20s} t={t:4d} objs {len(sc.objects):2d} dec {config['seconds']:6.1f}s zoom {zoom_n['seconds']:6.1f}s "
                  f"crops {zoom_n['run']:3d} acc {zoom_n['accepted']:3d} post {cfg['post_seconds']:4.1f}s write {time.time() - tw:5.1f}s "
                  f"peak {config['peak_gpu_gb']:.1f}G masks {nw} checks {'OK' if ok else 'MISMATCH ' + json.dumps(chk)}", flush=True)


def _r1(root, scene, t, cam, o, hw):
    p = os.path.join(root, scene, f"{t:06d}", "_r1", cam, f"{int(o)}.png")
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    return read_pred_mask(p)


if __name__ == "__main__":
    main()
