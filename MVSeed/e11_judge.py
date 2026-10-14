#!/usr/bin/env python3
"""S1-E11 judge (docs/stage1-E11-prereg.md sections 3-7), keyed by RESULT FOLDER (E10 structure).

  --stage 1   keyframe scores (MVSeed/runs/e11/kf_<folder>.json): X1, X2 (gated) and D (diagnostic) against C0,
              scene-cluster bootstrap on J, F, J&F (mvseg2d.paired, 10 000, seed 0), gate G1/G2 (X1, X2), splits
              (good/failing, size, kind, twin, contested/non-contested, exempt-touched, spill-14), the GT argmax-accuracy
              diagnostic, predictions P1-P6, P9-P10, watch values Z1 (counters == PNG-derived), Z1m, Z2, Z5x, Z7.
              Run with stage1/.venv (numpy, cv2):  stage1/.venv/bin/python MVSeed/e11_judge.py --stage 1
  --stage 2   MUVOD basic/complete of the passing arms' in-process headline runs (Data/MVSeg/jf_e11.json) against the
              current headline, the E8 rule, selection (tie -> X1), P7-P8, Z3/Z3b.  Also under the venv.
Both write MVSeed/runs/e11/e11_judge.json (stage 2 merges into stage 1's file).
"""
import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(REPO, "stage1")
OUT_DIR = os.path.join(REPO, "MVSeed", "runs", "e11")
OUT = os.path.join(OUT_DIR, "e11_judge.json")
D0 = "geoset_pred_ft_rep_e8"
D0_R1 = "geoset_pred_ft_r1_e8"
ARMS = [("X1", "geoset_pred_ft_rep_x_e11", "geoset_rep_x", "geoset_pred_ft_r1_x_e11"),
        ("X2", "geoset_pred_ft_rep_xm_e11", "geoset_rep_xm", "geoset_pred_ft_r1_xm_e11"),
        ("D", "geoset_pred_ft_rep_xu_e11", None, "geoset_pred_ft_r1_xu_e11")]
GATED = ("X1", "X2")
HEADLINE = "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs"
DEFENCE, TIE = -1.0, 0.05
SAFETY = -0.01
ORACLE = 0.0054                 # registered ceiling (P1): an arm above it is a bug
SCENES = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "CBABasketball", "Carpark", "CoffeeMartini",
          "Dog", "Fencing", "FlameSteak", "Frog", "MATF", "MartialArts", "Painter", "PoznanStreet", "Welder")


def _load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _result(scene, folder):
    return os.path.join(S1, "Data", scene, "Result", folder)


def _mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


def _pngs(d, skip=()):
    out = {}
    if not os.path.isdir(d):
        return out
    for cam in sorted(os.listdir(d)):
        if cam in skip or not os.path.isdir(os.path.join(d, cam)):
            continue
        for f in sorted(os.listdir(os.path.join(d, cam))):
            if f.endswith(".png"):
                out[(cam, f)] = os.path.join(d, cam, f)
    return out


def _bytes_equal(a, b):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def _kind(r):
    """Failure type of a seed (MVSeed/refloss/seedcat.py rules)."""
    if r["seed_px"] == 0 or (r["recall"] or 0) < 0.05:
        return "miss" if r["seed_px"] == 0 or r["gt_px"] < 50 else "wrong-object"
    if (r["precision"] or 0) < 0.3:
        return "wrong-object"
    if r["frac_other"] >= 0.2:
        return "spill-onto-other" + ("(twin)" if r["twin"] else "")
    if r["recall"] < 0.85 and r["precision"] >= 0.85:
        return "under-cover" + ("(beyond c_ini)" if r["vis_ratio"] > 1.2 else "")
    return "boundary/mixed"


def z2_round1(r1_folder, rep_folder):
    out = dict(scenes_ok=0, png_same=0, png_diff=0, key_mismatch=0, conf_same=0, missing=[])
    for sc in SCENES:
        a, b = _pngs(_result(sc, r1_folder)), _pngs(_result(sc, D0_R1))
        if not a:
            out["missing"].append(sc); continue
        ca, cb = os.path.join(_result(sc, rep_folder), "round1_conf.json"), os.path.join(_result(sc, D0), "round1_conf.json")
        if os.path.isfile(ca) and os.path.isfile(cb) and _bytes_equal(ca, cb):
            out["conf_same"] += 1
        if set(a) != set(b):
            out["key_mismatch"] += 1; continue
        same = sum(_bytes_equal(a[k], b[k]) for k in a)
        out["png_same"] += same; out["png_diff"] += len(a) - same
        if same == len(a):
            out["scenes_ok"] += 1
    out["ok"] = out["scenes_ok"] == 17 and out["conf_same"] == 17 and out["png_diff"] == 0 and out["key_mismatch"] == 0 and not out["missing"]
    return out


def z1_z5x(folder, ref_cams, capped):
    """Z1 (counters == PNG-derived) and Z5x (exact containment) for X1 / D, plus the GT argmax-accuracy diagnostic and
    the exempt-touched / contested pair sets.  Returns (z1, z5x, diag, sets)."""
    import cv2
    import numpy as np
    sys.path.insert(0, S1)
    from mvseg2d.io import load_labelmap
    z1 = dict(scenes=0, counters_ok=0, contested_px=0, removed_entries=0, masks_changed=0, masks_exempt=0, masks_emptied=0)
    z5 = dict(scenes_checked=0, added_px=0, outside_contested_px=0, double_survivor_px=0, key_mismatch=[], missing=[])
    diag = dict(contested_px_scored=0, gt_owner_contender_px=0, argmax_to_gt_px=0, argmax_to_smaller_px=0)
    contested_pairs, exempt_pairs = set(), set()
    for sc in SCENES:
        d = _result(sc, folder)
        if not os.path.isdir(d):
            z5["missing"].append(sc); continue
        z1["scenes"] += 1; z5["scenes_checked"] += 1
        cfg = _load_json(os.path.join(d, "RUN.json")).get("config", {})
        recs = _load_json(os.path.join(d, "xcl_conf.json"))
        meta = _load_json(os.path.join(S1, "Data", sc, "meta.json"))
        mask_cams = set(meta["mask_cams"]) - {ref_cams[sc]}
        got = dict(contested_px=0, removed_entries=0, masks_changed=0, masks_exempt=0, masks_emptied=0)
        base_all, arm_all = _pngs(_result(sc, D0), skip=(ref_cams[sc],)), _pngs(d, skip=(ref_cams[sc],))
        cams = sorted({k[0] for k in base_all} | {k[0] for k in arm_all})
        for cam in cams:
            objs = sorted({k[1] for k in base_all if k[0] == cam} | {k[1] for k in arm_all if k[0] == cam})
            if not objs:
                continue
            B0 = np.stack([(cv2.imread(base_all[(cam, f)], 0) > 127) if (cam, f) in base_all else None for f in objs if (cam, f) in base_all]) \
                if any((cam, f) in base_all for f in objs) else None
            if B0 is None:
                z5["key_mismatch"].append((sc, cam)); continue
            objs_b = [f for f in objs if (cam, f) in base_all]
            if set(objs) != set(objs_b):
                z5["key_mismatch"].append((sc, cam)); continue          # an arm mask without a C0 mask
            shape = B0.shape[1:]
            A = np.stack([(cv2.imread(arm_all[(cam, f)], 0) > 127) if (cam, f) in arm_all else np.zeros(shape, bool) for f in objs_b])
            contested = B0.sum(0) >= 2
            got["contested_px"] += int(contested.sum())
            removed = B0 & ~A
            got["removed_entries"] += int(removed.sum())
            got["masks_changed"] += int(removed.reshape(len(objs_b), -1).any(1).sum())
            exempt = np.array([bool(recs.get(cam, {}).get(f[:-4], {}).get("exempt", False)) for f in objs_b])
            got["masks_exempt"] += int(exempt.sum())
            got["masks_emptied"] += int(((B0.reshape(len(objs_b), -1).any(1)) & ~(A.reshape(len(objs_b), -1).any(1))).sum())
            z5["added_px"] += int((A & ~B0).sum())
            z5["outside_contested_px"] += int(((A != B0) & ~contested[None]).sum())
            if (~exempt).any():
                z5["double_survivor_px"] += int((A[~exempt].sum(0) > 1).sum())
            if cam in mask_cams:
                gt = load_labelmap(os.path.join(S1, "Data", sc, "masks", f"{cam}.png"))
                ids = np.array([int(f[:-4]) for f in objs_b])
                for b, f in enumerate(objs_b):
                    if contested.any() and (B0[b] & contested).any():
                        contested_pairs.add((sc, cam, int(f[:-4])))
                    if exempt[b] or (removed[b].any() and exempt.any()):
                        pass
                    if exempt[b]:
                        exempt_pairs.add((sc, cam, int(f[:-4])))
                # argmax accuracy on scored views: at each contested pixel the survivor (non-exempt) vs the GT owner
                surv = A.copy(); surv[exempt] = False
                one = surv.sum(0) == 1
                px = contested & one
                diag["contested_px_scored"] += int(contested.sum())
                if px.any():
                    winner = ids[surv[:, px].argmax(0)]
                    gtv = gt[px]
                    owner_is_contender = (B0[:, px] & (ids[:, None] == gtv[None])).any(0)
                    diag["gt_owner_contender_px"] += int(owner_is_contender.sum())
                    diag["argmax_to_gt_px"] += int((winner == gtv).sum())
                    sizes = B0.reshape(len(objs_b), -1).sum(1)
                    cont_sizes = np.where(B0[:, px], sizes[:, None], np.iinfo(np.int64).max)
                    smaller = ids[cont_sizes.argmin(0)]
                    diag["argmax_to_smaller_px"] += int((winner == smaller).sum())
        ok = all(int(cfg.get(f"n_xcl_{k}", -1)) == v for k, v in got.items())
        z1["counters_ok"] += int(ok)
        for k, v in got.items():
            z1[k] += v
    z1["ok"] = z1["scenes"] == 17 and z1["counters_ok"] == 17
    z5["ok"] = z5["scenes_checked"] == 17 and z5["added_px"] == 0 and z5["outside_contested_px"] == 0 and not z5["missing"] and \
        (z5["double_survivor_px"] == 0) and (not z5["key_mismatch"] or not capped)
    diag["argmax_accuracy"] = diag["argmax_to_gt_px"] / diag["gt_owner_contender_px"] if diag["gt_owner_contender_px"] else None
    diag["smaller_fraction"] = diag["argmax_to_smaller_px"] / max(diag["argmax_to_gt_px"] + (diag["gt_owner_contender_px"] - diag["argmax_to_gt_px"]), 1)
    return z1, z5, diag, dict(contested=contested_pairs, exempt=exempt_pairs)


def stage1():
    sys.path.insert(0, S1)
    from mvseg2d.paired import paired_scene_bootstrap, scene_means
    from mvseg2d.report import flatten

    def rows_of(name):
        rs = flatten(_load_json(os.path.join(OUT_DIR, f"kf_{name}.json")))
        for r in rs:
            r["JF"] = 0.5 * (r["J"] + r["F"])
        return rs

    def hallu(name):
        return sum(s["hallucinations"] for s in _load_json(os.path.join(OUT_DIR, f"kf_{name}.json"))["scenes"].values() if s["status"] == "ok")

    def summary(rs):
        m = {k: scene_means(rs, k) for k in ("J", "F", "JF")}
        return {k: statistics.mean(v.values()) for k, v in m.items()}

    key = lambda r: (r["scene"], r["cam"], r["obj"])
    r0 = rows_of(D0); k0 = {key(r): r for r in r0}
    good = {k for k, r in k0.items() if r["JF"] >= 0.8}
    ref_cams = {sc: _load_json(os.path.join(S1, "Data", sc, "meta.json"))["ref_cam"] for sc in SCENES}
    sf0 = {(r["scene"], r["cam"], r["obj"]): r for r in _load_json(os.path.join(REPO, "MVSeed", "runs", "seedfail.json"))["rows"]}
    spill14 = {k for k, r in sf0.items() if r["JF"] < 0.8 and r.get("kind", "").startswith("spill")}
    under22 = {k for k, r in sf0.items() if r["JF"] < 0.8 and r.get("kind", "").startswith("under")}
    res = {"C0": dict(name=D0, n=len(r0), **summary(r0), hallucinations=hallu(D0), fails=sum(r["J"] < 0.5 for r in r0), good_seeds=len(good))}
    passing = []
    sets = {}
    for rid, folder, tag, r1_folder in ARMS:
        p = os.path.join(OUT_DIR, f"kf_{folder}.json")
        if not os.path.isfile(p):
            print(f"{rid}: {p} missing (not decoded / scored yet)"); continue
        rs = rows_of(folder); kk = {key(r): r for r in rs}
        if set(kk) != set(k0):
            st_ = {sc: v["status"] for sc, v in _load_json(p)["scenes"].items() if v["status"] != "ok"}
            raise SystemExit(f"{rid}: pair set differs from C0 ({len(kk)} vs {len(k0)}); scenes not ok: {st_} -- fix the decode first")
        st = {k: paired_scene_bootstrap(r0, rs, key=k) for k in ("J", "F", "JF")}
        worst = min(st["JF"]["per_scene"].items(), key=lambda x: x[1])
        gate = dict(G1_mean_gt_0=st["JF"]["delta"] > 0, G1_ci_lower_gt_0=st["JF"]["ci"][0] > 0,
                    G2_no_scene_below=worst[1] >= SAFETY, worst_scene=worst[0], worst_delta=worst[1])
        gate["pass"] = gate["G1_mean_gt_0"] and gate["G1_ci_lower_gt_0"] and gate["G2_no_scene_below"]
        d = {k: kk[k]["JF"] - k0[k]["JF"] for k in k0}

        def split(sel):
            sel = list(sel)
            return dict(n=len(sel), dJF=_mean(d[k] for k in sel), dJ=_mean(kk[k]["J"] - k0[k]["J"] for k in sel),
                        dF=_mean(kk[k]["F"] - k0[k]["F"] for k in sel),
                        up=sum(d[k] > 0.01 for k in sel), down=sum(d[k] < -0.01 for k in sel), nonzero=sum(d[k] != 0 for k in sel))
        z1 = z5 = diag = None
        if rid in ("X1", "D"):
            z1, z5, diag, sets[rid] = z1_z5x(folder, ref_cams, capped=(rid == "X1"))
        cont = sets.get("X1", sets.get("D", {})).get("contested", set())
        exem = sets.get(rid, {}).get("exempt", set())
        splits = dict(good=split(good), fail=split(k for k in k0 if k not in good),
                      contested=split(k for k in k0 if k in cont) if cont else None,
                      non_contested=split(k for k in k0 if k not in cont) if cont else None,
                      exempt_touched=split(k for k in k0 if k in exem) if exem else None,
                      spill14=split(spill14 & set(k0)), under22=split(under22 & set(k0)),
                      twin=split(k for k in k0 if sf0.get(k, {}).get("twin")),
                      **{f"otype_{t}": split(k for k, r in k0.items() if r["type"] == t) for t in ("dynamic", "static", "env")})
        by_size = {b: split(k for k, r in k0.items() if r["size_bin"] == b) for b in ("<2000", "2000-10000", ">=10000")}
        sfp = os.path.join(OUT_DIR, f"sf_{folder}.json")
        kinds = None
        if os.path.isfile(sfp):
            sfa = {(r["scene"], r["cam"], r["obj"]): r for r in _load_json(sfp)}
            kinds = dict(C0={}, arm={})
            for k in k0:
                if k in sf0 and sf0[k]["JF"] < 0.8:
                    kd = sf0[k].get("kind") or _kind(sf0[k]); kinds["C0"][kd] = kinds["C0"].get(kd, 0) + 1
                if k in sfa and sfa[k]["JF"] < 0.8:
                    kd = _kind(sfa[k]); kinds["arm"][kd] = kinds["arm"].get(kd, 0) + 1
        z2 = z2_round1(r1_folder, folder)
        z7 = dict(seconds=0.0, gpu_mb_max=0.0, gpu_mb_max_C0=0.0, seconds_C0=0.0)
        xmem = dict(contested_cells=0, clamped_cells=0, masks_exempt=0, eligible=0, scenes=0)
        for sc in SCENES:
            rp, rp0 = os.path.join(_result(sc, folder), "RUN.json"), os.path.join(_result(sc, D0), "RUN.json")
            if os.path.isfile(rp) and os.path.isfile(rp0):
                run, run0 = _load_json(rp), _load_json(rp0)
                z7["seconds"] += float(run.get("seconds", 0)); z7["seconds_C0"] += float(run0.get("seconds", 0))
                z7["gpu_mb_max"] = max(z7["gpu_mb_max"], float(run.get("gpu_mb", 0) or 0)); z7["gpu_mb_max_C0"] = max(z7["gpu_mb_max_C0"], float(run0.get("gpu_mb", 0) or 0))
                c = run.get("config", {})
                if rid == "X2":
                    xmem["scenes"] += 1
                    for k2 in ("contested_cells", "clamped_cells", "masks_exempt", "eligible"):
                        xmem[k2] += int(c.get(f"n_xmem_{k2}", 0))
        z7["ok"] = z7["seconds"] <= 45 * 60 and z7["gpu_mb_max"] <= z7["gpu_mb_max_C0"] + 1000 and z7["seconds"] <= 1.15 * z7["seconds_C0"] + 60
        if rid == "X2":
            xmem["ok"] = xmem["scenes"] == 17 and xmem["contested_cells"] > 0 and xmem["clamped_cells"] > 0
        r = dict(name=folder, tag=tag, **summary(rs), hallucinations=hallu(folder), fails=sum(x["J"] < 0.5 for x in rs),
                 paired={k: dict(delta=v["delta"], ci=list(v["ci"]), per_scene=v["per_scene"]) for k, v in st.items()},
                 gate=gate, splits=splits, by_size=by_size, failure_kinds=kinds, Z1=z1, Z1m=xmem if rid == "X2" else None, Z2=z2, Z5x=z5, Z7=z7, diag=diag)
        res[rid] = r
        if rid in GATED and gate["pass"]:
            passing.append(tag)
        print(f"{rid} {folder}: J {r['J']:.4f} F {r['F']:.4f} J&F {r['JF']:.4f} (C0 {res['C0']['JF']:.4f})  "
              f"ΔJ&F {st['JF']['delta']:+.4f} CI [{st['JF']['ci'][0]:+.4f}, {st['JF']['ci'][1]:+.4f}]  ΔJ {st['J']['delta']:+.4f} [{st['J']['ci'][0]:+.4f}, {st['J']['ci'][1]:+.4f}]"
              f"  ΔF {st['F']['delta']:+.4f} [{st['F']['ci'][0]:+.4f}, {st['F']['ci'][1]:+.4f}]  worst {worst[0]} {worst[1]:+.4f}  fails {res['C0']['fails']}→{r['fails']}  halluc {res['C0']['hallucinations']}→{r['hallucinations']}"
              f"  gate {'PASS' if gate['pass'] else 'FAIL'}" + ("" if rid in GATED else " (diagnostic, not gated)"))
        sp = splits
        print(f"   good {sp['good']['dJF']:+.4f} (↑{sp['good']['up']} ↓{sp['good']['down']})  fail {sp['fail']['dJF']:+.4f}"
              + (f"  | contested {sp['contested']['dJF']:+.4f} (n={sp['contested']['n']}, nonzero {sp['contested']['nonzero']})  non-contested {sp['non_contested']['dJF']:+.4f} (n={sp['non_contested']['n']}, nonzero {sp['non_contested']['nonzero']})" if sp['contested'] else "")
              + (f"  | exempt-touched {sp['exempt_touched']['dJF']:+.4f} (n={sp['exempt_touched']['n']})" if sp['exempt_touched'] else "")
              + f"  | spill14 {sp['spill14']['dJF']:+.4f} (↑{sp['spill14']['up']} ↓{sp['spill14']['down']})  under22 {sp['under22']['dJF']:+.4f}  twin {sp['twin']['dJF']:+.4f}"
              + f"  | size {{{', '.join(f'{b}: {v['dJF']:+.4f}' for b, v in by_size.items())}}}")
        print(f"   kinds {kinds}\n   diag {diag}\n   Z1 {z1}\n   Z1m {r['Z1m']}\n   Z2 {z2}\n   Z5x {z5}\n   Z7 {z7}")
    pr = {}
    if "X1" in res:
        x = res["X1"]; dj = x["paired"]["JF"]["delta"]
        pr["P1"] = dict(delta=dj, ok=-0.002 <= dj <= 0.004 and x["paired"]["J"]["delta"] > x["paired"]["F"]["delta"] and abs(x["paired"]["F"]["delta"]) <= 0.001,
                        above_ceiling=dj > ORACLE, argmax_accuracy=x["diag"]["argmax_accuracy"] if x["diag"] else None)
        sp = x["splits"]
        pr["P2"] = dict(non_contested_nonzero=sp["non_contested"]["nonzero"] if sp["non_contested"] else None, added_px=x["Z5x"]["added_px"] if x["Z5x"] else None,
                        good=sp["good"]["dJF"], fail=sp["fail"]["dJF"], spill14=sp["spill14"]["dJF"],
                        ok=(sp["non_contested"] is None or sp["non_contested"]["nonzero"] == 0) and (x["Z5x"] is None or x["Z5x"]["added_px"] == 0)
                        and -0.001 <= sp["good"]["dJF"] <= 0.0035 and 0 <= sp["fail"]["dJF"] <= 0.020 and -0.02 <= sp["spill14"]["dJF"] <= 0.05)
        pr["P3"] = dict(down=sp["good"]["down"] + sp["fail"]["down"], fails=x["fails"], halluc=x["hallucinations"], worst=x["gate"]["worst_delta"], exempt=x["Z1"]["masks_exempt"] if x["Z1"] else None,
                        ok=(sp["good"]["down"] + sp["fail"]["down"]) <= 10 and x["fails"] <= 15 and x["hallucinations"] == res["C0"]["hallucinations"] and x["gate"]["worst_delta"] >= -0.006
                        and (x["Z1"] is None or 5 <= x["Z1"]["masks_exempt"] <= 120))
        pr["P4"] = dict(smaller_fraction=x["diag"]["smaller_fraction"] if x["diag"] else None,
                        ok=x["diag"] is not None and x["diag"]["smaller_fraction"] is not None and 0.55 <= x["diag"]["smaller_fraction"] <= 0.85)
    if "X1" in res and "X2" in res:
        dd = res["X2"]["paired"]["JF"]["delta"] - res["X1"]["paired"]["JF"]["delta"]
        # rule live: X2 and X1 PNGs differ somewhere
        n_diff = n_all = 0
        for sc in SCENES:
            a, b = _pngs(_result(sc, res["X1"]["name"])), _pngs(_result(sc, res["X2"]["name"]))
            for k in set(a) & set(b):
                n_all += 1; n_diff += int(not _bytes_equal(a[k], b[k]))
        s14 = res["X2"]["splits"]["spill14"]["dJF"] - res["X1"]["splits"]["spill14"]["dJF"]
        pr["P5"] = dict(X2_minus_X1=dd, pngs_differ=n_diff, pngs=n_all, spill14_diff=s14, same_gate=res["X2"]["gate"]["pass"] == res["X1"]["gate"]["pass"],
                        ok=-0.0015 <= dd <= 0.002 and n_diff > 0 and -0.01 <= s14 <= 0.02 and res["X2"]["gate"]["pass"] == res["X1"]["gate"]["pass"])
        pr["P9"] = dict(ok=res["X1"]["paired"]["JF"]["delta"] >= res["X2"]["paired"]["JF"]["delta"] - 0.0015)
    if "X1" in res and "D" in res:
        dd = res["D"]["paired"]["JF"]["delta"] - res["X1"]["paired"]["JF"]["delta"]
        ex = sets.get("X1", {}).get("exempt", set())
        # every pair where D != X1 is an exempt-touched pair of X1 (same camera as an exempt mask)
        kd = {key(r): r for r in rows_of(res["D"]["name"])}; kx = {key(r): r for r in rows_of(res["X1"]["name"])}
        ex_cams = {(k[0], k[1]) for k in ex}
        differ = [k for k in kx if abs(kd[k]["JF"] - kx[k]["JF"]) > 1e-9]
        not_touched = [k for k in differ if (k[0], k[1]) not in ex_cams]
        pr["P6"] = dict(D_minus_X1=dd, emptied=res["D"]["Z1"]["masks_emptied"] if res["D"]["Z1"] else None, pairs_differ=len(differ), pairs_differ_outside_exempt_cams=len(not_touched),
                        reading="cap protected" if dd < -0.001 else ("cap forfeited" if dd > 0.001 else "cap neutral"),
                        ok=-0.005 <= dd <= 0.001 and (res["D"]["Z1"] is None or 0 <= res["D"]["Z1"]["masks_emptied"] <= 60) and len(not_touched) == 0)
    for rid in ("X1", "X2", "D"):
        if rid in res:
            pr[f"P10_{rid}_under22"] = dict(dJF=res[rid]["splits"]["under22"]["dJF"], ok=abs(res[rid]["splits"]["under22"]["dJF"]) <= 0.01)
            for z in ("Z1", "Z1m", "Z2", "Z5x", "Z7"):
                v = res[rid].get(z)
                if v is not None:
                    pr[f"{z}_{rid}"] = bool(v.get("ok"))
    out = _load_json(OUT) if os.path.isfile(OUT) else {}
    out.update({"prereg": "docs/stage1-E11-prereg.md", "stage1": dict(arms=res, passing_tags=passing, predictions=pr)})
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"stage-1 gate: passing {passing}   predictions {json.dumps(pr)}")


def stage2():
    sys.path.insert(0, os.path.join(REPO, "MVSeed"))
    import e7_judge as E7
    from e8_judge import ci_at
    from e9_judge import z3_digest
    out = _load_json(OUT) if os.path.isfile(OUT) else {}
    s1 = out.get("stage1")
    if s1 is None or "passing_tags" not in s1:
        raise SystemExit("stage-1 judgement missing (run --stage 1 first)")
    tags = s1["passing_tags"]
    if not tags:
        out["stage2"] = dict(runs={}, passed=[], adopted=None, adopted_tag=None, predictions={}, note="no arm passed the stage-1 gate")
        json.dump(out, open(OUT, "w"), indent=1)
        print("stage-2: no arm passed the stage-1 gate; nothing to judge (prereg section 5-1)"); return
    d = E7.load(["jf_e11.json"])
    res = {}
    for rid, folder, tag, _ in ARMS:
        if tag not in tags:
            continue
        m = f"SegMaskSam3XW1CGPS4MSd{tag}FsCs10kG2Rs"
        Rm = list(E7.rows(d, m)); Rb = list(E7.rows(d, HEADLINE))
        if not Rm or not Rb:
            print(f"{rid}: no rows for {m} or {HEADLINE} (not scored yet?)"); continue
        sm, sb = E7.scene_scores(Rm), E7.scene_scores(Rb)
        cm, cb = E7.scene_scores(list(E7.rows(d, m, "complete"))), E7.scene_scores(list(E7.rows(d, HEADLINE, "complete")))
        st, stc = E7.paired(sb, sm), E7.paired(cb, cm)
        worst = min(st["per_scene"].values())
        rule = dict(mean_gt_0=st["mean"] > 0, ci_lower_gt_0=st["ci"][0] > 0, defence=worst >= DEFENCE, worst=worst, complete_ge_0=stc["mean"] >= 0)
        rule["pass"] = all(rule[k] for k in ("mean_gt_0", "ci_lower_gt_0", "defence", "complete_ge_0"))
        z3 = z3_digest("geoset_pred_ft", m, result=folder)
        z3b = dict(same=0, different=0, missing=0)
        from PIL import Image
        import numpy as np
        for r in Rm:
            if r["ref"]: continue
            c = E7.CFG[r["scene"]]; f, s0 = c["folder"], c["start_frame"]
            a = os.path.join(_result(r["scene"], folder), r["cam"], f"{r['obj']}.png")
            b = os.path.join(E7.DATA, f, m, r["cam"], str(s0), f"{r['obj']}.png")
            if not os.path.exists(a) and not os.path.exists(b): continue
            if not os.path.exists(a) or not os.path.exists(b): z3b["missing"] += 1; continue
            ma = np.asarray(Image.open(a).convert("L")) > 127; mb = np.asarray(Image.open(b).convert("L")) > 127
            z3b["same" if ma.shape == mb.shape and bool((ma == mb).all()) else "different"] += 1
        ref0 = [E7.jf(r, 0) for r in Rm if r["ref"] and r["gt"][0]]
        p8 = all(x is not None and x > 0.999 for x in ref0)
        res[rid] = dict(method=m, tag=tag, base=HEADLINE, n_scenes=len(sm), basic=statistics.mean(sm.values()), base_basic=statistics.mean(sb.values()),
                        paired=st, ci_bonferroni2=ci_at(sb, sm, 1 - 0.05 / 2), complete=statistics.mean(cm.values()), base_complete=statistics.mean(cb.values()),
                        paired_complete=stc, rule=rule, Z3_mask_digest=z3, Z3b_frame0_pixels=z3b, P8_ref_frame0_is_gt=p8)
        print(f"{rid} {m}\n   basic {res[rid]['basic']:.3f} vs {res[rid]['base_basic']:.3f}  Δ {st['mean']:+.3f}  CI [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]"
              f"  (97.5% [{res[rid]['ci_bonferroni2'][0]:+.3f}, {res[rid]['ci_bonferroni2'][1]:+.3f}])  {st['wins']}↑/{st['losses']}↓  worst {st['worst'][1]} {st['worst'][0]:+.2f}"
              f"\n   complete {res[rid]['complete']:.3f} vs {res[rid]['base_complete']:.3f}  Δ {stc['mean']:+.3f}  rule {rule}\n   Z3 {z3}  Z3b {z3b}  P8 {p8}")
    passed = [r for r in GATED if r in res and res[r]["rule"]["pass"]]
    adopted = None
    if passed:
        best = max(res[r]["basic"] for r in passed)
        adopted = next(r for r in GATED if r in passed and res[r]["basic"] >= best - TIE)      # tie -> X1
    pr = {}
    for r in res:
        kf = s1.get("arms", {}).get(r, {}).get("paired", {}).get("JF", {}).get("delta")
        if kf:
            ratio = res[r]["paired"]["mean"] / (66.7 * kf)
            pr[f"P7_{r}"] = dict(ratio=ratio, basic_delta=res[r]["paired"]["mean"], ok=0.3 <= ratio <= 1.5 and 0 <= res[r]["paired"]["mean"] <= 0.3)
        pr[f"P8_{r}"] = res[r]["P8_ref_frame0_is_gt"]
        pr[f"Z3_{r}"] = isinstance(res[r]["Z3_mask_digest"], dict) and res[r]["Z3_mask_digest"]["bad"] == [] and res[r]["Z3_mask_digest"]["ok"] == 17
    out["stage2"] = dict(runs=res, passed=passed, adopted=adopted, adopted_tag=res[adopted]["tag"] if adopted else None, predictions=pr)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nstage-2: passed {passed} -> adopted {adopted} ({out['stage2']['adopted_tag']})   predictions {pr}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    a = ap.parse_args()
    stage1() if a.stage == 1 else stage2()
