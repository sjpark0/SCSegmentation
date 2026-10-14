#!/usr/bin/env python3
"""S1-E12 judge (docs/stage1-E12-prereg.md sections 3-7), keyed by result folder (E11 structure).

  --zt ARM K  after every training fold (chain stage zt): the registered training invariants of runs/<ARM>/fold<K>
              (config.json Z1, rng-stream identity against the other trained arm's same fold, the zero-dose arm's step-1
              loss against the adopted log, cut coin share, fold wall time <= 1.30x the adopted fold).  Exit 1 aborts.
  --stage 1   keyframe scores (MVSeed/runs/e12/kf_<folder>.json): M+ and C0' against C0 and M+ against C0'
              (mvseg2d.paired.paired_scene_bootstrap on J, F, J&F, 10 000, seed 0), gate G1/G2/G3 for M+, splits, the
              visibility block from MVSeed/runs/e12/vis.json (recall per class, FP, added/removed precision), the
              mechanism verdict V1-V4, training records, predictions P1-P8, watch values.  stage1/.venv (numpy, cv2).
  --stage 2   MUVOD basic/complete of M+'s in-process headline run (Data/MVSeg/jf_e12.json) against the headline,
              the E8 rule, P9, Z3/Z3b.
All write MVSeed/runs/e12/e12_judge.json (stage 2 merges into stage 1's file).
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(REPO, "stage1")
RUNS = os.environ.get("E12_RUNS", os.path.join(S1, "geomem", "runs"))      # override only for dry runs of --zt
OUT_DIR = os.path.join(REPO, "MVSeed", "runs", "e12")
OUT = os.path.join(OUT_DIR, "e12_judge.json")
VIS = os.path.join(OUT_DIR, "vis.json")
D0, D0_R1 = "geoset_pred_ft_rep_x_e11", "geoset_pred_ft_r1_x_e11"
ADOPTED_ARM = "geoset_pred_ft"
ARMS = {"C0p": ("geoset_cut0_ft", None), "Mp": ("geoset_cutall_ft", "geoset_cutall_rep_x")}
FOLDER = {k: f"{arm}_rep_x_e12" for k, (arm, _) in ARMS.items()}
FOLDER_R1 = {k: f"{arm}_r1_x_e12" for k, (arm, _) in ARMS.items()}
HEADLINE = "SegMaskSam3XW1CGPS4MSdgeoset_rep_xFsCs10kG2Rs"
DEFENCE, SAFETY = -1.0, -0.01
LEAK, BUG = 0.012, 0.025
STEPS = {0: 2574, 1: 2574, 2: 2772, 3: 2772, 4: 2772}
SCENES = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "CBABasketball", "Carpark", "CoffeeMartini",
          "Dog", "Fencing", "FlameSteak", "Frog", "MATF", "MartialArts", "Painter", "PoznanStreet", "Welder")
INV = ("outside", "occluded", "nodepth")
CLS = ("vis",) + INV


def _load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _rows(path):
    return list(csv.DictReader(open(path)))


def _mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


# ---------------------------------------------------------------- zt: training invariants per fold
def _same_code(a, b):
    """True when the stage-1 training/decoding code is identical at commits a and b (doc-only commits during a chain)."""
    import subprocess
    if not a or not b:
        return False
    r = subprocess.run(["git", "-C", REPO, "diff", "--quiet", a, b, "--", "stage1/geomem", "stage1/methods"], capture_output=True)
    return r.returncode == 0


def zt(arm, k, chain_head=None):
    d = os.path.join(RUNS, arm, f"fold{k}")
    cfg = _load_json(os.path.join(d, "config.json"))
    log = _rows(os.path.join(d, "log.csv"))
    ad = os.path.join(RUNS, ADOPTED_ARM, f"fold{k}")
    acfg, alog = _load_json(os.path.join(ad, "config.json")), _rows(os.path.join(ad, "log.csv"))
    prob = {"geoset_cut0_ft": 0.0, "geoset_cutall_ft": 1.0}[arm]
    chk = {
        "trainable_params": cfg.get("trainable_params") == 5_922_321,
        "cut_prob": cfg.get("cut_prob") == prob, "cut_keep": cfg.get("cut_keep") == [0.3, 0.7], "cut_min_cells": cfg.get("cut_min_cells") == 20,
        "extra_mem_pred": cfg.get("extra_mem_pred") is True, "extra_mem_prob": cfg.get("extra_mem_prob") == 0.5,
        "train_decoder": cfg.get("train_decoder") is False,
        "recipe": cfg.get("epochs") == 3 and cfg.get("frame_stride") == 2 and cfg.get("seed") == 0,
        "steps": cfg.get("steps") == STEPS[k] and len(log) == STEPS[k],
        "git": chain_head is None or cfg.get("git") == chain_head or _same_code(chain_head, cfg.get("git")),
        "coin_share": all(r["cut"] == ("1" if prob == 1.0 else "0") for r in log),
        "c0p_no_cut": prob == 1.0 or all(r["n_cut"] == "0" for r in log),
    }
    time_ratio = float(cfg["seconds"]) / float(acfg["seconds"])
    chk["time"] = time_ratio <= 1.30
    if arm == "geoset_cut0_ft":
        chk["step1_loss"] = float(log[0]["loss"]) == float(alog[0]["loss"])
    other = os.path.join(RUNS, "geoset_cutall_ft" if arm == "geoset_cut0_ft" else "geoset_cut0_ft", f"fold{k}", "log.csv")
    if os.path.isfile(other):
        olog = _rows(other)
        chk["stream_identity"] = len(olog) == len(log) and all(a[c] == b[c] for a, b in zip(log, olog) for c in ("objs", "mem_coin", "mem_cam"))
    # reported watch values (P8 ranges; not abort criteria)
    n_mem2 = sum(int(r["n_mem"]) == 2 for r in log) / len(log)
    gf = [float(r["gated_frac"]) for r in log if r["gated_frac"] not in ("", None)]
    last = _mean(float(r["loss"]) for r in log[-100:])
    alast = _mean(float(r["loss"]) for r in alog[-100:])
    watch = dict(time_ratio=time_ratio, n_mem2_share=n_mem2, gated_frac_mean=_mean(gf), last100_loss=last, adopted_last100_loss=alast,
                 step1_loss=float(log[0]["loss"]), adopted_step1_loss=float(alog[0]["loss"]))
    if prob == 1.0:
        nc = [int(r["n_cut"]) for r in log]
        no = [len(r["objs"].split()) for r in log]
        fr = [float(r["cut_frac"]) for r in log if r["cut_frac"]]
        watch.update(steps_with_cut=sum(x >= 1 for x in nc) / len(nc), n_cut_per_obj=_mean(a / b for a, b in zip(nc, no) if b),
                     cut_frac_mean=_mean(fr))
    ok = all(chk.values())
    return dict(arm=arm, fold=k, ok=ok, checks=chk, watch=watch)


# ---------------------------------------------------------------- stage 1
def eligible_pairs(keys):
    """(scene, cam, obj) -> object has >= 20 cells on the 72x72 reference grid (nearest 1008 resize, >= half a cell)."""
    sys.path.insert(0, S1)
    from mvseg2d.io import load_labelmap, load_meta
    out, labs = {}, {}
    for sc, cam, o in keys:
        if sc not in labs:
            m = load_meta(os.path.join(S1, "Data"), sc)
            labs[sc] = load_labelmap(os.path.join(S1, "Data", sc, "masks", f"{m['ref_cam']}.png"))
        lab = labs[sc]
        h, w = lab.shape
        l = lab[(np.arange(1008) * h // 1008)][:, (np.arange(1008) * w // 1008)]
        out[(sc, cam, o)] = int(((l == o).reshape(72, 14, 72, 14).mean(axis=(1, 3)) >= 0.5).sum()) >= 20
    return out


def vis_block(vrows, groups, ref_cam_unused=None, n_boot=10_000, seed=0):
    """Pixel-weighted recall per class (ratio of sums) per folder and group; scene-cluster bootstrap of the differences
    Mp-C0p and C0p-C0 (final and round 1); pair-mean recall (pairs >= 100 px in the class); FP; pred/GT; added/removed."""
    folders = {"C0": D0, "C0p": FOLDER["C0p"], "Mp": FOLDER["Mp"], "C0_r1": D0_R1, "C0p_r1": FOLDER_R1["C0p"], "Mp_r1": FOLDER_R1["Mp"]}
    key = lambda r: (r["scene"], r["cam"], r["obj"])
    res = {}
    rng = np.random.default_rng(seed)
    for gname, gset in groups.items():
        rs = [r for r in vrows if key(r) in gset]
        scenes = sorted({r["scene"] for r in rs})
        si = {s: i for i, s in enumerate(scenes)}
        S = len(scenes)

        def sums(field_fn):
            a = np.zeros(S)
            for r in rs:
                a[si[r["scene"]]] += field_fn(r)
            return a
        g = {}
        for c in CLS + ("inv",):
            gt = sums(lambda r: sum(r[f"gt_{x}"] for x in INV) if c == "inv" else r[f"gt_{c}"])
            hit = {f: sums(lambda r, f=f: sum(r["folders"][folders[f]][f"hit_{x}"] for x in INV) if c == "inv" else r["folders"][folders[f]][f"hit_{c}"])
                   for f in folders}
            rec = {f: float(hit[f].sum() / gt.sum()) if gt.sum() else None for f in folders}
            idx = rng.integers(0, S, size=(n_boot, S)) if S else None
            boots = {}
            for name, (a, b) in {"Mp-C0p": ("Mp", "C0p"), "C0p-C0": ("C0p", "C0"), "Mp-C0": ("Mp", "C0"),
                                 "Mp-C0p_r1": ("Mp_r1", "C0p_r1"), "C0p-C0_r1": ("C0p_r1", "C0_r1")}.items():
                if not S or not gt.sum():
                    boots[name] = None; continue
                G = gt[idx].sum(1)
                d = hit[a][idx].sum(1) / G - hit[b][idx].sum(1) / G
                boots[name] = dict(delta=rec[a] - rec[b], ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
            pm = {f: _mean(r["folders"][folders[f]][f"hit_{c}"] / r[f"gt_{c}"] for r in rs if c != "inv" and r[f"gt_{c}"] >= 100)
                  for f in folders} if c != "inv" else None
            g[c] = dict(gt=int(gt.sum()), recall=rec, diff=boots, pair_mean=pm)
        fp = {f: dict(fp=int(sum(r["folders"][folders[f]]["fp_px"] for r in rs)),
                      fp_inv=int(sum(sum(r["folders"][folders[f]][f"fp_{x}"] for x in INV) for r in rs)),
                      fp_vis=int(sum(r["folders"][folders[f]]["fp_vis"] for r in rs)),
                      pred_gt=sum(r["folders"][folders[f]]["pred_px"] for r in rs) / max(sum(r["gt_px"] for r in rs), 1)) for f in folders}
        pairs = {}
        for pname in (f"{FOLDER['Mp']}:{FOLDER['C0p']}", f"{FOLDER['C0p']}:{D0}", f"{FOLDER_R1['Mp']}:{FOLDER_R1['C0p']}", f"{FOLDER_R1['C0p']}:{D0_R1}"):
            if not rs or pname not in rs[0]["pairs"]:
                continue
            tot = {k2: sum(r["pairs"][pname][k2] for r in rs) for k2 in rs[0]["pairs"][pname]}
            add_inv = sum(tot[f"add_{x}"] for x in INV); add_tp_inv = sum(tot[f"add_tp_{x}"] for x in INV)
            rem_inv = sum(tot[f"rem_{x}"] for x in INV); rem_tp_inv = sum(tot[f"rem_tp_{x}"] for x in INV)
            pairs[pname] = dict(totals=tot, add_inv=add_inv, add_inv_precision=add_tp_inv / add_inv if add_inv else None,
                                rem_inv=rem_inv, rem_inv_precision=rem_tp_inv / rem_inv if rem_inv else None,
                                net_add_inv=add_inv - rem_inv, net_tp_inv=add_tp_inv - rem_tp_inv)
        res[gname] = dict(n=len(rs), classes=g, fp=fp, pairs=pairs)
    return res


def stage1():
    sys.path.insert(0, S1)
    from mvseg2d.paired import paired_scene_bootstrap
    from mvseg2d.report import flatten

    def rows_of(name):
        rs = flatten(_load_json(os.path.join(OUT_DIR, f"kf_{name}.json")))
        for r in rs:
            r["JF"] = 0.5 * (r["J"] + r["F"])
        return rs

    def hallu(name):
        return sum(s["hallucinations"] for s in _load_json(os.path.join(OUT_DIR, f"kf_{name}.json"))["scenes"].values() if s["status"] == "ok")

    key = lambda r: (r["scene"], r["cam"], r["obj"])
    R = {"C0": rows_of(D0)}
    for a in ARMS:
        p = os.path.join(OUT_DIR, f"kf_{FOLDER[a]}.json")
        if not os.path.isfile(p):
            raise SystemExit(f"{a}: {p} missing (not decoded / scored yet)")
        R[a] = rows_of(FOLDER[a])
    K = {a: {key(r): r for r in rs} for a, rs in R.items()}
    k0 = K["C0"]
    for a in ARMS:
        if set(K[a]) != set(k0):
            raise SystemExit(f"{a}: pair set differs from C0 ({len(K[a])} vs {len(k0)}) -- fix the decode first")
    sf0 = {(r["scene"], r["cam"], r["obj"]): r for r in _load_json(os.path.join(REPO, "MVSeed", "runs", "seedfail.json"))["rows"]}
    good = {k for k in k0 if sf0[k]["JF"] >= 0.8}
    under = {k for k in k0 if sf0[k]["JF"] < 0.8 and sf0[k].get("kind", "").startswith("under")}
    beyond = {k for k in under if "beyond" in sf0[k]["kind"]}
    spill = {k for k in k0 if sf0[k]["JF"] < 0.8 and sf0[k].get("kind", "").startswith("spill")}
    elig = eligible_pairs(k0)
    vis = _load_json(VIS) if os.path.isfile(VIS) else None
    vrows = vis["rows"] if vis else []
    inv_frac = {key(r): sum(r[f"gt_{x}"] for x in INV) / max(r["gt_px"], 1) for r in vrows}
    sets = dict(all=set(k0), good=good, fail=set(k0) - good, under22=under, under_beyond9=beyond, under_within13=under - beyond,
                under21_noDogTree={k for k in under if not (k[0] == "Dog" and k[1] == "camera_0004" and k[2] == 3)},
                spill14=spill, twin={k for k in k0 if sf0[k].get("twin")}, eligible={k for k in k0 if elig[k]},
                ineligible={k for k in k0 if not elig[k]},
                **{f"otype_{t}": {k for k, r in k0.items() if r["type"] == t} for t in ("dynamic", "static", "env")},
                **{f"rig_{t}": {k for k, r in k0.items() if r["rig"] == t} for t in ("pinhole", "hemisphere")},
                **{f"size_{b}": {k for k, r in k0.items() if r["size_bin"] == b} for b in ("<2000", "2000-10000", ">=10000")},
                **{f"vd_{b}": {k for k, r in k0.items() if lo <= r["view_dist"] <= hi} for b, lo, hi in (("<=1", 0, 1), ("2", 2, 2), ("3-4", 3, 4), (">=5", 5, 10 ** 6))},
                **({f"inv_{b}": {k for k in k0 if lo <= inv_frac.get(k, -1) < hi} for b, lo, hi in
                    (("0-.2", 0, 0.2), (".2-.5", 0.2, 0.5), (".5-.8", 0.5, 0.8), (".8-1", 0.8, 1.01))} if inv_frac else {}),
                **({"inv_ge.2": {k for k in k0 if inv_frac.get(k, -1) >= 0.2}} if inv_frac else {}))

    def contrast(a, b):
        st = {m: paired_scene_bootstrap(R[b], R[a], key=m) for m in ("J", "F", "JF")}
        d = {k: K[a][k]["JF"] - K[b][k]["JF"] for k in k0}
        splits = {n: dict(n=len(s), dJF=_mean(d[k] for k in s), dJ=_mean(K[a][k]["J"] - K[b][k]["J"] for k in s),
                          dF=_mean(K[a][k]["F"] - K[b][k]["F"] for k in s), up=sum(d[k] > 0.01 for k in s), down=sum(d[k] < -0.01 for k in s),
                          sum_dJF=float(sum(d[k] for k in s)))
                  for n, s in sets.items() if s}
        worst = min(st["JF"]["per_scene"].items(), key=lambda x: x[1])
        return dict(paired={m: dict(delta=v["delta"], ci=list(v["ci"]), per_scene=v["per_scene"]) for m, v in st.items()},
                    worst_scene=worst[0], worst_delta=worst[1], splits=splits,
                    pairs={f"{k[0]}|{k[1]}|{k[2]}": d[k] for k in sorted(under)})
    C = {"Mp-C0": contrast("Mp", "C0"), "C0p-C0": contrast("C0p", "C0"), "Mp-C0p": contrast("Mp", "C0p")}
    g = C["Mp-C0"]
    gate = dict(G1_mean_gt_0=g["paired"]["JF"]["delta"] > 0, G1_ci_lower_gt_0=g["paired"]["JF"]["ci"][0] > 0,
                G2_no_scene_below=g["worst_delta"] >= SAFETY, worst_scene=g["worst_scene"], worst_delta=g["worst_delta"],
                G3_vs_C0p=C["Mp-C0p"]["paired"]["JF"]["delta"] > 0)
    gate["pass"] = all(gate[k] for k in ("G1_mean_gt_0", "G1_ci_lower_gt_0", "G2_no_scene_below", "G3_vs_C0p"))
    dmp = g["paired"]["JF"]["delta"]
    gate["leak_check"] = dmp > LEAK
    gate["bug"] = dmp > BUG
    if gate["bug"]:
        gate["pass"] = False
    # visibility block + mechanism verdict
    vb = vis_block(vrows, {"all": sets["all"], "good": good, "under22": under}) if vrows else None
    verdict = None
    if vb:
        v1 = vb["good"]["classes"]["outside"]["diff"]
        v1d, v1n = v1["Mp-C0p_r1"], v1["C0p-C0_r1"]
        V1 = bool(v1d and v1d["delta"] > 0 and v1d["ci"][0] > 0 and v1d["delta"] > abs(v1n["delta"]))
        v2 = vb["all"]["classes"]["outside"]["diff"]
        V2 = bool(v2["Mp-C0p"] and v2["Mp-C0p"]["delta"] > 0 and v2["Mp-C0p"]["ci"][0] > 0 and v2["Mp-C0p"]["delta"] > abs(v2["C0p-C0"]["delta"]))
        pr = vb["all"]["pairs"].get(f"{FOLDER['Mp']}:{FOLDER['C0p']}", {})
        prec = pr.get("add_inv_precision")
        V3 = prec is not None and prec >= 0.5
        visd = vb["all"]["classes"]["vis"]["diff"]["Mp-C0p"]
        fpv = vb["all"]["fp"]
        fp_growth = fpv["Mp"]["fp_vis"] / fpv["C0p"]["fp_vis"] - 1 if fpv["C0p"]["fp_vis"] else None
        V4 = visd is not None and abs(visd["delta"]) <= 0.003 and fp_growth is not None and fp_growth <= 0.10
        reading = "completes" if (V1 and V2 and V3 and V4) else ("extends but cannot localise" if (V1 or V2) and not V3 else
                                                                   ("no mechanism effect" if not V1 else "partial"))
        verdict = dict(V1=V1, V2=V2, V3=V3, V4=V4, reading=reading, V1_r1_outside_good=v1d, V1_noise=v1n,
                       V2_outside_all=v2["Mp-C0p"], V2_noise=v2["C0p-C0"], V3_add_inv_precision=prec, V4_vis_recall=visd, V4_vis_fp_growth=fp_growth)
    # training records
    train = {}
    for a, (arm, _) in ARMS.items():
        train[a] = [zt(arm, k) for k in range(5) if os.path.isfile(os.path.join(RUNS, arm, f"fold{k}", "config.json"))]
    # decode-side records
    def run_stats(folder):
        sec = gpu = 0.0; conf_n = conf_t = 0; ref_only = []
        for sc in SCENES:
            p = os.path.join(S1, "Data", sc, "Result", folder, "RUN.json")
            if os.path.isfile(p):
                r = _load_json(p); sec += float(r.get("seconds", 0)); gpu = max(gpu, float(r.get("gpu_mb", 0) or 0))
                ref_only.append(r.get("config", {}).get("views_with_ref_only_frac"))
            c = os.path.join(S1, "Data", sc, "Result", folder, "round1_conf.json")
            if os.path.isfile(c):
                for cam, objs in _load_json(c).items():
                    for o, v in objs.items():
                        conf_n += 1
                        ov = v["overlap"]
                        conf_t += int(v["presence"] > 0 and ((ov >= 0.5) if ov is not None else v["iou"] >= 0.7))
        return dict(seconds=sec, gpu_mb=gpu, conf_share=conf_t / conf_n if conf_n else None,
                    views_with_ref_only_frac=_mean(x for x in ref_only if x is not None))
    dec = {a: run_stats(FOLDER[a]) for a in ARMS}
    dec["C0"] = run_stats(D0)
    z7 = {a: dict(ok=dec[a]["seconds"] <= min(45 * 60, 1.15 * 465) and dec[a]["gpu_mb"] <= dec["C0"]["gpu_mb"] + 1000) for a in ARMS}
    res = {"C0": dict(name=D0, n=len(R["C0"]), hallucinations=hallu(D0), fails=sum(r["J"] < 0.5 for r in R["C0"]))}
    for a in ARMS:
        res[a] = dict(name=FOLDER[a], hallucinations=hallu(FOLDER[a]), fails=sum(r["J"] < 0.5 for r in R[a]))
    passing = [ARMS["Mp"][1]] if gate["pass"] else []
    # predictions
    P = {}
    P["P1"] = dict(Mp_C0=dmp, C0p_C0=C["C0p-C0"]["paired"]["JF"]["delta"], ok=-0.004 <= dmp <= 0.005)
    if verdict:
        o = vb["all"]["classes"]
        P["P2"] = dict(r1_outside_good=verdict["V1_r1_outside_good"], outside_all=o["outside"]["diff"]["Mp-C0p"], occluded=o["occluded"]["diff"]["Mp-C0p"],
                       nodepth=o["nodepth"]["diff"]["Mp-C0p"], vis=o["vis"]["diff"]["Mp-C0p"],
                       pred_gt=vb["all"]["fp"]["Mp"]["pred_gt"],
                       ok=bool(verdict["V1"]) and abs(o["vis"]["diff"]["Mp-C0p"]["delta"]) <= 0.003)
        fpg = vb["good"]["fp"]
        P["P3"] = dict(add_inv_precision=verdict["V3_add_inv_precision"],
                       good_inv_fp_growth=fpg["Mp"]["fp_inv"] / fpg["C0p"]["fp_inv"] - 1 if fpg["C0p"]["fp_inv"] else None,
                       ok=verdict["V3_add_inv_precision"] is not None and 0.35 <= verdict["V3_add_inv_precision"] <= 0.60)
    u = C["Mp-C0p"]["splits"]["under22"]
    P["P4"] = dict(under22=u["dJF"], ok=-0.02 <= u["dJF"] <= 0.06)
    gd = C["Mp-C0p"]["splits"]["good"]
    P["P5"] = dict(good=gd["dJF"], halluc=res["Mp"]["hallucinations"], fails=res["Mp"]["fails"],
                   ok=-0.004 <= gd["dJF"] <= 0.004 and res["Mp"]["hallucinations"] <= 16 and res["Mp"]["fails"] <= 17)
    sp = C["Mp-C0p"]
    dj, df = sp["paired"]["J"]["delta"], sp["paired"]["F"]["delta"]
    inv20 = sp["splits"].get("inv_ge.2"); tot = sum(x["sum_dJF"] for n, x in sp["splits"].items() if n == "all")
    P["P6"] = dict(dJ=dj, dF=df, ineligible=sp["splits"]["ineligible"]["dJF"],
                   inv20_share=(inv20["sum_dJF"] / tot) if inv20 and tot else None,
                   ok=dj > df and abs(sp["splits"]["ineligible"]["dJF"]) <= 0.002)
    P["P7"] = dict(conf_share=dict(C0=dec["C0"]["conf_share"], Mp=dec["Mp"]["conf_share"]),
                   ok=dec["Mp"]["conf_share"] is not None and dec["C0"]["conf_share"] is not None and abs(dec["Mp"]["conf_share"] - dec["C0"]["conf_share"]) <= 0.05)
    P["P8"] = dict(ok=all(t["ok"] for a in train for t in train[a]) and len(train["Mp"]) == 5 and len(train["C0p"]) == 5)
    out = _load_json(OUT) if os.path.isfile(OUT) else {}
    out.update({"prereg": "docs/stage1-E12-prereg.md", "stage1": dict(arms=res, contrasts=C, gate=gate, passing_tags=passing, visibility=vb,
                                                                       verdict=verdict, training=train, decode=dec, Z7=z7, predictions=P)})
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for name, c in C.items():
        p = c["paired"]
        print(f"{name}: ΔJ&F {p['JF']['delta']:+.4f} CI [{p['JF']['ci'][0]:+.4f}, {p['JF']['ci'][1]:+.4f}]  ΔJ {p['J']['delta']:+.4f}  ΔF {p['F']['delta']:+.4f}"
              f"  worst {c['worst_scene']} {c['worst_delta']:+.4f}  | good {c['splits']['good']['dJF']:+.4f}  fail {c['splits']['fail']['dJF']:+.4f}"
              f"  under22 {c['splits']['under22']['dJF']:+.4f}  eligible {c['splits']['eligible']['dJF']:+.4f}  ineligible {c['splits']['ineligible']['dJF']:+.4f}")
    print(f"arms {json.dumps(res)}")
    print(f"gate {gate}")
    if verdict:
        print(f"verdict {json.dumps(verdict)}")
        for gname in ("all", "good", "under22"):
            v = vb[gname]
            print(f"  vis[{gname}] recall " + "  ".join(f"{c}: " + "/".join(f"{v['classes'][c]['recall'][f]:.4f}" for f in ("C0", "C0p", "Mp")) for c in CLS + ("inv",))
                  + f"  | FP inv C0/C0p/Mp {v['fp']['C0']['fp_inv']}/{v['fp']['C0p']['fp_inv']}/{v['fp']['Mp']['fp_inv']}")
    print(f"training {json.dumps({a: [dict(fold=t['fold'], ok=t['ok'], **t['watch']) for t in train[a]] for a in train})}")
    print(f"decode {json.dumps(dec)}  Z7 {z7}")
    print(f"stage-1 gate: passing {passing}   predictions {json.dumps(P)}")
    if gate["bug"]:
        print(f"ABORT: M+ ΔJ&F {dmp:+.4f} > {BUG} -- treated as a bug (prereg P1)")
        sys.exit(2)


# ---------------------------------------------------------------- stage 2
def stage2():
    sys.path.insert(0, os.path.join(REPO, "MVSeed"))
    import e7_judge as E7
    from e9_judge import z3_digest
    out = _load_json(OUT) if os.path.isfile(OUT) else {}
    s1 = out.get("stage1")
    if s1 is None or "passing_tags" not in s1:
        raise SystemExit("stage-1 judgement missing (run --stage 1 first)")
    if not s1["passing_tags"]:
        out["stage2"] = dict(runs={}, passed=[], adopted=None, adopted_tag=None, predictions={}, note="M+ did not pass the stage-1 gate")
        json.dump(out, open(OUT, "w"), indent=1)
        print("stage-2: M+ did not pass the stage-1 gate; nothing to judge (prereg section 5-1)"); return
    d = E7.load(["jf_e12.json"])
    tag = ARMS["Mp"][1]
    m = f"SegMaskSam3XW1CGPS4MSd{tag}FsCs10kG2Rs"
    Rm, Rb = list(E7.rows(d, m)), list(E7.rows(d, HEADLINE))
    if not Rm or not Rb:
        raise SystemExit(f"no rows for {m} or {HEADLINE} (not scored yet?)")
    sm, sb = E7.scene_scores(Rm), E7.scene_scores(Rb)
    cm, cb = E7.scene_scores(list(E7.rows(d, m, "complete"))), E7.scene_scores(list(E7.rows(d, HEADLINE, "complete")))
    st, stc = E7.paired(sb, sm), E7.paired(cb, cm)
    worst = min(st["per_scene"].values())
    rule = dict(mean_gt_0=st["mean"] > 0, ci_lower_gt_0=st["ci"][0] > 0, defence=worst >= DEFENCE, worst=worst, complete_ge_0=stc["mean"] >= 0)
    rule["pass"] = all(rule[k] for k in ("mean_gt_0", "ci_lower_gt_0", "defence", "complete_ge_0"))
    z3 = z3_digest(ARMS["Mp"][0], m, result=FOLDER["Mp"])
    z3b = dict(same=0, different=0, missing=0)
    from PIL import Image
    for r in Rm:
        if r["ref"]: continue
        c = E7.CFG[r["scene"]]; f, s0 = c["folder"], c["start_frame"]
        a = os.path.join(S1, "Data", r["scene"], "Result", FOLDER["Mp"], r["cam"], f"{r['obj']}.png")
        b = os.path.join(E7.DATA, f, m, r["cam"], str(s0), f"{r['obj']}.png")
        if not os.path.exists(a) and not os.path.exists(b): continue
        if not os.path.exists(a) or not os.path.exists(b): z3b["missing"] += 1; continue
        ma = np.asarray(Image.open(a).convert("L")) > 127; mb = np.asarray(Image.open(b).convert("L")) > 127
        z3b["same" if ma.shape == mb.shape and bool((ma == mb).all()) else "different"] += 1
    ref0 = [E7.jf(r, 0) for r in Rm if r["ref"] and r["gt"][0]]
    kf = s1["contrasts"]["Mp-C0"]["paired"]["JF"]["delta"]
    ratio = st["mean"] / (66.7 * kf) if kf else None
    run = dict(method=m, tag=tag, base=HEADLINE, basic=statistics.mean(sm.values()), base_basic=statistics.mean(sb.values()), paired=st,
               complete=statistics.mean(cm.values()), base_complete=statistics.mean(cb.values()), paired_complete=stc, rule=rule,
               Z3_mask_digest=z3, Z3b_frame0_pixels=z3b, ref_frame0_is_gt=all(x is not None and x > 0.999 for x in ref0))
    P = dict(P9=dict(ratio=ratio, basic_delta=st["mean"], ok=ratio is not None and 0.5 <= ratio <= 1.5 and 0 <= st["mean"] <= 0.30 and stc["mean"] >= 0),
             Z3=isinstance(z3, dict) and z3["bad"] == [] and z3["ok"] == 17)
    out["stage2"] = dict(runs={"Mp": run}, passed=["Mp"] if rule["pass"] else [], adopted="Mp" if rule["pass"] else None,
                         adopted_tag=tag if rule["pass"] else None, predictions=P)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"Mp {m}\n   basic {run['basic']:.3f} vs {run['base_basic']:.3f}  Δ {st['mean']:+.3f}  CI [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]  "
          f"{st['wins']}↑/{st['losses']}↓  worst {st['worst'][1]} {st['worst'][0]:+.2f}\n   complete {run['complete']:.3f} vs {run['base_complete']:.3f}"
          f"  Δ {stc['mean']:+.3f}  rule {rule}\n   Z3 {z3}  Z3b {z3b}  ref frame 0 = GT {run['ref_frame0_is_gt']}")
    print(f"\nstage-2: adopted {out['stage2']['adopted_tag']}   predictions {P}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2))
    ap.add_argument("--zt", nargs=2, metavar=("ARM", "FOLD"))
    ap.add_argument("--head", default=None, help="chain HEAD (zt: config.json git must equal it)")
    a = ap.parse_args()
    if a.zt:
        r = zt(a.zt[0], int(a.zt[1]), a.head)
        print(json.dumps(r))
        sys.exit(0 if r["ok"] else 1)
    stage1() if a.stage == 1 else stage2()
