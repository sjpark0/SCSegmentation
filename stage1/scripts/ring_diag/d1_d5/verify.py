"""Independent re-scoring of the ring D1-D5 arms (read-only on all inputs).

For every arm: per (camera, person) pair with the person present in the GT of a non-reference camera,
  J_full   : IoU at 3840x2160 (missing PNG = empty mask = J 0)
  J_q      : IoU at 1/4 resolution, nearest resize (swap_diag rule)
  cov_q    : GT id (1..6) with the largest overlap of the predicted mask at 1/4 res, 0 if it covers no person
  cov_f    : same at full resolution (robustness check of the swap count)
  cov_iou  : GT id with the largest IoU with the predicted mask at full res (alternative swap rule)
Also: byte identity checks, D1 status transitions, round-1 gate statistics, near/far split, per-distance means.
"""
import hashlib
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image

REPO = "/home/sjpark/Documents/SCSegmentation"
D15 = "/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_d15"
OUT = os.path.join(D15, "verify")
GT = os.path.join(REPO, "Data/PlenopticFull/T06_flag_ganggang_F/Mask")
NAS = os.path.join(REPO, "PlenopticServer1/CROOM/SegResult/T06_flag_ganggang_F")
REF = "cam_06"
CAMS = [f"cam_{i:02d}" for i in range(32)]
S = 4


def res(datadir, arm):
    return os.path.join(D15, datadir, "T06_flag_ganggang_F/Result", arm)


ARMS = {
    "nas_chain": (os.path.join(NAS, "SegMaskSam3XW1CGPS4MFsCs10kG2"), "nas"),
    "nas_lat": (os.path.join(NAS, "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs"), "nas"),
}
for a in ("preset", "preset_r1", "preset_post", "star", "untr", "untr_r1", "bias0", "bias0_r1", "xnone", "xnone_r1", "xout"):
    ARMS[a] = (res("data", a), "local")
for k in (0, 1, 3, 4):
    ARMS[f"preset_f{k}"] = (res(f"data_f{k}", f"preset_f{k}"), "local")
    ARMS[f"preset_f{k}_r1"] = (res(f"data_f{k}", f"preset_f{k}_r1"), "local")


def path(arm, cam, p):
    root, kind = ARMS[arm]
    return os.path.join(root, cam, "900", f"{p}.png") if kind == "nas" else os.path.join(root, cam, f"{p}.png")


def dist(cam):
    i = int(cam[4:])
    return min(abs(i - 6), 32 - abs(i - 6))


def quarter(a):
    return np.array(Image.fromarray(a).resize((a.shape[1] // S, a.shape[0] // S), Image.NEAREST))


def score_cam(cam):
    lab = np.array(Image.open(os.path.join(GT, cam, "000900.png")))
    labq = quarter(lab)
    gt_area = np.bincount(lab.ravel(), minlength=7)
    gt_area_q = np.bincount(labq.ravel(), minlength=7)
    rows = []
    for arm in ARMS:
        for p in range(1, 7):
            f = path(arm, cam, p)
            exists = os.path.isfile(f)
            if exists:
                m8 = np.array(Image.open(f).convert("L"))
                pm = m8 > 127
            else:
                pm = np.zeros(lab.shape, bool)
            pmq = quarter(pm.astype(np.uint8)) > 0
            cnt = np.bincount(lab[pm], minlength=7)
            cntq = np.bincount(labq[pmq], minlength=7)
            inter, union = cnt[p], pm.sum() + gt_area[p] - cnt[p]
            interq, unionq = cntq[p], pmq.sum() + gt_area_q[p] - cntq[p]
            cov_f = int(np.argmax(cnt[1:]) + 1) if cnt[1:].sum() else 0
            cov_q = int(np.argmax(cntq[1:]) + 1) if cntq[1:].sum() else 0
            ious = [cnt[j] / (pm.sum() + gt_area[j] - cnt[j]) if gt_area[j] else 0.0 for j in range(1, 7)]
            cov_iou = int(np.argmax(ious) + 1) if max(ious) > 0 else 0
            rows.append(dict(arm=arm, cam=cam, d=dist(cam), p=p, exists=exists, gt_px=int(gt_area[p]), gt_px_q=int(gt_area_q[p]),
                             pred_px=int(pm.sum()), J_full=float(inter / union) if union else float("nan"),
                             J_q=float(interq / unionq) if unionq else float("nan"), cov_q=cov_q, cov_f=cov_f, cov_iou=cov_iou))
    return rows


def md5(f):
    return hashlib.md5(open(f, "rb").read()).hexdigest() if os.path.isfile(f) else None


def status(r, key="cov_q"):
    return "empty" if r[key] == 0 else ("ok" if r[key] == r["p"] else "wrong")


def main():
    with Pool(8) as pool:
        rows = [r for rs in pool.map(score_cam, [c for c in CAMS]) for r in rs]
    json.dump(rows, open(os.path.join(OUT, "verify_rows.json"), "w"))
    out = {}
    # pair set: non-reference, GT present at full res (and at 1/4 res)
    for arm in ARMS:
        R = [r for r in rows if r["arm"] == arm and r["cam"] != REF and r["gt_px"] > 0]
        Rq = [r for r in R if r["gt_px_q"] > 0]
        near = [r for r in R if 1 <= r["d"] <= 6]
        far = [r for r in R if 7 <= r["d"] <= 16]
        o = dict(n=len(R), n_q=len(Rq), n_near=len(near), n_far=len(far),
                 J_full=np.mean([r["J_full"] for r in R]), J_q=np.mean([r["J_q"] for r in Rq]),
                 J_near=np.mean([r["J_q"] for r in near]), J_far=np.mean([r["J_q"] for r in far]),
                 Jf_near=np.mean([r["J_full"] for r in near]), Jf_far=np.mean([r["J_full"] for r in far]),
                 J_lt05=sum(r["J_full"] < 0.5 for r in R), n_missing_png=sum(not r["exists"] for r in R))
        for key in ("cov_q", "cov_f", "cov_iou"):
            for nm, S_ in (("near", near), ("far", far)):
                st = Counter(status(r, key) for r in S_)
                o[f"wrong_{nm}_{key}"] = st["wrong"]
                o[f"empty_{nm}_{key}"] = st["empty"]
        o["by_d"] = {d: round(float(np.mean([r["J_q"] for r in R if r["d"] == d])), 3) for d in range(1, 17)}
        out[arm] = o
    # pair membership
    R0 = [r for r in rows if r["arm"] == "preset" and r["cam"] != REF]
    out["_absent_gt_pairs"] = [(r["cam"], r["p"]) for r in R0 if r["gt_px"] == 0]
    # byte identity
    ident = {}
    for a, b in (("preset", "nas_lat"), ("preset", "preset_post"), ("preset_r1", "preset_post_r1")):
        pass
    def cmp(a, b, rootb=None):
        same = diff = onlya = onlyb = 0
        for c in CAMS:
            for p in range(1, 7):
                fa = path(a, c, p)
                fb = path(b, c, p) if rootb is None else os.path.join(rootb, c, f"{p}.png")
                ha, hb = md5(fa), md5(fb)
                if ha is None and hb is None:
                    continue
                if ha is None:
                    onlyb += 1
                elif hb is None:
                    onlya += 1
                elif ha == hb:
                    same += 1
                else:
                    diff += 1
        return dict(same=same, diff=diff, only_a=onlya, only_b=onlyb)
    ident["preset_vs_nas_lat"] = cmp("preset", "nas_lat")
    ident["preset_vs_preset_post"] = cmp("preset", "preset_post")
    ident["preset_r1_vs_preset_post_r1"] = cmp("preset_r1", None, rootb=res("data", "preset_post_r1"))
    ident["preset_r1_vs_xnone_r1"] = cmp("preset_r1", None, rootb=res("data", "xnone_r1"))
    ident["preset_r1_vs_xout_r1"] = cmp("preset_r1", None, rootb=res("data", "xout_r1"))
    ident["star_vs_untr_r1"] = cmp("star", "untr_r1")
    out["_identity"] = ident
    # D1 transitions on the far half (cov_q)
    idx = {(r["arm"], r["cam"], r["p"]): r for r in rows}
    pairs_far = [(r["cam"], r["p"]) for r in R0 if r["gt_px"] > 0 and r["d"] >= 7]
    pairs_all = [(r["cam"], r["p"]) for r in R0 if r["gt_px"] > 0]
    def trans(a, b, P):
        c = Counter(f"{status(idx[(a, *k)])}->{status(idx[(b, *k)])}" for k in P)
        return dict(c)
    out["_D1"] = {"far r1->xnone": trans("preset_r1", "xnone", pairs_far), "far xnone->preset": trans("xnone", "preset", pairs_far),
                  "far r1->preset": trans("preset_r1", "preset", pairs_far),
                  "all r1->preset": trans("preset_r1", "preset", pairs_all)}
    same_wrong = [k for k in pairs_far if status(idx[("preset", *k)]) == "wrong" and status(idx[("preset_r1", *k)]) == "wrong"]
    out["_D1"]["final_wrong_also_r1_wrong"] = len(same_wrong)
    out["_D1"]["...same_person"] = sum(idx[("preset", *k)]["cov_q"] == idx[("preset_r1", *k)]["cov_q"] for k in same_wrong)
    out["_D1"]["changes"] = [(k, status(idx[("preset_r1", *k)]), idx[("preset_r1", *k)]["cov_q"], status(idx[("xnone", *k)]),
                              idx[("xnone", *k)]["cov_q"], round(idx[("preset", *k)]["J_q"], 3))
                             for k in pairs_far if status(idx[("preset_r1", *k)]) != status(idx[("xnone", *k)])
                             or idx[("preset_r1", *k)]["cov_q"] != idx[("xnone", *k)]["cov_q"]]
    # swap overlaps
    def wrong_set(a, P=pairs_far):
        return {k for k in P if status(idx[(a, *k)]) == "wrong"}
    W = {a: wrong_set(a) for a in ARMS}
    out["_swap_overlap"] = {f"preset&{a}": len(W["preset"] & W[a]) for a in ARMS if a != "preset"}
    out["_swap_overlap"]["bias0-preset"] = len(W["bias0"] - W["preset"])
    out["_swap_overlap"]["preset_r1&star"] = len(W["preset_r1"] & W["star"])
    out["_swap_overlap"]["star-preset_r1"] = len(W["star"] - W["preset_r1"])
    out["_swap_overlap"]["preset_r1-star"] = len(W["preset_r1"] - W["star"])
    # swap patterns
    def pat(a):
        return Counter(f"{k[1]}->{idx[(a, *k)]['cov_q']}" for k in W[a]).most_common(6)
    out["_patterns"] = {a: pat(a) for a in ("preset", "preset_r1", "star", "untr", "bias0")}
    # better/worse than chain per pair (J_q, >0.01)
    out["_vs_chain"] = {a: (sum(idx[(a, *k)]["J_q"] > idx[("nas_chain", *k)]["J_q"] + 0.01 for k in pairs_all),
                            sum(idx[(a, *k)]["J_q"] < idx[("nas_chain", *k)]["J_q"] - 0.01 for k in pairs_all))
                        for a in ("preset", "preset_r1", "star", "untr", "bias0", "xnone")}
    json.dump(out, open(os.path.join(OUT, "verify_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
