#!/usr/bin/env python3
"""S1-E14 multimask (scratch, git-ignored; host numpy): the pre-registered multimask arms (prereg_multimask.json) on DEV-A.

Per eval view (scene, t, cam) of the logit dump (_e14_logitdump/geoset_rep_x_A):
  * A0 = the adopted masks, recomputed from low_res through the bit-exact host path (e14dump.masks_numpy) and asserted
    equal to the _e13_dev PNGs; its J&F per pair asserted equal to _e13_dev/dev_pairs.json.
  * per present object, all 4 decoder tokens through the adopted upsampling (e14dump.logits_full_np, bit-exact):
      stability_k (288, delta = bf16(0.05) as the decoder compares in bf16; token 0 asserted equal to the dump's value),
      agreement_k with the reprojected reference GT (proj_ref.py: P, vis) = |M_k & P| / |(M_k | P) & vis|,
      oracle J&F_k of the pair object with exclusivity recomputed for it against the other objects at A0.
  * every arm: per-object 288 map -> adopted upsampling -> threshold 0 -> exclusive_masks_np (cap 0.5) over the view;
    J&F (mvseg2d.metrics.jf_pair) recomputed for pairs whose mask changed, else A0's value.
Writes <out>/parts/<scene>__<t>__<cam>.json (pair rows with every arm's J, F; per-object stats).
    PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python mm_eval.py --workers 8
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, S1)
sys.path.insert(0, os.path.join(S1, "geomem", "runs", "_e14_logitdump", "code"))
import e14dump as E  # noqa: E402
from mvseg2d.io import load_labelmap  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")
DUMP = os.path.join(S1, "geomem", "runs", "_e14_logitdump", "geoset_rep_x_A")
PROJ = os.path.join(S1, "geomem", "runs", "_e14_multimask", "proj")
PNG = os.path.join(S1, "geomem", "runs", "_e13_dev")
PAIRS = os.path.join(PNG, "dev_pairs.json")
OUT = os.path.join(S1, "geomem", "runs", "_e14_multimask")

STAB_D = np.float32(0.050048828125)     # bf16(0.05): the decoder's `mask_logits > 0.05` on a bf16 tensor
STAB_T = 0.98
MIN_VALID = 64                          # run.MIN_VALID
SEL_ARMS = ("I4", "I3", "S4", "SF", "R4", "RF")
AGG_ARMS = ("M4", "M3", "W4", "X4")
ARMS = SEL_ARMS + AGG_ARMS


def stability(x):
    """x [..., 288, 288] float32 -> |x > d| / |x > -d| (1 where the union is empty), float32 as torch."""
    f = x.reshape(*x.shape[:-2], -1)
    ai = (f > STAB_D).sum(-1).astype(np.float32)
    au = (f > -STAB_D).sum(-1).astype(np.float32)
    return np.where(au > 0, ai / np.where(au > 0, au, 1), np.float32(1.0)).astype(np.float32)


def select(arm, b, ious, stab, agree, nP, tok0):
    """Token chosen for object b by a selection arm (prereg)."""
    stable = stab[b, 0] >= STAB_T
    if arm == "I4":
        return int(np.argmax(ious[b]))
    if arm == "I3":
        return 1 + int(np.argmax(ious[b, 1:]))
    if arm == "S4":
        return int(np.argmax(stab[b]))
    if arm == "SF":
        return 0 if stable else 1 + int(np.argmax(stab[b, 1:]))
    if arm == "R4":
        return int(np.argmax(agree[b])) if nP[b] >= MIN_VALID else tok0
    if arm == "RF":
        if stable:
            return 0
        return 1 + int(np.argmax(agree[b, 1:])) if nP[b] >= MIN_VALID else tok0
    raise ValueError(arm)


def aggregate(arm, mm, ious):
    """288 map of an aggregation arm from mm [4,288,288] float32."""
    x = mm.astype(np.float64)
    if arm == "M4":
        return x.mean(0).astype(np.float32)
    if arm == "M3":
        return x[1:].mean(0).astype(np.float32)
    if arm == "W4":
        w = np.clip(ious.astype(np.float64), 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(4, 0.25)
        return np.tensordot(w, x, axes=1).astype(np.float32)
    if arm == "X4":
        return mm.max(0).astype(np.float32)
    raise ValueError(arm)


def up(x288, hw):
    return E.logits_full_np(np.asarray(x288, np.float32)[None], hw)[0]


def jf(g, p, bp, se):
    J, F = jf_pair(g, p, bp, se)
    return float(J), float(F)


def view_job(args):
    scene, t, cam, rows, out_dir = args
    t0 = time.time()
    z = E.load(DUMP, scene, t, cam, mm=True)
    objs = [int(o) for o in z["objects"]]
    hw = tuple(int(v) for v in z["hw"])
    B = len(objs)
    pres = z["presence"] > 0
    tok = z["token"].astype(int)
    mm = z["mm_low_res"].astype(np.float32)                 # [B,4,288,288]
    ious = z["mm_ious"].astype(np.float32)                  # [B,4]
    pz = np.load(os.path.join(PROJ, scene, f"{t:06d}", f"{cam}.npz"))
    assert [int(o) for o in pz["objects"]] == objs and tuple(pz["hw"]) == hw
    HW = hw[0] * hw[1]
    P = np.unpackbits(pz["P"], axis=1, count=HW).astype(bool).reshape(B, *hw)
    vis = np.unpackbits(pz["vis"], count=HW).astype(bool).reshape(hw)
    nP = pz["nP"]
    gt = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    assert gt.shape == hw
    bp = bound_pix(*hw)
    se = disk(bp)

    # --- A0: adopted path, asserted equal to the PNGs and dev_pairs J&F
    stab = stability(mm)                                     # [B,4]
    assert np.array_equal(stab[:, 0], z["stability"]), (scene, t, cam)
    L0 = E.logits_full_np(z["low_res"], hw)
    M0 = E.exclusive_masks_np(L0)
    png = E.read_png_masks(PNG, scene, t, cam, objs, hw)
    assert np.array_equal(M0, png), (scene, t, cam, "A0 != _e13_dev PNG")
    pair_objs = {int(r["obj"]): r for r in rows}
    oi = {o: i for i, o in enumerate(objs)}
    G = {o: gt == o for o in pair_objs}
    base = {}
    for o, r in pair_objs.items():
        J, F = jf(G[o], M0[oi[o]], bp, se)
        assert J == r["J"] and F == r["F"], (scene, t, cam, o, J, r["J"], F, r["F"])
        base[o] = (J, F)

    # --- per-token pass: agreement with P, oracle J&F (exclusivity against the others at A0)
    agree = np.zeros((B, 4), np.float64)
    a_inter = np.zeros((B, 4), np.int64)
    a_union = np.zeros((B, 4), np.int64)
    m_px = np.zeros((B, 4), np.int64)
    oracle = {}
    suf = np.full((B, *hw), -np.inf, np.float32)
    for b in range(B - 2, -1, -1):
        np.maximum(suf[b + 1], L0[b + 1], out=suf[b])
    pre = np.full(hw, -np.inf, np.float32)
    for b in range(B):
        o = objs[b]
        if pres[b]:
            for k in range(4):
                row = L0[b] if k == tok[b] else up(mm[b, k], hw)
                pos = row > 0
                inter = int((pos & P[b]).sum())                  # P is inside vis
                uni = int(((pos & vis) | P[b]).sum())
                a_inter[b, k], a_union[b, k], m_px[b, k] = inter, uni, int(pos.sum())
                agree[b, k] = inter / uni if uni > 0 else 0.0
                if o in pair_objs:
                    win = (row > pre) & (row >= suf[b])
                    npos = int(pos.sum())
                    lost = int((pos & ~win).sum())
                    exempt = npos > 0 and np.float32(lost) >= np.float32(0.5) * np.float32(npos)
                    keep = pos & (win | exempt)
                    if k == tok[b]:
                        assert np.array_equal(keep, M0[b]), (scene, t, cam, o, "oracle path != A0")
                        oracle.setdefault(o, [None] * 4)[k] = base[o]
                    else:
                        oracle.setdefault(o, [None] * 4)[k] = jf(G[o], keep, bp, se)
        elif o in pair_objs:
            oracle[o] = [base[o]] * 4
        np.maximum(pre, L0[b], out=pre)
    del suf, pre

    # --- arms
    arm_tokens, res = {}, {o: {} for o in pair_objs}
    arm_stats = {}
    for arm in ARMS:
        L = L0.copy()
        changed, toks = [], []
        for b in range(B):
            if not pres[b]:
                toks.append(-1)
                continue
            if arm in SEL_ARMS:
                k = select(arm, b, ious, stab, agree, nP, int(tok[b]))
                toks.append(k)
                if k != tok[b]:
                    L[b] = up(mm[b, k], hw)
                    changed.append(b)
            else:
                L[b] = up(aggregate(arm, mm[b], ious[b]), hw)
                changed.append(b)
        M = E.exclusive_masks_np(L)
        del L
        arm_tokens[arm] = toks
        n_mask_changed = 0
        for o in pair_objs:
            b = oi[o]
            if np.array_equal(M[b], M0[b]):
                res[o][arm] = base[o]
            else:
                n_mask_changed += 1
                res[o][arm] = jf(G[o], M[b], bp, se)
        diff = M != M0
        arm_stats[arm] = dict(n_obj_changed=len(changed), n_pair_mask_changed=n_mask_changed,
                              px_added=int((diff & M).sum()), px_removed=int((diff & M0).sum()))
        del M, diff

    out_rows = []
    for o, r in pair_objs.items():
        b = oi[o]
        g = G[o]
        gv = g & vis
        out_rows.append(dict(scene=scene, t=t, cam=cam, obj=o, view=f"{t}/{cam}", label=r.get("label"), kind=r.get("kind"),
                             gt_px=int(g.sum()), present=bool(pres[b]), token=int(tok[b]),
                             base_J=base[o][0], base_F=base[o][1],
                             arms={a: res[o][a] for a in ARMS}, oracle=oracle[o],
                             ious=ious[b].tolist(), stab=stab[b].tolist(), agree=agree[b].tolist(), nP=int(nP[b]),
                             m_px=m_px[b].tolist(),
                             P_iou_gt_vis=(float((P[b] & gv).sum() / max(int((P[b] | gv).sum()), 1))),
                             gt_vis_px=int(gv.sum()), base_agree_xcl=float((M0[b] & P[b]).sum() / max(int(((M0[b] & vis) | P[b]).sum()), 1)),
                             tokens={a: arm_tokens[a][b] for a in SEL_ARMS}))
    obj_rows = [dict(obj=objs[b], present=bool(pres[b]), token=int(tok[b]), ious=ious[b].tolist(), stab=stab[b].tolist(),
                     agree=agree[b].tolist(), nP=int(nP[b]), tokens={a: arm_tokens[a][b] for a in SEL_ARMS}) for b in range(B)]
    rec = dict(scene=scene, t=t, cam=cam, B=B, hw=list(hw), nvis=int(pz["nvis"]), arm_stats=arm_stats, rows=out_rows,
               objects=obj_rows, seconds=time.time() - t0)
    with open(os.path.join(out_dir, f"{scene}__{t:06d}__{cam}.json"), "w") as f:
        json.dump(rec, f)
    return scene, t, cam, rec["seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="scene filter (comma list)")
    a = ap.parse_args()
    parts = os.path.join(a.out, "parts")
    os.makedirs(parts, exist_ok=True)
    rows = json.load(open(PAIRS))["rows"]
    byv = {}
    for r in rows:
        byv.setdefault((r["scene"], int(r["t"]), r["cam"]), []).append(r)
    views = E.views(DUMP)
    assert set(byv) <= set(views), set(byv) - set(views)
    if a.only:
        views = [v for v in views if v[0] in a.only.split(",")]
    views = [v for v in views if not os.path.isfile(os.path.join(parts, f"{v[0]}__{v[1]:06d}__{v[2]}.json"))]
    sizes = {}
    for v in views:
        z = np.load(E.logit_path(DUMP, *v))
        sizes[v] = len(z["objects"]) * int(z["hw"][0]) * int(z["hw"][1])
    views.sort(key=lambda v: -sizes[v])
    if a.limit:
        views = views[:a.limit]
    jobs = [(s, t, c, byv.get((s, t, c), []), parts) for (s, t, c) in views]
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for i, (s, t, c, sec) in enumerate(pool.imap_unordered(view_job, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {s} {t} {c} {sec:.1f}s  (wall {time.time() - t0:.0f}s)", flush=True)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
