"""S1-E14 synthesis check (DEV-A only, host): mechanism invariants of the shift arm keep_tau = exclusive_masks(L + tau)
vs keep_0 = exclusive_masks(L) from the round-2 logit dump: removed px, exemption flips, newly non-empty masks, and
empty-GT ("hallucination") object-views that become non-empty.  Also checks keep_0 == the adopted PNGs."""
import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e14_logitdump/code")
import e14dump as E

ROOT = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e14_logitdump/geoset_rep_x_A"
PAIRS = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_dev/dev_pairs.json"
TAUS = (0.25, 0.5, 1.0)
EMPTY = {}
for r in json.load(open(PAIRS))["empty_gt"]:
    EMPTY.setdefault((r["scene"], int(r["t"]), r["cam"]), {})[int(r["obj"])] = r["seed_px"]

def excl(L):
    B = L.shape[0]
    pos = L > 0
    w = L.argmax(0)
    win = w[None] == np.arange(B)[:, None, None]
    npos = pos.reshape(B, -1).sum(1)
    lost = (pos & ~win).reshape(B, -1).sum(1)
    exempt = (npos > 0) & (lost.astype(np.float32) >= np.float32(0.5) * npos.astype(np.float32))
    return pos & (win | exempt[:, None, None]), exempt

def job(v):
    scene, t, cam = v
    d = E.load(ROOT, scene, t, cam)
    objs = [int(o) for o in d["objects"]]
    L = E.logits_full_np(d["low_res"], d["hw"])
    k0, ex0 = excl(L)
    png = E.read_png_masks(ROOT, scene, t, cam, objs, d["hw"])
    out = dict(view=f"{scene}/{t}/{cam}", png_diff=int((png != k0).sum()), B=len(objs))
    emp = EMPTY.get((scene, t, cam), {})
    for i, o in enumerate(objs):
        if o in emp:
            assert int(k0[i].sum()) == emp[o], (v, o, int(k0[i].sum()), emp[o])
    for tau in TAUS:
        kt, ext = excl(L + np.float32(tau))
        n0, nt = k0.reshape(len(objs), -1).sum(1), kt.reshape(len(objs), -1).sum(1)
        rem = (k0 & ~kt).reshape(len(objs), -1).sum(1)
        add = (kt & ~k0).reshape(len(objs), -1).sum(1)
        flip = ex0 != ext
        out[str(tau)] = dict(
            removed=int(rem.sum()), removed_noflip=int(rem[~flip].sum()), added=int(add.sum()),
            flips_on=int((~ex0 & ext).sum()), flips_off=int((ex0 & ~ext).sum()),
            newly_nonempty=int(((n0 == 0) & (nt > 0)).sum()),
            emptyGT_nonempty0=int(sum(n0[i] > 0 for i, o in enumerate(objs) if o in emp)),
            emptyGT_nonemptyT=int(sum(nt[i] > 0 for i, o in enumerate(objs) if o in emp)),
            emptyGT_px0=int(sum(n0[i] for i, o in enumerate(objs) if o in emp)),
            emptyGT_pxT=int(sum(nt[i] for i, o in enumerate(objs) if o in emp)),
            multi_claim=int((kt.sum(0) >= 2).sum()), multi_claim0=int((k0.sum(0) >= 2).sum()))
    return out

if __name__ == "__main__":
    vs = E.views(ROOT)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        res = p.map(job, vs, chunksize=1)
    tot = dict(views=len(res), png_diff=sum(r["png_diff"] for r in res))
    for tau in TAUS:
        tot[str(tau)] = {k: sum(r[str(tau)][k] for r in res) for k in res[0][str(tau)]}
    print(json.dumps(tot, indent=1))
    json.dump(dict(total=tot, views=res), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_invariants.json"), "w"), indent=1)
