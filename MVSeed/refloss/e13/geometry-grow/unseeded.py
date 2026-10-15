# GT label category of the primary's unclaimed candidate pixels C_o = G_o & ref-invisible & ~D_o & ~D_other (DEV only)
import glob, json, os, sys
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, "/home/sjpark/Documents/SCSegmentation/stage1")
from mvseg2d.io import load_labelmap, read_pred_mask
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/sjpark/Documents/SCSegmentation/stage1/Data"; DEV = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_dev"
SETS = json.load(open("/home/sjpark/Documents/SCSegmentation/docs/raw/muvod_object_sets.json"))
idx = json.load(open(DEV + "/index.json"))["scenes"]
def rd(p, hw):
    return read_pred_mask(p) if os.path.isfile(p) else np.zeros(hw, bool)
def job(a):
    scene, t, T, var = a
    S = idx[scene]; assert t != S["ref_frame"]
    basic = set(int(x) for x in S["basic"])
    Lc = load_labelmap(f"{DATA}/{scene}/frames/{t:06d}/masks/{S['ref_cam']}.png")
    objs = [o for o in sorted(basic) if (Lc == o).any()]
    gt = load_labelmap(f"{DATA}/{scene}/frames/{t:06d}/masks/{T}.png"); hw = gt.shape
    F = json.load(open(f"{HERE}/main/frames/{scene}_{t:06d}.json"))
    z = np.load(f"{HERE}/main/masks/{scene}/{t:06d}/{T}.npz"); hw_w = tuple(z["hw"])
    # vis from the classes stored implicitly: recompute via grow
    import grow as Gm
    zc = Gm.load_cam(scene, t, S["ref_cam"]); zt = Gm.load_cam(scene, t, T)
    pc = Gm.to_reference(Gm.local_points(zt["depth"], zt["K"]), zt["c2w"], zc["c2w"]).astype(np.float32)
    Kc = Gm.full_K(zc["K"], hw_w, hw); dcf = cv2.resize(zc["depth"], (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    visg, _, _ = Gm.project(pc, Kc, hw, dcf); vis = Gm.up(visg & zt["valid"], hw)
    D = {o: rd(f"{DEV}/{scene}/{t:06d}/{T}/{o}.png", hw) for o in objs}
    cnt = sum(D[o].astype(np.uint8) for o in objs)
    out = np.zeros(4)   # on o, seeded other, basic-but-not-in-c-at-t, non-basic
    for o in objs:
        k = f"{var}|{o}"
        if k not in z.files: continue
        g = Gm.up(np.unpackbits(z[k])[:hw_w[0] * hw_w[1]].reshape(hw_w).astype(bool), hw)
        C = g & ~vis & ~D[o] & ~((cnt - D[o]) > 0)
        l = gt[C]
        out += [(l == o).sum(), np.isin(l, [q for q in objs if q != o]).sum(),
                np.isin(l, [q for q in basic if q not in objs]).sum(), (~np.isin(l, list(basic))).sum()]
    return out
if __name__ == "__main__":
    sys.path.insert(0, HERE)
    var = sys.argv[1] if len(sys.argv) > 1 else "mv"
    jobs = [(s, t, T, var) for s, S in idx.items() for t in S["frames"] for T in S["eval_cams"]]
    with Pool(14) as p:
        r = np.sum(p.map(job, jobs), 0)
    print(f"[{var}] unclaimed candidate C_o px {r.sum():,.0f}: on o {r[0]/r.sum():.3f} | other seeded object {r[1]/r.sum():.3f} | basic object absent from c_ini at t {r[2]/r.sum():.3f} | non-basic (complete-only / unlabelled 0) object {r[3]/r.sum():.3f}")
