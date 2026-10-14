"""Round-1 / round-2 time split of the adopted preset (geoset_rep) on a few scenes, in-process (stage1.Stage1), by
timing GeoMemStar.decode_view (depth-0 calls: chunk=None -> round 1, chunk=OBJ_CHUNK -> round 2),
encode_memory_from_prediction and reproj_overlap.  Nothing is written.  Runs in scsam3:
    stage1/scripts/docker_run.sh scsam3 python -c "$(cat gpu_timing.py)"  FencingBlocksMATFBreakfast
"""
import sys, time, json, os
import torch
sys.path.insert(0, "/host/home/sjpark/Documents/SCSegmentation/stage1")
from methods.geomem import run as R
from methods.geomem import stage1 as S
from geomem import wrapper as Wm

T = {}
depth = [0]
orig_decode = Wm.GeoMemStar.decode_view
orig_enc = Wm.GeoMemStar.encode_memory_from_prediction
orig_ov = R.reproj_overlap


def acc(key, dt):
    T[key] = T.get(key, 0.0) + dt
    T["n_" + key] = T.get("n_" + key, 0) + 1


def decode_view(self, backbone_out, B, memory, p_v, valid_v, chunk=None, **kw):
    depth[0] += 1
    try:
        if depth[0] > 1:
            return orig_decode(self, backbone_out, B, memory, p_v, valid_v, chunk=chunk, **kw)
        torch.cuda.synchronize(); t0 = time.time()
        out = orig_decode(self, backbone_out, B, memory, p_v, valid_v, chunk=chunk, **kw)
        torch.cuda.synchronize(); acc("round2_decode" if chunk is not None else "round1_decode", time.time() - t0)
        return out
    finally:
        depth[0] -= 1


def enc(self, *a, **kw):
    torch.cuda.synchronize(); t0 = time.time()
    out = orig_enc(self, *a, **kw)
    torch.cuda.synchronize(); acc("mem_encode", time.time() - t0)
    return out


def ov(*a, **kw):
    torch.cuda.synchronize(); t0 = time.time()
    out = orig_ov(*a, **kw)
    torch.cuda.synchronize(); acc("reproj_overlap", time.time() - t0)
    return out


Wm.GeoMemStar.decode_view = decode_view
Wm.GeoMemStar.encode_memory_from_prediction = enc
R.reproj_overlap = ov

scenes = sys.argv[1:] or ["Fencing", "Blocks", "MATF", "Breakfast"]
s1 = S.Stage1("geoset_rep")
t0 = time.time(); s1.load(); print(f"model load {time.time() - t0:.1f} s", flush=True)
res = {}
for sc in scenes:
    T.clear()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = s1.run(sc)
    total = time.time() - t0
    cfg = r["config"]
    res[sc] = dict(total=total, peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20, n_targets=cfg["n_targets"],
                   n_objects=cfg["n_objects"], mem_per_view=cfg["mem_per_view_mean"], **dict(T))
    print(sc, json.dumps(res[sc]), flush=True)
s1.close()
print("RESULT", json.dumps(res))
