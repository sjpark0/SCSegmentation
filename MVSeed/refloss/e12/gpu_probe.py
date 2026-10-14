"""Judge probe (scsam3, GPU, read-only): (1) is a geoset_pred_ft training step bit-reproducible in-process?
(2) does a per-object per-cell -inf on the reference / predicted view, routed through MemorySet.gate_mask, reach the
patched SDPA and change only the dropped object?  (3) cost of a materialised per-object mask on a 1-memory step.
Run from stage1/:  scripts/docker_run.sh scsam3 python -c "$(cat gpu_probe.py)"
"""
import json
import sys
import time

import numpy as np
import torch

from geomem import wrapper as Wm
from geomem.data import Sample, choose_objects, load_sample, objects_for
from geomem.train import build_wrapper, train_one_step
from methods import sam3_single

DATA = "Data"
res = {}
torch.manual_seed(0)
rng = np.random.default_rng(0)
model = sam3_single.load_model()
w = build_wrapper(model, "geoset_pred_ft")
w.freeze("geoset_pred_ft")
ck = torch.load("geomem/runs/geoset_pred_ft/fold0/ckpt.pt", map_location="cuda")   # fold0 held Blocks out
w.tracker.transformer.encoder.load_state_dict(ck["memory_attention"])
w.geo_bias.load_state_dict(ck["geo_bias"])
s = Sample("Blocks", 0, "cam4", "cam0")
objs = choose_objects(objects_for(s, DATA), 16, rng)
b2 = load_sample(DATA, s, objs, "cuda", extra_mem_cam="cam2", extra_mem_gt=False)
b1 = load_sample(DATA, s, objs, "cuda")
B = b1["ref_masks"].shape[0]
res["B"] = int(B)

# ---- per-cell drop routed through gate_mask (monkeypatch; the designs' proposed field emulated)
orig = Wm.MemorySet.gate_mask
DROP = {}


def gm(self, dtype, device):
    m = orig(self, dtype, device)
    if not DROP:
        return m
    V, Bm = len(self.views), self.views[0].feats.shape[1]
    L = self.views[0].feats.shape[0]
    if m is None:
        m = torch.zeros(Bm, 1, 1, V * L + 4 * V, dtype=dtype, device=device)
    for i, d in DROP.items():
        if i < V:
            cols = torch.zeros(Bm, V * L + 4 * V, dtype=torch.bool, device=device)
            cols[:, i * L:(i + 1) * L] = d.to(device)
            m = m.masked_fill(cols[:, None, None, :], float("-inf"))
    return m


Wm.MemorySet.gate_mask = gm


def grads():
    return [p.grad.detach().clone() for p in w.trainable_parameters() if p.grad is not None]


def one_step(batch):
    for p in w.trainable_parameters():
        p.grad = None
    t = train_one_step(w, batch, "geoset_pred_ft", rng)
    return t, grads()


# (1) in-process repeatability of a training step (no optimizer step: weights unchanged)
for name, batch in (("1mem", b1), ("2mem_pred", b2)):
    DROP.clear()
    ta, ga = one_step(batch)
    tb, gb = one_step(batch)
    diff = [float((x.float() - y.float()).abs().max()) for x, y in zip(ga, gb)]
    res[f"repeat_{name}"] = dict(loss_equal=ta["loss"] == tb["loss"], loss=ta["loss"], n_mem=ta["n_mem"],
                                 terms_equal=all(ta[k] == tb[k] for k in ("focal", "dice", "iou", "presence")),
                                 grads_bitwise_equal=all(torch.equal(x, y) for x, y in zip(ga, gb)),
                                 n_grad_tensors=len(ga), max_abs_grad_diff=max(diff) if diff else None,
                                 n_tensors_differ=sum(d > 0 for d in diff))
    # materialised all-False drop (gate built, nothing dropped) vs the ungated path
    DROP[0] = torch.zeros(B, 5184, dtype=torch.bool)
    tz, gz = one_step(batch)
    res[f"zero_gate_vs_none_{name}"] = dict(loss_equal=tz["loss"] == ta["loss"], dloss=tz["loss"] - ta["loss"],
                                            grads_bitwise_equal=all(torch.equal(x, y) for x, y in zip(ga, gz)))
    DROP.clear()

# (2) forward-only drop effect, reference view and predicted view
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    ref = w.encode_reference(b2["ref_image"], b2["ref_backbone"], b2["ref_masks"], b2["p_ref"], b2["valid_ref"])
    DROP.clear()
    oA = w.decode_view(b2["tgt_backbone"], B, ref, b2["p_tgt"], b2["valid_tgt"])["logits_1008"].float()
    DROP[0] = torch.zeros(B, 5184, dtype=torch.bool)
    oZ = w.decode_view(b2["tgt_backbone"], B, ref, b2["p_tgt"], b2["valid_tgt"])["logits_1008"].float()
    d = torch.zeros(B, 72, 72, dtype=torch.bool)
    d[0, :, :36] = True                                           # object 0: left half-plane of the reference grid
    DROP[0] = d.reshape(B, 5184)
    oD = w.decode_view(b2["tgt_backbone"], B, ref, b2["p_tgt"], b2["valid_tgt"])["logits_1008"].float()
    dall = torch.zeros(B, 5184, dtype=torch.bool)
    dall[0] = True                                                # object 0: every feature column, pointers only
    DROP[0] = dall
    oP = w.decode_view(b2["tgt_backbone"], B, ref, b2["p_tgt"], b2["valid_tgt"])["logits_1008"].float()
    res["ref_drop"] = dict(
        none_vs_zero_gate_bitwise=bool(torch.equal(oA, oZ)), none_vs_zero_gate_maxdiff=float((oA - oZ).abs().max()),
        others_unchanged_bitwise=bool(torch.equal(oZ[1:], oD[1:])), others_maxdiff=float((oZ[1:] - oD[1:]).abs().max()),
        obj0_maxdiff=float((oZ[0] - oD[0]).abs().max()),
        obj0_sign_flips_frac=float(((oZ[0] > 0) != (oD[0] > 0)).float().mean()),
        obj0_fg_frac_before=float((oZ[0] > 0).float().mean()), obj0_fg_frac_after=float((oD[0] > 0).float().mean()),
        pointer_only_finite=bool(torch.isfinite(oP).all()), pointer_only_obj0_fg_frac=float((oP[0] > 0).float().mean()))
    # predicted second memory view (tau 0 so every present object is confident), drop on view 1 for object 1
    DROP.clear()
    pred = w.decode_view(b2["mem_backbone"], B, ref, b2["p_mem"], b2["valid_mem"])
    view = w.encode_memory_from_prediction(b2["mem_backbone"], B, pred["logits_1008"], pred["presence"], pred["obj_ptr"],
                                           b2["p_mem"], b2["valid_mem"], b2["mem_cam"], ious=pred["ious"], tau=0.0)
    mem = ref.with_view(view)
    DROP[1] = torch.zeros(B, 5184, dtype=torch.bool)
    p0 = w.decode_view(b2["tgt_backbone"], B, mem, b2["p_tgt"], b2["valid_tgt"])["logits_1008"].float()
    d1 = torch.zeros(B, 5184, dtype=torch.bool)
    d1[1, :2592] = True                                           # object 1: top half of the predicted view's grid
    DROP[1] = d1
    p1 = w.decode_view(b2["tgt_backbone"], B, mem, b2["p_tgt"], b2["valid_tgt"])["logits_1008"].float()
    keep = [i for i in range(B) if i != 1]
    res["pred_view_drop"] = dict(conf=[bool(x) for x in view.conf.cpu()],
                                 others_unchanged_bitwise=bool(torch.equal(p0[keep], p1[keep])),
                                 obj1_maxdiff=float((p0[1] - p1[1]).abs().max()))
    DROP.clear()

# (3) cost of a materialised per-object mask on a 1-memory training step
for label, drop in (("none", None), ("zero_gate", torch.zeros(B, 5184, dtype=torch.bool))):
    DROP.clear()
    if drop is not None:
        DROP[0] = drop
    one_step(b1)
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        t0 = time.time()
        one_step(b1)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    res[f"step_seconds_1mem_{label}"] = float(np.median(ts))
res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
print(json.dumps(res, indent=1))
