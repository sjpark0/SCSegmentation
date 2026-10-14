"""scsam3 container (stage1/scripts/docker_run.sh scsam3 python <this>): how far the S1-E9 fine-tuning moved the
weights.  For every fold checkpoint of the three arms, the relative Frobenius distance ||W_ft - W_0|| / ||W_0|| of the
mask decoder (trained in D1/D2, conv_s0/s1 excluded) and of the memory attention (trained in all three arms), overall
and the largest per-tensor value -> MVSeed/runs/e9/decmove.json.  W_0 = the SAM 3 weights the training started from."""
import json
import os
import sys

import torch

S1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "stage1"))
sys.path.insert(0, S1)
from geomem.wrapper import DECODER_FROZEN  # noqa: E402
from methods import sam3_single  # noqa: E402

ARMS = ("geoset_pred_ft", "geoset_dec_ft", "geoset_decb_ft")
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "e9", "decmove.json")


def rel_move(init, ft, skip=()):
    num = den = 0.0
    worst = ("", 0.0)
    for n, w0 in init.items():
        if n.startswith(skip) or not torch.is_floating_point(w0):
            continue
        d = (ft[n].float().cpu() - w0.float().cpu())
        dn, wn = float(d.norm()), float(w0.float().norm())
        num += dn ** 2
        den += wn ** 2
        r = dn / max(wn, 1e-12)
        if r > worst[1]:
            worst = (n, r)
    return {"rel": (num / max(den, 1e-24)) ** 0.5, "worst_tensor": worst[0], "worst_rel": worst[1]}


def main():
    tracker = sam3_single.load_model().tracker          # same modules as methods/geomem/run.INIT_MODULES
    dec0 = {k: v.detach().clone() for k, v in tracker.sam_mask_decoder.state_dict().items()}
    att0 = {k: v.detach().clone() for k, v in tracker.transformer.encoder.state_dict().items()}
    out = {}
    for arm in ARMS:
        for k in range(5):
            p = os.path.join(S1, "geomem", "runs", arm, f"fold{k}", "ckpt.pt")
            if not os.path.exists(p):
                continue
            ck = torch.load(p, map_location="cpu")
            row = {"memory_attention": rel_move(att0, ck["memory_attention"])}
            if "mask_decoder" in ck:
                row["mask_decoder"] = rel_move(dec0, ck["mask_decoder"], skip=DECODER_FROZEN)
                row["conv_s_unchanged"] = all(torch.equal(ck["mask_decoder"][n].cpu(), dec0[n].cpu())
                                              for n in dec0 if n.startswith(DECODER_FROZEN))
            out[f"{arm}/fold{k}"] = row
            print(arm, f"fold{k}", json.dumps(row))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("->", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
