#!/usr/bin/env python3
"""Host: aggregate the per-frame _logits/meta.json checks of the S1-E14 dump -> <root>/dump_meta.json."""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)
import e14dump as E  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(S1, "geomem", "runs", "_e14_logitdump", "geoset_rep_x_A")
pairs = E.dev_pairs(os.path.join(S1, "geomem", "runs", "_e13_dev", "dev_pairs.json"))
idx = json.load(open(os.path.join(root, "index.json")))
expected = {(s, t) for s, v in idx["scenes"].items() for t in v["frames"]}
metas = sorted(glob.glob(os.path.join(root, "*", "*", "_logits", "meta.json")))
done = {(p.split(os.sep)[-4], int(p.split(os.sep)[-3])) for p in metas
        if os.path.isfile(os.path.join(os.path.dirname(p), "DONE"))}
agg = dict(frames_expected=len(expected), frames_done=len(done), frames_missing=sorted(map(list, expected - done)),
           views=0, object_views=0, n_absent=0, n_fallback=0, fp16_inexact_low_res=0, fp16_inexact_mm=0,
           views_recon_low_res_equal=0, views_recon_ious_equal=0, views_recon_presence_equal=0, views_bf16_valued=0,
           obj_rethr_vs_ref_identical=0, obj_rethr_vs_mem_identical=0, obj_mem_vs_ref_identical=0,
           obj_own_png_vs_ref_identical=0, obj_png_bytes_equal=0,
           pairs_total=len(pairs), pairs_found=0, pairs_rethr_vs_ref_identical=0, pairs_png_bytes_equal=0,
           npz_bytes=0, mm_npz_bytes=0, dump_seconds=0.0, low_res_dtype=set(), mm_dtype=set(), env=None)
bad = []
for p in metas:
    m = json.load(open(p))
    agg["env"] = m["env"]
    agg["dump_seconds"] += m["dump_seconds"]
    for cam, v in m["views"].items():
        agg["views"] += 1
        agg["object_views"] += v["B"]
        for k in ("n_absent", "n_fallback", "fp16_inexact_low_res", "fp16_inexact_mm", "npz_bytes", "mm_npz_bytes"):
            agg[k] += v[k]
        agg["views_recon_low_res_equal"] += v["recon_low_res_equal"]
        agg["views_recon_ious_equal"] += v["recon_ious_equal"]
        agg["views_recon_presence_equal"] += v["recon_presence_equal"]
        agg["views_bf16_valued"] += v["bf16_valued_low_res"]
        agg["low_res_dtype"].add(v["low_res_dtype"])
        agg["mm_dtype"].add(v["mm_dtype"])
        for o, r in v["objects"].items():
            ok = r["rethr_vs_ref_px"] == 0
            agg["obj_rethr_vs_ref_identical"] += ok
            agg["obj_rethr_vs_mem_identical"] += r["rethr_vs_mem_px"] == 0
            agg["obj_mem_vs_ref_identical"] += r["mem_vs_ref_px"] == 0
            agg["obj_own_png_vs_ref_identical"] += r["own_png_vs_ref_px"] == 0
            agg["obj_png_bytes_equal"] += r["png_bytes_equal"]
            if (m["scene"], m["t"], cam, int(o)) in pairs:
                agg["pairs_found"] += 1
                agg["pairs_rethr_vs_ref_identical"] += ok
                agg["pairs_png_bytes_equal"] += r["png_bytes_equal"]
            if not ok or not r["png_bytes_equal"] or r["rethr_vs_mem_px"] or r["mem_vs_ref_px"]:
                bad.append(dict(scene=m["scene"], t=m["t"], cam=cam, obj=int(o), **r))
agg["low_res_dtype"], agg["mm_dtype"] = sorted(agg["low_res_dtype"]), sorted(agg["mm_dtype"])
agg["bad"] = bad[:100]
du = subprocess.run(["du", "-sb", root], capture_output=True, text=True).stdout.split()
agg["root_bytes"] = int(du[0]) if du else None
with open(os.path.join(root, "dump_meta.json"), "w") as f:
    json.dump(agg, f, indent=1)
print(json.dumps({k: v for k, v in agg.items() if k != "bad"}, indent=1))
print("bad", len(bad), bad[:5])
