import json, sys, statistics as st, os
r = json.load(open(sys.argv[1]))
man = json.load(open(sys.argv[2]))["provenance"]["ref_snap"]
off = set()
for f in man["snapped_files"]:
    cam, t, o = f.split("/"); off.add((int(t), int(o.split(".")[0])))
on = set(); n = 0; eq = []; mx = []; keys = {}
for fr in r["frames"]:
    for o, rec in fr["objs"].items():
        n += 1
        if rec.get("snap"): on.add((fr["t"], int(o)))
        if "reencode_equal" in rec:
            eq.append(rec["reencode_equal"]); mx.append(rec["reencode_maxabs"])
        for k in ("rel_vs_prev_frame", "rel_m0bin_vs_stored", "rel_m0rt_vs_stored", "rel_m0bin_vs_m0rt",
                  "rel_cond0_vs_stored", "rel_cond0_vs_m0rt", "cos_ptr_m0_vs_stored", "cos_ptr_cond0_vs_stored", "cos_ptr_vs_prev", "eff_iou_score"):
            if rec.get(k) is not None:
                keys.setdefault((k, bool(rec.get("snap"))), []).append(rec[k])
print(f"object-frames {n}; online snaps (mainline mask_t) {len(on)}; offline snaps (final PNG) {len(off)}; both {len(on & off)}; online-only {sorted(on - off)[:12]}..{len(on-off)}; offline-only {sorted(off - on)[:12]}..{len(off-on)}")
print(f"re-encode of stored mask reproduces stored maskmem_features: {sum(eq)}/{len(eq)} equal; max|diff| {max(mx) if mx else None}")
for (k, s), v in sorted(keys.items()):
    v = sorted(v)
    print(f"{k:26s} snap={s!s:5s} n={len(v):3d} median {st.median(v):.4f}  p10 {v[len(v)//10]:.4f}  p90 {v[(9*len(v))//10]:.4f}  min {v[0]:.4f} max {v[-1]:.4f}")
tm = [fr["t_mem_s"] for fr in r["frames"]]; td = [fr["t_decide_s"] + fr["t_read_s"] for fr in r["frames"]]; te = [fr["enc_ms_mean"] for fr in r["frames"]]
print(f"per frame: read+decide {st.mean(td):.3f} s, memory probe {st.mean(tm):.3f} s (all encodes+ptr for all objs), one re-encode {st.mean(te):.2f} ms; total {r['timing']}")
