import json, sys
S = sys.argv[1]; ds = sys.argv[2]
H = "/home/sjpark/Documents/SCSegmentation/Data/MVSeg/%s/SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs/MANIFEST.json" % ds
def offl(p):
    s = set()
    for f in json.load(open(p))["provenance"]["ref_snap"]["snapped_files"]:
        c, t, o = f.split("/"); s.add((int(t), int(o.split(".")[0])))
    return s
h = offl(H); m = offl(S + "/data/%s/SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs_probeM1/MANIFEST.json" % ds)
r = json.load(open(S + "/probe_%s_inject_mem.json" % ds))
on = {(fr["t"], int(o)) for fr in r["frames"] for o, x in fr["objs"].items() if x.get("snap")}
print("headline offline", len(h), "| M1 online", len(on), "| M1 offline", len(m))
print("M1 online - headline offline", sorted(on - h)); print("headline offline - M1 online", sorted(h - on))
print("M1 offline vs M1 online diff", sorted(m ^ on))
tf = [(fr["t"], int(o), x["iou"], x["diff"]) for fr in r["frames"] for o, x in fr["objs"].items() if x.get("diff") is not None and x["diff"] <= 4 and not x["snap"]]
print("M1 delta-pass tau-fail:", tf)
tm = [fr["t_mem_s"] for fr in r["frames"]]; td = [fr["t_decide_s"] + fr["t_read_s"] for fr in r["frames"]]
print("per frame read+decide %.2f s, mem probe %.2f s, enc %.2f ms; total %s" % (sum(td)/len(td), sum(tm)/len(tm), sum(fr["enc_ms_mean"] for fr in r["frames"])/len(r["frames"]), r["timing"]))
print("reencode equal", sum(x.get("reencode_equal", False) for fr in r["frames"] for x in fr["objs"].values()), "of", sum("reencode_equal" in x for fr in r["frames"] for x in fr["objs"].values()))
