import json, collections, os, statistics
REPO = "/home/sjpark/Documents/SCSegmentation"
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")))
OBJ = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
SCENES = sorted(OBJ)
A = {(r["scene"], r["cam"], r["obj"], r["t"]): r for r in json.load(open(os.path.join(HERE, "anat2_H.json")))}
S = {(r["scene"], r["cam"], r["obj"], r["t"]): r for r in json.load(open(os.path.join(HERE, "snap2_H.json")))}
P = collections.defaultdict(list)
for k, r in A.items(): P[k[:3]].append(r)
cams = collections.defaultdict(set); objs = collections.defaultdict(set)
for (s, c, o) in P: cams[s].add(c); objs[(s, c)].add(o)
NT = {k: len(v) for k, v in P.items()}
W = {k: 100 / (17 * len(cams[k[0]]) * len(objs[(k[0], k[1])])) / NT[k] for k in P}   # benchmark points per object-frame
def jf(r, a="J", b="F"): return 0.5 * (r[a] + r[b])
def snap(f, tau=0.8, d=4): return f["iou_p_g0"] >= tau and f["imgdiff"] <= d
def manifest_snaps(method, provenance=False):
    out = set()
    for sc in SCENES:
        m = json.load(open(f"{REPO}/Data/MVSeg/{CFG[sc]['folder']}/{method}/MANIFEST.json"))
        st = m["provenance"]["ref_snap"] if provenance else m["stats"]
        for x in st["snapped_files"]:
            c, t, o = x.split("/"); out.add((sc, c, int(o.split(".")[0]), int(t) - CFG[sc]["start_frame"]))
    return out
VI = {}
for _r in json.load(open(f"{REPO}/Data/MVSeg/jf_e14.json")): VI.setdefault(_r["dataset"], {})[_r["camera"]] = _r["meta"]["view_index"]
def dist(sc, cam): return VI[sc][cam] - VI[sc][OBJ[sc]["c_ini"]]
def jfrows(fname, method):
    """{(scene, cam, obj): (J list, F list, gt list)} for basic objects."""
    out = {}
    for r in json.load(open(f"{REPO}/Data/MVSeg/{fname}")):
        if r.get("method") != method or not r.get("result"): continue
        sc = r["dataset"]; basic = set(OBJ[sc]["seed_ids_c_ini"]); pf = r["result"]["per_frame"]
        for i, o in enumerate(r["objects"]):
            if o in basic: out[(sc, r["camera"], o)] = (pf["J"][i], pf["F"][i], pf["gt_area"][i])
    return out
def pair_w(rows):
    """benchmark weight in points per pair (MUVOD basic aggregation) and per object-frame."""
    cams = collections.defaultdict(set); objs = collections.defaultdict(set)
    for (s, c, o) in rows: cams[s].add(c); objs[(s, c)].add(o)
    n = len(cams)
    return {k: 100.0 / (n * len(cams[k[0]]) * len(objs[(k[0], k[1])])) for k in rows}
def pjf(J, F, t): return None if J[t] is None or F[t] is None else 0.5 * (J[t] + F[t])
