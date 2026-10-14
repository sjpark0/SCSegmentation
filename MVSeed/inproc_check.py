#!/usr/bin/env python3
"""In-process pipeline check (docs/pipeline-inprocess.md): does runMVSeg.py --stage1 geoset_rep ... --ref-snap
--save-stages reproduce the file route, stage by stage, byte for byte?

Per scene, the run <M>_ip (M = the headline SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs) against:
  stage1/final   <-> stage1/Data/<scene>/Result/geoset_pred_ft_rep_e8      PNG bytes (+ round1_conf.json)
  stage1/round1  <-> stage1/Data/<scene>/Result/geoset_pred_ft_r1_e8       PNG bytes
  seeds          <-> Data/MVSeg/<folder>/MVSeed_geoset_rep                  pixels (> 127), every (cam, obj)
  stage2         <-> Data/MVSeg/<folder>/SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2    PNG bytes
  final output   <-> Data/MVSeg/<folder>/SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs  PNG bytes
Host python (PIL).

    python3 MVSeed/inproc_check.py [--scenes Fencing ...] [--suffix _ip]   ->  MVSeed/runs/inproc_check.json
"""
import argparse, hashlib, json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data", "MVSeg")
S1 = os.path.join(REPO, "stage1", "Data")
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")))
CENSUS = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
HEAD = "SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2Rs"


def tree(d):
    """{relative path: sha256} of every PNG / JSON below d (top-level MANIFEST, RUN, config, round1_conf excluded: compared apart or volatile)."""
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), d)
            if rel in ("MANIFEST.json", "RUN.json", "config.json", "round1_conf.json", "zoom_conf.json", "xcl_conf.json") or not f.endswith((".png", ".json")):
                continue
            out[rel] = hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()
    return out


def cmp_bytes(a, b):
    ta, tb = tree(a), tree(b)
    same = sum(ta[k] == tb[k] for k in set(ta) & set(tb))
    return dict(identical=same, different=len(set(ta) & set(tb)) - same,
                only_new=len(set(ta) - set(tb)), only_old=len(set(tb) - set(ta)),
                ok=same == len(ta) == len(tb) and len(ta) > 0)


def cmp_pixels(a, b):
    from PIL import Image
    import numpy as np
    pa = {os.path.relpath(os.path.join(r, f), a) for r, _, fs in os.walk(a) for f in fs if f.endswith(".png")}
    pb = {os.path.relpath(os.path.join(r, f), b) for r, _, fs in os.walk(b) for f in fs if f.endswith(".png")}
    diff = [k for k in sorted(pa & pb)
            if not np.array_equal(np.asarray(Image.open(os.path.join(a, k)).convert("L")) > 127,
                                  np.asarray(Image.open(os.path.join(b, k)).convert("L")) > 127)]
    return dict(identical=len(pa & pb) - len(diff), different=len(diff), only_new=len(pa - pb), only_old=len(pb - pa),
                ok=not diff and pa == pb and len(pa) > 0)


def check(scene, suffix):
    f = CFG[scene]["folder"]
    run = os.path.join(DATA, f, HEAD + suffix)
    st = run + "_stages"
    r1 = os.path.join(S1, scene, "Result", "geoset_pred_ft_rep_e8")
    rows = dict(
        stage1_final=cmp_bytes(os.path.join(st, "stage1", "final"), r1),
        stage1_round1=cmp_bytes(os.path.join(st, "stage1", "round1"), os.path.join(S1, scene, "Result", "geoset_pred_ft_r1_e8")),
        seeds=cmp_pixels(os.path.join(st, "seeds"), os.path.join(DATA, f, "MVSeed_geoset_rep")),
        stage2=cmp_bytes(os.path.join(st, "stage2"), os.path.join(DATA, f, HEAD[:-2])),
        final=cmp_bytes(run, os.path.join(DATA, f, HEAD)))
    a, b = os.path.join(st, "stage1", "round1_conf.json"), os.path.join(r1, "round1_conf.json")
    rows["round1_conf"] = dict(ok=open(a, "rb").read() == open(b, "rb").read())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=sorted(CENSUS))
    ap.add_argument("--suffix", default="_ip")
    a = ap.parse_args()
    out, all_ok = {}, True
    for s in a.scenes:
        rows = check(s, a.suffix)
        ok = all(r["ok"] for r in rows.values())
        all_ok &= ok
        out[s] = rows
        print(f"{s:20s} " + "  ".join(f"{k} {r.get('identical', '')}{'' if r['ok'] else ' FAIL ' + str(r)}"
                                      for k, r in rows.items()) + ("  -> OK" if ok else "  -> FAIL"), flush=True)
    out["all_ok"] = all_ok
    json.dump(out, open(os.path.join(REPO, "MVSeed", "runs", "inproc_check.json"), "w"), indent=1)
    print("all identical:", all_ok)


if __name__ == "__main__":
    main()
