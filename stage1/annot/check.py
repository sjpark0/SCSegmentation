#!/usr/bin/env python3
"""CROOM annotation checks and the hash manifest (docs/croom-annotation.md §3).

    python stage1/annot/check.py [--root stage1/Data/_annot/CROOM] [--manifest stage1/annot/croom_sha256.json]

Per instant: cameras without a label file, objects per camera (ids, areas), cameras per object, empty label maps,
log counts (sam / save / strokes).  With --manifest, writes the sha256 of every labels/*.png, objects.json and
index.json (commit it before any decoding of these instants).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(HERE)
ROOT = os.path.join(S1, "Data", "_annot", "CROOM")
sys.path.insert(0, S1)
from annot.labels import is_2d_png, read_label  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_instant(d):
    cams = sorted((f[:-4] for f in os.listdir(os.path.join(d, "images")) if f.endswith(".jpg")), key=lambda c: (len(c), c))
    ad = os.path.join(d, "annot")
    objs = json.load(open(os.path.join(ad, "objects.json"))) if os.path.isfile(os.path.join(ad, "objects.json")) else {}
    per_cam, missing, empty = {}, [], []
    for c in cams:
        p = os.path.join(ad, "labels", f"{c}.png")
        if not os.path.isfile(p):
            missing.append(c)
            continue
        if not is_2d_png(p):
            raise SystemExit(f"{p}: not a 2-D label PNG (run annot/fix_rgba.py)")
        lab = read_label(p)
        ids = {int(i): int((lab == i).sum()) for i in np.unique(lab) if i}
        per_cam[c] = ids
        if not ids:
            empty.append(c)
    cams_per_obj = {o: sum(1 for c in per_cam if int(o) in per_cam[c]) for o in objs}
    log = dict(sam=0, save=0, strokes=0, not_visible=0)
    lp = os.path.join(ad, "log.jsonl")
    if os.path.isfile(lp):
        for line in open(lp):
            ev = json.loads(line)
            if ev["ev"] in log:
                log[ev["ev"]] += 1
            if ev["ev"] == "save":
                log["strokes"] += int(ev.get("strokes") or 0)
    return dict(n_cams=len(cams), labelled=len(per_cam), missing=missing, empty=empty, objects=objs, cams_per_object=cams_per_obj,
                per_cam=per_cam, log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    manifest = {}
    for take in sorted(os.listdir(a.root)):
        td = os.path.join(a.root, take)
        if not os.path.isdir(td):
            continue
        for fr in sorted(os.listdir(td)):
            d = os.path.join(td, fr)
            if not os.path.isdir(os.path.join(d, "images")):
                continue
            r = check_instant(d)
            print(f"== {take} #{int(fr)}: {r['labelled']}/{r['n_cams']} cameras labelled, missing {r['missing']}, empty {r['empty']}")
            for o, meta in r["objects"].items():
                print(f"   obj {o:>2s} {meta['name']:24s} in {r['cams_per_object'][o]:2d} cameras")
            print(f"   log: {r['log']}")
            if a.manifest:
                for name in ("objects.json", "index.json"):
                    p = os.path.join(d, "annot" if name == "objects.json" else "", name)
                    if os.path.isfile(p):
                        manifest[os.path.relpath(p, a.root)] = sha256(p)
                for c in r["per_cam"]:
                    p = os.path.join(d, "annot", "labels", f"{c}.png")
                    manifest[os.path.relpath(p, a.root)] = sha256(p)
    if a.manifest:
        with open(a.manifest, "w") as f:
            json.dump(dict(root=os.path.relpath(a.root, os.path.dirname(S1)), files=manifest), f, indent=1)
        print(f"manifest: {len(manifest)} files -> {a.manifest}")


if __name__ == "__main__":
    main()
