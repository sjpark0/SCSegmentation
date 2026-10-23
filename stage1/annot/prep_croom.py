#!/usr/bin/env python3
"""Copy the chosen CROOM instants (one frame of every camera) from the NAS into the annotation root (host only; the NAS
is a CIFS mount).  The frame choice is the rule of docs/croom-annotation.md; this script just copies what it is told.

    python stage1/annot/prep_croom.py --take T03_walking_rsp_out --frames 600 1800 [--root stage1/Data/_annot/CROOM]
    python stage1/annot/prep_croom.py --list                      # takes, camera counts, frame counts

Output: <root>/<take>/<frame:06d>/images/<cam>.jpg (cam = the NAS folder name, "0".."31") + index.json.
"""
import argparse
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(HERE)
MAIN = os.path.dirname(S1)
NAS = os.path.join(MAIN, "PlenopticServer1", "CROOM", "JPG")
ROOT = os.path.join(S1, "Data", "_annot", "CROOM")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take")
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for t in sorted(os.listdir(NAS)):
            cams = sorted(os.listdir(os.path.join(NAS, t)), key=lambda c: (len(c), c))
            n = len(os.listdir(os.path.join(NAS, t, cams[0]))) if cams else 0
            print(f"{t:28s} {len(cams):2d} cams  {n:5d} frames (cam {cams[0]})")
        return
    assert a.take and a.frames, "--take and --frames"
    cams = sorted(os.listdir(os.path.join(NAS, a.take)), key=lambda c: (len(c), c))
    for fr in a.frames:
        d = os.path.join(a.root, a.take, f"{fr:06d}", "images")
        os.makedirs(d, exist_ok=True)
        rec = {}
        for c in cams:
            src = os.path.join(NAS, a.take, c, f"{fr}.jpg")
            dst = os.path.join(d, f"{c}.jpg")
            if not os.path.isfile(dst):
                shutil.copyfile(src, dst)
            rec[c] = sha256(dst)
        with open(os.path.join(a.root, a.take, f"{fr:06d}", "index.json"), "w") as f:
            json.dump(dict(take=a.take, frame=fr, source=os.path.relpath(os.path.join(NAS, a.take), MAIN), cams=cams,
                           image_sha256=rec), f, indent=1)
        print(f"{a.take} frame {fr}: {len(cams)} images -> {d}")


if __name__ == "__main__":
    main()
