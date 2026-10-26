#!/usr/bin/env python3
"""One-time repair (2026-10-26): rewrite RGBA label PNGs stored by the annotation server before the as_label fix as
2-D uint8 label maps (channel 0; the colour channels are checked to agree, the alpha channel held the first saved id
and is dropped).  Every rewrite is appended to the instant's log.jsonl (ev "fix_rgba", sha256 before / after).

    python annot/fix_rgba.py [--root Data/_annot/CROOM] [--dry]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(HERE)
sys.path.insert(0, S1)
from annot.labels import as_label, sha256  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(S1, "Data", "_annot", "CROOM"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    n_fixed = n_ok = 0
    for take in sorted(os.listdir(a.root)):
        for fr in sorted(os.listdir(os.path.join(a.root, take))):
            ld = os.path.join(a.root, take, fr, "annot", "labels")
            if not os.path.isdir(ld):
                continue
            for f in sorted(os.listdir(ld)):
                p = os.path.join(ld, f)
                im = Image.open(p)
                if im.mode == "L":
                    n_ok += 1
                    continue
                raw = np.array(im)
                lab = as_label(raw)
                before = sha256(p)
                ids = {int(i): int((lab == i).sum()) for i in np.unique(lab) if i}
                print(f"{take}/{fr}/{f}: {im.mode} {raw.shape} -> L {lab.shape} ids {ids}")
                if a.dry:
                    continue
                Image.fromarray(lab).save(p, compress_level=3)
                assert Image.open(p).mode == "L" and (np.array(Image.open(p)) == lab).all()
                with open(os.path.join(a.root, take, fr, "annot", "log.jsonl"), "a") as g:
                    g.write(json.dumps(dict(t=time.time(), ev="fix_rgba", cam=f[:-4], mode=im.mode, sha256_before=before,
                                            sha256_after=sha256(p))) + "\n")
                n_fixed += 1
    print(f"fixed {n_fixed}, already 2-D {n_ok}")


if __name__ == "__main__":
    main()
