#!/usr/bin/env python3
"""Renumber COLMAP image IDs to 1..N (text model in -> text model out).

`colmap image_undistorter` keeps the original image IDs, so a partially registered scene ends up
with a sparse ID set (e.g. 8 images numbered up to 10).  LLFF's imgs2poses.py assumes IDs are
1..N (`cams[ind-1] = 1` over a list sized by the image count) and raises IndexError otherwise.
Only the IDs change; poses, intrinsics, points and tracks are untouched.
"""
import os
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
shutil.copyfile(os.path.join(src, "cameras.txt"), os.path.join(dst, "cameras.txt"))

lines = open(os.path.join(src, "images.txt")).read().splitlines()
body = [l for l in lines if l.strip() and not l.startswith("#")]
head = [l for l in lines if l.startswith("#")]
firsts = body[0::2]
old_ids = [int(l.split()[0]) for l in firsts]
remap = {o: i + 1 for i, o in enumerate(sorted(old_ids))}
out = list(head)
for f, pts in zip(firsts, body[1::2]):
    t = f.split()
    t[0] = str(remap[int(t[0])])
    out += [" ".join(t), pts]
open(os.path.join(dst, "images.txt"), "w").write("\n".join(out) + "\n")

out = []
for l in open(os.path.join(src, "points3D.txt")):
    if l.startswith("#") or not l.strip():
        out.append(l.rstrip("\n"))
        continue
    t = l.split()
    for i in range(8, len(t), 2):
        t[i] = str(remap[int(t[i])])
    out.append(" ".join(t))
open(os.path.join(dst, "points3D.txt"), "w").write("\n".join(out) + "\n")
print(f"renumbered {len(old_ids)} images: max id {max(old_ids)} -> {len(old_ids)}")
