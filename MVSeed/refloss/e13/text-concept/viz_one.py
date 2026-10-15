"""Diagnostic viz of one dev view/object: image | GT_o | D_o | cls (vis/out/occ/nodepth) | union of name detections | each
of the top detections.  DEV only.  usage: viz_one.py scene t cam obj out.jpg"""
import json
import os
import sys

import cv2
import numpy as np

S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
from mvseg2d.io import load_labelmap, read_pred_mask  # noqa: E402

scene, t, cam, obj, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
R = f"{S1}/geomem/runs"
ddir = f"{R}/_e13_text-concept/dets/{scene}/{t:06d}/{cam}"
pdir = f"{R}/_e13_sam3-prompt-extend/props/{scene}/{t:06d}/{cam}"
meta = json.load(open(f"{ddir}/meta.json"))
H, W = meta["hw"]
img = cv2.imread(f"{S1}/Data/{scene}/frames/{t:06d}/images/{cam}.jpg")
gt = load_labelmap(f"{S1}/Data/{scene}/frames/{t:06d}/masks/{cam}.png")
G = gt == obj
p = f"{R}/_e13_dev/{scene}/{t:06d}/{cam}/{obj}.png"
D = read_pred_mask(p) if os.path.isfile(p) else np.zeros((H, W), bool)
cls = cv2.imread(f"{pdir}/cls.png", cv2.IMREAD_UNCHANGED)
pp = f"{pdir}/proj/{obj}.png"
P = read_pred_mask(pp) if os.path.isfile(pp) else np.zeros((H, W), bool)
z = np.load(f"{ddir}/dets.npz")
sid = {s: i for i, s in enumerate(meta["strings"])}
ids = {sid[s] for s in meta["obj_prompts"][str(obj)]["split"]}
ks = [k for k in range(len(z["prob"])) if int(z["prompt"][k]) in ids and z["prob"][k] > 0.5]
M = {k: np.unpackbits(z["packed"][k], count=H * W).reshape(H, W).astype(bool) for k in ks}
iou = {k: (M[k] & G).sum() / max(1, (M[k] | G).sum()) for k in ks}
ovp = {k: int((M[k] & P).sum()) for k in ks}
U = np.zeros((H, W), bool)
for k in ks:
    U |= M[k]
sc = 6
small = cv2.resize(img, (W // sc, H // sc))


def ov(m, col=(0, 0, 255)):
    o = small.copy()
    ms = cv2.resize(m.astype(np.uint8), (W // sc, H // sc), interpolation=cv2.INTER_NEAREST) > 0
    o[ms] = (0.4 * o[ms] + 0.6 * np.array(col)).astype(np.uint8)
    return o


cl = small.copy()
cs = cv2.resize(cls, (W // sc, H // sc), interpolation=cv2.INTER_NEAREST)
for c, col in ((1, (255, 0, 0)), (2, (0, 255, 255)), (3, (255, 0, 255))):
    cl[cs == c] = (0.4 * cl[cs == c] + 0.6 * np.array(col)).astype(np.uint8)
tiles = [small, ov(G), ov(D, (0, 255, 0)), cl, ov(P, (255, 255, 0)), ov(U, (0, 128, 255))]
top = sorted(ks, key=lambda k: -iou[k])[:6]
for k in top:
    tiles.append(ov(M[k], (0, 128, 255)))
    cv2.putText(tiles[-1], f"k{k} p{z['prob'][k]:.2f} iouGT{iou[k]:.2f} ovP{ovp[k]}", (5, 20), 0, 0.5, (255, 255, 255), 1)
while len(tiles) % 4:
    tiles.append(np.zeros_like(small))
rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
cv2.imwrite(out, np.concatenate(rows, 0))
print(len(ks), "dets;", {k: (round(float(z['prob'][k]), 2), round(float(iou[k]), 3), ovp[k], int(M[k].sum())) for k in top})
print("GT", int(G.sum()), "D", int(D.sum()), "P", int(P.sum()), "U&miss", int((U & G & ~D).sum()), "miss", int((G & ~D).sum()))
