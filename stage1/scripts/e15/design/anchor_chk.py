import sys, json
sys.path.insert(0, '/home/sjpark/Documents/SCSegmentation/stage1')
import numpy as np
from PIL import Image
from geomem.pointmap import CONF_MIN, local_points
D = '/home/sjpark/Documents/SCSegmentation/stage1/Data'
for s, t in (('CBABasketball', 94), ('Blocks', 8)):
    m = json.load(open(f'{D}/{s}/meta.json')); ref = m['ref_cam']
    C = f'{D}/{s}/Cache/pi3_frames/{t:06d}'
    z = np.load(f'{C}/{ref}.npz'); T = z['c2w']; v = (z['conf'] > CONF_MIN) & (z['depth'] > 0)
    P = local_points(z['depth'], z['K'])
    lab = np.array(Image.open(f'{D}/{s}/frames/{t:06d}/masks/{ref}.png'))
    h, w = v.shape; labs = np.array(Image.fromarray(lab).resize((w, h), Image.NEAREST))
    Xs = T[:3, :3] @ np.median(P[v], 0) + T[:3, 3]; Xo = T[:3, :3] @ np.median(P[(labs > 0) & v], 0) + T[:3, 3]
    cs = {c: np.load(f'{C}/{c}.npz')['c2w'][:3, 3] for c in m['cams']}
    print(s, t, '|Xs-Xo|', round(float(np.linalg.norm(Xs - Xo)), 4), 'ref->Xs', round(float(np.linalg.norm(T[:3, 3] - Xs)), 4),
          'ref->Xo', round(float(np.linalg.norm(T[:3, 3] - Xo)), 4), 'max cam spread', round(float(max(np.linalg.norm(cs[c] - T[:3, 3]) for c in cs)), 4))
    def ang(X, c):
        u = T[:3, 3] - X; w_ = cs[c] - X
        return float(np.degrees(np.arccos(np.clip(u @ w_ / np.linalg.norm(u) / np.linalg.norm(w_), -1, 1))))
    for c in sorted(cs, key=lambda c: -ang(Xs, c))[:2]:
        print('   ', c, round(ang(Xs, c), 3), round(ang(Xo, c), 3))
