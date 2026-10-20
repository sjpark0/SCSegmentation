import sys, json, os
sys.path.insert(0, '/home/sjpark/Documents/SCSegmentation/stage1')
import numpy as np
from PIL import Image
from geomem import devset
from geomem.pointmap import CONF_MIN
D = '/home/sjpark/Documents/SCSegmentation/stage1/Data'
for s in ('CBABasketball', 'Blocks', 'Fencing', 'Dog', 'CoffeeMartini'):
    m = json.load(open(f'{D}/{s}/meta.json')); ref = m['ref_cam']
    t = devset.dev_frames(D, s, "A")[0]
    C = f'{D}/{s}/Cache/pi3_frames/{t:06d}'
    z = np.load(f'{C}/{ref}.npz'); v = (z['conf'] > CONF_MIN) & (z['depth'] > 0)
    lab = np.array(Image.open(f'{D}/{s}/frames/{t:06d}/masks/{ref}.png'))
    h, w = v.shape; labs = np.array(Image.fromarray(lab).resize((w, h), Image.NEAREST))
    basic = devset.basic_set(s)
    u = np.isin(labs, basic) & (labs > 0)
    print(s, t, 'union/valid', round(float((u & v).sum() / v.sum()), 3), 'labelled frac', round(float((labs > 0).mean()), 3), 'n basic', len(basic))
