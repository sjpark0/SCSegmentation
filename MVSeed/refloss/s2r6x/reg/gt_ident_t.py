# GT-only: per frame index t, share of object-frames whose GT mask equals GT_{t-1} (consecutive) and GT_0.
import json, os, numpy as np
from PIL import Image
R='/home/sjpark/Documents/SCSegmentation'
sets=json.load(open(f'{R}/docs/raw/muvod_object_sets.json'))
eq0=np.zeros(21); eqp=np.zeros(21); n=np.zeros(21)
for s,v in sets.items():
    if not isinstance(v,dict): continue
    for cam in v['cameras']:
        d=f'{R}/Data/MVSeg/{s}/Mask/{cam}'
        fs=sorted(os.listdir(d))
        L=[np.array(Image.open(os.path.join(d,f))) for f in fs]
        if L[0].ndim==3: L=[x[...,0].astype(np.int64)*65536+x[...,1].astype(np.int64)*256+x[...,2] for x in L]
        for o in [o for o in np.unique(L[0]) if o!=0]:
            M=[x==o for x in L]
            for t in range(1,len(L)):
                n[t]+=1; eq0[t]+=np.array_equal(M[t],M[0]); eqp[t]+=np.array_equal(M[t],M[t-1])
print('t  eq0   eqprev')
for t in range(1,21): print(t, round(eq0[t]/n[t],3), round(eqp[t]/n[t],3))
