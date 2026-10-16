# GT-only annotation statistic: share of (object, frame>=1) whose GT mask is pixel-identical to its frame-0 GT mask.
import json, os, numpy as np
from PIL import Image
R='/home/sjpark/Documents/SCSegmentation'
sets=json.load(open(f'{R}/docs/raw/muvod_object_sets.json'))
out={}
tot={'ref':[0,0],'non':[0,0]}
for s,v in sets.items():
    if not isinstance(v,dict): continue
    for cam in v['cameras']:
        d=f'{R}/Data/MVSeg/{s}/Mask/{cam}'
        fs=sorted(os.listdir(d))
        L=[np.array(Image.open(os.path.join(d,f))) for f in fs]
        if L[0].ndim==3: L=[x[...,0].astype(np.int64)*65536+x[...,1].astype(np.int64)*256+x[...,2] for x in L]
        ids=[o for o in np.unique(L[0]) if o!=0]
        n=same=0
        for o in ids:
            m0=L[0]==o
            for t in range(1,len(L)):
                n+=1; same+= bool(np.array_equal(L[t]==o,m0))
        k='ref' if cam==v['c_ini'] else 'non'
        tot[k][0]+=same; tot[k][1]+=n
        out[f'{s}/{cam}']=(same,n,k)
for k,(a,b) in tot.items(): print(k,a,b,round(a/b,3))
json.dump(out,open(os.path.join(os.path.dirname(__file__),'gt_ident.json'),'w'),indent=0)
