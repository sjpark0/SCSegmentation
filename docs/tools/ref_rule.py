import json, os
import numpy as np, cv2
ROOT='/home/sjpark/Documents/SCSegmentation/Data/MVSeg'
CFG=json.load(open('/home/sjpark/Documents/SCSegmentation/SCSam3/demo/MVSeg.json'))
DS=['AlexaMeadeFacePaint','Barn','Blocks','Breakfast','Carpark','Dog','Fencing','Frog','MATF','Painter','PoznanStreet','Welder','AlexaMeadeExhibit','CoffeeMartini','FlameSteak']
def cn(c,p,p1): return f"{p}{c:0{p1}d}"
for ds in DS:
    d=CFG[ds]; st=d['start_frame']; cams=[cn(c,d['prefix'],d['prefix1']) for c in d['cam_list']]
    seed={}; allids={}
    for cam in cams:
        g=cv2.imread(os.path.join(ROOT,ds,'Mask',cam,f"{st:06d}.png"),0); seed[cam]=set(int(x) for x in np.unique(g))-{0}
        files=sorted(f for f in os.listdir(os.path.join(ROOT,ds,'Mask',cam)) if f.endswith('.png'))
        allids[cam]={}
        for f in files:
            g=cv2.imread(os.path.join(ROOT,ds,'Mask',cam,f),0)
            for o in set(int(x) for x in np.unique(g))-{0}: allids[cam][o]=allids[cam].get(o,0)+1
    maxid={cam:max(seed[cam]) if seed[cam] else 0 for cam in cams}
    ref_maxid=max(cams,key=lambda c:maxid[c]); ref_most=max(cams,key=lambda c:len(seed[c]))
    def struct0(ref):
        return sum(n for cam in cams for o,n in allids[cam].items() if o not in seed[ref])
    tot=sum(n for cam in cams for n in allids[cam].values())
    print(f"{ds:20s} seed-ids per cam: {[(c,len(seed[c]),maxid[c]) for c in cams]} | max-id rule -> {ref_maxid} (struct-zero pairs {struct0(ref_maxid)}/{tot}) | most-objects rule -> {ref_most} ({struct0(ref_most)}/{tot}) | union of seed ids {len(set().union(*seed.values()))}")
