# GT-free: count frozen-rule snaps on non-reference annotated views (reads output PNGs + video only; frame-0 GT for a Z check)
import json, os, sys, importlib.util
from multiprocessing import Pool
import numpy as np, cv2
REPO='/home/sjpark/Documents/SCSegmentation'; R=REPO+'/Data/MVSeg'
spec=importlib.util.spec_from_file_location('ref_snap', REPO+'/SCSam3/ref_snap.py'); RS=importlib.util.module_from_spec(spec); spec.loader.exec_module(RS)
CFG=json.load(open(REPO+'/SCSam3/demo/MVSeg.json')); CEN=json.load(open(REPO+'/docs/raw/muvod_object_sets.json'))
M=sys.argv[1]
def job(a):
    sc,cam=a; c=CFG[sc]; d=f'{R}/{c["folder"]}/{M}/{cam}'; st=c['start_frame']
    frames=sorted((x for x in os.listdir(d) if x.isdigit()), key=int)
    assert frames[0]==str(st)
    objs=sorted(os.listdir(f'{d}/{frames[0]}'))
    m0={o: cv2.imread(f'{d}/{frames[0]}/{o}',0)>127 for o in objs}
    gt=cv2.imread(f'{R}/{c["folder"]}/Mask/{cam}/{st:06d}.png',0)
    f0gt=sum(int(np.array_equal(m0[o], gt==int(o.split('.')[0]))) for o in objs) if gt is not None else None
    sh=next(iter(m0.values())).shape
    def img(t):
        im=cv2.imread(f'{R}/{c["folder"]}/Video/{cam}/{int(t):06d}.jpg',0)
        if im.shape!=sh: im=cv2.resize(im,(sh[1],sh[0]),interpolation=cv2.INTER_AREA)
        return im.astype(np.int16)
    i0=img(frames[0]); n=0; of=0; basic=set(CEN[sc]['seed_ids_c_ini']); nb=0; ofb=0
    for t in frames[1:]:
        it=img(t)
        for o in sorted(os.listdir(f'{d}/{t}')):
            if o not in m0: continue
            of+=1; isb=int(o.split('.')[0]) in basic; ofb+=isb
            s,_,_=RS.decide(m0[o], cv2.imread(f'{d}/{t}/{o}',0)>127, i0, it, np, cv2)
            n+=s; nb+=s and isb
    return sc,cam,n,of,nb,ofb,f0gt,len(objs)
if __name__=='__main__':
    jobs=[]
    for sc in sorted(CFG):
        if sc not in CEN: continue
        ref=CEN[sc]['c_ini']
        for cam in CEN[sc]['cameras']:
            if cam!=ref: jobs.append((sc,cam))
    with Pool(14) as p: res=p.map(job,jobs)
    per={}
    for sc,cam,n,of,nb,ofb,f0,no in res:
        a=per.setdefault(sc,[0,0,0,0]); a[0]+=n;a[1]+=of;a[2]+=nb;a[3]+=ofb
        print(sc,cam,n,of,nb,ofb,'frame0==GT',f0,'/',no)
    print({k:v[0] for k,v in per.items()})
    print('total snaps',sum(v[0] for v in per.values()),'of',sum(v[1] for v in per.values()),'basic',sum(v[2] for v in per.values()),'of',sum(v[3] for v in per.values()), 'views',len(res))
