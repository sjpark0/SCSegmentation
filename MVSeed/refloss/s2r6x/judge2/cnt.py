import sys, os, json, numpy as np, cv2
sys.path.insert(0, '/home/sjpark/Documents/SCSegmentation/SCSam3')
import ref_snap
R='/home/sjpark/Documents/SCSegmentation'
sets=json.load(open(f'{R}/docs/raw/muvod_object_sets.json'))
cfg=json.load(open(f'{R}/SCSam3/demo/MVSeg.json'))
meth=sys.argv[1]
for s in sys.argv[2:]:
    folder=s
    start=cfg[s]['start_frame'] if 'start_frame' in cfg[s] else 0
    cini=sets[s]['c_ini']; tot=0; n=0; f0ok=0; f0n=0
    for cam in sets[s]['cameras']:
        if cam==cini: continue
        out=f'{R}/Data/MVSeg/{folder}/{meth}'
        cam_dir=f'{out}/{cam}'
        frames=sorted((x for x in os.listdir(cam_dir) if x.isdigit()), key=int)
        objs=sorted(os.listdir(f'{cam_dir}/{frames[0]}'))
        gt=cv2.imread(f'{R}/Data/MVSeg/{folder}/Mask/{cam}/{int(frames[0]):06d}.png', cv2.IMREAD_GRAYSCALE)
        m0={o:cv2.imread(f'{cam_dir}/{frames[0]}/{o}',0)>127 for o in objs}
        for o in objs:
            f0n+=1; f0ok+= int(np.array_equal(m0[o], gt==int(o.split('.')[0])))
        vd=f'{R}/Data/MVSeg/{folder}/Video/{cam}'
        def img(t):
            im=cv2.imread(f'{vd}/{int(t):06d}.jpg',0)
            sh=next(iter(m0.values())).shape
            if im.shape!=sh: im=cv2.resize(im,(sh[1],sh[0]),interpolation=cv2.INTER_AREA)
            return im.astype(np.int16)
        i0=img(frames[0])
        for t in frames[1:]:
            it=img(t)
            for o in sorted(os.listdir(f'{cam_dir}/{t}')):
                if o not in m0: continue
                n+=1
                mt=cv2.imread(f'{cam_dir}/{t}/{o}',0)>127
                tot+=ref_snap.decide(m0[o],mt,i0,it,np,cv2)[0]
    print(s, start, 'snaps',tot,'of',n,'frame0==GT',f0ok,'/',f0n, flush=True)
