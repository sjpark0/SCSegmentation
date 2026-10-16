# GT-free: for replayed snaps on c_ini crop-kept objects, is the MAINLINE tracker mask at that frame empty? (view_areas)
import json, os
R='/home/sjpark/Documents/SCSegmentation/Data/MVSeg'; cfg=json.load(open('/home/sjpark/Documents/SCSegmentation/SCSam3/demo/MVSeg.json'))
H='SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs'
tot=0; empty=0; small=0; per={}
for sc in sorted(cfg):
    p=f'{R}/{cfg[sc]["folder"]}/{H}/MANIFEST.json'
    if not os.path.exists(p): continue
    pv=json.load(open(p))['provenance']; ref=pv['ref_snap']['ref']; ri=str(pv['reference_view_index'])
    tr=pv['view_areas']['tracked']
    kept={str(t['obj']) for t in pv['crop_small']['targets'] if t['cam']==ref and t['kept']}
    st=pv['start_frame']
    for f in pv['ref_snap']['snapped_files']:
        cam,t,o=f.split('/'); o=o.split('.')[0]
        if o not in kept: continue
        tot+=1
        rec=tr.get(ri,{})
        a=None
        # try layouts
        if isinstance(rec,dict):
            fr=rec.get(t) or rec.get(str(int(t)-st))
            if isinstance(fr,dict): a=fr.get(o)
        if a is None or a==0: empty+=1; per[sc]=per.get(sc,0)+1
print('crop-kept replay entries',tot,'mainline area missing/0',empty,per)
m=json.load(open(f'{R}/CoffeeMartini/{H}/MANIFEST.json'))['provenance']['view_areas']['tracked']
k=list(m)[0]; print('tracked keys',list(m)[:5], 'inner', list(m[k])[:5])
