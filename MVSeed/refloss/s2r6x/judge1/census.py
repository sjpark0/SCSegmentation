import json, hashlib, os
R='/home/sjpark/Documents/SCSegmentation/Data/MVSeg'
cfg=json.load(open('/home/sjpark/Documents/SCSegmentation/SCSam3/demo/MVSeg.json'))
H='SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs'
lines=[];tot=0;tk=0;tt=0;per={}
for sc in sorted(cfg):
    p=f'{R}/{cfg[sc]["folder"]}/{H}/MANIFEST.json'
    if not os.path.exists(p): continue
    m=json.load(open(p)); pv=m['provenance']
    rs=pv['ref_snap']; cs=pv['crop_small']
    ref=rs['ref']
    sf=rs['snapped_files']
    tgt=[t for t in cs['targets'] if t['cam']==ref]
    kept={str(t['obj']) for t in tgt if t['kept']}
    allt={str(t['obj']) for t in tgt}
    nk=sum(1 for f in sf if f.split('/')[2].split('.')[0] in kept)
    nt=sum(1 for f in sf if f.split('/')[2].split('.')[0] in allt)
    per[sc]=(len(sf),nk,nt,len(tgt),len(kept),pv['reference_view_index'],pv['track_idx'][:3],len(pv['track_idx']),pv['scored_view_idx'])
    tot+=len(sf);tk+=nk;tt+=nt
    lines+= [f'{cfg[sc]["folder"]}/{f}' for f in sf]
for k,v in per.items(): print(k,v)
print('total',tot,'on crop-kept',tk,'on crop targets',tt,'non-target',tot-tt)
s='\n'.join(sorted(lines))
print(hashlib.sha256(s.encode()).hexdigest())
print(hashlib.sha256((s+'\n').encode()).hexdigest())
