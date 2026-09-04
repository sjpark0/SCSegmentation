import json, os, statistics as st
import numpy as np, cv2
os.chdir('/home/sjpark/Documents/SCSegmentation/Data/MVSeg')
CFG=json.load(open('/home/sjpark/Documents/SCSegmentation/SCSam3/demo/MVSeg.json'))
def load(f):
    d={}
    for e in json.load(open(f)):
        if e.get('result') is None: continue
        d[(e['dataset'],e['camera'],e['method'])]=e
    return d
raw=load('jf_raw.json'); raw.update(load('jf_sam3_onestage.json')); raw.update(load('jf_sam3_onestagenew.json'))
stale=load('jf_raw_sam3_onestage.STALE-prefix-bug.json')
DS12=['AlexaMeadeFacePaint','Barn','Blocks','Breakfast','Carpark','Dog','Fencing','Frog','MATF','Painter','PoznanStreet','Welder']
DS3=['AlexaMeadeExhibit','CoffeeMartini','FlameSteak']
def cn(c,p,p1): return f"{p}{c:0{p1}d}"
def info(name):
    d=CFG[name]; perms=d.get('perms') or list(range(d['start_cam'],d['num_cam']+d['start_cam']))
    best,bn=None,-1
    for c in d['cam_list']:
        n=int(np.max(cv2.imread(os.path.join(d['folder'],'Mask',cn(c,d['prefix'],d['prefix1']),f"{d['start_frame']:06d}.png"),0)))
        if n>bn: best,bn=c,n
    return dict(cams=[cn(c,d['prefix'],d['prefix1']) for c in d['cam_list']], ref=cn(best,d['prefix'],d['prefix1']), vi={cn(c,d['prefix'],d['prefix1']):perms.index(c) for c in d['cam_list']}, N=len(perms))
def jf(e,var='all'):
    r=e['result']; return st.mean(r['J_'+var]), st.mean(r['F_'+var])
M=['SegMaskNew1','SegMaskNew2','SegMaskNew3','SegMaskSam3OneStage','SegMaskSam3OneStageNew']
print("per-camera J / F  (view index, #nbrs=min(idx,4), ref marked *)")
rows=[]
for ds in DS12+DS3:
    I=info(ds)
    for cam in I['cams']:
        vi=I['vi'][cam]; line=f"{ds:20s} {cam:12s} idx={vi:2d} nb={min(vi,4)} {'*' if cam==I['ref'] else ' '} N={I['N']:2d} nobj={len(raw[(ds,cam,'SegMaskNew1')]['objects']):2d}"
        vals={}
        for m in M:
            e=raw.get((ds,cam,m))
            if e: j,f=jf(e); vals[m]=(j,f); line+=f" {m[-6:]}:J{j:.3f}/F{f:.3f}"
            else: line+=f" {m[-6:]}:   -    "
        e=stale.get((ds,cam,'SegMaskSam3OneStage'))
        if e: j,f=jf(e); line+=f" STALE:J{j:.3f}/F{f:.3f}"
        if 'SegMaskSam3OneStage' in vals and 'SegMaskSam3OneStageNew' in vals:
            dJ=vals['SegMaskSam3OneStageNew'][0]-vals['SegMaskSam3OneStage'][0]; dF=vals['SegMaskSam3OneStageNew'][1]-vals['SegMaskSam3OneStage'][1]
            line+=f"  dNew-One: J{dJ:+.4f} F{dF:+.4f}"; rows.append((ds,cam,vi,min(vi,4),cam==I['ref'],dJ,dF))
        print(line)
print()
print("delta OneStageNew-OneStage grouped by #cross-view neighbours")
for nb in range(5):
    r=[x for x in rows if x[3]==nb]
    if r: print(f" nb={nb}: n={len(r):2d} meanJ{st.mean([x[5] for x in r]):+.4f} meanF{st.mean([x[6] for x in r]):+.4f}  cams={[ (x[0][:6],x[1]) for x in r]}")
print("ref cams:", [(x[0][:8],x[1],x[2],round(x[5],4),round(x[6],4)) for x in rows if x[4]])
print()
def dsmean(ds,m,src=raw,var='all'):
    js=[];fs=[]
    for cam in info(ds)['cams']:
        e=src.get((ds,cam,m))
        if e is None: return None
        js+=e['result']['J_'+var]; fs+=e['result']['F_'+var]
    return st.mean(js),st.mean(fs)
print(f"{'dataset':20s}"+''.join(f"{m[-11:]:>12s}" for m in M)+"  STALE_1stage  inner:One/New")
for ds in DS12+DS3:
    line=f"{ds:20s}"
    for m in M:
        r=dsmean(ds,m); line+=f"{(r[0]+r[1])/2:12.4f}" if r else f"{'-':>12s}"
    r=dsmean(ds,'SegMaskSam3OneStage',stale); line+=f"{(r[0]+r[1])/2:12.4f}" if r else f"{'-':>12s}"
    a=dsmean(ds,'SegMaskSam3OneStage',var='inner'); b=dsmean(ds,'SegMaskSam3OneStageNew',var='inner')
    line+=f"   {(a[0]+a[1])/2 if a else float('nan'):.4f}/{(b[0]+b[1])/2 if b else float('nan'):.4f}"
    print(line)
for label,DS in (('12 datasets',DS12),('15 datasets',DS12+DS3)):
    print(label)
    for m in M+['STALE']:
        src=stale if m=='STALE' else raw; mm='SegMaskSam3OneStage' if m=='STALE' else m
        rs=[dsmean(d,mm,src) for d in DS]
        if all(rs): print(f"  {m:24s} J={st.mean(r[0] for r in rs):.4f} F={st.mean(r[1] for r in rs):.4f} J&F={st.mean((r[0]+r[1])/2 for r in rs):.4f}")
        else: print(f"  {m:24s} incomplete ({sum(1 for r in rs if r)}/{len(DS)})")
print("SAM2 repeat-run spread per dataset (J&F): New1/New2/New3 ; SAM3 OneStage stale vs final")
for ds in DS12+DS3:
    v=[ (lambda r:(r[0]+r[1])/2)(dsmean(ds,m)) for m in M[:3]]
    s=dsmean(ds,'SegMaskSam3OneStage',stale); f=dsmean(ds,'SegMaskSam3OneStage')
    print(f"  {ds:20s} {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} range={max(v)-min(v):.4f}   stale={((s[0]+s[1])/2 if s else float('nan')):.4f} final={(f[0]+f[1])/2:.4f}")
