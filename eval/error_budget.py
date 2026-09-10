"""Where do the missing 9 points of the headline live?  (docs/error-budget.md)

Oracle decomposition of the adopted configuration on the MUVOD basic protocol: each line replaces one
part of the prediction with the ground truth and re-scores, so the gain is the price we are paying for
that part.  Diagnosis only -- it reads the stored per-frame J/F of Data/MVSeg/jf_r2.json and touches
nothing.  Run on the host:  python3 eval/error_budget.py
"""
import json, statistics, sys
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M='SegMaskSam3XW1CGPS4MFsCs10kG2'
d=[x for x in json.load(open(f'{REPO}/Data/MVSeg/jf_r2.json')) if x['method']==M]
cen=json.load(open(f'{REPO}/docs/raw/muvod_object_sets.json'))
def rows():
    for r in d:
        sc=r['dataset']; basic=set(cen[sc]['seed_ids_c_ini']); ref=cen[sc]['c_ini']
        pf=r['result']['per_frame']
        for i,o in enumerate(r['objects']):
            if o not in basic: continue
            yield dict(scene=sc, cam=r['camera'], obj=o, ref=r['camera']==ref,
                       J=pf['J'][i], F=pf['F'][i], gt=pf['gt_area'][i], pred=pf['pred_area'][i])
R=list(rows())
def score(f):
    """MUVOD J&F^3: mean over objects -> mean over the 3 cameras -> mean over scenes."""
    per={}
    for r in R:
        vals=[f(r,t) for t in range(len(r['J'])) if r['gt'][t] is not None or True]
        vals=[v for v in vals if v is not None]
        if vals: per.setdefault(r['scene'],{}).setdefault(r['cam'],[]).append(statistics.mean(vals))
    return 100*statistics.mean([statistics.mean([statistics.mean(v) for v in cams.values()]) for cams in per.values()])
jf=lambda r,t: None if r['J'][t] is None else (r['J'][t]+r['F'][t])/2
print(f"headline           {score(jf):.2f}   (기록 91.02)")
print(f"J만                {score(lambda r,t: r['J'][t]):.2f}")
print(f"F만(경계)          {score(lambda r,t: r['F'][t]):.2f}")
print()
def oracle(name, f): print(f"  {name:42s} {score(f):6.2f}   +{score(f)-score(jf):5.2f}")
print("오라클(그 부분만 완벽하다고 두면 얼마가 되는가)")
oracle("① 기준 카메라(c_ini) 완벽", lambda r,t: 1.0 if r['ref'] else jf(r,t))
oracle("② 비기준 두 카메라 완벽", lambda r,t: jf(r,t) if r['ref'] else 1.0)
oracle("③ 프레임 0 완벽", lambda r,t: 1.0 if t==0 else jf(r,t))
oracle("④ 프레임 1~20 완벽", lambda r,t: jf(r,t) if t==0 else 1.0)
oracle("⑤ 경계 F 완벽 (J 유지)", lambda r,t: None if r['J'][t] is None else (r['J'][t]+1.0)/2)
oracle("⑥ 영역 J 완벽 (F 유지)", lambda r,t: None if r['J'][t] is None else (1.0+r['F'][t])/2)
sz=lambda r: statistics.median([a for a in r['gt'] if a]) if any(r['gt']) else 0
oracle("⑦ <2,000 px 객체 완벽", lambda r,t: 1.0 if sz(r)<2000 else jf(r,t))
oracle("⑧ 2k~10k px 객체 완벽", lambda r,t: 1.0 if 2000<=sz(r)<10000 else jf(r,t))
oracle("⑨ ≥10,000 px 객체 완벽", lambda r,t: 1.0 if sz(r)>=10000 else jf(r,t))
bad=lambda r: (jf(r,0) or 0)<0.5
oracle("⑩ 프레임 0이 실패한 쌍(J&F<0.5) 완벽", lambda r,t: 1.0 if bad(r) else jf(r,t))
oracle("⑪ 그 외 정상 쌍 완벽", lambda r,t: jf(r,t) if bad(r) else 1.0)
print()
print("구성")
n=len(R); nb=sum(1 for r in R if bad(r))
print(f"  (카메라,객체) 쌍 {n},  프레임 0 실패 {nb} ({100*nb/n:.1f}%),  기준 카메라 {sum(r['ref'] for r in R)}")
for lo,hi,name in ((0,2000,'<2k'),(2000,10000,'2k-10k'),(10000,10**9,'>=10k')):
    g=[r for r in R if lo<=sz(r)<hi]
    if g: print(f"  {name:8s} n={len(g):4d}  J&F {statistics.mean([statistics.mean([jf(r,t) for t in range(21) if jf(r,t) is not None]) for r in g]):.3f}")
print()
print("프레임별 평균 J&F (전 쌍)")
for t in (0,1,2,5,10,15,20):
    v=[jf(r,t) for r in R if jf(r,t) is not None]
    print(f"  프레임 {t:2d}: {statistics.mean(v):.4f}")

print()
print("=== 결손이 집중돼 있는가, 퍼져 있는가")
pair=[(statistics.mean([jf(r,t) for t in range(21) if jf(r,t) is not None]), r) for r in R]
pair.sort(key=lambda x:x[0])
tot=sum(1-v for v,_ in pair)
for q in (0.02,0.05,0.10,0.25,0.50):
    k=int(len(pair)*q); part=sum(1-v for v,_ in pair[:k])
    print(f"  가장 나쁜 {int(q*100):2d}% 쌍({k:4d}개)이 전체 결손의 {100*part/tot:4.1f}%")
print(f"  전체 결손 합 {tot:.1f} (쌍당 평균 {tot/len(pair):.3f})")
print(f"  J&F>0.95인 쌍 {sum(1 for v,_ in pair if v>0.95)}/{len(pair)},  0.8~0.95 {sum(1 for v,_ in pair if 0.8<v<=0.95)},  <0.8 {sum(1 for v,_ in pair if v<=0.8)}")
print()
print("=== 기준 카메라(정답을 그대로 주고 재예측)의 프레임별 J")
ref=[r for r in R if r['ref']]
for t in (0,1,5,10,20):
    print(f"  프레임 {t:2d}: J {statistics.mean([r['J'][t] for r in ref if r['J'][t] is not None]):.4f}  F {statistics.mean([r['F'][t] for r in ref if r['F'][t] is not None]):.4f}")
print()
print("=== 비기준 카메라, 프레임 0이 좋은(≥0.8) 쌍만: 이후 프레임에서 얼마나 잃는가")
good=[r for r in R if not r['ref'] and (jf(r,0) or 0)>=0.8]
print(f"  n={len(good)}  프레임0 {statistics.mean([jf(r,0) for r in good]):.4f} -> 프레임1~20 {statistics.mean([statistics.mean([jf(r,t) for t in range(1,21) if jf(r,t) is not None]) for r in good]):.4f}")
print()
print("=== 예측 면적 / 정답 면적 (편향 진단, 프레임 1~20)")
import math
rat=[]
for r in R:
    for t in range(1,21):
        if r['gt'][t] and r['pred'][t] is not None: rat.append(r['pred'][t]/r['gt'][t])
rat.sort()
print(f"  n={len(rat)}  p10 {rat[len(rat)//10]:.2f}  중앙값 {statistics.median(rat):.2f}  p90 {rat[9*len(rat)//10]:.2f}  (1.0이면 편향 없음)")
print(f"  과소예측(<0.9) {100*sum(1 for x in rat if x<0.9)/len(rat):.0f}%   과대예측(>1.1) {100*sum(1 for x in rat if x>1.1)/len(rat):.0f}%")
