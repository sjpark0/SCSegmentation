"""Host: join seedfail.json + cost.json, failure types of the seeds < 0.8 and cost by size/type (docs/stage1-seedfail.md section 2) -> seedfail_joined.json."""
import json, collections
S=json.load(open("seedfail.json")); C=json.load(open("cost.json")); R=[]
for r in S:
    k=f"{r['scene']}|{r['cam']}|{r['obj']}"
    if k in C and r["gt_px"]>0: r.update(C[k]); r["cost"]=r["LH"]-r["LG"]; R.append(r)
def kind(r):
    if r["seed_px"]==0 or (r["recall"] or 0)<0.05: return "miss" if r["seed_px"]==0 or r["gt_px"]<50 else "wrong-object"
    if (r["precision"] or 0)<0.3: return "wrong-object"
    if r["frac_other"]>=0.2: return "spill-onto-other" + ("(twin)" if r["twin"] else "")
    if r["recall"]<0.85 and r["precision"]>=0.85: return "under-cover" + ("(beyond c_ini)" if r["vis_ratio"]>1.2 else "")
    return "boundary/mixed"
fail=[r for r in R if r["JF"]<0.8]; good=[r for r in R if r["JF"]>=0.8]
for name,P in (("FAIL <0.8",fail),):
    g=collections.defaultdict(list)
    for r in P: g[kind(r)].append(r)
    print(name, len(P), "cost %.3f"%sum(r["cost"] for r in P))
    for k,v in sorted(g.items(), key=lambda x:-sum(r["cost"] for r in x[1])):
        print(f"  {k:28s} n={len(v):3d} cost {sum(r['cost'] for r in v):+.3f}  env {sum(r['otype']=='env' for r in v)}  e.g. "+", ".join(f"{r['scene'][:8]}/{r['cam'][-4:]}/{r['obj']}" for r in sorted(v,key=lambda r:-r['cost'])[:4]))
def size(r): return "<2k" if r["gt_px"]<2000 else "2k-10k" if r["gt_px"]<1e4 else "10k-100k" if r["gt_px"]<1e5 else ">=100k"
for name,key in (("size",size),("otype",lambda r:r["otype"]),("twin",lambda r:r["twin"]),("border",lambda r:r["border"]),("beyond c_ini (vis>1.2)",lambda r:r["vis_ratio"]>1.2)):
    for P,lab in ((fail,"fail"),(good,"good")):
        g=collections.defaultdict(list)
        for r in P: g[key(r)].append(r)
        print(f"{name:22s} {lab}: "+"  ".join(f"{k}: n={len(v)} cost {sum(r['cost'] for r in v):+.3f} JF0 {sum(r['JF'] for r in v)/len(v):.3f}" for k,v in sorted(g.items(),key=lambda x:str(x[0]))))
# good seeds: is the error boundary-band? oracle uplift
for P,lab in ((good,"good"),(fail,"fail")):
    import statistics
    print(lab, "mean JF0 %.4f  r2 %.4f  r4 %.4f  r8 %.4f"%tuple(statistics.mean(r[k] for r in P) for k in ("JF","oracle_r2","oracle_r4","oracle_r8")))
# good seeds: under vs over segmentation
under=sum((1-r["recall"]) for r in good); over=sum((1-r["precision"]) for r in good)
print("good: mean 1-recall %.4f  mean 1-precision %.4f"%(under/len(good), over/len(good)))
json.dump([dict(r,kind=kind(r)) for r in R], open("seedfail_joined.json","w"))
