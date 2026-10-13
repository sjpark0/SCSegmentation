"""Host: <seedfail a.json> <seedfail b.json>: seed J&F change b - a by population and size (docs/stage1-seedfail.md section 4)."""
import json, sys, statistics as S, collections
k=lambda r:(r["scene"],r["cam"],r["obj"])
A={k(r):r for r in json.load(open(sys.argv[1]))}; B={k(r):r for r in json.load(open(sys.argv[2]))}
M={k(r):r for r in json.load(open("seedfail.json"))}
assert all(abs(A[x]["JF"]-M[x]["JF"])<1e-9 for x in A), "s1 folder != MVSeed seeds"
C=json.load(open("cost.json"))
P=[x for x in A if A[x]["gt_px"]>0 and "|".join(map(str,x)) in C]
E=[x for x in A if A[x]["gt_px"]==0]
def iou(a,b): return None
size=lambda r:"<2k" if r["gt_px"]<2000 else "2k-10k" if r["gt_px"]<1e4 else "10k-100k" if r["gt_px"]<1e5 else ">=100k"
# GT-free gate: accept refined if IoU(refined, seed) >= tau -- approximate IoU from recall/precision not possible; need masks -> use stored? skip
for nm,ks in (("all",P),("good",[x for x in P if A[x]["JF"]>=0.8]),("fail",[x for x in P if A[x]["JF"]<0.8])):
    d=[B[x]["JF"]-A[x]["JF"] for x in ks]
    print(f"{nm:5s} n={len(ks)} seed {S.mean(A[x]['JF'] for x in ks):.4f} -> {S.mean(B[x]['JF'] for x in ks):.4f}  d {S.mean(d):+.4f}  up {sum(v>0.01 for v in d)} down {sum(v<-0.01 for v in d)}  bigdown(<-0.1) {sum(v<-0.1 for v in d)}")
for s in ("<2k","2k-10k","10k-100k",">=100k"):
    ks=[x for x in P if size(A[x])==s]; d=[B[x]["JF"]-A[x]["JF"] for x in ks]
    print(f"  {s:9s} n={len(ks)} d {S.mean(d):+.4f} up {sum(v>0.01 for v in d)} down {sum(v<-0.01 for v in d)}")
print("halluc on empty GT: seed", sum(A[x]["seed_px"]>0 for x in E), "refined", sum(B[x]["seed_px"]>0 for x in E))
# rough stage-2 transfer: seed cost ~ (1-JF0)-weighted; estimate with per-pair weight: LH/(1-jf0) ratio not robust; use pair weight w*21
