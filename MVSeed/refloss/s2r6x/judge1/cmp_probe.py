import os, sys, numpy as np, cv2, json
A=sys.argv[1]; B=sys.argv[2]
for cam in sorted(x for x in os.listdir(A) if os.path.isdir(os.path.join(A,x))):
    ch=[]; xs=0; us=0; first=None; n=0
    for t in sorted((x for x in os.listdir(os.path.join(A,cam)) if x.isdigit()), key=int):
        for o in sorted(os.listdir(os.path.join(A,cam,t))):
            pa=os.path.join(A,cam,t,o); pb=os.path.join(B,cam,t,o); n+=1
            ba=open(pa,'rb').read(); bb=open(pb,'rb').read() if os.path.exists(pb) else None
            if ba!=bb:
                a=cv2.imread(pa,0)>127; b=cv2.imread(pb,0)>127 if bb else np.zeros_like(a)
                x=np.count_nonzero(a^b); u=np.count_nonzero(a|b)
                if x: ch.append((int(t),o)); xs+=x; us+=u; first=int(t) if first is None else min(first,int(t))
    print(cam, 'pngs',n,'changed',len(ch),'first changed frame',first, 'XOR/union(changed) %.4f'%(xs/max(us,1)))
