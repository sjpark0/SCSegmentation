import os, sys, hashlib
a, b = sys.argv[1], sys.argv[2]
def files(d):
    out = {}
    for root, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(".png"):
                p = os.path.join(root, f); out[os.path.relpath(p, d)] = p
    return out
fa, fb = files(a), files(b)
print("files", len(fa), len(fb), "only a", len(set(fa) - set(fb)), "only b", len(set(fb) - set(fa)))
diff = {}
for k in sorted(set(fa) & set(fb)):
    if open(fa[k], "rb").read() != open(fb[k], "rb").read():
        cam = k.split("/")[0]; diff.setdefault(cam, []).append(k)
for cam in sorted(set(k.split("/")[0] for k in fa)):
    print(cam, "differ", len(diff.get(cam, [])), (diff.get(cam, [])[:6]))
