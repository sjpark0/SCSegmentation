"""Read geo_bias.{w,b_none} from torch zip checkpoints without torch (read-only)."""
import pickle, sys, zipfile
import numpy as np

DT = {"FloatStorage": np.float32, "BFloat16Storage": None, "HalfStorage": np.float16, "DoubleStorage": np.float64,
      "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}

def load(path, want=("geo_bias",)):
    zf = zipfile.ZipFile(path)
    root = zf.namelist()[0].split("/")[0]
    class Stor:
        def __init__(self, key, dt): self.key, self.dt = key, dt
    def rebuild(storage, offset, size, stride, *a):
        if storage.dt is None: return None
        buf = np.frombuffer(zf.read(f"{root}/data/{storage.key}"), dtype=storage.dt)
        n = int(np.prod(size)) if size else 1
        return buf[offset:offset + n].reshape(size) if size else buf[offset]
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2": return rebuild
            if name == "_rebuild_parameter": return lambda t, rg, hooks, *a: t
            if mod.startswith("torch") and name.endswith("Storage"): return name
            if mod == "collections" and name == "OrderedDict":
                import collections; return collections.OrderedDict
            if mod.startswith("torch"): return lambda *a, **k: None
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            _, stype, key, loc, numel = pid
            return Stor(key, DT.get(stype if isinstance(stype, str) else str(stype)))
    return U(zf.open(f"{root}/data.pkl")).load()

for p in sys.argv[1:]:
    ck = load(p)
    gb = ck.get("geo_bias")
    w = np.asarray(gb["w"], np.float64); b = float(np.asarray(gb["b_none"]).ravel()[0])
    centres = np.linspace(0, 2, 16)
    d = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0])
    bias = np.exp(-((d[:, None] - centres) ** 2) / (2 * 0.125 ** 2)) @ w
    print(p.split("runs/")[1], "w", np.round(w, 2).tolist(), "b_none", round(b, 2))
    print("   bias(d) at d=", d.tolist(), "->", np.round(bias, 2).tolist())
