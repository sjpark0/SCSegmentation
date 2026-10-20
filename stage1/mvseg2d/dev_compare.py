#!/usr/bin/env python3
"""Compare two DEV scorings (methods.geomem.dev_score outputs) -- host, numpy only (docs/stage1-dev-protocol.md).

Paired scene-cluster bootstrap (mvseg2d.paired, views = "<t>/<cam>", 10 000 resamples, seed 0) on J, F, J&F over the
common pairs, plus splits by the base's seedcat kind, failing (< 0.8) vs good, and per scene.
Reference-rotated scorings (docs/stage1-dev-protocol.md §4, DEV-A' / DEV-B'): rows with `ref` pair on
(scene, t, ref, cam, obj) and use the view "<t>/<ref>/<cam>"; BASE / CAND may each be several scorings joined by commas
(e.g. the two rotations), concatenated before pairing -- a frame's rotations stay in their scene's cluster.
    .venv/bin/python -m mvseg2d.dev_compare BASE.json CAND.json [--out report.json]
    .venv/bin/python -m mvseg2d.dev_compare BASE_rot1.json,BASE_rot2.json CAND_rot1.json,CAND_rot2.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from mvseg2d.paired import paired_scene_bootstrap


def load(p):
    """One scoring, or several joined by commas (concatenated)."""
    rows = []
    for q in p.split(","):
        with open(q) as f:
            rows += json.load(f)["rows"]
    for r in rows:
        r.setdefault("view", f"{r['t']}/{r['ref']}/{r['cam']}" if r.get("ref") is not None else f"{r['t']}/{r['cam']}")
    return rows


def pair_key(r):
    """(scene, t, ref, cam, obj); ref = "" for an unrotated row (the original c_ini), so unrotated pairs sort as before."""
    return (r["scene"], r["t"], r.get("ref") or "", r["cam"], r["obj"])


def compare(base, cand, n=10_000, seed=0):
    key = pair_key
    kb, kc = {key(r): r for r in base}, {key(r): r for r in cand}
    if len(kb) != len(base) or len(kc) != len(cand):
        raise SystemExit(f"duplicated pairs: base {len(base)} rows / {len(kb)} keys, cand {len(cand)} rows / {len(kc)} keys")
    common = sorted(set(kb) & set(kc))
    if len(common) != len(kb) or len(common) != len(kc):
        raise SystemExit(f"pair sets differ: base {len(kb)}, cand {len(kc)}, common {len(common)}")
    B, C = [kb[k] for k in common], [kc[k] for k in common]
    out = {m: paired_scene_bootstrap(B, C, key=m, n=n, seed=seed, view_key="view") for m in ("J", "F", "JF")}
    d = {k: kc[k]["JF"] - kb[k]["JF"] for k in common}
    groups = defaultdict(list)
    for k in common:
        groups["fail" if kb[k]["JF"] < 0.8 else "good"].append(k)
        groups[f"kind:{kb[k].get('kind')}"].append(k)
    splits = {g: dict(n=len(ks), dJF=float(np.mean([d[k] for k in ks])), up=sum(d[k] > 0.01 for k in ks),
                      down=sum(d[k] < -0.01 for k in ks)) for g, ks in sorted(groups.items())}
    return dict(n_pairs=len(common), paired={m: dict(delta=v["delta"], ci=list(v["ci"]), per_scene=v["per_scene"]) for m, v in out.items()},
                splits=splits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = compare(load(a.base), load(a.cand))
    p = r["paired"]
    print(f"{r['n_pairs']} pairs  dJ&F {p['JF']['delta']:+.4f} CI [{p['JF']['ci'][0]:+.4f}, {p['JF']['ci'][1]:+.4f}]  "
          f"dJ {p['J']['delta']:+.4f}  dF {p['F']['delta']:+.4f}")
    for g, v in r["splits"].items():
        print(f"  {g:32s} n={v['n']:5d}  dJF {v['dJF']:+.4f}  up {v['up']}  down {v['down']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
