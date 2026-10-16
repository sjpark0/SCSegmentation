#!/usr/bin/env python3
"""S2-R7 watch values Z1-Z6 (docs/stage2-R7-prereg.md section 7), GT-free, before any scoring.
Arm M = SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2RmRs against base H = ...Cs10kG2Rs (the E14 headline folder).
Host python:  PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python MVSeed/s2r7_check.py [--scenes ...]
-> MVSeed/runs/s2r7_check.json; exit 2 if any watch value fails (no verdict then: fix, erratum, rerun).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data", "MVSeg")
OUT = os.path.join(REPO, "MVSeed", "runs", "s2r7_check.json")
H = "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs"
M = "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2RmRs"
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json"), encoding="utf-8"))
CEN = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
START1 = dict(AlexaMeadeExhibit=18, AlexaMeadeFacePaint=11, Barn=14, Blocks=11, Breakfast=10, CBABasketball=7, Carpark=9,
              CoffeeMartini=27, Dog=7, Fencing=4, FlameSteak=36, Frog=0, MATF=13, MartialArts=11, Painter=12,
              PoznanStreet=8, Welder=12)                     # prereg Z2(c): H's non-target snaps at start+1, total 210
RULE = {"tau": 0.8, "delta": 4.0, "dilate": 31}
P1_MAX = 3183                     # erratum section 13: the GT-free delta-pass set D of H (3 181 snaps + 2 FlameSteak frames)


def _ref_snap():
    spec = importlib.util.spec_from_file_location("ref_snap", os.path.join(REPO, "SCSam3", "ref_snap.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def delta_pass(sc, dh, T):
    """D: the (c_ini cam/t/obj.png) with obj not in T, a non-empty frame-start PNG in H and mean |I_t - I_0| over the
    dilated m0 <= DELTA -- video and H's frame-start PNGs only (GT-free).  Every injection must lie in D."""
    import cv2
    import numpy as np
    rs = _ref_snap()
    f, start, cini = CFG[sc]["folder"], CFG[sc]["start_frame"], CEN[sc]["c_ini"]
    on = rs.OnlineSnap(dh, os.path.join(DATA, f, "Video", cini), cini, start, T, np, cv2)
    out = set()
    for t in range(start + 1, start + CFG[sc]["num_frame"]):
        # IoU is 1 when the "mask" is m0 itself, so decide_frame then passes iff the delta test passes
        if on.m0 is None:
            on._load()
        for o in on.decide_frame(t, {o: m for o, m in on.m0.items()}):
            out.add(f"{cini}/{t}/{o}.png")
    return out


def perms(sc):
    c = CFG[sc]
    return c.get("perms") or list(range(c["start_cam"], c["start_cam"] + c["num_cam"]))


def _j(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _bytes(p):
    with open(p, "rb") as f:
        return f.read()


def _sha(p):
    return hashlib.sha256(_bytes(p)).hexdigest()


def _obj(rel):
    return int(rel.split("/")[-1].split(".")[0])


def _frame(rel):
    return int(rel.split("/")[-2])


def check_scene(sc):
    f, start = CFG[sc]["folder"], CFG[sc]["start_frame"]
    dh, dm = os.path.join(DATA, f, H), os.path.join(DATA, f, M)
    bad, info = [], {}
    if not os.path.isfile(os.path.join(dm, "MANIFEST.json")):
        return dict(bad=[("missing", sc)], info={})
    mh, mm = _j(os.path.join(dh, "MANIFEST.json"))["provenance"], _j(os.path.join(dm, "MANIFEST.json"))["provenance"]
    rsm = mm.get("ref_snap_mem")
    if not rsm:
        return dict(bad=[("no ref_snap_mem block", sc)], info={})
    cini = CEN[sc]["c_ini"]
    T = sorted(int(o) for o in rsm["excluded_crop_targets"])
    # ---- Z1: frame-start PNGs of every camera identical; crop block identical; crop-kept PNGs identical; T == H's c_ini targets
    cams = sorted(c for c in os.listdir(dh) if os.path.isdir(os.path.join(dh, c)))
    if cams != sorted(c for c in os.listdir(dm) if os.path.isdir(os.path.join(dm, c))):
        bad.append(("Z1-cams", sc))
    for cam in cams:
        a, b = os.path.join(dh, cam, str(start)), os.path.join(dm, cam, str(start))
        fa, fb = sorted(os.listdir(a)), sorted(os.listdir(b))
        if fa != fb or any(_bytes(os.path.join(a, x)) != _bytes(os.path.join(b, x)) for x in fa):
            bad.append(("Z1-frame0", sc, cam))
    ch, cm = mh["crop_small"], mm["crop_small"]
    keyf = lambda t: (t["cam"], t["obj"], t["kept"], t["why"], t["areas"])
    if [keyf(t) for t in ch["targets"]] != [keyf(t) for t in cm["targets"]]:
        bad.append(("Z1-crop-targets", sc))
    for t in cm["targets"]:
        if not t["kept"]:
            continue
        for fr in range(start + 1, start + CFG[sc]["num_frame"]):
            a, b = os.path.join(dh, t["cam"], str(fr), f"{t['obj']}.png"), os.path.join(dm, t["cam"], str(fr), f"{t['obj']}.png")
            if os.path.isfile(a) != os.path.isfile(b) or (os.path.isfile(a) and _bytes(a) != _bytes(b)):
                bad.append(("Z1-crop-kept-png", sc, t["cam"], t["obj"], fr))
    t_h = sorted(int(t["obj"]) for t in ch["targets"] if t["cam"] == cini)
    if T != t_h:
        bad.append(("Z1-T", sc, T, t_h))
    # ---- Z2
    inj = sorted(rsm["injected"])
    snap_m = mm["ref_snap"]["snapped_files"]
    snap_h = mh["ref_snap"]["snapped_files"]
    m_minus_T = sorted(x for x in snap_m if _obj(x) not in set(T))
    h_minus_T = set(x for x in snap_h if _obj(x) not in set(T))
    if inj != m_minus_T or not rsm["injected_equals_snap_minus_targets"]:
        bad.append(("Z2a", sc, len(inj), len(m_minus_T)))
    D = delta_pass(sc, dh, T)
    if not set(inj) <= D:
        bad.append(("Z2b", sc, sorted(set(inj) - D)[:5]))
    f1 = start + 1
    inj1 = sorted(x for x in inj if _frame(x) == f1)
    h1 = sorted(x for x in h_minus_T if _frame(x) == f1)
    if inj1 != h1 or len(inj1) != START1[sc]:
        bad.append(("Z2c", sc, len(inj1), len(h1), START1[sc]))
    if rsm["errors"] or (inj and not (rsm["obj_score_min"] is not None and rsm["obj_score_min"] > 0)):
        bad.append(("Z2d", sc, rsm["errors"][:3], rsm["obj_score_min"]))
    if rsm["cache_misses"]:
        bad.append(("Z2e", sc, rsm["cache_misses"]))
    # ---- Z3: snapped c_ini PNGs == frame-start PNG; rule recorded
    for x in snap_m:
        cam, fr, o = x.split("/")
        if _bytes(os.path.join(dm, cam, fr, o)) != _bytes(os.path.join(dm, cam, str(start), o)):
            bad.append(("Z3", sc, x))
    if mm["ref_snap"]["rule"] != RULE or rsm["rule"] != RULE:
        bad.append(("Z3-rule", sc))
    # ---- Z4: the cone (session position r of c_ini, mode C)
    tidx = mm["track_idx"]
    r = tidx.index(mm["reference_view_index"])
    va_h, va_m = mh["view_areas"]["tracked"], mm["view_areas"]["tracked"]
    first = {}
    for pos, view in enumerate(tidx):
        a, b = va_h.get(str(view), {}), va_m.get(str(view), {})
        frames = sorted(set(a) | set(b), key=int)
        diff = [int(t) for t in frames if a.get(t) != b.get(t)]
        first[pos - r] = min(diff) if diff else None
    cam_view = {}                                               # scored cameras: first differing PNG frame
    pm = perms(sc)
    for cam in cams:
        idx = [i for i, p in enumerate(pm) if f"{CFG[sc]['prefix']}{p:0{CFG[sc]['prefix1']}d}" == cam]
        cam_view[cam] = idx[0] if idx else None
    png_first = {}
    for cam in cams:
        fr_dirs = sorted((x for x in os.listdir(os.path.join(dh, cam)) if x.isdigit()), key=int)
        fd = None
        for fr in fr_dirs:
            a, b = os.path.join(dh, cam, fr), os.path.join(dm, cam, fr)
            la = sorted(os.listdir(a)); lb = sorted(os.listdir(b)) if os.path.isdir(b) else None
            if la != lb or any(_bytes(os.path.join(a, x)) != _bytes(os.path.join(b, x)) for x in la):
                fd = int(fr)
                break
        v = cam_view[cam]
        if v in tidx:
            png_first[tidx.index(v) - r] = fd
    for d, fd in list(first.items()) + list(png_first.items()):
        if fd is None:
            continue
        lo = f1 if d >= 1 else (f1 + 1 if d == 0 else f1 + abs(d))
        if fd < lo:
            bad.append(("Z4", sc, d, fd, lo))
    info.update(n_injected=len(inj), T=T, inj_start1=len(inj1), snaps_arm=len(snap_m), snaps_H=len(snap_h), n_D=len(D),
                inj_not_in_H=sorted(set(inj) - h_minus_T),
                rel_change=rsm["rel_change"], seconds_inject=rsm["seconds"],
                first_diff_by_d={str(k): v for k, v in sorted(first.items())},
                png_first_diff_by_d={str(k): v for k, v in sorted(png_first.items())})
    # ---- Z5: no injection -> byte-identical
    if not inj:
        sys.path.insert(0, os.path.join(REPO, "eval"))
        import manifest as Mf
        if Mf.content_digest(dh)[0] != Mf.content_digest(dm)[0]:
            bad.append(("Z5", sc))
    # ---- Z6: image, package digest, argv
    if mm["docker_image_id"] != mh["docker_image_id"] or mm["source_digest"] != mh["source_digest"]:
        bad.append(("Z6-image/source", sc))
    if [a for a in mm["argv"] if a != "--ref-snap-mem"] != mh["argv"] or "--ref-snap-mem" not in mm["argv"]:
        bad.append(("Z6-argv", sc, mm["argv"]))
    rev = str(mm.get("git_rev"))
    info["git_rev"], info["git_dirty"] = rev, mm.get("git_dirty")
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        bad.append(("Z6-git", sc, rev))
    sys.path.insert(0, os.path.join(REPO, "eval"))
    import manifest as Mf
    info["content_digest_M"] = Mf.content_digest(dm)[0]
    return dict(bad=bad, info=info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=sorted(CEN))
    a = ap.parse_args()
    res = {sc: check_scene(sc) for sc in a.scenes}
    bad = [b for r in res.values() for b in r["bad"]]
    revs = {r["info"].get("git_rev") for r in res.values()}
    if len(revs) != 1:
        bad.append(("Z6-git-one-rev", sorted(map(str, revs))))
    else:
        rev = next(iter(revs))
        src = subprocess.run(["git", "-C", REPO, "show", f"{rev}:SCSam3/runMVSeg.py"], capture_output=True, text=True).stdout
        chk = subprocess.run(["git", "-C", REPO, "cat-file", "-e", f"{rev}:MVSeed/s2r7_check.py"]).returncode
        if "--ref-snap-mem" not in src or chk != 0:
            bad.append(("Z6-git-rev-lacks-S2-R7", rev))
    tot = sum(r["info"].get("n_injected", 0) for r in res.values())
    out = dict(prereg="docs/stage2-R7-prereg.md", scenes=res, n_bad=len(bad), bad=bad[:100], total_injected=tot,
               total_snaps_arm=sum(r["info"].get("snaps_arm", 0) for r in res.values()),
               content_digest_M={sc: r["info"].get("content_digest_M") for sc, r in res.items()},
               total_start1=sum(r["info"].get("inj_start1", 0) for r in res.values()),
               ok=not bad and len(res) == 17)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    for sc, r in res.items():
        i = r["info"]
        print(f"{sc:20s} injected {i.get('n_injected')}  (start+1 {i.get('inj_start1')})  snaps arm/H {i.get('snaps_arm')}/{i.get('snaps_H')}  "
              f"T {i.get('T')}  bad {r['bad'][:3]}")
    print(f"Z1-Z6: {'OK' if out['ok'] else 'FAIL'}  injected {tot}  start+1 {out['total_start1']}  bad {len(bad)}")
    if not out["ok"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
