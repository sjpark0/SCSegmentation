#!/usr/bin/env python3
"""S1-E15 Z-checks (docs/stage1-E15-prereg.md §8) on the decodes run_e15.sh made -> scripts/e15/runs/zcheck.json (host).

    Z4     DEV-A Dog: `dev_decode --preset geoset_rep_xs --scenes Dog --validate` into the separate root
           Data/_dev/e15/_z4_geoset_rep_xs_A (run_e15.sh z4-dog) = Data/_dev/geoset_rep_xs_A/Dog: every PNG (eval finals,
           c_ini GT, _r1), round1_conf.json and xcl_conf.json byte for byte, and every _validate.json entry equal.
    Z-R    AB(s*) at theta 60 on DEV-A (run_e15.sh dev AB 60 <s*> 0) = geoset_rep_xs_A: every PNG + round1_conf.json +
           xcl_conf.json byte for byte when no DEV-A camera is in F at 60 (angles.json), else the §7.1 case-Z rule; and
           the case-Z variant (eval-cam PNG bytes, eval-cam xcl_conf entries by value, round1_conf.json bytes) at every
           grid theta in [A_E, A_all) other than 60 that was decoded on DEV-A with s* (AB_<theta>_<s*>_A).  Step 2 marks a
           theta >= A_E (inf included) DEV-A-safe only with the whole Z stage passing (e15_select.zcheck_missing).
    Z-I2   the stress decode stress_20_<s*>_A (theta 20; full camera sets where an eval cam is in F): the eval cams in S
           keep C0's final PNG bytes; C0's round-1 cams keep their round-1 PNG bytes and round1_conf entries
           (e15_judge.frame_compare, not case Z).
    Z-sim  id3d_conf.json "round1" = id3d.rejudge of the written round-1 PNGs with the round1_conf presence / reproj record:
           every T06 E15 decode with Fix A states (Result/e15_*), and the DEV-A Dog frames of G(s*) at 60 (G_60_<s*>_A).
    Z-T06  every T06 E15 decode so far (Result/e15_*): I1 (round-1 PNG tree, round1_conf.json bytes = e15_C0) and, in
           scope R, I2 (the star views' final PNGs = e15_C0).
Each record names the compared roots and carries the number and a sha256 digest of the decode files it compared.
Every decoded frame's config must carry the run's relay_theta / id3d / §4 row.  A component whose decode does not exist
is recorded as missing (pass null); a present one that differs anywhere fails (pass false: a bug -- fix, decode again).

    .venv/bin/python scripts/e15/e15_zcheck.py                          -> runs/zcheck.json (exit 2 if a present check fails)
    .venv/bin/python scripts/e15/e15_zcheck.py --t06-invariants <name>  I1 (round-1 PNG tree, round1_conf.json bytes) and,
           in scope R, I2 (the star views' final PNGs) of Result/e15_<name> against Result/e15_C0 (exit 2 = a bug)
"""
import argparse
import hashlib
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, S1)
import numpy as np  # noqa: E402

_spec = importlib.util.spec_from_file_location("e15_judge", os.path.join(HERE, "e15_judge.py"))
J = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(J)
SEL = J.SEL

DATA, OUT = SEL.DATA, SEL.OUT
DEV_C0_A = SEL.DEV_C0_A
E15_DEV = os.path.join(DATA, "_dev", "e15")
Z4_ROOT = os.path.join(E15_DEV, "_z4_geoset_rep_xs_A")
Z4_SCENE, ZSIM_DEV_SCENE = "Dog", "Dog"
T06_RESULT = J.T06_RESULT
MAX_LISTED = 20                                             # differences listed per frame / run (all are counted)


def _bytes(p):
    with open(p, "rb") as f:
        return f.read()


def tree_digest(dirs):
    """(number of files, sha256 over relative path + bytes of every file) under the given folders, sorted."""
    h, n = hashlib.sha256(), 0
    for d in dirs:
        for dp, dns, fs in os.walk(d):
            dns.sort()
            for f in sorted(fs):
                p = os.path.join(dp, f)
                h.update(os.path.relpath(p, os.path.dirname(d)).encode() + b"\0" + _bytes(p))
                n += 1
    return n, h.hexdigest()


def png_tree(d):
    """{relative path: bytes} of every PNG under d (absent dir = {})."""
    out = {}
    for dp, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(".png"):
                p = os.path.join(dp, f)
                out[os.path.relpath(p, d).replace(os.sep, "/")] = _bytes(p)
    return out


def tree_compare(c0_dir, arm_dir, jsons=("round1_conf.json", "xcl_conf.json")):
    """Byte identity of two frame / Result folders: every PNG (same set), and the given json files."""
    a, b = png_tree(c0_dir), png_tree(arm_dir)
    diffs = [("png", k) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    for f in jsons:
        pa, pb = os.path.join(c0_dir, f), os.path.join(arm_dir, f)
        if not (os.path.isfile(pa) and os.path.isfile(pb)) or _bytes(pa) != _bytes(pb):
            diffs.append((f[:-5], "*"))
    return diffs


def config_mismatch(cfg, theta, family, row="AB"):
    """The decoded config's E15 keys (run.RelayRound.config) against the run: relay_theta, id3d, the §4 row."""
    from methods.geomem import run as R
    if cfg is None or "relay_theta" not in cfg:
        return [("config", "no relay_theta")]
    th = cfg["relay_theta"]
    th = math.inf if th == "inf" else float(th)
    fam = None if cfg.get("id3d") in (None, "off") else cfg["id3d"]
    o = dict(family=fam, scope=cfg.get("id3d_scope"), mode=cfg.get("relay_mode"), order=cfg.get("relay_order"),
             mem=cfg.get("relay_mem"), trust=cfg.get("relay_trust"), flagged=cfg.get("flagged"))
    bad = []
    if th != theta:
        bad.append(("relay_theta", cfg["relay_theta"]))
    if fam != family:
        bad.append(("id3d", cfg.get("id3d")))
    if R.e15_row(o) != row:
        bad.append(("row", R.e15_row(o)))
    return bad


def _frames(c0_root=None, scenes=None):
    idx = SEL.load(os.path.join(c0_root or DEV_C0_A, "index.json"))
    for s, S in idx["scenes"].items():
        if scenes is not None and s not in scenes:
            continue
        for t in S["frames"]:
            yield s, int(t), S


def _record(name, diffs, n, n_expected, root=None, c0_root=None, frames=(), **kw):
    n_diff = sum(len(v) for v in diffs.values())
    n_files, digest = tree_digest([os.path.join(root, s, f"{t:06d}") for s, t in frames]) if root else (0, None)
    return dict(run=name, root=root and os.path.relpath(root, S1), c0_root=c0_root and os.path.relpath(c0_root, S1),
                n_frames=n, n_expected=n_expected, n_files=n_files, files_sha256=digest, n_frames_differing=len(diffs),
                n_differences=n_diff, diffs={k: v[:MAX_LISTED] for k, v in list(diffs.items())[:MAX_LISTED]}, **kw,
                **{"pass": bool(n == n_expected and not diffs)})


# ------------------------------------------------------------------------------------------------- Z4 (DEV-A Dog)
def z4(z4_root=None, c0_root=None, scene=Z4_SCENE):
    z4_root, c0_root = z4_root or Z4_ROOT, c0_root or DEV_C0_A
    if not os.path.isdir(os.path.join(z4_root, scene)):
        return None
    diffs, n, n_exp, done = {}, 0, 0, []
    for s, t, _ in _frames(c0_root, [scene]):
        n_exp += 1
        fz, fc = (os.path.join(r, s, f"{t:06d}") for r in (z4_root, c0_root))
        if not os.path.isfile(os.path.join(fz, "DONE")):
            diffs[f"{s}/{t}"] = [("missing", "DONE")]
            continue
        n += 1
        done.append((s, t))
        d = tree_compare(fc, fz)
        if d:
            diffs[f"{s}/{t}"] = d
    vp = os.path.join(z4_root, "_validate.json")
    val = SEL.load(vp) if os.path.isfile(vp) else {}
    val = {k: v for k, v in val.items() if k.startswith(scene + "/")}
    if not val:
        diffs["_validate"] = [("missing", "_validate.json")]
    elif not all(v["equal"] for v in val.values()):
        diffs["_validate"] = [("validate", k) for k, v in val.items() if not v["equal"]]
    return _record(os.path.relpath(z4_root, S1), diffs, n, n_exp, z4_root, c0_root, done, scene=scene, validate=sorted(val))


# ------------------------------------------------------------------------------------------------- Z-R (DEV-A)
def zr(theta, family, summary, c0_root=None):
    """Z-R at one theta -> record or None (not decoded).  mode full: no DEV-A camera in F at theta (every decoded
    camera's PNG, round1_conf, xcl_conf bytes); case_z otherwise (§7.1 rule on the eval cams)."""
    c0_root = c0_root or DEV_C0_A
    name = SEL.run_name("AB", theta, family) + "_A"
    root = os.path.join(E15_DEV, name)
    if not os.path.isfile(os.path.join(root, "index.json")):
        return None
    tag = SEL.theta_tag(theta)
    full = summary["in_F"][tag]["A"]["n_all_in_F"] == 0
    diffs, n, n_exp, done = {}, 0, 0, []
    for s, t, S in _frames(c0_root):
        n_exp += 1
        fa, fc = (os.path.join(r, s, f"{t:06d}") for r in (root, c0_root))
        if not os.path.isfile(os.path.join(fa, "DONE")):
            diffs[f"{s}/{t}"] = [("missing", "DONE")]
            continue
        n += 1
        done.append((s, t))
        d = tree_compare(fc, fa) if full else J.frame_compare(fc, fa, S["eval_cams"], True, [])
        d += config_mismatch(SEL.load(os.path.join(fa, "config.json")), theta, family)
        if d:
            diffs[f"{s}/{t}"] = d
    return _record(name, diffs, n, n_exp, root, c0_root, done, theta=tag, family=family, mode="full" if full else "case_z")


def zr_thetas(A_E, A_all):
    """60, and every finite grid theta in [A_E, A_all) (the case-Z variant candidates for theta_s*)."""
    return [60.0] + [th for th in SEL.THETA_GRID if math.isfinite(th) and A_E <= th < A_all and th != 60.0]


# ------------------------------------------------------------------------------------------------- Z-I2 (stress)
def zi2(family, c0_root=None):
    c0_root = c0_root or DEV_C0_A
    name = SEL.run_name("stress", SEL.STRESS, family) + "_A"
    root = os.path.join(E15_DEV, name)
    if not os.path.isfile(os.path.join(root, "index.json")):
        return None
    diffs, n, n_exp, n_star, n_far, n_frames_F, done = {}, 0, 0, 0, 0, 0, []
    for s, t, S in _frames(c0_root):
        n_exp += 1
        fa, fc = (os.path.join(r, s, f"{t:06d}") for r in (root, c0_root))
        if not os.path.isfile(os.path.join(fa, "DONE")):
            diffs[f"{s}/{t}"] = [("missing", "DONE")]
            continue
        n += 1
        done.append((s, t))
        rp = os.path.join(fa, "relay.json")
        if not os.path.isfile(rp):
            diffs[f"{s}/{t}"] = [("missing", "relay.json")]
            continue
        rel = SEL.load(rp)
        star = [c for c in S["eval_cams"] if c in rel["S"] or not rel["active"]]
        n_star += len(star)
        n_far += len(S["eval_cams"]) - len(star)
        n_frames_F += bool(rel["F"])
        d = J.frame_compare(fc, fa, S["eval_cams"], False, star)
        d += config_mismatch(SEL.load(os.path.join(fa, "config.json")), SEL.STRESS, family)
        if d:
            diffs[f"{s}/{t}"] = d
    return _record(name, diffs, n, n_exp, root, c0_root, done, family=family, n_star_eval_views=n_star,
                   n_far_eval_views=n_far, n_frames_with_F=n_frames_F)


# ------------------------------------------------------------------------------------------------- Z-sim
def _consts(rec):
    c = {k: (math.inf if v == "inf" else v) for k, v in (rec.get("consts") or {}).items()}
    return c if rec["family"] == "cstar" else None


def zsim_one(cache, ref_cam, ref_masks, objects, rec, r1_dir, conf, hw):
    """The decode's round-1 Fix A states vs the host re-judgement -> list of (cam, obj, decode, host)."""
    from geomem import id3d
    cams = sorted(rec["round1"])
    host = id3d.rejudge(cache, ref_cam, ref_masks, objects, rec["family"], r1_dir, conf, cams, hw, consts=_consts(rec),
                        rig_ok=bool(rec.get("rig_ok", True)))
    return [(c, o, s, host[c].get(o)) for c in cams for o, s in rec["round1"][c].items() if host[c].get(o) != s]


def zsim_t06(result=None):
    """Every Result/e15_<name> with Fix A round-1 states (status_source id3d)."""
    from methods import common
    from mvseg2d.io import cache_dir
    result = result or T06_RESULT
    if not os.path.isdir(result):
        return None
    runs = sorted(d for d in os.listdir(result) if d.startswith("e15_") and not d.endswith("_r1")
                  and os.path.isfile(os.path.join(result, d, "id3d_conf.json")))
    if not runs:
        return None
    sc = common.load_scene(DATA, SEL.T06)
    masks = np.stack([sc.ref_labels == o for o in sc.objects])
    cache = cache_dir(DATA, SEL.T06, "pi3")
    out = {}
    for d in runs:
        rec = SEL.load(os.path.join(result, d, "id3d_conf.json"))
        if rec.get("status_source") != "id3d" or not rec.get("round1"):
            continue
        conf = SEL.load(os.path.join(result, d, "round1_conf.json"))
        diff = zsim_one(cache, sc.ref_cam, masks, sc.objects, rec, os.path.join(result, d + "_r1"), conf, sc.hw)
        n_files, digest = tree_digest([os.path.join(result, d + "_r1")])
        out[d] = dict(family=rec["family"], n_cams=len(rec["round1"]), n_diff=len(diff), diffs=diff[:MAX_LISTED],
                      id3d_conf_sha256=SEL.sha256(os.path.join(result, d, "id3d_conf.json")), n_r1_files=n_files,
                      r1_sha256=digest, **{"pass": not diff})
    return dict(runs=out, **{"pass": bool(out) and all(v["pass"] for v in out.values())}) if out else None


def zsim_dev(family, scene=ZSIM_DEV_SCENE, c0_root=None):
    """The DEV-A <scene> frames of G(s*) at theta 60."""
    from geomem.cache import pi3_frame_cache
    from methods.geomem import dev_decode as D
    c0_root = c0_root or DEV_C0_A
    name = SEL.run_name("G", 60.0, family) + "_A"
    root = os.path.join(E15_DEV, name)
    if not os.path.isdir(os.path.join(root, scene)):
        return None
    diffs, n, n_exp, done = {}, 0, 0, []
    for s, t, S in _frames(c0_root, [scene]):
        n_exp += 1
        f = os.path.join(root, s, f"{t:06d}")
        if not os.path.isfile(os.path.join(f, "DONE")) or not os.path.isfile(os.path.join(f, "id3d_conf.json")):
            diffs[f"{s}/{t}"] = [("missing", "DONE / id3d_conf.json")]
            continue
        n += 1
        done.append((s, t))
        sc = D.frame_scene(s, t, S["basic"])
        rec = SEL.load(os.path.join(f, "id3d_conf.json"))
        d = zsim_one(pi3_frame_cache(DATA, s, t), sc.ref_cam, np.stack([sc.ref_labels == o for o in sc.objects]),
                     sc.objects, rec, os.path.join(f, "_r1"), SEL.load(os.path.join(f, "round1_conf.json")), sc.hw)
        d += config_mismatch(SEL.load(os.path.join(f, "config.json")), 60.0, family, row="G")
        if d:
            diffs[f"{s}/{t}"] = d
    return _record(name, diffs, n, n_exp, root, None, done, family=family, scene=scene)


# ------------------------------------------------------------------------------------------------- T06 I1 / I2
def t06_invariants(name, result=None):
    """I1: Result/e15_<name>_r1 PNG tree and round1_conf.json bytes = e15_C0's; I2 (scope R): the star views' final PNGs.
    -> list of differences."""
    result = result or T06_RESULT
    d, d0 = os.path.join(result, f"e15_{name}"), os.path.join(result, "e15_C0")
    diffs = []
    a, b = png_tree(d0 + "_r1"), png_tree(d + "_r1")
    if not a:
        diffs.append(("I1", "e15_C0_r1 has no PNG (run t06-c0 first)"))
    diffs += [("I1 r1_png", k) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    p, p0 = os.path.join(d, "round1_conf.json"), os.path.join(d0, "round1_conf.json")
    if not (os.path.isfile(p) and os.path.isfile(p0)) or _bytes(p) != _bytes(p0):
        diffs.append(("I1 round1_conf", "*"))
    rp = os.path.join(d, "relay.json")
    if not os.path.isfile(rp):
        diffs.append(("relay.json", "missing"))
        return diffs
    rel = SEL.load(rp)
    if rel["options"]["scope"] == "relay":                  # I2 holds in scope R only (G / GN change the star views)
        star = list(rel["S"]) + ([] if rel["active"] else list(rel["F"]))
        fa, fb = png_tree(d0), png_tree(d)
        for k in sorted(set(fa) | set(fb)):
            if k.split("/", 1)[0] in star and fa.get(k) != fb.get(k):
                diffs.append(("I2 png", k))
    return diffs


def zt06(result=None):
    """Z-T06: I1 / I2 of every T06 E15 decode so far against e15_C0 (None: no T06 E15 decode yet)."""
    result = result or T06_RESULT
    if not os.path.isdir(os.path.join(result, "e15_C0")):
        return None
    runs = sorted(d[4:] for d in os.listdir(result) if d.startswith("e15_") and not d.endswith("_r1") and d != "e15_C0"
                  and os.path.isdir(os.path.join(result, d)))
    if not runs:
        return None
    out = {}
    for name in runs:
        diff = t06_invariants(name, result)
        n_files, digest = tree_digest([os.path.join(result, f"e15_{name}"), os.path.join(result, f"e15_{name}_r1")])
        out[name] = dict(n_diff=len(diff), diffs=diff[:MAX_LISTED], n_files=n_files, files_sha256=digest,
                         **{"pass": not diff})
    return dict(runs=out, c0_root=os.path.relpath(os.path.join(result, "e15_C0"), S1),
                **{"pass": all(v["pass"] for v in out.values())})


# ------------------------------------------------------------------------------------------------- all
def run_all():
    sel, ang = SEL.load(os.path.join(SEL.OUT, "select.json")), SEL.load(os.path.join(SEL.OUT, "angles.json"))
    if not sel.get("s_star"):
        raise SystemExit("Step 1 found no feasible statistic: no Fix-A Z-check (§6.2)")
    fam = sel["s_star"]["family"]
    summ = ang["summary"]
    A_E, A_all = summ["A_E"]["a"], summ["A_all"]["a"]
    res = dict(prereg="docs/stage1-E15-prereg.md §6.3b, §8", family=fam, A_E=A_E, A_all=A_all,
               Z4=z4(), ZR={}, ZI2=zi2(fam), Zsim=dict(T06=zsim_t06(), DEV_A_G=zsim_dev(fam)), ZT06=zt06())
    for th in zr_thetas(A_E, A_all):
        r = zr(th, fam, summ)
        if r is not None:
            res["ZR"][SEL.theta_tag(th)] = r
    items = list(SEL.ZSTAGE) + [f"ZR {t}" for t in res["ZR"] if t != "60"]
    res["missing"] = [k for k in SEL.ZSTAGE if SEL.zcheck_item(res, k) is None]
    res["pass_present"] = all(SEL.zcheck_item(res, k)["pass"] for k in items if SEL.zcheck_item(res, k) is not None)
    res["stage_pass"] = bool(not res["missing"] and res["pass_present"])          # the §6.3b Z stage (Step 2 reads it)
    SEL.save(SEL.ZCHECK, res)
    for k in items:
        v = SEL.zcheck_item(res, k)
        print(f"{k:14s} {'missing' if v is None else ('PASS' if v['pass'] else 'FAIL')}", flush=True)
    if not res["pass_present"]:
        raise SystemExit(2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t06-invariants", default=None, dest="t06_inv", metavar="NAME",
                    help="I1 / I2 of Result/e15_<NAME> against Result/e15_C0 (exit 2 on a difference)")
    a = ap.parse_args()
    if a.t06_inv:
        d = t06_invariants(a.t06_inv)
        if d:
            print(f"T06 e15_{a.t06_inv}: invariant differences (a bug, prereg §8): {d[:MAX_LISTED]}", flush=True)
            raise SystemExit(2)
        print(f"T06 e15_{a.t06_inv}: I1 / I2 hold", flush=True)
        return
    run_all()


if __name__ == "__main__":
    main()
