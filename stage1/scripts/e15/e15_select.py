#!/usr/bin/env python3
"""S1-E15 constant selection (docs/stage1-E15-prereg.md §6.1-§6.5).  Commit this script before Step 0 (§11.2).

    --step angles  N1 (§3.1, §6.1, L1): a_v of every camera under the decode's own inputs (c_ini masks + Pi3 poses only;
                   no target-view GT, no prediction) on DEV-A / DEV-B / keyframes (original c_ini) and DEV-A' / DEV-B'
                   (rotated input camera, protocol §4); A_E, A_all, per-scene maxima, object vs scene anchor, the eval cams
                   in F for every theta in Theta + {20}, the rotated-set distribution and N2 (training ordered pairs)
                   -> runs/angles.json.  Exit 2 when A_E >= 60 (stop; amend the registration).  Host (numpy).
    --step rows    the per-mask Fix A quantities that Step 0 / Step 1 decide from -> geomem/runs/_e15/rows.json.gz (not
                   tracked): DEV-A GT and C0 round-1 (+ the round-1 memory candidates, N6), DEV-A' GT and synthetic swaps,
                   T06 C0 round-1 / final.  A-h needs the reprojection of GT masks: torch on CPU (scsam3 without --gpus).
    --step 0       angles + rows + the N2-N6 reproduction and the P0 check -> runs/step0.json.
    --step 1       F1-F3, the objective and the tie-breaks over the 652 candidates -> s*, c_best -> runs/select.json (host).
                   Then the Step-1 commit: runs/{angles,step0,select}.json + id3d.CSTAR = c_best, CSTAR_SELECTED = True.
    --step 2       theta_s* (§6.4) from the T06 AB(theta) scores, the DEV-A' compares and, for theta < A_E only, the DEV-A
                   full decodes -> runs/step2.json (host).
    --step 2b      the statistic swap check (§6.5; a swap re-checks the §6.4 DEV-A' conditions at theta_s* for s', and
                   the DEV-A safety when theta_s* < A_E) -> runs/step2b.json (host).
    --step rescore-check --split A|B
                   the C0 re-scoring with the current dev_score (covers / other_obj_mask, read by Step 2 and stage B) must
                   reproduce the E14 J, F, J&F of Data/_dev/geoset_rep_xs_<split>/dev_pairs.json pair by pair
                   -> runs/c0_rescore.json (host; the re-score itself: run_e15.sh c0-rescore <split>).
    --step check-step1 --family F
                   the Step-1 commit guard of a Fix-A decode (run.e15_require_step1; run_e15.sh calls it on the host).
The Z stage (§6.3b: Z4 Dog, Z-R, Z-I2, Z-sim, Z-T06) is scripts/e15/e15_zcheck.py -> runs/zcheck.json; Step 2 marks a
theta >= A_E (inf included) DEV-A-safe only with the whole stage passing there for s*, plus the case-Z Z-R at theta when
theta < A_all (else need_zcheck); the C0 fill rule (§6.4) is used only after Z-I2 passed.
T06 C0 (Step 0 input, the byte reference of the T06 C0 re-decode) = the ring-diagnosis D1 decode persisted at
geomem/runs/_e15/d1_ref/{preset,preset_r1} (untracked) and checked against the tracked manifest
scripts/e15/design/d1_t06_c0.sha256 (prereg §6.1).

Names (shared with run_e15.sh / e15_judge.py): a decode of arm <arm> at threshold theta with family <fam> is
"<arm>_<theta>_<fam>" (theta "35" / "inf"; fam "off" for B-rep / B-obj / oracle): T06 Result folder e15_<name>, T06
score-json arm <name>; DEV root Data/_dev/e15/<name>_<split>[_rot<r>] and its dev_score json
geomem/runs/_e15/scores/<root basename>.json; C0: geoset_rep_xs_<split>[_rot<r>].
"""
import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(S1)
sys.path.insert(0, S1)
import numpy as np  # noqa: E402

from geomem import devset, id3d  # noqa: E402
from geomem.id3d import ABSTAIN, EMPTY, FLAG, PASS  # noqa: E402
from mvseg2d.io import load_meta  # noqa: E402

DATA = os.path.join(S1, "Data")
OUT = os.path.join(HERE, "runs")                                   # tracked: the Step-0/1/2 records
WORK = os.path.join(S1, "geomem", "runs", "_e15")                  # untracked: rows, scores
ROWS = os.path.join(WORK, "rows.json.gz")
SCORES = os.path.join(WORK, "scores")
DEV_C0_A = os.path.join(DATA, "_dev", "geoset_rep_xs_A")
T06, T06_REF = "T06_flag_ganggang_F", "cam_06"
T06_GT = os.path.join(REPO, "Data", "PlenopticFull", T06, "Mask", "{cam}", "000900.png")
# the D1 decode of the ring diagnosis (§6.1: T06 C0 = D1 output; final = NAS bytes); a later T06 C0 re-decode must match it.
# Persisted (untracked, prereg §6.1) from the diagnosis session's scratch ring_d15/data/<T06>/Result/{preset,preset_r1};
# the tracked manifest holds the sha256 of each of its files.
T06_C0 = os.path.join(WORK, "d1_ref")
T06_D1_MANIFEST = os.path.join(HERE, "design", "d1_t06_c0.sha256")
C0_E14 = os.path.join(DATA, "_dev", "geoset_rep_xs_{split}", "dev_pairs.json")   # E14 C0 scorings (J, F, J&F reference)
ZCHECK = os.path.join(OUT, "zcheck.json")                                         # scripts/e15/e15_zcheck.py
FISHEYE = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder")
THETA_GRID = (35.0, 40.0, 45.0, 50.0, 55.0, 60.0, math.inf)        # §6.4 Theta
STRESS = 20.0
THETAS_F = (20.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0)              # Theta + {20}, finite (eval cams in F per theta)
A_E_STOP = 60.0                                                     # §3.1: A_E >= 60 breaks the fallback-(a) premise
WIDE = 55.0                                                         # §6.2 tie-break 1: DEV-A' pairs with a_v >= 55
ANGLE_BINS = (0.0, 20.0, 35.0, 45.0, 55.0, 60.0, 75.0, 90.0, 180.0)
F1_ALL, F1_SCENE, F1_FISH, F2_ALL, F3_PASS, F3_WRONG = 0.003, 0.05, 0.005, 0.003, 0.9, 3   # §6.2
WRONG_C0, DEV_LO, DEV_SCENE, J_TOL = 28, -0.003, -0.010, 0.01      # §6.4
SWAP_J = 0.02                                                       # §6.5
FAMILY = {"vote": "A-v", "vote_inf": "A-v", "cstar": "A-c*", "hybrid": "A-h", "c0": "A-c0"}
FAMILY_ORDER = ("A-v", "A-c*", "A-h", "A-c0")                       # §6.2 tie-break 4
# P1's predicted A-c* constants (§5.2) = what P0 reproduces at Step 0; pinned here so that a re-run after the Step-1
# commit (id3d.CSTAR = c_best) still checks the registered P0 numbers
P1_CSTAR = dict(beta=1.0, kappa=1.25, sigma=0.05, rho_env=0.5, b_max=0.01, gamma=math.inf, phi_rig=math.inf)
P0 = dict(cstar_gt_ff=(5, 3), t06_far_wrong_flag_cstar=(24, 2), t06_far_wrong_flag_vote=(29, 2), A_E=(47.1, 0.2),
          A_all=(58.12, 0.5))                                       # §5.2 P0 (reproduction check): value, tolerance


# ------------------------------------------------------------------------------------------------- small helpers
def theta_tag(theta):
    return "inf" if math.isinf(theta) else f"{theta:g}"


def run_name(arm, theta, family):
    return f"{arm}_{theta_tag(theta)}_{family or 'off'}"


def _jsonable(x):
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return _jsonable(float(x))
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=1, allow_nan=False)


def load(path):
    with open(path) as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ring_d(cam, ref=T06_REF, n=32):
    i, r = int(cam[4:]), int(ref[4:])
    return min(abs(i - r), n - abs(i - r))


def muvod_scenes():
    return devset.dev_scenes()


def check_manifest(root=T06_C0, manifest=T06_D1_MANIFEST):
    """`root` holds exactly the files of the sha256 manifest (paths relative to root) under its top folders, byte for byte.
    -> dict(ok, n, bad: [(path, why)])."""
    want = {}
    with open(manifest) as f:
        for line in f:
            if line.strip():
                h, rel = line.rstrip("\n").split("  ", 1)
                want[rel] = h
    bad = []
    for rel, h in sorted(want.items()):
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            bad.append((rel, "missing"))
        elif sha256(p) != h:
            bad.append((rel, "sha256 differs"))
    for top in sorted({rel.split("/", 1)[0] for rel in want}):
        for dp, _, fs in os.walk(os.path.join(root, top)):
            for fn in fs:
                rel = os.path.relpath(os.path.join(dp, fn), root).replace(os.sep, "/")
                if rel not in want:
                    bad.append((rel, "not in the manifest"))
    return dict(ok=not bad, n=len(want), bad=bad)


def require_t06_c0(t06_c0):
    """Step 0 / rows read the T06 C0 decode only when it is the D1 decode (tracked manifest)."""
    chk = check_manifest(t06_c0)
    if not chk["ok"]:
        raise SystemExit(f"{t06_c0} is not the D1 decode of {T06_D1_MANIFEST}: {chk['bad'][:5]} (prereg §6.1)")
    return chk


# ------------------------------------------------------------------------------------------------- frames and inputs
SETS = ("A", "B", "K", "A'", "B'")                                  # K = the keyframe (TEST: inputs only, original c_ini)


def frame_specs(scenes, sets=SETS):
    """(set, scene, t, rot) of every frame-reference the angles cover; rotated keyframes are never listed."""
    out = []
    for s in scenes:
        for st in sets:
            if st == "K":
                out.append(("K", s, int(load_meta(DATA, s)["ref_frame"]), 0))
                continue
            split = st[0]
            for t in devset.dev_frames(DATA, s, split):
                for r in ((0,) if len(st) == 1 else (1, 2)):
                    out.append((st, s, int(t), r))
    return out


def frame_inputs(spec):
    """The decode's inputs of one frame-reference -> (cache, cams, ref, masks [B,H,W] bool, objects, ref_labels):
    DEV frames as methods.geomem.dev_decode.frame_scene (basic set, rotation); the keyframe as its decode (load_scene)."""
    from methods import common
    from methods.geomem import dev_decode as D
    from mvseg2d.io import cache_dir
    from geomem.cache import pi3_frame_cache
    st, s, t, rot = spec
    if st == "K":
        sc = common.load_scene(DATA, s)
        cache = cache_dir(DATA, s, "pi3")
    else:
        sc = D.frame_scene(s, t, devset.basic_set(s), rot)
        cache = pi3_frame_cache(DATA, s, t)
    masks = np.stack([sc.ref_labels == o for o in sc.objects]) if sc.objects else np.zeros((0,) + tuple(sc.hw), bool)
    return cache, list(sc.cams), sc.ref_cam, masks, list(sc.objects), sc.ref_labels


def scene_anchor(cache, ref):
    """The ring diagnosis' scene anchor: median of every valid c_ini Pi3 point (world) -> X or None."""
    from geomem.pointmap import CONF_MIN, local_points
    z = np.load(os.path.join(cache, f"{ref}.npz"))
    valid = (z["conf"] > CONF_MIN) & (z["depth"] > 0)
    if not valid.any():
        return None
    T = np.asarray(z["c2w"], np.float64)
    return T[:3, :3] @ np.median(local_points(z["depth"], z["K"])[valid], 0) + T[:3, 3]


def frame_angles(spec):
    """N1 of one frame-reference: object-anchor a_v (run.relay_geometry, the decode's code) and scene-anchor a_v."""
    from methods.geomem import run as R
    st, s, t, rot = spec
    cache, cams, ref, masks, objects, _ = frame_inputs(spec)
    g = R.relay_geometry(cache, cams, ref, masks, math.inf)
    Xs = scene_anchor(cache, ref)
    ang_s = id3d.view_angles(g["centres"], ref, Xs) if Xs is not None else {}
    evals = devset.eval_cams(DATA, s, rot)
    rec = dict(set=st, scene=s, t=t, rot=rot, ref=ref, kind=g["kind"], fallback=g["fallback"], n_objects=len(objects),
               all=g["angles"], eval={c: g["angles"][c] for c in evals if c in g["angles"]},
               eval_scene={c: ang_s[c] for c in evals if c in ang_s})
    if g["X"] is not None and Xs is not None:
        rec["anchor_dist"] = float(np.linalg.norm(np.asarray(g["X"]) - Xs))
        rec["max_dangle"] = float(max([abs(g["angles"][c] - ang_s[c]) for c in g["angles"]] or [0.0]))
    return rec


def _max_entry(entries, key):
    best = None
    for e in entries:
        for c, a in e[key].items():
            if best is None or a > best[0]:
                best = (a, e["set"], e["scene"], e["t"], c)
    return None if best is None else dict(a=best[0], set=best[1], scene=best[2], t=best[3], cam=best[4])


def angle_summary(entries):
    """A_E / A_all over the original-reference sets (A, B, K), per-set and per-scene maxima, the anchors, the eval /
    all cams in F per theta, the rotated sets' distribution."""
    orig = [e for e in entries if e["set"] in ("A", "B", "K")]
    out = dict(A_E=_max_entry(orig, "eval"), A_all=_max_entry(orig, "all"), by_set={}, per_scene={}, in_F={},
               anchors={}, rotated={})
    for st in SETS:
        es = [e for e in entries if e["set"] == st]
        if not es:
            continue
        out["by_set"][st] = dict(n_frames=len(es), eval_max=_max_entry(es, "eval"), all_max=_max_entry(es, "all"))
        out["anchors"][st] = dict(kinds=dict(Counter(e["kind"] for e in es)),
                                  max_anchor_dist=max([e.get("anchor_dist", 0.0) for e in es]),
                                  max_dangle=max([e.get("max_dangle", 0.0) for e in es]))
        per = defaultdict(dict)
        for e in es:
            p = per[e["scene"]]
            p["eval"] = max([p.get("eval", 0.0)] + list(e["eval"].values()))
            p["all"] = max([p.get("all", 0.0)] + list(e["all"].values()))
        out["per_scene"][st] = dict(per)
    for th in THETAS_F:
        rec = {}
        for st in ("A", "B", "K"):
            es = [e for e in entries if e["set"] == st]
            ev = [[e["scene"], e["t"], c, a] for e in es for c, a in e["eval"].items() if a > th]
            al = sum(a > th for e in es for a in e["all"].values())
            rec[st] = dict(eval_in_F=ev, n_eval_in_F=len(ev), n_all_in_F=int(al))
        out["in_F"][theta_tag(th)] = rec
    for st in ("A'", "B'"):
        es = [e for e in entries if e["set"] == st]
        if not es:
            continue
        a = np.array([x for e in es for x in e["eval"].values()])
        hist = np.histogram(a, bins=ANGLE_BINS)[0] if a.size else np.zeros(len(ANGLE_BINS) - 1)
        out["rotated"][st] = dict(n_pairs_views=int(a.size), bins=list(ANGLE_BINS), hist=[int(x) for x in hist],
                                  n_wide=int((a >= WIDE).sum()), max=float(a.max()) if a.size else None,
                                  wide_by_scene=dict(Counter(e["scene"] for e in es for x in e["eval"].values() if x >= WIDE)))
    return out


def n2_pairs(entries, scenes, stride=None):
    """N2 (report only): viewpoint change of the training samples (geomem/data.py enumerate_samples: every ordered pair of
    annotated cameras on train_frames[::frame_stride]).  From the angle entries: (r, v) at frame t = the eval angle of v
    in the entry whose input camera is r.  Rotated keyframe pairs are TEST inputs and are left out (counted)."""
    from geomem.cache import train_frames
    idx = {(e["scene"], e["t"], e["ref"]): e for e in entries}
    a, per, excluded = [], {}, 0
    for s in scenes:
        st = stride or frame_stride_of(s)
        meta = load_meta(DATA, s)
        cams = list(meta["mask_cams"])
        vals = []
        for t in train_frames(DATA, s)[::max(1, int(st))]:
            for r in cams:
                e = idx.get((s, int(t), r))
                if e is None:
                    excluded += len(cams) - 1
                    continue
                vals += [e["eval"][v] for v in cams if v != r and v in e["eval"]]
        a += vals
        per[s] = dict(n=len(vals), max=float(max(vals)) if vals else None,
                      frac_over_45=float(np.mean(np.array(vals) > 45)) if vals else None)
    a = np.array(a)
    return dict(n=int(a.size), excluded_rotated_keyframe_pairs=int(excluded), frac_over_45=float((a > 45).mean()) if a.size else None,
                quantiles={q: float(np.percentile(a, q)) for q in (50, 90, 99, 100)} if a.size else {}, per_scene=per)


def frame_stride_of(scene):
    """The training frame stride of the scene's own-fold geoset_pred_ft checkpoint (config.json; 1 when absent)."""
    from geomem.folds import load_folds
    p = os.path.join(S1, "geomem", "runs", "geoset_pred_ft", f"fold{load_folds()[scene]}", "config.json")
    try:
        return int(load(p).get("frame_stride") or 1)
    except OSError:
        return 1


def step_angles(scenes, workers):
    specs = frame_specs(scenes)
    entries = _pmap(frame_angles, specs, workers)
    summ = angle_summary(entries)
    res = dict(prereg="docs/stage1-E15-prereg.md §6.1 N1", inputs="c_ini masks + Pi3 poses only", scenes=scenes,
               summary=summ, n2=n2_pairs(entries, scenes), entries=entries)
    save(os.path.join(OUT, "angles.json"), res)
    A_E, A_all = summ["A_E"], summ["A_all"]
    print(f"A_E {A_E['a']:.2f} ({A_E['set']} {A_E['scene']} t={A_E['t']} {A_E['cam']})  "
          f"A_all {A_all['a']:.2f} ({A_all['set']} {A_all['scene']} t={A_all['t']} {A_all['cam']})", flush=True)
    for th in ("20", "35", "40", "45", "50", "55", "60"):
        f = summ["in_F"][th]
        print(f"  theta {th:>3}: eval cams in F  A {f['A']['n_eval_in_F']:3d}  B {f['B']['n_eval_in_F']:3d}  "
              f"K {f['K']['n_eval_in_F']:3d}   all cams in F  A {f['A']['n_all_in_F']} B {f['B']['n_all_in_F']} K {f['K']['n_all_in_F']}")
    if A_E["a"] >= A_E_STOP:
        print(f"STOP (§3.1): A_E = {A_E['a']:.2f} >= {A_E_STOP}: the 'byte-identical by construction' premise of the "
              "fallbacks breaks; amend the registration before any DEV decode", flush=True)
        raise SystemExit(2)
    return res


def _pmap(fn, items, workers):
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(fn, items, chunksize=1)


# ------------------------------------------------------------------------------------------------- rows (per-mask quantities)
def light_models(models):
    """What the decide functions read of a reference model: has, rho, border (int keys)."""
    return {int(o): dict(has=bool(m["has"]), rho=m["rho"], border=float(m["border"])) for o, m in models.items()}


def mask_quantities(Q, models, vref):
    """centroid_features + the A-v label counts at r_max 0.2 and inf (from one vote_nearest pass)."""
    feat = id3d.centroid_features(Q, models)
    out = dict(n=feat["n"])
    if feat["n"] >= id3d.N_MIN:
        out.update(spread=feat["spread"], d={int(j): v for j, v in feat["d"].items()})
        near, dist = id3d.vote_nearest(Q, vref)
        out["nq"] = int(len(near))
        out["v02"] = {int(k): int(v) for k, v in Counter(id3d.vote_threshold((near, dist), id3d.VOTE_RMAX["vote"]).tolist()).items()}
        out["vinf"] = {int(k): int(v) for k, v in Counter(id3d.vote_threshold((near, dist), math.inf).tolist()).items()}
    return out


def frame_key(spec, ref):
    st, s, t, _ = spec
    return f"{st}|{s}|{t}|{ref}"


def _row(kind, spec, ref, cam, o, mask, presence, q, **kw):
    st, s, t, rot = spec
    return dict(kind=kind, set=st, scene=s, t=t, rot=rot, ref=ref, frame=frame_key(spec, ref), cam=cam, o=int(o),
                n_px=int(mask.sum()), presence=float(presence), **q, **kw)


def _read_png(path, hw):
    from mvseg2d.io import read_pred_mask
    if not os.path.isfile(path):
        return np.zeros(hw, bool)
    m = read_pred_mask(path)
    assert m.shape == tuple(hw), (path, m.shape, hw)
    return m


def _overlap_torch(cache, cam, ref, hw, preds, ref_masks):
    """run.reproj_overlap on CPU (the decode's A-h input) -> (overlap [B] float with nan, n_valid [B] int)."""
    import torch
    from methods.geomem import run as R
    torch.set_num_threads(1)                                  # one frame per worker process
    with torch.no_grad():
        ov, nv = R.reproj_overlap(torch.from_numpy(np.asarray(preds, bool)), *R.reproj_inputs(cache, cam, ref, hw, "cpu"),
                                  torch.from_numpy(np.asarray(ref_masks, bool)))
    return ov.numpy().astype(np.float64), nv.numpy().astype(np.int64)


def frame_rows(spec, angles=None, overlap_fn=_overlap_torch, dev_c0=DEV_C0_A, wide=WIDE):
    """Rows of one DEV frame-reference: GT masks of the eval cams (A, A'), synthetic swaps on wide A' pairs, C0 round-1
    masks of the eval cams and the round-1 memory candidates (A).  -> (frame record, rows)."""
    from geomem.pointmap import ref_frame_points
    from mvseg2d.io import load_labelmap
    from methods import common
    st, s, t, rot = spec
    cache, cams, ref, masks, objects, _ = frame_inputs(spec)
    P_r, valid_r, _ = ref_frame_points(cache, ref, ref)
    models = id3d.ref_models(masks, objects, P_r, valid_r)
    vref = id3d.vote_reference(models)
    med, _ = id3d.rig_phi(cache, ref, list(load_meta(DATA, s)["cams"]))
    frame = dict(key=frame_key(spec, ref), set=st, scene=s, t=t, rot=rot, ref=ref, models=light_models(models), rig_phi=med)
    fdir = common.frame_dir(DATA, s, t)
    rows = []
    hw = masks.shape[1:]
    lift_cache = {}

    def lift(cam, m):
        if cam not in lift_cache:
            lift_cache[cam] = ref_frame_points(cache, cam, ref)[:2]
        P, v = lift_cache[cam]
        return id3d.lift(id3d.mask_to_work(m, P.shape[:2]), P, v)
    for cam in devset.eval_cams(DATA, s, rot):
        gt_lab = load_labelmap(os.path.join(fdir, "masks", f"{cam}.png"))
        a_v = (angles or {}).get(cam)
        pairs = [o for o in objects if (gt_lab == o).any()]
        if not pairs:
            continue
        gts = np.stack([gt_lab == o for o in pairs])
        refm = np.stack([masks[objects.index(o)] for o in pairs])
        ov, nv = overlap_fn(cache, cam, ref, hw, gts, refm)
        for i, o in enumerate(pairs):
            rows.append(_row("gt", spec, ref, cam, o, gts[i], 1.0, mask_quantities(lift(cam, gts[i]), models, vref),
                             a=a_v, overlap=float(ov[i]), n_valid=int(nv[i])))
        if st == "A'" and a_v is not None and a_v >= wide:              # tie-break 1 swaps (§6.2)
            sw, swo = [], []
            for o in pairs:
                if not models[o]["has"]:
                    continue
                cand = sorted((float(np.linalg.norm(models[o]["c"] - models[j]["c"])), j) for j in models
                              if j != o and models[j]["has"] and (gt_lab == j).any())
                if cand:
                    sw.append(gt_lab == cand[0][1])
                    swo.append((o, cand[0][1]))
            if sw:
                refm = np.stack([masks[objects.index(o)] for o, _ in swo])
                ov, nv = overlap_fn(cache, cam, ref, hw, np.stack(sw), refm)
                for i, (o, j) in enumerate(swo):
                    rows.append(_row("swap", spec, ref, cam, o, sw[i], 1.0, mask_quantities(lift(cam, sw[i]), models, vref),
                                     a=a_v, true=int(j), overlap=float(ov[i]), n_valid=int(nv[i])))
        if st == "A":                                                      # C0 round-1 of the eval cams (F2, N3-N4)
            d = os.path.join(dev_c0, s, f"{t:06d}")
            conf = load(os.path.join(d, "round1_conf.json"))
            for o in pairs:
                m = _read_png(os.path.join(d, "_r1", cam, f"{o}.png"), hw)
                c = conf[cam][str(o)]
                rows.append(_row("r1", spec, ref, cam, o, m, c["presence"], mask_quantities(lift(cam, m), models, vref),
                                 a=a_v, covers=id3d.covers(m, gt_lab), iou=c["iou"],
                                 overlap=float("nan") if c["overlap"] is None else c["overlap"], n_valid=c["n_valid"]))
    if st == "A":                                                          # N6: round-1 memory candidates (no GT)
        d = os.path.join(dev_c0, s, f"{t:06d}")
        cfg, conf = load(os.path.join(d, "config.json")), load(os.path.join(d, "round1_conf.json"))
        for cam in sorted({n for v in cfg["k3"].values() for n in v}):
            for o in objects:
                c = conf[cam][str(o)]
                if c["presence"] <= 0:
                    continue
                m = _read_png(os.path.join(d, "_r1", cam, f"{o}.png"), hw)
                rows.append(_row("mem", spec, ref, cam, o, m, c["presence"], mask_quantities(lift(cam, m), models, vref),
                                 iou=c["iou"], overlap=float("nan") if c["overlap"] is None else c["overlap"],
                                 n_valid=c["n_valid"]))
    return frame, rows


def t06_rows(t06_c0=T06_C0):
    """T06 frame 900 (contaminated dev): C0 round-1 and final masks of every (cam, person) pair of score.py, with covers
    (1/4 resolution) and the ring distance."""
    from PIL import Image
    from geomem.pointmap import ref_frame_points
    from methods import common
    from mvseg2d.io import cache_dir
    spec = ("T06", T06, 900, 0)
    sc = common.load_scene(DATA, T06)
    assert sc.ref_cam == T06_REF
    cache = cache_dir(DATA, T06, "pi3")
    masks = np.stack([sc.ref_labels == o for o in sc.objects])
    P_r, valid_r, _ = ref_frame_points(cache, T06_REF, T06_REF)
    models = id3d.ref_models(masks, sc.objects, P_r, valid_r)
    vref = id3d.vote_reference(models)
    med, _ = id3d.rig_phi(cache, T06_REF, list(sc.cams))
    frame = dict(key=frame_key(spec, T06_REF), set="T06", scene=T06, t=900, rot=0, ref=T06_REF,
                 models=light_models(models), rig_phi=med)
    conf = load(os.path.join(t06_c0, "preset", "round1_conf.json"))
    rows = []
    for cam in sc.cams:
        if cam == T06_REF:
            continue
        gt = np.array(Image.open(T06_GT.format(cam=cam)))
        H, W = gt.shape
        gq = np.array(Image.fromarray(gt.astype(np.uint8)).resize((W // 4, H // 4), Image.NEAREST))
        P, v, _ = ref_frame_points(cache, cam, T06_REF)
        for o in sc.objects:
            if not (gq == o).any():
                continue
            for kind, sub in (("r1", "preset_r1"), ("final", "preset")):
                m = _read_png(os.path.join(t06_c0, sub, cam, f"{o}.png"), (H, W))
                q = mask_quantities(id3d.lift(id3d.mask_to_work(m, P.shape[:2]), P, v), models, vref)
                c = conf[cam][str(o)]
                extra = dict(overlap=float("nan") if c["overlap"] is None else c["overlap"], n_valid=c["n_valid"],
                             iou=c["iou"]) if kind == "r1" else {}
                rows.append(_row(kind, spec, T06_REF, cam, o, m, c["presence"] if kind == "r1" else (1.0 if m.any() else -1.0), q,
                                 covers=id3d.covers(m, gt), ring_d=ring_d(cam), **extra))
    return frame, rows


def _frame_rows_job(args):
    spec, angles = args
    return frame_rows(spec, angles)


def step_rows(scenes, workers, angles_rec, t06_c0):
    import importlib.util
    require_t06_c0(t06_c0)
    if importlib.util.find_spec("torch") is None:            # not imported here: the workers fork from this process
        raise SystemExit("--step rows needs torch for the A-h reprojection of GT masks: run it in scsam3 without --gpus "
                         "(scripts/e15/run_e15.sh step0)")
    ang = {(e["set"], e["scene"], e["t"], e["rot"]): e["eval"] for e in angles_rec["entries"]}
    specs = [s for s in frame_specs(scenes, ("A", "A'"))]
    res = _pmap(_frame_rows_job, [(s, ang.get(s)) for s in specs], workers)
    frames, rows = [f for f, _ in res], [r for _, rs in res for r in rs]
    f6, r6 = t06_rows(t06_c0)
    frames.append(f6)
    rows += r6
    os.makedirs(WORK, exist_ok=True)
    with gzip.open(ROWS, "wt") as f:
        json.dump(_jsonable(dict(t06_c0=t06_c0, scenes=scenes, frames=frames, rows=rows)), f)
    print(f"rows: {len(rows)} ({dict(Counter((r['set'], r['kind']) for r in rows))}) -> {ROWS}", flush=True)
    return frames, rows


def load_rows(path=ROWS):
    """rows.json.gz -> (frames {key: frame}, rows) with int keys / float nan restored."""
    with gzip.open(path, "rt") as f:
        x = json.load(f)
    frames = {}
    for fr in x["frames"]:
        fr["models"] = {int(k): v for k, v in fr["models"].items()}
        frames[fr["key"]] = fr
    for r in x["rows"]:
        for k in ("d", "v02", "vinf"):
            if k in r:
                r[k] = {int(j): v for j, v in r[k].items()}
        if r.get("overlap") == "nan":
            r["overlap"] = float("nan")
    return frames, x["rows"]


# ------------------------------------------------------------------------------------------------- decisions
def candidates():
    """The 652 registered candidates (§6.2): A-c0, A-c* grid (CSTAR_GRID product order), A-v r_max 0.2 / inf, A-h."""
    out = [dict(family="c0", consts=None)]
    out += [dict(family="cstar", consts=dict(zip(id3d.CSTAR_GRID, v))) for v in itertools.product(*id3d.CSTAR_GRID.values())]
    out += [dict(family="vote", consts=None), dict(family="vote_inf", consts=None), dict(family="hybrid", consts=None)]
    for i, c in enumerate(out):
        c["index"] = i
    return out


def cand_name(c):
    if c["family"] != "cstar":
        return c["family"]
    k = c["consts"]
    return "cstar[" + ",".join(f"{n}={theta_tag(float(k[n]))}" for n in id3d.CSTAR_GRID) + "]"


def vote_from_counts(counts, nq, n, o, models):
    """id3d.vote_decide from the label counts of the query subsample (same rule, same ties)."""
    if n < id3d.N_MIN:
        return ABSTAIN
    if not models.get(o, {}).get("has"):
        return ABSTAIN
    ov3d = counts.get(o, 0) / nq
    ids = sorted(j for j in models if models[j]["has"])
    cs = [counts.get(j, 0) for j in ids]
    plur = ids[int(np.argmax(cs))] if sum(cs) else 0
    if ov3d >= id3d.THETA_V:
        return PASS
    if plur not in (0, o):
        return FLAG
    return ABSTAIN


def decide(row, cand, frame):
    """The decode's Fix A state of one row under one candidate (Judge.statuses order: EMPTY, rig, the statistic)."""
    if row["presence"] <= 0 or row["n_px"] == 0:
        return EMPTY
    fam, models, o = cand["family"], frame["models"], row["o"]
    if fam == "cstar":
        rig_ok = math.isinf(cand["consts"]["phi_rig"]) or frame["rig_phi"] <= cand["consts"]["phi_rig"]
        if not rig_ok:
            return ABSTAIN
        return id3d.cstar_decide(row, o, models, rig_ok=True, **cand["consts"])[0]
    if fam == "c0":
        return id3d.c0_decide(row, o, models)[0]
    key = "vinf" if fam == "vote_inf" else "v02"
    sv = vote_from_counts(row.get(key, {}), row.get("nq", 0), row["n"], o, models)
    if fam != "hybrid":
        return sv
    return id3d.hybrid_decide(sv, {}, row["n_px"], row["n_valid"], row["overlap"])[0]


def trusted(row, state):
    """id3d.trust of one row (presence > 0 & (PASS | ABSTAIN & the adopted reproj rule))."""
    rep = bool(id3d.reproj_rule([row["presence"]], [row.get("iou", 0.0)], [row["overlap"]])[0])
    return row["presence"] > 0 and (state == PASS or (state == ABSTAIN and rep))


# ------------------------------------------------------------------------------------------------- Step 1 (§6.2)
def step1_sets(rows):
    """The Step-1 data: GT (DEV-A + DEV-A'), DEV-A C0 round-1 correct, T06 round-1 far half (correct / wrong), swaps."""
    gt = [r for r in rows if r["kind"] == "gt" and r["set"] in ("A", "A'")]
    r1c = [r for r in rows if r["kind"] == "r1" and r["set"] == "A" and r["covers"] == r["o"]]
    t06 = [r for r in rows if r["kind"] == "r1" and r["set"] == "T06" and r["ring_d"] >= 7]
    return dict(gt=gt, r1_correct=r1c, t06_correct=[r for r in t06 if r["covers"] == r["o"]],
                t06_wrong=[r for r in t06 if r["covers"] not in (0, r["o"])],
                swap=[r for r in rows if r["kind"] == "swap" and r["set"] == "A'" and r["a"] is not None and r["a"] >= WIDE])


def metrics(cand, sets, frames):
    """F1-F3 quantities, the objective and tie-break 1 of one candidate."""
    def st(r):
        return decide(r, cand, frames[r["frame"]])
    gt_flag = [st(r) == FLAG for r in sets["gt"]]
    by_scene = defaultdict(lambda: [0, 0])
    for r, f in zip(sets["gt"], gt_flag):
        by_scene[r["scene"]][0] += f
        by_scene[r["scene"]][1] += 1
    fish = [by_scene[s] for s in FISHEYE if s in by_scene]
    corr = [st(r) for r in sets["t06_correct"]]
    wrong = [st(r) for r in sets["t06_wrong"]]
    swap = [st(r) == FLAG for r in sets["swap"]]
    m = dict(gt_ff=int(sum(gt_flag)), gt_n=len(gt_flag),
             gt_scene_max=max([a / b for a, b in by_scene.values()] or [0.0]),
             gt_scene_worst=max(by_scene, key=lambda s: by_scene[s][0] / by_scene[s][1]) if by_scene else None,
             fish_max=max([a / b for a, b in fish] or [0.0]),
             r1_ff=int(sum(st(r) == FLAG for r in sets["r1_correct"])), r1_n=len(sets["r1_correct"]),
             t06_correct_pass=int(sum(x == PASS for x in corr)), t06_correct_n=len(corr),
             t06_wrong_pass=int(sum(x == PASS for x in wrong)), t06_wrong_flag=int(sum(x == FLAG for x in wrong)),
             t06_wrong_abstain=int(sum(x == ABSTAIN for x in wrong)), t06_wrong_n=len(wrong),
             swap_flag=int(sum(swap)), swap_n=len(swap))
    m.update(feasibility(m))
    return m


def feasibility(m):
    f1 = (m["gt_n"] > 0 and m["gt_ff"] / m["gt_n"] <= F1_ALL and m["gt_scene_max"] <= F1_SCENE and m["fish_max"] <= F1_FISH)
    f2 = m["r1_n"] > 0 and m["r1_ff"] / m["r1_n"] <= F2_ALL
    f3 = (m["t06_correct_n"] > 0 and m["t06_correct_pass"] / m["t06_correct_n"] >= F3_PASS and m["t06_wrong_pass"] <= F3_WRONG)
    return dict(F1=bool(f1), F2=bool(f2), F3=bool(f3), feasible=bool(f1 and f2 and f3))


def components(cand):
    """§6.2 tie-break 2: the enabled A-c* components (gamma, b_max, phi_rig finite); other families 0."""
    if cand["family"] != "cstar":
        return 0
    return sum(not math.isinf(float(cand["consts"][k])) for k in ("gamma", "b_max", "phi_rig"))


def choose(scored):
    """§6.2 order over feasible (cand, metrics): objective (most T06 far wrong FLAGs), then tie-breaks 1 (swap detection
    rate), 2 (fewer components), 3 (conservative: larger beta, kappa, sigma -- within A-c*), 4 (family order A-v, A-c*,
    A-h, A-c0); a remaining tie goes to the registered grid order (candidates() order), recorded.  -> (best, log)."""
    pool = [(c, m) for c, m in scored if m["feasible"]]
    log = dict(n_feasible=len(pool))
    if not pool:
        return None, log
    best = max(m["t06_wrong_flag"] for _, m in pool)
    pool = [(c, m) for c, m in pool if m["t06_wrong_flag"] == best]
    log["after_objective"] = len(pool)

    def rate(m):                       # exact comparison: same denominator for every candidate
        return m["swap_flag"]
    top = max(rate(m) for _, m in pool)
    pool = [(c, m) for c, m in pool if rate(m) == top]
    log["after_tie1_swap"] = len(pool)
    low = min(components(c) for c, _ in pool)
    pool = [(c, m) for c, m in pool if components(c) == low]
    log["after_tie2_components"] = len(pool)
    cs = [(c, m) for c, m in pool if c["family"] == "cstar"]
    if len(cs) > 1:
        cons = max((c["consts"]["beta"], c["consts"]["kappa"], c["consts"]["sigma"]) for c, _ in cs)
        pool = [(c, m) for c, m in pool if c["family"] != "cstar" or
                (c["consts"]["beta"], c["consts"]["kappa"], c["consts"]["sigma"]) == cons]
    log["after_tie3_conservative"] = len(pool)
    fam = min(FAMILY_ORDER.index(FAMILY[c["family"]]) for c, _ in pool)
    pool = [(c, m) for c, m in pool if FAMILY_ORDER.index(FAMILY[c["family"]]) == fam]
    log["after_tie4_family"] = len(pool)
    pool.sort(key=lambda cm: cm[0]["index"])
    if len(pool) > 1:
        log["residual_tie"] = [cand_name(c) for c, _ in pool]       # not ordered by the prereg: grid order decides
    return pool[0], log


def best_per_family(scored):
    """§6.5: the best feasible candidate of each family by the §6.2 order (A-c*: c_best)."""
    out = {}
    for fam in FAMILY_ORDER:
        sub = [(c, m) for c, m in scored if FAMILY[c["family"]] == fam]
        b, _ = choose(sub)
        out[fam] = None if b is None else dict(family=b[0]["family"], consts=b[0]["consts"], name=cand_name(b[0]))
    return out


def step1(rows_path=ROWS):
    frames, rows = load_rows(rows_path)
    sets = step1_sets(rows)
    cands = candidates()
    scored = [(c, metrics(c, sets, frames)) for c in cands]
    best, log = choose(scored)
    cbest, clog = choose([(c, m) for c, m in scored if c["family"] == "cstar"])
    res = dict(prereg="docs/stage1-E15-prereg.md §6.2", rows=dict(path=rows_path, sha256=sha256(rows_path)),
               data={k: len(v) for k, v in sets.items()},
               thresholds=dict(F1_all=F1_ALL, F1_scene=F1_SCENE, F1_fisheye=F1_FISH, F2=F2_ALL, F3_pass=F3_PASS, F3_wrong=F3_WRONG),
               n_candidates=len(cands), n_feasible=log["n_feasible"], tie_log=log,
               s_star=None if best is None else dict(family=best[0]["family"], consts=best[0]["consts"],
                                                     name=cand_name(best[0]), metrics=best[1]),
               c_best=None if cbest is None else dict(consts=cbest[0]["consts"], name=cand_name(cbest[0]), metrics=cbest[1],
                                                       tie_log=clog),
               best_per_family=best_per_family(scored),
               candidates=[dict(name=cand_name(c), family=c["family"], **m) for c, m in scored])
    if best is None:
        res["outcome"] = "no feasible candidate: Fix A closed, the registration closes as 'no effect' (§6.2); G1 fails"
    save(os.path.join(OUT, "select.json"), res)
    print(f"Step 1: {log['n_feasible']} / {len(cands)} feasible; s* = {res['s_star'] and res['s_star']['name']}; "
          f"c_best = {res['c_best'] and res['c_best']['name']}", flush=True)
    if res["c_best"]:
        print("Step-1 commit: set geomem/id3d.py CSTAR = " + json.dumps(_jsonable(res["c_best"]["consts"])) +
              " (inf -> math.inf) and CSTAR_SELECTED = True", flush=True)
    return res


# ------------------------------------------------------------------------------------------------- Step 0 report (§6.1)
def reproduce(frames, rows):
    """N3-N6 with the tracked code; one denominator rule (correct = covers == own, MUVOD as T06)."""
    cand = {f: dict(family=f, consts=None) for f in ("c0", "vote", "vote_inf", "hybrid")}
    cand["cstar"] = dict(family="cstar", consts=dict(P1_CSTAR))     # P0's "expected constants" (not id3d.CSTAR)
    gtA = [r for r in rows if r["kind"] == "gt" and r["set"] == "A"]
    r1A = [r for r in rows if r["kind"] == "r1" and r["set"] == "A"]
    t1 = [r for r in rows if r["kind"] == "r1" and r["set"] == "T06"]
    tf = [r for r in rows if r["kind"] == "final" and r["set"] == "T06"]
    mem = [r for r in rows if r["kind"] == "mem"]

    def counts(rs, c):
        return dict(Counter(decide(r, c, frames[r["frame"]]) for r in rs))

    def truth(r):
        return "correct" if r["covers"] == r["o"] else ("empty" if r["covers"] == 0 else "wrong")
    out = {}
    for name, c in cand.items():
        gt_by = defaultdict(Counter)
        for r in gtA:
            gt_by[r["scene"]][decide(r, c, frames[r["frame"]])] += 1
        out[name] = dict(
            devA_gt=dict(n=len(gtA), flag=sum(v[FLAG] for v in gt_by.values()),
                         fisheye_flag=sum(gt_by[s][FLAG] for s in FISHEYE),
                         fisheye_n=sum(sum(gt_by[s].values()) for s in FISHEYE),
                         worst=max(((s, v[FLAG] / sum(v.values())) for s, v in gt_by.items()), key=lambda x: x[1], default=None)),
            devA_r1_correct=counts([r for r in r1A if truth(r) == "correct"], c),
            t06_r1={f"{half}_{tr}": counts([r for r in t1 if (r["ring_d"] >= 7) == (half == "far") and truth(r) == tr], c)
                    for half in ("near", "far") for tr in ("correct", "wrong", "empty")},
            t06_final=None if name == "hybrid" else       # A-h needs the reprojection record: round-1 masks only
            {f"far_{tr}": counts([r for r in tf if r["ring_d"] >= 7 and truth(r) == tr], c) for tr in ("correct", "wrong")})
    n6 = dict(n_present=len(mem))
    if mem:
        rep = [bool(id3d.reproj_rule([r["presence"]], [r["iou"]], [r["overlap"]])[0]) for r in mem]
        n6["reproj_pass"] = float(np.mean(rep))
        for name in ("vote", "c0", "cstar"):
            stt = [decide(r, cand[name], frames[r["frame"]]) for r in mem]
            tr = [trusted(r, s) for r, s in zip(mem, stt)]
            n6[name] = dict(trust=float(np.mean(tr)), flag=float(np.mean([s == FLAG for s in stt])),
                            differs_from_reproj=float(np.mean([a != b for a, b in zip(tr, rep)])))
    out["N6_memory_candidates"] = n6
    out["n_rows"] = dict(Counter(f"{r['set']}/{r['kind']}" for r in rows))
    return out


def p0_check(rep, angles_summary):
    got = dict(cstar_gt_ff=rep["cstar"]["devA_gt"]["flag"],
               t06_far_wrong_flag_cstar=rep["cstar"]["t06_r1"]["far_wrong"].get(FLAG, 0),
               t06_far_wrong_flag_vote=rep["vote"]["t06_r1"]["far_wrong"].get(FLAG, 0),
               A_E=angles_summary["A_E"]["a"], A_all=angles_summary["A_all"]["a"])
    return {k: dict(predicted=v, tol=t, got=got[k], within=abs(got[k] - v) <= t) for k, (v, t) in P0.items()}


def step0(scenes, workers, t06_c0):
    require_t06_c0(t06_c0)
    ang = step_angles(scenes, workers)
    frames, rows = step_rows(scenes, workers, ang, t06_c0)
    frames = {f["key"]: f for f in frames}
    rep = reproduce(frames, rows)
    res = dict(prereg="docs/stage1-E15-prereg.md §6.1", angles=dict(A_E=ang["summary"]["A_E"], A_all=ang["summary"]["A_all"],
                                                                     anchors=ang["summary"]["anchors"]),
               n2=ang["n2"], reproduction=rep, P0=p0_check(rep, ang["summary"]), t06_c0=t06_c0,
               t06_c0_manifest=dict(path=os.path.relpath(T06_D1_MANIFEST, REPO), sha256=sha256(T06_D1_MANIFEST)),
               rows=dict(path=ROWS, sha256=sha256(ROWS)))
    save(os.path.join(OUT, "step0.json"), res)
    for k, v in res["P0"].items():
        print(f"P0 {k}: predicted {v['predicted']} +- {v['tol']}, got {v['got']}  {'ok' if v['within'] else 'MISS (report both)'}")
    return res


# ------------------------------------------------------------------------------------------------- Step 2 / 2b (§6.4, §6.5)
def wide_subset_delta(base_rows, cand_rows, angles_by_view, theta, key="J"):
    """Scene-cluster paired point estimate of `key` over the pairs whose view has a_v > theta (None: no such pair)."""
    from mvseg2d.dev_compare import pair_key
    from mvseg2d.paired import paired_scene_bootstrap
    keep = {pair_key(r) for r in base_rows if angles_by_view.get(_view3(r), -1.0) > theta}
    if not keep:
        return None
    B = [r for r in base_rows if pair_key(r) in keep]
    C = [r for r in cand_rows if pair_key(r) in keep]
    return paired_scene_bootstrap(B, C, key=key, n=10_000, seed=0, view_key="view")["delta"]


def _view3(r):
    return (r["scene"], int(r["t"]), r.get("ref"), r["cam"])


def rotated_angles(angles_rec, st="A'"):
    """{(scene, t, ref, cam): a_v} of the eval views of one angle set ("A'", "B'", or "A" / "B" with the original c_ini)."""
    return {(e["scene"], e["t"], e["ref"], c): a for e in angles_rec["entries"] if e["set"] == st for c, a in e["eval"].items()}


def fill_from_c0(base_rows, cand_rows, angles_by_view, theta):
    """Scenes a scope-R AB(theta) DEV run did not decode keep C0's rows when none of their eval views has a_v > theta (I2:
    a star view's output is C0's, byte for byte), and only after Z-I2 passed (§6.4); a missing scene with an eval view
    in F is an incomplete run.  -> (rows, filled scenes)."""
    have = {r["scene"] for r in cand_rows}
    filled, bad = [], []
    for sc in sorted({r["scene"] for r in base_rows} - have):
        if any(a > theta for (s2, _, _, _), a in angles_by_view.items() if s2 == sc):
            bad.append(sc)
        else:
            filled.append(sc)
    if bad:
        raise SystemExit(f"the AB run at theta {theta_tag(theta)} lacks scenes with an eval view in F: {bad}")
    if filled and not zi2_passed():
        raise SystemExit(f"the C0 fill rule (§6.4) needs a passing Z-I2 in {ZCHECK} first (scenes to fill: {filled}); "
                         "or decode those scenes")
    return cand_rows + [r for r in base_rows if r["scene"] in filled], filled


def dev_safe(compare):
    """§6.4 DEV-A / DEV-A' safety on a dev_compare result: CI lower bound > -0.003 and no scene below -0.010."""
    jf = compare["paired"]["JF"]
    lo = jf["ci"][0]
    worst = min(jf["per_scene"].values()) if jf["per_scene"] else float("nan")
    return dict(ci_lo=lo, worst_scene=worst, ok=bool(math.isfinite(lo) and lo > DEV_LO and math.isfinite(worst) and worst >= DEV_SCENE))


ZSTAGE = ("Z4", "ZR 60", "ZI2", "Zsim T06", "Zsim DEV-A G", "ZT06")    # §6.3b: the Z stage before Step 2


def zcheck_item(zcheck, item):
    """One record of runs/zcheck.json by its §6.3b name ("ZR <theta>" for a Z-R record)."""
    if item.startswith("ZR "):
        return (zcheck.get("ZR") or {}).get(item[3:])
    if item.startswith("Zsim "):
        return (zcheck.get("Zsim") or {}).get("T06" if item == "Zsim T06" else "DEV_A_G")
    return zcheck.get(item)


def zcheck_missing(zcheck, theta, A_all, family):
    """§6.3b / §6.4: before theta >= A_E (inf included) counts as DEV-A-safe, runs/zcheck.json must hold the whole Z stage
    passing for the Step-2 family -- Z4, Z-R at 60, Z-I2, Z-sim (T06, DEV-A Dog G), Z-T06 -- and, for a finite theta in
    [A_E, A_all) other than 60, the case-Z Z-R at theta.  -> the missing items; a failing one is a bug (SystemExit 2:
    fix and decode again, §6.3b)."""
    need = list(ZSTAGE)
    if math.isfinite(theta) and theta < A_all and theta != 60.0:
        need.append(f"ZR {theta_tag(theta)}")
    if zcheck is None or zcheck.get("family") != family:
        return need
    missing = []
    for item in need:
        r = zcheck_item(zcheck, item)
        if r is None or r.get("pass") is None or r.get("family", family) != family:
            missing.append(item)
        elif not r["pass"]:
            print(f"{item} ({family}) FAILED in {ZCHECK}: a bug -- fix and decode again (prereg §6.3b)", flush=True)
            raise SystemExit(2)
    return missing


def zi2_passed(zcheck=None):
    """§6.4 C0 fill rule: usable only after Z-I2 passed (runs/zcheck.json)."""
    zc = zcheck if zcheck is not None else (load(ZCHECK) if os.path.isfile(ZCHECK) else {})
    return bool((zc.get("ZI2") or {}).get("pass"))


def theta_rule(t06, devA2, devA, A_E, zcheck=None, A_all=math.inf, family=None):
    """§6.4.  t06: {theta: dict(J_all, wrong_all)}; devA2: {finite theta: dict(safe (dev_safe), wide_dJ)}; devA: {theta <
    A_E: dev_safe} for the DEV-A full decodes that exist; zcheck: runs/zcheck.json (a theta >= A_E is DEV-A-safe only with
    the Z stage passing there, zcheck_missing).  -> dict(theta_s, allowed, C, need_devA / need_zcheck, ...)."""
    allowed = []
    for th in THETA_GRID:
        if th not in t06:
            raise SystemExit(f"Step 2: T06 AB at theta {theta_tag(th)} is missing")
        ok_t06 = t06[th]["wrong_all"] <= WRONG_C0
        if math.isinf(th):
            ok_dev = True                                  # AB(inf) = C0 by construction; nothing beyond theta
        else:
            if th not in devA2:
                raise SystemExit(f"Step 2: DEV-A' AB at theta {theta_tag(th)} is missing")
            d = devA2[th]
            ok_dev = d["safe"]["ok"] and (d["wide_dJ"] is None or d["wide_dJ"] >= 0)
        if ok_t06 and ok_dev:
            allowed.append(th)
    out = dict(allowed=[theta_tag(x) for x in allowed])
    if not allowed:
        out.update(theta_s=None, note="no allowed theta (prereg silent; theta_s* = inf: relay closed -> G1 fallback (b))",
                   theta_s_value=math.inf)
        return out
    jstar = max(t06[th]["J_all"] for th in allowed)
    C = sorted((th for th in allowed if t06[th]["J_all"] >= jstar - J_TOL), reverse=True)
    out.update(J_star=jstar, C=[theta_tag(x) for x in C], checked=[])
    for th in C:
        if th >= A_E:
            missing = zcheck_missing(zcheck, th, A_all, family)
            if missing:
                out.update(theta_s=None, need_zcheck=missing,
                           note=f"theta {theta_tag(th)} >= A_E: the Z stage (§6.3b) lacks {missing} for {family}: run the "
                                "decodes (run_e15.sh z4-dog / zr <theta> <family> / dev stress 20 <family> 0 / dev G 60 "
                                "<family> 0), then run_e15.sh zcheck, then rerun --step 2")
                return out
            out["checked"].append(dict(theta=theta_tag(th), safe="theta >= A_E: no DEV-A eval view in F (angles.json) and "
                                                                 "the Z stage passed (runs/zcheck.json)",
                                       zcheck=list(ZSTAGE) + ([f"ZR {theta_tag(th)}"] if math.isfinite(th) and th < A_all
                                                              and th != 60.0 else [])))
            out.update(theta_s=theta_tag(th), theta_s_value=th)
            break
        if th not in devA:
            out.update(theta_s=None, need_devA=theta_tag(th),
                       note=f"decode DEV-A (full) at theta {theta_tag(th)} with s*, score, rerun --step 2")
            return out
        out["checked"].append(dict(theta=theta_tag(th), **devA[th]))
        if devA[th]["ok"]:
            out.update(theta_s=theta_tag(th), theta_s_value=th)
            break
    else:
        out.update(theta_s=None, theta_s_value=math.inf, note="no DEV-A-safe theta in C (prereg silent; theta_s* = inf: "
                                                                "relay closed -> G1 fallback (b))")
    if out.get("theta_s") == "inf":
        out["note"] = "theta_s* = inf: relay closed -> G1 fallback (b) (§6.4)"
    return out


def swap_rule(t06_main, t06_alt):
    """§6.5: t06_main = dict(J_all, wrong_all) of AB(s*, theta_s*); t06_alt = {family: dict(J_all, wrong_all, name)} of
    AB(s', theta_s*).  A swap needs J_all >= main + 0.02 and fewer wrong_all; among several the highest J_all, then fewer
    wrong, then the family order.  -> dict(swap, to, qualifying)."""
    q = {f: v for f, v in t06_alt.items() if v is not None and v["J_all"] >= t06_main["J_all"] + SWAP_J - 1e-12
         and v["wrong_all"] < t06_main["wrong_all"]}
    if not q:
        return dict(swap=False, to=None, qualifying=[])
    to = sorted(q, key=lambda f: (-q[f]["J_all"], q[f]["wrong_all"], FAMILY_ORDER.index(f)))[0]
    return dict(swap=True, to=to, qualifying=sorted(q))


def _t06_scores(path):
    return load(path)["arms"]


def _score_rows(name):
    p = os.path.join(SCORES, f"{name}.json")
    if not os.path.isfile(p):
        raise SystemExit(f"missing DEV score {p}")
    from mvseg2d.dev_compare import load as dc_load
    return dc_load(p)


def _load_scoring(name):
    """The rows (GT non-empty) and the empty_gt entries (gt_px 0) of a DEV scoring geomem/runs/_e15/scores/<name>.json."""
    p = os.path.join(SCORES, f"{name}.json")
    if not os.path.isfile(p):
        raise SystemExit(f"missing DEV score {p}")
    x = load(p)
    return list(x["rows"]) + [dict(e, gt_px=0) for e in x["empty_gt"]]


def _rot_rows(name_rot):
    """Both rotations of a DEV-A' run, concatenated (protocol §4)."""
    return _score_rows(f"{name_rot}_rot1") + _score_rows(f"{name_rot}_rot2")


def rescore_check(split, new_path=None, e14_path=None):
    """The C0 re-scoring (current dev_score: + covers, other_obj_mask) against the E14 scoring of the same masks: the same
    pairs, J / F / J&F equal pair by pair, the same empty-GT pairs, every row with covers / other_obj_mask.
    -> dict(ok, n_pairs, n_diff, ...)."""
    new_path = new_path or os.path.join(SCORES, f"geoset_rep_xs_{split}.json")
    e14_path = e14_path or C0_E14.format(split=split)
    if not os.path.isfile(new_path):
        raise SystemExit(f"missing the C0 re-scoring {new_path} (run_e15.sh c0-rescore {split})")
    new, old = load(new_path), load(e14_path)

    def key(r):
        return (r["scene"], int(r["t"]), r.get("ref") or "", r["cam"], int(r["obj"]))
    kn, ko = {key(r): r for r in new["rows"]}, {key(r): r for r in old["rows"]}
    diff = [list(k) + [m] for k in sorted(set(kn) & set(ko)) for m in ("J", "F", "JF") if kn[k][m] != ko[k][m]]
    no_field = [list(key(r)) for r in new["rows"] if "covers" not in r or "other_obj_mask" not in r]
    en, eo = ({key(r) for r in x["empty_gt"]} for x in (new, old))
    res = dict(split=split, new=os.path.relpath(new_path, S1), e14=os.path.relpath(e14_path, S1), n_pairs=len(kn),
               n_pairs_e14=len(ko), only_new=len(set(kn) - set(ko)), only_e14=len(set(ko) - set(kn)), n_diff=len(diff),
               diffs=diff[:20], missing_fields=len(no_field), empty_gt_equal=en == eo,
               duplicates=len(kn) != len(new["rows"]) or len(ko) != len(old["rows"]))
    res["ok"] = bool(res["n_pairs"] == res["n_pairs_e14"] and not res["only_new"] and not res["only_e14"]
                     and not res["n_diff"] and not res["missing_fields"] and res["empty_gt_equal"] and not res["duplicates"])
    return res


def c0_rows(split):
    """The C0 (geoset_rep_xs, unrotated) rows that Step 2 / stage B read: the re-scoring, only when it reproduces E14."""
    chk = rescore_check(split)
    if not chk["ok"]:
        raise SystemExit(f"the C0 re-scoring of DEV-{split} does not reproduce E14: {json.dumps(_jsonable(chk))[:400]}")
    return _score_rows(f"geoset_rep_xs_{split}")


def step_rescore_check(split):
    res = rescore_check(split)
    p = os.path.join(OUT, "c0_rescore.json")
    rec = load(p) if os.path.isfile(p) else {}
    rec[split] = res
    save(p, rec)
    print(f"C0 re-score DEV-{split}: {'OK' if res['ok'] else 'DIFFERS'} ({res['n_pairs']} pairs, {res['n_diff']} J/F/J&F "
          f"differences, missing fields {res['missing_fields']})", flush=True)
    if not res["ok"]:
        raise SystemExit(2)
    return res


def devA2_conditions(arm, family, theta, ang):
    """§6.4 DEV-A' conditions of <arm>(family, theta) against the rotated C0: all-pair J&F CI lower bound > -0.003 and no
    scene below -0.010 (dev_safe), and the pairs beyond theta with point dJ >= 0.  None when not decoded / scored."""
    from mvseg2d.dev_compare import compare
    name = run_name(arm, theta, family) + "_A"
    if not all(os.path.isfile(os.path.join(SCORES, f"{name}_rot{r}.json")) for r in (1, 2)):
        return None
    rot_a = rotated_angles(ang, "A'")
    base2 = _rot_rows("geoset_rep_xs_A")
    cand, filled = fill_from_c0(base2, _rot_rows(name), rot_a, theta)
    safe = dev_safe(compare(base2, cand))
    wide = wide_subset_delta(base2, cand, rot_a, theta)
    return dict(run=name, safe=safe, wide_dJ=wide, filled_from_c0=filled,
                ok=bool(safe["ok"] and (wide is None or wide >= 0)))


def step2(t06_scores):
    from mvseg2d.dev_compare import compare
    sel, ang = load(os.path.join(OUT, "select.json")), load(os.path.join(OUT, "angles.json"))
    if not sel.get("s_star"):
        raise SystemExit("Step 1 found no feasible statistic: no Step 2 (§6.2)")
    fam = sel["s_star"]["family"]
    A_E, A_all = ang["summary"]["A_E"]["a"], ang["summary"]["A_all"]["a"]
    arms = _t06_scores(t06_scores)
    t06 = {}
    for th in THETA_GRID:
        a = arms.get(run_name("AB", th, fam))
        if a is not None:
            t06[th] = dict(J_all=a["J_all"], wrong_all=a["wrong_all"])
    ang_a = rotated_angles(ang, "A")
    devA2 = {}
    for th in THETA_GRID:
        if math.isinf(th):
            continue
        d = devA2_conditions("AB", fam, th, ang)
        if d is not None:
            devA2[th] = d
    devA = {}
    base = c0_rows("A")
    for th in THETA_GRID:
        name = run_name("AB", th, fam) + "_A"
        if not math.isinf(th) and th < A_E and os.path.isfile(os.path.join(SCORES, f"{name}.json")):
            cand, filled = fill_from_c0(base, _score_rows(name), ang_a, th)
            devA[th] = dict(dev_safe(compare(base, cand)), filled_from_c0=filled)
    zc = load(ZCHECK) if os.path.isfile(ZCHECK) else None
    rule = theta_rule(t06, devA2, devA, A_E, zcheck=zc, A_all=A_all, family=fam)
    if not rule.get("need_devA") and not rule.get("need_zcheck"):
        miss = zcheck_missing(zc, math.inf, A_all, fam)    # §6.3b: the Z stage is mandatory before Step 2, whatever theta_s*
        if miss:
            rule.update(theta_s=None, need_zcheck=miss, note="Z stage incomplete (prereg §6.3b)")
    res = dict(prereg="docs/stage1-E15-prereg.md §6.4", family=fam, A_E=A_E, A_all=A_all,
               t06={theta_tag(k): v for k, v in t06.items()},
               devA2={theta_tag(k): v for k, v in devA2.items()}, devA={theta_tag(k): v for k, v in devA.items()}, rule=rule,
               t06_scores=dict(path=t06_scores, sha256=sha256(t06_scores)),
               zcheck=None if zc is None else dict(path=os.path.relpath(ZCHECK, S1), sha256=sha256(ZCHECK)))
    save(os.path.join(OUT, "step2.json"), res)
    print(f"Step 2: theta_s* = {rule.get('theta_s')}  {rule.get('note', '')}", flush=True)
    if rule.get("need_devA") or rule.get("need_zcheck"):
        raise SystemExit(3)
    return res


def step2b(t06_scores):
    from mvseg2d.dev_compare import compare
    sel, s2, ang = (load(os.path.join(OUT, f)) for f in ("select.json", "step2.json", "angles.json"))
    if s2["rule"].get("need_devA") or s2["rule"].get("need_zcheck"):
        raise SystemExit("Step 2 is unfinished (need_devA / need_zcheck): rerun --step 2 first")
    th = s2["rule"].get("theta_s")
    if th is None or th == "inf":
        raise SystemExit("Step 2 chose no finite theta_s*: no Step 2b")
    theta = float(th)
    fam = sel["s_star"]["family"]
    arms = _t06_scores(t06_scores)
    main = arms[run_name("AB", theta, fam)]
    alt = {}
    for f, b in sel["best_per_family"].items():
        if b is None or f == FAMILY[fam]:
            continue
        a = arms.get(run_name("AB", theta, b["family"]))
        if a is None:
            raise SystemExit(f"Step 2b: T06 {run_name('AB', theta, b['family'])} is missing")
        alt[f] = dict(J_all=a["J_all"], wrong_all=a["wrong_all"], name=b["name"], family=b["family"])
    rule = swap_rule(dict(J_all=main["J_all"], wrong_all=main["wrong_all"]), alt)
    res = dict(prereg="docs/stage1-E15-prereg.md §6.5", theta_s=th, s_star=fam,
               main=dict(J_all=main["J_all"], wrong_all=main["wrong_all"]), alternatives=alt, rule=rule)
    if rule["swap"]:
        s_new = alt[rule["to"]]["family"]
        A_E = ang["summary"]["A_E"]["a"]
        d2 = devA2_conditions("AB", s_new, theta, ang)      # the §6.4 DEV-A' conditions at theta_s* for s' (review 1 #2)
        if d2 is None:
            res["need_devA2"] = run_name("AB", theta, s_new) + "_A_rot{1,2}"
            save(os.path.join(OUT, "step2b.json"), res)
            print(f"decode DEV-A' {res['need_devA2']} (run_e15.sh dev AB {theta_tag(theta)} {s_new} 1|2), rerun --step 2b",
                  flush=True)
            raise SystemExit(3)
        res["devA2_conditions"] = d2
        if not d2["ok"]:
            rule.update(swap=False, note="the swapped statistic fails the §6.4 DEV-A' conditions at theta_s*: no swap (§6.5)")
        if rule["swap"] and theta < A_E:                    # the DEV-A safety check again with the new statistic
            p = os.path.join(SCORES, run_name("AB", theta, s_new) + "_A.json")
            if not os.path.isfile(p):
                res["need_devA"] = run_name("AB", theta, s_new) + "_A"
                save(os.path.join(OUT, "step2b.json"), res)
                print(f"decode DEV-A (full) {res['need_devA']}, score, rerun --step 2b", flush=True)
                raise SystemExit(3)
            base = c0_rows("A")
            cand, _ = fill_from_c0(base, _score_rows(run_name("AB", theta, s_new) + "_A"), rotated_angles(ang, "A"), theta)
            safe = dev_safe(compare(base, cand))
            res["devA_safety"] = safe
            if not safe["ok"]:
                rule.update(swap=False, note="the swapped statistic fails the DEV-A safety check: no swap (§6.5)")
        res["main_family"] = s_new if rule["swap"] else fam
    else:
        res["main_family"] = fam
    save(os.path.join(OUT, "step2b.json"), res)
    print(f"Step 2b: swap {rule['swap']} -> main family {res['main_family']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=("angles", "rows", "0", "1", "2", "2b", "rescore-check", "check-step1"))
    ap.add_argument("--split", choices=("A", "B"), default="A", help="--step rescore-check: the C0 split")
    ap.add_argument("--family", default=None, help="--step check-step1: the Fix A family of the decode to start")
    ap.add_argument("--scenes", default="all", help="MUVOD scenes (default: the 17 of geomem/folds.json)")
    ap.add_argument("--workers", type=int, default=max(1, min(14, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--t06-c0", default=T06_C0, dest="t06_c0", help="T06 C0 Result folder holding preset/ and preset_r1/ (D1)")
    ap.add_argument("--t06-scores", default=os.path.join(WORK, "t06_scores.json"), dest="t06_scores",
                    help="ring_diag/d1_d5/score.py output with arms named <arm>_<theta>_<family>")
    a = ap.parse_args()
    scenes = muvod_scenes() if a.scenes == "all" else a.scenes.split(",")
    if a.step == "angles":
        step_angles(scenes, a.workers)
    elif a.step == "rows":
        step_rows(scenes, a.workers, load(os.path.join(OUT, "angles.json")), a.t06_c0)
    elif a.step == "0":
        step0(scenes, a.workers, a.t06_c0)
    elif a.step == "1":
        step1()
    elif a.step == "2":
        step2(a.t06_scores)
    elif a.step == "2b":
        step2b(a.t06_scores)
    elif a.step == "rescore-check":
        step_rescore_check(a.split)
    else:
        from methods.geomem import run as R
        try:
            R.e15_require_step1(None if a.family in (None, "off") else a.family)
        except ValueError as exc:
            raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
