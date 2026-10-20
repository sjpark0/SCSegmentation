"""S1-E15 (docs/stage1-E15-prereg.md §3): the 3D identity check (Fix A) and the relay geometry (Fix B).  numpy + cv2, no
torch, so the decode (methods/geomem/run.py) and the host scripts (Step 0/1, Z-sim) run the same code.

Geometry (§3.1).  The OBJECT anchor X = c2w_r[:3,:3] @ median(local points of c_ini inside the union of the input masks,
valid) + c2w_r[:3,3] (the scene anchor = median of all valid c_ini points when the union has fewer than N_MIN valid points).
a_v = angle at X between the c_ini and view-v camera centres; hop(m, v) = the same angle between two views.  Star views
S = {a_v <= theta}, far views F = {a_v > theta}; the relay order and the reachable F views are fixed by geometry alone.

Fix A (§3.2, §3.3).  A mask is lifted with ITS OWN view's Pi3 depth into the c_ini frame (`pointmap.ref_frame_points`
coordinates and scale) and compared with reference models built once from the c_ini input masks.  States PASS / FLAG /
ABSTAIN / EMPTY; four statistic families (A-c0 `c0`, A-c* `cstar`, A-v `vote` (r_max 0.2) / `vote_inf` (r_max inf),
A-h `hybrid`).  `trust` turns a state into the memory confidence; `select_final` is the §3.4 choice between the second
output X, the round-1 output S1 and the fallback.
"""
import math
import os
from collections import deque

import cv2
import numpy as np

from geomem.pointmap import CONF_MIN, local_points, ref_frame_points

PASS, FLAG, ABSTAIN, EMPTY = "PASS", "FLAG", "ABSTAIN", "EMPTY"
STATES = (PASS, FLAG, ABSTAIN, EMPTY)
FAMILIES = ("c0", "cstar", "vote", "vote_inf", "hybrid")

# §3.7 fixed constants (not variables)
N_MIN = 20                      # lifted points (work resolution) below which a set is "small"
ERODE_KERNEL = np.ones((3, 3), np.uint8)   # one 3x3 erosion at the work resolution; skipped when it leaves < N_MIN points
COVER_MIN = 0.5                 # mask -> work resolution: INTER_AREA coverage >= 0.5
RADIUS_PCT = 90                 # rho = 90th percentile of the distances to the coordinate-wise median
N_REF, N_Q = 1024, 4096         # A-v deterministic stride subsamples (reference model / query)
THETA_V = 0.5                   # A-v PASS: ov3d >= THETA_V (= run.THETA_REPROJ)
OVERLAP_PASS = 0.5              # A-h: reproj overlap >= 0.5 -> PASS (= run.THETA_REPROJ)
NU = 0.5                        # A-h: the reprojection decides where vis = n_valid / |m| >= NU
TAU_IOU = 0.7                   # the adopted reproj rule's IoU fallback (= run.TAU_CONF)
PHI_TOL, PHI_STRIDE = 0.95, 3   # free-space rate: z < 0.95 D_b(u), every 3rd pixel of a
K_RELAY, K_SEARCH = 3, 6        # relay neighbours (= round-2 k) among the K_SEARCH nearest admissible decoded views
BORDER_FRAC, BORDER_MIN_PX = 0.005, 2   # the c_ini border band: max(2 px, 0.5 % of min(H, W)) at the original resolution
VOTE_RMAX = {"vote": 0.2, "vote_inf": math.inf, "hybrid": 0.2}   # A-h reads A-v at r_max 0.2 (the A-v default)
_CHUNK = 128                    # A-v query rows per distance block (speed / memory; the result does not depend on it)

# A-c* constants: the §3.3 grid, chosen by the §6.2 Step-1 rule (scripts/e15/e15_select.py --step 1 -> runs/select.json
# c_best) and written here at the Step-1 commit (§6.2, L5): CSTAR = c_best, CSTAR_SELECTED = True (it stays False when no
# A-c* candidate is feasible).  Until then the values are P1's PREDICTED pick and run.py refuses `--id3d cstar`.
CSTAR_GRID = dict(beta=(1.0, 1.25, 1.5), kappa=(1.0, 1.25, 1.5), sigma=(0.03, 0.05), rho_env=(0.5, 1.0),
                  b_max=(0.01, 0.05, math.inf), gamma=(2.0, 3.0, math.inf), phi_rig=(0.12, math.inf))
CSTAR = dict(beta=1.0, kappa=1.25, sigma=0.05, rho_env=1.0, b_max=0.01, gamma=math.inf, phi_rig=0.12)   # Step 1, 2026-10-20
CSTAR_SELECTED = True


# ------------------------------------------------------------------ angles, anchor, relay order (§3.1, §3.5)
def angle_deg(u, v):
    """Angle between two vectors in degrees (0 for a zero vector)."""
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(u @ v / nu / nv, -1.0, 1.0))))


def mask_to_work(mask_bool, work_hw):
    """Original-resolution mask -> work resolution (INTER_AREA coverage >= 0.5)."""
    h, w = work_hw
    return cv2.resize(np.asarray(mask_bool).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= COVER_MIN


def object_anchor(depth, conf, K, c2w, union_work, n_min=N_MIN):
    """§3.1 anchor in world (cache) coordinates -> (X [3] float64, "obj" | "scene")."""
    valid = (conf > CONF_MIN) & (depth > 0)
    assert bool(valid.any()), "c_ini has no valid Pi3 point"
    P = local_points(depth, K)
    sel, kind = np.asarray(union_work, bool) & valid, "obj"
    if int(sel.sum()) < n_min:
        sel, kind = valid, "scene"
    T = np.asarray(c2w, np.float64)
    return T[:3, :3] @ np.median(P[sel], 0) + T[:3, 3], kind


def view_angles(centres, ref_cam, X):
    """{view: a_v} for every view but c_ini (degrees at the anchor X)."""
    return {c: angle_deg(centres[ref_cam] - X, centres[c] - X) for c in centres if c != ref_cam}


def hop_angle(centres, X, m, v):
    return angle_deg(centres[m] - X, centres[v] - X)


def split_views(angles, cams, theta):
    """(S, F) in camera order: S = {a_v <= theta}, F = {a_v > theta}; theta = inf -> F empty."""
    S = [c for c in cams if c in angles and angles[c] <= theta]
    F = [c for c in cams if c in angles and angles[c] > theta]
    return S, F


def reachable(S, F, hop, theta):
    """F views reachable from S through hops <= theta (BFS over F) -> set.  `hop(m, v)` -> degrees."""
    seen, q = set(), deque(S)
    while q:
        m = q.popleft()
        for v in F:
            if v not in seen and hop(m, v) <= theta:
                seen.add(v)
                q.append(v)
    return seen


def relay_order(S, F, angles, hop, theta, cams, order="hop"):
    """§3.5 steps 2-3 (geometry only) -> (relay sequence, unreachable F views in camera order).
    hop:   repeatedly the unsolved reachable F view with the smallest a_v among those within theta of a solved view
           (S or already relayed; ties -> camera order); F views no path reaches are unreachable.
    angle: (ABnh) every F view by a_v (ties -> camera order), no reachability rule."""
    rank = {c: i for i, c in enumerate(cams)}

    def key(c):
        return (angles[c], rank[c])
    if order == "angle":
        return sorted(F, key=key), []
    assert order == "hop", order
    reach = reachable(S, F, hop, theta)
    solved, seq, todo = list(S), [], set(reach)
    while todo:
        E = [v for v in todo if any(hop(m, v) <= theta for m in solved)]
        if not E:
            break
        v = min(E, key=key)
        seq.append(v)
        solved.append(v)
        todo.discard(v)
    return seq, [c for c in F if c not in reach]


def relay_candidates(v, pool, centres, cams, hop=None, theta=None, k_search=K_SEARCH):
    """§3.5 step 4: the pool members (decoded views) within theta of v (`hop` None = no hop rule, ABnh), by Pi3 centre
    distance (ties -> camera order), the first k_search."""
    rank = {c: i for i, c in enumerate(cams)}
    cand = [m for m in pool if m != v and (hop is None or hop(m, v) <= theta)]
    cand.sort(key=lambda m: (float(np.linalg.norm(np.asarray(centres[m], np.float64) - np.asarray(centres[v], np.float64))),
                             rank[m]))
    return cand[:k_search]


def relay_neighbours(cands, trusted_any, k=K_RELAY):
    """§3.5 step 4: the first k candidates with at least one trusted object."""
    return [m for m in cands if trusted_any[m]][:k]


def orphan_objects(neigh, trust):
    """Object indices no neighbour trusts (they see c_ini only).  trust: {view: bool [B]}."""
    if not neigh:
        return None
    t = np.stack([np.asarray(trust[m], bool) for m in neigh]).any(0)
    return [int(i) for i in np.flatnonzero(~t)]


# ------------------------------------------------------------------ rig free-space rate (§3.2)
def _valid(z):
    return (z["conf"] > CONF_MIN) & (z["depth"] > 0)


def free_space(a, b, stride=PHI_STRIDE, tol=PHI_TOL):
    """phi(a -> b): share of a's valid points (every `stride`-th pixel) that project inside b in front of b's valid depth
    (z_b < tol * D_b(u), nearest pixel).  a, b: dicts with depth, conf, K, c2w (a Pi3 cache entry)."""
    va = _valid(a)[::stride, ::stride]
    Pa = local_points(a["depth"], a["K"])[::stride, ::stride][va]
    if len(Pa) == 0:
        return 0.0
    T = np.linalg.inv(np.asarray(b["c2w"], np.float64)) @ np.asarray(a["c2w"], np.float64)
    Pb = Pa @ T[:3, :3].T + T[:3, 3]
    z = Pb[:, 2]
    front = z > 0
    zs = np.where(front, z, 1.0)
    Kb = np.asarray(b["K"], np.float64)
    ui = np.rint(Kb[0, 0] * Pb[:, 0] / zs + Kb[0, 2]).astype(np.int64)
    vi = np.rint(Kb[1, 1] * Pb[:, 1] / zs + Kb[1, 2]).astype(np.int64)
    h, w = b["depth"].shape
    inside = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    vb = _valid(b)
    hit = np.zeros(len(Pa), bool)
    idx = np.flatnonzero(inside)
    Db = b["depth"][vi[idx], ui[idx]]
    hit[idx] = vb[vi[idx], ui[idx]] & (z[idx] < tol * Db)
    return float(hit.sum() / len(Pa))


def rig_phi(cache_dir, ref_cam, cams):
    """Scene free-space rate: median over v != c_ini of max(phi(r -> v), phi(v -> r)) -> (median, {v: phi_v})."""
    r = dict(np.load(os.path.join(cache_dir, f"{ref_cam}.npz")))
    per = {}
    for c in cams:
        if c == ref_cam:
            continue
        v = dict(np.load(os.path.join(cache_dir, f"{c}.npz")))
        per[c] = max(free_space(r, v), free_space(v, r))
    return (float(np.median(list(per.values()))) if per else 0.0), per


# ------------------------------------------------------------------ lifting and reference models (§3.2)
def border_fraction(mask_bool):
    """Share of the mask's pixels inside the image border band max(2 px, 0.5 % of min(H, W)) (0 for an empty mask)."""
    m = np.asarray(mask_bool, bool)
    n = int(m.sum())
    if n == 0:
        return 0.0
    H, W = m.shape
    b = max(BORDER_MIN_PX, int(BORDER_FRAC * min(H, W)))
    edge = np.zeros_like(m)
    edge[:b] = True
    edge[-b:] = True
    edge[:, :b] = True
    edge[:, -b:] = True
    return float((m & edge).sum() / n)


def lift(mask_work, P, valid, n_min=N_MIN):
    """L(m, v): the view's points (c_ini frame, scene scale) under the eroded work-resolution mask; the un-eroded mask when
    the erosion leaves fewer than n_min valid points.  -> [n,3] (n may still be < n_min: the caller ABSTAINs)."""
    mw = np.asarray(mask_work, bool)
    sel = (cv2.erode(mw.astype(np.uint8), ERODE_KERNEL) > 0) & valid
    if int(sel.sum()) < n_min:
        sel = mw & valid
    return P[sel]


def subsample(X, n):
    """Deterministic stride subsample to at most n rows."""
    return X if len(X) <= n else X[np.linspace(0, len(X) - 1, n).astype(np.int64)]


def ref_models(masks_bool, objects, P_r, valid_r):
    """§3.2 reference model of every object from its c_ini input mask (original resolution); built once, never updated.
    -> {o: dict(R [n,3], c [3] | None, rho | None, n, has, border)}."""
    work = P_r.shape[:2]
    out = {}
    for i, o in enumerate(objects):
        R = lift(mask_to_work(masks_bool[i], work), P_r, valid_r)
        has = len(R) >= N_MIN
        c = np.median(R, 0) if has else None
        rho = float(np.percentile(np.linalg.norm(R - c, axis=1), RADIUS_PCT)) if has else None
        out[int(o)] = dict(R=R, c=c, rho=rho, n=int(len(R)), has=bool(has), border=border_fraction(masks_bool[i]))
    return out


# ------------------------------------------------------------------ the four statistic families (§3.3)
def centroid_features(Q, models):
    """The constant-free quantities of one lifted query that A-c0 and A-c* read: n = |Q|; for n >= N_MIN also the
    query spread (90th percentile of the distances to its coordinate-wise median q) and d_j = ||q - c_j|| for every
    modelled object j.  Computed once per mask (decode and the Step-1 grid both decide from it)."""
    n = len(Q)
    if n < N_MIN:
        return dict(n=n)
    q = np.median(Q, 0)
    return dict(n=n, spread=float(np.percentile(np.linalg.norm(Q - q, axis=1), RADIUS_PCT)),
                d={j: float(np.linalg.norm(q - models[j]["c"])) for j in models if models[j]["has"]})


def c0_decide(feat, o, models):
    """A-c0 from centroid_features (see status_c0)."""
    n = feat["n"]
    if n < N_MIN:
        return ABSTAIN, dict(n=n, why="small")
    mod = sorted(j for j in models if models[j]["has"])
    if not models.get(o, {}).get("has") or len(mod) < 2:
        return ABSTAIN, dict(n=n, why="model")
    d = feat["d"]
    near = min(mod, key=lambda j: (d[j], j))
    return (PASS if near == o else FLAG), dict(n=n, nearest=int(near), d_own=d[o])


def status_c0(Q, o, models):
    """A-c0 (the simulation rule): nearest modelled reference centroid == own -> PASS, else FLAG."""
    return c0_decide(centroid_features(Q, models), o, models)


def cstar_decide(feat, o, models, beta, kappa, sigma, rho_env, b_max, gamma, phi_rig=math.inf, rig_ok=True):
    """A-c* from centroid_features (see status_cstar); the checks run in the registered order."""
    n = feat["n"]
    if n < N_MIN:
        return ABSTAIN, dict(n=n, why="small")
    m = models.get(o)
    if m is None or not m["has"]:
        return ABSTAIN, dict(n=n, why="model")
    if m["rho"] > rho_env:
        return ABSTAIN, dict(n=n, why="wide")
    if m["border"] > b_max:
        return ABSTAIN, dict(n=n, why="cut")
    if not rig_ok:
        return ABSTAIN, dict(n=n, why="rig")
    spread = feat["spread"]
    if gamma != math.inf and spread > gamma * m["rho"] + sigma:
        return ABSTAIN, dict(n=n, why="spread", spread=spread)
    d_o = feat["d"][o]
    comp = sorted(j for j in models if j != o and models[j]["has"] and models[j]["rho"] <= rho_env)
    in_own = d_o <= beta * m["rho"] + sigma
    nearer = [j for j in comp if kappa * feat["d"][j] < d_o]
    info = dict(n=n, d_own=d_o, rho=m["rho"], spread=spread, nearer=[int(j) for j in nearer])
    if in_own and not nearer:
        return PASS, info
    if (not in_own) and nearer:
        return FLAG, info
    return ABSTAIN, dict(info, why="between")


def status_cstar(Q, o, models, beta, kappa, sigma, rho_env, b_max, gamma, phi_rig=math.inf, rig_ok=True):
    """A-c* (conjunctive rule).  ABSTAIN on a small query, an unmodelled / wide / cut own object, a query spread beyond
    gamma*rho_o + sigma, or rig distrust; in_own = d_o <= beta*rho_o + sigma; nearer = some competitor (modelled,
    not wide, cut ones included) with kappa*d_j < d_o; PASS = in_own & ~nearer, FLAG = ~in_own & nearer."""
    return cstar_decide(centroid_features(Q, models), o, models, beta, kappa, sigma, rho_env, b_max, gamma, phi_rig, rig_ok)


def vote_reference(models, n_ref=N_REF):
    """A-v reference point set: every modelled object's stride subsample (<= n_ref points), ascending id -> (P, labels)."""
    ids = sorted(j for j in models if models[j]["has"])
    if not ids:
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    refs = [subsample(models[j]["R"], n_ref) for j in ids]
    return np.concatenate(refs), np.concatenate([np.full(len(r), j, np.int64) for r, j in zip(refs, ids)])


def vote_nearest(Q, ref):
    """For each query point of the stride subsample (<= N_Q): (label of the nearest reference point (ties -> the first =
    lowest id), that distance).  Exact per-axis differences (platform-deterministic).  No reference point: (0, inf)."""
    P, lab = ref
    Qs = subsample(np.asarray(Q, np.float64), N_Q)
    near, dist = np.zeros(len(Qs), np.int64), np.full(len(Qs), np.inf)
    if len(P) == 0:
        return near, dist
    Pa = [np.ascontiguousarray(P[:, a], np.float64)[None] for a in range(3)]
    d2, tmp = np.empty((_CHUNK, len(P))), np.empty((_CHUNK, len(P)))
    for s in range(0, len(Qs), _CHUNK):
        q = Qs[s:s + _CHUNK]
        n = len(q)
        a, b = d2[:n], tmp[:n]
        np.subtract(q[:, 0:1], Pa[0], out=a)            # (dx^2 + dy^2) + dz^2, in place, fixed order
        np.multiply(a, a, out=a)
        for ax in (1, 2):
            np.subtract(q[:, ax:ax + 1], Pa[ax], out=b)
            np.multiply(b, b, out=b)
            np.add(a, b, out=a)
        j = a.argmin(1)
        near[s:s + n] = lab[j]
        dist[s:s + n] = np.sqrt(a[np.arange(n), j])
    return near, dist


def vote_threshold(nearest, r_max):
    """l(q) from vote_nearest: the nearest label where that distance <= r_max, else 0."""
    near, dist = nearest
    return np.where(dist <= r_max, near, 0)


def vote_labels(Q, ref, r_max):
    """l(q) for each query point of the stride subsample (<= N_Q): the label of the nearest reference point (ties -> the
    first = lowest id) when that distance <= r_max, else 0."""
    return vote_threshold(vote_nearest(Q, ref), r_max)


def vote_decide(lab, n, o, models):
    """A-v from the labels l(q) of a query of n lifted points (see status_vote)."""
    if n < N_MIN:
        return ABSTAIN, dict(n=n, why="small")
    if not models.get(o, {}).get("has"):
        return ABSTAIN, dict(n=n, why="model")
    ov3d = float((lab == o).mean())
    ids = sorted(j for j in models if models[j]["has"])
    counts = [int((lab == j).sum()) for j in ids]
    plur = ids[int(np.argmax(counts))] if sum(counts) else 0
    info = dict(n=n, ov3d=ov3d, plur=int(plur))
    if ov3d >= THETA_V:
        return PASS, info
    if plur not in (0, o):
        return FLAG, info
    return ABSTAIN, info


def status_vote(Q, o, models, r_max, ref=None):
    """A-v (3D vote): ov3d = own share of the labelled query points; plur = most frequent label >= 1 (ties -> lowest
    id; none -> 0).  ABSTAIN when o has no model or the query is small; PASS = ov3d >= THETA_V; FLAG = plur not in
    {0, o}; else ABSTAIN."""
    if len(Q) < N_MIN or not models.get(o, {}).get("has"):
        return vote_decide(None, len(Q), o, models)
    return vote_decide(vote_labels(Q, ref if ref is not None else vote_reference(models), r_max), len(Q), o, models)


def hybrid_decide(sv, info, n_px, n_valid, overlap):
    """A-h from A-v's (state, info) at r_max 0.2 and the reprojection record (see status_hybrid)."""
    vis = float(n_valid) / float(n_px) if n_px else 0.0
    info = dict(info, vis=vis, A_v=sv)
    if sv == FLAG:
        return FLAG, info
    if vis >= NU:
        ov = float(overlap) if overlap is not None else float("nan")
        return (PASS if (not math.isnan(ov) and ov >= OVERLAP_PASS) else ABSTAIN), info
    return sv, info


def status_hybrid(Q, o, models, n_px, n_valid, overlap, r_max=VOTE_RMAX["hybrid"], ref=None):
    """A-h: A-v FLAG -> FLAG; else where the mask is mostly visible from c_ini (vis = n_valid / |m| >= NU) the reprojection
    decides (overlap >= 0.5 -> PASS, else ABSTAIN); else A-v's state."""
    sv, info = status_vote(Q, o, models, r_max, ref)
    return hybrid_decide(sv, info, n_px, n_valid, overlap)


# ------------------------------------------------------------------ trust and the final choice (§3.2, §3.4)
def reproj_rule(presence, iou, overlap, tau=TAU_IOU, theta=OVERLAP_PASS):
    """run.object_conf_mode("reproj") in numpy: present & (overlap >= theta, IoU >= tau where overlap is nan)."""
    presence, iou, overlap = (np.asarray(x, np.float64) for x in (presence, iou, overlap))
    return (presence > 0) & np.where(np.isnan(overlap), iou >= tau, overlap >= theta)


def trust(states, presence, iou, overlap, tau=TAU_IOU):
    """Memory confidence (§3.2): presence > 0 & (PASS | (ABSTAIN & the adopted reproj rule)); FLAG / EMPTY never."""
    st = np.asarray(states, object)
    rep = reproj_rule(presence, iou, overlap, tau)
    return (np.asarray(presence, np.float64) > 0) & ((st == PASS) | ((st == ABSTAIN) & rep))


def select_final(status_x, status_s1, flagged="keep"):
    """§3.4: X when its state is PASS or ABSTAIN; else S1 when S1 PASSes; else the fallback (keep = X / chain / empty)."""
    assert flagged in ("keep", "chain", "empty"), flagged
    if status_x in (PASS, ABSTAIN):
        return "X"
    if status_s1 == PASS:
        return "S1"
    return flagged


# ------------------------------------------------------------------ one scene's judge
class Judge:
    """Fix A for one scene (frame): the c_ini reference models and the family's statistic.  `statuses` judges the masks
    of one view (the decoder's raw L > 0 masks, original resolution, before exclusivity)."""

    def __init__(self, cache_dir, ref_cam, ref_masks, objects, family, consts=None, rig_ok=True):
        assert family in FAMILIES, family
        self.cache_dir, self.ref_cam, self.family = cache_dir, ref_cam, family
        self.objects = [int(o) for o in objects]
        self.consts = dict(CSTAR if consts is None else consts) if family == "cstar" else {}
        self.rig_ok = bool(rig_ok)
        P_r, valid_r, self.scale = ref_frame_points(cache_dir, ref_cam, ref_cam)
        self.models = ref_models(np.asarray(ref_masks, bool), self.objects, P_r, valid_r)
        self.r_max = VOTE_RMAX.get(family)
        self.ref = vote_reference(self.models) if family in ("vote", "vote_inf", "hybrid") else None

    def model_record(self):
        return {str(o): dict(n=m["n"], has=m["has"], rho=m["rho"], border=m["border"]) for o, m in self.models.items()}

    def judge(self, Q, o, n_px, n_valid, overlap):
        f = self.family
        if f == "c0":
            return status_c0(Q, o, self.models)
        if f == "cstar":
            return status_cstar(Q, o, self.models, rig_ok=self.rig_ok, **self.consts)
        if f in ("vote", "vote_inf"):
            return status_vote(Q, o, self.models, self.r_max, self.ref)
        return status_hybrid(Q, o, self.models, n_px, n_valid, overlap, self.r_max, self.ref)

    def statuses(self, cam, pred, presence, n_valid=None, overlap=None):
        """pred [B,H,W] bool (original resolution), presence [B]; n_valid / overlap [B] (the reproj record; A-h only).
        -> (states [B], infos [B]).  EMPTY = presence <= 0 or no pixel; rig distrust -> every other state ABSTAIN."""
        pred = np.asarray(pred, bool)
        B = len(self.objects)
        assert self.family != "hybrid" or (n_valid is not None and overlap is not None), "A-h needs the reproj record"
        assert pred.shape[0] == B, (pred.shape, B)
        P_v, valid_v, _ = ref_frame_points(self.cache_dir, cam, self.ref_cam)
        work = P_v.shape[:2]
        states, infos = [], []
        for i, o in enumerate(self.objects):
            n_px = int(pred[i].sum())
            if float(presence[i]) <= 0 or n_px == 0:
                states.append(EMPTY)
                infos.append(dict(n=0))
                continue
            Q = lift(mask_to_work(pred[i], work), P_v, valid_v)
            if not self.rig_ok:
                states.append(ABSTAIN)
                infos.append(dict(n=len(Q), why="rig"))
                continue
            s, info = self.judge(Q, o, n_px, None if n_valid is None else n_valid[i],
                                 None if overlap is None else overlap[i])
            states.append(s)
            infos.append(info)
        return states, infos


# ------------------------------------------------------------------ ground-truth identity (AB-oracle, diagnostic only)
def covers(pred_bool, gt_labels, s=4):
    """ring_diag/d1_d5/score.py's rule: the GT label the predicted mask overlaps most at 1/s resolution (PIL nearest);
    0 when the mask is empty there or overlaps no label."""
    from PIL import Image
    gt = np.asarray(gt_labels)
    gt = gt.astype(np.uint8) if int(gt.max(initial=0)) <= 255 else gt.astype(np.int32)    # PIL mode L / I
    H, W = gt.shape
    pm = np.asarray(pred_bool, bool)
    if pm.shape != gt.shape:
        pm = np.array(Image.fromarray(pm.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    pq = np.array(Image.fromarray(pm.astype(np.uint8) * 255).resize((W // s, H // s), Image.NEAREST)) > 127
    gq = np.array(Image.fromarray(gt).resize((W // s, H // s), Image.NEAREST))
    if not pq.any():
        return 0
    top = int(gq.max())
    counts = [int((pq & (gq == j)).sum()) for j in range(1, top + 1)]
    return int(np.argmax(counts)) + 1 if sum(counts) else 0


def gt_states(pred, presence, objects, gt_labels):
    """AB-oracle identity: EMPTY as Fix A; PASS when the mask covers its own GT object, else FLAG."""
    out = []
    for i, o in enumerate(objects):
        if float(presence[i]) <= 0 or not bool(np.asarray(pred[i]).any()):
            out.append(EMPTY)
        else:
            out.append(PASS if covers(pred[i], gt_labels) == int(o) else FLAG)
    return out


# ------------------------------------------------------------------ host re-judgement (Z-sim)
def rejudge(cache_dir, ref_cam, ref_masks, objects, family, mask_dir, conf_rec, cams, hw, consts=None, rig_ok=True):
    """Fix A states of written masks <mask_dir>/<cam>/<o>.png (absent PNG = empty) with presence / n_valid / overlap from a
    round1_conf.json record -> {cam: {str(o): state}}; the decode's id3d_conf.json "round1" must equal it (Z-sim)."""
    judge = Judge(cache_dir, ref_cam, ref_masks, objects, family, consts, rig_ok)
    out = {}
    for cam in cams:
        pred = np.zeros((len(objects),) + tuple(hw), bool)
        for i, o in enumerate(objects):
            p = os.path.join(mask_dir, cam, f"{int(o)}.png")
            if os.path.isfile(p):
                pred[i] = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
        rec = conf_rec[cam]
        presence = [rec[str(o)]["presence"] for o in objects]
        n_valid = [rec[str(o)].get("n_valid", 0) for o in objects]
        overlap = [float("nan") if rec[str(o)].get("overlap") is None else rec[str(o)]["overlap"] for o in objects]
        st, _ = judge.statuses(cam, pred, presence, n_valid, overlap)
        out[cam] = {str(int(o)): s for o, s in zip(objects, st)}
    return out
