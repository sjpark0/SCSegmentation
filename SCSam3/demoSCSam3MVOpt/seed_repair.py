"""Degenerate cross-view seed detection and donor selection (Phase 5 / REPORT.md P5).

Torch-free so it is unit-testable on the host, like xview_gather.py.  The runner's
RepairSeeds() turns masks_spatial into plain area dicts, asks this module what to
repair and from where, then does the model work (a fresh spatial session) itself.

The rules are the ones pre-registered in docs/phase5-seed-repair-prereg.md section 1
as amended by section 8 (after the Welder smoke test, before the sweep).  If the code
here and that document disagree, the code is wrong.

  seed_areas   {view: {obj: area_px}}  first-frame masks the cross-view pass produced
                                       (an object missing from a view's dict is area 0)
  ref_areas    {obj: area_px}          the reference camera's ground-truth prompt
  ref_view     int                     the reference camera's view index (c_ini)
  views        which views to examine  (the runner passes the tracked session set:
                                       those are the only seeds ever consumed, and on
                                       wide rigs the far views are garbage)
  donors       which views may donate  (the runner passes the tracked set minus the
                                       reference: the reference is already the anchor)

A (view, obj) pair is DEGENERATE when all four hold:
  1. view != ref_view                        the reference seed comes from ground truth
  2. ref_areas[obj] >= MIN_PX                a few-pixel prompt is a prompt problem, not ours
  3. seed area < max(MIN_PX, REL_FRAC * ref_areas[obj])
                                             relative to the object's own prompt, not to a
                                             per-view median (a blob elsewhere in the view
                                             must not move the bar for this object)
  4. a donor exists                          see pick_donor

DONOR for (view, obj): among `donors`, within +-WINDOW of `view` (then +-FALLBACK), the
largest seed area for obj inside the plausibility band
    max(MIN_PX, DONOR_LO * ref_areas[obj]) <= area <= DONOR_HI * ref_areas[obj]
excluding `view`, every other view flagged for the same object, and donors already used
for this pair.  Ties: nearer view first, then the lower index.  The band is what keeps a
1.5-million-pixel blob from ever donating.
"""
from collections import namedtuple

MIN_PX = 64
REL_FRAC = 0.10
DONOR_LO = 0.25
DONOR_HI = 4.0
WINDOW = 4
FALLBACK = 8
MAX_ROUNDS = 2

Flag = namedtuple("Flag", "view obj area threshold")
Repair = namedtuple("Repair", "view obj area threshold donor donor_area")
Unrepairable = namedtuple("Unrepairable", "view obj area threshold")


def threshold(ref_area, min_px=MIN_PX, frac=REL_FRAC):
    """Rule 3's bar for one object."""
    return max(float(min_px), frac * ref_area)


def detect(seed_areas, ref_areas, ref_view, views=None, min_px=MIN_PX, frac=REL_FRAC):
    """Rules 1-3 -> [Flag], sorted by (view, obj).  Rule 4 is applied by plan()."""
    flags = []
    scope = sorted(seed_areas) if views is None else sorted(set(views) & set(seed_areas))
    for view in scope:
        if view == ref_view:
            continue                                            # rule 1
        areas = seed_areas[view]
        for obj in sorted(ref_areas):
            if ref_areas[obj] < min_px:
                continue                                        # rule 2
            thr = threshold(ref_areas[obj], min_px, frac)
            area = areas.get(obj, 0)
            if area < thr:                                      # rule 3
                flags.append(Flag(view, obj, int(area), thr))
    return flags


def in_band(area, ref_area, min_px=MIN_PX, lo=DONOR_LO, hi=DONOR_HI):
    return max(float(min_px), lo * ref_area) <= area <= hi * ref_area


def pick_donor(seed_areas, view, obj, ref_area, candidates=None, exclude=(), window=WINDOW,
               fallback=FALLBACK, min_px=MIN_PX, lo=DONOR_LO, hi=DONOR_HI):
    """Rule 4 -> (donor_view, donor_area) or None."""
    pool = sorted(seed_areas) if candidates is None else sorted(set(candidates) & set(seed_areas))
    excluded = set(exclude) | {view}
    for reach in (window, fallback):
        best = None
        for cand in pool:
            if cand in excluded or abs(cand - view) > reach:
                continue
            area = seed_areas[cand].get(obj, 0)
            if not in_band(area, ref_area, min_px, lo, hi):
                continue
            key = (-area, abs(cand - view), cand)               # largest, nearer, lower index
            if best is None or key < best[0]:
                best = (key, cand, int(area))
        if best is not None:
            return best[1], best[2]
    return None


def plan(seed_areas, ref_areas, ref_view, views=None, donors=None, used=None, **kw):
    """One round: detect, then a donor for every flag.

    `used`: {(view, obj): set(donor views already tried)} from earlier rounds.
    Returns ([Repair], [Unrepairable]); flags for the same object never donate to
    each other.
    """
    used = used or {}
    detect_kw = {k: kw[k] for k in ("min_px", "frac") if k in kw}
    donor_kw = {k: kw[k] for k in ("window", "fallback", "min_px", "lo", "hi") if k in kw}
    flags = detect(seed_areas, ref_areas, ref_view, views, **detect_kw)
    flagged_views = {}
    for f in flags:
        flagged_views.setdefault(f.obj, set()).add(f.view)
    repairs, unrepairable = [], []
    for f in flags:
        exclude = flagged_views[f.obj] | used.get((f.view, f.obj), set())
        donor = pick_donor(seed_areas, f.view, f.obj, ref_areas[f.obj], donors, exclude,
                           **donor_kw)
        if donor is None:
            unrepairable.append(Unrepairable(*f))
        else:
            repairs.append(Repair(f.view, f.obj, f.area, f.threshold, donor[0], donor[1]))
    return repairs, unrepairable


def as_records(items):
    """namedtuples -> plain dicts for MANIFEST.json."""
    return [dict(x._asdict()) for x in items]
