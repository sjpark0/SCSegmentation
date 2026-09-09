#!/usr/bin/env python3
"""Aggregate eval_jf.py raw files into per-dataset and overall J&F tables.

    python3 report_jf.py                                       everything in jf_v2.json
    python3 report_jf.py --raw jf_raw.json jf_sam3_onestage.json   several files, pooled
    python3 report_jf.py --methods SegMaskNew3 SegMaskSam3XW1GPS4   a subset, in that order
    python3 report_jf.py --paired SegMaskSam3XW0 SegMaskSam3XW1GPS4 --split nonref
    python3 report_jf.py --muvod --methods SegMaskSam3XW1GPS4M  MUVOD J&F^3 comparison
    python3 report_jf.py --paper docs/raw/paper_tables.md      every table the paper needs

Raw files are looked up as given, then under Data/MVSeg, then next to this
script.  When the same (dataset, camera, method) occurs in several files the
last file wins; one warning per (earlier file, later file) pair says how many
entries were overridden and names one of them.

Scores (version 2, 2026-09-07):

  object    J_o, F_o are the stored per-object means over frames (J_all/F_all,
            or J_inner/F_inner which drop the first and last frame the way the
            DAVIS code does); J&F_o = (J_o + F_o) / 2.
  camera    mean over the camera's objects of J_o and of F_o; J&F = (J + F) / 2.
  dataset   'pooled' (default): mean over every (camera, object) pair of the
            dataset; 'sequence': mean over the dataset's cameras of the camera
            score.  J&F = (J + F) / 2 either way.
  overall   unweighted mean over datasets.

Options that change what goes into those means:

  --frames inner       use the inner-frame scores everywhere
  --split ref|nonref   keep only the reference camera (max-id rule, the camera
                       the runner seeds from) or only the other cameras
  --area-weighted      weight every object by its mean GT area over the frames
                       (unweighted fallback when the weights of a group sum to 0)
  --common (default)   restrict to datasets every selected method has a result
                       for; --no-common keeps everything.  The dropped datasets
                       are always listed.

Extra readings:

  --bin-by-nb          cameras grouped by nb = min(view_index, W), W from
                       --window: camera count and mean camera J&F per method;
                       with --paired also the mean delta and wins/ties/losses
  --paired A B         delta = B - A at dataset level and at camera level: n,
                       mean, median, wins/ties/losses (tie iff |delta| < 1e-9),
                       exact two-sided sign test on wins vs losses (p = 1 when
                       there are neither), two-sided Wilcoxon signed-rank test
                       over the m non-tied deltas (exact for m <= 22, enumerating
                       all 2^m sign assignments, printed as 'exact, m=<m>' next
                       to the p-value; normal approximation with continuity and
                       tie correction above), 95% percentile bootstrap CI of the
                       mean delta: rng = random.Random(0), each of the 10000
                       resamples is rng.choices(units, k=n) over the n units,
                       the CI is the 2.5th and 97.5th percentile of the
                       resampled means with linear interpolation between order
                       statistics (numpy's default); at camera level the units
                       are cameras, and a cluster bootstrap additionally
                       resamples datasets with all their cameras.  Plus the
                       per-dataset delta table.  A and B must be in the raw
                       file; when no selected dataset has a result for both,
                       a one-line note replaces the tables.
  --ceiling RULE       the score a perfect tracker could reach: an object that
                       is absent from the reference camera's seed frame is never
                       prompted, so its ceiling is the fraction of frames where
                       its GT is empty; every other object scores 1.  RULE is
                       maxid (the runner's rule), count (REPORT.md P8) or both.
  --paper OUT.md       write every table the paper needs to a markdown file;
                       the reading flags above are fixed there and refused

Two aggregations of missing objects:

  as-is        every object in the ground truth counts; a method that exported
               no mask for an object is scored on an empty prediction, which is
               what the DAVIS protocol does and what an incomplete result costs.
  exported     (--exported-only) object/camera pairs for which the method
               produced no file at all are dropped.  It depends on the directory
               listings under Data/MVSeg and is nan for methods whose folders
               are gone, so it is off by default.  The objects it drops are
               listed after the tables; --no-missing suppresses that list
               (and does nothing without --exported-only).

Every table is preceded by one line stating aggregation, frames, split,
weights, window, dataset count and which of the two conventions it uses.
When a column covers fewer datasets than that count (--no-common) an
'n datasets' foot row gives the per-column count behind its AVERAGE.
Version 1 raw files (no meta/per_frame) still work for the plain tables and
--paired; the other readings need a version 2 file and say so.  Standard
library only.
"""
import argparse
import dataclasses
import json
import math
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
DATA = os.path.join(REPO, "Data", "MVSeg")

TIE = 1e-9                  # |delta| below this is a tie
N_BOOT = 10000              # bootstrap resamples
BOOT_SEED = 0
EXACT_MAX_N = 22            # exact Wilcoxon up to this many non-tied pairs
CEIL = "ceiling:"           # prefix of the ceiling pseudo-methods
METRIC_INDEX = {"J&F": 0, "J": 1, "F": 2}

# The internal comparison (--paper) runs at the *old* reference camera, because the
# SAM 2 folders exist only there; the external comparison against MUVOD runs at c_ini
# and lives in --muvod.  SAM 2 column is New3, not the published New1: New1 is not
# reproducible from any committed state and New3 is reproduced byte for byte
# (docs/sam2-baseline.md).  The 15-scene average moves by 0.0001.
PAPER_METHODS = ["SegMaskNew3", "SegMaskSam3XW0", "SegMaskSam3XW1GPS4"]
# The 12 datasets the original OneStageNew run completed; passed explicitly so
# the subset does not depend on which columns happen to be in the file.
PAPER_SUBSET12 = ["AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "Carpark",
                  "Dog", "Fencing", "Frog", "MATF", "Painter", "PoznanStreet",
                  "Welder"]


def paper_pairs(methods):
    """The three paired comparisons --paper reports, derived from the column order
    [external baseline, control, adopted]: control vs adopted (the mechanism), baseline
    vs adopted (the headline), baseline vs control (what the baseline is really beating).
    Deriving them means --methods stays consistent on its own; a hardcoded list silently
    produced `nan` columns whenever the two disagreed."""
    if len(methods) != 3:
        return [(a, b) for i, a in enumerate(methods) for b in methods[i + 1:]]
    base, control, adopted = methods
    return [(control, adopted), (base, adopted), (base, control)]

# --------------------------------------------------------------- MUVOD
# The benchmark our data comes from: Ashkani Chenarlogh et al., "MUVOD: A Novel
# Multi-view Video Object Segmentation Dataset and A Benchmark for 3D Segmentation",
# arXiv:2507.07519.  Its baseline is XMem applied both spatially and temporally.
# Tables III (basic) and IV (complete), J&F^3 in percent, in the paper's row order.
MUVOD_ORDER = ["Painter", "Breakfast", "Barn", "Frog", "Carpark", "PoznanStreet",
               "Fencing", "CBABasketball", "MartialArts", "Blocks", "MATF",
               "FlameSteak", "CoffeeMartini", "AlexaMeadeExhibit",
               "AlexaMeadeFacePaint", "Dog", "Welder"]
MUVOD_BASELINE = {
    "basic": {"Painter": 82.2, "Breakfast": 68.8, "Barn": 77.3, "Frog": 92.3,
              "Carpark": 85.3, "PoznanStreet": 80.2, "Fencing": 85.7,
              "CBABasketball": 66.4, "MartialArts": 83.7, "Blocks": 80.5, "MATF": 69.1,
              "FlameSteak": 84.8, "CoffeeMartini": 84.3, "AlexaMeadeExhibit": 82.8,
              "AlexaMeadeFacePaint": 64.9, "Dog": 75.5, "Welder": 85.2},
    "complete": {"Painter": 76.1, "Breakfast": 64.1, "Barn": 77.3, "Frog": 92.3,
                 "Carpark": 85.3, "PoznanStreet": 80.2, "Fencing": 85.7,
                 "CBABasketball": 65.6, "MartialArts": 83.7, "Blocks": 73.2,
                 "MATF": 59.3, "FlameSteak": 70.6, "CoffeeMartini": 76.7,
                 "AlexaMeadeExhibit": 82.8, "AlexaMeadeFacePaint": 60.4, "Dog": 67.2,
                 "Welder": 85.2},
}
MUVOD_GLOBAL = {"basic": 79.4, "complete": 75.6}
MUVOD_COL = "MUVOD XMem"
# The protocol itself: score the three annotated cameras, average the cameras into a
# scene score, average the scenes.  "basic" keeps only the objects visible in c_ini's
# reference frame; "complete" keeps every labelled object.
MUVOD_CFG = dict(agg="sequence", variant="all", split="all", weighted=False,
                 ref_rule="muvod")


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def resolve(name):
    for p in (name, os.path.join(DATA, name), os.path.join(ROOT, name)):
        if os.path.isfile(p):
            return p
    sys.exit(f"raw file not found: {name} (looked in ., {DATA}, {ROOT})")


def load(names):
    """(dataset, camera, method) -> entry.  A later file overrides an earlier one;
    one warning per (earlier, later) file pair reports the count and an example."""
    entries, origin, overridden = {}, {}, {}
    for name in names:
        with open(resolve(name)) as f:
            data = json.load(f)
        for e in data:
            k = (e["dataset"], e["camera"], e["method"])
            if k in entries:
                overridden.setdefault((origin[k], name), []).append(k)
            entries[k] = e
            origin[k] = name
    for (old, new), keys in overridden.items():
        print(f"warning: {len(keys)} entries of {old} overridden by {new}, "
              f"e.g. {'/'.join(keys[0])}", file=sys.stderr)
    return entries


class Store:
    """The loaded entries plus the indexes every reading needs."""

    def __init__(self, entries):
        self.E = entries
        self.datasets = sorted({d for d, _, _ in entries})
        self.methods = sorted({m for _, _, m in entries})
        self.cams = {d: sorted({c for dd, c, _ in entries if dd == d})
                     for d in self.datasets}
        # GT-only quantities (gt_area, seed ids, view index) are identical
        # across methods; keep one scored v2 entry per (dataset, camera).
        self.gt = {}
        for (d, c, _), e in sorted(entries.items()):
            if (d, c) not in self.gt and self.is_v2(e):
                self.gt[(d, c)] = e

    @staticmethod
    def is_v2(e):
        return (e.get("result") is not None and "meta" in e
                and "per_frame" in e["result"])

    def entry(self, d, cam, m):
        e = self.E.get((d, cam, m))
        return e if e is not None and e.get("result") is not None else None

    def require_v2(self, feature, methods):
        """Fail with a clear message when a v2-only feature meets a v1 file."""
        old = [k for k, e in self.E.items()
               if k[2] in methods and e.get("result") is not None and not self.is_v2(e)]
        if old:
            d, c, m = old[0]
            sys.exit(f"{feature} needs a version 2 raw file (schema 2, with meta and "
                     f"per_frame): {len(old)} entries lack them, e.g. {d}/{c}/{m}. "
                     f"Re-run eval/eval_jf.py or point --raw at jf_v2.json.")

    def ref_entry(self, d, rule):
        for cam in self.cams[d]:
            if (d, cam) in self.gt:
                ref = self.gt[(d, cam)]["meta"]["ref"]
                if rule not in ref:
                    sys.exit(f"{d}: this raw file has no '{rule}' reference rule "
                             f"(it has {', '.join(sorted(ref))}). Re-run "
                             f"eval/eval_jf.py to get schema 3.")
                return ref[rule]
        sys.exit(f"{d}: no version 2 entry to read the reference camera from")

    def ref_cam(self, d, rule="maxid"):
        return self.ref_entry(d, rule)["cam"]

    def object_filter(self, d, cfg):
        """The object ids a table may score, or None for 'every object'.  MUVOD's
        basic evaluation keeps only the objects visible in c_ini's reference frame."""
        if cfg.objects == "all":
            return None
        return frozenset(self.ref_entry(d, cfg.ref_rule)["seed_ids"])

    def view_index(self, d, cam):
        return self.gt[(d, cam)]["meta"]["view_index"]

    def n_views(self, d, cam):
        return self.gt[(d, cam)]["meta"]["n_views"]


def find_absent(store):
    """(dataset|camera|method) -> object ids with no exported mask in any frame.

    Read off the directory listings under Data/MVSeg, so it stays correct for
    any method set without a separate bookkeeping file, and wrong for methods
    whose folders were deleted (their objects all look absent).
    """
    absent = {}
    for (d, cam, m), e in store.E.items():
        if e.get("result") is None:
            continue
        mdir = os.path.join(DATA, d, m, cam)
        seen = set()
        if os.path.isdir(mdir):
            for fr in os.listdir(mdir):
                for f in os.listdir(os.path.join(mdir, fr)):
                    stem = os.path.splitext(f)[0]
                    if stem.isdigit():
                        seen.add(int(stem))
        gone = [o for o in e["objects"] if o not in seen]
        if gone:
            absent[f"{d}|{cam}|{m}"] = set(gone)
    return absent


def common_subset(store, methods, datasets):
    """Datasets where every selected method has a result, and the dropped ones."""
    have = {m: {d for (d, _, mm), e in store.E.items()
                if mm == m and e.get("result") is not None} for m in methods}
    keep = set(datasets).intersection(*have.values()) if have else set()
    return [d for d in datasets if d in keep], [d for d in datasets if d not in keep]


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Cfg:
    """What goes into the means: aggregation, frames, split, weights, window."""
    agg: str = "pooled"
    variant: str = "all"
    split: str = "all"
    weighted: bool = False
    window: int = 4
    nb_mode: str = "lower"      # lower = min(v, W); both = min(v, W) + min(N-1-v, W)
    objects: str = "all"        # all = every object in the camera's GT (MUVOD
                                # "complete"); basic = only those in c_ini's reference
                                # frame (MUVOD "basic")
    ref_rule: str = "maxid"     # which camera counts as the reference: maxid (the
                                # runner's own rule), count, or muvod (the config c_ini)

    def replace(self, **kw):
        """A copy with some fields changed; an unknown field is a TypeError."""
        return dataclasses.replace(self, **kw)

    def statement(self, n_datasets, exported=False, **override):
        """The one line printed before every table (D10)."""
        f = dict(aggregation=self.agg, frames=self.variant, split=self.split,
                 weights="gt_area" if self.weighted else "none",
                 window=self.window, datasets=n_datasets)
        if self.nb_mode != "lower":
            f["nb_mode"] = self.nb_mode         # the default keeps every existing statement line
        if self.objects != "all":
            f["objects"] = self.objects
        if self.ref_rule != "maxid":
            f["reference"] = self.ref_rule
        f.update(override)
        conv = ("exported only (objects with no output file dropped)" if exported
                else "as-is (missing objects scored as empty, the DAVIS convention)")
        return " | ".join(f"{k}={v}" for k, v in f.items()) + " | " + conv


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def frame_idx(variant, n):
    return range(n) if variant == "all" else range(1, n - 1)


def area_weights(e, variant):
    """w_o = mean GT area of the object over the variant's frames."""
    return [mean([row[f] for f in frame_idx(variant, len(row))])
            for row in e["result"]["per_frame"]["gt_area"]]


def ceiling_pairs(store, d, cam, rule, cfg):
    """[(J^, F^, w)] for one camera: 1 if the object is in the reference seed
    frame, else the fraction of frames whose GT is empty (scored 1 on an
    empty prediction), which is all an unprompted object can ever get."""
    e = store.gt.get((d, cam))
    if e is None:
        return None
    seeds = set(e["meta"]["ref"][rule]["seed_ids"])
    keep = store.object_filter(d, cfg)
    out = []
    for i, o in enumerate(e["objects"]):
        if keep is not None and o not in keep:
            continue
        row = e["result"]["per_frame"]["gt_area"][i]
        sel = [row[f] for f in frame_idx(cfg.variant, len(row))]
        j = 1.0 if o in seeds else sum(1 for a in sel if a == 0) / len(sel)
        out.append((j, j, mean(sel) if cfg.weighted else None))
    return out


def pairs_for(store, d, cam, m, cfg, absent=None):
    """[(J_o, F_o, w_o)] over the objects of one (dataset, camera, method);
    None when there is no result.  m may be a ceiling pseudo-method."""
    if m.startswith(CEIL):
        return ceiling_pairs(store, d, cam, m[len(CEIL):], cfg)
    e = store.entry(d, cam, m)
    if e is None:
        return None
    r = e["result"]
    w = area_weights(e, cfg.variant) if cfg.weighted else [None] * len(e["objects"])
    skip = absent.get(f"{d}|{cam}|{m}", set()) if absent else set()
    keep = store.object_filter(d, cfg)
    return [(r[f"J_{cfg.variant}"][i], r[f"F_{cfg.variant}"][i], w[i])
            for i, o in enumerate(e["objects"])
            if o not in skip and (keep is None or o in keep)]


def combine(pairs, weighted):
    """(J, F) over object scores, area-weighted when asked and the weights do
    not all vanish."""
    if weighted:
        sw = sum(w for _, _, w in pairs)
        if sw > 0:
            return (sum(w * j for j, _, w in pairs) / sw,
                    sum(w * f for _, f, w in pairs) / sw)
    return mean([j for j, _, _ in pairs]), mean([f for _, f, _ in pairs])


def jf(j, f):
    return (j + f) / 2, j, f


def select_cams(store, d, cfg):
    """The dataset's cameras after --split, reference under cfg.ref_rule."""
    cams = store.cams[d]
    if cfg.split == "all":
        return cams
    ref = store.ref_cam(d, cfg.ref_rule)
    return [c for c in cams if (c == ref) == (cfg.split == "ref")]


def score_all(store, methods, datasets, cfg, absent=None):
    """(dataset, method) -> (J&F, J, F) and (dataset, camera, method) -> same."""
    ds, cs = {}, {}
    for d in datasets:
        for m in methods:
            per_cam = {}
            for cam in select_cams(store, d, cfg):
                p = pairs_for(store, d, cam, m, cfg, absent)
                if p:
                    per_cam[cam] = p
                    cs[(d, cam, m)] = jf(*combine(p, cfg.weighted))
            if not per_cam:
                continue
            if cfg.agg == "pooled":
                ds[(d, m)] = jf(*combine([x for p in per_cam.values() for x in p],
                                         cfg.weighted))
            else:
                cam_scores = [cs[(d, c, m)] for c in per_cam]
                ds[(d, m)] = tuple(mean([s[i] for s in cam_scores]) for i in range(3))
    return ds, cs


# ----------------------------------------------------------------------------
# paired statistics
# ----------------------------------------------------------------------------

def avg_ranks(vals):
    """1-based ranks with ties averaged, and the tie group sizes."""
    order = sorted(range(len(vals)), key=vals.__getitem__)
    ranks, groups, i = [0.0] * len(vals), [], 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        groups.append(j - i + 1)
        i = j + 1
    return ranks, groups


def sign_test_p(wins, losses):
    """Exact two-sided binomial test on wins vs losses, ties excluded; 1 when
    there is nothing to test."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilcoxon_p(deltas):
    """Two-sided Wilcoxon signed-rank p-value and how it was obtained.

    Ties (|delta| < TIE) are dropped, leaving m deltas.  For m <= EXACT_MAX_N
    the null distribution of W+ is taken over all 2^m sign assignments (built
    by convolving the doubled ranks, which enumerates every assignment
    exactly); above that a normal approximation with continuity and tie
    correction.  The second value names the method and m.
    """
    d = [x for x in deltas if abs(x) >= TIE]
    n = len(d)
    if n == 0:
        return float("nan"), "none, m=0"
    ranks, groups = avg_ranks([abs(x) for x in d])
    wplus = sum(r for r, x in zip(ranks, d) if x > 0)
    if n <= EXACT_MAX_N:
        r2 = [int(round(2 * r)) for r in ranks]          # integer (doubled) ranks
        total = sum(r2)
        dist = [0] * (total + 1)                          # count of assignments per 2*W+
        dist[0] = 1
        for r in r2:
            for s in range(total - r, -1, -1):
                if dist[s]:
                    dist[s + r] += dist[s]
        dev = abs(2 * int(round(2 * wplus)) - total)      # distance from the centre
        count = sum(c for s, c in enumerate(dist) if abs(2 * s - total) >= dev)
        return count / 2 ** n, f"exact, m={n}"
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in groups) / 48
    z = max(0.0, abs(wplus - mu) - 0.5) / math.sqrt(var)
    return math.erfc(z / math.sqrt(2)), f"normal, m={n}"


def percentile(sorted_vals, q):
    """Linear interpolation between order statistics (numpy's default)."""
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


BOOT_SCHEME = (f"random.Random({BOOT_SEED}), rng.choices(units, k=n) per resample, "
               f"{N_BOOT} resamples, percentile CI with linear interpolation between "
               f"order statistics")
WILCOXON_SCHEME = (f"exact over the 2^m sign assignments of the m non-tied deltas for "
                   f"m<={EXACT_MAX_N}, normal approximation above")


def boot_ci(groups):
    """95% percentile bootstrap CI of the mean delta, resampling the groups
    with replacement (one delta per group = plain bootstrap; several = cluster
    bootstrap that keeps every camera of a resampled dataset).  Draw scheme:
    rng = random.Random(BOOT_SEED); each of the N_BOOT resamples is
    rng.choices(groups, k=len(groups)); the CI is the 2.5th and 97.5th
    percentile of the resampled means.  (nan, nan) without groups."""
    k = len(groups)
    if k == 0:
        return float("nan"), float("nan")
    rng = random.Random(BOOT_SEED)
    means = sorted(statistics.fmean([x for g in rng.choices(groups, k=k) for x in g])
                   for _ in range(N_BOOT))
    return percentile(means, 0.025), percentile(means, 0.975)


def paired_stats(deltas):
    """The paired-table row for one level; n=0 and nan fields without deltas."""
    wins = sum(1 for x in deltas if x >= TIE)
    losses = sum(1 for x in deltas if x <= -TIE)
    p_w, how = wilcoxon_p(deltas)
    lo, hi = boot_ci([[x] for x in deltas])
    return dict(n=len(deltas), mean=mean(deltas),
                median=statistics.median(deltas) if deltas else float("nan"),
                wins=wins, ties=len(deltas) - wins - losses, losses=losses,
                sign_p=sign_test_p(wins, losses), wilcoxon_p=p_w, wilcoxon=how,
                ci_lo=lo, ci_hi=hi)


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------

class Table:
    """A titled table: a label column plus named columns; rows and foot rows
    are (label, [cells]) with float / int / str / None cells.  `key` is the
    unique name used in the JSON dump, `name` what is displayed."""

    def __init__(self, key, name, statement, columns, rows, foot=(), fmt=None):
        self.key, self.name, self.statement = key, name, statement
        self.columns, self.rows, self.foot = list(columns), list(rows), list(foot)
        self.fmt = fmt or {}

    def cell(self, col, v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return format(v, self.fmt.get(col, ".4f"))
        return str(v)

    def as_json(self):
        return {"name": self.name, "statement": self.statement, "columns": self.columns,
                "rows": [dict([("label", lab)] + list(zip(self.columns, cells)))
                         for lab, cells in self.rows + self.foot]}


def H(text):
    return ("h", text)


def P(text):
    return ("p", text)


def render_text(items):
    out = []
    for it in items:
        if isinstance(it, tuple):
            kind, text = it
            if kind == "h":
                out += ["=" * 100, text, "=" * 100]
            else:
                out += [text, ""]
            continue
        t = it
        w = max([len(lab) for lab, _ in t.rows + t.foot] + [7]) + 2
        cw = [max([len(c) + 2, 13] + [len(t.cell(c, r[i])) + 2
                                      for _, r in t.rows + t.foot])
              for i, c in enumerate(t.columns)]
        rule = "-" * (w + sum(cw))
        out += [t.name, t.statement,
                " " * w + "".join(f"{c:>{cw[i]}s}" for i, c in enumerate(t.columns)),
                rule]
        out += [f"{lab:<{w}s}" + "".join(f"{t.cell(c, v):>{cw[i]}s}"
                                        for i, (c, v) in enumerate(zip(t.columns, r)))
                for lab, r in t.rows]
        if t.foot:
            out.append(rule)
            out += [f"{lab:<{w}s}" + "".join(f"{t.cell(c, v):>{cw[i]}s}"
                                            for i, (c, v) in enumerate(zip(t.columns, r)))
                    for lab, r in t.foot]
        out.append("")
    return "\n".join(out)


def render_md(items):
    out = []
    for it in items:
        if isinstance(it, tuple):
            kind, text = it
            out += [f"## {text}" if kind == "h" else text, ""]
            continue
        t = it
        out += [f"### {t.name}", "", f"*{t.statement}*", "",
                "| | " + " | ".join(t.columns) + " |",
                "|---|" + "---:|" * len(t.columns)]
        out += [f"| {lab} | " + " | ".join(t.cell(c, v) for c, v in zip(t.columns, r))
                + " |" for lab, r in t.rows]
        out += [f"| **{lab}** | " + " | ".join(
            f"**{t.cell(c, v)}**" for c, v in zip(t.columns, r)) + " |"
                for lab, r in t.foot]
        out.append("")
    return "\n".join(out)


def count_foot(counts, n):
    """The 'n datasets' foot row: how many datasets each column's AVERAGE
    covers, shown only when some column covers fewer than the n the statement
    claims (--no-common), so datasets=n never overstates a mean."""
    return [("n datasets", counts)] if any(c < n for c in counts) else []


def metric_tables(store, methods, datasets, cfg, metrics=("J&F", "J", "F"),
                  absent=None, exported=False, ceiling=(), key=""):
    """One per-dataset table per metric with an AVERAGE foot row (D3/D4)."""
    cols = list(methods) + [CEIL + r for r in ceiling]
    ds, _ = score_all(store, cols, datasets, cfg, absent)
    counts = [sum(1 for d in datasets if (d, m) in ds) for m in cols]
    tables = []
    for metric in metrics:
        i = METRIC_INDEX[metric]
        rows = [(d, [ds[(d, m)][i] if (d, m) in ds else None for m in cols])
                for d in datasets]
        foot = [("AVERAGE", [mean([ds[(d, m)][i] for d in datasets if (d, m) in ds])
                             for m in cols])] + count_foot(counts, len(datasets))
        tables.append(Table(f"{key}{metric}", metric,
                            cfg.statement(len(datasets), exported), cols, rows, foot))
    return tables


def summary_table(store, methods, datasets, cfg, absent, exported, key=""):
    """The AVERAGE row for every frames x convention combination."""
    modes = [(False, "as-is")]
    if exported:
        modes.append((True, "exported only"))
    rows, counts = [], []
    for variant, vlabel in (("all", "all frames"), ("inner", "DAVIS (drop first/last)")):
        for drop, dlabel in modes:
            ds, _ = score_all(store, methods, datasets, cfg.replace(variant=variant),
                              absent if drop else None)
            rows.append((f"J&F  {vlabel:<24s} {dlabel}",
                         [mean([ds[(d, m)][0] for d in datasets if (d, m) in ds])
                          for m in methods]))
            counts.append([sum(1 for d in datasets if (d, m) in ds) for m in methods])
    if all(c == counts[0] for c in counts):
        foot = count_foot(counts[0], len(datasets))
    else:                       # exported-only can empty a dataset for one row
        foot = [(f"n datasets ({lab.strip()})", c) for (lab, _), c in zip(rows, counts)]
    return Table(f"{key}summary", "AVERAGE row only, for every combination",
                 cfg.statement(len(datasets), frames="per row"), methods, rows, foot)


def ceiling_counts(store, datasets, cfg, rules, key=""):
    """Per dataset: unreachable objects and structurally-zero (object, frame)
    pairs (unreachable and GT non-empty) under each rule, on the kept cameras."""
    cols = [f"unreachable objects ({r})" for r in rules] + \
           [f"structural zeros ({r})" for r in rules]
    rows, tot = [], [0] * len(cols)
    for d in datasets:
        cnt = [0] * len(cols)
        for cam in select_cams(store, d, cfg):
            e = store.gt.get((d, cam))
            if e is None:
                continue
            for ri, rule in enumerate(rules):
                seeds = set(e["meta"]["ref"][rule]["seed_ids"])
                for i, o in enumerate(e["objects"]):
                    if o in seeds:
                        continue
                    row = e["result"]["per_frame"]["gt_area"][i]
                    cnt[ri] += 1
                    cnt[len(rules) + ri] += sum(
                        1 for f in frame_idx(cfg.variant, len(row)) if row[f] > 0)
        rows.append((d, cnt))
        tot = [a + b for a, b in zip(tot, cnt)]
    return Table(f"{key}ceiling counts", "ceiling: unreachable objects and structural zeros",
                 cfg.statement(len(datasets)), cols, rows, [("TOTAL", tot)])


def nb_of(store, d, cam, cfg):
    """Neighbour count of a camera under the gather's neighbourhood.  lower: min(v, W)
    (Phase 2 XW, modes A/D).  both: min(v, W) + min(N-1-v, W) (modes B/C/E), where N is
    the dataset's camera count (meta n_views)."""
    v, W = store.view_index(d, cam), cfg.window
    if cfg.nb_mode == "lower":
        return min(v, W)
    return min(v, W) + min(store.n_views(d, cam) - 1 - v, W)


def nb_max(cfg):
    return cfg.window if cfg.nb_mode == "lower" else 2 * cfg.window


def nb_table(store, methods, datasets, cfg, pair=None, key=""):
    """Cameras pooled across datasets and binned by nb (nb_of), capped at nb_max."""
    cols = list(methods) + [m for m in (pair or ()) if m not in methods]
    _, cs = score_all(store, cols, datasets, cfg)
    bins = {b: [] for b in range(nb_max(cfg) + 1)}
    skipped = []                # cameras only a version 1 entry knows about
    for d in datasets:
        for cam in select_cams(store, d, cfg):
            if (d, cam) not in store.gt:
                skipped.append(f"{d}/{cam}")
                continue
            bins[min(nb_of(store, d, cam, cfg), nb_max(cfg))].append((d, cam))
    if skipped:
        print(f"warning: nb bins: {len(skipped)} cameras without a version 2 entry "
              f"(no view index) left out: {', '.join(skipped)}", file=sys.stderr)
    columns = ["cameras"] + list(methods)
    if pair:
        columns += ["delta J&F", "delta J", "delta F", "wins", "ties", "losses",
                    "cluster CI lo", "cluster CI hi"]
    groups = [(f"nb={b}" if b < nb_max(cfg) else f"nb>={b}", cams) for b, cams in bins.items()]
    if cfg.nb_mode == "both":
        # SPEC_P4 amendment A4: the cumulative reading (every camera with a neighbour on
        # either side, the primary bin of modes B/C/E) next to the fixed-partition bins
        groups.append(("nb>=1", [c for b, cams in bins.items() if b >= 1 for c in cams]))
    rows = []
    for label, cams in groups:
        cells = [len(cams)] + [mean([cs[(d, c, m)][0] for d, c in cams if (d, c, m) in cs])
                               for m in methods]
        if pair:
            a, bm = pair
            dl = [[cs[(d, c, bm)][i] - cs[(d, c, a)][i] for i in range(3)]
                  for d, c in cams if (d, c, a) in cs and (d, c, bm) in cs]
            djf = [x[0] for x in dl]
            cells += [mean([x[i] for x in dl]) for i in range(3)]
            cells += [sum(1 for x in djf if x >= TIE), sum(1 for x in djf if abs(x) < TIE),
                      sum(1 for x in djf if x <= -TIE)]
            # cluster bootstrap of the bin's mean delta J&F, datasets as clusters
            # (same boot_ci / seed / scheme as the paired camera-level row)
            groups = [[cs[(d, c, bm)][0] - cs[(d, c, a)][0]
                       for d2, c in cams if d2 == d and (d, c, a) in cs and (d, c, bm) in cs]
                      for d in datasets]
            cells += list(boot_ci([g for g in groups if g]))
        rows.append((label, cells))
    name = "camera J&F by nb bin" + (f", delta = {pair[1]} - {pair[0]}" if pair else "")
    statement = (cfg.statement(len(datasets), bootstrap=BOOT_SCHEME) if pair
                 else cfg.statement(len(datasets)))
    return Table(f"{key}nb bins", name, statement, columns, rows)


def paired_tables(store, a, b, datasets, cfg, key=""):
    """D8: dataset-level and camera-level statistics of delta = B - A, and the
    per-dataset delta table sorted by delta.  A one-line note instead when no
    dataset has a result for both methods."""
    ds, cs = score_all(store, [a, b], datasets, cfg)
    d_rows = [(d, ds[(d, a)][0], ds[(d, b)][0]) for d in datasets
              if (d, a) in ds and (d, b) in ds]
    if not d_rows:
        return [P(f"paired {b} - {a}: none of the {len(datasets)} selected datasets "
                  f"has a result for both methods, paired tables skipped")]
    d_delta = [y - x for _, x, y in d_rows]
    groups = [[cs[(d, c, b)][0] - cs[(d, c, a)][0] for c in select_cams(store, d, cfg)
               if (d, c, a) in cs and (d, c, b) in cs] for d in datasets]
    groups = [g for g in groups if g]
    c_delta = [x for g in groups for x in g]
    cols = ["n", "mean", "median", "wins", "ties", "losses", "sign p", "wilcoxon p",
            "wilcoxon", "CI lo", "CI hi", "cluster CI lo", "cluster CI hi"]
    fmt = {"sign p": ".4g", "wilcoxon p": ".4g"}

    def row(st, cluster):
        return [st["n"], st["mean"], st["median"], st["wins"], st["ties"], st["losses"],
                st["sign_p"], st["wilcoxon_p"], st["wilcoxon"], st["ci_lo"], st["ci_hi"],
                cluster[0], cluster[1]]

    rows = [("dataset", row(paired_stats(d_delta), (None, None))),
            ("camera", row(paired_stats(c_delta), boot_ci(groups)))]
    stats = Table(f"{key}paired stats", f"paired: {b} - {a}",
                  cfg.statement(len(d_rows), bootstrap=BOOT_SCHEME,
                                wilcoxon=WILCOXON_SCHEME), cols, rows, fmt=fmt)
    deltas = Table(f"{key}paired per dataset", f"per-dataset delta: {b} - {a}",
                   cfg.statement(len(d_rows)), [a, b, "delta"],
                   sorted([(d, [x, y, y - x]) for d, x, y in d_rows], key=lambda r: r[1][2]))
    return [stats, deltas]


def agg_table(store, methods, datasets, cfg, key=""):
    """Pooled and sequence aggregation side by side, J&F."""
    pooled, _ = score_all(store, methods, datasets, cfg.replace(agg="pooled"))
    seq, _ = score_all(store, methods, datasets, cfg.replace(agg="sequence"))
    cols = [f"{m} {agg}" for m in methods for agg in ("pooled", "sequence")]
    pick = [(m, s) for m in methods for s in (pooled, seq)]
    rows = [(d, [s[(d, m)][0] if (d, m) in s else None for m, s in pick]) for d in datasets]
    foot = [("AVERAGE", [mean([s[(d, m)][0] for d in datasets if (d, m) in s])
                         for m, s in pick])]
    foot += count_foot([sum(1 for d in datasets if (d, m) in s) for m, s in pick],
                       len(datasets))
    return Table(f"{key}pooled vs sequence", "J&F, pooled vs sequence aggregation",
                 cfg.statement(len(datasets), aggregation="pooled+sequence (per column)"),
                 cols, rows, foot)


# ----------------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------------

def subset_note(store, methods, datasets, common):
    if not common:
        return datasets, P(f"all datasets: {len(datasets)}")
    keep, dropped = common_subset(store, methods, datasets)
    return keep, P(f"common subset: {len(keep)} datasets"
                   + (f"   (dropped: {', '.join(dropped)})" if dropped else ""))


def default_items(store, methods, datasets, cfg, args):
    """The plain report: per-dataset tables, the AVERAGE summary, extras."""
    # The common subset also honours the --paired pair, so a pair outside
    # --methods gets the same datasets as the statement claims.
    covered = list(methods) + [m for m in (args.paired or ()) if m not in methods]
    datasets, note = subset_note(store, covered, datasets, args.common)
    items = [note]
    if not datasets:
        return items + [P("no dataset has a result for every selected method")]
    absent = find_absent(store) if args.exported_only else None
    rules = {"maxid": ("maxid",), "count": ("count",), "both": ("maxid", "count"),
             None: ()}[args.ceiling]
    modes = [(False, "as-is  (missing exports scored as empty predictions)")]
    if args.exported_only:
        modes.append((True, "exported only  (objects with no output file dropped)"))
    for drop, label in modes:
        key = ("exported/" if drop else "as-is/")
        items.append(H(f"{label}   -   {cfg.variant} frames, {len(datasets)} datasets"))
        items += metric_tables(store, methods, datasets, cfg, ("J&F", "J", "F"),
                               absent if drop else None, drop, () if drop else rules, key)
        if rules and not drop:
            items.append(ceiling_counts(store, datasets, cfg, rules, key))
    items.append(H("AVERAGE row only, for every combination"))
    items.append(summary_table(store, methods, datasets, cfg, absent, args.exported_only))
    if args.bin_by_nb:
        items.append(H(f"cameras binned by nb = min(view_index, {cfg.window})"
                       if cfg.nb_mode == "lower" else
                       f"cameras binned by nb = min(view_index, {cfg.window}) + "
                       f"min(n_views-1-view_index, {cfg.window})   (+ cumulative nb>=1 row)"))
        items.append(nb_table(store, methods, datasets, cfg, args.paired))
    if args.paired:
        a, b = args.paired
        items.append(H(f"paired comparison: {b} - {a}"))
        items += paired_tables(store, a, b, datasets, cfg)
    if not args.no_missing and absent:
        lines = ["objects with no exported mask at all (camera, object) - only these "
                 "differ between the two aggregations:"]
        for k, v in sorted(absent.items()):
            d, cam, m = k.split("|")
            if m in methods:
                lines.append(f"  {d}/{cam}/{m}: {sorted(v)}")
        items.append(P("\n".join(lines)))
    return items


def muvod_table(store, methods, datasets, cfg, setting, key=""):
    """MUVOD Table III/IV shape: one row per scene, J&F^3 in percent, our methods
    beside the published XMem baseline, and a Global row of unweighted scene means."""
    ds, _ = score_all(store, methods, datasets, cfg)
    pub = MUVOD_BASELINE[setting]
    cols = list(methods) + [MUVOD_COL] + [f"{m} - {MUVOD_COL}" for m in methods]
    order = [d for d in MUVOD_ORDER if d in datasets] + \
            [d for d in datasets if d not in MUVOD_ORDER]
    rows = []
    for d in order:
        ours = [100 * ds[(d, m)][0] if (d, m) in ds else None for m in methods]
        base = pub.get(d)
        rows.append((d, ours + [base] +
                     [None if (o is None or base is None) else o - base for o in ours]))
    # Global: the mean over the scenes in the table, so our column and the baseline
    # column are averaged over exactly the same scenes.  The paper's own global is
    # over all 17 and is printed beside it whenever the table is short.
    def col_mean(i):
        vals = [r[1][i] for r in rows if r[1][i] is not None]
        return mean(vals) if vals else None
    foot = [("Global (%d scenes)" % len(rows), [col_mean(i) for i in range(len(cols))])]
    if len(rows) != len(MUVOD_ORDER):
        foot.append(("MUVOD global (17 scenes)",
                     [None] * len(methods) + [MUVOD_GLOBAL[setting]] + [None] * len(methods)))
    st = cfg.statement(len(rows), objects=setting,
                       protocol="MUVOD J&F^3 (per-camera score, averaged over the "
                                "three annotated cameras, then over scenes)")
    return Table(f"{key}muvod-{setting}", f"MUVOD {setting} evaluation, J&F^3 (%)",
                 st, cols, rows, foot, fmt={c: ".1f" for c in cols})


def muvod_items(store, methods, datasets, cfg, args):
    """The MUVOD comparison: the basic table (the paper's headline), the complete
    table, and the per-scene object counts each object set rests on."""
    datasets, note = subset_note(store, methods, datasets, args.common)
    items = [note,
             P("Baseline column: XMem applied spatially and temporally, as reported in "
               "arXiv:2507.07519 Tables III and IV.  c_ini per scene comes from "
               "MVSeg.json; see docs/muvod-protocol.md for how each one was fixed."),
             H("basic evaluation - only the objects visible in c_ini's reference frame")]
    if not datasets:
        return items + [P("no dataset has a result for every selected method")]
    items.append(muvod_table(store, methods, datasets, cfg.replace(objects="basic"),
                             "basic"))
    items.append(H("complete evaluation - every labelled object"))
    items.append(muvod_table(store, methods, datasets, cfg.replace(objects="all"),
                             "complete"))
    items.append(H("what each object set contains"))
    rows = []
    for d in [x for x in MUVOD_ORDER if x in datasets] + \
            [x for x in datasets if x not in MUVOD_ORDER]:
        ref = store.ref_entry(d, cfg.ref_rule)
        cams = select_cams(store, d, cfg)
        allobj = sorted({o for c in cams for o in store.gt[(d, c)]["objects"]})
        basic = set(ref["seed_ids"])
        rows.append((d, [ref["cam"], len(basic), len(allobj),
                         "yes" if basic >= set(allobj) else "no",
                         "yes" if MUVOD_BASELINE["basic"].get(d) ==
                                  MUVOD_BASELINE["complete"].get(d) else "no"]))
    items.append(Table("muvod-objects", "object sets per scene",
                       "c_ini and the two object sets; the last two columns must agree, "
                       "because MUVOD's basic and complete scores are equal exactly when "
                       "c_ini's reference frame already holds every labelled object",
                       ["c_ini", "basic objects", "all objects",
                        "basic == all (ours)", "basic == complete (MUVOD)"], rows))
    return items


def paper_items(store, methods, datasets, window, args):
    """Every table the paper needs, in the order of the specification (D11)."""
    datasets, note = subset_note(store, methods, datasets, args.common)
    sub12 = [d for d in datasets if d in PAPER_SUBSET12]
    sets = ((f"{len(datasets)} datasets", datasets, "all/"),
            (f"{len(sub12)}-dataset subset", sub12, "sub/"))
    base = Cfg(window=window)
    items = [P(f"methods: {', '.join(methods)}"), note,
             P(f"subset: {len(sub12)} datasets ({', '.join(sub12)})")]

    items.append(H(f"1. Headline, {len(datasets)} datasets"))
    items += metric_tables(store, methods, datasets, base, key="1 headline/")
    items += metric_tables(store, methods, datasets, base.replace(variant="inner"),
                           ("J&F",), key="1 headline/inner ")

    items.append(H(f"2. Headline, {len(sub12)}-dataset subset"))
    items += metric_tables(store, methods, sub12, base, key="2 subset/")
    items += metric_tables(store, methods, sub12, base.replace(variant="inner"),
                           ("J&F",), key="2 subset/inner ")

    items.append(H("3. Reference camera only / non-reference cameras only"))
    for split in ("ref", "nonref"):
        for label, dsl, k in sets:
            items.append(P(f"**{split}, {label}**"))
            items += metric_tables(store, methods, dsl, base.replace(split=split),
                                   key=f"3 {split} {k}")

    items.append(H(f"4. Camera J&F by nb = min(view_index, {window})"))
    pairs = paper_pairs(list(methods))
    for pair in pairs[:2]:
        for label, dsl, k in sets:
            items.append(P(f"**{pair[1]} - {pair[0]}, {label}**"))
            items.append(nb_table(store, methods, dsl, base, pair,
                                  key=f"4 {k}{pair[1]}-{pair[0]} "))

    items.append(H("5. Paired statistics"))
    for a, b in pairs:
        for label, dsl, k in sets:
            items.append(P(f"**{b} - {a}, {label}**"))
            items += paired_tables(store, a, b, dsl, base, key=f"5 {k}{b}-{a} ")

    items.append(H("6. Area-weighted headline"))
    for label, dsl, k in sets:
        items.append(P(f"**{label}**"))
        items += metric_tables(store, methods, dsl, base.replace(weighted=True),
                               key=f"6 area {k}")

    items.append(H(f"7. Ceiling, both reference rules, {len(datasets)} datasets"))
    rules = ("maxid", "count")
    items += metric_tables(store, methods, datasets, base, ceiling=rules, key="7 ceiling/")
    items.append(ceiling_counts(store, datasets, base, rules, key="7 ceiling/"))

    items.append(H("8. Pooled vs sequence aggregation"))
    items.append(agg_table(store, methods, datasets, base, key="8 "))
    return items


def non_negative(s):
    """argparse type for --window: an int >= 0."""
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", nargs="+", default=["jf_v2.json"],
                    help="one or more eval_jf.py outputs; entries are pooled, "
                         "the last file wins on duplicates")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="result folders to show, in this order (default: all in "
                         "the file; with --paper: %s)" % " ".join(PAPER_METHODS))
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="restrict to these datasets")
    ap.add_argument("--common", dest="common", action="store_true", default=True,
                    help="restrict to the datasets every selected method covers, "
                         "so the averages compare like with like (default)")
    ap.add_argument("--no-common", dest="common", action="store_false",
                    help="keep every dataset any selected method covers")
    ap.add_argument("--no-missing", action="store_true",
                    help="skip the list of objects with no exported mask that "
                         "--exported-only prints (no effect without it)")
    ap.add_argument("--exported-only", action="store_true",
                    help="also print the 'exported only' tables (directory listings)")
    ap.add_argument("--aggregation", choices=("pooled", "sequence"), default="pooled",
                    help="dataset score: over all (camera, object) pairs, or the "
                         "mean of the camera scores")
    ap.add_argument("--frames", choices=("all", "inner"), default="all",
                    help="which stored per-object means to use")
    ap.add_argument("--split", choices=("all", "ref", "nonref"), default="all",
                    help="reference camera only, or the non-reference cameras only")
    ap.add_argument("--window", type=non_negative, default=4,
                    help="W >= 0 for the nb = min(view_index, W) bins")
    ap.add_argument("--nb-mode", choices=("lower", "both"), default="lower",
                    help="nb definition for --bin-by-nb: lower = min(view_index, W) (XW, "
                         "modes A/D); both = min(view_index, W) + min(N-1-view_index, W) "
                         "(modes B/C/E)")
    ap.add_argument("--area-weighted", action="store_true",
                    help="weight objects by their mean GT area")
    ap.add_argument("--bin-by-nb", action="store_true",
                    help="camera J&F per nb bin, cameras pooled across datasets")
    ap.add_argument("--paired", nargs=2, metavar=("A", "B"),
                    help="paired statistics of B - A at dataset and camera level")
    ap.add_argument("--ceiling", choices=("maxid", "count", "both"),
                    help="add reachable-score ceiling columns under this rule")
    ap.add_argument("--objects", choices=("all", "basic"), default="all",
                    help="which objects go into the means: all of the camera's ground "
                         "truth (MUVOD's complete evaluation) or only those visible in "
                         "the reference frame (MUVOD's basic evaluation)")
    ap.add_argument("--ref-rule", choices=("maxid", "count", "muvod"), default="maxid",
                    help="which camera counts as the reference for --split and "
                         "--objects basic: the runner's max-id rule, the object-count "
                         "rule, or MUVOD's c_ini from MVSeg.json")
    ap.add_argument("--muvod", action="store_true",
                    help="the MUVOD comparison: J&F^3 per scene against the published "
                         "XMem baseline, basic and complete. Fixes the protocol "
                         "(--aggregation sequence, --ref-rule muvod, all frames) and "
                         "refuses flags that would change it")
    ap.add_argument("--paper", metavar="OUT.md",
                    help="write every paper table to this markdown file (its "
                         "settings are fixed: the reading flags are refused)")
    ap.add_argument("--dump-json", metavar="PATH",
                    help="also write every printed table to this JSON file")
    args = ap.parse_args()

    if args.muvod:
        if args.paper:
            sys.exit("--muvod and --paper each fix their own settings; run them separately")
        ignored = [flag for flag, on in (
            ("--aggregation", args.aggregation != "pooled"),
            ("--frames", args.frames != "all"),
            ("--split", args.split != "all"),
            ("--objects", args.objects != "all"),
            ("--ref-rule", args.ref_rule != "maxid"),
            ("--area-weighted", args.area_weighted),
            ("--exported-only", args.exported_only)) if on]
        if ignored:
            sys.exit("--muvod fixes the protocol and would ignore: "
                     f"{', '.join(ignored)}; drop them or run without --muvod")

    if args.paper:
        ignored = [flag for flag, on in (
            ("--aggregation", args.aggregation != "pooled"),
            ("--frames", args.frames != "all"),
            ("--split", args.split != "all"),
            ("--area-weighted", args.area_weighted),
            ("--bin-by-nb", args.bin_by_nb),
            ("--paired", args.paired is not None),
            ("--ceiling", args.ceiling is not None),
            ("--exported-only", args.exported_only),
            ("--nb-mode", args.nb_mode != "lower")) if on]
        if ignored:
            sys.exit(f"--paper fixes its own settings and would ignore: "
                     f"{', '.join(ignored)}; drop them or run without --paper")

    store = Store(load(args.raw))
    for m in args.paired or ():
        if m not in store.methods:
            sys.exit(f"--paired: {m} is not in {', '.join(args.raw)}; available "
                     f"methods: {', '.join(store.methods)}")
    methods = args.methods or (PAPER_METHODS if args.paper else store.methods)
    datasets = [d for d in store.datasets if not args.datasets or d in args.datasets]
    cfg = Cfg(args.aggregation, args.frames, args.split, args.area_weighted, args.window,
              args.nb_mode, args.objects, args.ref_rule)
    if args.muvod:
        cfg = cfg.replace(**MUVOD_CFG)
    for feature, on in (("--split ref|nonref", args.split != "all"),
                        ("--area-weighted", args.area_weighted),
                        ("--bin-by-nb", args.bin_by_nb),
                        ("--ceiling", args.ceiling is not None),
                        ("--objects basic", args.objects != "all"),
                        ("--ref-rule", args.ref_rule != "maxid"),
                        ("--muvod", args.muvod),
                        ("--paper", args.paper is not None)):
        if on:
            store.require_v2(feature, methods)

    if args.muvod:
        items = muvod_items(store, methods, datasets, cfg, args)
        print(render_text(items))
    elif args.paper:
        items = paper_items(store, methods, datasets, args.window, args)
        text = "\n".join(["# Paper tables", "",
                          f"generated by `eval/report_jf.py --paper` from "
                          f"{', '.join(args.raw)}", "", render_md(items)])
        with open(args.paper, "w") as f:
            f.write(text)
        sys.stdout.write(text)          # stdout is exactly the file
        n = sum(1 for it in items if isinstance(it, Table))
        print(f"wrote {args.paper}: {n} tables", file=sys.stderr)
    else:
        items = default_items(store, methods, datasets, cfg, args)
        print(render_text(items))

    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump({it.key: it.as_json() for it in items if isinstance(it, Table)},
                      f, indent=1)


if __name__ == "__main__":
    main()
