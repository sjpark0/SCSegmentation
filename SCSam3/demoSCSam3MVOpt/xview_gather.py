"""Cross-view memory gather of SCSam3TrackerPredictorNewMem (Phase 2, REPORT.md P2).

Torch-free so it is unit-testable on the host.  `output_dicts` is the per-object list
the tracker builds in propagate_in_video (:838-843): one entry per session, in
session_ids order (== camera index under --track-cams all/closure), None when the
session has no state for the object; each entry holds "cond_frame_outputs" and
"non_cond_frame_outputs" keyed by frame index.

LEGACY (window=4, hygiene=False) reproduces the loop that lived at :1262-1275 bit for
bit: Python's negative-index wrap, the IndexError when len(output_dicts) < window, and
the dead cond-frame fallback whose only effect is rebinding the caller's
`selected_cond_outputs` (REPORT.md C1/C2).  That rebinding is returned as `rebound`
so the caller can apply it exactly where the old code did.

HYGIENE (hygiene=True) = fixes (2) and (3) of ROADMAP Phase 2: negative neighbour
indices are skipped (no wrap, no IndexError) and the fallback is gone (no rebinding).

Consequence worth keeping in mind when reading XW results (SPEC amendment F4): under
hygiene the camera at index 0 has no non-negative neighbour for ANY window, so its
encoder inputs are identical across every W (and across closure/all).  An nb=0 delta
between two XW runs that is not exactly 0 therefore signals nondeterminism or a
cross-session leak outside this gather, never a control effect of W.

Phase 3 / P4 adds the neighbourhood variants (docs/phase3-neighbourhood.md): A = lower
views v-W..v-1 at frame t (today's gather), B = both sides v-W..v-1 and v+1..v+W at
t-1, C = lower at t and upper at t-1, D = lower at t-1, E = pass 1 as B, then a second
pass that recomputes t from v-W..v+W at t (the pass-1 outputs, Jacobi commit).  Frame
convention: t is the frame being computed, "tm1" is t-1 (t+1 when tracking in reverse);
the seed frame is found in cond_frame_outputs.  Under two-sided modes camera 0 DOES have
neighbours, so the nb=0 sentinel of Phase 2 does not apply; use the A-vs-D index-0
identity and closure==all instead (SPEC_P4 §6).

Phase 3 / P12 adds three conditioning knobs on the neighbour tokens (G gate, P pointer,
S row shift; `resolve_cross_view_knobs`, `gate_cross_view`); with all three at their
defaults the tracker's code path is today's, statement for statement
(docs/phase3-conditioning.md).
"""

UNCHANGED = object()      # "leave the caller's selected_cond_outputs as it is"
LEGACY_WINDOW = 4
LEGACY_HYGIENE = False
# Phase 3 / P4 neighbourhood variants (docs/phase3-neighbourhood.md).  The letter is the
# tracker's `cross_view_mode`; None means "A".  A is, statement for statement, the Phase 2
# gather.  Every other letter needs hygiene (no wrap, no C2 rebinding) and is its own
# lineage XW{W}{letter}.  Frame convention: t = frame being computed, "tm1" = t-1
# (t+1 when tracking in reverse); the seed frame is found in cond_frame_outputs.
MODES = ("A", "B", "C", "D", "E")
LEGACY_MODE = "A"
GATHER_MODES = ("lower_t", "lower_tm1", "both_tm1", "mixed", "all_t")
_MODE_GATHER = {"A": "lower_t",     # v-W..v-1 at t            (today)
                "B": "both_tm1",    # v-W..v-1, v+1..v+W at t-1
                "C": "mixed",       # v-W..v-1 at t, v+1..v+W at t-1   (REPORT P4)
                "D": "lower_tm1",   # v-W..v-1 at t-1          (frame-freshness control)
                "E": "both_tm1"}    # pass 1 = B; pass 2 = all_t
E_PASS2_GATHER = "all_t"            # v-W..v+W at t, every entry a pass-1 output


def resolve_cross_view(cross_view_window, cross_view_hygiene, num_maskmem):
    """(window, hygiene) for the constructor; None -> legacy.  Raises ValueError early."""
    window = LEGACY_WINDOW if cross_view_window is None else int(cross_view_window)
    hygiene = LEGACY_HYGIENE if cross_view_hygiene is None else bool(cross_view_hygiene)
    if not 0 <= window <= num_maskmem - 1:
        raise ValueError(
            f"cross_view_window={window}: neighbour s_pos=-k uses maskmem_tpos_enc row k-1 "
            f"and the parameter has {num_maskmem} rows (row {num_maskmem - 1} is the cond-frame "
            f"row), so 0 <= W <= {num_maskmem - 1}")
    if window != LEGACY_WINDOW and not hygiene:
        raise ValueError(
            f"cross_view_window={window} without cross_view_hygiene would wrap negative "
            "indices on a different neighbourhood (operations.md 함정 1); pass "
            "cross_view_hygiene=True")
    return window, hygiene


def resolve_cross_view_mode(cross_view_mode, hygiene):
    """Canonical mode letter for the constructor; None -> "A".  Raises ValueError early."""
    mode = LEGACY_MODE if cross_view_mode is None else str(cross_view_mode).upper()
    if mode not in MODES:
        raise ValueError(f"cross_view_mode={cross_view_mode!r}: expected one of {MODES}")
    if mode != LEGACY_MODE and not hygiene:
        raise ValueError(f"cross_view_mode={mode} needs cross_view_hygiene=True (the legacy "
                         "gather wraps negative indices and rebinds the cond pointer, REPORT C1/C2)")
    return mode


def gather_mode_for(cross_view_mode, xview_pass=None):
    """Which gather variant a call uses.  xview_pass None/1 = the generator's pass (every
    mode); 2 = the recompute pass, which exists only in mode E."""
    if xview_pass not in (None, 1, 2):
        raise ValueError(f"xview_pass={xview_pass!r}: expected None, 1 or 2")
    if xview_pass == 2:
        if cross_view_mode != "E":
            raise ValueError(f"xview_pass=2 is defined for mode E only, not {cross_view_mode!r}")
        return E_PASS2_GATHER
    return _MODE_GATHER[cross_view_mode]


# Phase 3 / P12 neighbour-token conditioning knobs (docs/phase3-conditioning.md).  All
# opt-in; (False, False, 0) is today's XW behaviour statement for statement.
#   G  cross_view_gate        use a neighbour entry only if it passes frame_filter's
#                             score test (eff_iou_score > mf_threshold); see `admits`
#                             for the two deliberate differences
#   P  cross_view_ptr         append the neighbours' object pointers after the own ones
#   S  cross_view_tpos_shift  neighbour v-k uses maskmem_tpos_enc row k-1+s (and, with P,
#                             pointer position k+s) instead of row k-1.  S only moves the
#                             neighbour inside the non-cond rows: REPORT P12's other two
#                             variants (put the neighbour on the cond row 6, or use a mean
#                             of rows) are OUT OF SCOPE here and are not implemented.
# Folder / lineage suffix: G, P, S<s> in that order after the mode letter (A: no letter).


def resolve_cross_view_knobs(cross_view_gate, cross_view_ptr, cross_view_tpos_shift,
                             window, hygiene, num_maskmem):
    """(gate, ptr, shift) for the constructor; None -> (False, False, 0).  Raises early.

    S bound: neighbour v-k adds maskmem_tpos_enc[k-1+s] and row num_maskmem-1 is the cond
    row, so k-1+s <= num_maskmem-2 for every k <= W  <=>  W + s <= num_maskmem-1.
    """
    gate = False if cross_view_gate is None else bool(cross_view_gate)
    ptr = False if cross_view_ptr is None else bool(cross_view_ptr)
    shift = 0 if cross_view_tpos_shift is None else int(cross_view_tpos_shift)
    if shift < 0:
        raise ValueError(f"cross_view_tpos_shift={shift}: must be >= 0")
    if window + shift > num_maskmem - 1:
        raise ValueError(
            f"cross_view_window={window} + cross_view_tpos_shift={shift} > {num_maskmem - 1}: "
            f"neighbour v-{window} would use maskmem_tpos_enc row {window - 1 + shift}, "
            f"but row {num_maskmem - 1} is the cond-frame row (W + s <= {num_maskmem - 1})")
    if (gate or ptr or shift) and not hygiene:
        raise ValueError("cross_view_gate / cross_view_ptr / cross_view_tpos_shift need "
                         "cross_view_hygiene=True (XW lineage only)")
    return gate, ptr, shift


def admits(out, threshold):
    """G's admission test for one stored neighbour output: eff_iou_score > threshold.

    This is frame_filter's score test (sam3_tracker_base.py:548, strict >) with two
    deliberate differences.  (i) No must-include: the temporal must-include (:554-555)
    keeps the OWN track continuous, and a neighbour provides no such continuity.  (ii)
    The no-key case is INVERTED: frame_filter SKIPS an entry that carries no
    "eff_iou_score" (:540-544), this returns True and ADMITS it, because for a neighbour
    "no key" means memory selection is off, or the entry is a seed cond entry reached
    under a t-1 mode -- neither is evidence against the entry.
    """
    score = out.get("eff_iou_score", None)
    return score is None or bool(score > threshold)


def gate_cross_view(s_pos_and_prevs, threshold, apply):
    """(entries, n_seen, n_fail).  n_seen = non-None entries, n_fail = those failing
    `admits`.  apply=True (G on) returns only the admitted entries; apply=False returns the
    input list unchanged (the counts are still taken, so a P-only run reports how many
    no-object pointers it injected)."""
    n_seen = n_fail = 0
    kept = []
    for s_pos, prev in s_pos_and_prevs:
        if prev is None:
            continue
        n_seen += 1
        if admits(prev, threshold):
            kept.append((s_pos, prev))
        else:
            n_fail += 1
    return (kept if apply else s_pos_and_prevs), n_seen, n_fail


def new_xview_stats():
    """Counters the tracker keeps while any knob is on; runMVSeg.xview_stats_summary reads
    them.  per_view[spatial_idx][frame_idx] = (n_seen, n_fail), summed over the per-object
    calls of that (session, frame)."""
    return {"calls": 0, "nb_seen": 0, "nb_fail": 0, "per_view": {}}


def record_xview(stats, spatial_idx, frame_idx, n_seen, n_fail):
    stats["calls"] += 1
    stats["nb_seen"] += n_seen
    stats["nb_fail"] += n_fail
    per = stats["per_view"].setdefault(spatial_idx, {})
    seen, fail = per.get(frame_idx, (0, 0))
    per[frame_idx] = (seen + n_seen, fail + n_fail)


def _offsets(mode, window):
    lower = list(range(-window, 0))            # -W..-1, farthest first (unchanged order)
    if mode in ("lower_t", "lower_tm1"):
        return lower
    return lower + list(range(1, window + 1))  # then +1..+W, nearest first (mirror)


def _frame_for(mode, s_pos, frame_idx, lag):
    """Frame of neighbour v+s_pos this variant reads (lag = +1 forward, -1 reverse)."""
    if mode in ("lower_t", "all_t"):
        return frame_idx
    if mode == "mixed":
        return frame_idx if s_pos < 0 else frame_idx - lag
    return frame_idx - lag                     # lower_tm1, both_tm1


def _lookup(prev_dict, f, mode):
    """A session's stored output at frame f.  A tracked frame lives in
    non_cond_frame_outputs; the seed lives in cond_frame_outputs (preflight pops a cond
    frame out of non_cond, SCSam3TrackerPredictorNewMem.py:761-765), so a t-1 == start
    read falls through to the cond dict.  all_t (E pass 2) reads the pass-1 output of
    frame t, which is what non_cond[t] holds until the Jacobi commit."""
    out = prev_dict["non_cond_frame_outputs"].get(f, None)
    if out is None and mode != "all_t":
        out = prev_dict["cond_frame_outputs"].get(f, None)
    return out


def gather_cross_view_memories(output_dicts, spatial_idx, frame_idx, window, hygiene,
                               max_cond_frames_in_attn, select_fn, mode="lower_t",
                               track_in_reverse=False):
    """Return (s_pos_and_prevs, rebound).

    s_pos_and_prevs: [(s_pos, out_or_None)] in loop order, consumed by the spatial
    token loop (row = abs(s_pos) - 1 for every entry).  rebound: UNCHANGED, or the dict
    the legacy code left bound to `selected_cond_outputs` (mode lower_t, hygiene off).
    """
    if mode not in GATHER_MODES:
        raise ValueError(f"mode={mode!r}: expected one of {GATHER_MODES}")
    if mode != "lower_t" and not hygiene:
        raise ValueError(f"mode={mode!r} requires hygiene=True")
    lag = -1 if track_in_reverse else 1
    n = len(output_dicts)
    s_pos_and_prevs = []
    rebound = UNCHANGED
    for s_pos in _offsets(mode, window):
        prev_spatial_idx = spatial_idx + s_pos
        if hygiene and prev_spatial_idx < 0:
            continue                                   # fix (2): no wrap, no IndexError
        if hygiene and prev_spatial_idx >= n:
            continue                                   # upper clip: positive offsets only
        if spatial_idx == prev_spatial_idx:            # never true (s_pos != 0); legacy guard kept
            continue
        prev_dict = output_dicts[prev_spatial_idx]     # legacy: wraps / raises here, same point
        if prev_dict is None:
            s_pos_and_prevs.append((s_pos, None))
            continue
        if mode == "lower_t":
            out = prev_dict["non_cond_frame_outputs"].get(frame_idx, None)
            if out is None and not hygiene:
                # Dead fallback (REPORT C2): `unselected` never contains frame_idx (proof in
                # SPEC §3), so `out` stays None; the rebinding is the published behaviour.
                rebound, unselected = select_fn(
                    frame_idx, prev_dict["cond_frame_outputs"], max_cond_frames_in_attn)
                out = unselected.get(frame_idx, None)
        else:
            out = _lookup(prev_dict, _frame_for(mode, s_pos, frame_idx, lag), mode)
        s_pos_and_prevs.append((s_pos, out))
    return s_pos_and_prevs, rebound
