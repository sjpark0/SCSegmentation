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
"""

UNCHANGED = object()      # "leave the caller's selected_cond_outputs as it is"
LEGACY_WINDOW = 4
LEGACY_HYGIENE = False


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


def gather_cross_view_memories(output_dicts, spatial_idx, frame_idx, window, hygiene,
                               max_cond_frames_in_attn, select_fn):
    """Return (s_pos_and_prevs, rebound).

    s_pos_and_prevs: [(s_pos, out_or_None)] in loop order, consumed by the spatial
    token loop.  rebound: UNCHANGED, or the dict the legacy code left bound to
    `selected_cond_outputs` (the LAST neighbour that took the fallback).
    """
    s_pos_and_prevs = []
    rebound = UNCHANGED
    for s_pos in range(-window, 0):
        prev_spatial_idx = spatial_idx + s_pos
        if hygiene and prev_spatial_idx < 0:
            continue                                   # fix (2): no wrap, no IndexError
        if spatial_idx == prev_spatial_idx:            # never true (s_pos != 0); legacy guard kept
            continue
        prev_dict = output_dicts[prev_spatial_idx]     # legacy: wraps / raises here, same point
        if prev_dict is None:
            s_pos_and_prevs.append((s_pos, None))
            continue
        out = prev_dict["non_cond_frame_outputs"].get(frame_idx, None)
        if out is None and not hygiene:
            # Dead fallback (REPORT C2): `unselected` never contains frame_idx (proof in
            # SPEC §3), so `out` stays None; the rebinding is the published behaviour.
            rebound, unselected = select_fn(
                frame_idx, prev_dict["cond_frame_outputs"], max_cond_frames_in_attn)
            out = unselected.get(frame_idx, None)
        s_pos_and_prevs.append((s_pos, out))
    return s_pos_and_prevs, rebound
