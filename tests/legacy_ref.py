"""Verbatim transliterations kept as the oracle for the pre-change code (SPEC.md section 4).

`select_closest_cond_frames` mirrors sam3/model/sam3_tracker_utils.py:271-329 so the
host (no sam3 install) can run the gather tests; `legacy_gather` is the loop that lived
at SCSam3TrackerPredictorNewMem.py:1262-1275 before Phase 2, including the rebinding of
`selected_cond_outputs` (REPORT.md C2).  Do not "improve" either function.
"""


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num, keep_first_cond_frame=False):
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2
        selected_outputs = {}
        if keep_first_cond_frame:
            idx_first = min((t for t in cond_frame_outputs if t < frame_idx), default=None)
            if idx_first is None:
                idx_first = max((t for t in cond_frame_outputs if t > frame_idx), default=None)
            if idx_first is not None:
                selected_outputs[idx_first] = cond_frame_outputs[idx_first]
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted((t for t in cond_frame_outputs if t not in selected_outputs),
                             key=lambda x: abs(x - frame_idx))[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}
    return selected_outputs, unselected_outputs


def legacy_gather(output_dicts, spatial_idx, frame_idx, selected_cond_outputs, max_cond_frames_in_attn=4,
                  select_fn=select_closest_cond_frames):
    """The pre-change loop.  `selected_cond_outputs` enters bound to the self-view dict
    (:1204) and may be rebound (:1273); the final binding is returned."""
    s_pos_and_prevs = []
    for s_pos in range(-4, 0):
        prev_spatial_idx = spatial_idx + s_pos
        if spatial_idx != prev_spatial_idx:
            if output_dicts[prev_spatial_idx] is None:
                s_pos_and_prevs.append((s_pos, None))
                continue
            out = output_dicts[prev_spatial_idx]["non_cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                selected_cond_outputs, unselected_cond_outputs1 = select_fn(
                    frame_idx, output_dicts[prev_spatial_idx]["cond_frame_outputs"], max_cond_frames_in_attn)
                out = unselected_cond_outputs1.get(frame_idx, None)
            s_pos_and_prevs.append((s_pos, out))
    return s_pos_and_prevs, selected_cond_outputs
