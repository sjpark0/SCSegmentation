frame_idx = 0
tracker_state = {}
tracker_state["output_dict"] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
tracker_state["output_dict"]["cond_frame_outputs"][frame_idx] = True
print(len(tracker_state["output_dict"]["cond_frame_outputs"]))