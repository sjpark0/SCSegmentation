# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict
from threading import Thread

import torch
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
import os
import numpy as np

from sam2.sam2_video_predictor import SAM2VideoPredictor

class SAM2VideoPredictorSpatial(SAM2VideoPredictor):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(self, *args, **kwargs):
        print("SAM2VideoPredictorSpatial")
        super().__init__(*args, **kwargs)

    @torch.inference_mode()
    def init_state(
        self,
        original_states,
        perms,
        start_frame = 0,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        numImage = len(perms)
        images = torch.zeros(numImage, 3, self.image_size, self.image_size, dtype=torch.float32)
        cpu_images = [None] * numImage
        for i in range(numImage):
            images[i], cpu_images[i] = original_states[i]["images"][start_frame]

        #images, video_height, video_width, cpu_images = 
        inference_state = {}
        inference_state["images"] = images
        inference_state["cpu_images"] = cpu_images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = original_states[0]["video_height"]
        inference_state["video_width"] = original_states[0]["video_width"]
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    