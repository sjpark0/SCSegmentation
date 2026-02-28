# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sam3 import perflib
from sam3.logger import get_logger
from sam3.model.act_ckpt_utils import clone_output_wrapper
from sam3.model.box_ops import box_xywh_to_cxcywh, box_xyxy_to_xywh
from sam3.model.data_misc import BatchedDatapoint, convert_my_tensors, FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model.io_utils import IMAGE_EXTS, load_resource_as_video_frames
from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3.model.sam3_video_base import MaskletConfirmationStatus, Sam3VideoBase
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.utils.misc import copy_data_to_device
from sam3.perflib.compile import compile_wrapper, shape_logging_wrapper
from sam3.perflib.masks_ops import masks_to_boxes as perf_masks_to_boxes
from torchvision.ops import masks_to_boxes
from tqdm.auto import tqdm
import cv2
import os


logger = get_logger(__name__)

def _load_img_as_tensor(image, image_size):
    img_np = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (image_size, image_size))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_height, video_width, _ = image.shape  # the original video size
    return img, video_height, video_width, np.array(image)

def _load_img_as_tensor_file(image, image_size):
    img_np = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (image_size, image_size))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_height, video_width, _ = image.shape  # the original video size
    return img, video_height, video_width, np.array(image)


def load_video_frames(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,
        compute_device=torch.device("cuda"),
    ):
        """
        Load the video frames from video_path. The frames are resized to image_size as in
        the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
        """
        is_bytes = isinstance(video_path, bytes)
        is_str = isinstance(video_path, str)
        is_mp4_path = is_str and os.path.splitext(video_path)[-1] in [".mp4", ".MP4", ".mov", ".MOV"]
        
        if is_bytes or is_mp4_path:
            return load_video_frames_from_video_file(
                video_path=video_path,
                image_size=image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                img_mean=img_mean,
                img_std=img_std,
                compute_device=compute_device,
            )
        elif is_str and os.path.isdir(video_path):
            return load_video_frames_from_jpg_images(
                video_path=video_path,
                image_size=image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                img_mean=img_mean,
                img_std=img_std,
                async_loading_frames=async_loading_frames,
                compute_device=compute_device,
            )
        else:
            raise NotImplementedError(
                "Only MP4 video and JPEG folder are supported at this moment"
            )


def load_video_frames_from_video_file(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
):
    """Load the video frames from a video file."""
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    lazy_images = AsyncVideoFrameLoaderFile(
                video_path,
                image_size,
                offload_video_to_cpu,
                img_mean,
                img_std,
                compute_device,
    )
    return lazy_images, lazy_images.video_height, lazy_images.video_width
  
def load_video_frames_from_jpg_images(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,
        compute_device=torch.device("cuda"),
    ):
        """
        Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

        The frames are resized to image_size x image_size and are loaded to GPU if
        `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

        You can load a frame asynchronously by setting `async_loading_frames` to `True`.
        """
        if isinstance(video_path, str) and os.path.isdir(video_path):
            jpg_folder = video_path
        else:
            raise NotImplementedError(
                "Only JPEG frames are supported at this moment. For video files, you may use "
                "ffmpeg (https://ffmpeg.org/) to extract frames into a folder of JPEG files, such as \n"
                "```\n"
                "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\n"
                "```\n"
                "where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks "
                "ffmpeg to start the JPEG file from 00000.jpg."
            )

        frame_names = [
            p
            for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"no images found in {jpg_folder}")
        img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        if async_loading_frames:
            lazy_images = AsyncVideoFrameLoader(
                img_paths,
                image_size,
                offload_video_to_cpu,
                img_mean,
                img_std,
                compute_device,
            )
            return lazy_images, lazy_images.video_height, lazy_images.video_width#, lazy_images.cpu_images

        images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
        cpu_images = [None] * num_frames
        res_img = [None] * num_frames

        for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
            images[n], video_height, video_width, cpu_images[n] = _load_img_as_tensor(cv2.imread(img_path), image_size)
            res_img[n] = images[n], cpu_images[n]

        if not offload_video_to_cpu:
            images = images.to(compute_device)
            img_mean = img_mean.to(compute_device)
            img_std = img_std.to(compute_device)
        # normalize by mean and std
        images -= img_mean
        images /= img_std
        return res_img, video_height, video_width

class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.frame_idx = -1        
        # items in `self.images` will be loaded asynchronously
        self.image = None
        self.cpu_image = None
        
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)        
        
    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        if self.frame_idx == index:
            return self.image, self.cpu_image
        
        self.frame_idx = index
        img, video_height, video_width, cpu_image = _load_img_as_tensor(
            cv2.imread(self.img_paths[index]), self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        self.image = img
        self.cpu_image = cpu_image
        return img, cpu_image

    def __len__(self):
        return len(self.img_paths)
    

class AsyncVideoFrameLoaderFile:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(
        self,
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        self.img_paths = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_idx = -1
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self.images` will be loaded asynchronously
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.image = None
        self.cpu_image = None

        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)
        
    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        if self.frame_idx == index:
            return self.image, self.cpu_image
        
        self.frame_idx = index
        ret, capture_img = self.cap.read()
        img, video_height, video_width, cpu_image = _load_img_as_tensor_file(
            capture_img, self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        
        self.image = img
        self.cpu_image = cpu_image

        return img, cpu_image

    def __len__(self):
        return self.num_frames

class SCSam3VideoInferenceWithInstanceInteractivity(Sam3VideoInferenceWithInstanceInteractivity):
    def __init__(
        self,
        use_prev_mem_frame=False,
        use_stateless_refinement=False,
        refinement_detector_cond_frame_removal_window=16,
        **kwargs,
    ):
        """
        use_prev_mem_frame: bool, whether to condition on previous memory frames for adding points
        use_stateless_refinement: bool, whether to enable stateless refinement behavior
        refinement_detector_cond_frame_removal_window: int, we remove a detector conditioning frame if it
            is within this many frames of a user refined frame. Set to a large value (e.g. 10000) to
            always remove detector conditioning frames if there is any user refinement in the video.
        """
        super().__init__(**kwargs)
        self.use_prev_mem_frame = use_prev_mem_frame
        self.use_stateless_refinement = use_stateless_refinement
        self.refinement_detector_cond_frame_removal_window = (
            refinement_detector_cond_frame_removal_window
        )
    
    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        video_loader_type="cv2",
    ):
        """Initialize an inference state from `resource_path` (an image or a video)."""
        images, orig_height, orig_width = load_video_frames(
            video_path=resource_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        
        inference_state = {}
        inference_state["images"] = images
        inference_state["image_size"] = self.image_size
        inference_state["num_frames"] = len(images)
        # the original video height and width, used for resizing final output scores
        inference_state["orig_height"] = orig_height
        inference_state["orig_width"] = orig_width
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # inputs on each frame
        self._construct_initial_input_batch(inference_state, images)
        # initialize extra states
        inference_state["tracker_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["action_history"] = []  # for logging user actions
        inference_state["is_image_only"] = is_image_type(resource_path)
        return inference_state
'''
    def _init_new_tracker_state(self, inference_state):
        return self.tracker.init_state(
            cached_features=inference_state["feature_cache"],
            video_height=inference_state["orig_height"],
            video_width=inference_state["orig_width"],
            num_frames=inference_state["num_frames"],
        )

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        # step 1: check which type of propagation to run, should be the same for all GPUs.
        propagation_type, obj_ids = self.parse_action_history_for_propagation(
            inference_state
        )
        self.add_action_history(
            inference_state,
            action_type=propagation_type,
            obj_ids=obj_ids,
            frame_idx=start_frame_idx,
        )

        # step 2: run full VG propagation
        if propagation_type == "propagation_full":
            logger.debug(f"Running full VG propagation (reverse={reverse}).")
            yield from super().propagate_in_video(
                inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
            )
            return

        # step 3: run Tracker partial propagation or direct fetch existing predictions
        assert propagation_type in ["propagation_partial", "propagation_fetch"]
        logger.debug(
            f"Running Tracker propagation for objects {obj_ids} and merging it with existing VG predictions (reverse={reverse})."
            if propagation_type == "propagation_partial"
            else f"Fetching existing VG predictions without running any propagation (reverse={reverse})."
        )
        processing_order, _ = self._get_processing_order(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )

        tracker_metadata = inference_state["tracker_metadata"]

        # if fetch just return from output
        if propagation_type == "propagation_fetch":
            for frame_idx in tqdm(processing_order):
                if self.rank == 0:
                    obj_id_to_mask = inference_state["cached_frame_outputs"].get(
                        frame_idx, {}
                    )
                    # post processing - remove suppressed obj_ids
                    obj_id_to_score = tracker_metadata["obj_id_to_score"]
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    obj_id_to_tracker_score = tracker_metadata[
                        "obj_id_to_tracker_score_frame_wise"
                    ][frame_idx]

                    out = {
                        "obj_id_to_mask": obj_id_to_mask,
                        "obj_id_to_score": obj_id_to_score,
                        "obj_id_to_tracker_score": obj_id_to_tracker_score,
                    }
                    yield (
                        frame_idx,
                        self._postprocess_output(
                            inference_state, out, suppressed_obj_ids=suppressed_obj_ids
                        ),
                    )
                else:
                    yield frame_idx, None

            return

        # get Tracker inference states containing selected obj_ids
        if propagation_type == "propagation_partial":
            # can be empty for GPUs where objects are not in their inference states
            tracker_states_local = self._get_tracker_inference_states_by_obj_ids(
                inference_state, obj_ids
            )
            for tracker_state in tracker_states_local:
                self.tracker.propagate_in_video_preflight(
                    tracker_state, run_mem_encoder=True
                )

        for frame_idx in tqdm(processing_order):
            # run Tracker propagation
            if propagation_type == "propagation_partial":
                self._prepare_backbone_feats(inference_state, frame_idx, reverse)
                obj_ids_local, low_res_masks_local, tracker_scores_local = (
                    self._propogate_tracker_one_frame_local_gpu(
                        tracker_states_local,
                        frame_idx=frame_idx,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                )

                # broadcast refined object tracker scores and masks to all GPUs
                # handle multiple objects that can be located on different GPUs
                refined_obj_data = {}  # obj_id -> (score, mask_video_res)

                # Collect data for objects on this GPU
                local_obj_data = {}
                for obj_id in obj_ids:
                    obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
                    if self.rank == obj_rank and obj_id in obj_ids_local:
                        refined_obj_idx = obj_ids_local.index(obj_id)
                        refined_mask_low_res = low_res_masks_local[
                            refined_obj_idx
                        ]  # (H_low_res, W_low_res)
                        refined_score = tracker_scores_local[refined_obj_idx]

                        # Keep low resolution for broadcasting to reduce communication cost
                        local_obj_data[obj_id] = (refined_score, refined_mask_low_res)

                # Broadcast data from each GPU that has refined objects
                if self.world_size > 1:
                    for obj_id in obj_ids:
                        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
                        if self.rank == obj_rank:
                            # This GPU has the object, broadcast its data
                            data_to_broadcast = local_obj_data.get(obj_id, None)
                            data_list = [
                                (data_to_broadcast[0].cpu(), data_to_broadcast[1].cpu())
                            ]
                            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
                            if data_to_broadcast is not None:
                                refined_obj_data[obj_id] = data_to_broadcast
                        elif self.rank != obj_rank:
                            # This GPU doesn't have the object, receive data
                            data_list = [None]
                            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
                            refined_obj_data[obj_id] = (
                                data_list[0][0].to(self.device),
                                data_list[0][1].to(self.device),
                            )
                else:
                    # Single GPU case
                    refined_obj_data = local_obj_data

                # Update Tracker scores for all refined objects
                for obj_id, (refined_score, _) in refined_obj_data.items():
                    tracker_metadata["obj_id_to_tracker_score_frame_wise"][
                        frame_idx
                    ].update({obj_id: refined_score.item()})

                if self.rank == 0:
                    # get predictions from Tracker inference states, it includes the original
                    # VG predictions and the refined predictions from interactivity.

                    # Prepare refined masks dictionary - upscale to video resolution after broadcast
                    refined_obj_id_to_mask = {}
                    for obj_id, (_, refined_mask_low_res) in refined_obj_data.items():
                        refined_mask_video_res = (
                            self._convert_low_res_mask_to_video_res(
                                refined_mask_low_res, inference_state
                            )
                        )  # (1, H_video, W_video) bool
                        refined_obj_id_to_mask[obj_id] = refined_mask_video_res

                    obj_id_to_mask = self._build_tracker_output(
                        inference_state, frame_idx, refined_obj_id_to_mask
                    )
                    out = {
                        "obj_id_to_mask": obj_id_to_mask,
                        "obj_id_to_score": tracker_metadata["obj_id_to_score"],
                        "obj_id_to_tracker_score": tracker_metadata[
                            "obj_id_to_tracker_score_frame_wise"
                        ][frame_idx],
                    }
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    self._cache_frame_outputs(
                        inference_state,
                        frame_idx,
                        obj_id_to_mask,
                        suppressed_obj_ids=suppressed_obj_ids,
                    )
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    yield (
                        frame_idx,
                        self._postprocess_output(
                            inference_state, out, suppressed_obj_ids=suppressed_obj_ids
                        ),
                    )
                else:
                    yield frame_idx, None

    def add_action_history(
        self, inference_state, action_type, frame_idx=None, obj_ids=None
    ):
        """
        action_history is used to automatically decide what to do during propagation.
        action_type: one of ["add", "remove", "refine"] + ["propagation_full", "propagation_partial", "propagation_fetch"]
        """
        instance_actions = ["add", "remove", "refine"]
        propagation_actions = [
            "propagation_full",
            "propagation_partial",
            "propagation_fetch",
        ]
        assert action_type in instance_actions + propagation_actions, (
            f"Invalid action type: {action_type}, must be one of {instance_actions + propagation_actions}"
        )
        action = {
            "type": action_type,
            "frame_idx": frame_idx,
            "obj_ids": obj_ids,
        }
        inference_state["action_history"].append(action)

    def _has_object_been_refined(self, inference_state, obj_id):
        action_history = inference_state["action_history"]
        for action in action_history:
            if action["type"] in ["add", "refine"] and action.get("obj_ids"):
                if obj_id in action["obj_ids"]:
                    return True
        return False

    def parse_action_history_for_propagation(self, inference_state):
        """
        Parse the actions in history before the last propagation and prepare for the next propagation.
        We support multiple actions (add/remove/refine) between two propagations. If we had an action
        history similar to this ["propagate", "add", "refine", "remove", "add"], the next propagation
        would remove the removed object, and also propagate the two added/refined objects.

        Returns:
            propagation_type: one of ["propagation_full", "propagation_partial", "propagation_fetch"]
                - "propagation_full": run VG propagation for all objects
                - "propagation_partial": run Tracker propagation for selected objects, useful for add/refine actions
                - "propagation_fetch": fetch existing VG predictions without running any propagation
            obj_ids: list of object ids to run Tracker propagation on if propagation_type is "propagation_partial".
        """
        action_history = inference_state["action_history"]
        if len(action_history) == 0:
            # we run propagation for the first time
            return "propagation_full", None

        if "propagation" in action_history[-1]["type"]:
            if action_history[-1]["type"] in ["propagation_fetch"]:
                # last propagation is direct fetch, we fetch existing predictions
                return "propagation_fetch", None
            elif action_history[-1]["type"] in [
                "propagation_partial",
                "propagation_full",
            ]:
                # we do fetch prediction if we have already run propagation twice or we have run
                # propagation once and it is from the first frame or last frame.
                if (
                    len(action_history) > 1
                    and action_history[-2]["type"]
                    in ["propagation_partial", "propagation_full"]
                ) or action_history[-1]["frame_idx"] in [
                    0,
                    inference_state["num_frames"] - 1,
                ]:
                    # we have run both forward and backward partial/full propagation
                    return "propagation_fetch", None
                else:
                    # we have run partial/full forward or backward propagation once, need run it for the rest of the frames
                    return action_history[-1]["type"], action_history[-1]["obj_ids"]

        # parse actions since last propagation
        obj_ids = []
        for action in action_history[::-1]:
            if "propagation" in action["type"]:
                # we reached the last propagation action, stop parsing
                break
            if action["type"] in ["add", "refine"]:
                obj_ids.extend(action["obj_ids"])
            # else action["type"] == "remove": noop
        obj_ids = list(set(obj_ids)) if len(obj_ids) > 0 else None
        propagation_type = (
            "propagation_partial" if obj_ids is not None else "propagation_fetch"
        )
        return propagation_type, obj_ids

    def remove_object(self, inference_state, obj_id, is_user_action=False):
        """
        We try to remove object from tracker states on every GPU, it will do nothing
        for states without this object.
        """
        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
        assert obj_rank is not None, f"Object {obj_id} not found in any GPU."

        tracker_states_local = inference_state["tracker_inference_states"]
        if self.rank == obj_rank:
            self._tracker_remove_object(tracker_states_local, obj_id)

        if is_user_action:
            self.add_action_history(
                inference_state, action_type="remove", obj_ids=[obj_id]
            )

        # update metadata
        tracker_metadata = inference_state["tracker_metadata"]
        _obj_ids = tracker_metadata["obj_ids_per_gpu"][obj_rank]
        tracker_metadata["obj_ids_per_gpu"][obj_rank] = _obj_ids[_obj_ids != obj_id]
        tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
            tracker_metadata["obj_ids_per_gpu"][obj_rank]
        )
        tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
            tracker_metadata["obj_ids_per_gpu"]
        )
        tracker_metadata["obj_id_to_score"].pop(obj_id, None)
        # tracker_metadata["max_obj_id"] # we do not reuse the object id, so we do not update it here

        # Clean up cached frame outputs to remove references to the deleted object
        if "cached_frame_outputs" in inference_state:
            for frame_idx in inference_state["cached_frame_outputs"]:
                frame_cache = inference_state["cached_frame_outputs"][frame_idx]
                if obj_id in frame_cache:
                    del frame_cache[obj_id]

    def _get_gpu_id_by_obj_id(self, inference_state, obj_id):
        """
        Locate GPU ID for a given object.
        """
        obj_ids_per_gpu = inference_state["tracker_metadata"]["obj_ids_per_gpu"]
        for rank, obj_ids in enumerate(obj_ids_per_gpu):
            if obj_id in obj_ids:
                return rank
        return None  # object not found in any GPU

    def _get_tracker_inference_states_by_obj_ids(self, inference_state, obj_ids):
        """
        Get the Tracker inference states that contain the given object ids.
        This is used to run partial Tracker propagation on a single object/bucket.
        Possibly multiple or zero states can be returned.
        """
        states = [
            state
            for state in inference_state["tracker_inference_states"]
            if set(obj_ids) & set(state["obj_ids"])
        ]
        return states

    def _prepare_backbone_feats(self, inference_state, frame_idx, reverse):
        input_batch = inference_state["input_batch"]
        feature_cache = inference_state["feature_cache"]
        num_frames = inference_state["num_frames"]
        geometric_prompt = (
            inference_state["constants"]["empty_geometric_prompt"]
            if inference_state["per_frame_geometric_prompt"][frame_idx] is None
            else inference_state["per_frame_geometric_prompt"][frame_idx]
        )
        _ = self.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=num_frames,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
            reverse=reverse,
            allow_new_detections=True,
        )

    @torch.inference_mode()
    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        boxes_xywh=None,
        box_labels=None,
        points=None,
        point_labels=None,
        obj_id=None,
        rel_coordinates=True,
    ):
        if points is not None:
            # Tracker instance prompts
            assert text_str is None and boxes_xywh is None, (
                "When points are provided, text_str and boxes_xywh must be None."
            )
            assert obj_id is not None, (
                "When points are provided, obj_id must be provided."
            )
            return self.add_tracker_new_points(
                inference_state,
                frame_idx,
                obj_id=obj_id,
                points=points,
                labels=point_labels,
                rel_coordinates=rel_coordinates,
                use_prev_mem_frame=self.use_prev_mem_frame,
            )
        else:
            # SAM3 prompts
            return super().add_prompt(
                inference_state,
                frame_idx,
                text_str=text_str,
                boxes_xywh=boxes_xywh,
                box_labels=box_labels,
            )

    @torch.inference_mode()
    def add_tracker_new_points(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points,
        labels,
        rel_coordinates=True,
        use_prev_mem_frame=False,
    ):
        """Add a new point prompt to Tracker. Suppporting instance refinement to existing
        objects by passing existing obj_id or adding a new object by passing a new obj_id.
        use_prev_mem_frame=False to disable cross attention to previous memory frames.
        Every GPU returns the same results, and results should contain all masks including
        these masks not refined or not added by the current user points.
        """
        assert obj_id is not None, "obj_id must be provided to add new points"
        tracker_metadata = inference_state["tracker_metadata"]
        if tracker_metadata == {}:
            # initialize masklet metadata if it's uninitialized (empty dict)
            tracker_metadata.update(self._initialize_metadata())

        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)

        # prepare feature
        self._prepare_backbone_feats(inference_state, frame_idx, reverse=False)

        object_has_been_refined = self._has_object_been_refined(inference_state, obj_id)
        if (
            obj_rank is not None
            and self.use_stateless_refinement
            and not object_has_been_refined
        ):
            # The first time we start refinement on the object, we remove it.
            logger.debug(
                f"[rank={self.rank}] Removing object {obj_id} before refinement."
            )
            self.remove_object(inference_state, obj_id, is_user_action=False)
            obj_rank = None

        if obj_rank is None:
            # new object, we assign it a GPU and create a new inference state if limit allows
            num_prev_obj = np.sum(tracker_metadata["num_obj_per_gpu"])
            if num_prev_obj >= self.max_num_objects:
                logger.warning(
                    f"add_tracker_new_points: cannot add a new object as we are already tracking {num_prev_obj=} "
                    f"masklets (under {self.max_num_objects=})"
                )
                obj_ids = []
                H_low_res = W_low_res = self.tracker.low_res_mask_size
                H_video_res = inference_state["orig_height"]
                W_video_res = inference_state["orig_width"]
                low_res_masks = torch.zeros(0, 1, H_low_res, W_low_res)
                video_res_masks = torch.zeros(0, 1, H_video_res, W_video_res)
                return frame_idx, obj_ids, low_res_masks, video_res_masks

            new_det_gpu_ids = self._assign_new_det_to_gpus(
                new_det_num=1,
                prev_workload_per_gpu=tracker_metadata["num_obj_per_gpu"],
            )
            obj_rank = new_det_gpu_ids[0]

            # get tracker inference state for the new object
            if self.rank == obj_rank:
                # for batched inference, we create a new inference state
                tracker_state = self._init_new_tracker_state(inference_state)
                inference_state["tracker_inference_states"].append(tracker_state)

            # update metadata
            tracker_metadata["obj_ids_per_gpu"][obj_rank] = np.concatenate(
                [
                    tracker_metadata["obj_ids_per_gpu"][obj_rank],
                    np.array([obj_id], dtype=np.int64),
                ]
            )
            tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
                tracker_metadata["obj_ids_per_gpu"][obj_rank]
            )
            tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
                tracker_metadata["obj_ids_per_gpu"]
            )
            tracker_metadata["max_obj_id"] = max(tracker_metadata["max_obj_id"], obj_id)

            logger.debug(
                f"[rank={self.rank}] Adding new object with id {obj_id} at frame {frame_idx}."
            )
            self.add_action_history(
                inference_state, "add", frame_idx=frame_idx, obj_ids=[obj_id]
            )
        else:
            # existing object, for refinement
            if self.rank == obj_rank:
                tracker_states = self._get_tracker_inference_states_by_obj_ids(
                    inference_state, [obj_id]
                )
                assert len(tracker_states) == 1, (
                    f"[rank={self.rank}] Multiple Tracker inference states found for the same object id."
                )
                tracker_state = tracker_states[0]

            # log
            logger.debug(
                f"[rank={self.rank}] Refining existing object with id {obj_id} at frame {frame_idx}."
            )
            self.add_action_history(
                inference_state, "refine", frame_idx=frame_idx, obj_ids=[obj_id]
            )

        # assign higher score to added/refined object
        tracker_metadata["obj_id_to_score"][obj_id] = 1.0
        tracker_metadata["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = 1.0

        if self.rank == 0:
            rank0_metadata = tracker_metadata.get("rank0_metadata", {})

            if "removed_obj_ids" in rank0_metadata:
                rank0_metadata["removed_obj_ids"].discard(obj_id)

            if "suppressed_obj_ids" in rank0_metadata:
                for frame_id in rank0_metadata["suppressed_obj_ids"]:
                    rank0_metadata["suppressed_obj_ids"][frame_id].discard(obj_id)

            if "masklet_confirmation" in rank0_metadata:
                obj_ids_all_gpu = tracker_metadata["obj_ids_all_gpu"]
                obj_indices = np.where(obj_ids_all_gpu == obj_id)[0]
                if len(obj_indices) > 0:
                    obj_idx = obj_indices[0]
                    if obj_idx < len(rank0_metadata["masklet_confirmation"]["status"]):
                        rank0_metadata["masklet_confirmation"]["status"][obj_idx] = 1
                        rank0_metadata["masklet_confirmation"]["consecutive_det_num"][
                            obj_idx
                        ] = self.masklet_confirmation_consecutive_det_thresh

        if self.rank == obj_rank:
            frame_idx, obj_ids, low_res_masks, video_res_masks = (
                self.tracker.add_new_points(
                    inference_state=tracker_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    clear_old_points=True,
                    rel_coordinates=rel_coordinates,
                    use_prev_mem_frame=use_prev_mem_frame,
                )
            )

            if video_res_masks is not None and len(video_res_masks) > 0:
                video_res_masks = fill_holes_in_mask_scores(
                    video_res_masks,  # shape (N, 1, H_video, W_video)
                    max_area=self.fill_hole_area,
                    fill_holes=True,
                    remove_sprinkles=True,
                )

            # Since the mem encoder has already run for the current input points?
            self.tracker.propagate_in_video_preflight(
                tracker_state, run_mem_encoder=True
            )
            # Clear detector conditioning frames when user clicks are received to allow
            # model updating masks on these frames. It is a noop if user is refining on the
            # detector conditioning frames or adding new objects.
            self.clear_detector_added_cond_frame_in_tracker(
                tracker_state, obj_id, frame_idx
            )

        # fetch results from states and gather across GPUs
        # Use optimized caching approach to avoid reprocessing unmodified objects
        if self.rank == obj_rank and len(obj_ids) > 0:
            new_mask_data = (video_res_masks[obj_ids.index(obj_id)] > 0.0).to(
                torch.bool
            )
        else:
            new_mask_data = None
        # Broadcast the new mask data across all ranks for consistency
        if self.world_size > 1:
            data_list = [new_mask_data.cpu() if new_mask_data is not None else None]
            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
            new_mask_data = data_list[0].to(self.device)

        if self.rank == 0:
            obj_id_to_mask = self._build_tracker_output(
                inference_state,
                frame_idx,
                {obj_id: new_mask_data} if new_mask_data is not None else None,
            )
            # post processing - remove suppressed obj_ids
            obj_id_to_score = tracker_metadata["obj_id_to_score"]
            suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                "suppressed_obj_ids"
            ][frame_idx]
            obj_id_to_tracker_score = tracker_metadata[
                "obj_id_to_tracker_score_frame_wise"
            ][frame_idx]

            out = {
                "obj_id_to_mask": obj_id_to_mask,
                "obj_id_to_score": obj_id_to_score,
                "obj_id_to_tracker_score": obj_id_to_tracker_score,
            }
            self._cache_frame_outputs(
                inference_state,
                frame_idx,
                obj_id_to_mask,
                suppressed_obj_ids=suppressed_obj_ids,
            )
            return frame_idx, self._postprocess_output(
                inference_state, out, suppressed_obj_ids=suppressed_obj_ids
            )
        else:
            return frame_idx, None  # no output on other GPUs

    def _gather_obj_id_to_mask_across_gpus(self, inference_state, obj_id_to_mask_local):
        """Gather obj_id_to_mask from all GPUs. Optionally resize the masks to the video resolution."""
        tracker_metadata = inference_state["tracker_metadata"]

        # concatenate the output masklets from all local inference states
        H_mask = W_mask = self.tracker.low_res_mask_size
        obj_ids_local = tracker_metadata["obj_ids_per_gpu"][self.rank]
        low_res_masks_local = []
        for obj_id in obj_ids_local:
            if obj_id in obj_id_to_mask_local:
                low_res_masks_local.append(obj_id_to_mask_local[obj_id])
            else:
                low_res_masks_local.append(
                    torch.full((H_mask, W_mask), -1024.0, device=self.device)
                )
        if len(low_res_masks_local) > 0:
            low_res_masks_local = torch.stack(low_res_masks_local, dim=0)  # (N, H, W)
            assert low_res_masks_local.shape[1:] == (H_mask, W_mask)
        else:
            low_res_masks_local = torch.zeros(0, H_mask, W_mask, device=self.device)

        # all-gather `low_res_masks_local` into `low_res_masks_global`
        # - low_res_masks_global: Tensor -- (num_global_obj, H_mask, W_mask)
        if self.world_size > 1:
            low_res_masks_local = low_res_masks_local.float().contiguous()
            low_res_masks_peers = [
                low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                for num_obj in tracker_metadata["num_obj_per_gpu"]
            ]
            dist.all_gather(low_res_masks_peers, low_res_masks_local)
            low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
        else:
            low_res_masks_global = low_res_masks_local
        return low_res_masks_global

    def _convert_low_res_mask_to_video_res(self, low_res_mask, inference_state):
        """
        Convert a low-res mask to video resolution, matching the format expected by _build_tracker_output.

        Args:
            low_res_mask: Tensor of shape (H_low_res, W_low_res)
            inference_state: Contains video dimensions

        Returns:
            video_res_mask: Tensor of shape (1, H_video, W_video) bool
        """
        if low_res_mask is None:
            return None

        # Convert to 3D for interpolation: (H_low_res, W_low_res) -> (1, H_low_res, W_low_res)
        low_res_mask_3d = low_res_mask.unsqueeze(0).unsqueeze(0)

        # Get video dimensions
        H_video = inference_state["orig_height"]
        W_video = inference_state["orig_width"]

        video_res_mask = F.interpolate(
            low_res_mask_3d.float(),
            size=(H_video, W_video),
            mode="bilinear",
            align_corners=False,
        )  # (1, H_video, W_video)

        # Convert to boolean - already in the right shape!
        return (video_res_mask.squeeze(0) > 0.0).to(torch.bool)

    def clear_detector_added_cond_frame_in_tracker(
        self, tracker_state, obj_id, refined_frame_idx
    ):
        """Clear detector added conditioning frame if it is within a predefined window
        of the refined frame. This allow model to update masks on these frames."""
        obj_idx = self.tracker._obj_id_to_idx(tracker_state, obj_id)

        mask_only_cond_frame_indices = []
        window = self.refinement_detector_cond_frame_removal_window
        for frame_idx in tracker_state["mask_inputs_per_obj"][obj_idx]:
            if frame_idx not in tracker_state["point_inputs_per_obj"][obj_idx]:
                # clear conditioning frames within a window of the refined frame
                if abs(frame_idx - refined_frame_idx) <= window:
                    mask_only_cond_frame_indices.append(frame_idx)

        # clear
        if len(mask_only_cond_frame_indices) > 0:
            for frame_idx in mask_only_cond_frame_indices:
                # obj_ids_on_this_frame is essentially all obj_ids in the state
                # since they are bucket batched
                obj_ids_on_this_frame = tracker_state["obj_id_to_idx"].keys()
                for obj_id2 in obj_ids_on_this_frame:
                    self.tracker.clear_all_points_in_frame(
                        tracker_state, frame_idx, obj_id2, need_output=False
                    )
            logger.debug(
                f"Cleared detector mask only conditioning frames ({mask_only_cond_frame_indices}) in Tracker."
            )
        return
'''

def is_image_type(resource_path: str) -> bool:
    if isinstance(resource_path, list):
        return len(resource_path) == 1
    return resource_path.lower().endswith(tuple(IMAGE_EXTS))
