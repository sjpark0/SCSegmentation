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
import cv2

from sam2.sam2_video_predictor import SAM2VideoPredictor


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


class SAM2VideoPredictorCustom(SAM2VideoPredictor):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(self, *args, **kwargs):
        print("SAM2VideoPredictorCustom")
        super().__init__(*args, **kwargs)

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            img, _ = inference_state["images"][frame_idx]
            image = img.to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        inference_state = {}
        
        inference_state["images"] = images
        #inference_state["cpu_images"] = cpu_images
        inference_state["num_frames"] = len(inference_state["images"])
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
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
