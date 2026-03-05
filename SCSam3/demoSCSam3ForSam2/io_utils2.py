# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import os
import numpy as np
from sam3.logger import get_logger
import cv2
import torch

logger = get_logger(__name__)

IS_MAIN_PROCESS = os.getenv("IS_MAIN_PROCESS", "1") == "1"
RANK = int(os.getenv("RANK", "0"))

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]



def _load_img_as_tensor(image, image_size):
    img_np = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (image_size, image_size))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    return img

def _load_img_as_tensor_file(image, image_size):
    img_np = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (image_size, image_size))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    return img


def load_video_frames(video_path):
        """
        Load the video frames from video_path. The frames are resized to image_size as in
        the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
        """
        is_bytes = isinstance(video_path, bytes)
        is_str = isinstance(video_path, str)
        is_mp4_path = is_str and os.path.splitext(video_path)[-1] in [".mp4", ".MP4", ".mov", ".MOV"]
        
        if is_bytes or is_mp4_path:
            return load_video_frames_from_video_file(video_path=video_path)
        elif is_str and os.path.isdir(video_path):
            return load_video_frames_from_jpg_images(video_path=video_path)
        else:
            raise NotImplementedError(
                "Only MP4 video and JPEG folder are supported at this moment"
            )


def load_video_frames_from_video_file(video_path):
    """Load the video frames from a video file."""
    lazy_images = AsyncVideoFrameLoaderFile(video_path)
    return lazy_images, lazy_images.video_height, lazy_images.video_width
  
def load_video_frames_from_jpg_images(video_path):
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
        lazy_images = AsyncVideoFrameLoader(img_paths)
        return lazy_images, lazy_images.video_height, lazy_images.video_width
        
class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.frame_idx = -1        
        # items in `self.images` will be loaded asynchronously
        self.cpu_image = None
        
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None
        
        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)        
        
    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        if self.frame_idx == index:
            return self.cpu_image
        
        self.frame_idx = index
        cpu_image = cv2.imread(self.img_paths[index])
        self.video_height = cpu_image.shape[0]
        self.video_width = cpu_image.shape[1]
        # normalize by mean and std
        self.cpu_image = cpu_image
        return cpu_image

    def __len__(self):
        return len(self.img_paths)
    

class AsyncVideoFrameLoaderFile:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(self, video_path):
        self.img_paths = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_idx = -1
        
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_height = None
        self.video_width = None
        
        self.__getitem__(0)
        
    def __getitem__(self, index):        
        if self.frame_idx != index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.frame_idx = index
        
        ret, cpu_image = self.cap.read()
        self.frame_idx += 1

        self.video_height = cpu_image.shape[0]
        self.video_width = cpu_image.shape[1]        
        return cpu_image
        
    def __len__(self):
        return self.num_frames

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
class AsyncVideoFrameCPUToGPU:
    def __init__(
        self, 
        origin,
        image_size=1008,
        offload_video_to_cpu=False,
        img_mean=(0.5, 0.5, 0.5),
        img_std=(0.5, 0.5, 0.5),        
    ):
        self.origin = origin
        self.frame_idx = -1
        # items in `self.images` will be loaded asynchronously
        self.image = None
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu

        self.img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        self.__getitem__(0)
        
    def __getitem__(self, index):
        
        if self.frame_idx == index:
            return self.image

        self.frame_idx = index        
        img = _load_img_as_tensor_file(self.origin[index], self.image_size)
        if not self.offload_video_to_cpu:
            img = img.cuda()
            self.img_mean = self.img_mean.cuda()
            self.img_std = self.img_mean.cuda()
        img -= self.img_mean
        img /= self.img_std
        self.image = img        

        return img

    def __len__(self):
        return len(self.origin)

