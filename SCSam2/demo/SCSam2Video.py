import torch
from build_sam import build_sam2_video_predictor
from SCSam2VideoPredictor import SAM2VideoPredictorCustom
from pose import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
    
class SCSam2Video:
    def __init__(self, device):
        sam_checkpoint = "../models/sam2.1_hiera_large.pt"
        model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam_checkpoint, device=device)
        self.input_points = []
        self.input_labels = []
        self.masks = []
        self.images = []
        self.inference_state = []

    def LoadVideo(self, folder, numImage):
        self.folder = folder
        foldername = self.folder + "/Param/"
        poses, pts3d, self.perms, self.w2c, self.c2w = load_colmap_data(foldername)
        cdepth, idepth = computecloseInfinity(poses, pts3d, self.perms)
        self.close_depth = np.min(cdepth) * 0.9
        self.inf_depth = np.max(idepth) * 2.0
        self.focals = poses[2, 4, :]
        self.numImage = numImage
        for i in range(numImage):
            foldername = self.folder + "/{:d}".format(i)
            state = self.predictor.init_state(video_path = foldername)
            self.predictor.reset_state(state)
            self.inference_state.append(state)
            self.input_points.append([])
            self.input_labels.append([])
            self.masks.append(None)
            self.images.append(None)    
    def AddPoint(self, refCamID, point, label, frame_idx, obj_id=1):
        _, out_obj_ids, masks = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state[refCamID],
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=np.array([point]),
            labels=np.array([label]),
        )
                
        h, w = masks.shape[-2:]
        mask_image = masks.reshape(h, w, 1)
        coords = np.where((mask_image > 0.0).cpu().numpy())
        boundingBox = int(np.min(coords[1])), int(np.max(coords[1])), int(np.min(coords[0])), int(np.max(coords[0]))
        for i in range(self.numImage):
            self.images[i] = self.inference_state[i]["cpu_images"][frame_idx]

        optZ, offsetX, offsetY = computeOffset(self.images, boundingBox, self.c2w, self.w2c, self.focals, 0, self.close_depth, self.inf_depth, self.perms)
        for i in range(self.numImage):
            self.input_points[i].append([point[0] + offsetX[i], point[1] + offsetY[i]])
            self.input_labels[i].append(label)
    
    def RunSegmentation(self, frame_idx):
        for i in range(self.numImage):
            _, out_obj_ids, masks = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state[i],
                frame_idx=frame_idx,
                obj_id=1,
                points=np.array(self.input_points[i]),
                labels=np.array(self.input_labels[i]),
            )
            self.masks[i] = (masks > 0.0).cpu().numpy()