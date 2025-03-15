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

        self.masks_spatial = {}
        self.obj_ids = None
        
    def LoadVideo_Folder(self, folder, numImage):    
        self.folder = folder
        self.numImage = numImage
        foldername = self.folder + "/Param/"
        poses, pts3d, self.perms, self.w2c, self.c2w = load_colmap_data(foldername)
        cdepth, idepth = computecloseInfinity(poses, pts3d, self.perms)
        self.close_depth = np.min(cdepth) * 0.9
        self.inf_depth = np.max(idepth) * 2.0
        self.focals = poses[2, 4, :]
        for i in range(self.numImage):
            foldername = self.folder + "/{:d}".format(i)
            state = self.predictor.init_state(video_path = foldername, offload_video_to_cpu = True, offload_state_to_cpu = True, async_loading_frames=True)
            self.predictor.reset_state(state)
            self.inference_state.append(state)
            self.input_points.append({})
            self.input_labels.append({})
            self.masks.append(None)
            self.images.append(None)    

    def LoadVideo_File(self, folder, numImage):
        frame_names = [
            p
            for p in os.listdir(folder)
            if os.path.splitext(p)[-1] in [".mp4", ".MP4", ".mov", ".MOV"]
        ]
        frame_names.sort()
        
        self.folder = folder
        self.numImage = numImage
        foldername = self.folder + "/Param/"
        poses, pts3d, self.perms, self.w2c, self.c2w = load_colmap_data(foldername)
        cdepth, idepth = computecloseInfinity(poses, pts3d, self.perms)
        self.close_depth = np.min(cdepth) * 0.9
        self.inf_depth = np.max(idepth) * 2.0
        self.focals = poses[2, 4, :]
        for i in range(self.numImage):
            filename = os.path.join(self.folder, frame_names[i])
            state = self.predictor.init_state(video_path = filename, offload_video_to_cpu = True, offload_state_to_cpu = True, async_loading_frames=True)            
            self.predictor.reset_state(state)
            self.inference_state.append(state)
            self.input_points.append({})
            self.input_labels.append({})
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
        for i, out_obj_id in enumerate(out_obj_ids):
            if out_obj_id == obj_id:
                h, w = masks[i].shape[-2:]
                mask_image = masks[i].reshape(h, w, 1)        
                break
        coords = np.where((mask_image > 0.0).cpu().numpy())
        boundingBox = int(np.min(coords[1])), int(np.min(coords[0])), int(np.max(coords[1])), int(np.max(coords[0]))
        for i in range(self.numImage):
            _, self.images[i] = self.inference_state[i]["images"][frame_idx]

        optZ, offsetX, offsetY = computeOffset(self.images, boundingBox, self.c2w, self.w2c, self.focals, 0, self.close_depth, self.inf_depth, self.perms)
        for i in range(self.numImage):
            if self.input_points[i].get(obj_id) is None:
                self.input_points[i][obj_id] = []
            if self.input_labels[i].get(obj_id) is None:
                self.input_labels[i][obj_id] = []

            self.input_points[i][obj_id].append([point[0] + offsetX[i], point[1] + offsetY[i]])
            self.input_labels[i][obj_id].append(label)
        
        self.obj_ids = out_obj_ids

    def InitializeSegmentation(self, frame_idx = 0):
        for i in range(self.numImage):
            for obj_id in self.obj_ids:
                _, out_obj_ids, masks = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state[i],
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=np.array(self.input_points[i][obj_id]),
                    labels=np.array(self.input_labels[i][obj_id]),
                )
                #self.masks_spatial[i] = (masks > 0.0).cpu().numpy()        
        self.tracking_result = [None] * self.numImage        
        for m in range(self.numImage):
            self.tracking_result[m] = self.predictor.propagate_in_video(self.inference_state[m])
        

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