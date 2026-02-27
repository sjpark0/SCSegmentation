import torch
from build_scsam3 import build_scsam3_video_model
#import build_sam2_video_predictor, build_sam2_video_predictor_spatial
from SCSam3VideoPredictor import SCSam3VideoInferenceWithInstanceInteractivity

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from misc import *
import os
from itertools import chain
class SCSam3Video:
    def __init__(self, device):
        self.predictor = build_scsam3_video_model()
        self.predictor_spatial = build_scsam3_video_model_spatial()
        self.masks = {}
        self.masks_spatial = {}
        self.obj_ids = None
        self.images = []
        self.inference_state = []
        self.inference_state_spatial = None
        self.input_points = {}
        self.input_labels = {}

    def LoadVideo_Folder(self, folder, perms):
        self.folder = folder
        self.numImage = len(perms)        
        for i in range(self.numImage):
            foldername = self.folder + "/{:d}".format(perms[i])
            state = self.predictor.init_state(video_path = foldername, offload_video_to_cpu = True, offload_state_to_cpu = True, async_loading_frames=True)
            self.predictor.reset_state(state)
            self.inference_state.append(state)
            self.images.append(None)    

        self.inference_state_spatial = self.predictor_spatial.init_state(original_states = self.inference_state, offload_video_to_cpu = True, perms = perms)
        self.predictor_spatial.reset_state(self.inference_state_spatial)
        
    def LoadVideo_Folder_MVSeg(self, folder, perms, start_frame, prefix, prefix1 = 0):
        self.folder = folder
        self.numImage = len(perms)        
        for i in range(self.numImage):
            foldername = self.folder + prefix + f"{perms[i]:0{prefix1}d}"
            state = self.predictor.init_state(video_path = foldername, offload_video_to_cpu = True, offload_state_to_cpu = True, async_loading_frames=True)
            self.predictor.reset_state(state)
            self.inference_state.append(state)
            self.images.append(None)    

        self.inference_state_spatial = self.predictor_spatial.init_state(original_states = self.inference_state, offload_video_to_cpu = True, perms = perms, start_frame=start_frame)
        self.predictor_spatial.reset_state(self.inference_state_spatial)

    def LoadVideo_File(self, folder, perms):        
        frame_names = [
            p
            for p in os.listdir(folder)
            if os.path.splitext(p)[-1] in [".mp4", ".MP4", ".mov", ".MOV"]
        ]
        frame_names.sort()
        
        self.folder = folder
        self.numImage = len(perms)        
        for i in range(self.numImage):
            filename = os.path.join(self.folder, frame_names[perms[i]])
            state = self.predictor.init_state(video_path = filename, offload_video_to_cpu = True, offload_state_to_cpu = True, async_loading_frames=True)
            self.predictor.reset_state(state)
            self.inference_state.append(state)
            self.images.append(None)    

        self.inference_state_spatial = self.predictor_spatial.init_state(original_states = self.inference_state, offload_video_to_cpu = True, perms = perms)
        self.predictor_spatial.reset_state(self.inference_state_spatial)
        

    def AddPoint(self, refCamID, point, label, obj_id):                
        if self.input_points.get(obj_id) is None:
            self.input_points[obj_id] = []
        if self.input_labels.get(obj_id) is None:
            self.input_labels[obj_id] = []
            
        self.input_points[obj_id].append(point)
        self.input_labels[obj_id].append(label)

        _, out_obj_ids, masks = self.predictor_spatial.add_new_points_or_box(
            inference_state=self.inference_state_spatial,
            frame_idx=refCamID,
            obj_id=obj_id,
            points=np.array(np.array(self.input_points[obj_id])),
            labels=np.array(np.array(self.input_labels[obj_id])),
        )
        
        self.obj_ids = out_obj_ids

    def AddMaskSingle(self, refCamID, mask, obj_id):
        _, out_obj_ids, masks = self.predictor_spatial.add_new_mask(
                inference_state=self.inference_state_spatial,
                frame_idx=refCamID,
                obj_id=obj_id,
                mask=mask // 255,
            ) 
        self.obj_ids = out_obj_ids

    def AddMask(self, camID, frame_idx, mask, obj_id):
        self.predictor.add_new_mask(
                inference_state=self.inference_state[camID],
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask // 255,
            )
    def InitializeSegmentation(self, refCamID = 0, reverse = False):
        for imageID, out_obj_ids, out_mask_logits in self.predictor_spatial.propagate_in_video(self.inference_state_spatial, start_frame_idx = refCamID, reverse = reverse):
            self.masks_spatial[imageID] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }     
    
    def RunTracking(self):
        self.tracking_result = [None] * self.numImage
        for m in range(self.numImage):
            self.tracking_result[m] = self.predictor.propagate_in_video(self.inference_state[m])
    
    def RunNaiveTracking(self, frame_idx, reverse = False):
        for m in range(self.numImage):            
           for i, obj_id in enumerate(self.obj_ids):
                _, out_obj_ids, masks = self.predictor.add_new_mask(
                    inference_state=self.inference_state[m],
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=self.masks_spatial[m][obj_id][0,...],
                )
        
        self.tracking_result = [None] * self.numImage
        
        if reverse:
            for m in range(self.numImage):
                self.tracking_result[m] = chain(self.predictor.propagate_in_video(self.inference_state[m], start_frame_idx = frame_idx, reverse = True), self.predictor.propagate_in_video(self.inference_state[m], start_frame_idx = frame_idx))
        else:
            for m in range(self.numImage):
                self.tracking_result[m] = self.predictor.propagate_in_video(self.inference_state[m], start_frame_idx = frame_idx)
        
        
        