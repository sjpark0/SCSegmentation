import torch
from build_scsam3 import build_scsam3_video_model, build_scsam3_video_model_spatial
from SCSam3VideoInference import SCSam3VideoInferenceWithInstanceInteractivity
from SCSam3VideoInferenceSpatial import SCSam3VideoInferenceWithInstanceInteractivitySpatial
from SCSam3TrackerPredictor import SCSam3TrackerPredictor
from SCSam3TrackerPredictorSpatial import SCSam3TrackerPredictorSpatial

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from misc import *
import os
from itertools import chain
class SCSam3Video:
    def __init__(self, device):
        self.model = build_scsam3_video_model(device=device)
        self.model_spatial = build_scsam3_video_model_spatial(device=device)

        self.predictor = self.model.tracker
        self.predictor.backbone = self.model.detector.backbone

        self.predictor_spatial = self.model_spatial.tracker
        self.predictor_spatial.backbone = self.model_spatial.detector.backbone

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

        self.inference_state_spatial = self.predictor_spatial.init_state(original_states = self.inference_state, offload_video_to_cpu = True)
        self.predictor_spatial.reset_state(self.inference_state_spatial)
        

    def AddPoint(self, refCamID, point, label, obj_id):                
        if self.input_points.get(obj_id) is None:
            self.input_points[obj_id] = []
        if self.input_labels.get(obj_id) is None:
            self.input_labels[obj_id] = []
        shape = self.inference_state_spatial["cpu_images"][0].shape
        point[0] = point[0] / shape[1]
        point[1] = point[1] / shape[0]
        self.input_points[obj_id].append(point)
        self.input_labels[obj_id].append(label)
        
        _, out_obj_ids, _, masks = self.predictor_spatial.add_new_points(
            inference_state=self.inference_state_spatial,
            frame_idx=refCamID,
            obj_id=obj_id,
            points=np.array(np.array(self.input_points[obj_id])),
            labels=np.array(np.array(self.input_labels[obj_id])),
            clear_old_points=False,
        )
        self.obj_ids = out_obj_ids
        
    def AddMaskSingle(self, refCamID, mask, obj_id):
        _, out_obj_ids, _, masks = self.predictor_spatial.add_new_mask(
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
        for imageID, out_obj_ids, _, out_mask_logits, _ in self.predictor_spatial.propagate_in_video(inference_state=self.inference_state_spatial, start_frame_idx=refCamID, max_frame_num_to_track=240, reverse=reverse, propagate_preflight=True):
            self.masks_spatial[imageID] = {
                out_obj_id: (out_mask_logits[i] > 0.0)
                for i, out_obj_id in enumerate(out_obj_ids)
            }     
    
    def RunTracking(self):
        self.tracking_result = [None] * self.numImage
        for m in range(self.numImage):
            self.tracking_result[m] = self.predictor.propagate_in_video(self.inference_state[m], start_frame_idx = 0, max_frame_num_to_track=240, propagate_preflight=True)
    
    def RunNaiveTracking(self, frame_idx, reverse = False):
        for m in range(self.numImage):            
           for i, obj_id in enumerate(self.obj_ids):
                _, out_obj_ids, _, masks = self.predictor.add_new_mask(
                    inference_state=self.inference_state[m],
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=self.masks_spatial[m][obj_id][0,...],
                )
        
        self.tracking_result = [None] * self.numImage
        
        if reverse:
            for m in range(self.numImage):
                self.tracking_result[m] = chain(self.predictor.propagate_in_video(self.inference_state[m], start_frame_idx = frame_idx, max_frame_num_to_track=240, reverse = True, propagate_preflight=True), self.predictor.propagate_in_video(self.inference_state[m], start_frame_idx = frame_idx, max_frame_num_to_track=240, reverse=False, propagate_preflight=True))
        else:
            for m in range(self.numImage):
                self.tracking_result[m] = self.predictor.propagate_in_video(self.inference_state[m], start_frame_idx = frame_idx, max_frame_num_to_track=240, reverse=False, propagate_preflight=True)
        
        
        