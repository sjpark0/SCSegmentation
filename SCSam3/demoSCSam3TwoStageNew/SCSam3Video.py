from urllib import response

import torch
from build_scsam3 import build_scsam3_video_model_newmem, build_scsam3_video_predictor
from SCSam3VideoInference import SCSam3VideoInferenceWithInstanceInteractivity
from SCSam3TrackerPredictor import SCSam3TrackerPredictor

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU
from itertools import chain
class SCSam3Video:
    def __init__(self, device):
        gpus_to_use = range(torch.cuda.device_count())
        self.model = build_scsam3_video_model_newmem(device=device)
        self.predictor = self.model.tracker
        self.predictor.backbone = self.model.detector.backbone

        self.predictor_spatial = build_scsam3_video_predictor(gpus_to_use=gpus_to_use)
        self.predictor_spatial.model.fill_hole_area = 0
        
        self.masks = {}
        self.masks_spatial = {}
        self.obj_ids = None
        self.images = []
        self.cpu_images = []
        self.inference_state = []

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
            self.cpu_images.append(None)
        #self.inference_state_spatial = self.predictor_spatial.init_state(original_states = self.inference_state, offload_video_to_cpu = True, perms = perms)
        #self.predictor_spatial.reset_state(self.inference_state_spatial)
        
    def LoadVideo_Folder_MVSeg(self, folder, perms, start_frame, prefix, prefix1 = 0):
        self.folder = folder
        self.numImage = len(perms)        
        for i in range(self.numImage):
            foldername = self.folder + prefix + f"{perms[i]:0{prefix1}d}"
            state = self.predictor.init_state(video_path = foldername, offload_video_to_cpu = True, offload_state_to_cpu = True, async_loading_frames=True)
            self.predictor.reset_state(state)
            self.inference_state.append(state)
            self.cpu_images.append(None)
            self.images.append(None)    

        #self.inference_state_spatial = self.predictor_spatial.init_state(original_states = self.inference_state, offload_video_to_cpu = True, perms = perms, start_frame=start_frame)
        #self.predictor_spatial.reset_state(self.inference_state_spatial)

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
            cpu_image, video_height, video_width = load_video_frames(video_path=filename)
            image = AsyncVideoFrameCPUToGPU(cpu_image, offload_video_to_cpu=True)  
            
            state = self.predictor.init_state(images=image, video_height=video_height, video_width=video_width, num_frames=len(image), offload_video_to_cpu = True, offload_state_to_cpu = True)
            self.predictor.reset_state(state)
            self.inference_state.append(state)
            self.images.append(image)    
            self.cpu_images.append(cpu_image)

        self.video_width = video_width
        self.video_height = video_height

        #첫프레임 이미지 가져오기
        images = []        
        for i in range(self.numImage):
            images.append(self.images[i][0])

        response = self.predictor_spatial.handle_request(
                request=dict(
                    type="start_session",
                    images=images,
                    orig_height=self.video_height,
                    orig_width=self.video_width,
                )
            )
        self.session_id_statial = response["session_id"]
        _ = self.predictor_spatial.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=self.session_id_statial,
                )
            )

    def AddPoint(self, refCamID, point, label, obj_id):                
        if self.input_points.get(obj_id) is None:
            self.input_points[obj_id] = []
        if self.input_labels.get(obj_id) is None:
            self.input_labels[obj_id] = []
        
        point[0] = point[0] / self.video_width
        point[1] = point[1] / self.video_height
        
        self.input_points[obj_id].append(point)
        self.input_labels[obj_id].append(label)
        
        #points_tensor = torch.tensor(np.array(self.input_points[obj_id]), dtype=torch.float32)
        #points_labels_tensor = torch.tensor(np.array(self.input_labels[obj_id]), dtype=torch.int32)
        
        response = self.predictor_spatial.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self.session_id_statial,
                frame_index=refCamID,
                points=np.array(self.input_points[obj_id]),
                point_labels=np.array(self.input_labels[obj_id]),
                obj_id=obj_id,
            )
        )
        out = response["outputs"]
        self.obj_ids = out["out_obj_ids"].tolist()
    

    def AddText(self, refCamID, text):
        response = self.predictor_spatial.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self.session_id_statial,
                frame_index=refCamID,
                text=text,
            )
        )
        
        out = response["outputs"]
        self.obj_ids = out["out_obj_ids"].tolist()

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
        responses = self.predictor_spatial.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=self.session_id_statial,
                start_frame_idx=refCamID,
            )
        )

        for response in responses:
            out = response["outputs"]            
            self.masks_spatial[response["frame_index"]] = {
                out_obj_id: (out["out_binary_masks"][i] > 0.0)
                for i, out_obj_id in enumerate(out["out_obj_ids"].tolist())
            }        
        
    
    def RunTracking(self):
        self.tracking_result = [None] * self.numImage
        for m in range(self.numImage):
            self.tracking_result[m] = self.predictor.propagate_in_video(self.inference_state[m], start_frame_idx = 0, max_frame_num_to_track=240, propagate_preflight=True)
    
    def RunNaiveTracking(self, frame_idx, reverse = False):
        for m in range(self.numImage):            
           for i, obj_id in enumerate(self.obj_ids):
                if self.masks_spatial[m].get(obj_id) is not None:
                    _, out_obj_ids, _, masks = self.predictor.add_new_mask(
                        inference_state=self.inference_state[m],
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask = torch.tensor(self.masks_spatial[m][obj_id], dtype=torch.float32),
                    )
                else:
                    _, out_obj_ids, _, masks = self.predictor.add_new_mask(
                        inference_state=self.inference_state[m],
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask = torch.zeros((self.video_height, self.video_width), dtype=torch.float32),                    
                    )
           
        self.tracking_result = [None] * self.numImage
        
        if reverse:
            for m in range(self.numImage):
                self.tracking_result[m] = chain(self.predictor.propagate_in_video(self.inference_state, spatial_idx=m, start_frame_idx = frame_idx, max_frame_num_to_track=240, reverse = True, propagate_preflight=True), self.predictor.propagate_in_video(self.inference_state, spatial_idx=m, start_frame_idx = frame_idx, max_frame_num_to_track=240, reverse=False, propagate_preflight=True))
        else:
            for m in range(self.numImage):
                self.tracking_result[m] = self.predictor.propagate_in_video(self.inference_state, spatial_idx=m, start_frame_idx = frame_idx, max_frame_num_to_track=240, reverse=False, propagate_preflight=True)
        

        
        