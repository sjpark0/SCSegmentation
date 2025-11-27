import torch
from misc import *
from pose import *
from SCSam2VideoNew import SCSam2VideoNew
#from SCSam2VideoNew import SCSam2VideoNew
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"


start_frame = 40
cam_list = [0, 6, 15]
num_frame = 21
folder = "../../Data/MVSeg/Painter/"
perms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]        

sc = SCSam2VideoNew(device)
sc.LoadVideo_Folder_MVSeg(folder + "Video", perms, start_frame)

filename = folder + "Mask/v{:d}/{:06d}.png".format(cam_list[0], start_frame)

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
obj_ids = np.max(img)
for i in range(obj_ids):
    mask = (img == (i+1)) * 255
    sc.AddMaskSingle(cam_list[0], mask, i+1)

if cam_list[0] != 0:
    sc.InitializeSegmentation(refCamID=cam_list[0], reverse=True)
sc.InitializeSegmentation(refCamID=cam_list[0])
sc.RunNaiveTracking(start_frame)

for idx in range(50):
    if idx >= num_frame:
        break
    for spatial_idx in range(len(perms)):
        frame_idx, out_obj_ids, out_mask_logits = next(sc.tracking_result[spatial_idx])
        #print(frame_idx, out_obj_ids)
        if spatial_idx in cam_list:    
            os.makedirs(folder + "SegMaskNew/v{:d}/{:d}".format(spatial_idx, frame_idx), exist_ok = True)
            res_mask = np.zeros(img.shape)
        
            _, cpu_image = sc.inference_state[spatial_idx]["images"][frame_idx]
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                res_mask = out_mask[0,...] * 255
                cv2.imwrite(folder + "SegMaskNew/v{:d}/{:d}/{:d}.png".format(spatial_idx, frame_idx, out_obj_id), res_mask)


start_frame = 0
cam_list = [5, 7, 9]
num_frame = 21
folder = "../../Data/MVSeg/Breakfast/"
perms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]        

sc = SCSam2VideoNew(device)
sc.LoadVideo_Folder_MVSeg(folder + "Video", perms, start_frame)

filename = folder + "Mask/v{:d}/{:06d}.png".format(cam_list[0], start_frame)

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
obj_ids = np.max(img)
for i in range(obj_ids):
    mask = (img == (i+1)) * 255
    sc.AddMaskSingle(cam_list[0], mask, i+1)

if cam_list[0] != 0:
    sc.InitializeSegmentation(refCamID=cam_list[0], reverse=True)
sc.InitializeSegmentation(refCamID=cam_list[0])
sc.RunNaiveTracking(start_frame)

for idx in range(50):
    if idx >= num_frame:
        break
    for spatial_idx in range(len(perms)):
        frame_idx, out_obj_ids, out_mask_logits = next(sc.tracking_result[spatial_idx])
        #print(frame_idx, out_obj_ids)
        if spatial_idx in cam_list:    
            os.makedirs(folder + "SegMaskNew/v{:d}/{:d}".format(spatial_idx, frame_idx), exist_ok = True)
            res_mask = np.zeros(img.shape)
        
            _, cpu_image = sc.inference_state[spatial_idx]["images"][frame_idx]
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                res_mask = out_mask[0,...] * 255
                cv2.imwrite(folder + "SegMaskNew/v{:d}/{:d}/{:d}.png".format(spatial_idx, frame_idx, out_obj_id), res_mask)
        

start_frame = 0
cam_list = [0, 4, 8]
num_frame = 21
folder = "../../Data/MVSeg/Carpark/"
perms = [0, 1, 2, 3, 4, 5, 6, 7, 8]        


sc = SCSam2VideoNew(device)
sc.LoadVideo_Folder_MVSeg(folder + "Video", perms, start_frame)

filename = folder + "Mask/v{:d}/{:06d}.png".format(cam_list[0], start_frame)

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
obj_ids = np.max(img)
for i in range(obj_ids):
    mask = (img == (i+1)) * 255
    sc.AddMaskSingle(cam_list[0], mask, i+1)

if cam_list[0] != 0:
    sc.InitializeSegmentation(refCamID=cam_list[0], reverse=True)
sc.InitializeSegmentation(refCamID=cam_list[0])
sc.RunNaiveTracking(start_frame)

for idx in range(50):
    if idx >= num_frame:
        break
    for spatial_idx in range(len(perms)):
        frame_idx, out_obj_ids, out_mask_logits = next(sc.tracking_result[spatial_idx])
        #print(frame_idx, out_obj_ids)
        if spatial_idx in cam_list:    
            os.makedirs(folder + "SegMaskNew/v{:d}/{:d}".format(spatial_idx, frame_idx), exist_ok = True)
            res_mask = np.zeros(img.shape)
        
            _, cpu_image = sc.inference_state[spatial_idx]["images"][frame_idx]
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                res_mask = out_mask[0,...] * 255
                cv2.imwrite(folder + "SegMaskNew/v{:d}/{:d}/{:d}.png".format(spatial_idx, frame_idx, out_obj_id), res_mask)
