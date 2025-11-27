import torch
from misc import *
from pose import *
from SCSam2VideoNew import SCSam2VideoNew
#from SCSam2VideoNew import SCSam2VideoNew
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    name = sys.argv[1]
    f = open("./MVSeg.json")
    data  = json.load(f)

    start_frame = data[name]["start_frame"]
    cam_list = data[name]["cam_list"]
    start_cam = data[name]["start_cam"]
    num_frame = data[name]["num_frame"]
    folder = "../../Data/MVSeg/" + data[name]["folder"]
    perms = data[name].get("perms", None)    
    if perms is None:        
        perms = list(range(start_cam, data[name]["num_cam"] + start_cam))        
    
    prefix = data[name]["prefix"]
    prefix1 = data[name]["prefix1"]

    sc = SCSam2VideoNew(device)
    sc.LoadVideo_Folder_MVSeg(folder + "/Video/", perms, start_frame, prefix, prefix1)

    obj_ids = 0
    max_mask = -1
    for cam in cam_list:
        filename = folder + "/Mask/" + prefix + f"{cam:0{prefix1}d}/{start_frame:06d}.png"
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        temp = np.max(img)
        if temp > obj_ids:
            obj_ids = temp
            max_mask = cam
    
    filename = folder + "/Mask/" + prefix + f"{max_mask:0{prefix1}d}/{start_frame:06d}.png"    
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    obj_ids = np.max(img)

    id = perms.index(max_mask)
    
    for i in range(obj_ids):
        mask = (img == (i+1)) * 255
        sc.AddMaskSingle(id, mask, i+1)

    if id != 0:
        sc.InitializeSegmentation(refCamID=id, reverse=True)
    sc.InitializeSegmentation(refCamID=id)
    sc.RunNaiveTracking(start_frame)

    for idx in range(50):
        if idx >= num_frame:
            break
        for spatial_idx in range(len(perms)):
            frame_idx, out_obj_ids, out_mask_logits = next(sc.tracking_result[spatial_idx])
            if perms[spatial_idx] in cam_list:    
                os.makedirs(folder + "/SegMaskNew3/" + prefix + f"{perms[spatial_idx]:0{prefix1}d}/{frame_idx:d}", exist_ok = True)
                res_mask = np.zeros(img.shape)
            
                _, cpu_image = sc.inference_state[spatial_idx]["images"][frame_idx]
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    res_mask = out_mask[0,...] * 255
                    cv2.imwrite(folder + "/SegMaskNew3/" + prefix + f"{perms[spatial_idx]:0{prefix1}d}/{frame_idx:d}/{out_obj_id:d}.png", res_mask)
