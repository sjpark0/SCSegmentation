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
    obj_list = data[name]["obj_list"]
    start_cam = data[name]["start_cam"]
    num_frame = data[name]["num_frame"]
    folder = "../../Data/MVSeg/" + data[name]["folder"]
    perms = data[name].get("perms", None)    
    if perms is None:        
        perms = list(range(start_cam, data[name]["num_cam"] + start_cam))        
    
    prefix = data[name]["prefix"]
    prefix1 = data[name]["prefix1"]   

    sc = SCSam2VideoNew(device)
    num_objs = data[name]["num_objs"]
            
    sc.LoadVideo_Folder_MVSeg(folder + "/Video/", perms, start_frame, prefix, prefix1)

    for i in range(len(perms)):
        for j in range(num_objs):
            filename = folder + "/Mask/{:02d}_p{:d}.png".format(perms[i], j+1)
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            sc.AddMask(i, 0, img, obj_list[j])

    sc.RunTracking() 

    for idx in range(50):
        if idx >= num_frame:
            break
        for spatial_idx in range(len(perms)):
            frame_idx, out_obj_ids, out_mask_logits = next(sc.tracking_result[spatial_idx])
            if perms[spatial_idx] in cam_list:    
                os.makedirs(folder + "/SegMaskNew_SA3D/" + prefix + f"{perms[spatial_idx]:0{prefix1}d}/{frame_idx:d}", exist_ok = True)
                res_mask = np.zeros(img.shape)
            
                _, cpu_image = sc.inference_state[spatial_idx]["images"][frame_idx]
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    res_mask = out_mask[0,...] * 255
                    cv2.imwrite(folder + "/SegMaskNew_SA3D/" + prefix + f"{perms[spatial_idx]:0{prefix1}d}/{frame_idx:d}/{out_obj_id:d}.png", res_mask)