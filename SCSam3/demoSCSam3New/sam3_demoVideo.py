import torch
from misc import *
from pose import *
from SCSam3Video import SCSam3Video
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"

sc = SCSam3Video(device)
#perms = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13, 14, 15]
perms = [0, 1]
         
#sc.LoadVideo_Folder("../../Data/VideoSample", perms)
sc.LoadVideo_File("../../Data/VideoSample_1", perms)
#sc.LoadVideo_Folder("D:\\CROOM\\T06_flag_ganggang\\JPG", perms)
#sc.LoadVideo_File("D:\\CROOM\\T06_flag_ganggang", perms)


sc.AddPoint(0, [2711, 1038], 1, 1)
sc.AddPoint(0, [2678, 1630], 1, 1)
sc.AddPoint(0, [1390, 1046], 1, 2)
sc.AddPoint(0, [1538, 1944], 1, 2)



sc.InitializeSegmentation()
#sc.RunNaiveTracking(0)

videos = []
for m in range(len(perms)):
    image, cpu_image = sc.inference_state[m]["images"][0]
    res_image = cpu_image
    
    for i, out_obj_id in enumerate(sc.obj_ids):
        out_mask = (sc.masks_spatial[m][out_obj_id].cpu().numpy() > 0.0)
        res_image = show_mask_cv(res_image, out_mask, obj_id=out_obj_id)
    
    cv2.imwrite(f"output_spatial_{m}.png", res_image)
'''
for m in range(len(perms)):
    os.makedirs(f"{m}", exist_ok=True)
    for frame_idx, out_obj_ids, _, out_mask_logits, _ in sc.tracking_result[m]:
        _, cpu_image = sc.inference_state[m]["images"][frame_idx]
        res_image = cpu_image
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            res_image = show_mask_cv(res_image, out_mask, obj_id=out_obj_id)
        cv2.imwrite(f"{m}/{frame_idx}.png", res_image)
'''