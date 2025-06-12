import torch
from misc import *
from pose import *
from SCSam2HQVideo import SCSam2HQVideo
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

sc = SCSam2HQVideo(device)
perms = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13, 14, 15]
        
sc.LoadVideo_Folder("../../Data/VideoSample", perms)
#sc.LoadVideo_File("../../Data/VideoSample_1", perms)
#sc.LoadVideo_Folder("D:\\CROOM\\T06_flag_ganggang\\JPG", perms)
#sc.LoadVideo_File("D:\\CROOM\\T06_flag_ganggang", perms)


sc.AddPoint(0, [2711, 1038], 1, 1)
sc.AddPoint(0, [2678, 1630], 1, 1)
sc.AddPoint(0, [1390, 1046], 1, 2)
sc.AddPoint(0, [1538, 1944], 1, 2)



sc.InitializeSegmentation()
sc.RunNaiveTracking(0)
index = 10

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
for frame_idx, out_obj_ids, out_mask_logits in sc.tracking_result[10]:
    _, cpu_image = sc.inference_state[10]["images"][frame_idx]
    res_image = cpu_image
    for i, out_obj_id in enumerate(out_obj_ids):
        out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        res_image = show_mask_cv(res_image, out_mask, obj_id=out_obj_id)
    cv2.imshow('image', res_image)    
    cv2.resizeWindow('image', 960, 540)
    cv2.waitKey(10)

cv2.destroyAllWindows()
