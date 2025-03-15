import torch
from misc import *
from pose import *
from SCSam2Video import SCSam2Video
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

sc = SCSam2Video(device)
perms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        
#sc.LoadVideo_Folder("../../Data/VideoSample", perms)
sc.LoadVideo_File("../../Data/VideoSample_2", perms)
#sc.LoadVideo_Folder("D:\\CROOM\\T06_flag_ganggang\\JPG", perms)
#sc.LoadVideo_File("D:\\CROOM\\T06_flag_ganggang", perms)

sc.AddPoint(0, [840, 1356], 1, 1)
sc.AddPoint(0, [784, 1620], 1, 1)
sc.AddPoint(0, [3514, 999], 1, 2)
sc.AddPoint(0, [3390, 1603], 1, 2)


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