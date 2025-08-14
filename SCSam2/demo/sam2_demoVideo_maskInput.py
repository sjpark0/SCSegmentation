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
perms = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13, 14, 15]
perms1 = [1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 9, 10, 11, 12, 13, 14, 15, 16]      
num_objs = 6
        
sc.LoadVideo_Folder("../../Data/VideoSample2", perms)

for i in range(len(perms1)):
    filename = "../../Data/Mask_SAM/mask_{:03d}.png".format(perms[i]) 
    img = cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), dsize=(3840, 2160))
    sc.AddMask(i, 0, img, 1)

#    for j in range(num_objs):
#        filename = "../../Data/Mask1/{:02d}".format(perms1[i]) + "_p" + str(j+1) + ".png"
#        img = cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), dsize=(3840, 2160))
#        sc.AddMask(i, 0, img, j+1)
sc.RunTracking() 

for frame_idx, out_obj_ids, out_mask_logits in sc.tracking_result[10]:
    _, cpu_image = sc.inference_state[10]["images"][frame_idx]
    res_image = cpu_image
    for i, out_obj_id in enumerate(out_obj_ids):
        out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        res_image = show_mask_cv(res_image, out_mask, obj_id=out_obj_id)
    cv2.imwrite("../../Data/Result/" + str(frame_idx) + ".png", res_image)