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

sc = SCSam2VideoNew(device)
perms = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13, 14, 15]
perms1 = [1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 9, 10, 11, 12, 13, 14, 15, 16]      
sc.LoadVideo_Folder("../../Data/VideoSample2", perms)
#sc.LoadVideo_File("../../Data/VideoSample_1", perms)
#sc.LoadVideo_Folder("D:\\CROOM\\T06_flag_ganggang\\JPG", perms)
#sc.LoadVideo_File("D:\\CROOM\\T06_flag_ganggang", perms)



num_objs = 6

'''
sc.AddPoint(0, [2711, 1038], 1, 1)
sc.AddPoint(0, [2678, 1630], 1, 1)
sc.AddPoint(0, [1390, 1046], 1, 2)
sc.AddPoint(0, [1538, 1944], 1, 2)

sc.InitializeSegmentation()
for i in range(len(perms)):    
    for j in range(num_objs):
        cv2.imwrite("../../Data/Masks/" + str(perms[i]) + "_" + str(j+1) + ".png", sc.masks_spatial[i][j+1][0,...].astype(int) * 255)    
'''

'''
masks = []
for i in perms:
    mask = {}
    for j in range(num_objs):
        filename = "../../Data/Masks/" + str(i) + "_" + str(j+1) + ".png"
        mask[j+1] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    masks.append(mask)
sc.RunTrackingMaskInput(0, masks)
'''
'''
for i in range(len(perms)):
    for j in range(num_objs):
        filename = "../../Data/Mask1/" + str(perms[i]) + "_" + str(j+1) + ".png"
        sc.AddMask(i, 0, cv2.imread(filename, cv2.IMREAD_GRAYSCALE), j+1)
sc.RunTracking()    
index = 10
'''
for i in range(len(perms1)):
    filename = "../../Data/Mask_YOLO/mask_{:03d}.png".format(perms[i]) 
    img = cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), dsize=(3840, 2160))
    sc.AddMask(i, 0, img, 1)

#    for j in range(num_objs):
#        filename = "../../Data/Mask1/{:02d}".format(perms1[i]) + "_p" + str(j+1) + ".png"
#        img = cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), dsize=(3840, 2160))
#        sc.AddMask(i, 0, img, j+1)
sc.RunTracking()    

for idx in range(50):
    for spatial_idx in range(len(perms)):
        frame_idx, out_obj_ids, out_mask_logits = next(sc.tracking_result[spatial_idx])
        if spatial_idx == 10:
            _, cpu_image = sc.inference_state[spatial_idx]["images"][frame_idx]
            res_image = cpu_image
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                res_image = show_mask_cv(res_image, out_mask, obj_id=out_obj_id)
            cv2.imwrite("../../Data/Result3/" + str(frame_idx) + ".png", res_image)


'''
sc.RunNaiveTracking(0)
index = 10

for idx in range(50):
    for spatial_idx in range(len(perms)):
        frame_idx, out_obj_ids, out_mask_logits = next(sc.tracking_result[spatial_idx])
        if spatial_idx == 10:
            _, cpu_image = sc.inference_state[spatial_idx]["images"][frame_idx]
            res_image = cpu_image
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                res_image = show_mask_cv(res_image, out_mask, obj_id=out_obj_id)
            cv2.imwrite("../../Data/Result2/" + str(frame_idx) + ".png", res_image)
'''