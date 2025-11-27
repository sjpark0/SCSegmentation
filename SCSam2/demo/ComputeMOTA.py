import numpy as np
import cv2
import sys
import json
from typing import Dict, List, Tuple

def ComputeIOU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()    
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0.0 else 0.0

def ComputeMOTA(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    num_mask1 = np.logical_and(mask1, 255).sum()
    num_mask2 = np.logical_and(mask2, 255).sum()
    fp = num_mask1 - intersection
    fn = num_mask2 - intersection    
    return fp, fn, num_mask1, num_mask2, intersection
'''
def ComputeMOTA_IDSwitch(mask1, mask2, prev1, prev2):
    intersection = np.logical_and(mask1, mask2).sum()
    num_mask1 = np.logical_and(mask1, 255).sum()
    num_mask2 = np.logical_and(mask2, 255).sum()
    fp = num_mask1 - intersection
    fn = num_mask2 - intersection    

    gt_temp = np.zeros((mask1.shape[1], mask1.shape[2]))
    gt_temp_prev = np.zeros((mask1.shape[1], mask1.shape[2]))
    pr_temp = np.zeros((mask1.shape[1], mask1.shape[2]))
    pr_temp_prev = np.zeros((mask1.shape[1], mask1.shape[2]))
    for i in range(mask1.shape[0]):
        gt_temp[mask1[i,:,:] > 0] = i
        gt_temp_prev[prev1[i, :,:] > 0] = i
        pr_temp[mask2[i,:,:] > 0] = i
        pr_temp_prev[prev2[i, :,:] > 0] = i

    same_gt_obj = (gt_temp > 0) & (gt_temp_prev > 0) & (gt_temp == gt_temp_prev)
    both_tracked = (pr_temp > 0) & (pr_temp_prev > 0)
    id_changed = (pr_temp != pr_temp_prev)

    idsw_mask = same_gt_obj & both_tracked & id_changed


    return fp, fn, num_mask1, num_mask2, int(idsw_mask.sum()), intersection
'''
def ComputeMOTA_IDSwitch(mask1, mask2, prev1, prev2):
    intersection = np.logical_and(mask1, mask2).sum()
    num_mask1 = np.logical_and(mask1, 255).sum()
    num_mask2 = np.logical_and(mask2, 255).sum()
    fp = num_mask1 - intersection
    fn = num_mask2 - intersection    

    gt_frame_intersection = np.logical_and(mask1, prev1)
    pr_frame_intersection = np.logical_and(mask2, prev2)
    pr_frame_union = np.logical_or(mask2, prev2)
    idsw = (pr_frame_union & ~pr_frame_intersection & ~gt_frame_intersection).sum()
    
    #pr_frame_intersection = np.logical_and(mask2, 255 - prev2)
    #idsw = np.logical_and(gt_frame_intersection, pr_frame_intersection).sum()
    
    #num_intersection = np.logical_and(gt_frame_intersection, pr_frame_intersection).sum()
    #num_gt_intersection = gt_frame_intersection.sum()
    #num_pr_intersection = pr_frame_intersection.sum()
    #idsw = num_gt_intersection - num_intersection + num_pr_intersection - num_intersection

    return fp, fn, num_mask1, num_mask2, idsw, intersection

if __name__ == "__main__":
    name = sys.argv[1]
    f = open("./MVSeg.json")
    data  = json.load(f)

    cam_list = data[name]["cam_list"]
    num_objs = data[name]["num_objs"]
    start_frame = data[name]["start_frame"]
    num_frame = data[name]["num_frame"]
    folder = data[name]["folder"]
    prefix = data[name]["prefix"]
    prefix1 = data[name]["prefix1"]
    folder1 = "../../Data/MVSeg/" + folder + "/ConMask/"
    folder2 = "../../Data/MVSeg/" + folder + "/SegMask1/"
    folder3 = "../../Data/MVSeg/" + folder + "/SegMaskNew1/"
    folder4 = "../../Data/MVSeg/" + folder + "/SegMaskNew2/"
    folder5 = "../../Data/MVSeg/" + folder + "/SegMaskNew3/"

    tmp = cv2.imread(folder1 + prefix + f"{cam_list[0]:0{prefix1}d}/{start_frame:d}/{1:d}.png", cv2.IMREAD_GRAYSCALE)
    mask1 = np.zeros((num_objs, tmp.shape[0], tmp.shape[1]))
    mask2 = np.zeros((num_objs, tmp.shape[0], tmp.shape[1]))
    mask3 = np.zeros((num_objs, tmp.shape[0], tmp.shape[1]))
    mask4 = np.zeros((num_objs, tmp.shape[0], tmp.shape[1]))
    mask5 = np.zeros((num_objs, tmp.shape[0], tmp.shape[1]))
       
    total_gt1 = 0
    total_fp1 = 0
    total_fn1 = 0
    total_idsw1 = 0

    total_gt2 = 0
    total_fp2 = 0
    total_fn2 = 0
    total_idsw2 = 0

    total_gt3 = 0
    total_fp3 = 0
    total_fn3 = 0
    total_idsw3 = 0

    total_gt4 = 0
    total_fp4 = 0
    total_fn4 = 0
    total_idsw4 = 0
    prev_mask1 = mask1
    prev_mask2 = mask2
    prev_mask3 = mask3
    prev_mask4 = mask4
    prev_mask5 = mask5
    
    for cam in range(len(cam_list)):        
        for frame in range(num_frame):
            for i in range(num_objs):
                filename1 = folder1 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
                filename2 = folder2 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
                filename3 = folder3 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
                filename4 = folder4 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
                filename5 = folder5 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
         
                t1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
                
                if t1 is not None:
                    mask1[i, :, :] = t1
                    mask2[i, :, :] = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
                    mask3[i, :, :] = cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)
                    mask4[i, :, :] = cv2.imread(filename4, cv2.IMREAD_GRAYSCALE)
                    mask5[i, :, :] = cv2.imread(filename5, cv2.IMREAD_GRAYSCALE)

            if frame == 0:
                fp1, fn1, gt1, pred1, intersection1 = ComputeMOTA(mask1, mask2)
                fp2, fn2, gt2, pred2, intersection2 = ComputeMOTA(mask1, mask3)
                fp3, fn3, gt3, pred3, intersection3 = ComputeMOTA(mask1, mask4)
                fp4, fn4, gt4, pred4, intersection4 = ComputeMOTA(mask1, mask5)
                
                total_fp1 = total_fp1 + fp1
                total_fn1 = total_fn1 + fn1
                total_gt1 = total_gt1 + gt1
                
                total_fp2 = total_fp2 + fp2
                total_fn2 = total_fn2 + fn2
                total_gt2 = total_gt2 + gt2
                
                total_fp3 = total_fp3 + fp3
                total_fn3 = total_fn3 + fn3
                total_gt3 = total_gt3 + gt3
                
                total_fp4 = total_fp4 + fp4
                total_fn4 = total_fn4 + fn4
                total_gt4 = total_gt4 + gt4

                prev_mask1 = mask1.copy()
                prev_mask2 = mask2.copy()
                prev_mask3 = mask3.copy()
                prev_mask4 = mask4.copy()
                prev_mask5 = mask5.copy()
            else:
                fp1, fn1, gt1, pred1, idsw1, intersection1 = ComputeMOTA_IDSwitch(mask1, mask2, prev_mask1, prev_mask2)
                fp2, fn2, gt2, pred2, idsw2, intersection2 = ComputeMOTA_IDSwitch(mask1, mask3, prev_mask1, prev_mask3)
                fp3, fn3, gt3, pred3, idsw3, intersection3 = ComputeMOTA_IDSwitch(mask1, mask4, prev_mask1, prev_mask4)
                fp4, fn4, gt4, pred4, idsw4, intersection4 = ComputeMOTA_IDSwitch(mask1, mask5, prev_mask1, prev_mask5)
                
                total_fp1 = total_fp1 + fp1
                total_fn1 = total_fn1 + fn1
                total_gt1 = total_gt1 + gt1
                total_idsw1 = total_idsw1 + idsw1
                
                total_fp2 = total_fp2 + fp2
                total_fn2 = total_fn2 + fn2
                total_gt2 = total_gt2 + gt2
                total_idsw2 = total_idsw2 + idsw2
                
                total_fp3 = total_fp3 + fp3
                total_fn3 = total_fn3 + fn3
                total_gt3 = total_gt3 + gt3
                total_idsw3 = total_idsw3 + idsw3
                
                total_fp4 = total_fp4 + fp4
                total_fn4 = total_fn4 + fn4
                total_gt4 = total_gt4 + gt4
                total_idsw4 = total_idsw4 + idsw4
                
                prev_mask1 = mask1.copy()
                prev_mask2 = mask2.copy()
                prev_mask3 = mask3.copy()
                prev_mask4 = mask4.copy()
                prev_mask5 = mask5.copy()

                #print(idsw1, idsw2, idsw3, idsw4)

        #print(cam, 1.0 - (total_fp1 + total_fn1 + total_idsw1) / float(total_gt1))

    mota1 = 1.0 - (total_fp1 + total_fn1 + total_idsw1) / float(total_gt1)
    mota2 = 1.0 - (total_fp2 + total_fn2 + total_idsw2) / float(total_gt2)
    mota3 = 1.0 - (total_fp3 + total_fn3 + total_idsw3) / float(total_gt3)
    mota4 = 1.0 - (total_fp4 + total_fn4 + total_idsw4) / float(total_gt4)
    
    print(mota1)
    print(mota2)
    print(mota3)
    print(mota4)
    