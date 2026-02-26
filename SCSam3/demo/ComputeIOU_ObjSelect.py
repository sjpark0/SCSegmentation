import numpy as np
import cv2
import sys
import json

def ComputeIOU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0.0 else 0.0

if __name__ == "__main__":
    name = sys.argv[1]
    f = open("./MVSeg.json")
    data  = json.load(f)

    cam_list = data[name]["cam_list"]
    obj_list = data[name]["obj_list"]

    #num_objs = data[name]["num_objs"]
    num_objs = len(obj_list)
    start_frame = data[name]["start_frame"]
    num_frame = data[name]["num_frame"]
    folder = data[name]["folder"]
    prefix = data[name]["prefix"]
    prefix1 = data[name]["prefix1"]
    folder1 = "../../Data/MVSeg/" + folder + "/ConMask/"
    #folder2 = "../../Data/MVSeg/" + folder + "/SegMask_SA3D/"
    #folder3 = "../../Data/MVSeg/" + folder + "/SegMaskNew_SA3D/"
    #folder4 = "../../Data/MVSeg/" + folder + "/SegMask_SAM2/"
    #folder5 = "../../Data/MVSeg/" + folder + "/SegMaskNew_SAM2/"

    folder2 = "../../Data/MVSeg/" + folder + "/SegMask/"
    folder3 = "../../Data/MVSeg/" + folder + "/SegMaskNew/"
    folder4 = "../../Data/MVSeg/" + folder + "/SegMask1/"
    folder5 = "../../Data/MVSeg/" + folder + "/SegMaskNew1/"

    tmp = cv2.imread(folder1 + prefix + f"{cam_list[0]:0{prefix1}d}/{start_frame:d}/{1:d}.png", cv2.IMREAD_GRAYSCALE)
    mask1 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    mask2 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    mask3 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    mask4 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    mask5 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))

    iou = []
    iouNew = []
    iou1 = []
    iouNew1 = []
    
    for i in range(num_objs):
        for frame in range(num_frame):
            for cam in range(len(cam_list)):
                filename1 = folder1 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{obj_list[i]:d}.png"
                filename2 = folder2 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{obj_list[i]:d}.png"
                filename3 = folder3 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{obj_list[i]:d}.png"
                filename4 = folder4 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{obj_list[i]:d}.png"
                filename5 = folder5 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{obj_list[i]:d}.png"
         
                mask1[cam, frame, :, :] = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
                mask2[cam, frame, :, :] = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
                mask3[cam, frame, :, :] = cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)
                mask4[cam, frame, :, :] = cv2.imread(filename4, cv2.IMREAD_GRAYSCALE)
                mask5[cam, frame, :, :] = cv2.imread(filename5, cv2.IMREAD_GRAYSCALE)

        #if i >= 1 and i <= 4:
        i1 = ComputeIOU(mask1, mask2)
        i2 = ComputeIOU(mask1, mask3)
        i3 = ComputeIOU(mask1, mask4)
        i4 = ComputeIOU(mask1, mask5)
        if i1 > 0.0 and i2 > 0.0:
            iou.append(ComputeIOU(mask1, mask2))
            iouNew.append(ComputeIOU(mask1, mask3))
            iou1.append(ComputeIOU(mask1, mask4))
            iouNew1.append(ComputeIOU(mask1, mask5))
            
    print(iou)
    print(iou1)
    print(iouNew)
    print(iouNew1)
    print(np.mean(iou))
    print(np.mean(iou1))
    print(np.mean(iouNew))
    print(np.mean(iouNew1))