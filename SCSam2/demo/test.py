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
    num_objs = data[name]["num_objs"]
    start_frame = data[name]["start_frame"]
    num_frame = data[name]["num_frame"]
    folder = data[name]["folder"]
    prefix = data[name]["prefix"]
    prefix1 = data[name]["prefix1"]
    folder1 = "../../Data/MVSeg/" + folder + "/ConMask/"
    #folder2 = "../../Data/MVSeg/" + folder + "/SegMask/"
    #folder3 = "../../Data/MVSeg/" + folder + "/SegMaskNew/"
    folder4 = "../../Data/MVSeg/" + folder + "/SegMask1/"
    folder5 = "../../Data/MVSeg/" + folder + "/SegMaskNew1/"

    tmp = cv2.imread(folder1 + prefix + f"{cam_list[0]:0{prefix1}d}/{start_frame:d}/{1:d}.png", cv2.IMREAD_GRAYSCALE)
    mask1 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    #mask2 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    #mask3 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    mask4 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    mask5 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))

    iou = []
    iouNew = []
    iou1 = []
    iouNew1 = []
    
    #for i in range(num_objs):
    i = 21
    for frame in range(num_frame):
        for cam in range(len(cam_list)):                
            filename1 = folder1 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
            #filename2 = folder2 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
            #filename3 = folder3 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
            filename4 = folder4 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
            filename5 = folder5 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"

            t1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)


            #if t1 is not None:
            mask1[cam, frame, :, :] = t1
            #mask2[cam, frame, :, :] = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
            #mask3[cam, frame, :, :] = cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)
            mask4[cam, frame, :, :] = cv2.imread(filename4, cv2.IMREAD_GRAYSCALE)
            mask5[cam, frame, :, :] = cv2.imread(filename5, cv2.IMREAD_GRAYSCALE)
        

    intersection = np.logical_and(mask1, mask4).sum()
    union = np.logical_or(mask1, mask4).sum()  
    print(intersection, union)

    intersection = np.logical_and(mask1, mask5).sum()
    union = np.logical_or(mask1, mask5).sum()  
    print(intersection, union)
    
    