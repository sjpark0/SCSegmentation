import numpy as np
import cv2
import sys
import json

def ComputeIOU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0.0 else 0.0
'''
cam_list = [0, 6, 15]
num_objs = 27
start_frame = 40
num_frame = 21
folder1 = "../../Data/MVSeg/Painter/ConMask/"
folder2 = "../../Data/MVSeg/Painter/SegMask/"
folder3 = "../../Data/MVSeg/Painter/SegMaskNew/"

tmp = cv2.imread(folder1 + "v{:d}/{:d}/{:d}.png".format(cam_list[0], start_frame, 1), cv2.IMREAD_GRAYSCALE)
mask1 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
mask2 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
mask3 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))

iou = []
iouNew = []
for i in range(num_objs):
    for frame in range(num_frame):
        for cam in range(len(cam_list)):
            filename1 = folder1 + "v{:d}/{:d}/{:d}.png".format(cam_list[cam], frame + start_frame, i+1)
            filename2 = folder2 + "v{:d}/{:d}/{:d}.png".format(cam_list[cam], frame + start_frame, i+1)
            filename3 = folder3 + "v{:d}/{:d}/{:d}.png".format(cam_list[cam], frame + start_frame, i+1)
            mask1[cam, frame, :, :] = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
            mask2[cam, frame, :, :] = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
            mask3[cam, frame, :, :] = cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)
    #if i >= 1 and i <= 4:
    i1 = ComputeIOU(mask1, mask2)
    i2 = ComputeIOU(mask1, mask3)
    if i1 > 0.0 and i2 > 0.0:
        iou.append(ComputeIOU(mask1, mask2))
        iouNew.append(ComputeIOU(mask1, mask3))

print(iou)
print(iouNew)

print(np.mean(iou))
print(np.mean(iouNew))

start_frame = 0
cam_list = [5, 7, 9]
num_frame = 21
num_objs = 30
folder1 = "../../Data/MVSeg/Breakfast/ConMask/"
folder2 = "../../Data/MVSeg/Breakfast/SegMask/"
folder3 = "../../Data/MVSeg/Breakfast/SegMaskNew/"

tmp = cv2.imread(folder1 + "v{:d}/{:d}/{:d}.png".format(cam_list[0], start_frame, 1), cv2.IMREAD_GRAYSCALE)
mask1 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
mask2 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
mask3 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))

iou = []
iouNew = []
for i in range(num_objs):
    for frame in range(num_frame):
        for cam in range(len(cam_list)):
            filename1 = folder1 + "v{:d}/{:d}/{:d}.png".format(cam_list[cam], frame + start_frame, i+1)
            filename2 = folder2 + "v{:d}/{:d}/{:d}.png".format(cam_list[cam], frame + start_frame, i+1)
            filename3 = folder3 + "v{:d}/{:d}/{:d}.png".format(cam_list[cam], frame + start_frame, i+1)
            mask1[cam, frame, :, :] = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
            mask2[cam, frame, :, :] = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
            mask3[cam, frame, :, :] = cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)
    #if i >= 1 and i <= 4:
    i1 = ComputeIOU(mask1, mask2)
    i2 = ComputeIOU(mask1, mask3)
    if i1 > 0.0 and i2 > 0.0:
        iou.append(ComputeIOU(mask1, mask2))
        iouNew.append(ComputeIOU(mask1, mask3))


print(iou)
print(iouNew)

print(np.mean(iou))
print(np.mean(iouNew))

start_frame = 0
cam_list = [0, 4, 8]
num_frame = 21
num_objs = 22
folder1 = "../../Data/MVSeg/Carpark/ConMask/"
folder2 = "../../Data/MVSeg/Carpark/SegMask/"
folder3 = "../../Data/MVSeg/Carpark/SegMaskNew/"

tmp = cv2.imread(folder1 + "v{:d}/{:d}/{:d}.png".format(cam_list[0], start_frame, 1), cv2.IMREAD_GRAYSCALE)
mask1 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
mask2 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
mask3 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))

iou = []
iouNew = []
for i in range(num_objs):
    for frame in range(num_frame):
        for cam in range(len(cam_list)):
            filename1 = folder1 + "v{:d}/{:d}/{:d}.png".format(cam_list[cam], frame + start_frame, i+1)
            filename2 = folder2 + "v{:d}/{:d}/{:d}.png".format(cam_list[cam], frame + start_frame, i+1)
            filename3 = folder3 + "v{:d}/{:d}/{:d}.png".format(cam_list[cam], frame + start_frame, i+1)
            mask1[cam, frame, :, :] = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
            mask2[cam, frame, :, :] = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
            mask3[cam, frame, :, :] = cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)
    #if i >= 1 and i <= 4:
    i1 = ComputeIOU(mask1, mask2)
    i2 = ComputeIOU(mask1, mask3)
    if i1 > 0.0 and i2 > 0.0:
        iou.append(ComputeIOU(mask1, mask2))
        iouNew.append(ComputeIOU(mask1, mask3))


print(iou)
print(iouNew)

print(np.mean(iou))
print(np.mean(iouNew))
'''

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
    #folder4 = "../../Data/MVSeg/" + folder + "/SegMask1/"
    folder5 = "../../Data/MVSeg/" + folder + "/SegMaskNew3/"

    tmp = cv2.imread(folder1 + prefix + f"{cam_list[0]:0{prefix1}d}/{start_frame:d}/{1:d}.png", cv2.IMREAD_GRAYSCALE)
    mask1 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    #mask2 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    #mask3 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    #mask4 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))
    mask5 = np.zeros((len(cam_list), num_frame, tmp.shape[0], tmp.shape[1]))

    iou = []
    iouNew = []
    iou1 = []
    iouNew1 = []
    
    for i in range(num_objs):
        for frame in range(num_frame):
            for cam in range(len(cam_list)):
                filename1 = folder1 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
                #filename2 = folder2 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
                #filename3 = folder3 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
                #filename4 = folder4 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
                filename5 = folder5 + prefix + f"{cam_list[cam]:0{prefix1}d}/{frame+start_frame:d}/{i+1:d}.png"
         
                #mask1[cam, frame, :, :] = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
                t1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
                
                if t1 is not None:
                    mask1[cam, frame, :, :] = t1
                    #mask2[cam, frame, :, :] = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
                    #mask3[cam, frame, :, :] = cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)
                    #mask4[cam, frame, :, :] = cv2.imread(filename4, cv2.IMREAD_GRAYSCALE)
                    mask5[cam, frame, :, :] = cv2.imread(filename5, cv2.IMREAD_GRAYSCALE)
                
        #if i >= 1 and i <= 4:
        #i1 = ComputeIOU(mask1, mask2)
        #i2 = ComputeIOU(mask1, mask3)
        #i3 = ComputeIOU(mask1, mask4)
        i4 = ComputeIOU(mask1, mask5)
        #if i3 > 0.0 and i4 > 0.0:                 
        if i4 > 0.0:
            #iou.append(ComputeIOU(mask1, mask2))
            #iouNew.append(ComputeIOU(mask1, mask3))
            #iou1.append(i3)
            iouNew1.append(i4)
            
    #print(iou)
    #print(iou1)
    #print(iouNew)
    print(iouNew1)
    #print(np.mean(iou))
    #print(np.mean(iou1))
    #print(np.mean(iouNew))
    print(np.mean(iouNew1))