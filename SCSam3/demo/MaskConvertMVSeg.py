import torch
from misc import *
from pose import *
from SCSam2Video import SCSam2Video
#from SCSam2VideoNew import SCSam2VideoNew
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json
import sys


if __name__ == "__main__":
    name = sys.argv[1]
    f = open("./MVSeg.json")
    data  = json.load(f)
    start_frame = data[name]["start_frame"]
    num_frame = data[name]["num_frame"]
    cam_list = data[name]["cam_list"]
    prefix = data[name]["prefix"]
    prefix1 = data[name]["prefix1"]    
    folder = "../../Data/MVSeg/" + data[name]["folder"]

    for cam in cam_list:
        for f in range(num_frame):
            filename = folder + "/Mask/" + prefix + f"{cam:0{prefix1}d}/{f+start_frame:06d}.png"
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            obj_ids = np.max(img)

            os.makedirs(folder + "/ConMask/" + prefix + f"{cam:0{prefix1}d}/{f+start_frame:d}", exist_ok = True)

            for i in range(obj_ids):
                mask = (img == (i+1)) * 255
                cv2.imwrite(folder + "/ConMask/" + prefix + f"{cam:0{prefix1}d}/{f+start_frame:d}/{i+1:d}.png", mask)
    