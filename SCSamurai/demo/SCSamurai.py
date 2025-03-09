import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pose import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
    
class SCSamurai:
    def __init__(self, device):
        sam_checkpoint = "../models/sam2.1_hiera_large.pt"
        model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam_checkpoint, device=device))
        self.input_points = []
        self.input_labels = []
        self.images = []
        self.masks = []

    def LoadImage(self, folder, numImage):
        self.folder = folder
        self.numImage = numImage
        
        poses, pts3d, self.perms, self.w2c, self.c2w = load_colmap_data(self.folder)
        cdepth, idepth = computecloseInfinity(poses, pts3d, self.perms)
        self.close_depth = np.min(cdepth) * 0.9
        self.inf_depth = np.max(idepth) * 2.0
        self.focals = poses[2, 4, :]


        for i in range(numImage):
            filename = self.folder + "/images/{:03d}.png".format(i)
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.input_points.append([])
            self.input_labels.append([])
            self.masks.append(None)

    def AddPoint(self, refCamID, point, label):
        self.predictor.set_image(self.images[refCamID])
        masks, scores, logits = self.predictor.predict(point_coords=np.array([point]), point_labels=np.array([label]), multimask_output=False,)
                
        print(masks.shape)
        h, w = masks.shape[-2:]
        mask_image = masks.reshape(h, w, 1)
        coords = np.where(mask_image > 0.0)
        #boundingBox = int(np.min(coords[1])), int(np.max(coords[1])), int(np.min(coords[0])), int(np.max(coords[0]))
        boundingBox = int(np.min(coords[1])), int(np.min(coords[0])), int(np.max(coords[1])), int(np.max(coords[0]))
        optZ, offsetX, offsetY = computeOffset(self.images, boundingBox, self.c2w, self.w2c, self.focals, 0, self.close_depth, self.inf_depth, self.perms)
        for i in range(self.numImage):
            self.input_points[i].append([point[0] + offsetX[i], point[1] + offsetY[i]])
            self.input_labels[i].append(label)
    
    def RunSegmentation(self):
        for i in range(self.numImage):
            self.predictor.set_image(self.images[i])
            masks, scores, logits = self.predictor.predict(point_coords=np.array(self.input_points[i]), point_labels=np.array(self.input_labels[i]), multimask_output=False,)
            self.masks[i] = masks