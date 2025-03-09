import torch
from misc import *
from pose import *
from SCSam2 import SCSam2
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
    
sc = SCSam2(device)
sc.LoadImage("../../Data/Sample1", 16)
sc.AddPoint(0, [840, 1356], 1)
sc.RunSegmentation()

for i in range(16):
    plt.imshow(sc.images[i])
    show_mask(sc.masks[i], plt.gca())
    show_points(np.array(sc.input_points[i]), np.array(sc.input_labels[i]), plt.gca())
    plt.axis('on')
    plt.show()