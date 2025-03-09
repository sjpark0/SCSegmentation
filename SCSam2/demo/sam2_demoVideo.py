import torch
from misc import *
from pose import *
from SCSam2Video import SCSam2Video
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
sc.LoadVideo("../../Data/VideoSample1", 32)
sc.AddPoint(0, [2711, 1038], 1, 0, 1)
sc.AddPoint(0, [2678, 1630], 1, 0, 1)

sc.AddPoint(0, [1390, 1046], 1, 0, 2)
sc.AddPoint(0, [1538, 1944], 1, 0, 2)

sc.RunSegmentation(0)
for m in range(32):
    plt.imshow(sc.images[m])
    for i, out_obj_id in enumerate(sc.obj_ids):
        show_points(np.array(sc.input_points[out_obj_id][m]), np.array(sc.input_labels[out_obj_id][m]), plt.gca())
        show_mask(sc.masks[out_obj_id][m], plt.gca())

    plt.axis('on')
    plt.show()

