import torch
from misc import *
from pose import *
from SCSam2 import SCSam2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


from build_sam import build_sam2_video_predictor
#from VideoPredictorCustom import build_sam2_video_predictor_custom


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"
   

sam_checkpoint = "../models/sam2.1_hiera_large.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam_checkpoint, device=device)

video_dir = "../../Data/VideoSample/0"

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
image = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(image)
plt.show()

inference_state = predictor.init_state(video_path = video_dir)

predictor.reset_state(inference_state)

prompts = {}

ann_frame_idx = 0

ann_obj_id = 1
points = np.array([[2711, 1038], [2678, 1630]], dtype=np.float32)
labels = np.array([1, 1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

ann_obj_id = 2
points = np.array([[1390, 1046], [1538, 1944]], dtype=np.float32)
labels = np.array([1, 1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(image)

for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

plt.show()

video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

vis_frame_stride = 10
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    
    image = cv2.imread(os.path.join(video_dir, frame_names[out_frame_idx]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.show()