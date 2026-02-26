import cv2
import os

input_folder = "../../Data/VideoSample"
output_folder = "../../Data/VideoSample_1"

os.makedirs(output_folder, exist_ok=True)

for i in range(32):
    foldername = input_folder + "/{:d}".format(i)
    
    filename = os.path.join(foldername, "0.jpg")
    image = cv2.imread(filename)
    height, width, _ = image.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 1
    out = cv2.VideoWriter(output_folder + "/{:03d}.mp4".format(i), fourcc, fps, (width, height))
    
    for j in range(50):
        filename = os.path.join(foldername, "{:d}.jpg".format(j))
        image = cv2.imread(filename)
        out.write(image)
    out.release()

        