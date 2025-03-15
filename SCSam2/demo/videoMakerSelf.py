import cv2
import os

input_folder = "../../Data/Sample1"
output_folder = "../../Data/VideoSample_2"

os.makedirs(output_folder, exist_ok=True)

for i in range(16):
    filename = input_folder + "/images/{:03d}.png".format(i)    
    image = cv2.imread(filename)
    height, width, _ = image.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 30
    out = cv2.VideoWriter(output_folder + "/{:03d}.mp4".format(i), fourcc, fps, (width, height))
    
    for j in range(100):
        filename = input_folder + "/images/{:03d}.png".format(i)   
        image = cv2.imread(filename)
        out.write(image)
    out.release()
        