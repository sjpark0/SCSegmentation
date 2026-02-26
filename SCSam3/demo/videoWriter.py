import cv2
import os

input_folder = "../../Data/VideoSample_2"
output_folder = "../../Data/VideoSample_3"


for i in range(16):
    filename = input_folder + "/{:03d}.mp4".format(i)    
    cap = cv2.VideoCapture(filename)
    loop = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        foldername = output_folder + "/{:d}".format(i)
        os.makedirs(foldername, exist_ok=True)
        filename = foldername + "/{:d}.jpg".format(loop)    
        cv2.imwrite(filename, frame)
        loop += 1
    cap.release()
        