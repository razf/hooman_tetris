import cv2
import pathlib
import os.path as osp
import numpy as np
from utils import load_cutout_to_contours_and_fill
fps = 60
cur_path = str(pathlib.Path().resolve())
images_save_path = cur_path + r'\\Playground'
fnum = 1

cutout_path = r"cutouts\sample_cutout.png"
fill, contours = load_cutout_to_contours_and_fill(cutout_path)

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cnt = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    frame = np.fliplr(frame)
    frame = cv2.drawContours(frame, contours, -1, (255,255,255), 10)
    cv2.imshow('Input', frame)

    cnt+=1
    if cnt >= fps:
        cnt=0
        cv2.imwrite(osp.join(images_save_path, f"{fnum}.png"), frame)
        # cv2.imwrite(osp.join(save_path,f"{fnum}.png"), frame)
        fnum+=1
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
