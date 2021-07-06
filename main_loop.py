import cv2
import numpy as np

from utils import load_cutout_to_contours_and_fill, preprocess_frame, draw_contours

cutout_path = r"cutouts\sample_cutout.png"
fill, contours = load_cutout_to_contours_and_fill(cutout_path)

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

started_bg = False
has_bg = False
bg_intensity_threshold = 50
bg_counter = 0
# Main loop
while True:
    ret, frame = cap.read()
    pframe = preprocess_frame(frame, scale_size=2)
    pframe = draw_contours(pframe,contours)
    cv2.imshow('Input', pframe)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
