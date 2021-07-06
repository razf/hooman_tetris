import cv2
import numpy as np

from utils import load_cutout_to_contours_and_fill, preprocess_frame, draw_contours

cutout_path = r"cutouts\raz_cutout.png"
fill, contours = load_cutout_to_contours_and_fill(cutout_path, (640,480))

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
    pframe = preprocess_frame(frame, scale_size=1)
    if not started_bg:
        bg_img = pframe.astype(np.float32)
        bg_int8 = cv2.convertScaleAbs(bg_img)
        final_bg = pframe
        diff = pframe
        started_bg = True
    elif not has_bg:
        bg_img = cv2.accumulateWeighted(pframe, bg_img, 0.05)
        bg_int8 = cv2.convertScaleAbs(bg_img)
        diff = cv2.absdiff(pframe, bg_int8)
        if diff[diff > 50].sum() < 100:
            bg_counter += 1
        else:
            bg_counter = 0
        if bg_counter > 60:
            final_bg = bg_int8
            has_bg = True
            cv2.imwrite("background.png", bg_int8)
            break
        print(diff[diff > 50].sum())

    cv2.imshow('Input', np.hstack([pframe, bg_int8, diff]))

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
