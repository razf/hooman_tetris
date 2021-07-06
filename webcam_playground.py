import cv2
import numpy as np

from utils import load_cutout_to_contours_and_fill, preprocess_frame, draw_contours

cutout_path = r"cutouts\sample_cutout.png"
fill, contours = load_cutout_to_contours_and_fill(cutout_path, (640,480))

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

final_bg = cv2.imread("background.png")
# Main loop
while True:
    ret, frame = cap.read()
    pframe = preprocess_frame(frame, scale_size=1)

    final_bg_blur = cv2.blur(final_bg, (5,5))
    pframe_blur = cv2.blur(pframe, (5,5))

    final_bg_blur_g = cv2.cvtColor(final_bg_blur, cv2.COLOR_RGB2GRAY)
    bg_pixel = final_bg_blur_g[80,80]

    pframe_g = cv2.cvtColor(pframe_blur, cv2.COLOR_RGB2GRAY)
    pframe_g = pframe_g*(bg_pixel/pframe_g[80,80]) #Normalize
    pframe_g = pframe_g.astype(np.uint8)

    diff = cv2.absdiff(pframe_g, final_bg_blur_g)
    diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

    kernel_erode = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((7, 7), np.uint8)
    er_image = cv2.erode(diff, kernel_erode, iterations=2)
    dil_image = cv2.dilate(er_image, kernel_dilate, iterations=3)
    diff=dil_image

    pframe_g = draw_contours(pframe_g, contours)

    cv2.imshow('Input', np.hstack([pframe_g, final_bg_blur_g, cv2.absdiff(diff,fill)]))

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
