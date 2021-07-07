import cv2
import numpy as np

def load_cutout_to_contours_and_fill(img_path, resize_dst_size = None):
    cutout = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    alpha = cutout[:, :, 3]
    if resize_dst_size is not None:
        alpha = cv2.resize(alpha,resize_dst_size)
    contours, hierarchy = cv2.findContours(alpha, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    return alpha, contours

def preprocess_frame(frame):
    frame = np.fliplr(frame)
    return frame

def draw_contours(frame, contours):
    frame = cv2.drawContours(frame, contours, -1, (255,255,255), 5)
    return frame


def extract_diff_from_bg(frame, bg, diff_intensity_thresh=30):
    diff = cv2.absdiff(frame, bg)
    diff = cv2.threshold(diff, diff_intensity_thresh, 255, cv2.THRESH_BINARY)[1]
    kernel_erode = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)
    er_image = cv2.erode(diff, kernel_erode, iterations=2)
    dil_image = cv2.dilate(er_image, kernel_dilate, iterations=3)
    diff = dil_image
    diff[np.sum((diff == 255), axis=2) > 1] = 255
    diff[np.sum((diff == 255), axis=2) <= 1] = 0

    return diff