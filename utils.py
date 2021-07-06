import cv2


def load_cutout_to_contours_and_fill(img_path):
    cutout = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    alpha = cutout[:, :, 3]
    contours, hierarchy = cv2.findContours(alpha, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return alpha, contours
