import cv2
import numpy as np
import queue
from utils import load_cutout_to_contours_and_fill, preprocess_frame, draw_contours, extract_diff_from_bg

cutout_path = r"cutouts\sample_cutout.png"
fill, contours = load_cutout_to_contours_and_fill(cutout_path, (640, 480))

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

final_bg = cv2.imread("background.png")
queue = []

# Main loop
def calc_similarity(diff, target):
    delta = cv2.absdiff(diff, target)
    return (delta == 0).sum() / delta.size

front_img = None


def calc_sim_image(diff, target):
    delta = cv2.absdiff(diff, target)
    delta = cv2.cvtColor(delta, cv2.COLOR_GRAY2RGB)
    delta[(diff > 0) & (target > 0)] = (0, 255, 0)
    delta[(diff > 0) & (target == 0)] = (255,0,0)
    return delta


while True:
    ret, frame = cap.read()
    pframe = preprocess_frame(frame, scale_size=1)

    final_bg_blur = cv2.blur(final_bg, (5, 5))
    pframe_blur = cv2.blur(pframe, (5, 5))

    final_bg_blur_g = cv2.cvtColor(final_bg_blur, cv2.COLOR_RGB2GRAY)
    bg_pixel = final_bg_blur_g[80, 80]

    pframe_g = cv2.cvtColor(pframe_blur, cv2.COLOR_RGB2GRAY)
    pframe_g = pframe_g * (bg_pixel / pframe_g[80, 80])  # Normalize
    pframe_g = pframe_g.astype(np.uint8)
    # if front_img is None:
    #     front_img = pframe_g.astype(np.float32)
    # front_img = cv2.accumulateWeighted(pframe_g, front_img, 0.8)
    # pframe_g = cv2.convertScaleAbs(front_img)

    diff = extract_diff_from_bg(pframe_g, final_bg_blur_g)

    pframe_g = draw_contours(pframe_g, contours)
    sim_image = calc_sim_image(diff, fill)
    similarity = calc_similarity(diff, fill)

    queue.append(similarity)
    if len(queue) > 50:
        queue.pop(0)
    sim_avg = np.average(queue)

    sim_image = cv2.putText(sim_image, "{:.3f}, {:.3f}".format(similarity,sim_avg), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    if sim_avg > 0.94:
        sim_image = cv2.putText(sim_image, "You Did it!".format(similarity, sim_avg), (300, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    pframe_g = cv2.cvtColor(pframe_g, cv2.COLOR_GRAY2RGB)
    cv2.imshow('Input', np.hstack([pframe_g, sim_image]))

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
