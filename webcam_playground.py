import cv2
import numpy as np
import queue
from utils import load_cutout_to_contours_and_fill, preprocess_frame, draw_contours, extract_diff_from_bg

cutout_path = r"cutouts\cutout4.png"
cutouts = [f"cutouts\\cutout{i}.png" for i in range(1,5)]
curr_cutout = 0
fill, contours = load_cutout_to_contours_and_fill(cutouts[0], (640, 480))

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

    return delta

score=0
num_succ_in_a_row = 0
while True:
    ret, frame = cap.read()
    pframe = preprocess_frame(frame)

    bg_pixel = final_bg[80, 80]

    pframe = pframe * (bg_pixel / pframe[80, 80])  # Normalize
    pframe = pframe.astype(np.uint8)
    if front_img is None:
        front_img = pframe.astype(np.float32)
    front_img = cv2.accumulateWeighted(pframe, front_img, 0.2)
    pframe = cv2.convertScaleAbs(front_img)

    diff = extract_diff_from_bg(pframe, final_bg, diff_intensity_thresh=40)

    pframe = draw_contours(pframe, contours)

    sim_image = calc_sim_image(diff, fill)
    similarity = calc_similarity(diff, fill)

    queue.append(similarity)
    if len(queue) > 50:
        queue.pop(0)
    sim_avg = np.average(queue)

    sim_image = cv2.putText(sim_image, "{:.3f}, {:.3f}".format(similarity,sim_avg), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    if sim_avg > 0.94:
        pframe = cv2.putText(pframe, "You Did it!".format(similarity, sim_avg), (250, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        num_succ_in_a_row+=1
    else:
        num_succ_in_a_row=0
    if num_succ_in_a_row > 10:
        num_succ_in_a_row=0
        curr_cutout+=1
        score+=1
        fill, contours = load_cutout_to_contours_and_fill(cutouts[curr_cutout], (640, 480))

    pframe = cv2.putText(pframe, "Score: {}".format(score), (20, 460),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    pframe = cv2.resize(pframe, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

    cv2.imshow('Input', np.hstack([pframe,np.vstack([diff, sim_image])]))

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
