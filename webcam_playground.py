import cv2
import numpy as np
from time import time
from utils import load_cutout_to_contours_and_fill, preprocess_frame, draw_contours, extract_diff_from_bg

logo = cv2.imread("logo.png")
sig = cv2.imread("sig.png")

logo = cv2.resize(logo, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
sig = cv2.resize(sig, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

logo_full_im = np.zeros((960, 1280, 3))
logo_full_im[10:logo.shape[0] + 10, 10:logo.shape[1] + 10] = logo
logo_full_im[logo_full_im==255] = 0
logo_full_im = logo_full_im.astype(np.uint8)

print(sig.shape)
sig_full_im = np.zeros((960, 1280, 3))
sig_full_im[870:sig.shape[0] + 870, 870:sig.shape[1] + 870] = sig
sig_full_im[sig_full_im==255] = 0
sig_full_im = sig_full_im.astype(np.uint8)

# cv2.imshow("meh", logo_full_im)

cutouts = [f"cutouts\\cutout{i}.png" for i in range(1, 8)]
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
    delta = cv2.absdiff(diff[60:420, :], target[60:420, :])
    return (delta == 0).sum() / delta.size


front_img = None


def calc_sim_image(diff, target):
    delta = cv2.absdiff(diff, target)

    return delta


score = 0
num_succ_in_a_row = 0
start_time = time()
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

    sim_image = cv2.putText(sim_image, "{:.3f}, {:.3f}".format(similarity, sim_avg), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255))
    if sim_avg > 0.94:
        pframe = cv2.putText(pframe, "You Did it!".format(similarity, sim_avg), (250, 40),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        num_succ_in_a_row += 1
    else:
        num_succ_in_a_row = 0
    if num_succ_in_a_row > 10:
        num_succ_in_a_row = 0
        curr_cutout += 1
        score += 1
        try:
            fill, contours = load_cutout_to_contours_and_fill(cutouts[curr_cutout], (640, 480))
        except:
            break

    pframe = cv2.putText(pframe, "Score: {}".format(score), (20, 460),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    pframe = cv2.resize(pframe, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    print(pframe.shape, logo_full_im.shape)
    pframe+=logo_full_im
    pframe+=sig_full_im

    pframe = cv2.putText(pframe, "Time: {:.2f}".format(time()-start_time), (1060, 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    # pframe = cv2.addWeighted(pframe, 0.4, logo_full_im, 0.1, 0)
    cv2.imshow('Input', np.hstack([pframe, np.vstack([diff, sim_image])]))

    c = cv2.waitKey(1)
    if c == 27:
        break
final_frame = np.zeros((960, 1280, 3))
final_frame = cv2.putText(final_frame, "Hooray! You finished Hooman Tetris in {:.2f} seconds!".format(time() - start_time), (200, 450),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

final_frame = cv2.putText(final_frame, "we hope you enjoyed the game :)", (350, 500),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

while True:
    cv2.imshow('Input', final_frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
