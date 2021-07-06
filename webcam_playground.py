import cv2
import os.path as osp

fps = 60
save_path = r"C:\Code\Datathon\Playground"
cap = cv2.VideoCapture(0)
fnum = 1
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cnt = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    cnt+=1
    if cnt >= 120:
        cnt=0
        cv2.imwrite(osp.join(save_path,f"{fnum}.png"), frame)
        fnum+=1
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
