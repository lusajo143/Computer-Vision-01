import cv2
import numpy as np

vid = cv2.VideoCapture(0)

x1, y1, k = 100, 100, 1


def select(event, x, y, flags, params):
    global x1, y1, k
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        k = -1


cv2.namedWindow("window")
cv2.setMouseCallback("window", select)

while True:

    _, frame = vid.read()

    cv2.imshow("window", frame)

    if cv2.waitKey(1) == 0 or k == -1:
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()
        break
old_points = np.array([[x1, y1]], dtype="float32").reshape(-1, 1, 2)
mask = np.zeros_like(frame)

while True:

    _, frame = vid.read()
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None,
                                                         maxLevel=1,
                                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                   15, 0.08))

    print(new_points)
    cv2.circle(mask, (new_points.ravel()[0], new_points.ravel()[1]), 10, (0, 255, 0), 2)
    old_points = new_points.copy()
    old_gray = new_gray.copy()
    cv2.imshow("mask", mask)

    cv2.imshow("window", frame)

    if cv2.waitKey(1) == 0:
        cv2.destroyAllWindows()
        break
