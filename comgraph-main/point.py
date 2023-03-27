import cv2
import sys


def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print((x, y))


if len(sys.argv) != 2:
    print("usage: python3", sys.argv[0], "image_file")
    exit(0)
img = cv2.imread(sys.argv[1])
cv2.imshow('sample', img)
cv2.setMouseCallback('sample', onMouse)
cv2.waitKey(0)
