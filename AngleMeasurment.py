import cv2
import math

path = 'test.JPG'
img = cv2.imread(path)

# get the mouse point
def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)

cv2.imshow('Image',img)
cv2.setMouseCallback('Image',mousePoints )
cv2.waitKey(0)

