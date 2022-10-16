import cv2
import math

path = 'test.JPG'
img = cv2.imread(path)
pointsList = []
# get the mouse point
def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # store all the point that we click on
        cv2.circle(img,(x,y),5,(0,0,255),cv2.FILLED)
        pointsList.append([x,y])
        print(pointsList)
        #print(x,y)
# store all the point that we click on
while True:
    cv2.imshow('Image',img)
    cv2.setMouseCallback('Image',mousePoints )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointsList = []
        img = cv2.imread(path)


