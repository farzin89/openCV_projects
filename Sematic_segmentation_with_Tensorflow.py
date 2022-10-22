import cv2
import os

import numpy as np

from util import get_detection
# (1) define path

cfg_path = ''
weight_path = ''
class_names_path = ''

img_path ='cat.png'

# 2 load image
img = cv2.imread(img_path)
H,W,C= img.shape
# 3 load model
net = cv2.dnn.readNetFromTensorflow(weight_path,cfg_path)
# 4 convert image
blob = cv2.dnn.blobFromImage(img)
# get mask
boxes,masks = get_detection(net,blob)
# 6 draw masks
empty_img= np.zeros((H,W,C))
detection_th = 0.5
for j in range(len(masks)):
    bbox = boxes[0,0,j]


    class_id = bbox[1]
    score = bbox[2]
    # do some filtering to reduce number of detections
    if score > detection_th:
        mask = masks[j]

        x1,y1,x2,y2 = int(bbox[3] * W),int(bbox[4]* H),int(bbox[5]* W),int(bbox[6]* H)

        mask = mask[int(class_id)]
        mask = cv2.resize(mask,(x2-x1,y2-y1))
        for c in range(3):
            empty_img[y1:y2,x1:x2,c] = mask


        print(bbox.shape)
        print(mask.shape)

# 7 visualization
cv2.imshow('mask', empty_img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



