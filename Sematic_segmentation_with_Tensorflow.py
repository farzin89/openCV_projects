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
empty_img= np.zeros((H,W))
detection_th = 0.5
for j in range(len(masks)):
    bbox = boxes[0,0,j]
    mask = masks[j]

    class_id = bbox[1]
    score = bbox[2]





