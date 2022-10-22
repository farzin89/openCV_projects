import cv2
import os

# (1) define path

cfg_path = ''
weight_path = ''
class_names_path = ''

img_path ='cat.png'

# 2 load image
img = cv2.imread(img_path)
# load model
net = cv2.dnn.readNetFromTensorflow(weight_path,cfg_path)
