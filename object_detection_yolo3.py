
import os

#define constants
import cv2.dnn

import util2

model_cfg_path = os.path.join("yolov3.cfg")
model_weight_path = os.path.join("darknet53.conv.74")
class_name_path = os.path.join("class.names")

img_path = "pexels-diana-huggins-615369.jpg"

# load class names

with open (class_name_path,'r') as f:
    class_names =(j[:-1] for j in f.readlines() if len(j)>2)
    f.close()
# load model
net= cv2.dnn.readNetFromDarknet(model_cfg_path,model_weight_path)
#load image
img =cv2.imread(img_path)
#convert image
blob = cv2.dnn.blobFromImage(img,1/255,(320,320),(0,0,0),True)

# get detection
net.setInput(blob)
outputs= util2.get_outputs(net)
# bboxes, class_ids,confidences
bboxes = []
class_ids =[]
scorse = []