
import os
import numpy as np
#define constants
import cv2
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
detections= util2.get_outputs(net)
# bboxes, class_ids,confidences
bboxes = []
class_ids =[]
scores = []
for detection in detections:
    #[x1,x2,x3,x4,x5,x6,...,xN]
    bbox = detection[:4]

    print(bbox)
    bbox_confidence =detection[4]

    class_id= np.argmax(detection[5:])
    score = np.amax(detection[5:])

    bboxes.append(bbox)
    class_ids.append(class_id)
    scores.append(score)

    break

#for bbox in bboxes:
#    img = util2.draw(bbox,img)