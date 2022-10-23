
import os

#define constants

model_cfg_path = os.path.join("yolov3.cfg")
model_weight_path = os.path.join("")
class_name_path = os.path.join("class.names")

img_path = "pexels-diana-huggins-615369.jpg"

# load class names

with open (class_name_path,'r') as f:
    class_names =(j[:-1] for j in f.readlines() if len(j)>2)
    f.close()
