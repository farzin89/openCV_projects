import numpy as np
import cv2


def get_outputs(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    outs= net.forward(output_layers)
    outs= [c for out in outs for c in out if c[4] > 0.1]
    return outs

def draw(bbox,img):
    xc,yc,w,h = bbox
    img = cv2.rectangle(img,
                        (xc - int(w/2),yc - int(h/2)),
                        (xc + int(w/2),yc + int(h/2)),
                        (0,255,0),20)
def NMS(boxes,class_ids,confidences,overlapThresh = 0.5):
    # Return an empty list, if no boxes given
    boxes = np.asarray(boxes)
    class_ids =np.asarray(class_ids)
    confidences = np.asarray(confidences)

    if len (boxes) == 0:
        return [],[],[]
    x1 = boxes[:, 0] - (boxes[:, 2] / 2) # x coordinate of the top-left corner
    y1 = boxes[:, 1] - (boxes[:, 3] / 2)  # y coordinate of the top-left corner
    x2 = boxes[:, 0] - (boxes[:, 2] / 2)  # x coordinate of the bottom-right corner
    y2 = boxes[:, 1] - (boxes[:, 3] / 2)  # x coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    #Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2- x1 + 1) * (y2 - y1 + 1) # we add 1,because the pixel at the start as well as at the end
    # counts the indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # create temporary indices
        tem_indices = indices[indices!=i]
        # find out the coordinats of the intersection box
        xx1 = np.maximum(box[0] - (box[2] / 2), boxes[tem_indices, 0] - (boxes[tem_indices, 2] / 2))
        yy1 = np.maximum(box[1] - (box[3] / 2), boxes[tem_indices, 1] - (boxes[tem_indices, 3] / 2))
        xx2 = np.maximum(box[0] + (box[2] / 2), boxes[tem_indices, 0] + (boxes[tem_indices, 2] / 2))
        yy2 = np.maximum(box[1] + (box[3] / 2), boxes[tem_indices, 1] + (boxes[tem_indices, 3] / 2))
        # Find out the width and height of the intersection box
        w = np.maximum(0,xx2 -  xx1 + 1)
        h = np.maximum(0,yy2 -  yy1 + 1)
        # compute the ratio of overlap
        overlap = (w+h) / areas[tem_indices]
        # if the actual bounding box has an overlap bigger than threshold with any box,remove it's index
        if np.any(overlap) > overlapThresh:
            indices = indices[indices!=i]
        # return only the boxes at the remaining indices
    return boxes[indices], class_ids[indices], confidences[indices]

def get_limits(color):
    c =np.uint8([[color]]) # here insert the BGR values which you want to convert to HSV
    hsvC = cv2.cvtColor(c,cv2.COLOR_BGR2HSV)

    lowerLimit = hsvC[0][0][0] -10,100,100
    upperLimit = hsvC[0][0][0] + 10,255,255

    lowerLimit = np.array(lowerLimit,dtype=np.uint8)
    upperLimit = np.array(upperLimit,dtype=np.uint8)

    return lowerLimit,upperLimit