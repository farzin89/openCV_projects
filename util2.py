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
