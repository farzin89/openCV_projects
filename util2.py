import numpy as np
import cv2


def get_outputs(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    outs= net.forward(output_layers)
    outs= [c for out in outs for c in out if c[4] > 0.1]
    return outs