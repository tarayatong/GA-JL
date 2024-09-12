# coding: utf-8
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image

from nets.pff_scalefactor_shuffle_spp import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)

def getOutParm():
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    classes_path    = 'model_data/plane_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    model_path = r"logs/MAPFF_balance0.5/113_139/2_ep060-loss0.024-val_loss0.022.pth"

    net    = YoloBody(anchors_mask, num_classes)
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net    = net.eval()
    '''image = Image.open(r"/home/liujing/Code/data/SAITD/SAITD/113/JPEGImages/0.png")
    image = cvtColor(image)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)), 0)
    images = torch.from_numpy(image_data)
    images = images.cuda()
    outputs = net(images)'''
    print(net.conv2d_22.conv)
    '''for name, layer in net.named_parameters(recurse=True):
        print(name, layer.shape, sep=" ")'''

if __name__ == "__main__":
    #getOutParm()
    image = Image.open(r"/home/liujing/Code/data/SAITD/SAITD/113/JPEGImages/0.png")
    image = image.convert('RGB')
    image.save(r"/home/liujing/Code/yolo3-After/yolo3-After/yolo3/yolov4/data/0.png")
