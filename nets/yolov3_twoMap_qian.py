from collections import OrderedDict
from nets.darknet_return4_2 import darknet53
import torch

import torch.nn as nn

def conv2d(in_filter, out_filter, kernel_size,stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    m=nn.Sequential(OrderedDict([
        ("conv",nn.Conv2d(in_filter,out_filter,kernel_size=kernel_size,padding=pad,bias=False,stride=stride)),
        ("bn",nn.BatchNorm2d(out_filter)),
        ("relu",nn.LeakyReLU(0.1))
    ]))
    return m

def Conv2d(in_filter, out_filter, kernel_size=1,stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    m=nn.Sequential(OrderedDict([
        ("conv",nn.Conv2d(in_filter,out_filter,kernel_size=kernel_size,padding=pad,bias=False,stride=stride)),
        ("bn",nn.BatchNorm2d(out_filter)),
        ("relu",nn.LeakyReLU(0.1))
    ]))
    return m

def conv2DBlock(filters_list, in_filter):
    m = nn.Sequential(
        conv2d(in_filter, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

def Upsample(in_filter,out_filter):
    m = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_filter, out_filter, 1)
    )
    return m

def last_layer(in_filter,out_filter,lastest_filter):
    m=nn.Sequential(OrderedDict([
        ("conv1",conv2d(in_filter,out_filter,kernel_size=3)),
        ("conv2",nn.Conv2d(out_filter,lastest_filter,kernel_size=1,stride=1,padding=0,bias=True))
    ]))
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes,pretrained=False):
        super(YoloBody,self).__init__()
        self.backbone=darknet53()
        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        lastest_filter=len(anchors_mask[0])*(num_classes + 5)
        self.conv2d_Block1 = conv2DBlock([256, 512], out_filters[-2])
        self.conv2d_1 = last_layer(256, 512, 18)
        self.conv2d_Block2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_Block2 = conv2DBlock([128, 256], out_filters[-3])
        self.conv2d_2 = last_layer(128, 256, 18)

    def forward(self,x):
        # ---------------------------------------------------#
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        # ---------------------------------------------------#
        x3, x2, x1= self.backbone(x)

        out1_branch = self.conv2d_Block1(x1)  #256, 40, 32
        out1 = self.conv2d_1(out1_branch)

        x2_in = self.conv2d_Block2_upsample(out1_branch)
        x2_in = x2 + x2_in
        out2_branch = self.conv2d_Block2(x2_in)

        out2 = self.conv2d_2(out2_branch)

        return out1, out2
