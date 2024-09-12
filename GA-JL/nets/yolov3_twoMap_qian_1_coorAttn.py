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

def PFF(in_filter, out_filter):
    m = nn.Sequential(
        conv2d(in_filter, out_filter, 1),
        conv2d(out_filter, in_filter, 3)
    )
    return m

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(planes[0])
        self.relu1  = nn.LeakyReLU(0.1)

        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out

class CoorAttn(nn.Module):
    def __init__(self,inplanes):
        super(CoorAttn, self).__init__()
        self.conv = conv2d(inplanes, inplanes//4, 3)
        self.conv_h = nn.Conv2d(inplanes//4, inplanes,1)
        self.conv_w = nn.Conv2d(inplanes//4, inplanes,1)
        self.sigmoid=nn.Sigmoid()
        self.AvgPool_h = nn.AdaptiveAvgPool2d([None, 1])
        self.AvgPool_w = nn.AdaptiveAvgPool2d([1, None])

        self.conv1 = conv2d(2*inplanes, inplanes, 3)
        self.ResBlock = ResBlock(inplanes, [inplanes//2, inplanes])

    def forward(self,x, x1):
        bs,c,h,w = x.shape
        x1 = self.conv1(torch.cat([x, x1],dim=1))
        x1 = self.ResBlock(x1)
        pool_h = self.AvgPool_h(x1)
        pool_w = self.AvgPool_w(x1)
        y = torch.cat([pool_h, pool_w.permute(0,1,3,2)],dim=2)
        y = self.conv(y)
        x_h,x_w = torch.split(y,[h,w],dim=2)
        x_h = self.sigmoid(self.conv_h(x_h))
        x_w = self.sigmoid(self.conv_w(x_w)).permute(0,1,3,2)
        x =  x + x * x_h * x_w
        return x

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes,pretrained=False):
        super(YoloBody,self).__init__()
        self.backbone=darknet53()
        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #PFF
        self.PFF1_conv = PFF(512, 256)  # 16,16, 512
        self.PFF1_upsample_2 = Upsample(512, 256)  # 32,32, 256
        #self.PFF1_cat_conv = conv2d(1024, 512, 1)

        self.PFF2_conv = PFF(256, 512)  # 32,32, 256
        self.PFF2_upsample_2 = Upsample(256, 128)  # 64,64, 128
        self.PFF2_cat_conv = conv2d(768, 256, 1)

        self.PFF3_downsample_2 = conv2d(128, 256, 3, stride=2)  # 32,32, 256
        self.PFF3_conv = PFF(128, 256)  # 64,64, 128

        self.coorAttn = CoorAttn(256)

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        lastest_filter=len(anchors_mask[0])*(num_classes + 5)
        #13*13*1024->13*13*512
        self.conv2d_Block0 = conv2DBlock([512,1024],out_filters[-1])

        self.conv2d_Block1_conv=conv2d(512,256,1)
        self.conv2d_Block1_upsample=nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_Block1=conv2DBlock([256,512],out_filters[-2])

        self.conv2d_Block2_conv = conv2d(256, 128, 1)
        self.conv2d_Block2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_Block2 = conv2DBlock([128, 256], out_filters[-3] )

        self.conv2d_2=last_layer(128,256,18)
        self.conv2d_1=last_layer(256,512,18)
        #self.conv2d_0 =last_layer(512, 1024, 18)

        self.conv2d_down2=nn.Conv2d(128,128,kernel_size=3,padding=1,stride=2)
        self.conv2d_down1 = nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=2)

    def forward(self,x):
        # ---------------------------------------------------#
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        # ---------------------------------------------------#
        x3, x2, x1= self.backbone(x)

        # PFF
        PFF3_3 = self.PFF3_conv(x3)  # 64,64,128
        PFF3_2 = self.PFF3_downsample_2(PFF3_3)  # 32,32,256
        #print(PFF3_2.shape)
        #print(x2.shape)

        x2 = self.coorAttn(x2, PFF3_2)

        out1_branch = self.conv2d_Block1(x1)  #256, 40, 32
        out1 = self.conv2d_1(out1_branch)

        x2_in = self.conv2d_Block2_upsample(out1_branch)
        x2_in = x2 + x2_in
        out2_branch = self.conv2d_Block2(x2_in)

        out2 = self.conv2d_2(out2_branch)

        return out1, out2
