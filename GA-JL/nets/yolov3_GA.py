from collections import OrderedDict
from nets.darknet_return4_2 import darknet53
import torch

import torch.nn as nn

def Dense0(in_filter,out_filter):
    outputs=nn.Sequential(OrderedDict([
        ("linear", nn.Linear(in_filter,out_filter,bias=False)),
        ("relu",   nn.LeakyReLU(0.1))
    ]))
    return outputs

def Dense1(in_filter,out_filter):
    outputs=nn.Sequential(OrderedDict([
        ("linear", nn.Linear(in_filter,out_filter,bias=False)),
        ("sigmoid",   nn.Sigmoid())
    ]))
    return outputs

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

class CoorAttn(nn.Module):
    def __init__(self,inplanes):
        super(CoorAttn, self).__init__()
        self.conv = conv2d(inplanes, inplanes//4, 3)
        self.conv_h = nn.Conv2d(inplanes//4, inplanes,1)
        self.conv_w = nn.Conv2d(inplanes//4, inplanes,1)
        self.sigmoid=nn.Sigmoid()
        self.AvgPool_h = nn.AdaptiveAvgPool2d([None, 1])
        self.AvgPool_w = nn.AdaptiveAvgPool2d([1, None])
        #self.MaxPool_h = nn.MaxPool2d([None, 1])
        #self.MaxPool_w = nn.MaxPool2d([1, None])

    def forward(self,x, x1):
        bs,c,h,w = x.shape
        pool_h = self.AvgPool_h(x)
        #print(pool_h.shape)
        pool_w = self.AvgPool_w(x)
        y=torch.cat([pool_h, pool_w.permute(0,1,3,2)],dim=2)
        y=self.conv(y)
        x_h,x_w = torch.split(y,[h,w],dim=2)
        x_h = self.sigmoid(self.conv_h(x_h))
        x_w = self.sigmoid(self.conv_w(x_w)).permute(0,1,3,2)
        x1 =  x1 + x1 * x_h * x_w
        return x1

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, x1):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        y = x1 + x1 * y.expand_as(x1)
        return y  # 注意力作用每一个通道上

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

        self.PFF3_downsample_2 = conv2d(128, 128, 3, stride=2)  # 32,32, 256
        self.PFF3_downsample_1 = conv2d(128, 128, 3, stride=2)  # 32,32, 256
        self.PFF3_conv = PFF(128, 256)  # 64,64, 128

        self.coorAttn = CoorAttn(128)
        self.SE= SE_Block(128)

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
        PFF3_coor = self.PFF3_downsample_2(PFF3_3)  # 32,32,256
        PFF3_se = self.PFF3_downsample_1(PFF3_3)  # 32,32,256
        #print(PFF3_2.shape)
        #print(x3.shape)

        x2_bs, x2_chs, x2_h, x2_w = x2.shape
        x2_coor, x2_se = torch.split(x2, x2_chs//2, dim=1)

        x2_coor = self.coorAttn(PFF3_coor, x2_coor)
        x2_se = self.SE(PFF3_se, x2_se)
        x2 = torch.cat([x2_coor, x2_se], dim=1)

        out1_branch = self.conv2d_Block1(x1)  #256, 40, 32
        out1 = self.conv2d_1(out1_branch)

        x2_in = self.conv2d_Block2_upsample(out1_branch)
        x2_in = x2 + x2_in
        out2_branch = self.conv2d_Block2(x2_in)

        out2 = self.conv2d_2(out2_branch)

        return out1, out2
