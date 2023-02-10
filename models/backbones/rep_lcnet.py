# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .repvggblock import RepVGGBlock,repvgg_model_convert,hswish,hsigmoid


class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 act='hardswish'):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups)

        self._batch_norm = nn.BatchNorm2d(num_filters)
        if act=="hardswish":
            self._relu = hswish()
        else:
            self._relu = nn.ReLU6()

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self._relu(y)
        return y


class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 act='hardswish',
                 dw_size=3,
                 padding=1,
                 use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self._depthwise_conv = RepVGGBlock(in_channels=num_channels, out_channels=int(num_filters1 * scale), 
                               kernel_size=dw_size, stride=stride, 
                               padding=padding,groups=int(num_groups * scale),
                               deploy=False,act=act, use_se=self.use_se) 
        self._pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        y = self._pointwise_conv(y)
        return y


class Lcnet(nn.Module):
    def __init__(self, in_channels=3, scale=0.5, **kwargs):
        super().__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(
            num_channels=in_channels,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1)

        conv2_1 = DepthwiseSeparable(
            num_channels=int(32 * scale),
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale)
        self.block_list.append(conv2_1)

        conv2_2 = DepthwiseSeparable(
            num_channels=int(64 * scale),
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=1,
            scale=scale)
        self.block_list.append(conv2_2)

        conv3_1 = DepthwiseSeparable(
            num_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale)
        self.block_list.append(conv3_1)

        conv3_2 = DepthwiseSeparable(
            num_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=(2, 1),
            scale=scale)
        self.block_list.append(conv3_2)

        conv4_1 = DepthwiseSeparable(
            num_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale)
        self.block_list.append(conv4_1)

        conv4_2 = DepthwiseSeparable(
            num_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=(2, 1),
            scale=scale)
        self.block_list.append(conv4_2)

        for _ in range(5):
            conv5 = DepthwiseSeparable(
                num_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                dw_size=5,
                padding=2,
                scale=scale,
                use_se=False)
            self.block_list.append(conv5)

        conv5_6 = DepthwiseSeparable(
            num_channels=int(512 * scale),
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=(2, 1),
            dw_size=5,
            padding=2,
            scale=scale,
            use_se=False)
        self.block_list.append(conv5_6)

        conv6 = DepthwiseSeparable(
            num_channels=int(1024 * scale),
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=1,
            dw_size=5,
            padding=2,
            use_se=False,
            scale=scale)
        self.block_list.append(conv6)
        self.block_list = nn.Sequential(*self.block_list)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        return y

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = hsigmoid()
    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hardsigmoid(outputs)
        return torch.mul(inputs, outputs)

def rep_lcnet(pretrained,in_channels,scale):
    model = Lcnet(in_channels=in_channels, scale=scale)
    if pretrained:
        pass
    return model

def CreateBackboneModel(modelName,pretrained,scale=1,isGray=False):
    assert modelName in ['rep_lcnet']
    in_channels = 1 if isGray else 3
    model = globals().get(modelName)(pretrained,scale=scale,in_channels=in_channels)
    return model

# model = CreateBackboneModel('lcnet',pretrained=False,scale=0.5,isGray=False)
# print(model)
# import torch
# img = torch.rand((1,3,32,280))
# out = model(img)
# print(out.shape)