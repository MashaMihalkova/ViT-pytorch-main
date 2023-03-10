# import cv2
# import matplotlib.pyplot as plt
# from os.path import isfile
# import torch.nn.init as init
# import torch
import torch.nn as nn
# import numpy as np
# import pandas as pd
# import os
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset
# from torchvision import transforms
# from torch.optim import Adam, SGD
# import time
# from torch.autograd import Variable
# import torch.functional as F
# from tqdm import tqdm
# from sklearn import metrics
# import urllib
# import pickle
# import cv2
# import torch.nn.functional as F
# from torchvision import models
# import seaborn as sns


class ResUnit(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        """
        Residual Unit
        """
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, int(outplanes / 4), kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(int(outplanes / 4), int(outplanes / 4), kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(int(outplanes / 4))
        self.conv3 = nn.Conv2d(int(outplanes / 4), outplanes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(int(outplanes / 4))
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride
        self.make_downsample = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if (self.inplanes != self.outplanes) or (self.stride != 1):
            residual = self.make_downsample(residual)

        out += residual

        return out


class AttentionModule_stage1(nn.Module):

    def __init__(self, inplanes, outplanes, size1, size2, size3):
        """
        max_pooling layers are used in mask branch size with input
        """
        super(AttentionModule_stage1, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        self.Resblock1 = ResUnit(inplanes, outplanes)  # first residual block

        self.trunkbranch = nn.Sequential(ResUnit(inplanes, outplanes),
                                         ResUnit(inplanes, outplanes))

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Resblock2 = ResUnit(inplanes, outplanes)
        self.skip1 = ResUnit(inplanes, outplanes)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Resblock3 = ResUnit(inplanes, outplanes)
        self.skip2 = ResUnit(inplanes, outplanes)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Resblock4 = nn.Sequential(ResUnit(inplanes, outplanes),
                                       ResUnit(inplanes, outplanes))

        # self.upsample3 = nn.UpsamplingBilinear2d(size=size3)
        self.Resblock5 = ResUnit(inplanes, outplanes)

        # self.upsample2 = nn.UpsamplingBilinear2d(size=size2)
        self.Resblock6 = ResUnit(inplanes, outplanes)

        # self.upsample1 = nn.UpsamplingBilinear2d(size=size1)

        self.output_block = nn.Sequential(nn.BatchNorm2d(outplanes),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, bias=False),
                                          nn.BatchNorm2d(outplanes),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, bias=False),
                                          nn.Sigmoid())

        self.last_block = ResUnit(inplanes, outplanes)

    def forward(self, x):
        # The Number of pre-processing Residual Units Before
        # Splitting into trunk branch and mask branch is 1.
        # 48*48
        x = self.Resblock1(x)

        # The output of trunk branch
        out_trunk = self.trunkbranch(x)

        # soft Mask Branch
        # The Number of Residual Units between adjacent pooling layer is 1.
        pool1 = self.maxpool1(x)  # (48,48) -> (24,24)
        out_softmask1 = self.Resblock2(pool1)
        skip_connection1 = self.skip1(pool1)  # "skip_connection"

        pool2 = self.maxpool2(out_softmask1)  # (24,24) -> (12,12)
        out_softmask2 = self.Resblock3(pool2)
        skip_connection2 = self.skip2(pool2)

        pool3 = self.maxpool3(out_softmask2)  # (12,12) -> (6,6)
        out_softmask3 = self.Resblock4(pool3)

        out_interp3 = nn.functional.interpolate(out_softmask3, size=self.size3, mode='bilinear',
                                                align_corners=True)  # (6,6)->(12,12)
        out = out_interp3 + skip_connection2
        out_softmask4 = self.Resblock5(out)

        out_interp2 = nn.functional.interpolate(out_softmask4, size=self.size2, mode='bilinear',
                                                align_corners=True)  # (12,12)->(24,24)
        out = out_interp2 + skip_connection1
        out_softmask5 = self.Resblock6(out)

        out_interp1 = nn.functional.interpolate(out_softmask5, size=self.size1, mode='bilinear',
                                                align_corners=True)  # (24,24)->(48,48)
        out_softmask6 = self.output_block(out_interp1)

        out = (1 + out_softmask6) * out_trunk
        last_out = self.last_block(out)

        return last_out


class AttentionModule_stage2(nn.Module):

    def __init__(self, inplanes, outplanes, size1, size2):
        """
        max_pooling layers are used in mask branch size with input
        """
        super(AttentionModule_stage2, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.Resblock1 = ResUnit(inplanes, outplanes)  # first residual block

        self.trunkbranch = nn.Sequential(ResUnit(inplanes, outplanes),
                                         ResUnit(inplanes, outplanes))

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (24,24) -> (12,12)
        self.Resblock2 = ResUnit(inplanes, outplanes)
        self.skip1 = ResUnit(inplanes, outplanes)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (12,12) -> (6,6)
        self.Resblock3 = nn.Sequential(ResUnit(inplanes, outplanes),
                                       ResUnit(inplanes, outplanes))
        # self.upsample2 = nn.UpsamplingBilinear2d(size2)
        self.Resblock4 = ResUnit(inplanes, outplanes)
        # self.upsample1 = nn.UpsamplingBilinear2d(size1)

        self.output_block = nn.Sequential(nn.BatchNorm2d(outplanes),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, bias=False),
                                          nn.BatchNorm2d(outplanes),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, bias=False),
                                          nn.Sigmoid())

        self.last_block = ResUnit(inplanes, outplanes)

    def forward(self, x):
        # The Number of pre-processing Residual Units Before
        # Splitting into trunk branch and mask branch is 1.
        # 48*48
        x = self.Resblock1(x)

        # The output of trunk branch
        out_trunk = self.trunkbranch(x)

        # soft Mask Branch
        # The Number of Residual Units between adjacent pooling layer is 1.
        pool1 = self.maxpool1(x)  # (24,24) -> (12,12)
        out_softmask1 = self.Resblock2(pool1)
        skip_connection1 = self.skip1(pool1)  # "skip_connection"

        pool2 = self.maxpool2(out_softmask1)  # (12,12) -> (6,6)
        out_softmask2 = self.Resblock3(pool2)

        out_interp2 = nn.functional.interpolate(out_softmask2, size=self.size2, mode='bilinear',
                                                align_corners=True)  # (6,6) ->(12,12)
        out = out_interp2 + skip_connection1
        out_softmask3 = self.Resblock4(out)
        out_interp1 = nn.functional.interpolate(out_softmask3, size=self.size1, mode='bilinear',
                                                align_corners=True)  # (24,24)
        out_softmask4 = self.output_block(out_interp1)

        out = (1 + out_softmask4) * out_trunk
        last_out = self.last_block(out)

        return last_out


class AttentionModule_stage3(nn.Module):

    def __init__(self, inplanes, outplanes, size1):
        """
        max_pooling layers are used in mask branch size with input
        """
        super(AttentionModule_stage3, self).__init__()
        self.size1 = size1
        self.Resblock1 = ResUnit(inplanes, outplanes)  # first residual block

        self.trunkbranch = nn.Sequential(ResUnit(inplanes, outplanes),
                                         ResUnit(inplanes, outplanes))

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (12,12) -> (6,6)
        self.Resblock2 = nn.Sequential(ResUnit(inplanes, outplanes),
                                       ResUnit(inplanes, outplanes))
        # self.upsample1 = nn.UpsamplingBilinear2d(size1)

        self.output_block = nn.Sequential(nn.BatchNorm2d(outplanes),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, bias=False),
                                          nn.BatchNorm2d(outplanes),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, bias=False),
                                          nn.Sigmoid())

        self.last_block = ResUnit(inplanes, outplanes)

    def forward(self, x):
        # The Number of pre-processing Residual Units Before
        # Splitting into trunk branch and mask branch is 1.
        x = self.Resblock1(x)

        # The output of trunk branch
        out_trunk = self.trunkbranch(x)

        # soft Mask Branch
        # The Number of Residual Units between adjacent pooling layer is 1.
        pool1 = self.maxpool1(x)  # (12,12) -> (6,6)
        out_softmask1 = self.Resblock2(pool1)
        out_interp1 = nn.functional.interpolate(out_softmask1, size=self.size1, mode='bilinear',
                                                align_corners=True)  # (6,6) ->(12,12)
        out_softmask2 = self.output_block(out_interp1)
        out = (1 + out_softmask2) * out_trunk
        last_out = self.last_block(out)

        return last_out


class ResidualAttentionNetwork(nn.Module):

    def __init__(self):
        super(ResidualAttentionNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16,
                                             kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Resblock1 = ResUnit(16, 64)
        self.attention_module1 = AttentionModule_stage1(64, 64, size1=(48, 48), size2=(24, 24),
                                                        size3=(12, 12))  # (48,48)
        self.attention_module2 = AttentionModule_stage1(64, 64, size1=(48, 48), size2=(24, 24), size3=(12, 12))
        self.Resblock2 = ResUnit(64, 128, 2)  # (24, 24)
        self.attention_module3 = AttentionModule_stage2(128, 128, size1=(24, 24), size2=(12, 12))
        self.attention_module4 = AttentionModule_stage2(128, 128, size1=(24, 24), size2=(12, 12))
        self.Resblock3 = ResUnit(128, 256, 2)
        self.attention_module5 = AttentionModule_stage3(256, 256, size1=(12, 12))
        self.attention_module6 = AttentionModule_stage3(256, 256, size1=(12, 12))
        self.Resblock4 = nn.Sequential(ResUnit(256, 512, 2),
                                       ResUnit(512, 512),
                                       ResUnit(512, 512))
        self.Avergepool = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=6, stride=1)
        )

        self.fc1 = nn.Linear(512, 1)
        # self.pred = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)  # (96,96)
        # print(x.shape)
        x = self.maxpool1(x)  #
        x = self.Resblock1(x)
        x = self.attention_module1(x)
        x = self.attention_module2(x)
        x = self.Resblock2(x)
        x = self.attention_module3(x)
        x = self.attention_module4(x)
        x = self.Resblock3(x)
        x = self.attention_module5(x)
        x = self.attention_module6(x)
        x = self.Resblock4(x)
        x = self.Avergepool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # print(x.shape)
        # x = self.pred(x)
        # x = x.squeeze()

        return x
