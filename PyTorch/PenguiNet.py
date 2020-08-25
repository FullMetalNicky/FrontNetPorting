import torch.nn as nn
import torch
from ConvBlock import ConvBlock
from ConvBlock import printRes
import numpy as np
import logging

np.set_printoptions(threshold=np.inf)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PenguiNet(nn.Module):
    def __init__(self, block, layers, isGray=False, w=160, h=96, c=32, fc_nodes=1920):
        super(PenguiNet, self).__init__()

        if isGray ==True:
            self.name = "PenguiNet"
        else:
            self.name = "PenguiNetRGB"
        self.inplanes = c
        self.width = w
        self.height = h
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d

        self.groups = 1
        self.base_width = 64
        if isGray == True:
            self.conv = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        else:
            self.conv = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU()

        self.layer1 = ConvBlock(self.inplanes, self.inplanes, stride=2)
        self.layer2 = ConvBlock(self.inplanes, self.inplanes*2, stride=2)
        self.layer3 = ConvBlock(self.inplanes*2, self.inplanes*4, stride=2)

        self.dropout = nn.Dropout()

        fcSize = fc_nodes
        self.fc = nn.Linear(fcSize, 4)


    def forward(self, x):

        conv5x5 = self.conv(x)
        btn = self.bn(conv5x5)
        relu1 = self.relu1(btn)
        max_pool = self.maxpool(relu1)

        l1 = self.layer1(max_pool)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        out = l3.flatten(1)

        out = self.dropout(out)
        out = self.fc(out)
        x = out[:, 0]
        y = out[:, 1]
        z = out[:, 2]
        phi = out[:, 3]
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)
        phi = phi.unsqueeze(1)
        # PrintFC(x, "Dense1")
        # PrintFC(y, "Dense2")
        # PrintFC(z, "Dense3")
        # PrintFC(phi, "Dense4")

        return [x, y, z, phi]


def PrintRelu(layer, name):
    logger = logging.getLogger('')
    enable = logger.isEnabledFor(logging.INFO)
    if (enable == True):
        tmp = layer.reshape(-1)
        logging.info("{}={}".format(name, list(tmp.numpy())))

def PrintFC(layer, name):
    logger = logging.getLogger('')
    enable = logger.isEnabledFor(logging.INFO)
    if (enable == True):
        logging.info("{}={}".format(name, layer.numpy()))
