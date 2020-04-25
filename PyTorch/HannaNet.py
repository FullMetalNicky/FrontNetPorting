import torch.nn as nn
from ConvBlock import ConvBlock
from ConvBlock import printRes
import numpy as np
import logging

np.set_printoptions(threshold=np.inf)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class HannaNet(nn.Module):
    def __init__(self, block, layers, isGray=False):
        super(HannaNet, self).__init__()

        if isGray ==True:
            self.name = "HannaNetGray"
        else:
            self.name = "HannaNetRGB"
        self.inplanes = 32
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

        self.layer1 = ConvBlock(32, 32, stride=2)
        self.layer2 = ConvBlock(32, 64, stride=2)
        self.layer3 = ConvBlock(64, 128, stride=2)

        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout()

        fcSize = 1920
        self.fc_x = nn.Linear(fcSize, 1)
        self.fc_y = nn.Linear(fcSize, 1)
        self.fc_z = nn.Linear(fcSize, 1)
        self.fc_phi = nn.Linear(fcSize, 1)


    def forward(self, x):

        conv5x5 = self.conv(x)
        btn = self.bn(conv5x5)
        relu1 = self.relu2(btn)
        max_pool = self.maxpool(relu1)

        l1 = self.layer1(max_pool)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        flat = l3.view(l3.size(0), -1)

        drop = self.dropout(flat)
        x = self.fc_x(drop)
        y = self.fc_y(drop)
        z = self.fc_z(drop)
        phi = self.fc_phi(drop)

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
