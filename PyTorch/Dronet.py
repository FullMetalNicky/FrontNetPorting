import torch.nn as nn
from PreActBlock import PreActBlock
from PreActBlock import printRes
import numpy as np
import logging

np.set_printoptions(threshold=np.inf)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Dronet(nn.Module):
    def __init__(self, block, layers, gray=False):
        super(Dronet, self).__init__()

        if gray ==True:
            self.name = "DronetGray"
        else:
            self.name = "DronetRGB"
        self.inplanes = 32
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d

        self.groups = 1
        self.base_width = 64
        if gray == True:
            self.conv = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        else:
            self.conv = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = PreActBlock(32, 32, stride=2)
        self.layer2 = PreActBlock(32, 64, stride=2)
        self.layer3 = PreActBlock(64, 128, stride=2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        # 160x96 -> 1920, 108x60 -> 1024, 80x48 -> 768
        fcSize = 1920
        self.fc_x = nn.Linear(fcSize, 1)
        self.fc_y = nn.Linear(fcSize, 1)
        self.fc_z = nn.Linear(fcSize, 1)
        self.fc_phi = nn.Linear(fcSize, 1)


    def forward(self, x):

        out = self.conv(x)
        out = self.maxpool(out)
        #printRes(out, "5x5ConvMax_1")
        out = self.layer1(out)
        #printRes(out, "Add_1")
        out = self.layer2(out)
        #printRes(out, "Add_2")
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(out)
        #logging.info("AddReLU_3")
        #logging.info(list(out.numpy()))
        out = self.dropout(out)
        x = self.fc_x(out)
        #logging.info("Dense1={}".format(x.numpy()))
        y = self.fc_y(out)
        #logging.info("Dense2={}".format(y.numpy()))
        z = self.fc_z(out)
        #logging.info("Dense3={}".format(z.numpy()))
        phi = self.fc_phi(out)
        #logging.info("Dense4={}".format(phi.numpy()))

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
