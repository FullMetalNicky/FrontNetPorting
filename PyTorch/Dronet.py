import torch.nn as nn
from PreActBlock import PreActBlock


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Dronet(nn.Module):
    def __init__(self, block, layers, isGray=False):
        super(Dronet, self).__init__()

        if isGray ==True:
            self.name = "DronetGray"
        else:
            self.name = "DronetRGB"
        self.inplanes = 32
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d

        self.groups = 1
        self.base_width = 64
        if isGray == True:
            self.conv = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        else:
            self.conv = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = PreActBlock(32, 32, stride=2)
        self.layer2 = PreActBlock(32, 64, stride=2)
        self.layer3 = PreActBlock(64, 128, stride=2)

        self.dropout = nn.Dropout()
        finalSize = 128*2*4
        self.fc_x = nn.Linear(finalSize, 1)
        self.fc_y = nn.Linear(finalSize, 1)
        self.fc_z = nn.Linear(finalSize, 1)
        self.fc_phi = nn.Linear(finalSize, 1)


    def forward(self, x):
        # n, c, h, w = x.shape
        # location = int((c * w * h) / 2)
        # tmp = x.reshape(-1)
        # print("input: val {} at {}".format(tmp[location:location + w], location))
        out = self.conv(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(out)
        out = self.dropout(out)
        x = self.fc_x(out)
        y = self.fc_y(out)
        z = self.fc_z(out)
        phi = self.fc_phi(out)

        return [x, y, z, phi]