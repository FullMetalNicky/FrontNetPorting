
import torch.nn as nn
from PreActBlock import PreActBlock




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



# FrontNet
class FrontNet(nn.Module):
    def __init__(self, block, layers, isGray=False):
        super(FrontNet, self).__init__()

        if isGray ==True:
            self.name = "FrontNetGray"
        else:
            self.name = "FrontNetRGB"
        self.inplanes = 32
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d

        replace_stride_with_dilation = [False, False, False]

        self.groups = 1
        self.base_width = 64
        if isGray == True:
            self.conv = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        else:
            self.conv = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = PreActBlock(32, 32, stride=1)
        self.layer2 = PreActBlock(32, 64, stride=2)
        self.layer3 = PreActBlock(64, 128, stride=2)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 7), stride=(1, 1))
        self.fc1 = nn.Linear(128 * block.expansion, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_x = nn.Linear(64, 1)
        self.fc_y = nn.Linear(64, 1)
        self.fc_z = nn.Linear(64, 1)
        self.fc_phi = nn.Linear(64, 1)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        x = self.fc_x(out)
        y = self.fc_y(out)
        z = self.fc_z(out)
        phi = self.fc_phi(out)

        return [x, y, z, phi]

