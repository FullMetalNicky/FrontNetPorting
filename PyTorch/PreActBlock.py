import torch.nn as nn

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    # I don't use all these parameters
    # I just wanted to keep the previous API. I will clean it up once you approve the net

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # n, c, h, w = x.shape
        # location = int((c * w * h) / 2)
        # tmp = x.reshape(-1)
        # print("entering resnet block: val {} at {}".format(tmp[location:location+w], location))
        out = self.bn1(x)
        out = self.relu1(out)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # n, c, h, w = out.shape
        # location = int((c * w * h ) / 2)
        # tmp = out.reshape(-1)
        # print("After resnet conv 1: val {} at {}".format(tmp[location:location+w], location))
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        # n, c, h, w = out.shape
        # location = int((c * w * h ) / 2)
        # tmp = out.reshape(-1)
        # print("After resnet conv 2: val {} at {}".format(tmp[location:location+w], location))
        # n, c, h, w = shortcut.shape
        # location = int((c * w * h ) / 2)
        # tmp = shortcut.reshape(-1)
        # print("After resnet shortcut: val {} at {}".format(tmp[location:location+w], location))
        out += shortcut
        return out
