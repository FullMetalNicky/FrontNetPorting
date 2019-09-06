import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class GrayFrontNet(nn.Module):
    def __init__(self, block, layers):
        super(GrayFrontNet, self).__init__()

        self.inplanes = 32
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d

        # if replace_stride_with_dilation is None:
        # each element in the tuple indicates if we should replace
        # the 2x2 stride with a dilated convolution instead
        replace_stride_with_dilation = [False, False, False]
        # if len(replace_stride_with_dilation) != 3:
        #   raise ValueError("replace_stride_with_dilation should be None "
        #                   "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = 1
        self.base_width = 64
        self.conv = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 7), stride=(1, 1))
        self.fc1 = nn.Linear(128 * block.expansion, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_x = nn.Linear(64, 1)
        self.fc_y = nn.Linear(64, 1)
        self.fc_z = nn.Linear(64, 1)
        self.fc_phi = nn.Linear(64, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


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